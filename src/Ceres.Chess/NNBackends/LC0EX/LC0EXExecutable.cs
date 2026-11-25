#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Provenance

/*
  lc0ex -- the executable format, its precompiled kernels, the builder that produces them and the
  reference CUDA runtime -- is the work of Kovax (GitHub: Kovax007) for Leela Chess Zero.

  This file is a C# port of parts of that runtime:
    src/neural/backends/lc0ex-cuda/runtime/lc0ex_cuda.cc     (module loading, kernel and symbol resolution,
                                                              persistent allocation)
    src/neural/backends/lc0ex-cuda/network_lc0ex_cuda.cc     (matching and uploading the carrier's weights)

  Original sources: https://github.com/Kovax007/lc0, branch tcec-season-30-swiss-lc0ex, commit 29ec4b44.
  Copyright (C) 2026 The LCZero Authors. Licensed under the GNU General Public License, version 3 or
  (at your option) any later version, with the additional permission for NVIDIA libraries stated there.

  Ported to C# and modified for Ceres in November 2026.
  Differences from the original: the loaded executable is shared (reference counted) between
  evaluators on a device, and some structural and architecture checks were added.
*/

#endregion

#region Using directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using ManagedCuda;
using ManagedCuda.BasicTypes;

using Ceres.Base.CUDA;

#endregion

namespace Ceres.Chess.NNBackends.LC0EX
{
  /// <summary>
  /// An lc0ex neural executable loaded onto one GPU: its CUDA modules, kernel functions,
  /// resolved symbols and the persistent allocation holding the weights.
  ///
  /// Read-only after load, so it is shared (reference counted) by all executions
  /// (evaluator instances) using the same (GPU, artifact, carrier). Executions own
  /// streams, execution memory and graphs (see LC0EXExecution).
  /// Obtained with Acquire and given back with Release (there is no public Dispose).
  ///
  /// Implemented entirely with CUDA driver API calls (via ManagedCuda), mirroring
  /// lc0's lc0ex runtime (src/neural/backends/lc0ex-cuda/runtime/lc0ex_cuda.cc).
  /// </summary>
  public sealed class LC0EXExecutable
  {
    public LC0EXExecutableDef Def { get; }
    public CUDADevice Device { get; }
    public int GPUID => Device.GPUID;
    public string ArtifactFileName { get; }
    public string CarrierFileName { get; }
    public long CarrierFileSizeBytes { get; }
    public LC0EXNetworkFormat NetworkFormat { get; }

    /// <summary>
    /// If the network has a moves-left head (every program then has a moves-left output).
    /// </summary>
    public bool HasMovesLeft => NetworkFormat.MovesLeft != 0;

    /// <summary>
    /// CUDA function for each entry of Def.Kernels (keyed by index, never by name:
    /// the same function name appears in many cubins).
    /// </summary>
    internal CUfunction[] Functions { get; private set; }

    /// <summary>
    /// Aligned base address of the persistent (weights) allocation.
    /// </summary>
    internal ulong PersistentBase { get; private set; }

    /// <summary>
    /// Device addresses of symbol arguments, keyed by (binary index, symbol name).
    /// </summary>
    internal Dictionary<(int, string), ulong> SymbolAddresses { get; } = new();

    CUmodule[] modules;
    CUdeviceptr persistentRaw;
    int refCount;

    /// <summary>
    /// Serializes measurement of program latencies (see LC0EXExecution), so that
    /// executions created concurrently for the same executable do not time each other.
    /// </summary>
    internal readonly object ProgramCostLock = new();

    /// <summary>
    /// Measured wall-clock latency (milliseconds) of one launch of each program, keyed by
    /// program batch size. Measured once per executable (by the first execution that needs
    /// a program) and then shared, so all executions on a device plan batches identically.
    /// Access only while holding ProgramCostLock.
    /// </summary>
    internal readonly Dictionary<int, double> ProgramCostsMs = new();


    #region Shared instance cache

    static readonly Dictionary<(int, string, string), LC0EXExecutable> cache = new();
    static readonly Dictionary<(int, string, string), object> loadLocks = new();
    static readonly Dictionary<string, WeakReference<LC0EXCarrier>> carrierCache = new();
    static readonly object cacheLock = new();

    /// <summary>
    /// File names are compared as the file system does (not case sensitive on Windows),
    /// so that one file named with different casing is not loaded twice.
    /// </summary>
    static string CacheFileName(string fullPath) => OperatingSystem.IsWindows() ? fullPath.ToUpperInvariant() : fullPath;

    readonly (int, string, string) cacheKey;


    /// <summary>
    /// Returns the (possibly shared) executable for the given GPU, artifact and carrier,
    /// loading it if necessary. Each Acquire must be balanced by a Release.
    /// </summary>
    public static LC0EXExecutable Acquire(int gpuID, string artifactFileName, string carrierFileName)
    {
      string artifactFull = Path.GetFullPath(artifactFileName);
      string carrierFull = Path.GetFullPath(carrierFileName);
      (int, string, string) key = (gpuID, CacheFileName(artifactFull), CacheFileName(carrierFull));

      object loadLock;
      lock (cacheLock)
      {
        if (cache.TryGetValue(key, out LC0EXExecutable existing))
        {
          existing.refCount++;
          return existing;
        }

        if (!loadLocks.TryGetValue(key, out loadLock))
        {
          loadLocks[key] = loadLock = new object();
        }
      }

      // Load outside the cache lock, so that a different executable can be acquired meanwhile,
      // but one at a time per key: a second request for the same executable waits and then shares it.
      // (The device part of every load is in any case serialized process-wide, see the constructor.)
      lock (loadLock)
      {
        lock (cacheLock)
        {
          if (cache.TryGetValue(key, out LC0EXExecutable existing))
          {
            existing.refCount++;
            return existing;
          }
        }

        LC0EXExecutableDef def = LC0EXExecutableDef.Load(artifactFull);
        LC0EXCarrier carrier = GetCarrier(carrierFull);
        LC0EXExecutable loaded = new(def, carrier, CUDADevice.GetContext(gpuID), artifactFull, key);

        lock (cacheLock)
        {
          loaded.refCount = 1;
          cache[key] = loaded;
          return loaded;
        }
      }
    }


    /// <summary>
    /// Releases one reference; frees device resources when the last reference is released.
    /// </summary>
    public void Release()
    {
      lock (cacheLock)
      {
        if (refCount <= 0)
        {
          throw new InvalidOperationException("LC0EXExecutable released more often than acquired.");
        }
        if (--refCount > 0)
        {
          return;
        }

        if (cache.TryGetValue(cacheKey, out LC0EXExecutable cached) && ReferenceEquals(cached, this))
        {
          cache.Remove(cacheKey);
        }
      }

      FreeDeviceResources();
    }


    /// <summary>
    /// Carriers are large (hundreds of MB); share one parsed instance between devices
    /// loading at the same time, but let it be collected once all loads are done.
    /// </summary>
    static LC0EXCarrier GetCarrier(string carrierFull)
    {
      lock (carrierCache)
      {
        if (carrierCache.TryGetValue(carrierFull, out WeakReference<LC0EXCarrier> weak)
         && weak.TryGetTarget(out LC0EXCarrier existing))
        {
          return existing;
        }

        LC0EXCarrier carrier = new(carrierFull);
        carrierCache[carrierFull] = new WeakReference<LC0EXCarrier>(carrier);
        return carrier;
      }
    }

    #endregion


    LC0EXExecutable(LC0EXExecutableDef def, LC0EXCarrier carrier, CUDADevice device, string artifactFull,
                    (int, string, string) cacheKey)
    {
      this.cacheKey = cacheKey;
      Def = def;
      Device = device;
      ArtifactFileName = artifactFull;
      CarrierFileName = Path.GetFullPath(carrier.FileName);
      CarrierFileSizeBytes = new FileInfo(carrier.FileName).Length;
      NetworkFormat = carrier.NetworkFormat;

      ValidateStructure();
      CheckFingerprint(carrier);
      CheckMovesLeft();

      // Loading (module load, allocation, synchronous copies) is done under the
      // process-wide graph capture write lock, per Ceres convention, so it cannot
      // disturb a stream capture in progress in another component.
      device.GraphCaptureRWLock.EnterWriteLock();
      try
      {
        device.SetCurrent();
        CheckArchitecture();
        LoadModules();
        ResolveFunctions();
        ResolveSymbols();
        AllocateAndUploadWeights(carrier);

        // The copies above are synchronous, but make certain everything has
        // reached the device before any execution stream reads the weights.
        CheckResult(DriverAPINativeMethods.ContextManagement.cuCtxSynchronize(), "cuCtxSynchronize");
      }
      catch
      {
        FreeDeviceResources();
        throw;
      }
      finally
      {
        device.GraphCaptureRWLock.ExitWriteLock();
      }
    }


    #region Validation

    void ValidateStructure()
    {
      if (Def.RuntimeParameters.Count > 0)
      {
        throw new NotSupportedException($"lc0ex executable {Path.GetFileName(ArtifactFileName)} declares runtime parameters "
                                      + $"({string.Join(",", Def.RuntimeParameters.Select(p => p.Name))}), not supported.");
      }

      foreach (LC0EXKernelDef k in Def.Kernels)
      {
        if (k.BinaryIndex < 0 || k.BinaryIndex >= Def.Binaries.Count)
        {
          throw new FormatException($"lc0ex kernel {k.Function} references missing binary {k.BinaryIndex}.");
        }
        if (k.Parameters.Any(p => p == LC0EXParameterType.Unknown || p > LC0EXParameterType.NullPointer))
        {
          throw new FormatException($"lc0ex kernel {k.Function} has an unknown parameter type.");
        }
      }

      foreach (LC0EXBufferDef b in Def.PersistentBuffers)
      {
        if (!b.IsRowMajor && !b.TryGetPitched2D(out _, out _, out _))
        {
          throw new NotSupportedException($"lc0ex persistent buffer '{b.Name}' has an unsupported layout "
                                        + $"(strides [{string.Join(",", b.Strides)}] for shape [{string.Join(",", b.Shape)}]).");
        }
        if (string.IsNullOrEmpty(b.Name))
        {
          throw new FormatException("lc0ex persistent buffer without a name.");
        }
        if (b.Offset > Def.PersistentSizeBytes || b.FootprintBytes > Def.PersistentSizeBytes - b.Offset)
        {
          throw new FormatException($"lc0ex persistent buffer '{b.Name}' extends past the persistent allocation.");
        }
      }

      foreach (LC0EXProgramDef p in Def.Programs)
      {
        for (int n = 0; n < p.Nodes.Length; n++)
        {
          LC0EXNodeDef node = p.Nodes[n];
          if (node.KernelIndex < 0 || node.KernelIndex >= Def.Kernels.Count)
          {
            throw new FormatException($"lc0ex program {p.Name} node {n} references missing kernel {node.KernelIndex}.");
          }
          if (node.Dependencies.Any(d => d < 0 || d >= n))
          {
            throw new FormatException($"lc0ex program {p.Name} node {n} has a dependency that is not an earlier node.");
          }

          LC0EXKernelDef kernel = Def.Kernels[node.KernelIndex];
          int numArgsExpected = kernel.Parameters.Count(t => t != LC0EXParameterType.NullPointer);
          if (node.Arguments.Length != numArgsExpected)
          {
            throw new FormatException($"lc0ex program {p.Name} node {n} ({kernel.Function}) has {node.Arguments.Length} "
                                    + $"arguments but its kernel ABI needs {numArgsExpected}.");
          }

          foreach (LC0EXArgumentDef a in node.Arguments)
          {
            if (a.Kind == LC0EXArgumentKind.Allocation)
            {
              ulong limit = a.AllocationKind == LC0EXAllocationKind.Persistent ? Def.PersistentSizeBytes : p.ExecutionSizeBytes;
              if (a.AllocationOffset >= limit)
              {
                throw new FormatException($"lc0ex program {p.Name} node {n} has an argument offset outside its allocation.");
              }
            }
          }
        }
      }
    }


    /// <summary>
    /// Requires the carrier to be the network architecture the executable was built for:
    /// the same comparison as lc0 (network format, head count and number of encoder layers).
    ///
    /// The head and layer counts come from the Lc0 weights section. A carrier without that
    /// section (ONNX only) cannot supply them, so then only the network format is compared
    /// (lc0 itself would reject such a carrier for an executable built from a net that has them).
    /// A different depth or width is still caught when the weight tensors are matched by name and shape.
    /// </summary>
    void CheckFingerprint(LC0EXCarrier carrier)
    {
      string artifactName = Path.GetFileName(ArtifactFileName);
      if (Def.Metadata == null || Def.Metadata.Length == 0)
      {
        throw new Exception($"lc0ex executable {artifactName} has no network fingerprint (metadata), "
                          + "so it cannot be matched to a network.");
      }

      LC0EXNetworkFingerprint expected = LC0EXNetworkFingerprint.FromSerializedNet(Def.Metadata, 0, Def.Metadata.Length, applyLoaderFixups: false,
                                                                                   out bool hasNetworkFormat, out _);
      if (!hasNetworkFormat)
      {
        throw new Exception($"lc0ex executable {artifactName} has a network fingerprint without a network format.");
      }

      bool matches = carrier.HasWeightsSection ? expected == carrier.Fingerprint
                                               : expected.Format == carrier.Fingerprint.Format;
      if (!matches)
      {
        throw new Exception($"lc0ex executable {artifactName} was built for a different network architecture "
                          + $"than carrier {Path.GetFileName(carrier.FileName)}: executable {expected}, carrier {carrier.Fingerprint}.");
      }

      if (!carrier.HasWeightsSection && (expected.Headcount != 0 || expected.NumEncoderLayers != 0))
      {
        Console.WriteLine($"lc0ex: carrier {Path.GetFileName(carrier.FileName)} has no Lc0 weights section, so the head count "
                        + $"({expected.Headcount}) and layer count ({expected.NumEncoderLayers}) expected by {artifactName} were not verified.");
      }
    }


    /// <summary>
    /// A network with a moves-left head requires a moves-left output in every program (as in lc0).
    /// </summary>
    void CheckMovesLeft()
    {
      if (!HasMovesLeft)
      {
        return;
      }

      foreach (LC0EXProgramDef p in Def.Programs)
      {
        if (!p.Buffers.ContainsKey("/output/mlh"))
        {
          throw new FormatException($"lc0ex program {p.Name} has no moves-left output buffer, but the network has a moves-left head.");
        }
      }
    }


    /// <summary>
    /// Requires the device to be able to run the architecture the cubins were compiled for.
    /// A plain target (e.g. sm_86) also runs on later devices of the same major version (e.g. 8.9);
    /// a target with an "a" suffix (e.g. sm_90a, as Triton emits) uses features of exactly that architecture.
    /// </summary>
    void CheckArchitecture()
    {
      (int major, int minor) = Def.TargetComputeCapability;
      int devMajor = Device.CUDACapabilityMajor;
      int devMinor = Device.CUDACapabilityMinor;
      bool exactOnly = (Def.TargetArchitecture ?? "").EndsWith('a');
      bool compatible = major == devMajor && (exactOnly ? minor == devMinor : minor <= devMinor);
      if (!compatible)
      {
        throw new NotSupportedException($"lc0ex executable {Path.GetFileName(ArtifactFileName)} targets {Def.TargetArchitecture} "
                                      + $"but GPU {GPUID} is compute capability {devMajor}.{devMinor} (sm_{devMajor}{devMinor}). "
                                      + "lc0ex executables are compiled for one GPU architecture; build or obtain one for this GPU.");
      }
    }

    #endregion


    #region Loading

    void LoadModules()
    {
      modules = new CUmodule[Def.Binaries.Count];
      for (int i = 0; i < Def.Binaries.Count; i++)
      {
        CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleLoadData(ref modules[i], Def.Binaries[i]);
        if (res != CUResult.Success)
        {
          // (Likely causes only; the driver does not say more.)
          string hint = res switch
          {
            CUResult.ErrorNoBinaryForGPU or CUResult.ErrorInvalidImage
              => $"possibly the cubin is not compatible with this GPU (artifact target {Def.TargetArchitecture})",
            CUResult.UnsupportedPTXVersion or CUResult.ErrorInvalidPtx or CUResult.ErrorSharedObjectInitFailed
              => "possibly the cubin needs a newer NVIDIA driver than the one installed",
            _ => "module load failed"
          };
          throw new Exception($"lc0ex: cuModuleLoadData failed for binary {i} of {Path.GetFileName(ArtifactFileName)}: {res} ({hint}).");
        }

        // The driver has its own copy now; do not keep the cubin on the managed heap.
        Def.Binaries[i] = null;
      }
    }


    void ResolveFunctions()
    {
      Functions = new CUfunction[Def.Kernels.Count];
      for (int k = 0; k < Def.Kernels.Count; k++)
      {
        LC0EXKernelDef kernel = Def.Kernels[k];
        CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetFunction(ref Functions[k], modules[kernel.BinaryIndex], kernel.Function);
        CheckResult(res, $"cuModuleGetFunction({kernel.Function})");
      }

      // Dynamic shared memory above 48 KB must be opted into per function.
      // Use the maximum over every node (in every program) that launches the function.
      int[] maxSmem = new int[Def.Kernels.Count];
      foreach (LC0EXProgramDef p in Def.Programs)
      {
        foreach (LC0EXNodeDef n in p.Nodes)
        {
          maxSmem[n.KernelIndex] = Math.Max(maxSmem[n.KernelIndex], checked((int)n.DynamicSharedMemoryBytes));
        }
      }

      int deviceOptinLimit = Device.Context.GetDeviceInfo().MaxSharedMemoryPerBlockOptin;
      for (int k = 0; k < Def.Kernels.Count; k++)
      {
        if (maxSmem[k] == 0)
        {
          continue;
        }

        if (maxSmem[k] > deviceOptinLimit)
        {
          throw new NotSupportedException($"lc0ex kernel {Def.Kernels[k].Function} needs {maxSmem[k]} bytes of shared memory, "
                                        + $"GPU {GPUID} allows {deviceOptinLimit}.");
        }

        CUResult res = DriverAPINativeMethods.FunctionManagement.cuFuncSetAttribute(Functions[k], CUFunctionAttribute.MaxDynamicSharedSizeBytes, maxSmem[k]);
        CheckResult(res, $"cuFuncSetAttribute({Def.Kernels[k].Function}, MaxDynamicSharedSizeBytes={maxSmem[k]})");
      }
    }


    void ResolveSymbols()
    {
      IEnumerable<(int, string)> symbols = Def.Programs.SelectMany(p => p.Nodes)
                                                       .SelectMany(n => n.Arguments)
                                                       .Where(a => a.Kind == LC0EXArgumentKind.Symbol)
                                                       .Select(a => (a.SymbolBinaryIndex, a.SymbolName))
                                                       .Distinct();
      foreach ((int binary, string name) in symbols)
      {
        if (binary < 0 || binary >= modules.Length)
        {
          throw new FormatException($"lc0ex symbol {name} references missing binary {binary}.");
        }

        CUdeviceptr ptr = default;
        SizeT size = default;
        CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref ptr, ref size, modules[binary], name);
        CheckResult(res, $"cuModuleGetGlobal({name})");
        SymbolAddresses[(binary, name)] = ptr.Pointer;
      }
    }


    unsafe void AllocateAndUploadWeights(LC0EXCarrier carrier)
    {
      if (Def.PersistentSizeBytes == 0)
      {
        if (Def.PersistentBuffers.Count > 0)
        {
          throw new FormatException("lc0ex executable has persistent buffers but no persistent allocation.");
        }
        return;
      }

      // Match every persistent buffer to an initializer before touching the device.
      List<string> missing = new();
      List<string> problems = new();
      List<(LC0EXBufferDef buffer, LC0EXInitializer init)> uploads = new();
      foreach (LC0EXBufferDef buffer in Def.PersistentBuffers)
      {
        if (!carrier.Initializers.TryGetValue(buffer.Name, out LC0EXInitializer init))
        {
          missing.Add(buffer.Name);
          continue;
        }

        if (init.DataType != (int)buffer.DataType)
        {
          problems.Add($"{buffer.Name}: data type {init.DataType} vs buffer {(int)buffer.DataType}");
        }
        else if (init.Dims.Length != buffer.Shape.Length
              || !init.Dims.Zip(buffer.Shape, (d, s) => d >= 0 && (ulong)d == s).All(x => x))
        {
          problems.Add($"{buffer.Name}: shape [{string.Join(",", init.Dims)}] vs buffer [{string.Join(",", buffer.Shape)}]");
        }
        else if (init.Offset < 0 || (ulong)init.Length != buffer.SizeBytes)
        {
          problems.Add($"{buffer.Name}: raw_data {(init.Offset < 0 ? "missing (typed data fields are not supported)" : init.Length + " bytes")}"
                     + $" vs buffer {buffer.SizeBytes} bytes");
        }
        else
        {
          uploads.Add((buffer, init));
        }
      }

      if (missing.Count > 0)
      {
        throw new Exception($"lc0ex carrier {Path.GetFileName(carrier.FileName)} is missing {missing.Count} weight tensor(s) needed by "
                          + $"{Path.GetFileName(ArtifactFileName)} (artifact and carrier must come from the same builder version): "
                          + string.Join(", ", missing.Take(12)) + (missing.Count > 12 ? ", ..." : ""));
      }
      if (problems.Count > 0)
      {
        throw new Exception($"lc0ex carrier {Path.GetFileName(carrier.FileName)} does not match {Path.GetFileName(ArtifactFileName)}: "
                          + string.Join("; ", problems.Take(8)) + (problems.Count > 8 ? "; ..." : ""));
      }

      HashSet<string> bufferNames = Def.PersistentBuffers.Select(b => b.Name).ToHashSet();
      int numUnused = carrier.Initializers.Keys.Count(k => !bufferNames.Contains(k));
      if (numUnused > 0)
      {
        Console.WriteLine($"lc0ex: {numUnused} carrier initializer(s) have no corresponding buffer and are ignored.");
      }

      // Aligned persistent allocation, zeroed so unnamed ranges are zero (as in lc0).
      ulong align = Math.Max(Def.PersistentAlignment, 1);
      CheckResult(DriverAPINativeMethods.MemoryManagement.cuMemAlloc_v2(ref persistentRaw, Def.PersistentSizeBytes + align - 1),
                  $"cuMemAlloc({Def.PersistentSizeBytes} bytes persistent)");
      PersistentBase = (persistentRaw.Pointer + (align - 1)) & ~(align - 1);
      CheckResult(DriverAPINativeMethods.Memset.cuMemsetD8_v2(new CUdeviceptr(PersistentBase), 0, Def.PersistentSizeBytes),
                  "cuMemsetD8(persistent)");

      fixed (byte* src = carrier.Data)
      {
        foreach ((LC0EXBufferDef buffer, LC0EXInitializer init) in uploads)
        {
          if (init.Length == 0)
          {
            continue;
          }

          CUResult res;
          if (buffer.TryGetPitched2D(out ulong rows, out ulong cols, out ulong pitch))
          {
            // Row-major source rows scattered to a pitched destination (one 2D copy).
            ulong elemSize = (ulong)LC0EXBufferDef.ElementSize(buffer.DataType);
            CUDAMemCpy2D copy = new()
            {
              srcMemoryType = CUMemoryType.Host,
              srcHost = (IntPtr)(src + init.Offset),
              srcPitch = cols * elemSize,
              dstMemoryType = CUMemoryType.Device,
              dstDevice = new CUdeviceptr(PersistentBase + buffer.Offset),
              dstPitch = pitch * elemSize,
              WidthInBytes = cols * elemSize,
              Height = rows,
            };
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copy);
          }
          else
          {
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(new CUdeviceptr(PersistentBase + buffer.Offset),
                                                                              (IntPtr)(src + init.Offset), init.Length);
          }
          CheckResult(res, $"upload of weights '{buffer.Name}'");
        }
      }
    }

    #endregion


    internal static void CheckResult(CUResult res, string what)
    {
      if (res != CUResult.Success)
      {
        throw new Exception($"lc0ex CUDA error in {what}: {res}");
      }
    }


    #region Dispose

    void FreeDeviceResources()
    {
      try
      {
        Device.SetCurrent();
      }
      catch
      {
        // Ignore; best effort cleanup.
      }

      if (persistentRaw.Pointer != 0)
      {
        DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(persistentRaw);
        persistentRaw = default;
        PersistentBase = 0;
      }

      if (modules != null)
      {
        foreach (CUmodule m in modules)
        {
          if (m.Pointer != IntPtr.Zero)
          {
            DriverAPINativeMethods.ModuleManagement.cuModuleUnload(m);
          }
        }
        modules = null;
      }
    }

    #endregion


    public override string ToString()
      => $"<LC0EXExecutable GPU {GPUID} {Path.GetFileName(ArtifactFileName)} + {Path.GetFileName(CarrierFileName)} "
       + $"({Def.TargetArchitecture}, programs [{string.Join(",", Def.Programs.Select(p => p.BatchSize))}])>";
  }
}
