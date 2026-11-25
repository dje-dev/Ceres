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

  This file reads the format he defined; its field numbers, enum values and semantics follow:
    proto/lc0ex.proto
    proto/lc0ex_metadata.proto

  Original sources: https://github.com/Kovax007/lc0, branch tcec-season-30-swiss-lc0ex, commit 29ec4b44.
  Copyright (C) 2026 The LCZero Authors. Licensed under the GNU General Public License, version 3 or
  (at your option) any later version, with the additional permission for NVIDIA libraries stated there.

  Ported to C# and modified for Ceres in November 2026.
  The reading code is written for Ceres (no generated protobuf classes are used).
*/

#endregion

#region Using directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

#endregion

namespace Ceres.Chess.NNBackends.LC0EX
{
  /// <summary>
  /// Kernel ABI parameter types (lc0ex.proto ParameterType).
  /// </summary>
  public enum LC0EXParameterType
  {
    Unknown = 0,
    U32 = 1,
    Pointer = 2,
    NullPointer = 3,  // pointer-sized zero supplied by the runtime; consumes no Node.Argument
  }


  /// <summary>
  /// Buffer element types; values match ONNX TensorProto.DataType (lc0ex.proto Buffer.DataType).
  /// </summary>
  public enum LC0EXDataType
  {
    Unknown = 0,
    F32 = 1,
    U8 = 2,
    F16 = 10,
    U64 = 13,
    BF16 = 16,
  }


  /// <summary>
  /// A typed named memory range (persistent or per-program execution memory).
  /// </summary>
  public sealed record LC0EXBufferDef(string Name, ulong Offset, LC0EXDataType DataType, ulong[] Shape, long[] Strides)
  {
    public static int ElementSize(LC0EXDataType dt) => dt switch
    {
      LC0EXDataType.F32 => 4,
      LC0EXDataType.U8 => 1,
      LC0EXDataType.F16 => 2,
      LC0EXDataType.U64 => 8,
      LC0EXDataType.BF16 => 2,
      _ => throw new NotSupportedException($"Unsupported lc0ex buffer data type {dt}")
    };

    public ulong NumElements => Shape.Aggregate(1UL, (a, b) => a * b);

    public ulong SizeBytes => NumElements * (ulong)ElementSize(DataType);

    /// <summary>
    /// True if no layout is given or the layout equals the default row-major strides.
    /// </summary>
    public bool IsRowMajor
    {
      get
      {
        if (Strides == null || Strides.Length == 0)
        {
          return true;
        }
        if (Strides.Length != Shape.Length)
        {
          return false;
        }
        long expected = 1;
        for (int i = Shape.Length - 1; i >= 0; i--)
        {
          if (Strides[i] != expected)
          {
            return false;
          }
          expected *= (long)Shape[i];
        }
        return true;
      }
    }


    /// <summary>
    /// Recognizes a "pitched" layout: contiguous innermost dimension (stride 1), rows separated
    /// by a pitch at least the row length, and any leading dimensions compact with respect to
    /// that pitch. This is how lc0ex packs several weight matrices side by side into one wider
    /// matrix (e.g. BT4's Q, K, V as column blocks of a [1024, 3072] QKV matrix).
    /// Returns false for row-major buffers and for any other layout.
    /// </summary>
    public bool TryGetPitched2D(out ulong rows, out ulong cols, out ulong pitchElements)
    {
      rows = cols = pitchElements = 0;
      if (IsRowMajor || Strides == null || Strides.Length != Shape.Length || Shape.Length < 2)
      {
        return false;
      }

      int r = Shape.Length;
      if (Strides[r - 1] != 1 || Strides[r - 2] < (long)Shape[r - 1])
      {
        return false;
      }

      for (int i = 0; i < r - 2; i++)
      {
        if (Strides[i] != Strides[i + 1] * (long)Shape[i + 1])
        {
          return false;
        }
      }

      cols = Shape[r - 1];
      pitchElements = (ulong)Strides[r - 2];
      rows = 1;
      for (int i = 0; i < r - 1; i++)
      {
        rows *= Shape[i];
      }
      return true;
    }


    /// <summary>
    /// Bytes spanned in memory from the buffer offset (larger than SizeBytes for pitched layouts).
    /// </summary>
    public ulong FootprintBytes
      => TryGetPitched2D(out ulong rows, out ulong cols, out ulong pitch)
           ? (rows == 0 ? 0 : ((rows - 1) * pitch + cols) * (ulong)ElementSize(DataType))
           : SizeBytes;
  }


  public sealed record LC0EXKernelDef(int BinaryIndex, string Function, LC0EXParameterType[] Parameters, ulong RuntimeNS);


  public enum LC0EXArgumentKind { Parameter, Allocation, Symbol }

  public enum LC0EXAllocationKind { Unknown = 0, Persistent = 1, Execution = 2 }


  public sealed record LC0EXArgumentDef(LC0EXArgumentKind Kind, string ParameterName,
                                        LC0EXAllocationKind AllocationKind, ulong AllocationOffset,
                                        int SymbolBinaryIndex, string SymbolName);


  public sealed record LC0EXNodeDef(int KernelIndex, int[] Dependencies, LC0EXArgumentDef[] Arguments,
                                    uint[] Grid, uint[] Block, uint DynamicSharedMemoryBytes);


  public sealed record LC0EXProgramDef(string Name, int BatchSize, ulong ExecutionSizeBytes, ulong ExecutionAlignment,
                                       Dictionary<string, LC0EXBufferDef> Buffers, LC0EXNodeDef[] Nodes);


  /// <summary>
  /// Decoded contents of a .lc0ex "neural executable" file (see lc0 proto/lc0ex.proto).
  /// The file holds cubins and a kernel launch graph per program; it holds no weights.
  /// </summary>
  public sealed class LC0EXExecutableDef
  {
    public const uint EXPECTED_MAGIC = 0x1c0e;
    public const uint SUPPORTED_FORMAT = 1;

    public string FileName { get; private set; }
    public uint Magic { get; private set; }
    public uint Format { get; private set; }

    /// <summary>
    /// Opaque metadata; in practice a serialized pblczero.Net architecture fingerprint.
    /// </summary>
    public byte[] Metadata { get; private set; }

    public int TargetVendor { get; private set; }
    public string TargetArchitecture { get; private set; }

    /// <summary>
    /// The embedded cubins. An entry is set to null once it has been loaded onto a device
    /// (see LC0EXExecutable), so the images are not kept on the managed heap.
    /// </summary>
    public List<byte[]> Binaries { get; } = new();
    public List<(string Name, LC0EXParameterType Type)> RuntimeParameters { get; } = new();
    public List<LC0EXKernelDef> Kernels { get; } = new();

    /// <summary>
    /// Programs sorted by ascending batch size.
    /// </summary>
    public List<LC0EXProgramDef> Programs { get; private set; } = new();

    public ulong PersistentSizeBytes { get; private set; }
    public ulong PersistentAlignment { get; private set; } = 1;
    public List<LC0EXBufferDef> PersistentBuffers { get; } = new();


    /// <summary>
    /// Loads and decodes the specified .lc0ex file.
    /// </summary>
    public static LC0EXExecutableDef Load(string fileName)
    {
      if (!File.Exists(fileName))
      {
        throw new FileNotFoundException($"lc0ex executable file not found: {fileName}");
      }

      LC0EXExecutableDef def = Parse(File.ReadAllBytes(fileName));
      def.FileName = fileName;
      return def;
    }


    public static LC0EXExecutableDef Parse(byte[] data)
    {
      LC0EXExecutableDef def = new();
      LC0EXProtoReader r = new(data);
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: def.Magic = wt == LC0EXProtoReader.WIRE_FIXED32 ? r.ReadFixed32() : (uint)r.ReadVarint(); break;
          case 2: def.Format = checked((uint)r.ReadVarint()); break;
          case 3: def.Metadata = r.ReadBytesCopy(); break;
          case 4: ParseTarget(r.ReadMessage(), def); break;
          case 10: def.Binaries.Add(ParseBinary(r.ReadMessage())); break;
          case 12: def.RuntimeParameters.Add(ParseParameter(r.ReadMessage())); break;
          case 13: def.PersistentBuffers.Add(ParseBuffer(r.ReadMessage())); break;
          case 14: def.Kernels.Add(ParseKernel(r.ReadMessage())); break;
          case 15: def.Programs.Add(ParseProgram(r.ReadMessage())); break;
          case 16:
            (ulong size, ulong align) = ParseAllocation(r.ReadMessage());
            def.PersistentSizeBytes = size;
            def.PersistentAlignment = align;
            break;
          default: r.Skip(wt); break;
        }
      }

      if (def.Magic != EXPECTED_MAGIC)
      {
        throw new Exception($"Not an lc0ex executable (magic 0x{def.Magic:x}, expected 0x{EXPECTED_MAGIC:x}).");
      }
      if (def.Format != SUPPORTED_FORMAT)
      {
        throw new Exception($"Unsupported lc0ex executable format generation {def.Format} (this Ceres build supports {SUPPORTED_FORMAT}).");
      }
      if (def.Programs.Count == 0)
      {
        throw new Exception("lc0ex executable contains no programs.");
      }

      def.Programs = def.Programs.OrderBy(p => p.BatchSize).ToList();
      if (def.Programs.Select(p => p.BatchSize).Distinct().Count() != def.Programs.Count)
      {
        throw new Exception("lc0ex executable contains multiple programs with the same batch size.");
      }

      return def;
    }


    static void ParseTarget(LC0EXProtoReader r, LC0EXExecutableDef def)
    {
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: def.TargetVendor = (int)r.ReadVarint(); break;
          case 2: def.TargetArchitecture = r.ReadString(); break;
          default: r.Skip(wt); break;
        }
      }
    }


    static byte[] ParseBinary(LC0EXProtoReader r)
    {
      int format = 0;
      byte[] data = null;
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: format = (int)r.ReadVarint(); break;
          case 2: data = r.ReadBytesCopy(); break;
          default: r.Skip(wt); break;
        }
      }

      const int FORMAT_CUBIN = 1;
      if (format != FORMAT_CUBIN)
      {
        throw new NotSupportedException($"lc0ex binary format {format} not supported (only CUBIN).");
      }
      return data ?? throw new FormatException("lc0ex binary without data.");
    }


    static (string, LC0EXParameterType) ParseParameter(LC0EXProtoReader r)
    {
      string name = null;
      LC0EXParameterType type = LC0EXParameterType.Unknown;
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: name = r.ReadString(); break;
          case 2: type = (LC0EXParameterType)r.ReadVarint(); break;
          default: r.Skip(wt); break;
        }
      }
      return (name, type);
    }


    static (ulong size, ulong align) ParseAllocation(LC0EXProtoReader r)
    {
      ulong size = 0;
      ulong align = 1;
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: size = r.ReadVarint(); break;
          case 2: align = r.ReadVarint(); break;
          default: r.Skip(wt); break;
        }
      }

      if (align == 0 || (align & (align - 1)) != 0)
      {
        throw new FormatException($"lc0ex allocation alignment {align} is not a nonzero power of two.");
      }
      return (size, align);
    }


    static LC0EXBufferDef ParseBuffer(LC0EXProtoReader r)
    {
      string name = null;
      ulong offset = 0;
      LC0EXDataType dataType = LC0EXDataType.Unknown;
      List<ulong> shape = new();
      List<ulong> strides = null;
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: name = r.ReadString(); break;
          case 2: offset = r.ReadVarint(); break;
          case 3: dataType = (LC0EXDataType)r.ReadVarint(); break;
          case 4: r.ReadRepeatedVarint(wt, shape); break;
          case 5:
            strides = new List<ulong>();
            LC0EXProtoReader layout = r.ReadMessage();
            while (layout.Next(out int lf, out int lwt))
            {
              if (lf == 1)
              {
                layout.ReadRepeatedVarint(lwt, strides);
              }
              else
              {
                layout.Skip(lwt);
              }
            }
            break;
          default: r.Skip(wt); break;
        }
      }

      return new LC0EXBufferDef(name, offset, dataType, shape.ToArray(),
                                strides?.Select(s => unchecked((long)s)).ToArray());
    }


    static LC0EXKernelDef ParseKernel(LC0EXProtoReader r)
    {
      int binaryIndex = 0;
      string function = null;
      List<ulong> parameters = new();
      ulong runtimeNS = 0;
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: binaryIndex = checked((int)r.ReadVarint()); break;
          case 2: function = r.ReadString(); break;
          case 3: r.ReadRepeatedVarint(wt, parameters); break;
          case 4: runtimeNS = r.ReadVarint(); break;
          default: r.Skip(wt); break;
        }
      }
      if (string.IsNullOrEmpty(function))
      {
        throw new FormatException("lc0ex kernel without a function name.");
      }
      return new LC0EXKernelDef(binaryIndex, function, parameters.Select(p => (LC0EXParameterType)p).ToArray(), runtimeNS);
    }


    static LC0EXProgramDef ParseProgram(LC0EXProtoReader r)
    {
      string name = null;
      int batchSize = 0;
      ulong execSize = 0;
      ulong execAlign = 1;
      Dictionary<string, LC0EXBufferDef> buffers = new();
      List<LC0EXNodeDef> nodes = new();

      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: name = r.ReadString(); break;
          case 2:
            // ProgramMetadata { required uint32 batch_size = 1; }
            LC0EXProtoReader meta = r.ReadMessage();
            while (meta.Next(out int mf, out int mwt))
            {
              if (mf == 1)
              {
                batchSize = checked((int)meta.ReadVarint());
              }
              else
              {
                meta.Skip(mwt);
              }
            }
            break;
          case 3: (execSize, execAlign) = ParseAllocation(r.ReadMessage()); break;
          case 4:
            LC0EXBufferDef buf = ParseBuffer(r.ReadMessage());
            if (string.IsNullOrEmpty(buf.Name))
            {
              throw new FormatException("lc0ex program buffer without a name.");
            }
            if (!buffers.TryAdd(buf.Name, buf))
            {
              throw new FormatException($"lc0ex program has more than one buffer named '{buf.Name}'.");
            }
            break;
          case 5: nodes.Add(ParseNode(r.ReadMessage())); break;
          default: r.Skip(wt); break;
        }
      }

      if (batchSize <= 0)
      {
        throw new FormatException($"lc0ex program '{name}' has no batch size in its metadata.");
      }
      return new LC0EXProgramDef(name, batchSize, execSize, execAlign, buffers, nodes.ToArray());
    }


    static LC0EXNodeDef ParseNode(LC0EXProtoReader r)
    {
      int kernelIndex = 0;
      List<ulong> deps = new();
      List<LC0EXArgumentDef> args = new();
      List<ulong> grid = new();
      List<ulong> block = new();
      uint smem = 0;
      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: kernelIndex = checked((int)r.ReadVarint()); break;
          case 2: r.ReadRepeatedVarint(wt, deps); break;
          case 3: args.Add(ParseArgument(r.ReadMessage())); break;
          case 4: r.ReadRepeatedVarint(wt, grid); break;
          case 5: r.ReadRepeatedVarint(wt, block); break;
          case 6: smem = checked((uint)r.ReadVarint()); break;
          default: r.Skip(wt); break;
        }
      }

      static uint[] Dim3(List<ulong> v, string what)
      {
        if (v.Count < 1 || v.Count > 3)
        {
          throw new FormatException($"lc0ex node {what} must have 1 to 3 dimensions, has {v.Count}.");
        }
        uint[] d = { 1, 1, 1 };
        for (int i = 0; i < v.Count; i++)
        {
          d[i] = checked((uint)v[i]);
        }
        return d;
      }

      return new LC0EXNodeDef(kernelIndex, deps.Select(d => checked((int)d)).ToArray(), args.ToArray(),
                              Dim3(grid, "grid"), Dim3(block, "block"), smem);
    }


    static LC0EXArgumentDef ParseArgument(LC0EXProtoReader r)
    {
      string parameterName = null;
      bool hasAllocation = false;
      LC0EXAllocationKind allocKind = LC0EXAllocationKind.Unknown;
      ulong allocOffset = 0;
      bool hasSymbol = false;
      int symbolBinary = 0;
      string symbolName = null;

      while (r.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: parameterName = r.ReadString(); break;
          case 2:
            hasAllocation = true;
            LC0EXProtoReader a = r.ReadMessage();
            while (a.Next(out int af, out int awt))
            {
              if (af == 1) allocKind = (LC0EXAllocationKind)a.ReadVarint();
              else if (af == 2) allocOffset = a.ReadVarint();
              else a.Skip(awt);
            }
            break;
          case 3:
            hasSymbol = true;
            LC0EXProtoReader s = r.ReadMessage();
            while (s.Next(out int sf, out int swt))
            {
              if (sf == 1) symbolBinary = checked((int)s.ReadVarint());
              else if (sf == 2) symbolName = s.ReadString();
              else s.Skip(swt);
            }
            break;
          default: r.Skip(wt); break;
        }
      }

      int numSources = (hasAllocation ? 1 : 0) + (hasSymbol ? 1 : 0) + (parameterName != null ? 1 : 0);
      if (numSources != 1)
      {
        throw new FormatException($"lc0ex node argument must have exactly one source, has {numSources}.");
      }

      if (hasAllocation)
      {
        if (allocKind != LC0EXAllocationKind.Persistent && allocKind != LC0EXAllocationKind.Execution)
        {
          throw new FormatException($"lc0ex argument has unknown allocation kind {allocKind}.");
        }
        return new LC0EXArgumentDef(LC0EXArgumentKind.Allocation, null, allocKind, allocOffset, 0, null);
      }
      if (hasSymbol)
      {
        return new LC0EXArgumentDef(LC0EXArgumentKind.Symbol, null, LC0EXAllocationKind.Unknown, 0, symbolBinary, symbolName);
      }
      return new LC0EXArgumentDef(LC0EXArgumentKind.Parameter, parameterName, LC0EXAllocationKind.Unknown, 0, 0, null);
    }


    /// <summary>
    /// Returns the target architecture as (major, minor) compute capability, e.g. "sm_90a" -> (9, 0).
    /// </summary>
    public (int major, int minor) TargetComputeCapability
    {
      get
      {
        string arch = TargetArchitecture ?? "";
        string digits = new string(arch.SkipWhile(c => !char.IsDigit(c)).TakeWhile(char.IsDigit).ToArray());
        if (!arch.StartsWith("sm_") || digits.Length < 2)
        {
          throw new FormatException($"Unrecognized lc0ex target architecture '{arch}'.");
        }
        int value = int.Parse(digits);
        return (value / 10, value % 10);
      }
    }


    public override string ToString()
      => $"<LC0EXExecutableDef {Path.GetFileName(FileName)} {TargetArchitecture} binaries={Binaries.Count} kernels={Kernels.Count} "
       + $"programs=[{string.Join(",", Programs.Select(p => p.BatchSize))}] persistent={PersistentSizeBytes / (1024 * 1024)}MB>";
  }
}
