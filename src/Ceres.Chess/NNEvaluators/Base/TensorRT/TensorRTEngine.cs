#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Managed wrapper for a TensorRT engine and execution context.
/// </summary>
public sealed class TensorRTEngine : IDisposable
{
  /// <summary>
  /// Batch size this engine was built for.
  /// </summary>
  public int BatchSize;

  /// <summary>
  /// Number of input tensors.
  /// </summary>
  public int NumInputs { get; }

  /// <summary>
  /// Number of output tensors.
  /// </summary>
  public int NumOutputs { get; }

  /// <summary>
  /// Total input size in elements across all input tensors.
  /// </summary>
  public long TotalInputSize { get; }

  /// <summary>
  /// Total output size in elements across all output tensors.
  /// </summary>
  public long TotalOutputSize { get; }

  /// <summary>
  /// Source file path (ONNX or engine file).
  /// </summary>
  public string SourcePath { get; }

  /// <summary>
  /// Whether this engine was loaded from cache.
  /// </summary>
  public bool WasLoadedFromCache { get; }

  /// <summary>
  /// Internal handle for native interop.
  /// </summary>
  internal IntPtr Handle => handle;

  private IntPtr handle;
  private bool _disposed;


  /// <summary>
  /// Private constructor that initializes from handle.
  /// </summary>
  private TensorRTEngine(IntPtr handle, int batchSize, string sourcePath, bool wasLoadedFromCache)
  {
    this.handle = handle;
    BatchSize = batchSize;
    SourcePath = sourcePath;
    WasLoadedFromCache = wasLoadedFromCache;

    NumInputs = TensorRTNative.GetNumInputs(this.handle);
    NumOutputs = TensorRTNative.GetNumOutputs(this.handle);

    TotalInputSize = 0;
    for (int i = 0; i < NumInputs; i++)
    {
      TotalInputSize += TensorRTNative.GetInputSize(this.handle, i);
    }

    TotalOutputSize = 0;
    for (int i = 0; i < NumOutputs; i++)
    {
      TotalOutputSize += TensorRTNative.GetOutputSize(this.handle, i);
    }
  }


  /// <summary>
  /// Load engine from ONNX file (legacy constructor for compatibility).
  /// </summary>
  public TensorRTEngine(string onnxPath, int batchSize, TensorRTBuildOptions? options = null, int deviceId = -1)
  {
    BatchSize = batchSize;
    SourcePath = onnxPath;
    WasLoadedFromCache = false;

    if (deviceId >= 0 && options.HasValue)
    {
      TensorRTBuildOptions opts = options.Value;
      handle = TensorRTNative.LoadONNXOnDevice(onnxPath, batchSize, ref opts, deviceId);
    }
    else if (options.HasValue)
    {
      TensorRTBuildOptions opts = options.Value;
      handle = TensorRTNative.LoadONNXWithOptions(onnxPath, batchSize, ref opts);
    }
    else
    {
      handle = TensorRTNative.LoadONNX(onnxPath, batchSize);
    }

    if (handle == IntPtr.Zero)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"Failed to load ONNX: {error ?? "unknown error"}");
    }

    NumInputs = TensorRTNative.GetNumInputs(handle);
    NumOutputs = TensorRTNative.GetNumOutputs(handle);

    TotalInputSize = 0;
    for (int i = 0; i < NumInputs; i++)
    {
      TotalInputSize += TensorRTNative.GetInputSize(handle, i);
    }

    TotalOutputSize = 0;
    for (int i = 0; i < NumOutputs; i++)
    {
      TotalOutputSize += TensorRTNative.GetOutputSize(handle, i);
    }
  }


  /// <summary>
  /// Load a pre-built engine file (.engine).
  /// </summary>
  public static TensorRTEngine LoadEngineFile(string enginePath, int batchSize, int deviceId = -1)
  {
    IntPtr handle = TensorRTNative.LoadEngineFile(enginePath, batchSize, deviceId);
    if (handle == IntPtr.Zero)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"Failed to load engine file: {error ?? "unknown error"}");
    }
    return new TensorRTEngine(handle, batchSize, enginePath, false);
  }


  /// <summary>
  /// Load ONNX with caching support. If cached engine exists and is valid, loads from cache.
  /// Otherwise builds from ONNX and caches the result.
  /// </summary>
  public static TensorRTEngine LoadWithCache(string onnxPath, int batchSize, TensorRTBuildOptions options,
                                              int deviceId = -1, string cacheDir = null, bool forceRebuild = false)
  {
    IntPtr handle = TensorRTNative.LoadONNXCached(onnxPath, batchSize, ref options, deviceId,
                                                   cacheDir, forceRebuild ? 1 : 0, out int wasCached);
    if (handle == IntPtr.Zero)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"Failed to load ONNX: {error ?? "unknown error"}");
    }
    return new TensorRTEngine(handle, batchSize, onnxPath, wasCached != 0);
  }


  /// <summary>
  /// Build a single multi-profile TensorRT engine with shared weights,
  /// returning one TensorRTEngine per batch size. Each engine wraps an
  /// independent execution context but shares the underlying ICudaEngine.
  /// </summary>
  public static unsafe TensorRTEngine[] LoadMultiProfileWithCache(string onnxPath, int[] batchSizes,
      TensorRTBuildOptions options, int deviceId = -1, string cacheDir = null, bool forceRebuild = false)
  {
    int numProfiles = batchSizes.Length;
    IntPtr* handles = stackalloc IntPtr[numProfiles];

    int wasCached;
    int result;
    fixed (int* sizesPtr = batchSizes)
    {
      result = TensorRTNative.LoadONNXMultiProfileCached(onnxPath, sizesPtr, numProfiles,
          ref options, deviceId, cacheDir, forceRebuild ? 1 : 0, out wasCached, handles);
    }

    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"Failed to load multi-profile ONNX ({result}): {error ?? "unknown error"}");
    }

    TensorRTEngine[] engines = new TensorRTEngine[numProfiles];
    for (int i = 0; i < numProfiles; i++)
    {
      engines[i] = new TensorRTEngine(handles[i], batchSizes[i], onnxPath, wasCached != 0);
    }
    return engines;
  }


  /// <summary>
  /// Load a pre-built multi-profile engine file (.engine) directly.
  /// Deserializes the engine and creates N execution contexts, one per batch size.
  /// Bypasses ONNX parsing and cache validation — useful for loading
  /// pre-refitted engines produced by external tooling (e.g., Python TensorRT).
  /// </summary>
  public static unsafe TensorRTEngine[] LoadMultiProfileEngineFile(string enginePath, int[] batchSizes,
      bool useCudaGraphs = false, bool useSpinWait = true, int deviceId = -1)
  {
    int numProfiles = batchSizes.Length;
    IntPtr* handles = stackalloc IntPtr[numProfiles];

    int result;
    fixed (int* sizesPtr = batchSizes)
    {
      result = TensorRTNative.LoadMultiProfileEngineFile(enginePath, sizesPtr, numProfiles,
          useCudaGraphs ? 1 : 0, useSpinWait ? 1 : 0, deviceId, handles);
    }

    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"Failed to load multi-profile engine file ({result}): {error ?? "unknown error"}");
    }

    TensorRTEngine[] engines = new TensorRTEngine[numProfiles];
    for (int i = 0; i < numProfiles; i++)
    {
      engines[i] = new TensorRTEngine(handles[i], batchSizes[i], enginePath, wasLoadedFromCache: true);
    }
    return engines;
  }


  /// <summary>
  /// Load from either ONNX or engine file based on file extension, with optional caching.
  /// </summary>
  public static TensorRTEngine Load(string path, int batchSize, TensorRTBuildOptions? options = null,
                                    int deviceId = -1, string cacheDir = null, bool forceRebuild = false)
  {
    string ext = Path.GetExtension(path).ToLowerInvariant();

    if (ext == ".engine" || ext == ".plan")
    {
      return LoadEngineFile(path, batchSize, deviceId);
    }
    else if (ext == ".onnx")
    {
      TensorRTBuildOptions opts = options ?? TensorRTBuildOptions.Default;
      return LoadWithCache(path, batchSize, opts, deviceId, cacheDir, forceRebuild);
    }
    else
    {
      throw new ArgumentException($"Unsupported file extension: {ext}. Expected .onnx, .engine, or .plan");
    }
  }


  /// <summary>
  /// Save the engine to a file for later reuse.
  /// </summary>
  public void SaveEngine(string enginePath)
  {
    int result = TensorRTNative.SaveEngine(handle, enginePath);
    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"Failed to save engine: {error ?? "unknown error"}");
    }
  }


  /// <summary>
  /// Get the cache filename that would be used for this ONNX file with given options.
  /// </summary>
  public static string GetCacheFilename(string onnxPath, int batchSize, TensorRTBuildOptions options, int deviceId = -1)
  {
    IntPtr ptr = TensorRTNative.GenerateCacheFilenameForDevice(onnxPath, batchSize, ref options, deviceId);
    if (ptr == IntPtr.Zero)
    {
      throw new InvalidOperationException("Failed to generate cache filename");
    }
    string filename = Marshal.PtrToStringAnsi(ptr) ?? "";
    TensorRTNative.FreeString(ptr);
    return filename;
  }


  /// <summary>
  /// Get number of layers in the engine.
  /// </summary>
  public int GetNumLayers() => TensorRTNative.GetNumLayers(handle);



  /// <summary>
  /// Get layer information as JSON string.
  /// </summary>
  public string GetLayerInfo(int layerIndex)
  {
    IntPtr ptr = TensorRTNative.GetEngineLayerInfo(handle, layerIndex);
    if (ptr == IntPtr.Zero)
    {
      return null;
    }
    string info = Marshal.PtrToStringAnsi(ptr) ?? "";
    TensorRTNative.FreeString(ptr);
    return info;
  }


  /// <summary>
  /// Get engine summary as JSON string (includes all layer info).
  /// </summary>
  public string GetEngineSummary()
  {
    IntPtr ptr = TensorRTNative.GetEngineSummary(handle);
    if (ptr == IntPtr.Zero)
    {
      return null;
    }
    string info = Marshal.PtrToStringAnsi(ptr) ?? "";
    TensorRTNative.FreeString(ptr);
    return info;
  }


  /// <summary>
  /// Get input tensor name by index.
  /// </summary>
  public string GetInputName(int index) => TensorRTNative.PtrToString(TensorRTNative.GetInputName(handle, index));

  /// <summary>
  /// Get output tensor name by index.
  /// </summary>
  public string GetOutputName(int index) => TensorRTNative.PtrToString(TensorRTNative.GetOutputName(handle, index));

  /// <summary>
  /// Get input tensor size in elements by index.
  /// </summary>
  public long GetInputSize(int index) => TensorRTNative.GetInputSize(handle, index);

  /// <summary>
  /// Get output tensor size in elements by index.
  /// </summary>
  public long GetOutputSize(int index) => TensorRTNative.GetOutputSize(handle, index);

  /// <summary>
  /// Get element size in bytes for an input tensor (1=INT8/UINT8, 2=FP16/BF16, 4=FP32/INT32, 8=INT64).
  /// </summary>
  public int GetInputElementSize(int index) => TensorRTNative.GetInputElementSize(handle, index);

  /// <summary>
  /// Get element size in bytes for an output tensor.
  /// </summary>
  public int GetOutputElementSize(int index) => TensorRTNative.GetOutputElementSize(handle, index);

  /// <summary>
  /// Returns true if the first input tensor expects byte (INT8/UINT8) inputs.
  /// </summary>
  public bool HasByteInput => NumInputs > 0 && GetInputElementSize(0) == 1;

  /// <summary>
  /// Returns true if this engine uses CUDA graphs for inference.
  /// </summary>
  public bool UsesCudaGraphs => TensorRTNative.UsesCudaGraphs(handle) == 1;

  /// <summary>
  /// Returns true if the CUDA graph for the specified stream has already been captured.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  public bool IsStreamGraphCaptured(int streamIdx) => TensorRTNative.IsStreamGraphCaptured(handle, streamIdx) == 1;


  /// <summary>
  /// Get metadata for all output tensors (up to 16).
  /// </summary>
  public unsafe OutputTensorInfo[] GetOutputTensorInfo()
  {
    const int MaxOutputs = 16;
    NativeOutputTensorInfo* infos = stackalloc NativeOutputTensorInfo[MaxOutputs];

    int count = TensorRTNative.GetOutputTensorInfo(handle, infos, MaxOutputs);
    if (count < 0)
    {
      throw new InvalidOperationException("Failed to get output tensor info");
    }

    OutputTensorInfo[] result = new OutputTensorInfo[count];
    for (int i = 0; i < count; i++)
    {
      string name = Marshal.PtrToStringAnsi(infos[i].Name) ?? $"output_{i}";
      result[i] = new OutputTensorInfo(name, infos[i].Offset, infos[i].Size);
    }

    return result;
  }


  /// <summary>
  /// Extract individual output tensors from a flat output buffer.
  /// </summary>
  public Dictionary<string, Half[]> ExtractOutputTensors(Half[] flatOutput)
  {
    OutputTensorInfo[] infos = GetOutputTensorInfo();
    Dictionary<string, Half[]> result = new Dictionary<string, Half[]>(infos.Length);

    foreach (OutputTensorInfo info in infos)
    {
      Half[] data = new Half[info.Size];
      Array.Copy(flatOutput, info.Offset, data, 0, info.Size);
      result[info.Name] = data;
    }

    return result;
  }


  /// <summary>
  /// Extract individual output tensors as ReadOnlySpan views (no copy).
  /// </summary>
  public IEnumerable<(string Name, ReadOnlyMemory<Half> Data)> GetOutputTensorsAsSpans(Half[] flatOutput)
  {
    OutputTensorInfo[] infos = GetOutputTensorInfo();
    foreach (OutputTensorInfo info in infos)
    {
      yield return (info.Name, flatOutput.AsMemory((int)info.Offset, (int)info.Size));
    }
  }


  /// <summary>
  /// Run inference with Half (FP16) inputs and outputs.
  /// </summary>
  public unsafe void InferHost(Half[] input, Half[] output)
  {
    if (input.Length != TotalInputSize)
    {
      throw new ArgumentException($"Input size mismatch: expected {TotalInputSize}, got {input.Length}");
    }
    if (output.Length != TotalOutputSize)
    {
      throw new ArgumentException($"Output size mismatch: expected {TotalOutputSize}, got {output.Length}");
    }

    fixed (Half* inputPtr = input)
    {
      fixed (Half* outputPtr = output)
      {
        int result = TensorRTNative.InferHost(handle, inputPtr, TotalInputSize, outputPtr, TotalOutputSize);
        if (result != 0)
        {
          string error = TensorRTNative.GetLastErrorString();
          throw new InvalidOperationException($"Inference failed ({result}): {error ?? "unknown error"}");
        }
      }
    }
  }


  /// <summary>
  /// Run inference with byte inputs (for -I8 models with squares_byte input tensor).
  /// Output is still FP16.
  /// </summary>
  public unsafe void InferHostBytes(byte[] input, Half[] output)
  {
    if (input.Length != TotalInputSize)
    {
      throw new ArgumentException($"Input size mismatch: expected {TotalInputSize}, got {input.Length}");
    }
    if (output.Length != TotalOutputSize)
    {
      throw new ArgumentException($"Output size mismatch: expected {TotalOutputSize}, got {output.Length}");
    }

    fixed (byte* inputPtr = input)
    {
      fixed (Half* outputPtr = output)
      {
        int result = TensorRTNative.InferHostBytes(handle, inputPtr, TotalInputSize, outputPtr, TotalOutputSize);
        if (result != 0)
        {
          string error = TensorRTNative.GetLastErrorString();
          throw new InvalidOperationException($"Inference failed ({result}): {error ?? "unknown error"}");
        }
      }
    }
  }


  /// <summary>
  /// Gets the number of input elements per position (total input elements / batch size).
  /// </summary>
  public long InputElementsPerPosition => TotalInputSize / BatchSize;

  /// <summary>
  /// Gets the number of output elements per position (total output elements / batch size).
  /// </summary>
  public long OutputElementsPerPosition => TotalOutputSize / BatchSize;


  /// <summary>
  /// Run inference with dynamic batch size (for range-mode engines).
  /// Only copies/processes the actual number of positions specified.
  /// Buffers may be larger than needed - only the first actualInputElements/actualOutputElements are used.
  /// </summary>
  /// <param name="input">Input buffer (may be oversized)</param>
  /// <param name="output">Output buffer (may be oversized)</param>
  /// <param name="actualBatchSize">The actual number of positions to process</param>
  /// <param name="actualInputElements">Actual input elements to use (must be &lt;= input.Length)</param>
  /// <param name="actualOutputElements">Actual output elements to use (must be &lt;= output.Length)</param>
  public unsafe void InferHostDynamic(Half[] input, Half[] output, int actualBatchSize,
                                       long actualInputElements, long actualOutputElements)
  {
    if (input.Length < actualInputElements)
    {
      throw new ArgumentException($"Input buffer too small: need {actualInputElements}, got {input.Length}");
    }
    if (output.Length < actualOutputElements)
    {
      throw new ArgumentException($"Output buffer too small: need {actualOutputElements}, got {output.Length}");
    }

    fixed (Half* inputPtr = input)
    {
      fixed (Half* outputPtr = output)
      {
        int result = TensorRTNative.InferHostDynamic(handle, inputPtr, actualInputElements, outputPtr, actualOutputElements, actualBatchSize);
        if (result != 0)
        {
          string error = TensorRTNative.GetLastErrorString();
          throw new InvalidOperationException($"Dynamic inference failed ({result}): {error ?? "unknown error"}");
        }
      }
    }
  }

  /// <summary>
  /// Run inference with dynamic batch size (for range-mode engines).
  /// Only copies/processes the actual number of positions specified.
  /// Buffers must be exactly sized for the batch.
  /// </summary>
  /// <param name="input">Input buffer sized for actualBatchSize positions</param>
  /// <param name="output">Output buffer sized for actualBatchSize positions</param>
  /// <param name="actualBatchSize">The actual number of positions to process</param>
  public void InferHostDynamic(Half[] input, Half[] output, int actualBatchSize)
  {
    long expectedInputSize = InputElementsPerPosition * actualBatchSize;
    long expectedOutputSize = OutputElementsPerPosition * actualBatchSize;

    if (input.Length != expectedInputSize)
    {
      throw new ArgumentException($"Input size mismatch: expected {expectedInputSize} for batch {actualBatchSize}, got {input.Length}");
    }
    if (output.Length != expectedOutputSize)
    {
      throw new ArgumentException($"Output size mismatch: expected {expectedOutputSize} for batch {actualBatchSize}, got {output.Length}");
    }

    InferHostDynamic(input, output, actualBatchSize, expectedInputSize, expectedOutputSize);
  }


  /// <summary>
  /// Run inference with byte inputs and dynamic batch size (for range-mode engines).
  /// Only copies/processes the actual number of positions specified.
  /// Buffers may be larger than needed - only the first actualInputElements/actualOutputElements are used.
  /// </summary>
  /// <param name="input">Input buffer (may be oversized)</param>
  /// <param name="output">Output buffer (may be oversized)</param>
  /// <param name="actualBatchSize">The actual number of positions to process</param>
  /// <param name="actualInputElements">Actual input elements to use (must be &lt;= input.Length)</param>
  /// <param name="actualOutputElements">Actual output elements to use (must be &lt;= output.Length)</param>
  public unsafe void InferHostBytesDynamic(byte[] input, Half[] output, int actualBatchSize,
                                            long actualInputElements, long actualOutputElements)
  {
    if (input.Length < actualInputElements)
    {
      throw new ArgumentException($"Input buffer too small: need {actualInputElements}, got {input.Length}");
    }
    if (output.Length < actualOutputElements)
    {
      throw new ArgumentException($"Output buffer too small: need {actualOutputElements}, got {output.Length}");
    }

    fixed (byte* inputPtr = input)
    {
      fixed (Half* outputPtr = output)
      {
        int result = TensorRTNative.InferHostBytesDynamic(handle, inputPtr, actualInputElements, outputPtr, actualOutputElements, actualBatchSize);
        if (result != 0)
        {
          string error = TensorRTNative.GetLastErrorString();
          throw new InvalidOperationException($"Dynamic inference failed ({result}): {error ?? "unknown error"}");
        }
      }
    }
  }

  /// <summary>
  /// Run inference with byte inputs and dynamic batch size (for range-mode engines).
  /// Only copies/processes the actual number of positions specified.
  /// Buffers must be exactly sized for the batch.
  /// </summary>
  /// <param name="input">Input buffer sized for actualBatchSize positions</param>
  /// <param name="output">Output buffer sized for actualBatchSize positions</param>
  /// <param name="actualBatchSize">The actual number of positions to process</param>
  public void InferHostBytesDynamic(byte[] input, Half[] output, int actualBatchSize)
  {
    long expectedInputSize = InputElementsPerPosition * actualBatchSize;
    long expectedOutputSize = OutputElementsPerPosition * actualBatchSize;

    if (input.Length != expectedInputSize)
    {
      throw new ArgumentException($"Input size mismatch: expected {expectedInputSize} for batch {actualBatchSize}, got {input.Length}");
    }
    if (output.Length != expectedOutputSize)
    {
      throw new ArgumentException($"Output size mismatch: expected {expectedOutputSize} for batch {actualBatchSize}, got {output.Length}");
    }

    InferHostBytesDynamic(input, output, actualBatchSize, expectedInputSize, expectedOutputSize);
  }


  #region Pipelined Sub-Batch Stream-Based Operations

  /// <summary>
  /// Asynchronously copies data from pinned host memory to GPU memory on the specified stream.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  /// <param name="gpuDst">GPU destination pointer</param>
  /// <param name="pinnedSrc">Pinned host source pointer</param>
  /// <param name="bytes">Number of bytes to copy</param>
  public void CopyToGPUOnStreamAsync(int streamIdx, IntPtr gpuDst, IntPtr pinnedSrc, long bytes)
  {
    int result = TensorRTNative.CopyToGPUOnStream(handle, streamIdx, gpuDst, pinnedSrc, bytes);
    if (result != 0)
    {
      throw new InvalidOperationException($"CopyToGPUOnStream failed on stream {streamIdx}");
    }
  }


  /// <summary>
  /// Asynchronously runs inference on the specified stream using pre-staged GPU buffers.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  /// <param name="gpuInput">GPU input buffer pointer</param>
  /// <param name="gpuOutput">GPU output buffer pointer</param>
  public void InferOnStreamAsync(int streamIdx, IntPtr gpuInput, IntPtr gpuOutput)
  {
    int result = TensorRTNative.InferOnStream(handle, streamIdx, gpuInput, gpuOutput);
    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"InferOnStream failed on stream {streamIdx}: {error ?? "unknown error"}");
    }
  }


  /// <summary>
  /// Asynchronously runs inference on the specified stream with CUDA graph support.
  /// On first call per stream, captures a CUDA graph. Subsequent calls replay the graph.
  /// For engines with useCudaGraphs=false, this behaves like InferOnStreamAsync.
  /// Use this for Exact mode engines where batch size is fixed.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  /// <param name="gpuInput">GPU input buffer pointer</param>
  /// <param name="gpuOutput">GPU output buffer pointer</param>
  public void InferOnStreamWithGraphAsync(int streamIdx, IntPtr gpuInput, IntPtr gpuOutput)
  {
    int result = TensorRTNative.InferOnStreamWithGraph(handle, streamIdx, gpuInput, gpuOutput);
    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"InferOnStreamWithGraph failed on stream {streamIdx}: {error ?? "unknown error"}");
    }
  }


  /// <summary>
  /// Asynchronously runs inference on the specified stream with CUDA graph support and proper locking.
  /// On first call per stream, acquires write lock on graphCaptureRWLock, captures a CUDA graph, then releases.
  /// Subsequent calls replay the graph without needing the write lock.
  /// For engines with useCudaGraphs=false, this behaves like InferOnStreamAsync.
  /// Use this for Exact mode engines where batch size is fixed.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  /// <param name="gpuInput">GPU input buffer pointer</param>
  /// <param name="gpuOutput">GPU output buffer pointer</param>
  /// <param name="graphCaptureRWLock">Reader/writer lock to acquire during graph capture (null to skip locking)</param>
  public void InferOnStreamWithGraphAsync(int streamIdx, IntPtr gpuInput, IntPtr gpuOutput,
                                          System.Threading.ReaderWriterLockSlim graphCaptureRWLock)
  {
    // Fast path: if graphs are disabled or already captured, no lock needed
    if (!UsesCudaGraphs || IsStreamGraphCaptured(streamIdx))
    {
      int result = TensorRTNative.InferOnStreamWithGraph(handle, streamIdx, gpuInput, gpuOutput);
      if (result != 0)
      {
        string error = TensorRTNative.GetLastErrorString();
        throw new InvalidOperationException($"InferOnStreamWithGraph failed on stream {streamIdx}: {error ?? "unknown error"}");
      }
      return;
    }

    // Slow path: graph capture may be needed - must acquire lock and re-check
    // This avoids TOCTOU race where multiple threads see needsCapture=true before any acquires the lock
    if (graphCaptureRWLock != null)
    {
      graphCaptureRWLock.EnterWriteLock();
      try
      {
        // Re-check inside lock - another thread may have captured while we waited
        // The native InferOnStreamWithGraph will handle this gracefully (replay if captured)
        int result = TensorRTNative.InferOnStreamWithGraph(handle, streamIdx, gpuInput, gpuOutput);
        if (result != 0)
        {
          string error = TensorRTNative.GetLastErrorString();
          throw new InvalidOperationException($"InferOnStreamWithGraph failed on stream {streamIdx}: {error ?? "unknown error"}");
        }
      }
      finally
      {
        graphCaptureRWLock.ExitWriteLock();
      }
    }
    else
    {
      // No lock provided but capture needed - caller's responsibility to ensure no concurrent capture
      int result = TensorRTNative.InferOnStreamWithGraph(handle, streamIdx, gpuInput, gpuOutput);
      if (result != 0)
      {
        string error = TensorRTNative.GetLastErrorString();
        throw new InvalidOperationException($"InferOnStreamWithGraph failed on stream {streamIdx}: {error ?? "unknown error"}");
      }
    }
  }


  /// <summary>
  /// Asynchronously runs inference on the specified stream with dynamic batch size.
  /// Sets the input shape to the actual batch size before inference.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  /// <param name="gpuInput">GPU input buffer pointer</param>
  /// <param name="gpuOutput">GPU output buffer pointer</param>
  /// <param name="actualBatchSize">The actual number of positions to process</param>
  public void InferOnStreamDynamicAsync(int streamIdx, IntPtr gpuInput, IntPtr gpuOutput, int actualBatchSize)
  {
    int result = TensorRTNative.InferOnStreamDynamic(handle, streamIdx, gpuInput, gpuOutput, actualBatchSize);
    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"InferOnStreamDynamic failed on stream {streamIdx} for batch {actualBatchSize}: {error ?? "unknown error"}");
    }
  }


  /// <summary>
  /// Asynchronously copies data from GPU memory to pinned host memory on the specified stream.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  /// <param name="pinnedDst">Pinned host destination pointer</param>
  /// <param name="gpuSrc">GPU source pointer</param>
  /// <param name="bytes">Number of bytes to copy</param>
  public void CopyFromGPUOnStreamAsync(int streamIdx, IntPtr pinnedDst, IntPtr gpuSrc, long bytes)
  {
    int result = TensorRTNative.CopyFromGPUOnStream(handle, streamIdx, pinnedDst, gpuSrc, bytes);
    if (result != 0)
    {
      throw new InvalidOperationException($"CopyFromGPUOnStream failed on stream {streamIdx}");
    }
  }


  /// <summary>
  /// Synchronizes the specified CUDA stream, blocking until all operations complete.
  /// </summary>
  /// <param name="streamIdx">Stream index (0 or 1)</param>
  public void SyncStream(int streamIdx)
  {
    int result = TensorRTNative.SyncStreamIdx(handle, streamIdx);
    if (result != 0)
    {
      throw new InvalidOperationException($"SyncStreamIdx failed on stream {streamIdx}");
    }
  }

  #endregion


  #region Weight Refitting

  /// <summary>
  /// Sets weights for a named tensor in the engine and refits immediately.
  /// The engine must have been built with Refittable=1 in build options.
  /// </summary>
  /// <param name="weightTensorName">Name of the weight tensor to update (e.g., "conv1.weight")</param>
  /// <param name="weights">FP16 weight data to set</param>
  /// <exception cref="InvalidOperationException">If the engine was not built with refit support or the operation fails</exception>
  public unsafe void SetNamedWeights(string weightTensorName, Half[] weights)
  {
    if (string.IsNullOrEmpty(weightTensorName))
    {
      throw new ArgumentException("Weight tensor name cannot be null or empty", nameof(weightTensorName));
    }
    if (weights == null || weights.Length == 0)
    {
      throw new ArgumentException("Weights array cannot be null or empty", nameof(weights));
    }

    fixed (Half* weightsPtr = weights)
    {
      int result = TensorRTNative.SetNamedWeights(handle, weightTensorName, weightsPtr, weights.Length);
      if (result != 0)
      {
        string error = TensorRTNative.GetLastErrorString();
        throw new InvalidOperationException($"Failed to set named weights ({result}): {error ?? "unknown error"}");
      }
    }
  }

  /// <summary>
  /// Sets weights for a named tensor in the engine and refits immediately.
  /// The engine must have been built with Refittable=1 in build options.
  /// </summary>
  /// <param name="weightTensorName">Name of the weight tensor to update (e.g., "conv1.weight")</param>
  /// <param name="weights">FP16 weight data as a span</param>
  /// <exception cref="InvalidOperationException">If the engine was not built with refit support or the operation fails</exception>
  public unsafe void SetNamedWeights(string weightTensorName, ReadOnlySpan<Half> weights)
  {
    if (string.IsNullOrEmpty(weightTensorName))
    {
      throw new ArgumentException("Weight tensor name cannot be null or empty", nameof(weightTensorName));
    }
    if (weights.IsEmpty)
    {
      throw new ArgumentException("Weights span cannot be empty", nameof(weights));
    }

    fixed (Half* weightsPtr = weights)
    {
      int result = TensorRTNative.SetNamedWeights(handle, weightTensorName, weightsPtr, weights.Length);
      if (result != 0)
      {
        string error = TensorRTNative.GetLastErrorString();
        throw new InvalidOperationException($"Failed to set named weights ({result}): {error ?? "unknown error"}");
      }
    }
  }

  /// <summary>
  /// Refits the CUDA engine after weight updates.
  /// This is typically called automatically by SetNamedWeights, but can be called
  /// separately if multiple weight updates need to be batched.
  /// </summary>
  /// <exception cref="InvalidOperationException">If the engine was not built with refit support or the operation fails</exception>
  public void RefitEngine()
  {
    int result = TensorRTNative.RefitEngine(handle);
    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"Failed to refit engine ({result}): {error ?? "unknown error"}");
    }
  }

  #endregion


  /// <summary>
  /// Dispose the engine and release native resources.
  /// </summary>
  public void Dispose()
  {
    if (_disposed)
    {
      return;
    }
    _disposed = true;

    if (handle != IntPtr.Zero)
    {
      TensorRTNative.FreeEngine(handle);
      handle = IntPtr.Zero;
    }
  }
}
