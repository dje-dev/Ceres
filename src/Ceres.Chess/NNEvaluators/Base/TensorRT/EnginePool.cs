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
using System.Linq;
using System.Threading;

using Ceres.Base.CUDA;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

public sealed class EnginePool : IDisposable
{
  /// <summary>
  /// If true, outputs a line for each batch executed showing batch details.
  /// </summary>
  private const bool VERBOSE_DUMP_BATCHES = false;

  /// <summary>
  /// Number of concurrent compute streams for pipelined sub-batch processing.
  /// 1 = each sub-batch processed sequentially (H2D, compute, D2H, handler).
  /// 2 = sub-batches processed in pairs: each stream issues the full
  ///     H2D -> Compute -> D2H sequence asynchronously, enabling concurrent
  ///     compute on streams 0 and 2 (separate TRT execution contexts).
  /// </summary>
  private const int NUM_COMPUTE_STREAMS = 2;

  // Stream index constants (stream 1 reserved for future use)
  private const int STREAM_COMPUTE_A = 0;
  private const int STREAM_COMPUTE_B = 2;

  /// <summary>
  /// Maps buffer index [0..NUM_COMPUTE_STREAMS-1] to CUDA stream IDs.
  /// </summary>
  private static readonly int[] COMPUTE_STREAM_IDS = [STREAM_COMPUTE_A, STREAM_COMPUTE_B];

  /// <summary>
  /// If true, uses the optimized batch scheduling algorithm that considers
  /// actual execution times per batch size to minimize total inference time.
  /// When false, uses the legacy greedy algorithm with ExactModeBatchFractionAcceptablePadding.
  /// </summary>
  public static bool OPTIMIZED_SCHEDULING = true;


  private readonly TensorRT trt;
  private readonly bool ownsTrt;
  private readonly List<TensorRTEngine> engines = new();
  private readonly List<(int min, int max)> ranges = new();
  private readonly EnginePoolMode mode;
  private readonly string onnxPath;
  private readonly int deviceId;
  private readonly bool useByteInputs;
  private bool disposed;

  /// <summary>
  /// Flag indicating WarmupWithGraphCapture() has been called and completed.
  /// When true, all CUDA graphs are captured and pipelining can safely use direct replay calls.
  /// Marked volatile for thread-safe reads in the fast path.
  /// </summary>
  private volatile bool warmupCompleted;

  /// <summary>
  /// Groups pinned memory, GPU memory, and managed output buffers for one compute stream.
  /// Encapsulates the buffer lifecycle: allocate in constructor, free in Dispose.
  /// </summary>
  private readonly struct StreamBufferSet
  {
    public readonly IntPtr PinnedInput;
    public readonly IntPtr GpuInput;
    public readonly IntPtr PinnedOutput;
    public readonly IntPtr GpuOutput;
    public readonly Half[] ManagedOutput;

    public StreamBufferSet(long alignedInputBytes, long alignedOutputBytes, long maxOutputElements)
    {
      PinnedInput = TensorRTNative.AllocPinned(alignedInputBytes);
      GpuInput = TensorRTNative.AllocGPU(alignedInputBytes);
      PinnedOutput = TensorRTNative.AllocPinned(alignedOutputBytes);
      GpuOutput = TensorRTNative.AllocGPU(alignedOutputBytes);
      ManagedOutput = new Half[maxOutputElements];
    }

    public void Free()
    {
      TensorRTNative.FreePinned(PinnedInput);
      TensorRTNative.FreeGPU(GpuInput);
      TensorRTNative.FreePinned(PinnedOutput);
      TensorRTNative.FreeGPU(GpuOutput);
    }
  }

  // Per-stream buffer sets for pipelined sub-batch processing [0..NUM_COMPUTE_STREAMS-1].
  private StreamBufferSet[] streamBuffers;
  private long maxInputBytes;
  private long maxOutputBytes;

  // Per-position output element counts for each tensor (from largest engine).
  // Used to compute aligned total output size for any batch count.
  private int[] outputPerPosSizes;

  // Pre-allocated managed buffers to reduce GC pressure
  private byte[] cachedByteInputBuffer;
  private Half[] cachedHalfInputBuffer;
  private Half[] cachedOutputBuffer;

  /// <summary>
  /// CUDA graph capture requires 256-byte aligned memory addresses.
  /// This rounds up byte sizes to ensure proper alignment.
  /// </summary>
  private const int CUDA_GRAPH_ALIGNMENT = 256;

  /// <summary>
  /// Rounds up a byte size to the next 256-byte boundary for CUDA graph compatibility.
  /// </summary>
  private static long AlignForCudaGraph(long bytes) => (bytes + CUDA_GRAPH_ALIGNMENT - 1) & ~(CUDA_GRAPH_ALIGNMENT - 1);

  // (Per-stream buffers grouped in streamBuffers[])

  // Per-engine cached buffers for Exact mode (indexed by engine index)
  // Eliminates per-batch allocations when using multiple exact-size engines
  private byte[][] perEngineByteInputBuffers;
  private Half[][] perEngineHalfInputBuffers;
  private Half[][] perEngineOutputBuffers;

  // Cached batch plan to avoid allocations in pipelined processing
  // Each entry: (start position, actual positions, engine, engine batch size, use dynamic inference, engine index)
  private readonly List<(int start, int count, TensorRTEngine engine, int engineBatchSize, bool useDynamic, int engineIndex)> cachedBatchPlan = new();
  private int lastBatchPlanPositions = -1;
  private bool verboseDumpPending;

  /// <summary>
  /// Signals that a new multi-GPU batch is starting, so the next ComputeBatchPlan
  /// should produce verbose output even if the position count is unchanged.
  /// </summary>
  internal void MarkNewBatch() => verboseDumpPending = true;

  // Session cache: batch plan per totalPositions (engine sizes/timings are constant)
  private readonly Dictionary<int, (int start, int count, TensorRTEngine engine, int engineBatchSize, bool useDynamic, int engineIndex)[]> batchPlanCache = new();

  /// <summary>
  /// In Exact mode, the maximum fraction of an engine's batch size that can be padding
  /// while still preferring a single oversized engine over multiple smaller engine launches.
  /// For example, 0.25 means if we can satisfy a request with one engine where padding
  /// does not exceed 25% of the engine's batch size, we use that single engine.
  /// This avoids the latency cost of multiple kernel launches (which varies by platform).
  /// </summary>
  public float ExactModeBatchFractionAcceptablePadding { get; set; }

  /// <summary>
  /// In Exact mode, the maximum absolute number of padding positions that is acceptable
  /// when using an oversized engine. This provides a floor for small batch sizes where
  /// the fraction-based threshold would be too restrictive.
  /// For example, 8 means if padding is at most 8 positions, use the oversized engine.
  /// Either this OR ExactModeBatchFractionAcceptablePadding being satisfied allows oversized use.
  /// </summary>
  public int ExactModeBatchNumPositionsAcceptablePadding { get; set; }

  /// <summary>
  /// Execution times in milliseconds for each engine batch size.
  /// Index corresponds to the engine index (same order as ranges).
  /// Set by MultiGPUEnginePool.Warmup() or manually for optimized scheduling.
  /// </summary>
  public float[] ExecutionTimes { get; set; }

  /// <summary>
  /// Engine sizes array (batch sizes) for use by the optimized scheduler.
  /// In Exact mode, these are the exact batch sizes sorted descending.
  /// </summary>
  public int[] EngineSizes => ranges.Select(r => r.min).ToArray();


  public int InputElementsPerPosition { get; private set; }
  public int OutputElementsPerPosition { get; private set; }
  public long MaxTotalOutputSize { get; private set; }
  public int DeviceId => deviceId;
  public bool UseByteInputs => useByteInputs;

  /// <summary>
  /// Returns true if WarmupWithGraphCapture() has been called.
  /// When true, all CUDA graphs are captured and direct replay calls can be used.
  /// </summary>
  public bool IsWarmedUp => warmupCompleted;

  /// <summary>
  /// Reader/writer lock for CUDA graph capture synchronization.
  /// Graph capture requires exclusive access - no other CUDA activity can happen concurrently.
  /// Uses the same lock as CUDADevice to synchronize across all CUDA backends.
  /// Only used during WarmupWithGraphCapture() - not needed after warmup.
  /// </summary>
  private static ReaderWriterLockSlim GraphCaptureRWLock => CUDADevice.GetContext(0).GraphCaptureRWLock;


  /// <summary>
  /// Gets output tensor info from the largest engine.
  /// For Exact mode, engines are sorted descending so [0] is largest.
  /// For Range mode, engines are sorted ascending so [^1] is largest.
  /// </summary>
  public OutputTensorInfo[] GetOutputTensorInfo() =>
      mode == EnginePoolMode.Range ? engines[^1].GetOutputTensorInfo() : engines[0].GetOutputTensorInfo();

  /// <summary>
  /// Gets input tensor name from the first engine.
  /// </summary>
  public string GetInputName(int index) => engines[0].GetInputName(index);

  /// <summary>
  /// Gets the largest engine batch size in the pool.
  /// For Exact mode, engines are sorted descending so _ranges[0].max is largest.
  /// For Range mode, engines are sorted ascending so _ranges[^1].max is largest.
  /// </summary>
  public int MaxEngineBatchSize => mode == EnginePoolMode.Range ? ranges[^1].max : ranges[0].max;

  /// <summary>
  /// Output tensor alignment (must match OUTPUT_TENSOR_ALIGN_ELEMS in TensorRTWrapper.cpp).
  /// </summary>
  private const int OUTPUT_TENSOR_ALIGN = 128;

  /// <summary>
  /// Computes the total aligned output element count for a given batch size.
  /// Each tensor's size is AlignUp(perPos * batchSize, 128) to match the C++ layout.
  /// </summary>
  public int ComputeAlignedOutputSize(int batchSize)
  {
    int total = 0;
    for (int i = 0; i < outputPerPosSizes.Length; i++)
    {
      int tensorSize = outputPerPosSizes[i] * batchSize;
      total += (tensorSize + OUTPUT_TENSOR_ALIGN - 1) / OUTPUT_TENSOR_ALIGN * OUTPUT_TENSOR_ALIGN;
    }
    return total;
  }


  public EnginePool(TensorRT trt, string onnxPath, int[] sizes, EnginePoolMode mode,
                    TensorRTBuildOptions options, int inputElementsPerPos, int outputElementsPerPos,
                    int deviceId = 0, string cacheDir = "/tmp/tensorrt_cache")
  {
    this.onnxPath = onnxPath;
    this.mode = mode;
    this.deviceId = deviceId;
    // Store initial values - will be updated from first engine if provided values are estimates
    InputElementsPerPosition = inputElementsPerPos;
    OutputElementsPerPosition = outputElementsPerPos;

    this.trt = trt;
    ownsTrt = false;

    // Ensure cache directory exists
    if (!string.IsNullOrEmpty(cacheDir))
    {
      System.IO.Directory.CreateDirectory(cacheDir);
    }

    if (mode == EnginePoolMode.Range)
    {
      // sizes defines range boundaries: [1..sizes[0]], [sizes[0]+1..sizes[1]], etc.
      // e.g., sizes = {15, 127, 1024} means [1..15], [16..127], [128..1024]
      int prevMax = 0;
      foreach (int maxSize in sizes)
      {
        int minSize = prevMax + 1;
        ranges.Add((minSize, maxSize));

        TensorRTBuildOptions opts = options;
        opts.MinBatchSize = minSize;
        opts.OptBatchSize = FindOptimalBatchSize(minSize, maxSize);
        opts.MaxBatchSize = maxSize;

        TensorRTEngine engine = this.trt.LoadEngineWithCache(onnxPath, maxSize, opts, cacheDir, deviceId);
        engines.Add(engine);
        prevMax = maxSize;
      }
    }
    else // Exact mode
    {
      // sizes defines exact batch sizes, sorted descending for processing
      Array.Sort(sizes);
      Array.Reverse(sizes);

      // Build a single multi-profile engine with shared weights across all batch sizes.
      // This eliminates N-fold weight duplication in VRAM and requires only one cache file.
      string ext = System.IO.Path.GetExtension(onnxPath).ToLowerInvariant();
      TensorRTEngine[] multiEngines;
      if (ext == ".engine" || ext == ".plan")
      {
        // Load pre-built engine file directly (bypasses ONNX parsing and cache)
        multiEngines = this.trt.LoadMultiProfileEngineFile(onnxPath, sizes, options, deviceId);
      }
      else
      {
        multiEngines = this.trt.LoadMultiProfileEngineWithCache(
          onnxPath, sizes, options, cacheDir, deviceId);
      }

      foreach (TensorRTEngine engine in multiEngines)
      {
        ranges.Add((engine.BatchSize, engine.BatchSize));
        engines.Add(engine);
      }
    }


    // Compute per-position sizes from largest engine
    // For Exact mode, engines are sorted descending so [0] is largest
    // For Range mode, engines are sorted ascending so [^1] is largest
    TensorRTEngine largestEngine = this.mode == EnginePoolMode.Range ? engines[^1] : engines[0];
    int largestBatchSize = this.mode == EnginePoolMode.Range ? ranges[^1].max : ranges[0].max;
    InputElementsPerPosition = (int)(largestEngine.TotalInputSize / largestBatchSize);
    OutputElementsPerPosition = (int)(largestEngine.TotalOutputSize / largestBatchSize);
    MaxTotalOutputSize = largestEngine.TotalOutputSize;

    // Compute per-position output sizes for aligned output calculations
    OutputTensorInfo[] tensorInfos = largestEngine.GetOutputTensorInfo();
    outputPerPosSizes = new int[tensorInfos.Length];
    for (int i = 0; i < tensorInfos.Length; i++)
    {
      outputPerPosSizes[i] = (int)(tensorInfos[i].Size / largestBatchSize);
    }

    // Detect input mode from largest engine's tensor data type (INT8 = byte inputs)
    useByteInputs = largestEngine.HasByteInput;


    // Allocate buffers sized for largest engine
    // Apply 256-byte alignment for CUDA graph capture compatibility
    maxInputBytes = 0;
    maxOutputBytes = 0;
    foreach (TensorRTEngine e in engines)
    {
      maxInputBytes = Math.Max(maxInputBytes, e.TotalInputSize * sizeof(float));
      maxOutputBytes = Math.Max(maxOutputBytes, e.TotalOutputSize * sizeof(float));
    }

    // Round up to 256-byte alignment for CUDA graph capture (required on some platforms)
    long alignedInputBytes = AlignForCudaGraph(maxInputBytes);
    long alignedOutputBytes = AlignForCudaGraph(maxOutputBytes);

    // Allocate NUM_COMPUTE_STREAMS sets of pinned/GPU buffers for pipelined processing
    long maxInputElements = largestEngine.TotalInputSize;
    long maxOutputElements = largestEngine.TotalOutputSize;

    streamBuffers = new StreamBufferSet[NUM_COMPUTE_STREAMS];
    for (int i = 0; i < NUM_COMPUTE_STREAMS; i++)
    {
      streamBuffers[i] = new StreamBufferSet(alignedInputBytes, alignedOutputBytes, maxOutputElements);
    }

    // Pre-allocate managed input buffers for non-pipelined paths (Process/ProcessBytes)
    if (useByteInputs)
    {
      cachedByteInputBuffer = new byte[maxInputElements];
    }
    else
    {
      cachedHalfInputBuffer = new Half[maxInputElements];
    }
    cachedOutputBuffer = new Half[maxOutputElements];

    // Pre-allocate per-engine buffers for Exact mode to eliminate GC pressure
    if (mode == EnginePoolMode.Exact)
    {
      perEngineByteInputBuffers = new byte[engines.Count][];
      perEngineHalfInputBuffers = new Half[engines.Count][];
      perEngineOutputBuffers = new Half[engines.Count][];

      for (int i = 0; i < engines.Count; i++)
      {
        if (useByteInputs)
        {
          perEngineByteInputBuffers[i] = new byte[engines[i].TotalInputSize];
        }
        else
        {
          perEngineHalfInputBuffers[i] = new Half[engines[i].TotalInputSize];
        }
        perEngineOutputBuffers[i] = new Half[engines[i].TotalOutputSize];
      }
    }

    // Synchronize device after loading all engines
    TensorRTNative.SynchronizeDevice(this.deviceId);

    // Detect if GPU has integrated memory (like GB10/Jetson) and adjust padding thresholds.
    // Integrated GPUs have unified memory with lower kernel launch overhead relative to compute,
    // so we should be more conservative about accepting padding (prefer exact-fit batches).
    bool isIntegrated = TensorRTNative.IsIntegratedGPU(this.deviceId) == 1;
    ExactModeBatchNumPositionsAcceptablePadding = isIntegrated ? 10 : 30; // was:10:20
    ExactModeBatchFractionAcceptablePadding = isIntegrated ? 0.10f : 0.25f; // was: 0.1:0.2

    if (options.UseCudaGraphs != 0)
    {
      WarmupWithGraphCapture();
    }
  }


  /// <summary>
  /// Warms up all engines by running inference and capturing CUDA graphs if enabled.
  /// This method uses only stream 0 (no pipelining) to safely capture graphs.
  /// Must be called before ProcessBytesWithHandler for optimal performance with CUDA graphs.
  /// Thread-safe: uses global graph capture lock to prevent concurrent capture across devices.
  /// After this method completes, IsWarmedUp returns true and pipelining uses direct graph replay.
  /// </summary>
  public unsafe void WarmupWithGraphCapture()
  {
    // Already warmed up - nothing to do
    if (warmupCompleted)
    {
      return;
    }

    // Warm up each engine to trigger CUDA graph capture on all compute streams.
    // Uses single-stream-at-a-time to avoid concurrent CUDA activity during capture.
    foreach (TensorRTEngine engine in engines)
    {
      if (!engine.UsesCudaGraphs || engine.IsStreamGraphCaptured(0))
      {
        continue;
      }

      int inputBytes = (int)engine.TotalInputSize * (useByteInputs ? 1 : 2);
      int outputBytes = (int)engine.TotalOutputSize * 2;

      // Capture CUDA graph on each compute stream
      for (int s = 0; s < NUM_COMPUTE_STREAMS; s++)
      {
        int streamId = COMPUTE_STREAM_IDS[s];
        if (s > 0 && engine.IsStreamGraphCaptured(streamId))
        {
          continue;
        }

        engine.CopyToGPUOnStreamAsync(streamId, streamBuffers[s].GpuInput, streamBuffers[s].PinnedInput, inputBytes);
        engine.InferOnStreamWithGraphAsync(streamId, streamBuffers[s].GpuInput, streamBuffers[s].GpuOutput, GraphCaptureRWLock);
        engine.CopyFromGPUOnStreamAsync(streamId, streamBuffers[s].PinnedOutput, streamBuffers[s].GpuOutput, outputBytes);
        engine.SyncStream(streamId);
      }
    }

    warmupCompleted = true;
  }


  public void Process(Half[] input, Half[] output, int totalPositions,
                      int inputElementOffset = 0, int outputElementOffset = 0)
  {
    int processed = 0;
    int lastBatchSize = 0;

    while (processed < totalPositions)
    {
      int remaining = totalPositions - processed;
      (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) = SelectEngineWithMode(remaining, lastBatchSize);

      int actualPositions = Math.Min(batchSize, remaining);
      int inputOffset = inputElementOffset + processed * InputElementsPerPosition;
      int outputOffset = outputElementOffset + processed * OutputElementsPerPosition;
      int inputElements = actualPositions * InputElementsPerPosition;
      int outputElements = useDynamic ? ComputeAlignedOutputSize(actualPositions) : actualPositions * OutputElementsPerPosition;

      if (useDynamic)
      {
        // Dynamic inference: reuse pre-allocated max-size buffers
        // Copy input to cached buffer
        if (inputElements > 0)
        {
          Array.Copy(input, inputOffset, cachedHalfInputBuffer, 0, inputElements);
        }

        // Dynamic inference with actual batch size using oversized buffers
        engine.InferHostDynamic(cachedHalfInputBuffer, cachedOutputBuffer, actualPositions, inputElements, outputElements);

        // Copy output from cached buffer   copy only the unaligned per-position data
        int unalignedElements = actualPositions * OutputElementsPerPosition;
        if (unalignedElements > 0)
        {
          Array.Copy(cachedOutputBuffer, 0, output, outputOffset, unalignedElements);
        }
      }
      else
      {
        // Static inference: use pre-allocated per-engine buffers
        Half[] batchInput = perEngineHalfInputBuffers[engineIndex];
        Half[] batchOutput = perEngineOutputBuffers[engineIndex];

        // Copy input to buffer
        if (inputElements > 0)
        {
          Array.Copy(input, inputOffset, batchInput, 0, inputElements);
        }

        // Synchronous inference
        engine.InferHost(batchInput, batchOutput);

        // Copy output
        if (outputElements > 0)
        {
          Array.Copy(batchOutput, 0, output, outputOffset, outputElements);
        }
      }

      processed += actualPositions;
      lastBatchSize = batchSize;
    }
  }

  public void ProcessBytes(byte[] input, Half[] output, int totalPositions,
                           int inputByteOffset = 0, int outputElementOffset = 0)
  {
    int processed = 0;
    int lastBatchSize = 0;

    while (processed < totalPositions)
    {
      int remaining = totalPositions - processed;
      (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) = SelectEngineWithMode(remaining, lastBatchSize);

      int actualPositions = Math.Min(batchSize, remaining);
      int inputOffset = inputByteOffset + processed * InputElementsPerPosition;
      int outputOffset = outputElementOffset + processed * OutputElementsPerPosition;
      int inputElements = actualPositions * InputElementsPerPosition;
      int outputElements = useDynamic ? ComputeAlignedOutputSize(actualPositions) : actualPositions * OutputElementsPerPosition;

      if (useDynamic)
      {
        // Dynamic inference: reuse pre-allocated max-size buffers
        // Copy input to cached buffer
        if (inputElements > 0)
        {
          Array.Copy(input, inputOffset, cachedByteInputBuffer, 0, inputElements);
        }

        // Dynamic inference with actual batch size using oversized buffers
        engine.InferHostBytesDynamic(cachedByteInputBuffer, cachedOutputBuffer, actualPositions, inputElements, outputElements);

        // Copy output from cached buffer   copy only the unaligned per-position data
        int unalignedElements = actualPositions * OutputElementsPerPosition;
        if (unalignedElements > 0)
        {
          Array.Copy(cachedOutputBuffer, 0, output, outputOffset, unalignedElements);
        }
      }
      else
      {
        // Static inference: use pre-allocated per-engine buffers
        byte[] batchInput = perEngineByteInputBuffers[engineIndex];
        Half[] batchOutput = perEngineOutputBuffers[engineIndex];

        // Copy input to buffer
        if (inputElements > 0)
        {
          Array.Copy(input, inputOffset, batchInput, 0, inputElements);
        }

        // Synchronous inference with byte inputs
        engine.InferHostBytes(batchInput, batchOutput);

        // Copy output
        if (outputElements > 0)
        {
          Array.Copy(batchOutput, 0, output, outputOffset, outputElements);
        }
      }

      processed += actualPositions;
      lastBatchSize = batchSize;
    }
  }

  /// <summary>
  /// Process Half inputs with callback for tensor-major output extraction.
  /// Uses the unified pipelined path with async stream-based inference.
  /// </summary>
  public void ProcessWithHandler(Half[] input, int totalPositions, SubBatchOutputHandler handler,
                                 int globalPositionOffset = 0, int inputElementOffset = 0)
  {
    ComputeBatchPlan(totalPositions);
    if (cachedBatchPlan.Count == 0)
    {
      return;
    }
    ProcessSubBatchesPipelined(input, handler, globalPositionOffset, inputElementOffset);
  }






  /// <summary>
  /// Process byte inputs with callback for tensor-major output extraction.
  /// Uses the unified pipelined path with async stream-based inference.
  /// </summary>
  public void ProcessBytesWithHandler(byte[] input, int totalPositions, SubBatchOutputHandler handler,
                                      int globalPositionOffset = 0, int inputByteOffset = 0)
  {
    ComputeBatchPlan(totalPositions);
    if (cachedBatchPlan.Count == 0)
    {
      return;
    }
    ProcessSubBatchesPipelined(input, handler, globalPositionOffset, inputByteOffset);
  }


  /// <summary>
  /// Unified pipelined sub-batch processing for both byte and Half inputs.
  /// Processes sub-batches in groups of NUM_COMPUTE_STREAMS with per-stream pipelining:
  ///
  /// Each stream issues the complete H2D -> Compute -> D2H sequence asynchronously,
  /// enabling true concurrent compute across streams (separate TRT execution contexts)
  /// and overlapping D2H with compute:
  ///
  ///   Stream A: [H2D(0)] [Compute(0)] [D2H(0)]
  ///   Stream B: [H2D(1)] [Compute(1)] [D2H(1)]
  ///                       ^concurrent^  ^overlap^
  ///
  /// For dynamic inference (no CUDA graphs), compute is serialized across streams
  /// since both streams share a single TRT execution context.
  /// </summary>
  private unsafe void ProcessSubBatchesPipelined<TInput>(TInput[] input, SubBatchOutputHandler handler,
                                                          int globalPositionOffset, int inputElementOffset)
      where TInput : unmanaged
  {
    int batchCount = cachedBatchPlan.Count;

    // Single batch: direct H2D -> compute -> D2H -> handler (no pipelining overhead)
    if (batchCount == 1)
    {
      (int start, int count, TensorRTEngine engine, int engineBatchSize, bool useDynamic, int engineIndex) = cachedBatchPlan[0];
      int inputBytes = count * InputElementsPerPosition * sizeof(TInput);
      int elementOffset = inputElementOffset + start * InputElementsPerPosition;

      fixed (TInput* srcPtr = input)
      {
        Buffer.MemoryCopy(srcPtr + elementOffset, (void*)streamBuffers[0].PinnedInput, inputBytes, inputBytes);
      }

      engine.CopyToGPUOnStreamAsync(STREAM_COMPUTE_A, streamBuffers[0].GpuInput, streamBuffers[0].PinnedInput, inputBytes);

      if (useDynamic)
      {
        engine.InferOnStreamDynamicAsync(STREAM_COMPUTE_A, streamBuffers[0].GpuInput, streamBuffers[0].GpuOutput, count);
      }
      else
      {
        engine.InferOnStreamWithGraphAsync(STREAM_COMPUTE_A, streamBuffers[0].GpuInput, streamBuffers[0].GpuOutput);
      }

      int outputPositions = useDynamic ? count : engineBatchSize;
      long outputBytes = (long)ComputeAlignedOutputSize(outputPositions) * sizeof(ushort);
      engine.CopyFromGPUOnStreamAsync(STREAM_COMPUTE_A, streamBuffers[0].PinnedOutput, streamBuffers[0].GpuOutput, outputBytes);
      engine.SyncStream(STREAM_COMPUTE_A);

      fixed (Half* dstPtr = streamBuffers[0].ManagedOutput)
      {
        Buffer.MemoryCopy((void*)streamBuffers[0].PinnedOutput, dstPtr, outputBytes, outputBytes);
      }
      handler(globalPositionOffset + start, count, outputPositions, streamBuffers[0].ManagedOutput);
      return;
    }

    // Multi-batch: process in groups of NUM_COMPUTE_STREAMS with per-stream pipelining.
    // Each stream issues H2D -> Compute -> D2H asynchronously, enabling concurrent
    // compute and overlapping D2H(s) with Compute(s+1) across streams.
    Span<long> groupOutputBytes = stackalloc long[NUM_COMPUTE_STREAMS];
    Span<int> groupOutputPositions = stackalloc int[NUM_COMPUTE_STREAMS];
    for (int groupStart = 0; groupStart < batchCount; groupStart += NUM_COMPUTE_STREAMS)
    {
      int groupEnd = Math.Min(groupStart + NUM_COMPUTE_STREAMS, batchCount);
      int groupSize = groupEnd - groupStart;

      // Check if any batch in this group uses dynamic inference.
      // Dynamic inference shares a single TRT execution context across streams,
      // so concurrent enqueueV3 is unsafe and compute must be serialized.
      bool anyDynamic = false;
      for (int s = 0; s < groupSize; s++)
      {
        if (cachedBatchPlan[groupStart + s].useDynamic)
        {
          anyDynamic = true;
          break;
        }
      }

      // Issue complete H2D -> Compute -> D2H per stream (all async on GPU)
      for (int s = 0; s < groupSize; s++)
      {
        int batchIdx = groupStart + s;
        int streamId = COMPUTE_STREAM_IDS[s];
        (int start, int count, TensorRTEngine engine, int engineBatchSize, bool useDynamic, int engineIndex) = cachedBatchPlan[batchIdx];

        int inputBytes = count * InputElementsPerPosition * sizeof(TInput);
        int elementOffset = inputElementOffset + start * InputElementsPerPosition;

        // Copy input to pinned memory (CPU-side, overlaps with previous stream's GPU work)
        fixed (TInput* srcPtr = input)
        {
          Buffer.MemoryCopy(srcPtr + elementOffset, (void*)streamBuffers[s].PinnedInput, inputBytes, inputBytes);
        }

        // H2D on this compute stream
        engine.CopyToGPUOnStreamAsync(streamId, streamBuffers[s].GpuInput, streamBuffers[s].PinnedInput, inputBytes);

        // Dynamic path safety: serialize compute when streams share execution context
        if (anyDynamic && s > 0)
        {
          cachedBatchPlan[groupStart + s - 1].engine.SyncStream(COMPUTE_STREAM_IDS[s - 1]);
        }

        // Launch compute
        if (useDynamic)
        {
          engine.InferOnStreamDynamicAsync(streamId, streamBuffers[s].GpuInput, streamBuffers[s].GpuOutput, count);
        }
        else
        {
          engine.InferOnStreamWithGraphAsync(streamId, streamBuffers[s].GpuInput, streamBuffers[s].GpuOutput);
        }

        // D2H on same stream (queued after compute completes on this stream)
        int outputPositions = useDynamic ? count : engineBatchSize;
        long outputBytes = (long)ComputeAlignedOutputSize(outputPositions) * sizeof(ushort);
        groupOutputBytes[s] = outputBytes;
        groupOutputPositions[s] = outputPositions;
        engine.CopyFromGPUOnStreamAsync(streamId, streamBuffers[s].PinnedOutput, streamBuffers[s].GpuOutput, outputBytes);
      }

      // Sync each stream and process results in order
      for (int s = 0; s < groupSize; s++)
      {
        int batchIdx = groupStart + s;
        int streamId = COMPUTE_STREAM_IDS[s];
        (int start, int count, TensorRTEngine engine, _, _, _) = cachedBatchPlan[batchIdx];

        engine.SyncStream(streamId);

        long outputBytes = groupOutputBytes[s];
        fixed (Half* dstPtr = streamBuffers[s].ManagedOutput)
        {
          Buffer.MemoryCopy((void*)streamBuffers[s].PinnedOutput, dstPtr, outputBytes, outputBytes);
        }
        handler(globalPositionOffset + start, count, groupOutputPositions[s], streamBuffers[s].ManagedOutput);
      }
    }
  }


  /// <summary>
  /// Computes the batch plan for processing totalPositions using the engine selection logic.
  /// Results are stored in cachedBatchPlan to avoid allocations.
  /// Each entry contains: (start position, actual positions, engine, engine batch size, use dynamic inference, engine index)
  /// </summary>
  private void ComputeBatchPlan(int totalPositions)
  {
    // Fast path: identical to last call, cachedBatchPlan already populated
    if (totalPositions == lastBatchPlanPositions)
    {
      if (verboseDumpPending)
      {
        verboseDumpPending = false;
        DumpBatchPlanVerbose(totalPositions);
      }
      return;
    }

    cachedBatchPlan.Clear();
    lastBatchPlanPositions = totalPositions;
    verboseDumpPending = false;

    // Session cache: engine sizes/timings are constant so the plan is invariant per totalPositions
    if (batchPlanCache.TryGetValue(totalPositions, out var cached))
    {
      foreach (var entry in cached)
      {
        cachedBatchPlan.Add(entry);
      }
      DumpBatchPlanVerbose(totalPositions);
      return;
    }

    // Use optimized scheduling if enabled and execution times are available (Exact mode only)
    if (OPTIMIZED_SCHEDULING && ExecutionTimes != null && mode == EnginePoolMode.Exact)
    {
      ComputeBatchPlanOptimized(totalPositions);
    }
    else
    {
      // Legacy greedy algorithm
      int processed = 0;
      int lastBatchSize = 0;

      while (processed < totalPositions)
      {
        int remaining = totalPositions - processed;
        (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) = SelectEngineWithMode(remaining, lastBatchSize);

        // For Exact mode, batchSize is the engine's configured size (may exceed remaining).
        // We process actualPositions but allocate/compute with engine's batchSize.
        int actualPositions = Math.Min(batchSize, remaining);
        cachedBatchPlan.Add((processed, actualPositions, engine, batchSize, useDynamic, engineIndex));

        processed += actualPositions;
        lastBatchSize = batchSize;
      }
    }

    batchPlanCache[totalPositions] = [.. cachedBatchPlan];
  }


  /// <summary>
  /// Computes batch plan using the optimized BatchScheduler algorithm.
  /// This considers actual execution times per batch size to minimize total inference time.
  /// </summary>
  private void ComputeBatchPlanOptimized(int totalPositions)
  {
    int numEngines = ranges.Count;

    // Build engine sizes array - ranges[i].min contains the size in Exact mode
    // Note: ranges is sorted descending, but ExecutionTimes is indexed by original (ascending) order
    Span<int> engineSizes = stackalloc int[numEngines];
    for (int i = 0; i < numEngines; i++)
    {
      engineSizes[i] = ranges[i].min;
    }

    // Build correctly-ordered execution times array to match engineSizes order
    // ExecutionTimes is indexed by original order (ascending), we need it in ranges order (descending)
    Span<float> orderedExecutionTimes = stackalloc float[numEngines];
    for (int i = 0; i < numEngines; i++)
    {
      int size = engineSizes[i];
      // Find this size's index in ascending order (original ExecutionTimes order)
      // Since ranges is descending and ExecutionTimes is ascending, index is (numEngines - 1 - position in descending)
      int originalIndex = 0;
      for (int j = 0; j < numEngines; j++)
      {
        if (ranges[j].min < size)
        {
          originalIndex++;
        }
      }
      orderedExecutionTimes[i] = ExecutionTimes[originalIndex];
    }

    // Use the optimized single-GPU scheduler (accepts ReadOnlySpan, no allocation needed)
    // When concurrent compute is enabled, use the concurrent-aware scheduler
    int[] batchSizes = BatchScheduler.ScheduleSingleGPU(engineSizes, orderedExecutionTimes, totalPositions,
                                                         deviceId, concurrent: NUM_COMPUTE_STREAMS > 1);

    // Convert batch sizes to batch plan entries
    int processed = 0;
    foreach (int batchSize in batchSizes)
    {
      // Find engine index by linear scan (numEngines is small, typically < 8)
      int engineIndex = -1;
      for (int i = 0; i < numEngines; i++)
      {
        if (ranges[i].min == batchSize)
        {
          engineIndex = i;
          break;
        }
      }

      TensorRTEngine engine = engines[engineIndex];

      // In Exact mode, we process up to batchSize positions but use the full engine batch
      int actualPositions = Math.Min(batchSize, totalPositions - processed);

      // Exact mode always uses static inference (useDynamic = false)
      cachedBatchPlan.Add((processed, actualPositions, engine, batchSize, false, engineIndex));

      processed += actualPositions;
      if (processed >= totalPositions)
      {
        break;
      }
    }

    if (VERBOSE_DUMP_BATCHES)
    {
      int totalPos = batchSizes.Sum();
      float totalTime = 0;
      foreach (int b in batchSizes)
      {
        for (int i = 0; i < numEngines; i++)
        {
          if (engineSizes[i] == b)
          {
            totalTime += orderedExecutionTimes[i];
            break;
          }
        }
      }
      Console.WriteLine($"OPTIMIZED_PLAN [device {deviceId}]: {totalPositions} -> [{string.Join(", ", batchSizes)}] " +
                        $"total={totalPos} padding={totalPos - totalPositions} time={totalTime:F1}ms");
    }
  }


  /// <summary>
  /// Dumps the current cached batch plan in OPTIMIZED_PLAN format.
  /// Called on cache hits so verbose output is always shown.
  /// </summary>
  private void DumpBatchPlanVerbose(int totalPositions)
  {
    if (!BatchScheduler.VERBOSE_DETAILS || cachedBatchPlan.Count == 0 || ExecutionTimes == null)
    {
      return;
    }

    int numEngines = ranges.Count;
    int totalBatched = 0;
    float totalTime = 0;
    Span<int> planSizes = stackalloc int[cachedBatchPlan.Count];
    for (int i = 0; i < cachedBatchPlan.Count; i++)
    {
      int batchSize = cachedBatchPlan[i].engineBatchSize;
      planSizes[i] = batchSize;
      totalBatched += batchSize;

      // Map engine batch size to ExecutionTimes index (ascending order)
      int originalIndex = 0;
      for (int j = 0; j < numEngines; j++)
      {
        if (ranges[j].min < batchSize)
        {
          originalIndex++;
        }
      }
      totalTime += ExecutionTimes[originalIndex];
    }

    // Add overhead (concurrent or sequential) matching the execution strategy
    if (cachedBatchPlan.Count > 1)
    {
      if (NUM_COMPUTE_STREAMS > 1)
      {
        int numGroups = (cachedBatchPlan.Count + NUM_COMPUTE_STREAMS - 1) / NUM_COMPUTE_STREAMS;
        totalTime += numGroups * BatchScheduler.PER_CONCURRENT_PAIR_OVERHEAD_MS;
      }
      else
      {
        totalTime += (cachedBatchPlan.Count - 1) * BatchScheduler.PER_BATCH_OVERHEAD_MS;
      }
    }

    int[] sizes = planSizes.ToArray();
    Console.WriteLine($"  OPTIMIZED_PLAN [device {deviceId}]: {totalPositions} -> [{string.Join(", ", sizes)}] " +
                      $"total={totalBatched} padding={totalBatched - totalPositions} time={totalTime:F1}ms");
  }


  /// <summary>
  /// Selects an engine for the given batch size and indicates whether to use dynamic inference.
  /// For Range mode, uses dynamic inference with actual batch size.
  /// For Exact mode, uses static inference with engine's configured batch size.
  /// </summary>
  private (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) SelectEngineWithMode(int remaining, int lastBatchSize)
  {
    (TensorRTEngine engine, int engineBatchSize, int engineIndex) = DoSelectEngineWithIndex(remaining, lastBatchSize);

    // For Range mode, use dynamic inference with actual batch size
    // For Exact mode, use static inference (engine expects its exact configured batch size)
    bool useDynamic = (mode == EnginePoolMode.Range);

    // For dynamic mode, the engineBatchSize returned is the engine's max, but we'll use 'remaining' 
    // as the actual batch size (up to the engine's max)
    int actualBatch = useDynamic ? Math.Min(remaining, engineBatchSize) : engineBatchSize;
    int padded = engineBatchSize - actualBatch;
    int remainingAfter = remaining - actualBatch;

    if (VERBOSE_DUMP_BATCHES)
    {
      DumpBatchInfo(actualBatch, engineBatchSize, engineIndex, padded, remainingAfter);
    }

    return (engine, actualBatch, useDynamic, engineIndex);
  }


  /// <summary>
  /// Dumps batch execution info to console with colored output.
  /// </summary>
  private void DumpBatchInfo(int actualBatch, int engineBatchSize, int engineIndex, int padded, int remaining)
  {
    string engineDesc;
    string paddedStr;

    if (mode == EnginePoolMode.Range)
    {
      (int min, int max) = ranges[engineIndex];
      engineDesc = $"Range({min}/{FindOptimalBatchSize(min, max)}/{max})";
      paddedStr = ""; // No padding in Range mode - we use dynamic batch size
    }
    else
    {
      engineDesc = $"Exact({engineBatchSize})";
      paddedStr = $" padded {padded}";
    }

    ConsoleColor color = padded > 0 ? ConsoleColor.Blue : ConsoleColor.Yellow;
    ConsoleColor prev = Console.ForegroundColor;
    Console.ForegroundColor = color;
    Console.WriteLine($"TRT_BATCH {actualBatch} allocated {engineBatchSize} from {engineDesc}{paddedStr} remaining {remaining}");
    Console.ForegroundColor = prev;
  }

  private (TensorRTEngine engine, int batchSize, int engineIndex) DoSelectEngineWithIndex(int remaining, int lastBatchSize)
  {
    if (mode == EnginePoolMode.Range)
    {
      // Find the range that contains 'remaining'
      for (int i = 0; i < ranges.Count; i++)
      {
        if (remaining >= ranges[i].min && remaining <= ranges[i].max)
        {
          return (engines[i], ranges[i].max, i);
        }
      }
      // If larger than all ranges, use largest engine
      int lastIndex = ranges.Count - 1;
      return (engines[^1], ranges[^1].max, lastIndex);
    }
    else // Exact mode
    {
      // Engines are sorted descending by size in Exact mode.
      // Acceptable padding is satisfied if EITHER:
      //   1. Fraction-based: (engineSize - remaining) <= engineSize * ExactModeBatchFractionAcceptablePadding
      //      Which simplifies to: remaining >= engineSize * (1 - ExactModeBatchFractionAcceptablePadding)
      //   2. Absolute: (engineSize - remaining) <= ExactModeBatchNumPositionsAcceptablePadding
      float minFillFraction = 1.0f - ExactModeBatchFractionAcceptablePadding;

      for (int i = 0; i < ranges.Count; i++)
      {
        // If the next smaller is already big enough then don't consider this one.
        if (i < ranges.Count - 1)
        {
          int nextSmallerEngineSize = ranges[i + 1].min;
          if (remaining <= nextSmallerEngineSize)
          {
            continue;
          }
        }

        int engineSize = ranges[i].min;

        // Check if padding is acceptable for this oversized engine
        int paddingPositions = engineSize - remaining;
        bool fractionAcceptable = remaining >= engineSize * minFillFraction;
        bool absoluteAcceptable = paddingPositions <= ExactModeBatchNumPositionsAcceptablePadding;

        if (fractionAcceptable || absoluteAcceptable)
        {
          // Padding is within acceptable limits, use this single engine
          return (engines[i], engineSize, i);
        }
      }

      // No engine with acceptable padding found; use smallest engine with padding
      int lastIndex = ranges.Count - 1;
      return (engines[^1], ranges[^1].min, lastIndex);
    }
  }



  /// <summary>
  /// Finds optimal batch size: prefer power of 2 closest to midpoint, else use midpoint.
  /// </summary>
  /// <param name="minSize"></param>
  /// <param name="maxSize"></param>
  /// <returns></returns>
  private static int FindOptimalBatchSize(int minSize, int maxSize)
  {
    int midpoint = (minSize + maxSize) / 2;

    // Find all powers of 2 within the range [minSize, maxSize]
    int bestPow2 = -1;
    int bestDistance = int.MaxValue;

    for (int pow2 = 1; pow2 <= maxSize; pow2 *= 2)
    {
      if (pow2 >= minSize && pow2 <= maxSize)
      {
        int distance = Math.Abs(pow2 - midpoint);
        if (distance < bestDistance)
        {
          bestDistance = distance;
          bestPow2 = pow2;
        }
      }
    }

    return bestPow2 > 0 ? bestPow2 : midpoint;
  }


  public string GetRangesDescription()
  {
    List<string> parts = new();
    foreach ((int min, int max) in ranges)
    {
      if (mode == EnginePoolMode.Range)
      {
        parts.Add($"[{min}..{max}]");
      }
      else
      {
        parts.Add($"{min}");
      }
    }
    return string.Join(", ", parts);
  }


  public void Dispose()
  {
    if (disposed) return;
    disposed = true;

    // Free stream buffers
    for (int i = 0; i < NUM_COMPUTE_STREAMS; i++)
    {
      streamBuffers[i].Free();
    }

    foreach (TensorRTEngine engine in engines)
    {
      engine.Dispose();
    }

    if (ownsTrt)
    {
      trt.Dispose();
    }
  }
}
