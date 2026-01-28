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
  /// If true, uses pipelined sub-batch processing where H2D of the next batch
  /// overlaps with compute of the current batch using two CUDA streams.
  /// NOTE: Pipelining is automatically disabled when CUDA graph capture is pending,
  /// since graph capture requires exclusive GPU access (no concurrent stream activity).
  /// </summary>
  private const bool USE_PIPELINED_SUBBATCHES = true;

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

  // Pinned memory and GPU buffers for the largest engine
  private IntPtr pinnedIn;
  private IntPtr pinnedOut;
  private IntPtr gpuIn;
  private IntPtr gpuOut;
  private long maxInputBytes;
  private long maxOutputBytes;

  // Pipelined sub-batch resources: second set of pinned + GPU buffers
  // Used to overlap H2D of next batch with compute of current batch
  private IntPtr pinnedIn2;
  private IntPtr gpuIn2;
  private IntPtr pinnedOut2;
  private IntPtr gpuOut2;

  // Pre-allocated managed buffers to reduce GC pressure
  private byte[] cachedByteInputBuffer;
  private Half[] cachedHalfInputBuffer;
  private Half[] cachedOutputBuffer;

  // Second input buffer for pipelined processing
  private byte[] cachedByteInputBuffer2;

  // Second output buffer for pipelined processing
  private Half[] cachedOutputBuffer2;

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
      foreach (int size in sizes)
      {
        ranges.Add((size, size));

        TensorRTBuildOptions opts = options;

        // Use tiling optimization level 3 for batch sizes >= 128
        // TODO: currently disabled, this may only help on certain GPUs (e.g. GB10) and is very slow to build
        if (size >= 128)
        {
          // opts.TilingOptimizationLevel = 3; // note: can be 5x to 20x slower to build
        }

        TensorRTEngine engine = this.trt.LoadEngineWithCache(onnxPath, size, opts, cacheDir, deviceId);
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

    // Detect input mode from largest engine's tensor data type (INT8 = byte inputs)
    useByteInputs = largestEngine.HasByteInput;

    // Allocate buffers sized for largest engine
    maxInputBytes = 0;
    maxOutputBytes = 0;
    foreach (TensorRTEngine e in engines)
    {
      maxInputBytes = Math.Max(maxInputBytes, e.TotalInputSize * sizeof(float));
      maxOutputBytes = Math.Max(maxOutputBytes, e.TotalOutputSize * sizeof(float));
    }

    pinnedIn = TensorRTNative.AllocPinned(maxInputBytes);
    pinnedOut = TensorRTNative.AllocPinned(maxOutputBytes);
    gpuIn = TensorRTNative.AllocGPU(maxInputBytes);
    gpuOut = TensorRTNative.AllocGPU(maxOutputBytes);

    if (USE_PIPELINED_SUBBATCHES)
    {
      // Allocate second set of input and output buffers for pipelined sub-batch processing
      pinnedIn2 = TensorRTNative.AllocPinned(maxInputBytes);
      gpuIn2 = TensorRTNative.AllocGPU(maxInputBytes);
      pinnedOut2 = TensorRTNative.AllocPinned(maxOutputBytes);
      gpuOut2 = TensorRTNative.AllocGPU(maxOutputBytes);
    }

    // Pre-allocate managed buffers sized for largest engine
    long maxInputElements = largestEngine.TotalInputSize;
    long maxOutputElements = largestEngine.TotalOutputSize;
    if (useByteInputs)
    {
      cachedByteInputBuffer = new byte[maxInputElements];

      if (USE_PIPELINED_SUBBATCHES)
      {
        cachedByteInputBuffer2 = new byte[maxInputElements];
      }
    }
    else
    {
      cachedHalfInputBuffer = new Half[maxInputElements];
    }
    cachedOutputBuffer = new Half[maxOutputElements];

    if (USE_PIPELINED_SUBBATCHES)
    {
      cachedOutputBuffer2 = new Half[maxOutputElements];
    }

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

    // Warm up each engine to trigger CUDA graph capture on stream 0
    // Uses single-stream path to avoid concurrent CUDA activity during capture
    foreach (TensorRTEngine engine in engines)
    {
      if (!engine.UsesCudaGraphs || engine.IsStreamGraphCaptured(0))
      {
        continue;
      }

      // Allocate dummy buffers sized for this engine's batch size
      int inputBytes = (int)engine.TotalInputSize * (useByteInputs ? 1 : 2);
      int outputBytes = (int)engine.TotalOutputSize * 2;

      // Copy dummy input to pinned memory (zeros are fine for warmup)
      // Use H2D, Infer (with graph capture), D2H all on stream 0
      engine.CopyToGPUOnStreamAsync(0, gpuIn, pinnedIn, inputBytes);

      // This call will capture the CUDA graph on first execution
      // Pass the lock to ensure exclusive access during capture (no concurrent CUDA activity allowed)
      engine.InferOnStreamWithGraphAsync(0, gpuIn, gpuOut, GraphCaptureRWLock);

      engine.CopyFromGPUOnStreamAsync(0, pinnedOut, gpuOut, outputBytes);
      engine.SyncStream(0);
    }

    warmupCompleted = true;
  }


  public void Process(Half[] input, Half[] output, int totalPositions)
  {
    int processed = 0;
    int lastBatchSize = 0;

    while (processed < totalPositions)
    {
      int remaining = totalPositions - processed;
      (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) = SelectEngineWithMode(remaining, lastBatchSize);

      int actualPositions = Math.Min(batchSize, remaining);
      int inputOffset = processed * InputElementsPerPosition;
      int outputOffset = processed * OutputElementsPerPosition;
      int inputElements = actualPositions * InputElementsPerPosition;
      int outputElements = actualPositions * OutputElementsPerPosition;

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

        // Copy output from cached buffer
        if (outputElements > 0)
        {
          Array.Copy(cachedOutputBuffer, 0, output, outputOffset, outputElements);
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

  public void ProcessBytes(byte[] input, Half[] output, int totalPositions)
  {
    int processed = 0;
    int lastBatchSize = 0;

    while (processed < totalPositions)
    {
      int remaining = totalPositions - processed;
      (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) = SelectEngineWithMode(remaining, lastBatchSize);

      int actualPositions = Math.Min(batchSize, remaining);
      int inputOffset = processed * InputElementsPerPosition;
      int outputOffset = processed * OutputElementsPerPosition;
      int inputElements = actualPositions * InputElementsPerPosition;
      int outputElements = actualPositions * OutputElementsPerPosition;

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

        // Copy output from cached buffer
        if (outputElements > 0)
        {
          Array.Copy(cachedOutputBuffer, 0, output, outputOffset, outputElements);
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
  /// Process with callback for tensor-major output extraction.
  /// The handler is called after each sub-batch inference with the raw output buffer.
  /// Uses pre-allocated buffers when possible to reduce GC pressure.
  /// </summary>
  public void ProcessWithHandler(Half[] input, int totalPositions, SubBatchOutputHandler handler, int globalPositionOffset = 0)
  {
    int processed = 0;
    int lastBatchSize = 0;

    while (processed < totalPositions)
    {
      int remaining = totalPositions - processed;
      (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) = SelectEngineWithMode(remaining, lastBatchSize);

      int actualPositions = Math.Min(batchSize, remaining);
      int inputOffset = processed * InputElementsPerPosition;
      int inputElements = actualPositions * InputElementsPerPosition;
      int outputElements = actualPositions * OutputElementsPerPosition;

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

        // Call handler with raw output (tensor-major layout) and actual batch size
        handler(globalPositionOffset + processed, actualPositions, actualPositions, cachedOutputBuffer);
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

        // Call handler with raw output (tensor-major layout)
        handler(globalPositionOffset + processed, actualPositions, batchSize, batchOutput);
      }

      processed += actualPositions;
      lastBatchSize = batchSize;
    }
  }






  /// <summary>
  /// Process byte inputs with callback for tensor-major output extraction.
  /// The handler is called after each sub-batch inference with the raw output buffer.
  /// Uses pre-allocated buffers when possible to reduce GC pressure.
  /// </summary>
  public void ProcessBytesWithHandler(byte[] input, int totalPositions, SubBatchOutputHandler handler, int globalPositionOffset = 0)
  {
    if (USE_PIPELINED_SUBBATCHES)
    {
      ProcessBytesWithHandlerPipelined(input, totalPositions, handler, globalPositionOffset);
    }
    else
    {
      // Pipelining disabled - use non-pipelined path (still uses graph replay after warmup)
      ProcessBytesWithHandlerNonPipelined(input, totalPositions, handler, globalPositionOffset);
    }
  }


  /// <summary>
  /// Non-pipelined processing path that uses synchronous host-based inference.
  /// </summary>
  private void ProcessBytesWithHandlerNonPipelined(byte[] input, int totalPositions, SubBatchOutputHandler handler, int globalPositionOffset)
  {
    int processed = 0;
    int lastBatchSize = 0;

    while (processed < totalPositions)
    {
      int remaining = totalPositions - processed;
      (TensorRTEngine engine, int batchSize, bool useDynamic, int engineIndex) = SelectEngineWithMode(remaining, lastBatchSize);

      int actualPositions = Math.Min(batchSize, remaining);
      int inputOffset = processed * InputElementsPerPosition;
      int inputElements = actualPositions * InputElementsPerPosition;
      int outputElements = actualPositions * OutputElementsPerPosition;

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

        // Call handler with raw output (tensor-major layout) and actual batch size
        handler(globalPositionOffset + processed, actualPositions, actualPositions, cachedOutputBuffer);
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

        // Call handler with raw output (tensor-major layout)
        handler(globalPositionOffset + processed, actualPositions, batchSize, batchOutput);
      }

      processed += actualPositions;
      lastBatchSize = batchSize;
    }
  }


  /// <summary>
  /// Pipelined sub-batch processing with CUDA graph replay.
  /// 
  /// PREREQUISITE: WarmupWithGraphCapture() must have been called first.
  /// All CUDA graphs are pre-captured during warmup, so this method uses direct
  /// graph replay calls without any locking overhead.
  /// 
  /// Pipeline pattern:
  ///   - While batch N computes: CPU prep + H2D for batch N+1 (on stream 1)
  ///   - Sync batch N compute
  ///   - Launch batch N+1 compute immediately (before post-processing N)
  ///   - D2H + post-process batch N (while batch N+1 computes)
  /// 
  /// This overlaps CPU prep and H2D with GPU compute, without concurrent GPU computes.
  /// </summary>
  private unsafe void ProcessBytesWithHandlerPipelined(byte[] input, int totalPositions, SubBatchOutputHandler handler, int globalPositionOffset)
  {
    // Compute batch boundaries using the centralized engine selection logic
    ComputeBatchPlan(totalPositions);

    if (cachedBatchPlan.Count == 0)
    {
      return;
    }

    // For single batch, just use the simple path
    if (cachedBatchPlan.Count == 1)
    {
      (int start, int count, TensorRTEngine engine, int engineBatchSize, bool useDynamic, int engineIndex) = cachedBatchPlan[0];
      int inputBytes = count * InputElementsPerPosition;

      // Copy to pinned memory
      fixed (byte* srcPtr = input)
      {
        Buffer.MemoryCopy(srcPtr, (void*)pinnedIn, inputBytes, inputBytes);
      }

      // H2D, Compute, D2H all on stream 0 - CUDA guarantees in-order execution
      engine.CopyToGPUOnStreamAsync(0, gpuIn, pinnedIn, inputBytes);

      // Use dynamic inference for Range mode to set correct batch size
      if (useDynamic)
      {
        engine.InferOnStreamDynamicAsync(0, gpuIn, gpuOut, count);
      }
      else
      {
        // CUDA graphs already captured during warmup - direct replay (no locking needed)
        engine.InferOnStreamWithGraphAsync(0, gpuIn, gpuOut);
      }

      // For dynamic mode, output size matches actual count; for static mode, it's engine batch size
      int outputPositions = useDynamic ? count : engineBatchSize;
      long outputBytes = outputPositions * OutputElementsPerPosition * sizeof(ushort);
      engine.CopyFromGPUOnStreamAsync(0, pinnedOut, gpuOut, outputBytes);
      engine.SyncStream(0);  // Single sync at end before CPU access

      // Copy to managed buffer and call handler
      fixed (Half* dstPtr = cachedOutputBuffer)
      {
        Buffer.MemoryCopy((void*)pinnedOut, dstPtr, outputBytes, outputBytes);
      }
      // For dynamic: engineBatchSize param = count; for static: engineBatchSize param = engine's batch size
      handler(globalPositionOffset, count, outputPositions, cachedOutputBuffer);
      return;
    }

    // Multi-batch pipelined processing
    // Ping-pong between two sets of input and output buffers
    IntPtr[] pinnedInputs = [pinnedIn, pinnedIn2];
    IntPtr[] gpuInputs = [gpuIn, gpuIn2];
    IntPtr[] pinnedOutputs = [pinnedOut, pinnedOut2];
    IntPtr[] gpuOutputs = [gpuOut, gpuOut2];
    Half[][] managedOutputs = [cachedOutputBuffer, cachedOutputBuffer2];
    byte[][] managedInputs = [cachedByteInputBuffer, cachedByteInputBuffer2];

    // Stage first batch: CPU prep + H2D
    (int batch0Start, int batch0Count, TensorRTEngine batch0Engine, int batch0EngineSize, bool batch0UseDynamic, int batch0EngineIndex) = cachedBatchPlan[0];
    int batch0InputBytes = batch0Count * InputElementsPerPosition;
    Array.Copy(input, 0, managedInputs[0], 0, batch0InputBytes);
    fixed (byte* srcPtr = managedInputs[0])
    {
      Buffer.MemoryCopy(srcPtr, (void*)pinnedInputs[0], batch0InputBytes, batch0InputBytes);
    }
    batch0Engine.CopyToGPUOnStreamAsync(0, gpuInputs[0], pinnedInputs[0], batch0InputBytes);
    // No sync needed - Infer on same stream will wait for H2D

    // Launch first batch compute (output to buffer 0)
    if (batch0UseDynamic)
    {
      batch0Engine.InferOnStreamDynamicAsync(0, gpuInputs[0], gpuOutputs[0], batch0Count);
    }
    else
    {
      batch0Engine.InferOnStreamWithGraphAsync(0, gpuInputs[0], gpuOutputs[0]);
    }

    // Process remaining batches with pre-staging
    for (int i = 1; i < cachedBatchPlan.Count; i++)
    {
      int prevBuffer = (i - 1) % 2;
      int currBuffer = i % 2;

      (int currStart, int currCount, TensorRTEngine currEngine, int currEngineSize, bool currUseDynamic, int currEngineIndex) = cachedBatchPlan[i];

      // WHILE PREVIOUS BATCH COMPUTES: Pre-stage current batch (CPU prep + H2D on stream 1)
      {
        int inputBytes = currCount * InputElementsPerPosition;
        int inputOffset = currStart * InputElementsPerPosition;

        // CPU prep: copy to managed buffer, then to pinned memory
        Array.Copy(input, inputOffset, managedInputs[currBuffer], 0, inputBytes);
        fixed (byte* srcPtr = managedInputs[currBuffer])
        {
          Buffer.MemoryCopy(srcPtr, (void*)pinnedInputs[currBuffer], inputBytes, inputBytes);
        }

        // H2D on stream 1 (overlaps with compute on stream 0)
        currEngine.CopyToGPUOnStreamAsync(1, gpuInputs[currBuffer], pinnedInputs[currBuffer], inputBytes);
      }

      // Wait for H2D first (typically faster than compute, so returns quickly)
      currEngine.SyncStream(1);

      // Wait for previous batch compute to finish (sync on previous engine's stream)
      TensorRTEngine prevEngine = cachedBatchPlan[i - 1].engine;
      prevEngine.SyncStream(0);

      // Launch current batch compute to alternating output buffer
      if (currUseDynamic)
      {
        currEngine.InferOnStreamDynamicAsync(0, gpuInputs[currBuffer], gpuOutputs[currBuffer], currCount);
      }
      else
      {
        // CUDA graphs already captured during warmup - direct replay (no locking needed)
        currEngine.InferOnStreamWithGraphAsync(0, gpuInputs[currBuffer], gpuOutputs[currBuffer]);
      }

      // D2H + post-process previous batch while current batch computes
      // Previous batch wrote to gpuOutputs[prevBuffer], so read from there (no race!)
      {
        (int prevStart, int prevCount, TensorRTEngine prevBatchEngine, int prevEngineSize, bool prevUseDynamic, int prevEngineIndex) = cachedBatchPlan[i - 1];

        // For dynamic mode, output size matches actual count; for static mode, it's engine batch size
        int prevOutputPositions = prevUseDynamic ? prevCount : prevEngineSize;
        long prevOutputBytes = prevOutputPositions * OutputElementsPerPosition * sizeof(ushort);

        // D2H for previous batch from its dedicated output buffer
        prevBatchEngine.CopyFromGPUOnStreamAsync(1, pinnedOutputs[prevBuffer], gpuOutputs[prevBuffer], prevOutputBytes);
        prevBatchEngine.SyncStream(1);  // Need data for handler

        // Copy to managed buffer and call handler
        fixed (Half* dstPtr = managedOutputs[prevBuffer])
        {
          Buffer.MemoryCopy((void*)pinnedOutputs[prevBuffer], dstPtr, prevOutputBytes, prevOutputBytes);
        }
        // For dynamic: engineBatchSize param = count; for static: engineBatchSize param = engine's batch size
        handler(globalPositionOffset + prevStart, prevCount, prevOutputPositions, managedOutputs[prevBuffer]);
      }
    }

    // Handle last batch (no next batch to pre-stage)
    {
      int lastIdx = cachedBatchPlan.Count - 1;
      int lastBuffer = lastIdx % 2;
      (int start, int count, TensorRTEngine lastEngine, int lastEngineSize, bool lastUseDynamic, int lastEngineIndex) = cachedBatchPlan[lastIdx];

      // For dynamic mode, output size matches actual count; for static mode, it's engine batch size
      int lastOutputPositions = lastUseDynamic ? count : lastEngineSize;
      long outputBytes = lastOutputPositions * OutputElementsPerPosition * sizeof(ushort);

      // D2H on same stream as compute - CUDA guarantees compute finishes first (in-order execution)
      lastEngine.CopyFromGPUOnStreamAsync(0, pinnedOutputs[lastBuffer], gpuOutputs[lastBuffer], outputBytes);
      lastEngine.SyncStream(0);  // Single sync before CPU access

      // Copy to managed buffer and call handler
      fixed (Half* dstPtr = managedOutputs[lastBuffer])
      {
        Buffer.MemoryCopy((void*)pinnedOutputs[lastBuffer], dstPtr, outputBytes, outputBytes);
      }
      // For dynamic: engineBatchSize param = count; for static: engineBatchSize param = engine's batch size
      handler(globalPositionOffset + start, count, lastOutputPositions, managedOutputs[lastBuffer]);
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
    int[] batchSizes = BatchScheduler.ScheduleSingleGPU(engineSizes, orderedExecutionTimes, totalPositions, deviceId);

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

    // Add per-batch overhead for each batch beyond the first (matches BatchScheduler DP formula)
    if (cachedBatchPlan.Count > 1)
    {
      totalTime += (cachedBatchPlan.Count - 1) * BatchScheduler.PER_BATCH_OVERHEAD_MS;
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

    TensorRTNative.FreePinned(pinnedIn);
    TensorRTNative.FreePinned(pinnedOut);
    TensorRTNative.FreeGPU(gpuIn);
    TensorRTNative.FreeGPU(gpuOut);

    // Free pipelined sub-batch buffers
    if (USE_PIPELINED_SUBBATCHES)
    {
      TensorRTNative.FreePinned(pinnedIn2);
      TensorRTNative.FreeGPU(gpuIn2);
      TensorRTNative.FreePinned(pinnedOut2);
      TensorRTNative.FreeGPU(gpuOut2);
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
