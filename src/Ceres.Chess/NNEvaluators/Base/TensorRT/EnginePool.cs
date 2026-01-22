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

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

public sealed class EnginePool : IDisposable
{
  /// <summary>
  /// If true, outputs a line for each batch executed showing batch details.
  /// </summary>
  private const bool VERBOSE_DUMP_BATCHES = false;

  private readonly TensorRT trt;
  private readonly bool ownsTrt;
  private readonly List<TensorRTEngine> engines = new();
  private readonly List<(int min, int max)> ranges = new();
  private readonly EnginePoolMode mode;
  private readonly string onnxPath;
  private readonly int deviceId;
  private readonly bool useByteInputs;
  private bool disposed;

  // Pinned memory and GPU buffers for the largest engine
  private IntPtr pinnedIn;
  private IntPtr pinnedOut;
  private IntPtr gpuIn;
  private IntPtr gpuOut;
  private long maxInputBytes;
  private long maxOutputBytes;

  // Pre-allocated managed buffers to reduce GC pressure
  private byte[] cachedByteInputBuffer;
  private Half[] cachedHalfInputBuffer;
  private Half[] cachedOutputBuffer;

  // Per-engine cached buffers for Exact mode (indexed by engine index)
  // Eliminates per-batch allocations when using multiple exact-size engines
  private byte[][] perEngineByteInputBuffers;
  private Half[][] perEngineHalfInputBuffers;
  private Half[][] perEngineOutputBuffers;



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


  public int InputElementsPerPosition { get; private set; }
  public int OutputElementsPerPosition { get; private set; }
  public int DeviceId => deviceId;
  public bool UseByteInputs => useByteInputs;

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
  public int MaxEngineBatchSize =>
      mode == EnginePoolMode.Range ? ranges[^1].max : ranges[0].max;

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
          // TODO: currently always disabled: opts.TilingOptimizationLevel = 3; // note: can be 5x to 20x slower to build
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

    // Pre-allocate managed buffers sized for largest engine
    long maxInputElements = largestEngine.TotalInputSize;
    long maxOutputElements = largestEngine.TotalOutputSize;
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
    ExactModeBatchFractionAcceptablePadding = isIntegrated ? 0.20f : 0.30f;
    ExactModeBatchNumPositionsAcceptablePadding = isIntegrated ? 4 : 12;

    // Note: Warmup is skipped here - caller should warmup before benchmarking
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
        // Dynamic inference: reuse pre-allocated max-size buffers (no new allocations!)
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
        // Static inference: use pre-allocated per-engine buffers (no new allocations!)
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
        // Dynamic inference: reuse pre-allocated max-size buffers (no new allocations!)
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
        // Static inference: use pre-allocated per-engine buffers (no new allocations!)
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
        // Dynamic inference: reuse pre-allocated max-size buffers (no new allocations!)
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
        // Static inference: use pre-allocated per-engine buffers (no new allocations!)
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
        // Dynamic inference: reuse pre-allocated max-size buffers (no new allocations!)
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
        // Static inference: use pre-allocated per-engine buffers (no new allocations!)
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
      // First, check if any engine can handle the request with acceptable padding.
      // Acceptable padding is satisfied if EITHER:
      //   1. Fraction-based: (engineSize - remaining) <= engineSize * ExactModeBatchFractionAcceptablePadding
      //      Which simplifies to: remaining >= engineSize * (1 - ExactModeBatchFractionAcceptablePadding)
      //   2. Absolute: (engineSize - remaining) <= ExactModeBatchNumPositionsAcceptablePadding
      float minFillFraction = 1.0f - ExactModeBatchFractionAcceptablePadding;

      for (int i = 0; i < ranges.Count; i++)
      {
        int engineSize = ranges[i].min;

        if (engineSize <= remaining)
        {
          // Engine fits exactly (no padding needed), use it
          return (engines[i], engineSize, i);
        }

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
