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
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Multi-GPU engine pool that distributes work across multiple GPUs.
/// </summary>
public sealed class MultiGPUEnginePool : IDisposable
{
  /// <summary>
  /// Maximum number of GPUs supported for stackalloc optimization.
  /// </summary>
  private const int MAX_GPUS_STACKALLOC = 16;

  /// <summary>
  /// When true, runs batch size optimization during warmup and actually rebuilds
  /// engines with the optimized sizes for improved throughput.
  /// N.B. This significantly slows down initialization and has unclear benefit.
  ///      Therefore disabled by default (but can be run manually for experimentation).
  /// </summary>
  public static bool APPLY_BATCH_SIZE_OPTIMIZATION = false;


  private readonly TensorRT trt;
  private List<EnginePool> pools = new();
  private readonly int[] deviceIDs;
  private readonly int minBatchSizePerGPU;
  private readonly EnginePoolMode mode;
  private int[][] sizesPerGPU;
  private readonly bool useByteInputs;
  private bool disposed;

  // Fields needed for rebuilding pools with optimized sizes
  private readonly string onnxPath;
  private readonly TensorRTBuildOptions buildOptions;
  private readonly string cacheDir;

  // Pinned memory buffers for async transfers
  private IntPtr pinnedInput;
  private IntPtr pinnedOutput;
  private long pinnedInputBytes;
  private long pinnedOutputBytes;
  // Cached arrays per GPU to avoid per-batch allocations
  private readonly Half[][] cachedHalfInputs;
  private readonly Half[][] cachedHalfOutputs;
  private readonly byte[][] cachedByteInputs;
  private readonly int[] cachedInputCapacities;
  private readonly int[] cachedOutputCapacities;

  // Cached unique device set (cleared and reused)
  private readonly HashSet<int> cachedUniqueDevices = new();

  // Execution times (average) by GPU (outer index) and batch size index (inner index)
  // Initialized during Warmup()
  private float[][] executionTimesPerGPU;

  // Device names for each GPU (e.g., "NVIDIA RTX PRO 6000 Blackwell")
  // Used for grouping identical GPUs when computing speed fractions
  private readonly string[] deviceNames;

  // Cached speed-normalized fractions for each GPU count (computed once after warmup)
  // Index is numGPUs, value is the fractions array for that GPU count
  private float[][] cachedSpeedFractions;

  // Precomputed per-GPU cost tables: gpuCostTable[g][n] = min execution time (ms)
  // for GPU g to process n positions using optimal engine combination.
  // Computed during Warmup() via DP over engine sizes.
  private float[][] gpuCostTable;
  private int maxCostTableSize;

  // Preallocated DP buffers for TryComputeOptimalDistribution (avoids per-call allocation)
  private float[] dpBufA, dpBufB;
  private int[] dpChoices;

  // Per-session distribution cache: since engine sizes, GPU timings, and penalty constants
  // are fixed for the session, the optimal distribution for a given totalPositions is invariant.
  // distCache[totalPositions * pools.Count + g] = positions assigned to GPU g.
  private int[] distCache;
  private bool[] distCacheValid;

  /// <summary>
  /// Input elements per position.
  /// </summary>
  public int InputElementsPerPosition { get; private set; }

  /// <summary>
  /// Output elements per position.
  /// </summary>
  public int OutputElementsPerPosition { get; private set; }

  /// <summary>
  /// Number of GPU devices in the pool.
  /// </summary>
  public int NumDevices => deviceIDs.Length;

  /// <summary>
  /// Whether inputs are byte (INT8) format.
  /// </summary>
  public bool UseByteInputs => useByteInputs;

  /// <summary>
  /// Gets output tensor info from the first pool (same layout for all).
  /// </summary>
  public OutputTensorInfo[] GetOutputTensorInfo() => pools[0].GetOutputTensorInfo();

  /// <summary>
  /// Gets input tensor name from the first pool.
  /// </summary>
  public string GetInputName(int index) => pools[0].GetInputName(index);

  /// <summary>
  /// Gets the largest engine batch size across all pools.
  /// </summary>
  public int MaxEngineBatchSize => pools[0].MaxEngineBatchSize;

  /// <summary>
  /// Execution times (in milliseconds) by GPU (outer index) and batch size index (inner index).
  /// Populated by Warmup() method. Returns null if Warmup() has not been called.
  /// </summary>
  public float[][] ExecutionTimesPerGPU => executionTimesPerGPU;

  /// <summary>
  /// Constructor.
  /// </summary>
  public MultiGPUEnginePool(TensorRT trt, string onnxPath, int[][] sizesPerGPU, EnginePoolMode mode,
                             TensorRTBuildOptions options, int inputElementsPerPos, int outputElementsPerPos,
                             int[] deviceIds, int minBatchSizePerGPU, string cacheDir)
  {
    this.trt = trt;
    this.onnxPath = onnxPath;
    this.buildOptions = options;
    this.cacheDir = cacheDir;
    deviceIDs = deviceIds;
    this.minBatchSizePerGPU = minBatchSizePerGPU;
    this.mode = mode;
    this.sizesPerGPU = sizesPerGPU;

    // Load engines in parallel across GPUs when enabled, multi-GPU, and all device IDs are unique
    // (duplicate device IDs can occur for testing purposes and require sequential loading)
    bool hasDuplicateDevices = deviceIds.Length != deviceIds.Distinct().Count();
    bool useParallel = NNEvaluatorTensorRT.PARALLEL_ENGINE_LOAD_ENABLED
                       && deviceIds.Length > 1
                       && !hasDuplicateDevices;

    if (useParallel)
    {
      // Pre-allocate array for thread-safe parallel assignment
      EnginePool[] poolsArray = new EnginePool[deviceIds.Length];

      Parallel.For(0, deviceIds.Length, i =>
      {
        poolsArray[i] = new EnginePool(trt, onnxPath, (int[])sizesPerGPU[i].Clone(), mode, options,
                                        inputElementsPerPos, outputElementsPerPos, deviceIds[i], cacheDir);
      });

      // Add to list in order
      for (int i = 0; i < deviceIds.Length; i++)
      {
        pools.Add(poolsArray[i]);
      }
    }
    else
    {
      // Sequential loading for single GPU or when parallel loading is disabled
      for (int i = 0; i < deviceIds.Length; i++)
      {
        EnginePool pool = new EnginePool(trt, onnxPath, (int[])sizesPerGPU[i].Clone(), mode, options,
                                          inputElementsPerPos, outputElementsPerPos, deviceIds[i], cacheDir);
        pools.Add(pool);
      }
    }

    InputElementsPerPosition = pools[0].InputElementsPerPosition;
    OutputElementsPerPosition = pools[0].OutputElementsPerPosition;

    useByteInputs = pools[0].UseByteInputs;

    int maxBatch = GetMaxBatchSize();
    pinnedInputBytes = maxBatch * InputElementsPerPosition * sizeof(ushort);
    pinnedOutputBytes = maxBatch * OutputElementsPerPosition * sizeof(ushort);
    pinnedInput = TensorRTNative.AllocPinned(pinnedInputBytes);
    pinnedOutput = TensorRTNative.AllocPinned(pinnedOutputBytes);
    // Initialize cached arrays for each GPU
    int numPools = pools.Count;
    cachedHalfInputs = new Half[numPools][];
    cachedHalfOutputs = new Half[numPools][];
    cachedByteInputs = new byte[numPools][];
    cachedInputCapacities = new int[numPools];
    cachedOutputCapacities = new int[numPools];

    // Get device names for each GPU (used for grouping identical GPUs)
    deviceNames = new string[numPools];
    for (int i = 0; i < numPools; i++)
    {
      deviceNames[i] = TensorRTNative.GetDeviceName(deviceIds[i]);
    }

    Warmup();
  }


  private int GetMaxBatchSize()
  {
    int max = 0;
    foreach (int[] gpuSizes in sizesPerGPU)
    {
      int gpuMax = mode == EnginePoolMode.Range ? gpuSizes[^1] : gpuSizes.Max();
      max = Math.Max(max, gpuMax);
    }
    return max;
  }


  /// <summary>
  /// Ensures cached Half input/output arrays for the given GPU index have sufficient capacity.
  /// </summary>
  private void EnsureHalfArrayCapacity(int gpuIndex, int requiredInputElements, int requiredOutputElements)
  {
    if (cachedInputCapacities[gpuIndex] < requiredInputElements)
    {
      cachedHalfInputs[gpuIndex] = new Half[requiredInputElements];
      cachedInputCapacities[gpuIndex] = requiredInputElements;
    }

    if (cachedOutputCapacities[gpuIndex] < requiredOutputElements)
    {
      cachedHalfOutputs[gpuIndex] = new Half[requiredOutputElements];
      cachedOutputCapacities[gpuIndex] = requiredOutputElements;
    }
  }


  /// <summary>
  /// Ensures cached byte input array for the given GPU index has sufficient capacity.
  /// </summary>
  private void EnsureByteInputCapacity(int gpuIndex, int requiredInputElements)
  {
    if (cachedInputCapacities[gpuIndex] < requiredInputElements)
    {
      cachedByteInputs[gpuIndex] = new byte[requiredInputElements];
      cachedInputCapacities[gpuIndex] = requiredInputElements;
    }
  }


  /// <summary>
  /// Synchronizes all unique devices in the pool.
  /// </summary>
  private void SynchronizeUniqueDevices()
  {
    cachedUniqueDevices.Clear();
    foreach (int deviceId in deviceIDs)
    {
      if (cachedUniqueDevices.Add(deviceId))
      {
        TensorRTNative.SynchronizeDevice(deviceId);
      }
    }
  }


  /// <summary>
  /// Synchronizes only the devices that were marked as unique.
  /// </summary>
  private void SynchronizeTrackedDevices()
  {
    foreach (int deviceId in cachedUniqueDevices)
    {
      TensorRTNative.SynchronizeDevice(deviceId);
    }
  }


  /// <summary>
  /// Warms up all GPUs and benchmarks execution times for each batch size.
  /// Runs each batch size 3 times and takes the minimum (best) time as the estimate.
  /// Must be called before using ExecutionTimesPerGPU property.
  /// </summary>
  public void Warmup()
  {
    const int WARMUP_ITERATIONS = 5;

    int numGPUs = pools.Count;

    executionTimesPerGPU = new float[numGPUs][];

    for (int gpuIndex = 0; gpuIndex < numGPUs; gpuIndex++)
    {
      int[] gpuSizes = sizesPerGPU[gpuIndex];
      int numBatchSizes = gpuSizes.Length;
      executionTimesPerGPU[gpuIndex] = new float[numBatchSizes];
      EnginePool pool = pools[gpuIndex];

      for (int sizeIndex = 0; sizeIndex < numBatchSizes; sizeIndex++)
      {
        int batchSize = gpuSizes[sizeIndex];

        // Allocate dummy input/output arrays for this batch size
        int inputElements = batchSize * InputElementsPerPosition;
        int outputElements = batchSize * OutputElementsPerPosition;

        float bestTimeMs = float.MaxValue;

        if (useByteInputs)
        {
          byte[] dummyInput = new byte[inputElements];
          Half[] dummyOutput = new Half[outputElements];

          for (int iter = 0; iter < WARMUP_ITERATIONS; iter++)
          {
            // Synchronize before timing
            TensorRTNative.SynchronizeDevice(deviceIDs[gpuIndex]);

            Stopwatch sw = Stopwatch.StartNew();
            pool.ProcessBytes(dummyInput, dummyOutput, batchSize);
            TensorRTNative.SynchronizeDevice(deviceIDs[gpuIndex]);
            sw.Stop();

            float elapsedMs = (float)sw.Elapsed.TotalMilliseconds;
            bestTimeMs = Math.Min(bestTimeMs, elapsedMs);
          }
        }
        else
        {
          Half[] dummyInput = new Half[inputElements];
          Half[] dummyOutput = new Half[outputElements];

          for (int iter = 0; iter < WARMUP_ITERATIONS; iter++)
          {
            // Synchronize before timing
            TensorRTNative.SynchronizeDevice(deviceIDs[gpuIndex]);

            Stopwatch sw = Stopwatch.StartNew();
            pool.Process(dummyInput, dummyOutput, batchSize);
            TensorRTNative.SynchronizeDevice(deviceIDs[gpuIndex]);
            sw.Stop();

            float elapsedMs = (float)sw.Elapsed.TotalMilliseconds;
            bestTimeMs = Math.Min(bestTimeMs, elapsedMs);
          }
        }

        executionTimesPerGPU[gpuIndex][sizeIndex] = bestTimeMs;
      }

      // Output raw timing estimates for this GPU
      string sizesStr = string.Join(", ", sizesPerGPU[gpuIndex]);
      string timingsStr = string.Join(", ", executionTimesPerGPU[gpuIndex].Select(t => t.ToString("F1")));
      Console.WriteLine($"DEVICE {deviceIDs[gpuIndex]} sizes: [{sizesStr}] timings: [{timingsStr}] ms ({deviceNames[gpuIndex]})");
    }

    // Average execution timings for identical GPU types to reduce measurement noise.
    // GPUs with the same device name have the same SM count and engine sizes.
    float[][] rawTimings = new float[numGPUs][];
    for (int g = 0; g < numGPUs; g++)
    {
      rawTimings[g] = (float[])executionTimesPerGPU[g].Clone();
    }
    for (int g = 0; g < numGPUs; g++)
    {
      string name = deviceNames[g];
      if (name == null)
      {
        continue;
      }
      int numSizes = rawTimings[g].Length;
      int count = 0;
      for (int other = 0; other < numGPUs; other++)
      {
        if (string.Equals(name, deviceNames[other], StringComparison.Ordinal)
            && rawTimings[other].Length == numSizes)
        {
          count++;
        }
      }
      if (count <= 1)
      {
        continue;
      }
      for (int s = 0; s < numSizes; s++)
      {
        float sum = 0;
        for (int other = 0; other < numGPUs; other++)
        {
          if (string.Equals(name, deviceNames[other], StringComparison.Ordinal))
          {
            sum += rawTimings[other][s];
          }
        }
        executionTimesPerGPU[g][s] = sum / count;
      }
    }

    // Log averaged timings and pass to each EnginePool
    for (int g = 0; g < numGPUs; g++)
    {
      if (executionTimesPerGPU[g][0] != rawTimings[g][0] || executionTimesPerGPU[g][^1] != rawTimings[g][^1])
      {
        string avgStr = string.Join(", ", executionTimesPerGPU[g].Select(t => t.ToString("F1")));
        Console.WriteLine($"DEVICE {deviceIDs[g]} averaged timings: [{avgStr}] ms");
      }
      pools[g].ExecutionTimes = executionTimesPerGPU[g];
    }

    // Run batch size optimization if enabled
    if (APPLY_BATCH_SIZE_OPTIMIZATION && mode == EnginePoolMode.Exact)
    {
      int maxBatchSize = sizesPerGPU[0][^1] * numGPUs;
      BatchSizeOptimizer.OptimizationResult result = BatchSizeOptimizer.Optimize(sizesPerGPU[0], executionTimesPerGPU, numGPUs, 1024);
      
      // Check if sizes actually changed
      bool changed = false;
      for (int i = 0; i < sizesPerGPU[0].Length; i++)
      {
        if (sizesPerGPU[0][i] != result.OptimizedSizes[i])
        {
          changed = true;
          break;
        }
      }

      if (changed)
      {
        Console.WriteLine($"Batch size optimization: [{string.Join(", ", result.OriginalSizes)}] -> [{string.Join(", ", result.OptimizedSizes)}]");
        float origThroughput = -result.OriginalScore / 1000f;
        float optThroughput = -result.OptimizedScore / 1000f;
        float pctImprove = origThroughput > 0 ? (optThroughput - origThroughput) / origThroughput * 100f : 0f;
        Console.WriteLine($"  Estimated throughput: {origThroughput:F0}k -> {optThroughput:F0}k nps ({pctImprove:+0.0;-0.0}%)");

        // Rebuild pools with optimized sizes
        RebuildPoolsWithSizes(result.OptimizedSizes);
        return; // Warmup will be called again by RebuildPoolsWithSizes
      }
      else
      {
        Console.WriteLine("Batch size optimization: sizes are near-optimal.");
      }
    }

    // Pre-compute speed fractions for all possible GPU counts (1 to numGPUs)
    // This avoids recomputation on every ComputeOptimalDistribution call
    cachedSpeedFractions = new float[numGPUs + 1][];
    for (int n = 1; n <= numGPUs; n++)
    {
      cachedSpeedFractions[n] = BatchScheduler.ComputeSpeedNormalizedFractions(executionTimesPerGPU, deviceNames, n);
    }

    // Precompute cost-per-position tables for engine-aware multi-GPU distribution.
    // gpuCostTable[g][n] = min execution time for GPU g to process n positions,
    // using DP: cost[n] = min over engines e of (cost[max(0,n-sizes[e])] + times[e] + overhead).
    if (mode == EnginePoolMode.Exact)
    {
      int maxSize = 0;
      foreach (int[] gpuSizes in sizesPerGPU)
      {
        maxSize = Math.Max(maxSize, gpuSizes[^1]);
      }
      int maxN = maxSize * numGPUs;
      maxCostTableSize = maxN;
      gpuCostTable = new float[numGPUs][];
      for (int g = 0; g < numGPUs; g++)
      {
        int[] gSizes = sizesPerGPU[g];
        int numGpuBatchSizes = gSizes.Length;
        gpuCostTable[g] = new float[maxN + 1];
        gpuCostTable[g][0] = 0;
        for (int n = 1; n <= maxN; n++)
        {
          float best = float.MaxValue;
          for (int e = 0; e < numGpuBatchSizes; e++)
          {
            int remainder = Math.Max(0, n - gSizes[e]);
            float credit = remainder > 0 ? BatchScheduler.PER_BATCH_OVERHEAD_MS : 0;
            float cost = gpuCostTable[g][remainder]
                       + executionTimesPerGPU[g][e]
                       + credit;
            if (cost < best)
            {
              best = cost;
            }
          }
          gpuCostTable[g][n] = best;
        }
      }

      dpBufA = new float[maxN + 1];
      dpBufB = new float[maxN + 1];
      dpChoices = new int[numGPUs * (maxN + 1)];
      distCache = new int[(maxN + 1) * numGPUs];
      distCacheValid = new bool[maxN + 1];
    }
  }


  /// <summary>
  /// Rebuilds all engine pools with new batch sizes.
  /// Disposes existing pools and creates new ones with optimized sizes.
  /// </summary>
  private void RebuildPoolsWithSizes(int[] newSizes)
  {
    // Dispose existing pools
    foreach (EnginePool pool in pools)
    {
      pool.Dispose();
    }
    pools.Clear();

    // Update sizesPerGPU with new sizes for all GPUs
    for (int i = 0; i < deviceIDs.Length; i++)
    {
      sizesPerGPU[i] = NNEvaluatorTensorRT.AdjustToSM(deviceIDs[i], newSizes);
    }

    Console.WriteLine($"Rebuilding engines with optimized sizes...");

    // Rebuild pools with new sizes (same logic as constructor)
    bool hasDuplicateDevices = deviceIDs.Length != deviceIDs.Distinct().Count();
    bool useParallel = NNEvaluatorTensorRT.PARALLEL_ENGINE_LOAD_ENABLED
                       && deviceIDs.Length > 1
                       && !hasDuplicateDevices;

    if (useParallel)
    {
      EnginePool[] poolsArray = new EnginePool[deviceIDs.Length];
      Parallel.For(0, deviceIDs.Length, i =>
      {
        poolsArray[i] = new EnginePool(trt, onnxPath, (int[])sizesPerGPU[i].Clone(), mode, buildOptions,
                                        InputElementsPerPosition, OutputElementsPerPosition, deviceIDs[i], cacheDir);
      });
      for (int i = 0; i < deviceIDs.Length; i++)
      {
        pools.Add(poolsArray[i]);
      }
    }
    else
    {
      for (int i = 0; i < deviceIDs.Length; i++)
      {
        EnginePool pool = new EnginePool(trt, onnxPath, (int[])sizesPerGPU[i].Clone(), mode, buildOptions,
                                          InputElementsPerPosition, OutputElementsPerPosition, deviceIDs[i], cacheDir);
        pools.Add(pool);
      }
    }

    // Clear caches that depend on old sizes
    cachedSpeedFractions = null;
    gpuCostTable = null;
    distCache = null;
    distCacheValid = null;

    // Re-run warmup with new engines (this will measure timings and rebuild cost tables)
    // Temporarily disable optimization to prevent infinite recursion
    bool savedFlag = APPLY_BATCH_SIZE_OPTIMIZATION;
    APPLY_BATCH_SIZE_OPTIMIZATION = false;
    Warmup();
    APPLY_BATCH_SIZE_OPTIMIZATION = savedFlag;
  }


  private bool ShouldUseSingleGPU(int totalPositions)
  {
    if (pools.Count == 1)
    {
      return true;
    }

    if (totalPositions < minBatchSizePerGPU * 2)
    {
      return true;
    }

    // When cost tables are available, compare single-GPU cost against
    // the best multi-GPU estimate (with per-GPU coordination overhead).
    if (gpuCostTable != null && EnginePool.OPTIMIZED_SCHEDULING && totalPositions <= maxCostTableSize)
    {
      float singleGPUCost = gpuCostTable[0][totalPositions];
      int maxGPUs = Math.Min(pools.Count, totalPositions / Math.Max(1, minBatchSizePerGPU));
      for (int k = 2; k <= maxGPUs; k++)
      {
        int perGPU = (totalPositions + k - 1) / k;
        float worstGPUCost = 0;
        for (int g = 0; g < k; g++)
        {
          float cost = gpuCostTable[g][perGPU];
          if (cost > worstGPUCost)
          {
            worstGPUCost = cost;
          }
        }
        if (worstGPUCost + k * BatchScheduler.PER_GPU_FIXED_COST_MS < singleGPUCost)
        {
          return false; // Multi-GPU with k GPUs beats single GPU
        }
      }
      return true; // No multi-GPU config beats single GPU with overhead
    }

    if (mode == EnginePoolMode.Exact)
    {
      foreach (int size in sizesPerGPU[0])
      {
        if (totalPositions <= size && (size - totalPositions) < minBatchSizePerGPU)
        {
          return true;
        }
      }
    }

    if (mode == EnginePoolMode.Range)
    {
      int prevMax = 0;
      foreach (int maxSize in sizesPerGPU[0])
      {
        int minSize = prevMax + 1;
        if (totalPositions >= minSize && totalPositions <= maxSize)
        {
          return true;
        }
        prevMax = maxSize;
      }
    }

    return false;
  }


  /// <summary>
  /// Process batch with Half inputs.
  /// </summary>
  public void Process(Half[] input, Half[] output, int totalPositions)
  {
    for (int i = 0; i < pools.Count; i++) pools[i].MarkNewBatch();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].Process(input, output, totalPositions);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    if (numGPUs > MAX_GPUS_STACKALLOC)
    {
      throw new InvalidOperationException($"Number of GPUs ({numGPUs}) exceeds maximum supported ({MAX_GPUS_STACKALLOC}).");
    }

    Span<int> starts = stackalloc int[numGPUs];
    Span<int> counts = stackalloc int[numGPUs];
    int offset = 0;

    // Try to get optimal distribution, fall back to equal split
    if (TryComputeOptimalDistribution(totalPositions, numGPUs, counts))
    {
      // Compute starts from the optimal counts
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        offset += counts[i];
      }
    }
    else
    {
      // Fall back to equal distribution
      int baseSize = totalPositions / numGPUs;
      int remainder = totalPositions % numGPUs;
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        counts[i] = baseSize + (i < remainder ? 1 : 0);
        offset += counts[i];
      }
    }

    // Trim unused trailing GPUs (distribution may use fewer than numGPUs)
    while (numGPUs > 1 && counts[numGPUs - 1] == 0)
    {
      numGPUs--;
    }

    SynchronizeUniqueDevices();

    // Prepare cached sub-arrays and copy input data
    for (int i = 0; i < numGPUs; i++)
    {
      int inputElements = counts[i] * InputElementsPerPosition;
      int outputElements = counts[i] * OutputElementsPerPosition;
      EnsureHalfArrayCapacity(i, inputElements, outputElements);
      Array.Copy(input, starts[i] * InputElementsPerPosition, cachedHalfInputs[i], 0, inputElements);
    }

    // Convert to arrays for lambda capture (Span cannot be captured)
    int[] startsArray = starts.Slice(0, numGPUs).ToArray();
    int[] countsArray = counts.Slice(0, numGPUs).ToArray();

    Parallel.For(0, numGPUs, i =>
    {
      pools[i].Process(cachedHalfInputs[i], cachedHalfOutputs[i], countsArray[i]);
    });

    SynchronizeTrackedDevices();

    for (int i = 0; i < numGPUs; i++)
    {
      int outputElements = counts[i] * OutputElementsPerPosition;
      Array.Copy(cachedHalfOutputs[i], 0, output, starts[i] * OutputElementsPerPosition, outputElements);
    }
  }


  /// <summary>
  /// Process batch with byte inputs.
  /// </summary>
  public void ProcessBytes(byte[] input, Half[] output, int totalPositions)
  {
    for (int i = 0; i < pools.Count; i++) pools[i].MarkNewBatch();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].ProcessBytes(input, output, totalPositions);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    if (numGPUs > MAX_GPUS_STACKALLOC)
    {
      throw new InvalidOperationException($"Number of GPUs ({numGPUs}) exceeds maximum supported ({MAX_GPUS_STACKALLOC}).");
    }

    Span<int> starts = stackalloc int[numGPUs];
    Span<int> counts = stackalloc int[numGPUs];
    int offset = 0;

    // Try to get optimal distribution, fall back to equal split
    if (TryComputeOptimalDistribution(totalPositions, numGPUs, counts))
    {
      // Compute starts from the optimal counts
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        offset += counts[i];
      }
    }
    else
    {
      // Fall back to equal distribution
      int baseSize = totalPositions / numGPUs;
      int remainder = totalPositions % numGPUs;
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        counts[i] = baseSize + (i < remainder ? 1 : 0);
        offset += counts[i];
      }
    }

    // Trim unused trailing GPUs (distribution may use fewer than numGPUs)
    while (numGPUs > 1 && counts[numGPUs - 1] == 0)
    {
      numGPUs--;
    }

    SynchronizeUniqueDevices();

    // Prepare cached sub-arrays and copy input data
    for (int i = 0; i < numGPUs; i++)
    {
      int inputElements = counts[i] * InputElementsPerPosition;
      int outputElements = counts[i] * OutputElementsPerPosition;
      EnsureByteInputCapacity(i, inputElements);
      EnsureHalfArrayCapacity(i, 0, outputElements);
      Array.Copy(input, starts[i] * InputElementsPerPosition, cachedByteInputs[i], 0, inputElements);
    }

    // Convert to arrays for lambda capture (Span cannot be captured)
    int[] startsArray = starts.Slice(0, numGPUs).ToArray();
    int[] countsArray = counts.Slice(0, numGPUs).ToArray();

    Parallel.For(0, numGPUs, i =>
    {
      pools[i].ProcessBytes(cachedByteInputs[i], cachedHalfOutputs[i], countsArray[i]);
    });

    SynchronizeTrackedDevices();

    for (int i = 0; i < numGPUs; i++)
    {
      int outputElements = counts[i] * OutputElementsPerPosition;
      Array.Copy(cachedHalfOutputs[i], 0, output, starts[i] * OutputElementsPerPosition, outputElements);
    }
  }


  /// <summary>
  /// Process with callback for tensor-major output extraction.
  /// Thread-safe: handler is called concurrently from multiple GPU threads.
  /// Handlers must use thread-local buffers and write to non-overlapping position ranges.
  /// </summary>
  public void ProcessWithHandler(Half[] input, int totalPositions, SubBatchOutputHandler handler)
  {
    for (int i = 0; i < pools.Count; i++) pools[i].MarkNewBatch();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].ProcessWithHandler(input, totalPositions, handler, globalPositionOffset: 0);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    if (numGPUs > MAX_GPUS_STACKALLOC)
    {
      throw new InvalidOperationException($"Number of GPUs ({numGPUs}) exceeds maximum supported ({MAX_GPUS_STACKALLOC}).");
    }

    Span<int> starts = stackalloc int[numGPUs];
    Span<int> counts = stackalloc int[numGPUs];
    int offset = 0;

    // Try to get optimal distribution, fall back to equal split
    if (TryComputeOptimalDistribution(totalPositions, numGPUs, counts))
    {
      // Compute starts from the optimal counts
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        offset += counts[i];
      }
    }
    else
    {
      // Fall back to equal distribution
      int baseSize = totalPositions / numGPUs;
      int remainder = totalPositions % numGPUs;
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        counts[i] = baseSize + (i < remainder ? 1 : 0);
        offset += counts[i];
      }
    }

    // Trim unused trailing GPUs (distribution may use fewer than numGPUs)
    while (numGPUs > 1 && counts[numGPUs - 1] == 0)
    {
      numGPUs--;
    }

    SynchronizeUniqueDevices();

    // Prepare cached sub-arrays and copy input data
    for (int i = 0; i < numGPUs; i++)
    {
      int inputElements = counts[i] * InputElementsPerPosition;
      EnsureHalfArrayCapacity(i, inputElements, 0);
      Array.Copy(input, starts[i] * InputElementsPerPosition, cachedHalfInputs[i], 0, inputElements);
    }

    // Convert to arrays for lambda capture (Span cannot be captured)
    int[] startsArray = starts.Slice(0, numGPUs).ToArray();
    int[] countsArray = counts.Slice(0, numGPUs).ToArray();

    Parallel.For(0, numGPUs, i =>
    {
      int capturedStart = startsArray[i];
      SubBatchOutputHandler wrappedHandler = (globalStart, count, engineBatchSize, rawOutput) =>
      {
        int trueGlobalStart = capturedStart + globalStart;
        // No lock needed: handler uses thread-local buffers for intermediate storage,
        // and writes to distinct position ranges in the result arrays (no overlap).
        handler(trueGlobalStart, count, engineBatchSize, rawOutput);
      };

      pools[i].ProcessWithHandler(cachedHalfInputs[i], countsArray[i], wrappedHandler, globalPositionOffset: 0);
    });

    SynchronizeTrackedDevices();
  }


  /// <summary>
  /// Process byte inputs with callback for tensor-major output extraction.
  /// Thread-safe: handler uses thread-local buffers and writes to non-overlapping position ranges.
  /// </summary>
  public void ProcessBytesWithHandler(byte[] input, int totalPositions, SubBatchOutputHandler handler)
  {
    for (int i = 0; i < pools.Count; i++) pools[i].MarkNewBatch();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].ProcessBytesWithHandler(input, totalPositions, handler, globalPositionOffset: 0);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    if (numGPUs > MAX_GPUS_STACKALLOC)
    {
      throw new InvalidOperationException($"Number of GPUs ({numGPUs}) exceeds maximum supported ({MAX_GPUS_STACKALLOC}).");
    }

    Span<int> starts = stackalloc int[numGPUs];
    Span<int> counts = stackalloc int[numGPUs];
    int offset = 0;

    // Try to get optimal distribution, fall back to equal split
    if (TryComputeOptimalDistribution(totalPositions, numGPUs, counts))
    {
      // Compute starts from the optimal counts
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        offset += counts[i];
      }
    }
    else
    {
      // Fall back to equal distribution
      int baseSize = totalPositions / numGPUs;
      int remainder = totalPositions % numGPUs;
      for (int i = 0; i < numGPUs; i++)
      {
        starts[i] = offset;
        counts[i] = baseSize + (i < remainder ? 1 : 0);
        offset += counts[i];
      }
    }

    // Trim unused trailing GPUs (distribution may use fewer than numGPUs)
    while (numGPUs > 1 && counts[numGPUs - 1] == 0)
    {
      numGPUs--;
    }

    SynchronizeUniqueDevices();

    // Prepare cached sub-arrays and copy input data
    for (int i = 0; i < numGPUs; i++)
    {
      int inputElements = counts[i] * InputElementsPerPosition;
      EnsureByteInputCapacity(i, inputElements);
      Array.Copy(input, starts[i] * InputElementsPerPosition, cachedByteInputs[i], 0, inputElements);
    }

    // Convert to arrays for lambda capture (Span cannot be captured)
    int[] startsArray = starts.Slice(0, numGPUs).ToArray();
    int[] countsArray = counts.Slice(0, numGPUs).ToArray();

    Parallel.For(0, numGPUs, i =>
    {
      int capturedStart = startsArray[i];
      SubBatchOutputHandler wrappedHandler = (globalStart, count, engineBatchSize, rawOutput) =>
      {
        int trueGlobalStart = capturedStart + globalStart;
        // No lock needed: handler uses thread-local buffers for intermediate storage,
        // and writes to distinct position ranges in the result arrays (no overlap).
        handler(trueGlobalStart, count, engineBatchSize, rawOutput);
      };

      pools[i].ProcessBytesWithHandler(cachedByteInputs[i], countsArray[i], wrappedHandler, globalPositionOffset: 0);
    });

    SynchronizeTrackedDevices();
  }


  /// <summary>
  /// Computes optimal position distribution across GPUs using DP over position counts.
  /// Uses precomputed gpuCostTable[g][n] (min time for GPU g to process n positions
  /// with optimal engine combination) to find the distribution minimizing makespan.
  /// The per-GPU EnginePool then independently schedules sub-batches for its allocation.
  /// Falls back to fraction-based distribution when cost tables are unavailable.
  /// </summary>
  private bool TryComputeOptimalDistribution(int totalPositions, int numGPUs, Span<int> distribution)
  {
    if (gpuCostTable == null || !EnginePool.OPTIMIZED_SCHEDULING ||
        numGPUs <= 0 || mode != EnginePoolMode.Exact || totalPositions > maxCostTableSize)
    {
      return TryFractionBasedDistribution(totalPositions, numGPUs, distribution);
    }

    // Check session cache (engine sizes, timings, and penalties are constant)
    if (distCacheValid[totalPositions])
    {
      int cBase = totalPositions * pools.Count;
      for (int g = 0; g < numGPUs; g++)
      {
        distribution[g] = distCache[cBase + g];
      }

      if (BatchScheduler.VERBOSE_DETAILS)
      {
        int usedGPUs = 0;
        for (int g = 0; g < numGPUs; g++)
        {
          if (distribution[g] > 0) usedGPUs++;
        }
        Console.WriteLine($"  MULTI_GPU_DIST (cached): {totalPositions} positions -> GPUs used={usedGPUs} " +
                          $"distribution=[{string.Join(", ", distribution.Slice(0, numGPUs).ToArray())}]");
      }

      return true;
    }

    // Select optimal GPU count by running lightweight DP for each k.
    // Unlike an even-split heuristic, this accounts for uneven distributions
    // across heterogeneous GPUs (e.g., NVL vs PCIe).
    int stride = maxCostTableSize + 1;
    int minPerGPU = minBatchSizePerGPU;
    int bestK = 2;
    float bestTotalCost = float.MaxValue;

    for (int k = 2; k <= numGPUs; k++)
    {
      if ((totalPositions + k - 1) / k < minPerGPU)
      {
        break;
      }

      // Lightweight DP (no choice tracking) to compute true makespan for k GPUs
      float[] dpA = dpBufA;
      float[] dpB = dpBufB;
      float[] lastCost = gpuCostTable[k - 1];
      for (int r = 0; r <= totalPositions; r++)
      {
        dpA[r] = r >= minPerGPU ? lastCost[r] : float.MaxValue;
      }

      for (int g = k - 2; g >= 0; g--)
      {
        float[] gCost = gpuCostTable[g];
        int trailing = k - 1 - g;
        for (int r = 0; r <= totalPositions; r++)
        {
          int maxN = r - trailing * minPerGPU;
          if (maxN < minPerGPU)
          {
            dpB[r] = float.MaxValue;
            continue;
          }
          maxN = Math.Min(maxN, maxCostTableSize);

          // Binary search for crossing: smallest ng where gCost[ng] >= dpA[r - ng]
          int lo = minPerGPU, hi = maxN;
          while (lo < hi)
          {
            int mid = lo + (hi - lo) / 2;
            if (gCost[mid] < dpA[r - mid])
            {
              lo = mid + 1;
            }
            else
            {
              hi = mid;
            }
          }
          float best = float.MaxValue;
          int cLo = Math.Max(minPerGPU, lo - 1);
          int cHi = Math.Min(maxN, lo + 1);
          for (int ng = cLo; ng <= cHi; ng++)
          {
            float hv = dpA[r - ng];
            if (hv >= float.MaxValue)
            {
              continue;
            }
            float m = Math.Max(gCost[ng], hv);
            if (m < best)
            {
              best = m;
            }
          }
          dpB[r] = best;
        }
        (dpA, dpB) = (dpB, dpA);
      }

      float makespan = dpA[totalPositions];
      if (makespan < float.MaxValue * 0.5f)
      {
        float total = makespan + k * BatchScheduler.PER_GPU_FIXED_COST_MS;
        if (total < bestTotalCost)
        {
          bestTotalCost = total;
          bestK = k;
        }
      }
    }

    // Base case: last GPU (index bestK-1) handles all remaining positions
    float[] dpNext = dpBufA;
    float[] dpThis = dpBufB;
    float[] lastGpuCost = gpuCostTable[bestK - 1];
    for (int r = 0; r <= totalPositions; r++)
    {
      dpNext[r] = r >= minPerGPU ? lastGpuCost[r] : float.MaxValue;
    }

    // DP from second-to-last GPU down to first.
    // dpNext[r] = min makespan distributing r positions across GPUs [g+1..bestK-1].
    // dpThis[r] = min makespan distributing r positions across GPUs [g..bestK-1].
    for (int g = bestK - 2; g >= 0; g--)
    {
      float[] gpuCost = gpuCostTable[g];
      int choiceBase = g * stride;
      int trailingGPUs = bestK - 1 - g;

      for (int r = 0; r <= totalPositions; r++)
      {
        // Ensure trailing GPUs can each get minPerGPU
        int maxN = r - trailingGPUs * minPerGPU;
        if (maxN < minPerGPU)
        {
          dpThis[r] = float.MaxValue;
          dpChoices[choiceBase + r] = 0;
          continue;
        }
        maxN = Math.Min(maxN, maxCostTableSize);

        // Binary search for crossing: smallest ng where gpuCost[ng] >= dpNext[r - ng]
        int lo = minPerGPU, hi = maxN;
        while (lo < hi)
        {
          int mid = lo + (hi - lo) / 2;
          if (gpuCost[mid] < dpNext[r - mid])
          {
            lo = mid + 1;
          }
          else
          {
            hi = mid;
          }
        }
        float bestMakespan = float.MaxValue;
        int bestN = minPerGPU;
        int checkLo = Math.Max(minPerGPU, lo - 1);
        int checkHi = Math.Min(maxN, lo + 1);
        for (int ng = checkLo; ng <= checkHi; ng++)
        {
          float rest = dpNext[r - ng];
          if (rest >= float.MaxValue)
          {
            continue;
          }
          float makespan = Math.Max(gpuCost[ng], rest);
          if (makespan <= bestMakespan)
          {
            bestMakespan = makespan;
            bestN = ng;
          }
        }

        dpThis[r] = bestMakespan;
        dpChoices[choiceBase + r] = bestN;
      }

      (dpNext, dpThis) = (dpThis, dpNext);
    }

    // Check feasibility
    if (dpNext[totalPositions] >= float.MaxValue * 0.5f)
    {
      if (!TryFractionBasedDistribution(totalPositions, numGPUs, distribution))
      {
        return false;
      }
      StoreDistCache(totalPositions, numGPUs, distribution);
      return true;
    }

    // Backtrack to recover distribution for bestK GPUs
    int rem = totalPositions;
    for (int g = 0; g < bestK - 1; g++)
    {
      int ng = dpChoices[g * stride + rem];
      distribution[g] = ng;
      rem -= ng;
    }
    distribution[bestK - 1] = rem;

    // Zero out unused GPU slots
    for (int g = bestK; g < numGPUs; g++)
    {
      distribution[g] = 0;
    }

    if (BatchScheduler.VERBOSE_DETAILS)
    {
      float makespan = dpNext[totalPositions];
      float penalty = BatchScheduler.PER_GPU_FIXED_COST_MS * Math.Max(0, bestK - 1);
      Console.WriteLine($"  MULTI_GPU_DIST: {totalPositions} positions -> GPUs used={bestK} " +
                        $"distribution=[{string.Join(", ", distribution.Slice(0, numGPUs).ToArray())}] " +
                        $"makespan={makespan:F1}ms penalty={penalty:F1}ms total={makespan + penalty:F1}ms");
    }

    StoreDistCache(totalPositions, numGPUs, distribution);
    return true;
  }


  private void StoreDistCache(int totalPositions, int numGPUs, Span<int> distribution)
  {
    int cBase = totalPositions * pools.Count;
    for (int g = 0; g < numGPUs; g++)
    {
      distCache[cBase + g] = distribution[g];
    }
    distCacheValid[totalPositions] = true;
  }


  /// <summary>
  /// Fraction-based distribution fallback using cached speed-normalized fractions.
  /// </summary>
  private bool TryFractionBasedDistribution(int totalPositions, int numGPUs, Span<int> distribution)
  {
    if (cachedSpeedFractions == null || numGPUs <= 0 || numGPUs >= cachedSpeedFractions.Length)
    {
      return false;
    }

    float[] fractions = cachedSpeedFractions[numGPUs];
    if (fractions == null)
    {
      return false;
    }

    int remaining = totalPositions;
    for (int gpu = 0; gpu < numGPUs && remaining > 0; gpu++)
    {
      int positionsForGpu;
      if (gpu == numGPUs - 1)
      {
        positionsForGpu = remaining;
      }
      else
      {
        positionsForGpu = (int)Math.Round(fractions[gpu] * totalPositions);
        positionsForGpu = Math.Min(positionsForGpu, remaining);
        if (positionsForGpu < minBatchSizePerGPU && remaining > minBatchSizePerGPU)
        {
          positionsForGpu = minBatchSizePerGPU;
        }
      }
      distribution[gpu] = positionsForGpu;
      remaining -= positionsForGpu;
    }

    return true;
  }


  /// <summary>
  /// Get description of the pool configuration.
  /// </summary>
  public string GetDescription()
  {
    return $"MultiGPU[{string.Join(",", deviceIDs)}] mode={mode} minPerGPU={minBatchSizePerGPU}";
  }


  /// <summary>
  /// Dispose the pool and all engine pools.
  /// </summary>
  public void Dispose()
  {
    if (disposed)
    {
      return;
    }
    disposed = true;

    if (pinnedInput != IntPtr.Zero)
    {
      TensorRTNative.FreePinned(pinnedInput);
    }
    if (pinnedOutput != IntPtr.Zero)
    {
      TensorRTNative.FreePinned(pinnedOutput);
    }

    foreach (EnginePool pool in pools)
    {
      pool.Dispose();
    }
  }
}
