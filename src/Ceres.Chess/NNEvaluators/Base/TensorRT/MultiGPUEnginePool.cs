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

  private readonly TensorRT trt;
  private readonly List<EnginePool> pools = new();
  private readonly int[] deviceIDs;
  private readonly int minBatchSizePerGPU;
  private readonly EnginePoolMode mode;
  private readonly int[] sizes;
  private readonly bool useByteInputs;
  private bool disposed;

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

  // Reusable lock object for handler synchronization
  private readonly object handlerLock = new();

  // Execution times (average) by GPU (outer index) and batch size index (inner index)
  // Initialized during Warmup()
  private float[][] executionTimesPerGPU;

  // Device names for each GPU (e.g., "NVIDIA RTX PRO 6000 Blackwell")
  // Used for grouping identical GPUs when computing speed fractions
  private readonly string[] deviceNames;

  // Cached speed-normalized fractions for each GPU count (computed once after warmup)
  // Index is numGPUs, value is the fractions array for that GPU count
  private float[][] cachedSpeedFractions;

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
  public MultiGPUEnginePool(TensorRT trt, string onnxPath, int[] sizes, EnginePoolMode mode,
                             TensorRTBuildOptions options, int inputElementsPerPos, int outputElementsPerPos,
                             int[] deviceIds, int minBatchSizePerGPU, string cacheDir)
  {
    this.trt = trt;
    deviceIDs = deviceIds;
    this.minBatchSizePerGPU = minBatchSizePerGPU;
    this.mode = mode;
    this.sizes = (int[])sizes.Clone();

    foreach (int deviceId in deviceIds)
    {
      EnginePool pool = new EnginePool(trt, onnxPath, (int[])sizes.Clone(), mode, options,
                                        inputElementsPerPos, outputElementsPerPos, deviceId, cacheDir);
      pools.Add(pool);
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
    if (mode == EnginePoolMode.Range)
    {
      return sizes[^1];
    }
    else
    {
      int max = 0;
      foreach (int s in sizes)
      {
        max = Math.Max(max, s);
      }
      return max;
    }
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
    int numBatchSizes = sizes.Length;

    executionTimesPerGPU = new float[numGPUs][];

    for (int gpuIndex = 0; gpuIndex < numGPUs; gpuIndex++)
    {
      executionTimesPerGPU[gpuIndex] = new float[numBatchSizes];
      EnginePool pool = pools[gpuIndex];

      for (int sizeIndex = 0; sizeIndex < numBatchSizes; sizeIndex++)
      {
        int batchSize = sizes[sizeIndex];

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

      // Output timing estimates for this GPU
      string timingsStr = string.Join(", ", executionTimesPerGPU[gpuIndex].Select(t => t.ToString("F1")));
      Console.WriteLine($"DEVICE {deviceIDs[gpuIndex]} timings: [{timingsStr}] ms ({deviceNames[gpuIndex]})");

      // Pass execution times to the EnginePool for optimized scheduling
      pool.ExecutionTimes = executionTimesPerGPU[gpuIndex];
    }

    // Pre-compute speed fractions for all possible GPU counts (1 to numGPUs)
    // This avoids recomputation on every ComputeOptimalDistribution call
    cachedSpeedFractions = new float[numGPUs + 1][];
    for (int n = 1; n <= numGPUs; n++)
    {
      cachedSpeedFractions[n] = BatchScheduler.ComputeSpeedNormalizedFractions(executionTimesPerGPU, deviceNames, n);
    }
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

    if (mode == EnginePoolMode.Exact)
    {
      foreach (int size in sizes)
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
      foreach (int maxSize in sizes)
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

    SynchronizeUniqueDevices();

    // Prepare cached sub-arrays and copy input data
    for (int i = 0; i < numGPUs; i++)
    {
      int inputElements = counts[i] * InputElementsPerPosition;
      int outputElements = counts[i] * OutputElementsPerPosition;
      EnsureHalfArrayCapacity(i, inputElements, outputElements);
      Array.Copy(input, starts[i] * InputElementsPerPosition, cachedHalfInputs[i], 0, inputElements);
    }

    // Copy to local arrays for lambda capture (Span cannot be captured)
    Span<int> startsLocal = stackalloc int[numGPUs];
    Span<int> countsLocal = stackalloc int[numGPUs];
    starts.CopyTo(startsLocal);
    counts.CopyTo(countsLocal);
    int[] startsArray = startsLocal.ToArray();
    int[] countsArray = countsLocal.ToArray();

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

    // Copy to local arrays for lambda capture (Span cannot be captured)
    Span<int> startsLocal = stackalloc int[numGPUs];
    Span<int> countsLocal = stackalloc int[numGPUs];
    starts.CopyTo(startsLocal);
    counts.CopyTo(countsLocal);
    int[] startsArray = startsLocal.ToArray();
    int[] countsArray = countsLocal.ToArray();

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
  /// Thread-safe: handler may be called concurrently from multiple GPU threads.
  /// </summary>
  public void ProcessWithHandler(Half[] input, int totalPositions, SubBatchOutputHandler handler)
  {
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

    SynchronizeUniqueDevices();

    // Prepare cached sub-arrays and copy input data
    for (int i = 0; i < numGPUs; i++)
    {
      int inputElements = counts[i] * InputElementsPerPosition;
      EnsureHalfArrayCapacity(i, inputElements, 0);
      Array.Copy(input, starts[i] * InputElementsPerPosition, cachedHalfInputs[i], 0, inputElements);
    }

    // Copy to local arrays for lambda capture (Span cannot be captured)
    Span<int> startsLocal = stackalloc int[numGPUs];
    Span<int> countsLocal = stackalloc int[numGPUs];
    starts.CopyTo(startsLocal);
    counts.CopyTo(countsLocal);
    int[] startsArray = startsLocal.ToArray();
    int[] countsArray = countsLocal.ToArray();

    Parallel.For(0, numGPUs, i =>
    {
      int capturedStart = startsArray[i];
      SubBatchOutputHandler wrappedHandler = (globalStart, count, engineBatchSize, rawOutput) =>
      {
        int trueGlobalStart = capturedStart + globalStart;
        lock (handlerLock)
        {
          handler(trueGlobalStart, count, engineBatchSize, rawOutput);
        }
      };

      pools[i].ProcessWithHandler(cachedHalfInputs[i], countsArray[i], wrappedHandler, globalPositionOffset: 0);
    });

    SynchronizeTrackedDevices();
  }


  /// <summary>
  /// Process byte inputs with callback for tensor-major output extraction.
  /// Thread-safe: handler may be called concurrently from multiple GPU threads.
  /// </summary>
  public void ProcessBytesWithHandler(byte[] input, int totalPositions, SubBatchOutputHandler handler)
  {
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

    SynchronizeUniqueDevices();

    // Prepare cached sub-arrays and copy input data
    for (int i = 0; i < numGPUs; i++)
    {
      int inputElements = counts[i] * InputElementsPerPosition;
      EnsureByteInputCapacity(i, inputElements);
      Array.Copy(input, starts[i] * InputElementsPerPosition, cachedByteInputs[i], 0, inputElements);
    }

    // Copy to local arrays for lambda capture (Span cannot be captured)
    Span<int> startsLocal = stackalloc int[numGPUs];
    Span<int> countsLocal = stackalloc int[numGPUs];
    starts.CopyTo(startsLocal);
    counts.CopyTo(countsLocal);
    int[] startsArray = startsLocal.ToArray();
    int[] countsArray = countsLocal.ToArray();

    Parallel.For(0, numGPUs, i =>
    {
      int capturedStart = startsArray[i];
      SubBatchOutputHandler wrappedHandler = (globalStart, count, engineBatchSize, rawOutput) =>
      {
        int trueGlobalStart = capturedStart + globalStart;
        lock (handlerLock)
        {
          handler(trueGlobalStart, count, engineBatchSize, rawOutput);
        }
      };

      pools[i].ProcessBytesWithHandler(cachedByteInputs[i], countsArray[i], wrappedHandler, globalPositionOffset: 0);
    });

    SynchronizeTrackedDevices();
  }


  /// <summary>
  /// Computes optimal position distribution across GPUs based on their measured performance.
  /// Uses cached speed-normalized fractions (computed once during warmup).
  /// Writes directly to the provided Span to avoid allocation.
  /// </summary>
  /// <param name="totalPositions">Total positions to distribute</param>
  /// <param name="numGPUs">Number of GPUs to use (may be less than pools.Count for small batches)</param>
  /// <param name="distribution">Span to write position counts per GPU</param>
  /// <returns>True if optimal distribution was computed, false to use default equal distribution</returns>
  private bool TryComputeOptimalDistribution(int totalPositions, int numGPUs, Span<int> distribution)
  {
    // Only use optimized scheduling when cached fractions are available
    if (cachedSpeedFractions == null || !EnginePool.OPTIMIZED_SCHEDULING || 
        numGPUs <= 0 || numGPUs >= cachedSpeedFractions.Length)
    {
      return false;
    }

    float[] fractions = cachedSpeedFractions[numGPUs];
    if (fractions == null)
    {
      return false;
    }

    // Distribute positions based on cached fractions
    int remaining = totalPositions;

    for (int gpu = 0; gpu < numGPUs && remaining > 0; gpu++)
    {
      int positionsForGpu;

      if (gpu == numGPUs - 1)
      {
        // Last GPU gets remainder to avoid rounding issues
        positionsForGpu = remaining;
      }
      else
      {
        positionsForGpu = (int)Math.Round(fractions[gpu] * totalPositions);
        positionsForGpu = Math.Min(positionsForGpu, remaining);

        // Enforce minimum batch size per GPU (except for last GPU which gets remainder)
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
