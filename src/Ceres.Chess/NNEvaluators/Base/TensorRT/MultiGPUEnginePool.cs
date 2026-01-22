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
using System.Threading.Tasks;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Multi-GPU engine pool that distributes work across multiple GPUs.
/// </summary>
public sealed class MultiGPUEnginePool : IDisposable
{
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
  /// Execution log for debugging multi-GPU distribution.
  /// </summary>
  public List<string> ExecutionLog { get; } = new();

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
  /// Constructor.
  /// </summary>
  public MultiGPUEnginePool(TensorRT trt, string onnxPath, int[] sizes, EnginePoolMode mode,
                             TensorRTBuildOptions options, int inputElementsPerPos, int outputElementsPerPos,
                             int[] deviceIds, int minBatchSizePerGPU = 8, string cacheDir = "/tmp/tensorrt_cache")
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
    ExecutionLog.Clear();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].Process(input, output, totalPositions);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    int baseSize = totalPositions / numGPUs;
    int remainder = totalPositions % numGPUs;

    int[] starts = new int[numGPUs];
    int[] counts = new int[numGPUs];
    int offset = 0;
    for (int i = 0; i < numGPUs; i++)
    {
      starts[i] = offset;
      counts[i] = baseSize + (i < remainder ? 1 : 0);
      offset += counts[i];
    }

    HashSet<int> uniqueDevices = new();
    foreach (int deviceId in deviceIDs)
    {
      if (uniqueDevices.Add(deviceId))
      {
        TensorRTNative.SynchronizeDevice(deviceId);
      }
    }

    Half[][] subInputs = new Half[numGPUs][];
    Half[][] subOutputs = new Half[numGPUs][];
    for (int i = 0; i < numGPUs; i++)
    {
      subInputs[i] = new Half[counts[i] * InputElementsPerPosition];
      subOutputs[i] = new Half[counts[i] * OutputElementsPerPosition];
      Array.Copy(input, starts[i] * InputElementsPerPosition, subInputs[i], 0, subInputs[i].Length);
    }

    string[] logs = new string[numGPUs];
    Parallel.For(0, numGPUs, i =>
    {
      pools[i].Process(subInputs[i], subOutputs[i], counts[i]);
      logs[i] = $"GPU{deviceIDs[i]}({counts[i]})";
    });

    foreach (int deviceId in uniqueDevices)
    {
      TensorRTNative.SynchronizeDevice(deviceId);
    }

    for (int i = 0; i < numGPUs; i++)
    {
      Array.Copy(subOutputs[i], 0, output, starts[i] * OutputElementsPerPosition, subOutputs[i].Length);
      ExecutionLog.Add(logs[i]);
    }
  }


  /// <summary>
  /// Process batch with byte inputs.
  /// </summary>
  public void ProcessBytes(byte[] input, Half[] output, int totalPositions)
  {
    ExecutionLog.Clear();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].ProcessBytes(input, output, totalPositions);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    int baseSize = totalPositions / numGPUs;
    int remainder = totalPositions % numGPUs;

    int[] starts = new int[numGPUs];
    int[] counts = new int[numGPUs];
    int offset = 0;
    for (int i = 0; i < numGPUs; i++)
    {
      starts[i] = offset;
      counts[i] = baseSize + (i < remainder ? 1 : 0);
      offset += counts[i];
    }

    HashSet<int> uniqueDevices = new();
    foreach (int deviceId in deviceIDs)
    {
      if (uniqueDevices.Add(deviceId))
      {
        TensorRTNative.SynchronizeDevice(deviceId);
      }
    }

    byte[][] subInputs = new byte[numGPUs][];
    Half[][] subOutputs = new Half[numGPUs][];
    for (int i = 0; i < numGPUs; i++)
    {
      subInputs[i] = new byte[counts[i] * InputElementsPerPosition];
      subOutputs[i] = new Half[counts[i] * OutputElementsPerPosition];
      Array.Copy(input, starts[i] * InputElementsPerPosition, subInputs[i], 0, subInputs[i].Length);
    }

    string[] logs = new string[numGPUs];
    Parallel.For(0, numGPUs, i =>
    {
      pools[i].ProcessBytes(subInputs[i], subOutputs[i], counts[i]);
      logs[i] = $"GPU{deviceIDs[i]}({counts[i]})";
    });

    foreach (int deviceId in uniqueDevices)
    {
      TensorRTNative.SynchronizeDevice(deviceId);
    }

    for (int i = 0; i < numGPUs; i++)
    {
      Array.Copy(subOutputs[i], 0, output, starts[i] * OutputElementsPerPosition, subOutputs[i].Length);
      ExecutionLog.Add(logs[i]);
    }
  }


  /// <summary>
  /// Process with callback for tensor-major output extraction.
  /// Thread-safe: handler may be called concurrently from multiple GPU threads.
  /// </summary>
  public void ProcessWithHandler(Half[] input, int totalPositions, SubBatchOutputHandler handler)
  {
    ExecutionLog.Clear();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].ProcessWithHandler(input, totalPositions, handler, globalPositionOffset: 0);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    int baseSize = totalPositions / numGPUs;
    int remainder = totalPositions % numGPUs;

    int[] starts = new int[numGPUs];
    int[] counts = new int[numGPUs];
    int offset = 0;
    for (int i = 0; i < numGPUs; i++)
    {
      starts[i] = offset;
      counts[i] = baseSize + (i < remainder ? 1 : 0);
      offset += counts[i];
    }

    HashSet<int> uniqueDevices = new();
    foreach (int deviceId in deviceIDs)
    {
      if (uniqueDevices.Add(deviceId))
      {
        TensorRTNative.SynchronizeDevice(deviceId);
      }
    }

    Half[][] subInputs = new Half[numGPUs][];
    for (int i = 0; i < numGPUs; i++)
    {
      subInputs[i] = new Half[counts[i] * InputElementsPerPosition];
      Array.Copy(input, starts[i] * InputElementsPerPosition, subInputs[i], 0, subInputs[i].Length);
    }

    string[] logs = new string[numGPUs];
    object handlerLock = new object();
    Parallel.For(0, numGPUs, i =>
    {
      SubBatchOutputHandler wrappedHandler = (globalStart, count, engineBatchSize, rawOutput) =>
      {
        int trueGlobalStart = starts[i] + globalStart;
        lock (handlerLock)
        {
          handler(trueGlobalStart, count, engineBatchSize, rawOutput);
        }
      };

      pools[i].ProcessWithHandler(subInputs[i], counts[i], wrappedHandler, globalPositionOffset: 0);
      logs[i] = $"GPU{deviceIDs[i]}({counts[i]})";
    });

    foreach (int deviceId in uniqueDevices)
    {
      TensorRTNative.SynchronizeDevice(deviceId);
    }

    for (int i = 0; i < numGPUs; i++)
    {
      ExecutionLog.Add(logs[i]);
    }
  }


  /// <summary>
  /// Process byte inputs with callback for tensor-major output extraction.
  /// Thread-safe: handler may be called concurrently from multiple GPU threads.
  /// </summary>
  public void ProcessBytesWithHandler(byte[] input, int totalPositions, SubBatchOutputHandler handler)
  {
    ExecutionLog.Clear();

    if (ShouldUseSingleGPU(totalPositions))
    {
      pools[0].ProcessBytesWithHandler(input, totalPositions, handler, globalPositionOffset: 0);
      return;
    }

    int numGPUs = Math.Min(pools.Count, totalPositions / minBatchSizePerGPU);
    numGPUs = Math.Max(1, numGPUs);

    int baseSize = totalPositions / numGPUs;
    int remainder = totalPositions % numGPUs;

    int[] starts = new int[numGPUs];
    int[] counts = new int[numGPUs];
    int offset = 0;
    for (int i = 0; i < numGPUs; i++)
    {
      starts[i] = offset;
      counts[i] = baseSize + (i < remainder ? 1 : 0);
      offset += counts[i];
    }

    HashSet<int> uniqueDevices = new();
    foreach (int deviceId in deviceIDs)
    {
      if (uniqueDevices.Add(deviceId))
      {
        TensorRTNative.SynchronizeDevice(deviceId);
      }
    }

    byte[][] subInputs = new byte[numGPUs][];
    for (int i = 0; i < numGPUs; i++)
    {
      subInputs[i] = new byte[counts[i] * InputElementsPerPosition];
      Array.Copy(input, starts[i] * InputElementsPerPosition, subInputs[i], 0, subInputs[i].Length);
    }

    string[] logs = new string[numGPUs];
    object handlerLock = new object();
    Parallel.For(0, numGPUs, i =>
    {
      SubBatchOutputHandler wrappedHandler = (globalStart, count, engineBatchSize, rawOutput) =>
      {
        int trueGlobalStart = starts[i] + globalStart;
        lock (handlerLock)
        {
          handler(trueGlobalStart, count, engineBatchSize, rawOutput);
        }
      };

      pools[i].ProcessBytesWithHandler(subInputs[i], counts[i], wrappedHandler, globalPositionOffset: 0);
      logs[i] = $"GPU{deviceIDs[i]}({counts[i]})";
    });

    foreach (int deviceId in uniqueDevices)
    {
      TensorRTNative.SynchronizeDevice(deviceId);
    }

    for (int i = 0; i < numGPUs; i++)
    {
      ExecutionLog.Add(logs[i]);
    }
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
