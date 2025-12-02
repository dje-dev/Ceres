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
using System.Runtime.InteropServices;
using System.Threading;

using Ceres.Base.OperatingSystem.NVML;

#endregion

namespace Ceres.Base.CUDA;

/// <summary>
/// Immutable statistics collected by MetricCUDAUtilization.
/// </summary>
public readonly record struct CUDAUtilizationStats
(
    double MinGpuUtilPct,
    double MaxGpuUtilPct,
    double AvgGpuUtilPct,
    double MedianGpuUtilPct,

    double MinMemUtilPct,
    double MaxMemUtilPct,
    double AvgMemUtilPct,
    double MedianMemUtilPct,

    double MinTempC,
    double MaxTempC,
    double AvgTempC,
    double MedianTempC
);


/// <summary>
/// Class to manage periodically sampling NVML for 
/// GPU utilization, temperature, and memory-usage information.
/// </summary>
public sealed class CUDAUtilizationMetrics : IDisposable
{
  const int SAMPLING_INTERVAL_MS = 100; // NVML calls are fast (circa 0.02ms)

  private readonly List<Sample> samples = [];
  private readonly TimeSpan interval;
  private Thread samplingThread;
  private bool collecting;
  private bool nvmlInitialised;
  private IntPtr deviceHandle;


  /// <summary>
  /// A single sample of GPU metrics.
  /// </summary>
  private readonly struct Sample
  {
    public Sample(DateTime timestamp, uint gpuUtilPct, uint memUtilPct, uint tempC)
    {
      Timestamp = timestamp;
      GpuUtilPct = gpuUtilPct;
      MemUtilPct = memUtilPct;
      TempC = tempC;
    }


    public DateTime Timestamp { get; }
    public uint GpuUtilPct { get; }
    public uint MemUtilPct { get; }
    public uint TempC { get; }
  }


  /// <summary>
  /// Creates a new sampler.
  /// </summary>
  /// <param name="interval">Sampling cadence; defaults to 500 ms.</param>
  public CUDAUtilizationMetrics(TimeSpan? interval = null)
  {
    this.interval = interval ?? TimeSpan.FromMilliseconds(SAMPLING_INTERVAL_MS);
  }


  /// <summary>
  /// Begins sampling on the first NVML device (index 0).
  /// </summary>
  public void Start()
  {
    if (collecting)
    {
      throw new InvalidOperationException("Sampling is already running.");
    }

    uint result = NVMLMethods.nvmlInit();
    if (result != 0)
    {
      throw new InvalidOperationException($"nvmlInit failed with error {result}.");
    }

    nvmlInitialised = true;

    result = NVMLMethods.nvmlDeviceGetCount(out uint deviceCount);
    if (result != 0 || deviceCount == 0)
    {
      throw new InvalidOperationException("No NVIDIA GPU detected or unable to enumerate devices.");
    }

    result = NVMLMethods.nvmlDeviceGetHandleByIndex(0u, out deviceHandle);
    if (result != 0)
    {
      throw new InvalidOperationException($"nvmlDeviceGetHandleByIndex failed with error {result}.");
    }

    collecting = true;
    samplingThread = new Thread(SamplingLoop)
    {
      IsBackground = true,
      Name = nameof(CUDAUtilizationMetrics) + ":SamplingThread"
    };
    samplingThread.Start();
  }


  /// <summary>
  /// Stops sampling and blocks until all pending samples are processed.
  /// </summary>
  public void Stop()
  {
    if (!collecting)
    {
      return;
    }

    collecting = false;
    samplingThread?.Join();

    if (nvmlInitialised)
    {
      uint nvmlError = NVMLMethods.nvmlShutdown();
      nvmlInitialised = false;
    }
  }


  /// <summary>
  /// Returns immutable snapshot of statistical aggregates 
  /// for the interval between Start and Stop./>.
  /// </summary>
  public CUDAUtilizationStats GetStats()
  {
    if (collecting)
    {
      throw new InvalidOperationException("Call Stop() before retrieving statistics.");
    }

    if (samples.Count == 0)
    {
      return default;
    }

    double[] gpuUtil = [.. samples.Select(s => (double)s.GpuUtilPct)];
    double[] memUtil = [.. samples.Select(s => (double)s.MemUtilPct)];
    double[] tempC = [.. samples.Select(s => (double)s.TempC)];

    return new CUDAUtilizationStats(gpuUtil.Min(), gpuUtil.Max(), gpuUtil.Average(), Median(gpuUtil),
                                    memUtil.Min(), memUtil.Max(), memUtil.Average(), Median(memUtil),
                                    tempC.Min(), tempC.Max(), tempC.Average(), Median(tempC));
  }

  private void SamplingLoop()
  {
    while (collecting)
    {
      CollectOnce();
      Thread.Sleep(interval);
    }
  }


  private void CollectOnce()
  {
    NVMLUtilization utilisation;

    uint result = NVMLMethods.nvmlDeviceGetUtilizationRates(deviceHandle, out utilisation);
    if (result != 0)
    {
      return; // Skip sample on failure.
    }

    result = NVMLMethods.nvmlDeviceGetTemperature(deviceHandle, 0u, out uint tempC);
    if (result != 0)
    {
      tempC = 0u;
    }

    Sample sample = new(DateTime.UtcNow, utilisation.UtilizationGPUPct, utilisation.UtilizationMemoryPct, tempC);
    lock (samples)
    {
      samples.Add(sample);
    }
  }


  private static double Median(IEnumerable<double> sequence)
  {
    double[] sorted = [.. sequence.OrderBy(x => x)];
    int n = sorted.Length;
    if (n == 0)
    {
      return double.NaN;
    }
    if ((n & 1) == 1)
    {
      return sorted[n / 2];
    }
    return (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
  }


  public void Dispose() => Stop();


  #region Native helpers (memory info)

  [DllImport(NVMLMethods.NVML_LIB_NAME)]
  private static extern ulong nvmlDeviceGetMemoryInfo(IntPtr device, out NvmlMemory memory);

  [StructLayout(LayoutKind.Sequential)]
  private struct NvmlMemory
  {
    public ulong Total;
    public ulong Free;
    public ulong Used;
  }

  #endregion


  /// <summary>
  /// Helper struct to enable using declaration with out variable for MetricCUDAUtilization
  /// </summary>
  public record CUDAUtilizationBlock : IDisposable
  {
    private readonly CUDAUtilizationMetrics utilization;
    public CUDAUtilizationBlock(out CUDAUtilizationMetrics utilization)
    {
      this.utilization = new CUDAUtilizationMetrics();
      utilization = this.utilization;
      this.utilization.Start();
    }

    public void Dispose() => utilization?.Stop();
  }
}
