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
using System.Runtime.InteropServices;
using System.Text;

#endregion

namespace Ceres.Base.OperatingSystem.NVML
{
  /// <summary>
  /// NVIDIA Management Library wrapper.
  /// </summary>
  public static class NVMLMethods
  {
    public const string NVML_LIB_NAME = "nvml";

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlInit();

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlShutdown();

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetCount(out uint deviceCount);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetName(IntPtr device, [MarshalAs(UnmanagedType.LPStr)] StringBuilder name, uint length);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetHandleByIndex(uint index, out IntPtr device);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetTemperature(IntPtr device, uint sensorType, out uint temp);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetUtilizationRates(IntPtr device, out NVMLUtilization utilization);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetPowerUsage(IntPtr device, out uint powerUsage);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetArchitecture(IntPtr device, out uint architecture);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetClockInfo(IntPtr device, nvmlClockType type, ref uint clock);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetCudaComputeCapability(IntPtr device, ref uint major, ref uint minor);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetTemperatureThreshold(IntPtr device, nvmlTemperatureThresholds thresholdType, ref uint temp);

    [DllImport(NVML_LIB_NAME)]
    public static extern uint nvmlDeviceGetCurrentClocksThrottleReasons(IntPtr device, ref NVMLClocksThrottleReasons clocksThrottleReasons);

    #region Resolve import

    internal const string NVML_LINUX = "libnvidia-ml.so";
    internal const string NVML_WINDOWS = "nvml.dll";

    static NVMLMethods()
    {
      NativeLibrary.SetDllImportResolver(typeof(NVMLMethods).Assembly, ImportResolver);
    }

    private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
      if (libraryName == NVML_LIB_NAME)
      {
        string libName = RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                          ? NVML_LINUX
                          : NVML_WINDOWS;
        NativeLibrary.TryLoad(libName, assembly, DllImportSearchPath.SafeDirectories, out IntPtr libHandle);
        return libHandle;
      }
      else
      {
        return default;
      }
    }

    #endregion

  }

  public struct NVMLUtilization
  {
    public uint UtilizationGPUPct;
    public uint UtilizationMemoryPct;
  }

  /// <summary>
  /// Temperature thresholds.
  /// </summary>
  public enum nvmlTemperatureThresholds
  {
    /// <summary>
    /// Temperature at which the GPU will shut down for HW protection
    /// </summary>
    Shutdown = 0,
    /// <summary>
    /// Temperature at which the GPU will begin slowdown
    /// </summary>
    Slowdown = 1
  }

  public enum nvmlClockType
  {
    Graphics = 0,
    SM = 1,
    Mem = 2,
    Video = 3
  }

  [Flags]
  public enum NVMLClocksThrottleReasons : ulong
  {
    GpuIdle = 0x0000000000000001L,
    ApplicationsClocksSetting = 0x0000000000000002L,
    SwPowerCap = 0x0000000000000004L,
    HwSlowdown = 0x0000000000000008L,
    SyncBoost = 0x0000000000000010L,

    DisplayClockSetting = 0x0000000000000100L,
    HwPowerBrakeSlowdown = 0x0000000000000080L,
    HwThermalSlowdown = 0x0000000000000040L,
    SwThermalSlowdown = 0x0000000000000020L,

    Unknown = 0x8000000000000000L,
    None = 0x0000000000000000L
  }


}