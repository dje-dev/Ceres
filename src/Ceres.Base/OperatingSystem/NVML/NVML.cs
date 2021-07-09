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
using System.Data;
using System.Text;
using Ceres.Base.Misc;

#endregion

namespace Ceres.Base.OperatingSystem.NVML
{
  /// <summary>
  /// Accesses information about installed NVIDIA GPUs via the NVML library.
  /// </summary>
  public static class NVML
  {
    /// <summary>
    /// Returns the number of GPU devices in the system.
    /// </summary>
    public static int DeviceCount
    {
      get
      {
        CheckInitialized();
        return (int)deviceCount;
      }
    }

    /// <summary>
    /// Returns information about each of the GPU devices in the system.
    /// </summary>
    /// <returns></returns>
    public static List<NVMLGPUInfo> GetGPUsInfo()
    {
      CheckInitialized();

      List<NVMLGPUInfo> infos = new List<NVMLGPUInfo>();

      for (uint i = 0; i < deviceCount; i++)
      {
        infos.Add(GetGPUInfo(i));
      }

      return infos;
    }

    public static NVMLGPUInfo GetGPUInfo(uint i)
    {
      uint ret = NVMLMethods.nvmlDeviceGetHandleByIndex(i, out IntPtr device);

      StringBuilder name = new StringBuilder(50);
      NVMLMethods.nvmlDeviceGetName(device, name, 50);
      NVMLMethods.nvmlDeviceGetTemperature(device, 0, out uint temperatureCentigrade);
      NVMLMethods.nvmlDeviceGetUtilizationRates(device, out NVMLUtilization utilization);

      NVMLMethods.nvmlDeviceGetPowerUsage(device, out uint powerUsage);
      NVMLMethods.nvmlDeviceGetArchitecture(device, out uint architecture);

      NVMLClocksThrottleReasons clocksThrottleReasons = 0;
      NVMLMethods.nvmlDeviceGetCurrentClocksThrottleReasons(device, ref clocksThrottleReasons);

      uint major = 0;
      uint minor = 0;
      NVMLMethods.nvmlDeviceGetCudaComputeCapability(device, ref major, ref minor);

      uint clocksSM = 0;
      NVMLMethods.nvmlDeviceGetClockInfo(device, nvmlClockType.Graphics, ref clocksSM);

      return new NVMLGPUInfo((int)i, name.ToString(),
                             (int)major, (int)minor, (int)architecture, (int)clocksSM,
                             (int)utilization.UtilizationGPUPct, (int)utilization.UtilizationMemoryPct,
                             (float)powerUsage / 1000.0f, (int)temperatureCentigrade, clocksThrottleReasons);
    }

    public const string InfoDescriptionHeaderLine1 = "ID  Name                     Ver  SMClk  GPU%  Mem%   Temp   Throttle Reasons";
    public const string InfoDescriptionHeaderLine2 = "--  -----------------------  ---  -----  ----  ----   ----   ----------------";


    /// <summary>
    /// Dumps information about each GPU to the Console.
    /// </summary>
    public static void DumpInfo()
    {
      Console.WriteLine(InfoDescriptionHeaderLine1);
      Console.WriteLine(InfoDescriptionHeaderLine2);

      foreach (NVMLGPUInfo info in GetGPUsInfo())
      {
        Console.WriteLine(GetInfoDescriptionLine(info));
      }
    }


    public static string GetInfoDescriptionLine(NVMLGPUInfo info)
    {
      StringBuilder desc = new StringBuilder();

      desc.Append($"{info.ID,2:F0}  {info.Name,-20}     {info.CapabilityMajor,1}{info.CapabilityMinor,1}  ");
      desc.Append($"{info.ClocksSMMhz,5}  {info.GPUUtilizationPct,3}%  {info.MemoryUtilizationPct,3}%  ");
      desc.Append($"{info.TemperatureCentigrade,4}C   {StringUtils.Sized(info.ClocksThrottleReasons.ToString(), 14)}");

      return desc.ToString();
    }


    public static void Shutdown()
    {
      NVMLMethods.nvmlShutdown();
      haveInitialized = false;
    }

    #region Internal helpers

    static uint deviceCount;
    static bool haveInitialized;

    static void CheckInitialized()
    {
      if (!haveInitialized)
      {
        NVMLMethods.nvmlInit();
        haveInitialized = true;
        NVMLMethods.nvmlDeviceGetCount(out deviceCount);
      }

    }

#endregion
  }
}
