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


#endregion


namespace Ceres.Base.OperatingSystem.NVML
{
  /// <summary>
  /// Summary hardware descriptive information 
  /// and operating state for an NVIDIA GPU.
  /// </summary>
  public record NVMLGPUInfo
  {
    /// <summary>
    /// GPU ID number.
    /// </summary>
    public readonly int ID;

    /// <summary>
    /// GPU model name.
    /// </summary>
    public readonly string Name;

    /// <summary>
    /// Compute architecture.
    /// </summary>
    public readonly int Architecture;

    /// <summary>
    /// Compute capability (major version).
    /// </summary>
    public readonly int CapabilityMajor;

    /// <summary>
    /// Compute capability (minor version).
    /// </summary>
    public readonly int CapabilityMinor;

    /// <summary>
    /// Current frequency (megahertz) of SM clock.
    /// </summary>
    public readonly int ClocksSMMhz;

    /// <summary>
    /// Curernt percentage utilization of GPU.
    /// </summary>
    public readonly int GPUUtilizationPct;

    /// <summary>
    /// Curernt power usage of GPU (in watts).
    /// </summary>
    public readonly float GPUPowerUsage;

    /// <summary>
    /// Current percentage utilization of memory.
    /// </summary>
    public readonly int MemoryUtilizationPct;

    /// <summary>
    /// Current temperature (in degrees Centigrade).
    /// </summary>
    public readonly int TemperatureCentigrade;

    /// <summary>
    /// Zero or more reason why the GPU is currently throttled.
    /// </summary>
    public readonly NVMLClocksThrottleReasons ClocksThrottleReasons;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="id">GPU id number</param>
    /// <param name="name">GPU model name</param>
    /// <param name="architecture">GPU model name</param>
    /// <param name="capabiiltyMajor">compute capability(major)</param>
    /// <param name="capabilityMinor">compute capability (minor)</param>
    /// <param name="clocksSMMhz">current operating frequencey</param>
    /// <param name="gpuUtilizationPct">current GPU utilization</param>
    /// <param name="memoryUtilizationPct">current memory utilization</param>
    /// <param name="gpuPowerUsage">current GPU utilization</param>
    /// <param name="tempreatureCentrigrade">current temperature</param>
    /// <param name="clocksThrottleReasons">reasons for current GPU throttling</param>
    public NVMLGPUInfo(int id, string name, int architecture,
                       int capabiiltyMajor, int capabilityMinor, int clocksSMMhz,
                       int gpuUtilizationPct, int memoryUtilizationPct, float gpuPowerUsage,
                       int tempreatureCentrigrade, NVMLClocksThrottleReasons clocksThrottleReasons)
    {
      ID = id;
      Name = name;
      Architecture = architecture;
      CapabilityMajor = capabiiltyMajor;
      CapabilityMinor = capabilityMinor;
      ClocksSMMhz = clocksSMMhz;
      GPUUtilizationPct = gpuUtilizationPct;
      MemoryUtilizationPct = memoryUtilizationPct;
      GPUPowerUsage = gpuPowerUsage;
      TemperatureCentigrade = tempreatureCentrigrade;
      ClocksThrottleReasons = clocksThrottleReasons;
    }
  }
}
