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
using System.Diagnostics;
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Ceres.Base.OperatingSystem;
using Ceres.Chess.Diagnostics;

#endregion

namespace Ceres.Chess.Initialization;

/// <summary>
/// Engine-agnostic, process-wide low-level initialization shared by all Ceres search engines.
/// </summary>
public static class CeresEngineInitialization
{
  static void DisableForegroundPriorityBoost()
  {
    // Windows-only concept; the property throws PlatformNotSupportedException elsewhere.
    if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
      return;

    using Process me = Process.GetCurrentProcess();
    me.PriorityBoostEnabled = false;   // false = boost DISABLED
  }


  /// <summary>
  /// Performs the common low-level process initialization: hardware/software compatibility
  /// verification, garbage collector latency mode, processor affinity, and (optionally) launch of
  /// the performance monitor.
  ///
  /// Intended to be called once per process; the per-engine BaseInitialize wrappers guard this with
  /// their own one-time flags.
  /// </summary>
  /// <param name="launchMonitor">if true (and on Windows) launches the dotnet-counters style monitor</param>
  /// <param name="eventSourceName">name of the EventSource to monitor when launchMonitor is true</param>
  public static void InitializeBaseProcess(bool launchMonitor, string eventSourceName)
  {
    HardwareManager.VerifyHardwareSoftwareCompatability();

    // The spin count for semaphores is directly configurable.
    // Because most of the Ceres multithreading is not extremely fine grained,
    // the awaited event almost always happens later than the default spinning period.
    // Therefore it is better to use a short spin and save the CPU cycles.
    // This reduces reported CPU time and also reduced wall clock time.
    // (default was 70: AppContextConfigHelper.GetInt32Config("System.Threading.ThreadPool.UnfairSemaphoreSpinLimit", 70, false))
    AppDomain.CurrentDomain.SetData("System.Threading.ThreadPool.UnfairSemaphoreSpinLimit", 2);

    // For more stable performance testing, don't enable foreground priority boost.
    // This is a Windows-only concept and doesn't seem to have significant wall-clock impact.
    DisableForegroundPriorityBoost();

    // Batch mode seems to speed up nps by a few percent.
    GCSettings.LatencyMode = GCLatencyMode.Batch;

    // Affinitize to a single NUMA node to reduce cross-node memory access.
    // In practice we affinitize to use only 32 logical processors
    // which limits typically to a single NUMA node on modern hardware.
    // This yields significant performance improvements on NUMA hardware (up to 10%).
    // Disabled:
    //   (1) proper implementation is tricky; for AMD CCDs one really needs to look
    //       replace NUMA-node with L3-domain-aware pinning (GetLogicalProcessorInformation -> RelationCache level 3).
    //   (2) logical versus physical processor distinctions and layout need to be further considered
    //   (3) this complicates tournaments with multiple engines on the same machine,
    // HardwareManager.AffinitizeSingleNUMANode(0);

    // Possibly launch the .NET logging.
    // Note that this is a legacy style of logging and typically no longer used.
    if (launchMonitor
     && !Console.IsOutputRedirected  // launched as a subprocess, e.g. in a touranment, suppress
     && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))     
    {
      EventSourceCeres.ENABLED = true;
      EventSourceCeres.LaunchConsoleMonitor(eventSourceName);
    }
  }
}
