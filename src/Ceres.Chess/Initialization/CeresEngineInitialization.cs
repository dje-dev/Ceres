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

    // For more consistent performance testing, don't enable foreground priority boost.
    DisableForegroundPriorityBoost();

    // Run with the concurrent (background) GC latency mode. Interactive keeps full (gen2)
    // collections concurrent - off the search's critical path - instead of blocking / stop-the-world.
    GCSettings.LatencyMode = GCLatencyMode.Interactive;

    // NUMA affinitiy disabled, problematic to attempt to
    // derive best value and balance across possibly multiple instances.
    //HardwareManager.AffinitizeSingleNUMANode(numaNode);

    if (launchMonitor
     && !Console.IsOutputRedirected  // launched as a subprocess, e.g. in a touranment, suppress
     && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))     
    {
      EventSourceCeres.ENABLED = true;
      EventSourceCeres.LaunchConsoleMonitor(eventSourceName);
    }
  }
}
