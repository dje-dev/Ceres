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
using System.Runtime;
using System.Runtime.InteropServices;
using System.Threading;
using Ceres.Base.Environment;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.Diagnostics;

#endregion

namespace Ceres.MCGS.Environment;

/// <summary>
/// Manages initialization of the MCGS engine.
/// </summary>
public static class MCGSEngineInitialization
{
  /// <summary>
  /// Lock object to ensure thread-safe initialization.
  /// </summary>
  private static readonly object initializationLock = new();

  /// <summary>
  /// Flag indicating if initialization has been completed.
  /// </summary>
  private static volatile bool isInitialized = false;

  /// <summary>
  /// Performs base initialization of the MCGS engine.
  /// This method is thread-safe and will only perform initialization once.
  /// </summary>
  /// <param name="launchMonitor">If true, launches a performance monitor</param>
  /// <param name="numaNode">NUMA node to use</param>
  public static void BaseInitialize(bool launchMonitor = false, int numaNode = 0)
  {
    // Quick check without lock for performance (double-checked locking pattern)
    if (isInitialized)
    {
      return;
    }

    lock (initializationLock)
    {
      // Check again inside lock to handle race condition
      if (isInitialized)
      {
        return;
      }

      HardwareManager.VerifyHardwareSoftwareCompatability();

      // TODO: consider setting GCSettings.LatencyMode to
      //       SustainedLowLatency when running under tight time constrints

      HardwareManager.Initialize(numaNode);

      if (launchMonitor && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
      {
        // TODO: log this. Console.WriteLine($"dotnet-counters  monitor --process-id {Process.GetCurrentProcess().Id} Ceres System.Runtime Ceres.MCTS.Environment.MCTSEventSource");
        EventSourceCeres.ENABLED = true;
        EventSourceCeres.LaunchConsoleMonitor("Ceres.MCGS.Environment.MCGSEventSource");
      }

      //      MCGSEventSource.Initialize();

      isInitialized = true;
    }
  }

  /// <summary>
  /// Returns true if initialization has been completed.
  /// </summary>
  public static bool IsInitialized => isInitialized;
}
