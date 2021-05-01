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
using Ceres.Base.Environment;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.Diagnostics;

#endregion

namespace Ceres.MCTS.Environment
{
  /// <summary>
  /// Manages initialization of the MCTS engine.
  /// </summary>
  public static class MCTSEngineInitialization
  {
    public static void BaseInitialize(bool launchMonitor, int? numaNode)
    {
      HardwareManager.VerifyHardwareSoftwareCompatability();

      int minNumThreads = Math.Min(96, System.Environment.ProcessorCount * 4);
      System.Threading.ThreadPool.SetMinThreads(minNumThreads, 32);
      GCSettings.LatencyMode = GCLatencyMode.Batch;

      HardwareManager.Initialize(numaNode);

      if (launchMonitor && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
      {
        // TODO: log this. Console.WriteLine($"dotnet-counters  monitor --process-id {Process.GetCurrentProcess().Id} Ceres System.Runtime Ceres.MCTS.Environment.MCTSEventSource");
        EventSourceCeres.ENABLED = true;
        EventSourceCeres.LaunchConsoleMonitor("Ceres.MCTS.Environment.MCTSEventSource");
      }

      MCTSEventSource.Initialize();
    }

  }

}


