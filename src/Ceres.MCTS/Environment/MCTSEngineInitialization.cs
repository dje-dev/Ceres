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

      // On .NET 6 the spin count for sempahores is directly configurable.
      // Because most of the Ceres multithreading is not extremely fine grained,
      // the awaited event almost always happens later than the default spinning period.
      // Therefore it is better to use a short spin and save the CPU cycles.
      // This reduces reported CPU time by about 15% to 20% with no slowdown.
      // For .NET 6.0: see WorkerThread.cs, default was 70: AppContextConfigHelper.GetInt32Config("System.Threading.ThreadPool.UnfairSemaphoreSpinLimit", 70, false)
      // (see for how to set this option: https://www.strathweb.com/2019/12/runtime-host-configuration-options-and-appcontext-data-in-net-core/)
      AppDomain.CurrentDomain.SetData("System.Threading.ThreadPool.UnfairSemaphoreSpinLimit", 5);

      // TODO: consider using SustainedLowLatency when running under timed time control
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


