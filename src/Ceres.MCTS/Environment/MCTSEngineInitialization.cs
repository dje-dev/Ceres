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

using Ceres.Chess.Initialization;

#endregion

namespace Ceres.MCTS.Environment
{
  /// <summary>
  /// Manages initialization of the MCTS engine.
  /// </summary>
  public static class MCTSEngineInitialization
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
    /// Performs base initialization of the MCTS engine.
    /// This method is thread-safe and will only perform initialization once.
    /// </summary>
    /// <param name="launchMonitor">If true, launches a performance monitor</param>
    public static void BaseInitialize(bool launchMonitor = false)
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

        // Shared, engine-agnostic process initialization (hardware checks, GC latency mode set
        // explicitly to Interactive/background GC, processor affinity, optional monitor). Centralized
        // so the in-process and external (UCI) launch paths configure the runtime identically.
        CeresEngineInitialization.InitializeBaseProcess(launchMonitor, "Ceres.MCTS.Environment.MCTSEventSource");

        MCTSEventSource.Initialize();

        isInitialized = true;
      }
    }

    /// <summary>
    /// Returns true if initialization has been completed.
    /// </summary>
    public static bool IsInitialized => isInitialized;
  }

}


