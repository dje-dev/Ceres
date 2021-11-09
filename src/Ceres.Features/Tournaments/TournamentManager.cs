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
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;

using Ceres.Chess.UserSettings;
using Ceres.Features.GameEngines;
using Ceres.Chess;
using Ceres.Features.Players;
using System.Threading;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manages execution of a tournament (match) of games
  /// of one engine against another, possibly multithreaded.
  /// </summary>
  public partial class TournamentManager
  {
    /// <summary>
    /// Number of touranment games to play in parallel.
    /// </summary>
    public readonly int NumConcurrent = 1;

    /// <summary>
    /// Definition of parameters of the tournament.
    /// </summary>
    public TournamentDef Def;


    /// <summary>
    /// Optionally a queue manager object which is used when running
    /// distributed model (running either as a coordinator or worker process).
    /// </summary>
    public TournamentGameQueueManager QueueManager;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="def"></param>
    /// <param name="numParallel"></param>
    /// <param name="separateGPUs"></param>
    public TournamentManager(TournamentDef def, int numParallel = 1)
    {
      Def = def;
      NumConcurrent = Math.Min(numParallel, def.NumGamePairs ?? int.MaxValue);

      // Turn off showing game moves if running parallel,
      // since the moves of various games would be intermixed.
      if (numParallel > 1)
      {
        def.ShowGameMoves = false;
      }

#if NOT
      if (def.EvaluatorDef2.DeviceCombo == NNEvaluators.Defs.NNEvaluatorDeviceComboType.Pooled)
      {
        // If already pooled then having separate GPUs does not make sense
        SeparateGPUs = false;
      }
#endif
    }

    int numGamePairsLaunched = 0;
    readonly object lockObj = new();


    /// <summary>
    /// Method called by threads to get the next available game to be played.
    /// </summary>
    /// <returns></returns>
    int GetNextOpeningIndexForLocalThread(int maxOpenings)
    {
      if (Def.RandomizeOpenings) throw new NotImplementedException();

      lock (lockObj)
      {
        if (numGamePairsLaunched < maxOpenings)
        {
          return numGamePairsLaunched++;
        }
        else
          return -1;
      }
    }


    /// <summary>
    /// If running multiple match threads then we need to make sure the 
    /// neural network evaluators are suitable for parallel execution, either by:
    ///   - being of type Pooled, so that all evaluator definitions reference
    ///     the same underlying evaluator which will pool positions from many
    ///     threads into one batch, or
    ///   - each thread being assigned to a different GPU
    ///   
    /// This method handles the latter case, if the evaluators are not pooled
    /// then it will sequentially apply them to GPUs with successive indices.
    /// 
    /// TODO: detect error condition where the number of threads exceeds number of GPUs.
    /// </summary>
    /// <param name="def"></param>
    /// <param name="relativeDeviceIndex"></param>
    static void TrySetRelativeDeviceIDIfNotPooled(TournamentDef def, int relativeDeviceIndex)
    {
      if (def.Engines != null)
      {
        foreach (EnginePlayerDef engine in def.Engines)
        {
          engine.EngineDef.ModifyDeviceIndexIfNotPooled(relativeDeviceIndex);
        }
      }
      else
      {
        def.Player1Def.EngineDef.ModifyDeviceIndexIfNotPooled(relativeDeviceIndex);
        def.Player2Def.EngineDef.ModifyDeviceIndexIfNotPooled(relativeDeviceIndex);
      }
    }

    void VerifyEnginesCompatible()
    {
      if (Def.Engines != null)
      {
        foreach (EnginePlayerDef engine in Def.Engines)
        {
          if (engine.SearchLimit.Type == SearchLimitType.NodesForAllMoves
              && !engine.EngineDef.SupportsNodesPerGameMode)
          {
            throw new Exception($"Requested NodesPerGame mode is not supported by engine: {engine.EngineDef.ID}");
          }
        }
      }

      else
      {
        if (Def.Player1Def.SearchLimit.Type == SearchLimitType.NodesForAllMoves
         && !Def.Player1Def.EngineDef.SupportsNodesPerGameMode)
        {
          throw new Exception($"Requested NodesPerGame mode is not supported by engine 1: {Def.Player1Def.EngineDef.ID}");
        }

        if (Def.Player2Def.SearchLimit.Type == SearchLimitType.NodesForAllMoves
         && !Def.Player2Def.EngineDef.SupportsNodesPerGameMode)
        {
          throw new Exception($"Requested NodesPerGame mode is not supported by engine 2: {Def.Player2Def.EngineDef.ID}");
        }
      }
    }


    internal ManualResetEvent shutdownComplete = new (false);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="queueManager">optional associated queue manager if running in distributed mode</param>
    /// <returns></returns>
    public TournamentResultStats RunTournament(TournamentGameQueueManager queueManager = null)
    {
      shutdownComplete.Reset();
      Def.ShouldShutDown = false;
        
      if (Def.Engines.Length > 0)
      {
        foreach (EnginePlayerDef engine in Def.Engines)
        {
          if (engine == null) throw new ArgumentNullException("engine is null)");
        }
      }
      else
      {
        if (Def.Player1Def == null) throw new ArgumentNullException("Def.Player1Def is null)");
        if (Def.Player2Def == null) throw new ArgumentNullException("Def.Player2Def is null)");
      }


      VerifyEnginesCompatible();

      // Install Ctrl-C handler to allow ad hoc clean termination of tournament (with stats).
      ConsoleCancelEventHandler ctrlCHandler = new ConsoleCancelEventHandler((object sender, 
        ConsoleCancelEventArgs args) => 
        {
          Console.WriteLine("Tournament pending shutdown....");
          Def.parentDef.ShouldShutDown = true;
          shutdownComplete.WaitOne();
        }); ;
      Console.CancelKeyPress += ctrlCHandler;

      QueueManager = queueManager;
      TournamentResultStats parentTest = new();


      // Show differences between engine 1 and engine 2
      Def.DumpParams();

      // Prepare to create a set of task threads to run games
      List<Task> tasks = new List<Task>();
      List<TournamentGameThread> gameThreads = new List<TournamentGameThread>();

      Directory.CreateDirectory(CeresUserSettingsManager.Settings.DirCeresOutput);

      // If we are a worker process then run only a single game thread.
      int numConcurrent = queueManager != null && !queueManager.IsCoordinator ? 1 : NumConcurrent;

      for (int i = 0; i < numConcurrent; i++)
      {
        TournamentDef tournamentDefClone = Def.Clone();

        // Make sure the threads will use either different or pooled evaluators
        if (NumConcurrent > 1)
        {
          TrySetRelativeDeviceIDIfNotPooled(tournamentDefClone, i);
        }

        //if (Def.Player1Def is GameEngineDefCeres
        // && Def.Player2Def is GameEngineDefCeres
        // && Def.Player1Def.SearchLimit.IsNodesLimit
        // && Def.Player2Def.SearchLimit.IsNodesLimit)
        //{
        //    GameEngineDefCeres thisDefCeres1 = Def.Player1Def.EngineDef as GameEngineDefCeres;
        //    GameEngineDefCeres thisDefCeres2 = Def.Player2Def.EngineDef as GameEngineDefCeres;

        //    // TODO: possibly add optimization here which will share trees
        //    //       with ReusePositionEvaluationsFromOtherTree.
        //    //       See the suite manager for an example of how this is done.
        //}

        TournamentGameThread gameTest = new TournamentGameThread(tournamentDefClone, parentTest);
        gameThreads.Add(gameTest);

        Action action;
        if (QueueManager == null)
        {
          // Everything happens locally, data structures updated as part of processing.
          action = () => ThreadProcLocalWorker(i, gameTest);
        }
        else if (QueueManager.IsCoordinator)
        {
          // We are coordinator. Repeatedly enqueue request to play game pairs and retireve/show results.
          action = () => ThreadProcCoordinator(i, gameTest);
        }
        else
        {
          // Worker method (which will forward result data structure back to coordinator).
          action = () => ThreadProcDistributedWorker(i, gameTest);
        }


        Task thisTask = new Task(action);
        tasks.Add(thisTask);
        thisTask.Start();
      }
      Task.WaitAll(tasks.ToArray());

      parentTest.DumpTournamentSummary(Def.Logger, Def.ReferenceEngineId);

      shutdownComplete.Set();

      return parentTest;
    }

  }
}
