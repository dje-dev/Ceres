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
using System.Diagnostics;

using Ceres.Chess.UserSettings;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Features.GameEngines;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manages execution of a tournament (match) of games
  /// of one engine against another, possibly multithreaded.
  /// </summary>
  public class TournamentManager
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
    /// Constructor.
    /// </summary>
    /// <param name="def"></param>
    /// <param name="numParallel"></param>
    /// <param name="separateGPUs"></param>
    public TournamentManager(TournamentDef def, int numParallel = 1)
    {
      Def = def;
      NumConcurrent = numParallel;

      // Turn off showing game moves if running parallel,
      // since the moves of various games would be intermixed.
      if (numParallel > 1) def.ShowGameMoves = false;

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
    (int gameSequenceNum, int openingIndex) GetNextGameForThread(int maxOpenings)
    {
      if (Def.RandomizeOpenings) throw new NotImplementedException();

      lock (lockObj)
      {
        if (numGamePairsLaunched < maxOpenings)
        {
          int thisOpeningIndex = numGamePairsLaunched++;
          return (thisOpeningIndex * 2, thisOpeningIndex);
        }
        else
          return (-1, -1);
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
      def.Player1Def.EngineDef.ModifyDeviceIndexIfNotPooled(relativeDeviceIndex);
      def.Player2Def.EngineDef.ModifyDeviceIndexIfNotPooled(relativeDeviceIndex);
    }

    public TournamentResultStats RunTournament()
    {
      TournamentResultStats parentTest = new TournamentResultStats(Def.Player1Def.ID, Def.Player2Def.ID);

      // Show differences between engine 1 and engine 2
      Def.DumpParams();

      // Prepare to create a set of task threads to run games
      List<Task> tasks = new List<Task>();
      List<TournamentGameThread> gameThreads = new List<TournamentGameThread>();

      Directory.CreateDirectory(CeresUserSettingsManager.Settings.DirCeresOutput);

      for (int i = 0; i < NumConcurrent; i++)
      {
        TournamentDef tournamentDefClone = Def.Clone();

        // Make sure the threads will use either different or pooled evaluators
        if (NumConcurrent > 1)
        {
          TrySetRelativeDeviceIDIfNotPooled(tournamentDefClone, i);
        }

        if (Def.Player1Def is GameEngineDefCeres 
         && Def.Player2Def is GameEngineDefCeres
         && Def.Player1Def.SearchLimit.IsNodesLimit
         && Def.Player2Def.SearchLimit.IsNodesLimit)
        {
          GameEngineDefCeres thisDefCeres1 = Def.Player1Def.EngineDef as GameEngineDefCeres;
          GameEngineDefCeres thisDefCeres2 = Def.Player2Def.EngineDef as GameEngineDefCeres;

          // TODO: possibly add optimization here which will share trees
          //       with ReusePositionEvaluationsFromOtherTree.
          //       See the suite manager for an example of how this is done.
        }

        TournamentGameThread gameTest = new TournamentGameThread(tournamentDefClone, parentTest);
        gameThreads.Add(gameTest);


#if NOT
        // TODO: need to clean up, figure out how to not exceed the maximum avaliable GPU ID
        int? gpuID = null;
        if (Def.EvaluatorDef2 != null && SeparateGPUs)
        {
          gpuID = i + Def.EvaluatorDef2.Devices[0].Device.DeviceIndex;
        }

        TournamentDef tournamentDefClone = Def.Clone();
        tournamentDefClone.ForceUseGPUIndex = gpuID;
        tournamentDefClone.NumGamePairs = (Def.NumGamePairs / NumParallel)
                                        + ((Def.NumGamePairs % NumParallel > i) ? 1 : 0);

#endif

        //        if (NUM_PARALLEL > new ParamsNN().NNEVAL_NUM_GPUS && limit1.Type != SearchLimit.LimitType.NodesPerMove)
        //          throw new Exception("Running in timed mode yet NUM_PARALLEL > NNEVAL_NUM_GPUS, this will not be fair test.")

        Task thisTask = new Task(() => TournamentManagerThreadMainMethod(i, gameTest));
        tasks.Add(thisTask);
        thisTask.Start();
      }
      Task.WaitAll(tasks.ToArray());

      // Calculate and write summary line
      long totalNodesEngine1 = gameThreads.Sum(g => g.TotalNodesEngine1);
      long totalNodesEngine2 = gameThreads.Sum(g => g.TotalNodesEngine2);
      int totalMovesEngine1 = gameThreads.Sum(g => g.TotalMovesEngine1);
      int totalMovesEngine2 = gameThreads.Sum(g => g.TotalMovesEngine2);
      float totalTimeEngine1 = gameThreads.Sum(g => g.TotalTimeEngine1);
      float totalTimeEngine2 = gameThreads.Sum(g => g.TotalTimeEngine2);
      float numGames = gameThreads.Sum(g => g.NumGames);

      Def.Logger.WriteLine("					          ------  ------    --------------  --------------   ----"); 
      Def.Logger.Write("                                                 ");
      Def.Logger.Write($"{totalTimeEngine1,7:F2} {totalTimeEngine2,7:F2} ");
      Def.Logger.Write($"{totalNodesEngine1,17:N0} {totalNodesEngine2,15:N0}   ");
      Def.Logger.Write($"{totalMovesEngine1 / numGames,4:F0}");

      Def.Logger.WriteLine();
      parentTest.Dump();
      return parentTest;
    }


    private void TournamentManagerThreadMainMethod(int i, TournamentGameThread gameTest)
    {
      try
      {
        gameTest.RunGameTests(i, GetNextGameForThread);
      }
      catch (Exception exc)
      {
        Console.WriteLine("Exception in TournamentManager thread: " + exc.ToString());
      }
    }

  }
}
