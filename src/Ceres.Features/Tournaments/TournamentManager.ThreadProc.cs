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

#endregion

namespace Ceres.Features.Tournaments
{
  public partial class TournamentManager
  {

    /// <summary>
    /// Main processing loop when running as a local (non-distributed) worker thread.
    /// </summary>
    /// <param name="threadIndex"></param>
    /// <param name="gameTest"></param>
    private void ThreadProcLocalWorker(int threadIndex, TournamentGameThread gameTest)
    {
      try
      {
        gameTest.RunGameTests(threadIndex, () => GetNextOpeningIndexForLocalThread(Def.NumGamePairs ?? int.MaxValue));
      }
      catch (Exception exc)
      {
        Console.WriteLine("Exception in TournamentManager thread: " + exc.ToString());
      }
    }

    /// <summary>
    /// Main processing loop when running as the coordinator for a distributed tournament.
    /// </summary>
    /// <param name="threadIndex"></param>
    /// <param name="gameTest"></param>
    private void ThreadProcCoordinator(int threadIndex, TournamentGameThread gameTest)
    {
      while (true)
      {
        try
        {
          int openingIndex = GetNextOpeningIndexForLocalThread(Def.NumGamePairs ?? int.MaxValue);
          if (openingIndex == -1)
          {
            return;
          }

          // Post this game request for some worker to accept and process.
          QueueManager.EnqueueOpeningRequest(openingIndex);

          // Retrieve some result from a worker.
          // Note that this retrieved result is not necessarily for this just-enqueued game request.
          (TournamentGameInfo gameInfo, TournamentGameInfo gameReverseInfo) = QueueManager.GetGamePairResult();

//Console.WriteLine(openingIndex + " retrieved game pair result " + gameInfo.OpeningIndex + " " + gameReverseInfo.OpeningIndex + "count was " + gameTest.NumGames);
          // Finally, integrate this into our overall results.
          string pgnFileName = null; // TODO: collect these someday as well?
          gameTest.UpdateStatsAndOutputSummaryFromGameResult(pgnFileName, gameInfo.Engine2IsWhite, gameInfo.OpeningIndex, gameInfo.GameSequenceNum, gameInfo);
          gameTest.UpdateStatsAndOutputSummaryFromGameResult(pgnFileName, gameReverseInfo.Engine2IsWhite, gameReverseInfo.OpeningIndex, gameReverseInfo.GameSequenceNum, gameReverseInfo);
          
//Console.WriteLine("Count now " + gameTest.NumGames);
        }
        catch (Exception exc)
        {
          //Console.WriteLine("Exception in TournamentManager thread: " + exc.ToString());
        }
      }
    }


    /// <summary>
    /// Main processing loop when running as a distributed worker.
    /// </summary>
    /// <param name="threadIndex"></param>
    /// <param name="gameTest"></param>
    private void ThreadProcDistributedWorker(int threadIndex, TournamentGameThread gameTest)
    {
      try
      {
        gameTest.RunGameTests(threadIndex, QueueManager.GetEnqueuedOpeningRequest,
          delegate (TournamentGameInfo gameInfo, TournamentGameInfo gameReverse)
          {
          // In this postprocessing step we post the games result info
          // so the coordinator can retrieve and integrate into the full tournament.
          QueueManager.WriteGamePairResult(gameInfo, gameReverse);
          });
      }
      catch (Exception exc)
      {
        Console.WriteLine("Exception in TournamentManager thread: " + exc.ToString());
      }
    }


  }

}