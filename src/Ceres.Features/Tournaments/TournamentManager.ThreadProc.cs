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

using Ceres.Base.Misc;

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
        gameTest.RunGameTests(threadIndex, (int numOpeningsAvailable) => GetNextOpeningIndexForLocalThread(numOpeningsAvailable, Def.NumGamePairs ?? int.MaxValue));
      }
      catch (Exception exc)
      {
        Console.WriteLine("Exception in TournamentManager thread: " + exc.ToString());
        RetireFailedGameThread(threadIndex, gameTest);
      }
    }


    /// <summary>
    /// Called when a game thread terminated with an exception (for example because the remote
    /// evaluation server of its device slot remained unreachable). The thread is retired and
    /// the remaining threads continue the tournament.
    ///
    /// If the thread had not yet recorded any game from the opening it was working on then
    /// that opening is returned to be played by another thread, so no opening is lost.
    /// </summary>
    /// <param name="threadIndex"></param>
    /// <param name="gameTest"></param>
    private void RetireFailedGameThread(int threadIndex, TournamentGameThread gameTest)
    {
      gameTest.WasRetired = true;

      string where = gameTest.Def.DeviceStream is { } stream ? $" (device slot {stream})" : "";
      string openingStr = "";

      int openingIndex = gameTest.OpeningIndexInProgress;
      if (openingIndex >= 0)
      {
        if (gameTest.NumGamesRecordedForOpeningInProgress == 0)
        {
          lock (lockObj)
          {
            openingsToRetry.Enqueue(openingIndex);
          }
          openingStr = $" Opening {openingIndex} was returned unplayed and will be played by another thread (if any is still running).";
        }
        else
        {
          openingStr = $" Opening {openingIndex} is left incomplete ({gameTest.NumGamesRecordedForOpeningInProgress} game(s) of it were already recorded).";
        }
      }

      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        $"WARNING: tournament game thread {threadIndex}{where} failed and was retired; remaining threads continue.{openingStr}");
    }

    /// <summary>
    /// Main processing loop when running as the coordinator for a distributed tournament.
    /// </summary>
    /// <param name="threadIndex"></param>
    /// <param name="gameTest"></param>
    private void ThreadProcCoordinator(int threadIndex, TournamentGameThread gameTest)
    {
      while (!gameTest.Def.parentDef.ShouldShutDown)
      {
        try
        {
          int openingIndex = GetNextOpeningIndexForLocalThread(-1, Def.NumGamePairs ?? int.MaxValue);
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
          Console.WriteLine("Exception in TournamentManager thread: " + exc.ToString());
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