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
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.OperatingSystem.NVML;

using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNFiles;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.Features.GameEngines;
using Ceres.Chess.LC0.Positions;
using Ceres.Features.Analysis;
using Ceres.MCTS.Iteration;
using Ceres.Chess.Positions;
using System.Collections.Generic;
using Ceres.MCTS.Params;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.Base.OperatingSystem;
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.Features.UCI
{
  /// <summary>
  /// Manager of UCI game loop, parsing and executing commands from Console
  /// and outputting appropriate UCI lines such as bestmove and info.
  /// </summary>
  public class UCIManager
  {
    /// <summary>
    /// Input stream.
    /// </summary>
    public readonly TextReader InStream;

    /// <summary>
    /// Output stream.
    /// </summary>
    public readonly TextWriter OutStream;

    /// <summary>
    /// Action to be called upon end of Ceres search.
    /// </summary>
    public Action<MCTSManager> SearchFinishedEvent;

    public readonly NNEvaluatorDef EvaluatorDef;
    public readonly ParamsSearch ParamsSearch;
    public readonly ParamsSelect ParamsSelect;

    /// <summary>
    /// Ceres engine instance used for current UCI game.
    /// </summary>
    public GameEngineCeresInProcess CeresEngine;

    Task<GameEngineSearchResultCeres> taskSearchCurrentlyExecuting;

    bool haveInitializedEngine;

    /// <summary>
    /// The position and history associated with the current evaluation.
    /// </summary>
    PositionWithHistory curPositionAndMoves;
    bool curPositionIsContinuationOfPrior;

    MCTSIterator curContext => curManager.Context;

    List<GameMoveStat> gameMoveHistory = new List<GameMoveStat>();
    MCTSManager curManager;

    bool stopIsPending;
    bool debug = false;



    /// <summary>
    /// Construtor.
    /// </summary>
    /// <param name="evaluatorDef"></param>
    /// <param name="inStream"></param>
    /// <param name="outStream"></param>
    /// <param name="searchFinishedEvent"></param>
    public UCIManager(NNEvaluatorDef evaluatorDef, 
                      TextReader inStream = null, TextWriter outStream = null,                       
                      Action<MCTSManager> searchFinishedEvent = null,
                      bool disablePruning = false)
    {
      InStream = inStream ?? Console.In;
      OutStream = outStream ?? Console.Out;
      SearchFinishedEvent = searchFinishedEvent;
      
      EvaluatorDef = evaluatorDef;      

      ParamsSearch = new ParamsSearch();
      ParamsSelect = new ParamsSelect();

      if (disablePruning) ParamsSearch.FutilityPruningStopSearchEnabled = false;
    }

    /// <summary>
    /// Outputs line to UCI.
    /// </summary>
    /// <param name="result"></param>
    void Send(string result) => OutStream.WriteLine(result);
    



    /// <summary>
    /// Runs the UCI loop.
    /// </summary>
    public void PlayUCI()
    {
      // Default to the startpos.
      curPositionAndMoves = PositionWithHistory.FromFENAndMovesUCI(Position.StartPosition.FEN, null);
      curManager = null;
      gameMoveHistory = new List<GameMoveStat>();

      while (true)
      {
        string command = InStream.ReadLine();

        switch (command)
        {
          case null:
          case "":
            break;

          case "uci":
            Send("id name Ceres"); // TODO: Add executable version
            Send("id author David Elliott and the Ceres Authors");
            // todo output options such as:
            //   option name Logfile type check default false
            Send("uciok");
            break;

          case "setoption":
            OutStream.WriteLine("Not processing option " + command);
            return;

          case "stop":
            if (taskSearchCurrentlyExecuting != null && !stopIsPending)
            {
              stopIsPending = true;

              // TODO: cleanup
              //       Possible race condition, curManager is only set in search callback which may not have hit yet
              //       Fix eventually by rewriting SerchManager to have a constructor and then non-static Search method,
              //       os we can get the context in this class directly after construction
              while (curManager == null) Thread.Sleep(1); // **** TEMPORARY ***

              curManager.ExternalStopRequested = true;
              if (taskSearchCurrentlyExecuting != null)
              {
                taskSearchCurrentlyExecuting.Wait();
                if (!debug && taskSearchCurrentlyExecuting != null) taskSearchCurrentlyExecuting.Result?.Search?.Manager?.Dispose();
                taskSearchCurrentlyExecuting = null;
              }

            }

            curManager = null;
            stopIsPending = false;

            break;

          case "ponderhit":
            throw new NotImplementedException("Ceres does not yet support UCI ponder mode.");
            return;

          case "xboard":
            // ignore
            break;

          case "debug on":
            debug = true;
            break;

          case "debug off":
            debug = false;
            break;

          case "isready":
            InitializeEngineIfNeeded();
            Send("readyok");
            break;

          case "ucinewgame":
            gameMoveHistory = new List<GameMoveStat>();
            CeresEngine?.ResetGame();
            break;

          case "quit":
            if (curManager != null)
            {
              curManager.ExternalStopRequested = true;
              taskSearchCurrentlyExecuting?.Wait();
            }

            if (CeresEngine != null)
              CeresEngine.Dispose();

            System.Environment.Exit(0);
            break;

          case string c when c.StartsWith("go"):
            if (taskSearchCurrentlyExecuting != null)
              throw new Exception("Received go command when another search was running and not stopped first");

            InitializeEngineIfNeeded();

            taskSearchCurrentlyExecuting = ProcessGo(command);
            break;

          case string c when c.StartsWith("position"):
            try
            {
              ProcessPosition(c);
            }
            catch (Exception e)
            {
              Send($"Illegal position command: \"{c}\"" + System.Environment.NewLine + e.ToString());
            }
            break;

          // Proprietary commands
          case "lc0-config":
            if (curManager != null)
            {
              string netID = EvaluatorDef.Nets[0].Net.NetworkID;
              INNWeightsFileInfo netDef = NNWeightsFiles.LookupNetworkFile(netID);
              (string exe, string options) = LC0EngineConfigured.GetLC0EngineOptions(null, null, curContext.EvaluatorDef, netDef, false, false);
              Console.WriteLine("info string " + exe + " " + options);
            }
            else
              Console.WriteLine("info string No search manager created");

            break;

          case "dump-params":
            if (curManager != null)
              curManager.DumpParams();
            else
              Console.WriteLine("info string No search manager created");
            break;

          case "dump-processor":
            HardwareManager.DumpProcessorInfo();
            break;

          case "dump-time":
            if (curManager != null)
              curManager.DumpTimeInfo();
            else
              Console.WriteLine("info string No search manager created");
            break;

          case "dump-store": 
            if (curManager != null)
            {
              using (new SearchContextExecutionBlock(curContext))
                curManager.Context.Tree.Store.Dump(true);
            }
            else
              Console.WriteLine("info string No search manager created");
            break;

          case "dump-move-stats":
            if (curManager != null)
            {
              using (new SearchContextExecutionBlock(curContext))
                curManager.Context.Root.Dump(1, 1, prefixString:"info string ");
            }
            else
              Console.WriteLine("info string No search manager created");
            break;

          case "dump-pv":
            DumpPV(false);
            break;

          case "dump-pv-detail":
            DumpPV(true);
            break;

          case "dump-nvidia":
            NVML.DumpInfo();
            break;


          case "waitdone": // proprietary verb
            taskSearchCurrentlyExecuting?.Wait();
            break;

          default:
            Console.WriteLine($"error Unknown command: {command}");
            break;
        }
      }
    }

    /// <summary>
    /// Dumps the PV (principal variation) to the output stream.
    /// </summary>
    /// <param name="withDetail"></param>
    private void DumpPV(bool withDetail)
    {
      if (curManager != null)
      {
        using (new SearchContextExecutionBlock(curContext))
        {
          SearchPrincipalVariation pv2 = new SearchPrincipalVariation(curContext.Root);
          MCTSPosTreeNodeDumper.DumpPV(curContext.StartPosAndPriorMoves, curContext.Root, withDetail, null);
        }
      }
      else
        Console.WriteLine("info string No search manager created");
    }


    private void InitializeEngineIfNeeded()
    {
      if (!haveInitializedEngine)
      {
        // Create the engine (to be subsequently reused).
        CeresEngine = new GameEngineCeresInProcess("Ceres", EvaluatorDef, ParamsSearch, ParamsSelect);

        // Initialize engine
        CeresEngine.Warmup();
        haveInitializedEngine = true;
      }
    }


    /// <summary>
    /// Processes a specified position command, 
    /// with the side effect of resetting the curPositionAndMoves.
    /// </summary>
    /// <param name="command"></param>
    private void ProcessPosition(string command)
    {
      string[] parts = command.Split(" ");

      string fen;
      int nextIndex;

      string startFEN;
      switch (parts[1])
      {
        case "fen":
          fen = command.Substring(command.IndexOf("fen") + 4);
          nextIndex = 2;
          while (parts.Length > nextIndex && parts[nextIndex] != "moves")
            fen = fen + " " + parts[nextIndex++];
          startFEN = fen;
          break;

        case "startpos":
          startFEN = Position.StartPosition.FEN;
          nextIndex = 2;
          break;

        default:
          throw new Exception("invalid " + command);
      }

      string movesSubstring = "";
      if (parts.Length > nextIndex && parts[nextIndex] == "moves")
      {
        for (int i = nextIndex + 1; i < parts.Length; i++)
          movesSubstring += parts[i] + " ";
      }

      PositionWithHistory newPositionAndMoves = PositionWithHistory.FromFENAndMovesUCI(startFEN, movesSubstring);

      curPositionIsContinuationOfPrior = newPositionAndMoves.IsIdenticalToPriorToLastMove(curPositionAndMoves);
      if (!curPositionIsContinuationOfPrior && CeresEngine != null)
      {
        CeresEngine.ResetGame();
      }

      // Switch to the new position and moves
      curPositionAndMoves = newPositionAndMoves;
    }


    /// <summary>
    /// Processes the go command.
    /// </summary>
    /// <param name="command"></param>
    /// <returns></returns>
    private Task<GameEngineSearchResultCeres> ProcessGo(string command)
    {
      return Task.Run(() =>
      {
        try
        {
          SearchLimit searchLimit;

          // Parse the search limit
          searchLimit = GetSearchLimit(command);

          GameEngineSearchResultCeres result = null;
          if (searchLimit != null)
          {
            if (searchLimit.Value > 0)
            {
              // Run the actual search
              result = RunSearch(searchLimit);
            }
          }

          taskSearchCurrentlyExecuting = null;
          return result;
        }
        catch (Exception exc)
        {
          Console.WriteLine("Exception in Ceres engine execution:");
          Console.WriteLine(exc);
          Console.WriteLine(exc.StackTrace);

          System.Environment.Exit(3);
          return null;
        }
      });
    }


    /// <summary>
    /// Parses a specification of search time in UCI format into an equivalent SearchLimit.
    /// Returns null if parsing failed.
    /// </summary>
    /// <param name="command"></param>
    /// <returns></returns>
    private SearchLimit GetSearchLimit(string command)
    {
      bool weAreWhite = curPositionAndMoves.FinalPosition.MiscInfo.SideToMove == SideType.White;
      UCIGoCommandParsed goInfo = new UCIGoCommandParsed(command, weAreWhite);
      if (!goInfo.IsValid) return null;

      if (goInfo.Nodes.HasValue)
      {
        return SearchLimit.NodesPerMove(goInfo.Nodes.Value);
      }
      else if (goInfo.MoveTime.HasValue)
      {
        return SearchLimit.SecondsPerMove(goInfo.MoveTime.Value / 1000.0f);
      }
      else if (goInfo.Infinite)
      {
        return SearchLimit.NodesPerMove(MCTSNodeStore.MAX_NODES);
      }
      else if (goInfo.TimeOurs.HasValue)
      {
        float increment = 0;
        if (goInfo.IncrementOurs.HasValue) increment = goInfo.IncrementOurs.Value / 1000.0f;

        int? movesToGo = null;
        if (goInfo.MovesToGo.HasValue) movesToGo = goInfo.MovesToGo.Value;

        return SearchLimit.SecondsForAllMoves(goInfo.TimeOurs.Value / 1000.0f, increment, movesToGo, true);
      }
      else
      {
        Console.WriteLine($"Unsupported time control in UCI go command {command}");
        return null;
      }
    }


    /// <summary>
    /// Actually runs a search with specified limits.
    /// </summary>
    /// <param name="searchLimit"></param>
    /// <returns></returns>
    private GameEngineSearchResultCeres RunSearch(SearchLimit searchLimit)
    {
      DateTime lastInfoUpdate = DateTime.Now;

      int numUpdatesSent = 0;

      MCTSManager.MCTSProgressCallback callback =
        (manager) =>
        {
          curManager = manager;

          DateTime now = DateTime.Now;
          float timeSinceLastUpdate = (float)(now - lastInfoUpdate).TotalSeconds;

          bool isFirstUpdate = numUpdatesSent == 0;
          float UPDATE_INTERVAL_SECONDS = isFirstUpdate ? 0.1f : 0.5f;
          if (curManager != null && timeSinceLastUpdate > UPDATE_INTERVAL_SECONDS && curManager.Root.N > 0)
          {
            Send(UCIInfoString(curManager));

            numUpdatesSent++;
            lastInfoUpdate = now;
          }
        };

      GameEngineCeresInProcess.ProgressCallback callbackPlain = obj => callback((MCTSManager)obj);

      // use this? movesSinceNewGame

      // Search from this position (possibly with tree reuse)
      GameEngineSearchResultCeres result = CeresEngine.Search(curPositionAndMoves, searchLimit, gameMoveHistory, callbackPlain) as GameEngineSearchResultCeres;

      GameMoveStat moveStat = new GameMoveStat(gameMoveHistory.Count,
                                               curPositionAndMoves.FinalPosition.MiscInfo.SideToMove,
                                               result.ScoreQ, result.ScoreCentipawns,
                                               float.NaN, //engine1.CumulativeSearchTimeSeconds, 
                                               curPositionAndMoves.FinalPosition.PieceCount,
                                               result.MAvg, result.FinalN, result.FinalN - result.StartingN,
                                               searchLimit,
                                               (float)result.TimingStats.ElapsedTimeSecs);

      gameMoveHistory.Add(moveStat);

      if (SearchFinishedEvent != null) SearchFinishedEvent(result.Search.Manager);

      // Send the final info string (unless this was an instamove).
      Send(UCIInfoString(result.Search.Manager, result.Search.BestMoveRoot));

      // Send the best move
      Send("bestmove " + result.Search.BestMove.MoveStr(MGMoveNotationStyle.LC0Coordinate));
      if (debug) Send("info string " + result.Search.BestMoveRoot.BestMoveInfo(false));

      return result;
    }

    public static string UCIInfoString(MCTSManager manager, MCTSNode bestMoveRoot = null)
    {
      // If no override bestMoveRoot was specified
      // then it is assumed the move chosen was from the root (not an instamove)
      if (bestMoveRoot == null) bestMoveRoot = manager.Root;

      bool wasInstamove = manager.Root != bestMoveRoot;

      float elapsedTimeSeconds = wasInstamove ? 0 : (float)(DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;

      float scoreCentipawn = MathF.Round(EncodedEvalLogistic.LogisticToCentipawn((float)bestMoveRoot.Q), 0);
      float nps = manager.NumStepsTakenThisSearch / elapsedTimeSeconds;

      SearchPrincipalVariation pv;
      using (new SearchContextExecutionBlock(manager.Context))
      {
        pv = new SearchPrincipalVariation(bestMoveRoot);
      }

      //info depth 12 seldepth 27 time 30440 nodes 51100 score cp 105 hashfull 241 nps 1678 tbhits 0 pv e6c6 c5b4 d5e4 d1e1 
      int selectiveDepth = pv.Nodes.Count - 1;
      int depthOfBestMoveInTree = wasInstamove ? bestMoveRoot.Depth : 0;
      int depth = (int)MathF.Round(manager.Context.AvgDepth - depthOfBestMoveInTree, 0);


      if (wasInstamove)
      {
        // Note that the correct tablebase hits cannot be easily calculated and reported
        return $"info depth {depth} seldepth {selectiveDepth} time 0 "
             + $"nodes {bestMoveRoot.N:F0} score cp {scoreCentipawn:F0} tbhits {manager.CountTablebaseHits} nps 0 "
             + $"pv {pv.ShortStr()} string M= {bestMoveRoot.MAvg:F0} instamove";
      }
      else
      {
        return $"info depth {depth} seldepth {selectiveDepth} time {elapsedTimeSeconds * 1000.0f:F0} "
             + $"nodes {manager.Root.N:F0} score cp {scoreCentipawn:F0} tbhits {manager.CountTablebaseHits} nps {nps:F0} "
             + $"pv {pv.ShortStr()} string M= {manager.Root.MAvg:F0}";
      }
    }

  }
}