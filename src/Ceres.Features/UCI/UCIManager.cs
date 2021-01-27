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
using System.Threading.Tasks;
using System.Collections.Generic;

using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Base.OperatingSystem.NVML;

using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNFiles;

using Ceres.MCTS.Iteration;
using Ceres.MCTS.Params;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.Utils;

using Ceres.Features.GameEngines;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Managers;
using Ceres.Chess.LC0VerboseMoves;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;

#endregion

namespace Ceres.Features.UCI
{
  /// <summary>
  /// Manager of UCI game loop, parsing and executing commands from Console
  /// and outputting appropriate UCI lines such as bestmove and info.
  /// </summary>
  public partial class UCIManager
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

    volatile Task<GameEngineSearchResultCeres> taskSearchCurrentlyExecuting;

    bool haveInitializedEngine;

    /// <summary>
    /// The position and history associated with the current evaluation.
    /// </summary>
    PositionWithHistory curPositionAndMoves;
    bool curPositionIsContinuationOfPrior;

    List<GameMoveStat> gameMoveHistory = new List<GameMoveStat>();

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
      curPositionAndMoves = PositionWithHistory.FromFENAndMovesUCI(Position.StartPosition.FEN);
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
            Send($"id name Ceres {CeresVersion.VersionString}");
            Send("id author David Elliott and the Ceres Authors");
            Send(SetOptionUCIDescriptions);
            Send("uciok");
            break;

          case string c when c.StartsWith("setoption"):
            ProcessSetOption(command);
            break;

          case "stop":
            if (taskSearchCurrentlyExecuting != null && !stopIsPending)
            {
              stopIsPending = true;

              CeresEngine.Search.Manager.ExternalStopRequested = true;
              if (taskSearchCurrentlyExecuting != null)
              {
                taskSearchCurrentlyExecuting.Wait();
                if (!debug && taskSearchCurrentlyExecuting != null) taskSearchCurrentlyExecuting.Result?.Search?.Manager?.Dispose();
                taskSearchCurrentlyExecuting = null;
              }

            }

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
            if (taskSearchCurrentlyExecuting != null)
            {
              CeresEngine.Search.Manager.ExternalStopRequested = true;
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
            if (CeresEngine?.Search != null)
            {
              string netID = EvaluatorDef.Nets[0].Net.NetworkID;
              INNWeightsFileInfo netDef = NNWeightsFiles.LookupNetworkFile(netID);
              (string exe, string options) = LC0EngineConfigured.GetLC0EngineOptions(null, null, CeresEngine.Search.Manager.Context.EvaluatorDef, netDef, false, false);
              OutStream.WriteLine("info string " + exe + " " + options);
            }
            else
              OutStream.WriteLine("info string No search manager created");

            break;

          case "dump-params":
            if (CeresEngine?.Search != null)
              CeresEngine?.Search.Manager.DumpParams();
            else
              OutStream.WriteLine("info string No search manager created");
            break;

          case "dump-processor":
            HardwareManager.DumpProcessorInfo();
            break;

          case "dump-time":
            if (CeresEngine?.Search != null)
              CeresEngine?.Search.Manager.DumpTimeInfo(OutStream);
            else
              OutStream.WriteLine("info string No search manager created");
            break;

          case "dump-store": 
            if (CeresEngine?.Search != null)
            {
              using (new SearchContextExecutionBlock(CeresEngine.Search.Manager.Context))
                CeresEngine.Search.Manager.Context.Tree.Store.Dump(true);
            }
            else
              OutStream.WriteLine("info string No search manager created");
            break;

          case "dump-move-stats":
            if (CeresEngine?.Search != null)
            {
              OutputVerboseMoveStats(CeresEngine.Search.SearchRootNode);
            }
            else
              OutStream.WriteLine("info string No search manager created");
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

          case "waitdone": // proprietary verb used for test driver
            taskSearchCurrentlyExecuting?.Wait();
            break;

          default:
            OutStream.WriteLine($"error Unknown command: {command}");
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
      if (CeresEngine?.Search != null)
      {
        using (new SearchContextExecutionBlock(CeresEngine?.Search.Manager.Context))
        {
          SearchPrincipalVariation pv2 = new SearchPrincipalVariation(CeresEngine.Search.Manager.Root);
          MCTSPosTreeNodeDumper.DumpPV(CeresEngine.Search.Manager.Context.Root, withDetail);
        }
      }
      else
        OutStream.WriteLine("info string No search manager created");
    }


    private void InitializeEngineIfNeeded()
    {
      if (!haveInitializedEngine)
      {
        // Create the engine (to be subsequently reused).
        CeresEngine = new GameEngineCeresInProcess("Ceres", EvaluatorDef, ParamsSearch, ParamsSelect, logFileName:logFileName);

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
      command = StringUtils.WhitespaceRemoved(command);
      
      string commandLower = command.ToLower();

      string posString;
      if (commandLower.StartsWith("position fen "))
        posString = command.Substring(13);
      else if (commandLower.StartsWith("position startpos"))
        posString = command.Substring(9);
      else
        throw new Exception($"Illegal position command, expected to start with position fen or position startpos");
      
      PositionWithHistory newPositionAndMoves = PositionWithHistory.FromFENAndMovesUCI(posString);

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
          OutStream.WriteLine("Exception in Ceres engine execution:");
          OutStream.WriteLine(exc);
          OutStream.WriteLine(exc.StackTrace);

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
      SearchLimit searchLimit;
      bool weAreWhite = curPositionAndMoves.FinalPosition.MiscInfo.SideToMove == SideType.White;
      UCIGoCommandParsed goInfo = new UCIGoCommandParsed(command, weAreWhite);
      if (!goInfo.IsValid) return null;

      if (goInfo.Nodes.HasValue)
      {
        searchLimit =  SearchLimit.NodesPerMove(goInfo.Nodes.Value);
      }
      else if (goInfo.MoveTime.HasValue)
      {
        searchLimit = SearchLimit.SecondsPerMove(goInfo.MoveTime.Value / 1000.0f);
      }
      else if (goInfo.Infinite)
      {
        searchLimit = SearchLimit.NodesPerMove(MCTSNodeStore.MAX_NODES);
      }
      else if (goInfo.TimeOurs.HasValue)
      {
        float increment = 0;
        if (goInfo.IncrementOurs.HasValue) increment = goInfo.IncrementOurs.Value / 1000.0f;

        int? movesToGo = null;
        if (goInfo.MovesToGo.HasValue) movesToGo = goInfo.MovesToGo.Value;

        searchLimit = SearchLimit.SecondsForAllMoves(goInfo.TimeOurs.Value / 1000.0f, increment, movesToGo, true);
      }
      else
      {
        OutStream.WriteLine($"Unsupported time control in UCI go command {command}");
        return null;
      }

      // Add on possible search moves restriction.
      return searchLimit with { SearchMoves = goInfo.SearchMoves };
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
          DateTime now = DateTime.Now;
          float timeSinceLastUpdate = (float)(now - lastInfoUpdate).TotalSeconds;

          bool isFirstUpdate = numUpdatesSent == 0;
          float UPDATE_INTERVAL_SECONDS = isFirstUpdate ? 0.1f : 0.5f;
          if (manager != null && timeSinceLastUpdate > UPDATE_INTERVAL_SECONDS && manager.Root.N > 0)
          {
            OutputUCIInfo();

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

      OutputUCIInfo(true);

      // Send the best move
      Send("bestmove " + result.Search.BestMove.MoveStr(MGMoveNotationStyle.LC0Coordinate));
      if (debug) Send("info string " + result.Search.SearchRootNode.BestMoveInfo(false));

      return result;
    }


    void OutputUCIInfo(bool isFinalInfo = false)
    {
      if (numPV == 1)
      {
        Send(UCIInfo.UCIInfoString(CeresEngine.Search.Manager, CeresEngine.Search.SearchRootNode, 
                                  showWDL:showWDL, scoreAsQ:scoreAsQ));
      }
      else
      {
        MCTSManager manager = CeresEngine.Search.Manager;
        MCTSNode searchRootNode = CeresEngine.Search.SearchRootNode;
        BestMoveInfo best = searchRootNode.BestMoveInfo(false);

        // Send top move
        Send(UCIInfo.UCIInfoString(manager, searchRootNode, best.BestMoveNode, 1, 
                                   showWDL: showWDL, useParentN: !perPVCounters, scoreAsQ: scoreAsQ));

        // Send other moves visited
        MCTSNode[] sortedN = searchRootNode.ChildrenSorted(s => -(float)s.N);
        int multiPVIndex = 2;
        for (int i=0;i<sortedN.Length && i <numPV; i++)
        {
          if (!object.ReferenceEquals(sortedN[i], best.BestMoveNode))
          {
            Send(UCIInfo.UCIInfoString(manager, searchRootNode, sortedN[i], multiPVIndex, 
                                       showWDL: showWDL, useParentN: !perPVCounters, scoreAsQ: scoreAsQ));
            multiPVIndex++;
          }
        }


        // Finally show moves that had no visits
        float elapsedTimeSeconds = (float)(DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;
        string timeStr = $"{ elapsedTimeSeconds * 1000.0f:F0}";
        for (int i = multiPVIndex-1; i<searchRootNode.NumPolicyMoves; i++)
        {
          (MCTSNode node, EncodedMove move, FP16 p) info = searchRootNode.ChildAtIndexInfo(i);
          if (info.node == null)
          {
            string str = $"info depth 0 seldepth 0 time { timeStr } nodes 1 score cp 0 tbhits 0 multipv {multiPVIndex} pv {info.move.AlgebraicStr} ";
            OutStream.WriteLine(str);
            multiPVIndex++;
          }
        }

      }
      if (verboseMoveStats && (logLiveStats || isFinalInfo)) OutputVerboseMoveStats(CeresEngine.Search.SearchRootNode);
    }

    /// <summary>
    /// Since "log live stats" is a LC0 only feature (used for example Nibbler)
    /// if in this mode we use the LC0 format instead of Ceres' own.
    /// </summary>
    bool ShouldUseLC0FormatForVerboseMoves => logLiveStats;

    void OutputVerboseMoveStats(MCTSNode searchRootNode)
    {
      using (new SearchContextExecutionBlock(CeresEngine.Search.Manager.Context))
      {
        if (ShouldUseLC0FormatForVerboseMoves)
        {
          foreach (LC0VerboseMoveStat stat in LC0VerboseMoveStatsFromMCTSNode.BuildStats(searchRootNode))
          {
            OutStream.WriteLine(stat.LC0String);
          }
        }
        else
        {
          CeresEngine.Search.Manager.Context.Root.Dump(1, 1, prefixString: "info string ");
        }
      }
    }

  }
}