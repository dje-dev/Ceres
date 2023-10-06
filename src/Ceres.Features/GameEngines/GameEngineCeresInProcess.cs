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
using System.Collections.Generic;

using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.Params;
using Ceres.MCTS.Environment;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.NodeCache;
using Ceres.Features.UCI;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Features.Visualization.AnalysisGraph;
using Ceres.Features.Commands;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Sublcass of GameEngine specialized for Ceres engine (running in-process).
  /// </summary>
  public class GameEngineCeresInProcess : GameEngine
  {
    /// <summary>
    /// Definition of neural network evaluator used for execution.
    /// </summary>
    public readonly NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// Definition of neural network evaluator used for secondary execution.
    /// </summary>
    public readonly NNEvaluatorDef EvaluatorDefSecondary;

    /// <summary>
    /// General search parameters used.
    /// </summary>
    public readonly ParamsSearch SearchParams;

    /// <summary>
    /// MCTS leaf selection parameters used.
    /// </summary>
    public readonly ParamsSelect ChildSelectParams;

    /// <summary>
    /// Manager used for apportioning node or time limits at the game
    /// level to individual moves.
    /// </summary>
    public IManagerGameLimit GameLimitManager;

    /// <summary>
    /// Current or most recently executed search.
    /// </summary>
    public MCTSearch Search;

    /// <summary>
    /// Search conducted prior to current search.
    /// </summary>
    public MCTSearch LastSearch;

    /// <summary>
    /// Optional name of file to which detailed log information 
    /// will be written after each move.
    /// </summary>
    public string SearchLogFileName;

    /// <summary>
    /// If the VerboseMoveStats should be populated at end of each search.
    /// </summary>
    public bool GatherVerboseMoveStats;

    /// <summary>
    /// If detailed information relating to search status of
    /// moves at root should be output at end of a search.
    /// </summary>
    public bool OutputVerboseMoveStats;

    /// <summary>
    /// Optional descriptive information for current game.
    /// </summary>
    public string CurrentGameID;

    /// <summary>
    /// If search should be short-circuited if only one legal move at root.
    /// </summary>
    public bool MoveImmediateIfOnlyOneMove;

    /// <summary>
    /// Optional list of moves to be forced to be made at each ply.
    /// </summary>
    public List<MGMove> ForcedMoves = null;


    #region Internal data

    /// <summary>
    /// Once created the NN evaluator pair is reused (until Dispose is called).
    /// </summary>
    public NNEvaluatorSet Evaluators { get; private set; }

    /// <summary>
    /// Attempt to retain node cache across all searches and games
    /// because it is typically a very large data structure to allocate and initialize.
    /// </summary>
    IMCTSNodeCache reuseNodeCache = null;

    #endregion

    Action<string> InfoLogger = null;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id">identifying string</param>
    /// <param name="evaluatorDef">primary evalutor for all nodes</param>
    /// <param name="evaluatorDefSecondary">optional secondary evaluator to be run on subset of tree</param>
    /// <param name="searchParams">optional non-default search parameters</param>
    /// <param name="childSelectParams">optional non-default child selection parameters </param>
    /// <param name="gameLimitManager">optional override manager for search limits</param>
    /// <param name="logFileName">optional name of file to which to write detailed log</param>
    /// <param name="moveImmediateIfOnlyOneMove">if engine should chose best move immediately without search if only one legal move</param>
    /// <param name="processorGroupID">id of processor group on which engine should execute</param>
    public GameEngineCeresInProcess(string id,
                                    NNEvaluatorDef evaluatorDef,
                                    NNEvaluatorDef evaluatorDefSecondary = null,
                                    ParamsSearch searchParams = null,
                                    ParamsSelect childSelectParams = null,
                                    IManagerGameLimit gameLimitManager = null,
                                    string logFileName = null,
                                    bool moveImmediateIfOnlyOneMove = true,
                                    int processorGroupID = 0,
                                    Action<string> infoLogger = null,
                                    List<MGMove> forcedMoves = null) : base(id, processorGroupID)
    {
      if (evaluatorDef == null)
      {
        throw new ArgumentNullException(nameof(evaluatorDef));
      }

      // Use default settings for search and select params if not specified.
      if (searchParams == null)
      {
        searchParams = new ParamsSearch();
      }

      if (childSelectParams == null)
      {
        childSelectParams = new ParamsSelect();
      }

      // Use default limit manager if not specified.
      if (gameLimitManager == null)
      {
        // Check for alternate limits manager specified in Ceres settings.
        string altManager = CeresUserSettingsManager.Settings.LimitsManagerName;
        if (altManager == null)
        {
          gameLimitManager = new ManagerGameLimitCeres(searchParams.GameLimitUsageAggressiveness);
        }
        else
        {
          if (altManager.ToUpper() == "TEST")
          {
            gameLimitManager = new ManagerGameLimitTest(searchParams.GameLimitUsageAggressiveness);
          }
          else
          {
            throw new NotImplementedException(altManager + " not supported for setting AlternateLimitsManagerName");
          }
        }
      }

      EvaluatorDef = evaluatorDef;
      EvaluatorDefSecondary = evaluatorDefSecondary;
      SearchParams = searchParams;
      GameLimitManager = gameLimitManager;
      ChildSelectParams = childSelectParams;
      SearchLogFileName = logFileName;
      MoveImmediateIfOnlyOneMove = moveImmediateIfOnlyOneMove;
      OutputVerboseMoveStats = CeresUserSettingsManager.Settings.VerboseMoveStats;
      InfoLogger = infoLogger;
      ForcedMoves = forcedMoves;

      if (logFileName == null && !string.IsNullOrEmpty(CeresUserSettingsManager.Settings.SearchLogFile))
      {
        SearchLogFileName = CeresUserSettingsManager.Settings.SearchLogFile;
      }
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => true;


    bool isFirstMoveOfGame = true;

    /// <summary>
    /// Resets all state between games.
    /// </summary>
    /// <param name="gameID">optional game descriptive string</param>
    public override void ResetGame(string gameID = null)
    {
      reuseNodeCache = Search?.Manager.Context.Tree.NodeCache;

      LastSearch?.Manager.Dispose();
      Search?.Manager.Dispose();

      Search = null;
      LastSearch = null;

      isFirstMoveOfGame = true;
      CurrentGameID = gameID;
    }

    /// <summary>
    /// Executes any preparatory steps (that should not be counted in thinking time) before a search.
    /// </summary>
    protected override void DoSearchPrepare()
    {
    }


    static readonly object logFileWriteObj = new object();

    /// <summary>
    /// Runs a search, calling DoSearch and adjusting the cumulative search time
    /// (convenience method with same functionality but returns the as the subclass
    /// GameEngineSearchResultCeres.
    /// </summary>
    /// <param name="curPositionAndMoves"></param>
    /// <param name="searchLimit"></param>
    /// <param name="callback"></param>
    /// <returns></returns>
    public GameEngineSearchResultCeres SearchCeres(PositionWithHistory curPositionAndMoves,
                                                   SearchLimit searchLimit,
                                                   List<GameMoveStat> gameMoveHistory = null,
                                                   ProgressCallback callback = null,
                                                   bool verbose = false)
    {
      return Search(curPositionAndMoves, searchLimit, gameMoveHistory, callback, verbose) as GameEngineSearchResultCeres;
    }


  /// <summary>
  /// Overridden virtual method which executes search.
  /// </summary>
  /// <param name="curPositionAndMoves"></param>
  /// <param name="searchLimit"></param>
  /// <param name="gameMoveHistory"></param>
  /// <param name="callback"></param>
  /// <returns></returns>
  protected override GameEngineSearchResult DoSearch(PositionWithHistory curPositionAndMoves,
                                                       SearchLimit searchLimit,
                                                       List<GameMoveStat> gameMoveHistory,
                                                       ProgressCallback callback,
                                                       bool verbose)
    {
      if (LastSearch != null && curPositionAndMoves.InitialPosMG != LastSearch.Manager.Context.StartPosAndPriorMoves.InitialPosMG)
      {
        throw new Exception("ResetGame must be called if not continuing same line");
      }

      // Possibly set a forced move if a list of such moves was provided and is not yet exhausted.
      MGMove forcedMove = (ForcedMoves == null || ForcedMoves.Count < curPositionAndMoves.Count) 
                            ? default 
                            : ForcedMoves[curPositionAndMoves.Count - 1];

      // Set max tree nodes to maximum value based on memory (unless explicitly overridden in passed SearchLimit)
      searchLimit = searchLimit with
      {
        MaxTreeVisits = searchLimit.MaxTreeVisits ?? MCTSParamsFixed.MAX_VISITS,
      };

      MCTSearch searchResult;

      // Set up callback passthrough if provided
      MCTSManager.MCTSProgressCallback callbackMCTS = null;
      if (callback != null)
      {
        callbackMCTS = callbackContext => callback((MCTSManager)callbackContext);
      }

      // Possibly use the context of opponent to reuse position evaluations
      MCTSIterator shareContext = null;
      if (OpponentEngine is GameEngineCeresInProcess)
      {
        GameEngineCeresInProcess ceresOpponentEngine = OpponentEngine as GameEngineCeresInProcess;

        if (LastSearch is not null
         && LastSearch.Manager.Context.ParamsSearch.ReusePositionEvaluationsFromOtherTree
         && ceresOpponentEngine?.LastSearch.Manager != null
         && LeafEvaluatorReuseOtherTree.ContextsCompatibleForReuse(LastSearch.Manager.Context, ceresOpponentEngine.LastSearch.Manager.Context))
        {
          shareContext = ceresOpponentEngine.LastSearch.Manager.Context;

          // Clear any prior shared context from the shared context
          // to prevent unlimited backward chaining (keeping unneeded prior contexts alive)
          shareContext.ClearSharedContext();
        }
      }


    void InnerCallback(MCTSManager manager)
      {
        // Check for possible externally enqueued command.
        (string command, string options) = InterprocessCommandManager.TryDequeuePendingCommand();
        if (command != default)
        {
          AnalysisGraphOptions optionsObj = AnalysisGraphOptions.FromString(options);
          Console.WriteLine($"Writing Analysis Graph (detail level {optionsObj.DetailLevel})");
          AnalysisGraphGenerator graphGenerator = new AnalysisGraphGenerator(Search, optionsObj);
          graphGenerator.Write(true);
        }
        callbackMCTS?.Invoke(manager);
      }

      // Run the search
      searchResult = RunSearchPossiblyTreeReuse(shareContext, curPositionAndMoves, gameMoveHistory,
                                                searchLimit, InnerCallback, 
                                                (string infoMsg) => InfoLogger?.Invoke(infoMsg), 
                                                verbose, forcedMove);

      int scoreCeresCP;
      BestMoveInfo bestMoveInfo = null;
      int N;
      bestMoveInfo = searchResult.Manager.Root.BestMoveInfo(false, forcedMove);
      N = searchResult.SearchRootNode.N;
#if NOT
        bool wouldBeDrawByRepetition = PositionRepetitionCalc.DrawByRepetitionWouldBeClaimable(curPositionAndMoves.FinalPosition, bestMoveInfo.BestMove, curPositionAndMoves.GetPositions());
        if (wouldBeDrawByRepetition)
        {
        }
#endif

      scoreCeresCP = (int)MathF.Round(EncodedEvalLogistic.WinLossToCentipawn(bestMoveInfo.QOfBest), 0);


      MGMove bestMoveMG = searchResult.BestMove;

      // Save (do not dispose) this search in case we can reuse it next time.
      LastSearch = searchResult;

      isFirstMoveOfGame = false;
      // TODO is the RootNWhenSearchStarted correct because we may be following a continuation (BestMoveRoot)
      GameEngineSearchResultCeres result =
        new GameEngineSearchResultCeres(bestMoveMG.MoveStr(MGMoveNotationStyle.LC0Coordinate),
                                        (float)bestMoveInfo.QOfBest, scoreCeresCP, searchResult.SearchRootNode.MAvg, searchResult.Manager.SearchLimit,
                                        this.LastSearch.TimingInfo,
                                        searchResult.Manager.RootNWhenSearchStarted, N, (int)Math.Round(searchResult.Manager.Context.AvgDepth),
                                        searchResult, bestMoveInfo);


#if FLAG_POSSIBLE_FALSE_DRAWS
        if (result.Depth < 4 && Math.Abs(result.ScoreCentipawns) < 2 
         && Math.Abs(result.Search.SearchRootNode.V) > 0.5f)
        {
          Console.WriteLine("RETRY " + curPositionAndMoves.FinalPosition.FEN);
        }
#endif

      // Append search result information to log file (if any).
      StringWriter dumpInfo = new StringWriter();
      if (SearchLogFileName != null)
      {
        result.Search.Manager.DumpFullInfo(bestMoveMG, result.Search.SearchRootNode,
                                           result.Search.LastReuseDecision, result.Search.LastMakeNewRootTimingStats,
                                           result.Search.LastGameLimitInputs,
                                           dumpInfo, CurrentGameID);
        lock (logFileWriteObj)
        {
          File.AppendAllText(SearchLogFileName, dumpInfo.GetStringBuilder().ToString());
        }
      }

      if (GatherVerboseMoveStats)
      {
        result.VerboseMoveStats = GetVerboseMoveStats();
      }

      if (OutputVerboseMoveStats)
      {
        result.Search.Manager.Context.Root.Dump(1, 1);
      }

      return result;
    }

    public override void Warmup(int? knownMaxNumNodes = null)
    { 
      PrepareEvaluators();
    }

    void PrepareEvaluators()
    {
      if (Evaluators == null)
      {
        Evaluators = new NNEvaluatorSet(EvaluatorDef, SearchParams.Execution.FlowDirectOverlapped, EvaluatorDefSecondary);
        Evaluators.Warmup(false);
      }
    }

    /// <summary>
    /// Launches search, possibly as continuation from last search.
    /// </summary>
    /// <param name="reuseOtherContextForEvaluatedNodes"></param>
    /// <param name="curPositionAndMoves"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="searchLimit"></param>
    /// <param name="VERBOSE"></param>
    /// <param name="callback"></param>
    /// <returns></returns>
    private MCTSearch RunSearchPossiblyTreeReuse(MCTSIterator reuseOtherContextForEvaluatedNodes,
                                                 PositionWithHistory curPositionAndMoves,
                                                 List<GameMoveStat> gameMoveHistory,
                                                 SearchLimit searchLimit,
                                                 MCTSManager.MCTSProgressCallback callback,
                                                 MCTSearch.MCTSInfoLogger infoLogger,
                                                 bool verbose,
                                                 MGMove forcedMove)
    {
      PositionEvalCache positionCacheOpponent = null;

      Search = new MCTSearch(infoLogger);

      if (LastSearch == null)
      {
        PrepareEvaluators();
        Search.Search(Evaluators, ChildSelectParams, SearchParams, GameLimitManager,
                      reuseOtherContextForEvaluatedNodes,
                      curPositionAndMoves, searchLimit, verbose, lastSearchStartTime,
                      gameMoveHistory, callback, null, reuseNodeCache, false, isFirstMoveOfGame,
                      MoveImmediateIfOnlyOneMove, forcedMove:forcedMove);
      }
      else
      {
        if (LastSearch.Manager.Context.StartPosAndPriorMoves.InitialPosMG != curPositionAndMoves.InitialPosMG)
        {
          throw new Exception("Internal error: not same starting position");
        }

        List<MGMove> forwardMoves = new List<MGMove>();
        List<MGMove> lastMoves = LastSearch.Manager.Context.StartPosAndPriorMoves.Moves;
        for (int i = 0; i < curPositionAndMoves.Moves.Count; i++)
        {
          if (i < lastMoves.Count)
          {
            if (lastMoves[i] != curPositionAndMoves.Moves[i])
            {
              throw new Exception("Internal error: move sequence is not a prefix");
            }
          }
          else
            forwardMoves.Add(curPositionAndMoves.Moves[i]);
        }

        // Determine the minimum fraction of tree that would need to be useful
        // before we would possibly reuse part of tree from prior search.
        // Below a certain level the tree reuse would increase memory consumption
        // (because unused nodes remain in node store) but not appreciably improve search speed.
        int currentN = LastSearch.SearchRootNode.N;

        // Determine threshold to decide if tree is big enough to be reused.
        // Extraction time is about linear in number of nodes extracted (averages about 2.5mm/sec)
        // except for a fixed time required to build BitArray of included nodes (which is only about 2% of total time).
        // Therefore we almost always reuse the tree, unless extremely small.
        float THRESHOLD_FRACTION_NODES_REUSABLE = 0.03f;

        Search.SearchContinue(LastSearch, reuseOtherContextForEvaluatedNodes,
                                     forwardMoves, curPositionAndMoves,
                                     gameMoveHistory, searchLimit, verbose, lastSearchStartTime,
                                     callback, THRESHOLD_FRACTION_NODES_REUSABLE,
                                     isFirstMoveOfGame, MoveImmediateIfOnlyOneMove, forcedMove);
      }

      // Update the statistic on possible overshoot of internally alloted time and actual
      // (unless first move, since extra overhead is perhaps unavoidable in that situation).
      if (!isFirstMoveOfGame)
      {
        TimeSpan elapsedTime = DateTime.Now - lastSearchStartTime;
        if (Search.Manager.SearchLimit.IsTimeLimit)
        {
          if (Search.Manager.SearchLimit.IsPerGameLimit)
          {
            throw new NotImplementedException();
          }

          float timeOvershoot = (float)elapsedTime.TotalSeconds - Search.Manager.SearchLimit.Value;
          MCTSEventSource.MaximumTimeAllotmentOvershoot = Math.Max(MCTSEventSource.MaximumTimeAllotmentOvershoot, timeOvershoot);
        }
      }

      return Search;
    }


    public override void DumpMoveHistory(List<GameMoveStat> gameMoveHistory, SideType? side)
    {
      // TODO: fill in increemental time below (last argument)
      ManagerGameLimitInputs timeManagerInputs = new(LastSearch.Manager.Context.StartPosAndPriorMoves.FinalPosition,
                                                LastSearch.Manager.Context.ParamsSearch,
                                                gameMoveHistory, SearchLimitType.SecondsPerMove,
                                                LastSearch.Manager.Root.N, (float)LastSearch.Manager.Root.Q,
                                                LastSearch.Manager.SearchLimit.Value, 0, null, null, 0, 0,
                                                null, gameMoveHistory.Count == 0,
                                                LastSearch.Manager.Context.ParamsSearch.TestFlag);
      timeManagerInputs.Dump(side);
    }


    /// <summary>
    /// Returns UCI information string 
    /// (such as would appear in a chess GUI describing search progress) 
    /// based on last state of search.
    /// </summary>
    public override UCISearchInfo UCIInfo
    {
      get
      {
        if (LastSearch != null)
        {
          return new UCISearchInfo(MCTS.Utils.UCIInfo.UCIInfoString(Search.Manager, Search.SearchRootNode));
        }
        else
        {
          return null;
        }
      }
    }


    /// <summary>
    /// Returns list of verbose move statistics pertaining to current search root node.
    /// </summary>
    /// <returns></returns>
    public List<VerboseMoveStat> GetVerboseMoveStats()
    {
      if (Search == null)
      {
        throw new Exception("GetVerboseMoveStats cannot return search statistics because no search has run yet.");
      }

      return VerboseMoveStatsFromMCTSNode.BuildStats(Search.SearchRootNode);
    }



    /// <summary>
    /// Diposes underlying search engine.
    /// </summary>
    public override void Dispose()
    {
      Evaluators?.Dispose();
      Evaluators = null;

      Search?.Manager.Dispose();
      Search = null;

      LastSearch?.Manager.Dispose();
      LastSearch = null;
      reuseNodeCache = null;
    }

  }
}
