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
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.NodeCache;
using Ceres.Features.UCI;
using Ceres.Chess.LC0VerboseMoves;

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
    /// Manager used for approprtioning node or time limits at the game
    /// level to individual moves.
    /// </summary>
    public readonly IManagerGameLimit GameLimitManager;

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
    /// If detailed information relating to search status of
    /// moves at root should be output at end of a search.
    /// </summary>
    public bool VerboseMoveStats;

    /// <summary>
    /// Optional descriptive information for current game.
    /// </summary>
    public string CurrentGameID;

    /// <summary>
    /// If search should be short-circuited if only one legal move at root.
    /// </summary>
    public bool MoveImmediateIfOnlyOneMove;


    #region Internal data

    /// <summary>
    /// Once created the NN evaluator pair is reused (until Dispose is called).
    /// </summary>
    NNEvaluatorSet evaluators = null;

    /// <summary>
    /// Attempt to retain node cache across all searches and games
    /// because it is typically a very large data structure to allocate and initialize.
    /// </summary>
    IMCTSNodeCache reuseNodeCache = null;

    #endregion


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
    public GameEngineCeresInProcess(string id,
                                    NNEvaluatorDef evaluatorDef,
                                    NNEvaluatorDef evaluatorDefSecondary = null,
                                    ParamsSearch searchParams = null,
                                    ParamsSelect childSelectParams = null,
                                    IManagerGameLimit gameLimitManager = null,
                                    string logFileName = null,
                                    bool moveImmediateIfOnlyOneMove = true) : base(id)
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
        gameLimitManager = new ManagerGameLimitCeres(searchParams.GameLimitUsageAggressiveness);
      }

      EvaluatorDef = evaluatorDef;
      EvaluatorDefSecondary = evaluatorDefSecondary;
      SearchParams = searchParams;
      GameLimitManager = gameLimitManager;
      ChildSelectParams = childSelectParams;
      SearchLogFileName = logFileName;
      MoveImmediateIfOnlyOneMove = moveImmediateIfOnlyOneMove;
      VerboseMoveStats = CeresUserSettingsManager.Settings.VerboseMoveStats;

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
    /// Overriden virtual method which executes search.
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
        callbackMCTS?.Invoke(manager);
      }

      // Run the search
      searchResult = RunSearchPossiblyTreeReuse(shareContext, curPositionAndMoves, gameMoveHistory,
                                                searchLimit, InnerCallback, verbose);

      int scoreCeresCP;
      BestMoveInfo bestMoveInfo = null;
      using (new SearchContextExecutionBlock(searchResult.Manager.Context))
      {
        bestMoveInfo = searchResult.Manager.Root.BestMoveInfo(false);

#if NOT
        bool wouldBeDrawByRepetition = PositionRepetitionCalc.DrawByRepetitionWouldBeClaimable(curPositionAndMoves.FinalPosition, bestMoveInfo.BestMove, curPositionAndMoves.GetPositions());
        if (wouldBeDrawByRepetition)
        {
        }
#endif
      }

      scoreCeresCP = (int)MathF.Round(EncodedEvalLogistic.WinLossToCentipawn(bestMoveInfo.QOfBest), 0);


      MGMove bestMoveMG = searchResult.BestMove;

      int N = searchResult.SearchRootNode.N;

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

      using (new SearchContextExecutionBlock(result.Search.Manager.Context))
      {

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

        if (VerboseMoveStats)
        {
          result.Search.Manager.Context.Root.Dump(1, 1);
        }
      }

      return result;
    }

    public override void Warmup(int? knownMaxNumNodes = null)
    { 
      PrepareEvaluators();
    }

    void PrepareEvaluators()
    {
      if (evaluators == null)
      {
        evaluators = new NNEvaluatorSet(EvaluatorDef, SearchParams.Execution.FlowDirectOverlapped, EvaluatorDefSecondary);
        evaluators.Warmup(false);
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
                                                 bool verbose)
    {
      PositionEvalCache positionCacheOpponent = null;

      Search = new MCTSearch();

      if (LastSearch == null)
      {
        PrepareEvaluators();
        Search.Search(evaluators, ChildSelectParams, SearchParams, GameLimitManager,
                      reuseOtherContextForEvaluatedNodes,
                      curPositionAndMoves, searchLimit, verbose, lastSearchStartTime,
                      gameMoveHistory, callback, null, reuseNodeCache, false, isFirstMoveOfGame,
                      MoveImmediateIfOnlyOneMove);
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
        using (new SearchContextExecutionBlock(LastSearch.Manager.Context))
        {
          int currentN = LastSearch.SearchRootNode.N;
        }

        // Determine threshold to decide if tree is big enough to be reused.
        // Extraction time is about linear in number of nodes extracted (averages about 2.5mm/sec)
        // except for a fixed time required to build BitArray of included nodes (which is only about 2% of total time).
        // Therefore we almost always reuse the tree, unless extremely small.
        float THRESHOLD_FRACTION_NODES_REUSABLE = 0.03f;

        Search.SearchContinue(LastSearch, reuseOtherContextForEvaluatedNodes,
                                     forwardMoves, curPositionAndMoves,
                                     gameMoveHistory, searchLimit, verbose, lastSearchStartTime,
                                     callback, THRESHOLD_FRACTION_NODES_REUSABLE,
                                     isFirstMoveOfGame, MoveImmediateIfOnlyOneMove);
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
                                                LastSearch.Manager.SearchLimit.Value, 0, null, null, 0, 0);
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
    public List<LC0VerboseMoveStat> GetVerboseMoveStats()
    {
      if (Search == null)
      {
        throw new Exception("GetVerboseMoveStats cannot return search statistics because no search has run yet.");
      }

      using (new SearchContextExecutionBlock(Search.Manager.Context))
      {
        return LC0VerboseMoveStatsFromMCTSNode.BuildStats(Search.SearchRootNode);
      }
    }



    /// <summary>
    /// Diposes underlying search engine.
    /// </summary>
    public override void Dispose()
    {
      evaluators?.Dispose();
      evaluators = null;

      Search?.Manager.Dispose();
      Search = null;

      LastSearch?.Manager.Dispose();
      LastSearch = null;
      reuseNodeCache = null;
    }

  }
}
