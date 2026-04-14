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
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;

using Ceres.Base.Benchmarking;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;

using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.UserSettings;

using Ceres.MCGS.UCI;

using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Utils;
using Ceres.MCGS.Visualization.AnalysisGraph;
using static Ceres.MCGS.Search.Phases.MCGSSelect;

#endregion

namespace Ceres.MCGS.GameEngines;

/// <summary>
/// Subclass of GameEngine specialized for Ceres MCGSEngine (running in-process).
/// </summary>
public class GameEngineCeresMCGSInProcess : GameEngine
{
  public const string CERES_MCGS_VERSION_STR = "2.30";

  /// <summary>
  /// Definition of neural network evaluator used for execution.
  /// </summary>
  public readonly NNEvaluatorDef EvaluatorDef;

  /// <summary>
  /// General search parameters used.
  /// </summary>
  public readonly ParamsSearch SearchParams;

  /// <summary>
  /// MCTS leaf selection parameters used.
  /// </summary>
  public readonly ParamsSelect SelectParams;

  /// <summary>
  /// Manager used for apportioning node or time limits at the game
  /// level to individual moves.
  /// </summary>
  public IManagerGameLimit GameLimitManager;

  /// <summary>
  /// Search in progress or last concluded, if any.
  /// </summary>
  public MCGSSearch Search;

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

  public WorkerPool<ExtendPathsWorkerInfo>[] SelectWorkerPools = new WorkerPool<ExtendPathsWorkerInfo>[2];


  public bool DisposeGraphAfterSearch;

  /// <summary>
  /// Optional fixed search limit known at engine creation time.
  /// When specified, allows optimizations for small searches.
  /// </summary>
  public readonly SearchLimit FixedSearchLimit;


  #region Internal data

  /// <summary>
  /// Once created the NN evaluator pair is reused (until Dispose is called).
  /// </summary>
  public NNEvaluatorSet Evaluators { get; private set; }


  readonly BestMoveInfoMCGS lastBestMoveInfo;

  readonly Action<string> InfoLogger;

  #endregion


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="id">identifying string</param>
  /// <param name="evaluatorDef">primary evaluator for all nodes</param>
  /// <param name="evaluatorDefSecondary">optional secondary evaluator to be run on subset of tree</param>
  /// <param name="searchParams">optional non-default search parameters</param>
  /// <param name="selectParams">optional non-default child selection parameters </param>
  /// <param name="gameLimitManager">optional override manager for search limits</param>
  /// <param name="logFileName">optional name of file to which to write detailed log</param>
  /// <param name="moveImmediateIfOnlyOneMove">if engine should chose best move immediately without search if only one legal move</param>
  /// <param name="processorGroupID">id of processor group on which engine should execute</param>
  /// <param name="disposeGraphAfterSearch">if the graph should be disposed after each search</param>
  /// <param name="infoLogger">optional action to log info messages</param>
  /// <param name="forcedMoves">optional list of moves to force</param>
  /// <param name="fixedSearchLimit">optional fixed search limit known at engine creation time</param>
  public GameEngineCeresMCGSInProcess(string id,
                                      NNEvaluatorDef evaluatorDef,
                                      ParamsSearch searchParams = null,
                                      ParamsSelect selectParams = null,
                                      IManagerGameLimit gameLimitManager = null,
                                      string logFileName = null,
                                      bool moveImmediateIfOnlyOneMove = true,
                                      int processorGroupID = 0,
                                      bool disposeGraphAfterSearch = true,
                                      Action<string> infoLogger = null,
                                      List<MGMove> forcedMoves = null,
                                      SearchLimit fixedSearchLimit = null) : base(id, processorGroupID)
  {
    // Use default settings for search and select params if not specified.
    if (searchParams == null)
    {
      searchParams = new ParamsSearch();
    }

    if (selectParams == null)
    {
      selectParams = new ParamsSelect();
    }

    // Optimization: disable dual evaluators/iterators for small searches
    // since the overhead is not worth it. This must be done before storing SearchParams
    // because MCGSManager will check these flags.
    if (ShouldDisableDualEvaluatorsForLimit(fixedSearchLimit))
    {
      searchParams = searchParams with
      {
        Execution = searchParams.Execution with
        {
          DualOverlappedIterators = false,
          DualEvaluators = false
        }
      };
    }

    gameLimitManager = InitializeGameLimitManager(searchParams, gameLimitManager);

    EvaluatorDef = evaluatorDef ?? throw new ArgumentNullException(nameof(evaluatorDef));
    SearchParams = searchParams;
    GameLimitManager = gameLimitManager;
    SelectParams = selectParams;
    SearchLogFileName = logFileName;
    MoveImmediateIfOnlyOneMove = moveImmediateIfOnlyOneMove;
    DisposeGraphAfterSearch = disposeGraphAfterSearch;
    OutputVerboseMoveStats = CeresUserSettingsManager.Settings.VerboseMoveStats;
    InfoLogger = infoLogger;
    ForcedMoves = forcedMoves;
    FixedSearchLimit = fixedSearchLimit;

    if (logFileName == null && !string.IsNullOrEmpty(CeresUserSettingsManager.Settings.SearchLogFile))
    {
      SearchLogFileName = CeresUserSettingsManager.Settings.SearchLogFile;
    }

    PrepareEvaluators();
    Warmup();
  }

  private static IManagerGameLimit InitializeGameLimitManager(ParamsSearch searchParams, IManagerGameLimit gameLimitManager)
  {
    // Use default limit manager if not specified.
    if (gameLimitManager == null)
    {
      // Check for alternate limits manager specified in Ceres settings.
      string altManager = CeresUserSettingsManager.Settings.LimitsManagerName;
      if (altManager == null)
      {
        gameLimitManager = new ManagerGameLimitCeresMCGS(searchParams.GameLimitUsageAggressiveness);
      }
      else
      {
        if (altManager.ToUpper(CultureInfo.InvariantCulture) == "TEST")
        {
          gameLimitManager = new ManagerGameLimitTest(searchParams.GameLimitUsageAggressiveness);
        }
        else
        {
          throw new NotImplementedException(altManager + " not supported for setting AlternateLimitsManagerName");
        }
      }
    }

    return gameLimitManager;
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
    Search?.Manager.Engine.Graph.Dispose();

    Search = null;

    isFirstMoveOfGame = true;
    CurrentGameID = gameID;
  }


  /// <summary>
  /// Executes any preparatory steps (that should not be counted in thinking time) before a search.
  /// </summary>
  protected override void DoSearchPrepare()
  {
  }


  static readonly Lock logFileWriteObj = new();




  /// <summary>
  /// Runs a search, calling DoSearch and adjusting the cumulative search time
  /// (convenience method with same functionality but returns the as the subclass
  /// GameEngineSearchResultCeres.
  /// </summary>
  /// <param name="curPositionAndMoves"></param>
  /// <param name="searchLimit"></param>
  /// <param name="callback"></param>
  /// <returns></returns>
  public GameEngineSearchResultCeresMCGS SearchCeres(PositionWithHistory curPositionAndMoves,
                                                 SearchLimit searchLimit,
                                                 List<GameMoveStat> gameMoveHistory = null,
                                                 ProgressCallback callback = null,
                                                 bool verbose = false)
  {
    return Search(curPositionAndMoves, searchLimit, gameMoveHistory, callback, verbose) as GameEngineSearchResultCeresMCGS;
  }


  bool haveWarnedPrefetch = false;


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
    if (SearchParams.PrefetchParams != null && searchLimit.IsNodesLimit && haveWarnedPrefetch)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "WARNING: With Prefetching enabled the NodesLimit is not inclusive of prefetched nodes.");
    }

    // Validate that the search limit is compatible with FixedSearchLimit (if specified).
    if (FixedSearchLimit != null 
        && FixedSearchLimit.IsNodesLimit 
        && searchLimit.IsNodesLimit
        && searchLimit.Value > FixedSearchLimit.Value)
    {
      throw new InvalidOperationException($"Search limit ({searchLimit.Value} nodes) is incompatible with FixedSearchLimit ({FixedSearchLimit.Value} nodes) ");
    }

    // Possibly set a forced move if a list of such moves was provided and is not yet exhausted.
    MGMove forcedMove = (ForcedMoves == null || ForcedMoves.Count < curPositionAndMoves.Count)
                          ? default
                          : ForcedMoves[curPositionAndMoves.Count - 1];

    // Set max tree nodes to maximum value based on memory (unless explicitly overridden in passed SearchLimit)
    searchLimit = searchLimit with
    {
      MaxTreeVisits = searchLimit.MaxTreeVisits ?? MCGSParamsFixed.MAX_VISITS,
    };

    // Set up callback passthrough if provided
    MCGSManager.MCGSProgressCallback callbackMCGS = null;
    if (callback != null)
    {
      callbackMCGS = callbackContext => callback((MCGSManager)callbackContext);
    }

    // Attempt to initialize for graph reuse
    // (this might not succeed first time if we or opponent is not yet initialized).
    PossiblyInitializeForOpponentGraphReuse();

    void InnerCallback(MCGSManager manager)
    {
      // Check for possible externally enqueued command.
      (string command, string options) = InterprocessCommandManager.TryDequeuePendingCommand();
      if (command != default)
      {
        AnalysisGraphOptions optionsObj = AnalysisGraphOptions.FromString(options);
        Console.WriteLine($"Writing Analysis Graph (detail level {optionsObj.DetailLevel})");
        throw new Exception("AnalysisGraphGenerator constructor below needs remeidation to use MCGSSearch as argument, not null");
        AnalysisGraphGenerator graphGenerator = new AnalysisGraphGenerator(null, optionsObj);
        graphGenerator.Write(true);
      }
      callbackMCGS?.Invoke(manager);
    }

    // Run the search
    MCGSSearch searchResult;
    TimingStats searchTimingStats = new();
    using (new TimingBlock(searchTimingStats, TimingBlock.LoggingType.None))
    {
      searchResult = RunSearchPossiblyTreeReuse(curPositionAndMoves, gameMoveHistory,
                                                           searchLimit, InnerCallback,
                                                           infoMsg => InfoLogger?.Invoke(infoMsg),
                                                           verbose, forcedMove);
    }
    isFirstMoveOfGame = false;

    int scoreCeresCP;
    BestMoveInfoMCGS bestMoveInfo = searchResult.Manager.GetBestMove(out GEdge bestChild,
                                                                     out GNode bestMoveNode,
                                                                     out MGMove bestMove, true);

#if NOT
      bool wouldBeDrawByRepetition = PositionRepetitionCalc.DrawByRepetitionWouldBeClaimable(curPositionAndMoves.FinalPosition, bestMoveInfo.BestMove, curPositionAndMoves.GetPositions());
      if (wouldBeDrawByRepetition)
      {
      }
#endif

    // TODO:  bestMoveInfo is not used, removed?
    //      MGMove bestMoveMG = searchResult.BestMove;
    // TODO is the RootNWhenSearchStarted correct because we may be following a continuation (BestMoveRoot)
    //      string moveStr = bestMoveMG.MoveStr(MGMoveNotationStyle.Coordinates);

    MGMove bestMoveMG = bestMoveInfo.BestMove;
    // TODO is the RootNWhenSearchStarted correct because we may be following a continuation (BestMoveRoot)
    string moveStr = bestMoveMG.MoveStr(MGMoveNotationStyle.Coordinates);
    scoreCeresCP = (int)MathF.Round(EncodedEvalLogistic.WinLossToCentipawn(bestMoveInfo.QOfBest), 0);
    int eps = 0;
    int depth = 0;
    //this..NumEvalsThisSearch / elapsedTimeSeconds; // positions evaluated per second

    GameEngineSearchResultCeresMCGS result = new(Search, moveStr, bestMoveMG, (float)Search.Manager.Engine.SearchRootNode.Q, bestMoveInfo.QOfBest,
                                                 scoreCeresCP, 0,
                                                 searchLimit, searchTimingStats,
                                                 Search.StartSearchN, Search.Manager.Engine.SearchRootNode.N,
                                                 eps, depth,
                                                 bestMoveInfo, Search.Manager.Engine.Graph.RatioVisitsToNodes);

    // Append search result information to log file (if any).
    StringWriter dumpInfo = new();
    if (SearchLogFileName != null)
    {
      result.Search.Manager.DumpFullInfo(result.BestMoveInfo, result.Search.SearchRootNode,
                                         result.Search.Manager.LastGameLimitInputs,
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
      Console.WriteLine("NOTE: pending fix in GameEngineCeresMCGSInProcessNEW");
      //UCIManagerMCGS.OutputVerboseMoveStats(result.BestMoveInfo);
    }

    return result;
  }


  bool haveEstablishedOpponentGraphReuse = false;

  private void PossiblyInitializeForOpponentGraphReuse()
  {
    // Possibly use the context of opponent to reuse position evaluations
    if (OpponentEngine is not null && !haveEstablishedOpponentGraphReuse)
    {
      if (Search is not null && Search.Manager.ParamsSearch.ReusePositionEvaluationsFromOtherGraph)
      {
        GameEngineCeresMCGSInProcess ceresOpponentEngine = OpponentEngine as GameEngineCeresMCGSInProcess;
        if (ceresOpponentEngine is null)
        {
          throw new Exception($"ReusePositionEvaluationsFromOtherTree not possible: Opponent engine of type {OpponentEngine.GetType()} not GameEngineCeresMCGSInProcess.");
        }

        bool evaluatorDefsCompatible = EvaluatorDef.NetEvaluationsIdentical(ceresOpponentEngine.EvaluatorDef);

        if (evaluatorDefsCompatible)
        {
          // The Graph from the other engine may not be static (graph rebuilding).
          // Therefore use a delegate to get the current graph.
          Search.Manager.Engine.Graph.ReuseGraphProvider = () => ceresOpponentEngine.Search.Manager.Engine.Graph;
          haveEstablishedOpponentGraphReuse = true;

          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "NOTE: NN evaluations will be shared across engines (ReusePositionEvaluationsFromOtherTree).");
        }
        else
        {
          Console.WriteLine("Engine  : " + EvaluatorDef);
          Console.WriteLine("Opponent: " + ceresOpponentEngine.EvaluatorDef);
          throw new NotImplementedException("ReusePositionEvaluationsFromOtherTree not possible; opponent engine evaluator definition not compatible.");
        }
      }
    }
  }

  public override void Warmup(int? knownMaxNumNodes = null)
  {
    Evaluators.Warmup(knownMaxNumNodes ?? int.MaxValue);
  }


  /// <summary>
  /// Determines if dual evaluators/iterators should be disabled based on a search limit.
  /// Small searches (below THRESHOLD_BEGIN_OVERLAPPING) don't benefit from dual evaluators.
  /// </summary>
  /// <param name="searchLimit">The search limit to check (can be null).</param>
  /// <returns>True if the search is small enough to disable dual evaluators.</returns>
  internal static bool ShouldDisableDualEvaluatorsForLimit(SearchLimit searchLimit)
  {
    return searchLimit != null
           && searchLimit.IsNodesLimit
           && searchLimit.Value < ParamsSearchExecutionChooser.THRESHOLD_BEGIN_OVERLAPPING;
  }


  void PrepareEvaluators()
  {
    if (Evaluators == null)
    {
      // Use SearchParams.Execution.DualEvaluators which was already adjusted
      // in the constructor based on FixedSearchLimit.
      Evaluators = new NNEvaluatorSet(EvaluatorDef, SearchParams.Execution.DualEvaluators, null);
      if (overrideEvaluator1 != null)
      {
        Evaluators.OverrideEvaluators(overrideEvaluator1, overrideEvaluator2, overrideEvaluatorSecondary);
      }
    }
  }

  NNEvaluator overrideEvaluator1;
  NNEvaluator overrideEvaluator2;
  NNEvaluator overrideEvaluatorSecondary;


  /// <summary>
  /// Overrides the evaluators used for the search.
  /// This will cause the NNEvaluatorDefs (EvaluatorDef, Evaluator2Def and EvaluatorDefSecondary) to be ignored.
  /// Intended mainly for testing purposes.
  /// </summary>
  /// <param name="evaluator1"></param>
  /// <param name="evaluator2"></param>
  /// <param name="evaluatorSecondary"></param>
  public void OverrideEvaluators(NNEvaluator evaluator1, NNEvaluator evaluator2, NNEvaluator evaluatorSecondary)
  {
    overrideEvaluator1 = evaluator1;
    overrideEvaluator2 = evaluator2;
    overrideEvaluatorSecondary = evaluatorSecondary;
  }


  /// <summary>
  /// Launches search, possibly as continuation from last search.
  /// </summary>
  /// <param name="curPositionAndMoves"></param>
  /// <param name="gameMoveHistory"></param>
  /// <param name="searchLimit"></param>
  /// <param name="callback"></param>
  /// <param name="infoLogger"></param>
  /// <param name="verbose"></param>
  /// <param name="forcedMove"></param>
  /// <returns></returns>
  private MCGSSearch RunSearchPossiblyTreeReuse(PositionWithHistory curPositionAndMoves,
                                                List<GameMoveStat> gameMoveHistory,
                                                SearchLimit searchLimit,
                                                MCGSManager.MCGSProgressCallback callback,
                                                MCGSSearch.MCGSInfoLogger infoLogger,
                                                bool verbose,
                                                MGMove forcedMove)
  {
    Graph reuseGraph = Search?.Manager.Engine.Graph;

    Search?.Manager.Dispose();
    Search = new MCGSSearch(infoLogger);

    Search.Search(Evaluators, reuseGraph, SelectWorkerPools,
                  SelectParams, SearchParams, GameLimitManager,
                  curPositionAndMoves, searchLimit, verbose, lastSearchStartTime,
                  gameMoveHistory, callback, null, isFirstMoveOfGame,
                  MoveImmediateIfOnlyOneMove, forcedMove: forcedMove,
                  fixedSearchLimit: FixedSearchLimit);
    return Search;
  }


#if NOT
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
#endif


  /// <summary>
  /// Returns UCI information string 
  /// (such as would appear in a chess GUI describing search progress) 
  /// based on last state of search.
  /// </summary>
  public override UCISearchInfo UCIInfo => Search != null ? new(UCIInfoMCGS.UCIInfoString(Search.Manager)) : null;



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

    return VerboseMoveStatsFromMCGSNode.BuildStats(Search.Manager, lastBestMoveInfo);
  }



  /// <summary>
  /// Diposes underlying search engine.
  /// </summary>
  public override void Dispose()
  {
    Search?.Manager.Engine.Graph.Dispose();
    Search?.Manager?.Dispose();
    Search = null;
    Evaluators?.Dispose();
    Evaluators = null;
    SelectWorkerPools[0]?.Dispose();
    SelectWorkerPools[1]?.Dispose();
  }


  public void DumpStoreUsageSummary()
  {
    if (Search != null)
    {
      GraphStore store = Search.Manager.Engine.Graph.Store;
      Console.WriteLine("Store used item counts " + store.NodesStore.NumUsedNodes);
      float nodes = store.NodesStore.NumUsedNodes;
      store.DumpUsageSummary();
    }
  }
}
