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
using System.Text;
using System.Threading;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.UserSettings;

using Ceres.MCGS.Analysis;
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
using Ceres.MCGS.Search.Paths;
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
  /// The result of the most recently completed search (used for post-hoc diagnostic dumps).
  /// </summary>
  public GameEngineSearchResultCeresMCGS LastSearchResult { get; private set; }

  /// <summary>
  /// Set asynchronously (via RequestDumpInfo) to request that this engine dump its current search
  /// diagnostics to the console. Consumed and cleared at the next safe quiescent point during the
  /// running search, or at search end if the search finishes first.
  /// </summary>
  volatile bool dumpInfoRequested;

  /// <summary>
  /// Caller-supplied label identifying what requested the pending dump (forwarded as the dump
  /// description and shown in its header, exactly like the "UCI"/"AUTO" descriptions). Set by
  /// RequestDumpInfo before the request flag, so the consuming thread observes it.
  /// </summary>
  string dumpInfoDescription = "DUMP-INFO";

  /// <summary>
  /// Output format for a requested diagnostics dump. Set by RequestDumpInfo before the request flag.
  /// </summary>
  public enum DumpInfoFormat
  {
    /// <summary>Plain dump (yellow header followed by the full search info).</summary>
    Plain,

    /// <summary>
    /// Dump wrapped as a "dump-info-block": prefixed with a process/GC/machine header and bracketed
    /// by begin/end markers (see <see cref="DiagnosticsBlock"/>) for clean programmatic capture.
    /// </summary>
    Block
  }

  /// <summary>
  /// Format requested for the pending dump. Set by RequestDumpInfo before the request flag.
  /// </summary>
  DumpInfoFormat dumpInfoFormat = DumpInfoFormat.Plain;

  /// <summary>
  /// Serializes diagnostic dumps so concurrent engines (multi-threaded tournaments) do not
  /// interleave their output on the console. Shared with the UCI-engine dump wrapper so in-process
  /// and external-UCI engine dumps also do not interleave.
  /// </summary>
  static readonly object dumpConsoleLock = DiagnosticsBlock.ConsoleLock;

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

  /// <summary>
  /// Per-iterator pools of path visit slots. Owned here (rather than by the per-search
  /// MCGSEngine) so the multi-megabyte buffers are allocated once and reused by every search.
  /// </summary>
  public ArraySegmentPool<MCGSPathVisit>[] PathVisitPools = new ArraySegmentPool<MCGSPathVisit>[2];

  /// <summary>
  /// Per-evaluator neural network input batches. Owned here (rather than by the per-search
  /// evaluator wrapper) so that the plane / history arrays grown during one search are reused
  /// by the next instead of being rebuilt (and discarded) on every move.
  /// </summary>
  public EncodedPositionBatchFlat[] ReusableNNBatches = new EncodedPositionBatchFlat[2];


  public bool DisposeGraphAfterSearch;

  /// <summary>
  /// Optional fixed search limit known at engine creation time.
  /// When specified, allows optimizations for small searches.
  /// </summary>
  public readonly SearchLimit FixedSearchLimit;

  /// <summary>
  /// If true this engine emits a per-tournament diagnostic "minilog" file (default false).
  /// The tournament path enables logging explicitly via InitMiniLog; this flag additionally
  /// drives standalone (non-tournament) lazy initialization on the first search.
  /// </summary>
  public readonly bool EmitMiniLog;

  /// <summary>
  /// Active diagnostic minilog writer (null unless logging is enabled).
  /// </summary>
  MCGSMiniLog miniLog;

  /// <summary>
  /// Number of move lines written to the minilog for the game currently in progress.
  /// </summary>
  int miniLogMovesThisGame;

  /// <summary>
  /// Whether a result footer has already been written for the game currently in progress.
  /// The tournament supplies a full footer of its own (MiniLogWriteGameResult); this flag stops
  /// the synthetic standalone footer from duplicating it.
  /// </summary>
  bool miniLogFooterWritten;

  /// <summary>
  /// Whether the "=== NEW GAME ===" separator for the game in progress still needs to be written.
  /// The separator is emitted lazily, immediately before that game's first content, rather than at ResetGame.
  /// </summary>
  bool miniLogSeparatorPending = true;

  /// <summary>
  /// Per-game accumulator behind the minilog "=== TIME USAGE ===" summary section: aggregates
  /// per-move allocation vs. consumption and search stop reasons so end-of-game time usage
  /// can be characterized (how much of the per-move budgets went unused, and why).
  /// </summary>
  sealed class MiniLogTimeUsageStats
  {
    public int NumTimedMoves;
    public double SumAllocSec;
    public double SumElapsedSec;
    public readonly List<double> BudgetFracsPct = new();
    public int StopFutility, StopTimeLimit, StopOnlyMove, StopTablebase, StopInstamove, StopOther;
    public int NumMovesUnderIncrement;
    public float MinPostMoveClock = float.NaN;
    public double FirstMoveElapsed = double.NaN;

    public void Reset()
    {
      NumTimedMoves = 0;
      SumAllocSec = 0;
      SumElapsedSec = 0;
      BudgetFracsPct.Clear();
      StopFutility = StopTimeLimit = StopOnlyMove = StopTablebase = StopInstamove = StopOther = 0;
      NumMovesUnderIncrement = 0;
      MinPostMoveClock = float.NaN;
      FirstMoveElapsed = double.NaN;
    }
  }

  /// <summary>
  /// Accumulator for the current game's minilog time-usage summary (reset per game).
  /// </summary>
  readonly MiniLogTimeUsageStats miniLogTimeUsage = new();

  /// <summary>
  /// Time remaining on the tournament clock (in seconds) at the start of the current move, or
  /// null when not playing in a timed tournament. Set by the tournament before each search and
  /// logged as "TimeRem" on the move line.
  /// </summary>
  public float? TournamentClockRemainingSeconds { get; set; }

  /// <summary>
  /// Time remaining on the tournament clock (in seconds) for the OPPONENT at the start of the
  /// current move, or null when not playing in a timed tournament (or the opponent's clock is not
  /// known, e.g. an external/UCI opponent). Set by the tournament before each search and logged as
  /// "OppTimeRem" on the move line. Exposed so the engine can (now or in future) reason about the
  /// opponent's remaining time. Only populated when the opponent is driven by this same in-process
  /// tournament (a Ceres tournament).
  /// </summary>
  public float? TournamentClockRemainingOpponentSeconds { get; set; }


  #region Internal data

  /// <summary>
  /// Once created the NN evaluator pair is reused (until Dispose is called).
  /// </summary>
  public NNEvaluatorSet Evaluators { get; private set; }


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
  /// <param name="emitMiniLog">if a per-tournament diagnostic minilog file should be written</param>
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
                                      SearchLimit fixedSearchLimit = null,
                                      bool emitMiniLog = false) : base(id, processorGroupID)
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
    EmitMiniLog = emitMiniLog;

    // If a diagnostic minilog will be emitted, have the limit manager capture its per-move
    // allocation reasoning so it can be recorded near each move header (see BuildMiniLogMoveLine).
    if (GameLimitManager != null && emitMiniLog)
    {
      GameLimitManager.CaptureDiagnostics = true;
    }

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
      // Check for an alternate limits manager: the per-engine ParamsSearch setting takes
      // precedence over the process-wide Ceres settings (so two engines in one tournament
      // can A/B different managers).
      string altManager = searchParams.LimitsManagerName;
      if (string.IsNullOrEmpty(altManager))
      {
        altManager = CeresUserSettingsManager.Settings.LimitsManagerName;
      }

      if (string.IsNullOrEmpty(altManager))
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
  /// Starting time control of the current game (seconds), captured from the first time-based search
  /// limit seen since the last ResetGame, or null if none yet seen (or the limit is not time-based).
  /// Used to resolve the move overhead in effect for the game (see MCGSSearch), which must be based
  /// on the starting time control rather than the remaining clock (which shrinks as the game plays out).
  /// </summary>
  float? gameStartingTimeLimitSeconds;

  /// <summary>
  /// Resets all state between games.
  /// </summary>
  /// <param name="gameID">optional game descriptive string</param>
  public override void ResetGame(string gameID = null)
  {
    Search?.Manager.Engine.Graph.Dispose();

    Search = null;

    isFirstMoveOfGame = true;
    gameStartingTimeLimitSeconds = null;

    MiniLogFinishOpenGame();

    CurrentGameID = gameID;

    miniLogSeparatorPending = true;
    miniLogMovesThisGame = 0;
    miniLogFooterWritten = false;
    miniLogTimeUsage.Reset();
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

    // Capture the starting time control of this game (the first time-based limit seen since ResetGame).
    // For a per-game clock this is the full time control, since no time has yet been consumed;
    // for a per-move limit it is that per-move allotment.
    if (gameStartingTimeLimitSeconds is null && searchLimit.IsTimeLimit && searchLimit.Value > 0)
    {
      gameStartingTimeLimitSeconds = searchLimit.Value;
    }

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
      // Honor any pending diagnostics-dump request. This callback fires at a quiescent point
      // (holding the backup lock with no other iterator in its select/backup phase), so reading
      // the graph and dumping here is safe.
      if (dumpInfoRequested)
      {
        dumpInfoRequested = false;
        DumpDiagnosticsWithHeader(manager, liveMidSearch: true);
      }

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
    // Evaluations per second (neural network position evaluations) made during this search.
    int eps = searchTimingStats.ElapsedTimeSecs > 0
            ? (int)MathF.Round(Search.Manager.NumEvalsThisSearch / (float)searchTimingStats.ElapsedTimeSecs)
            : 0;
    int depth = 0;

    GameEngineSearchResultCeresMCGS result = new(Search, moveStr, bestMoveMG, (float)Search.Manager.Engine.SearchRootNode.Q, bestMoveInfo.QOfBest,
                                                 scoreCeresCP, 0,
                                                 searchLimit, searchTimingStats,
                                                 Search.StartSearchN, Search.Manager.Engine.SearchRootNode.N,
                                                 eps, depth,
                                                 bestMoveInfo, Search.Manager.Engine.Graph.RatioVisitsToNodes);

    // Retain the most recent search result so diagnostics can be dumped post-hoc (e.g. blunder analysis).
    LastSearchResult = result;

    // If a diagnostics dump was requested but the search completed before the next quiescent
    // callback could fire (e.g. a very short search), honor it now against the just-completed search.
    if (dumpInfoRequested)
    {
      dumpInfoRequested = false;
      DumpDiagnosticsWithHeader(Search.Manager, liveMidSearch: false);
    }

    // If configured, always emit the full search info dump after every completed search. This lives
    // at the GameEngine level (below UCI) so it happens for ALL callers -- UCI, tournaments, suites,
    // and direct programmatic searches -- exactly as if the "dump-info" command had been issued.
    if (MCGSParamsFixed.ALWAYS_DUMP_SEARCH_INFO)
    {
      result.Search.Manager.DumpFullInfo(result, Console.Out, "AUTO");

      // Also run and display a revaluation analysis, exactly as if a "revalue-root N" command
      // had been issued with N scaled to the search size. Analysis only (the best move above is
      // already final); note the rollout visits do grow the graph, like any deep-rollout command.
      MCGSManager revalManager = result.Search.Manager;
      int revalRoundsPerStage = Math.Max(1, revalManager.Engine.SearchRootNode.N / 20);
      PrincipalRevaluationResult reval = PrincipalRevaluation.Run(revalManager, revalRoundsPerStage);
      PrincipalRevaluationDumper.DumpToConsole(reval, bestMoveMG);

      // Purely informational: report whether the rollout evidence would prefer a different move.
      RevaluationSwitchDecision revalDecision = PrincipalRevaluation.CalcBlendedQSwitchDecision(
          revalManager, revalManager.Engine.SearchRootNode, bestMoveInfo, reval);
      Console.WriteLine(revalDecision.WouldSwitch
        ? $"info string reval decision: rollout evidence would prefer {revalDecision.CandidateMove} over {revalDecision.BaselineMove} ({revalDecision.Description})"
        : $"info string reval decision: rollout evidence keeps {revalDecision.BaselineMove} ({revalDecision.Description})");
    }

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

    // Emit the compact per-move diagnostic minilog line (if enabled). For standalone (non-tournament)
    // use, lazily initialize on the first search using an auto-derived file name. The write is wrapped
    // so a logging failure can never disrupt the game.
    if (miniLog == null && EmitMiniLog)
    {
      InitMiniLog(AutoMiniLogFileName(), FixedSearchLimit ?? searchLimit);
    }
    if (miniLog != null)
    {
      try
      {
        // Emit the limits-manager allocation reasoning (if captured) immediately before the move
        // line it governs, so it sits next to that move header (mirrors the inline blunder block).
        MiniLogEnsureGameStarted();
        miniLog.AppendLimitsSection(result.Search.Manager.LastGameLimitOutputs?.DiagnosticText);
        miniLog.WriteMoveLine(BuildMiniLogMoveLine(result, bestMoveInfo));
        UpdateMiniLogTimeUsage(result);
        miniLogMovesThisGame++;
      }
      catch (Exception exc)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Minilog write failed: " + exc.Message);
      }
    }

    if (GatherVerboseMoveStats)
    {
      result.VerboseMoveStats = GetVerboseMoveStats(result.BestMoveInfo);
    }

    if (OutputVerboseMoveStats)
    {
      Console.WriteLine("NOTE: pending fix in GameEngineCeresMCGSInProcessNEW");
      //UCIManagerMCGS.OutputVerboseMoveStats(result.BestMoveInfo);
    }

    return result;
  }


  /// <summary>
  /// Requests that this engine asynchronously dump its current search diagnostics to the console.
  /// The dump is emitted at the next safe quiescent point during the running search (typically
  /// within ~0.5s), or at the end of the current search if it finishes sooner. Safe to call from
  /// another thread. The description identifies the requester (e.g. "UCI", "AUTO") and appears in
  /// the dump header; the engine itself is agnostic to who or what triggered the request.
  /// When format is Block the dump is emitted as a marker-delimited dump-info-block (with a
  /// process/GC/machine header) suitable for programmatic capture.
  /// </summary>
  public void RequestDumpInfo(string description = "DUMP-INFO", DumpInfoFormat format = DumpInfoFormat.Plain)
  {
    dumpInfoDescription = description;
    dumpInfoFormat = format;
    dumpInfoRequested = true;
  }


  /// <summary>
  /// Emits a yellow header (identifying this engine and the current move) followed by the full
  /// search diagnostics dump. When liveMidSearch is true the dump reflects the in-progress search
  /// (so it must only be called at a quiescent point where reading the graph is safe); otherwise
  /// it dumps the most recently completed search. Output is serialized across engines via
  /// dumpConsoleLock, and the whole operation is guarded so a dump failure cannot disrupt search.
  /// </summary>
  void DumpDiagnosticsWithHeader(MCGSManager manager, bool liveMidSearch)
  {
    string description = dumpInfoDescription;
    DumpInfoFormat format = dumpInfoFormat;
    try
    {
      lock (dumpConsoleLock)
      {
        if (format == DumpInfoFormat.Block)
        {
          // Wrap the dump in begin/end markers plus a process/GC/machine header so a consumer
          // (e.g. the tournament manager driving this engine over UCI) can capture it cleanly.
          DiagnosticsBlock.WriteBlock(Console.Out, w => WriteDiagnosticsBody(manager, liveMidSearch, description, w));
        }
        else
        {
          WriteDiagnosticsBody(manager, liveMidSearch, description, Console.Out);
        }
      }
    }
    catch (Exception e)
    {
      Console.WriteLine("Search diagnostics dump failed: " + e.Message);
    }
  }


  /// <summary>
  /// Writes the diagnostics body (a header line identifying this engine and the current move,
  /// followed by the full search info dump) to the specified writer. Must be called while holding
  /// dumpConsoleLock; when liveMidSearch is true it must only be called at a quiescent point.
  /// </summary>
  void WriteDiagnosticsBody(MCGSManager manager, bool liveMidSearch, string description, TextWriter writer)
  {
    int moveNum;
    try
    {
      int priorPlies = manager.Engine.SearchRootNode.Graph.Store.HistoryHashes.PriorPositionsMG.Length;
      moveNum = 1 + priorPlies / 2;
    }
    catch
    {
      moveNum = 0;
    }

    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
      $"===== {description}  engine={ID}  move={moveNum} =====");

    string phase = liveMidSearch ? " (in-search)" : " (search end)";
    if (liveMidSearch)
    {
      // Non-final, read-only best-move peek (we are not finalizing a move, just reporting).
      BestMoveInfoMCGS bestMoveInfo = manager.GetBestMove(out _, out _, out _, isFinalBestMoveCalc: false);
      manager.DumpFullInfo(bestMoveInfo, manager.Engine.SearchRootNode, default, writer, description + phase);
    }
    else if (LastSearchResult?.Search?.Manager != null)
    {
      LastSearchResult.Search.Manager.DumpFullInfo(LastSearchResult, writer, description + phase);
    }
  }


  /// <summary>
  /// Dumps detailed diagnostics about the most recently completed search to the specified writer.
  /// Returns false if no search has yet completed for this engine.
  /// </summary>
  public override bool TryDumpLastSearchDiagnostics(TextWriter writer, string description)
  {
    GameEngineSearchResultCeresMCGS result = LastSearchResult;
    if (result?.Search?.Manager == null)
    {
      return false;
    }

    result.Search.Manager.DumpFullInfo(result, writer, description);
    return true;
  }


  /// <summary>
  /// Minimum fraction of a node's visits which a child must have received before its Q is
  /// considered reliable enough to appear in the root move summary (with an absolute floor below).
  /// </summary>
  const float ROOT_SUMMARY_MIN_CHILD_FRAC_N = 0.005f;

  /// <summary>
  /// Absolute minimum visits required of a child appearing in the root move summary.
  /// </summary>
  const int ROOT_SUMMARY_MIN_CHILD_N = 32;


  /// <summary>
  /// Returns a compact summary of the root move statistics of the most recently completed search.
  ///
  /// Sign conventions: the Q on an edge is stated from the perspective of the side to move at the
  /// child node. Root children are therefore negated to reach our perspective, whereas the replies
  /// one level further down are already stated from our perspective and are used as-is.
  /// </summary>
  public override GameEngineRootMoveSummary TryGetLastSearchRootMoveSummary()
  {
    try
    {
      GameEngineSearchResultCeresMCGS result = LastSearchResult;
      MCGSManager manager = result?.Search?.Manager;
      if (manager == null)
      {
        return null;
      }

      GNode root = manager.Engine.SearchRootNode;
      if (root.IsNull || root.NumEdgesExpanded == 0)
      {
        return null;
      }

      // Use the authoritative search root position, not GNode.CalcPosition(): under
      // position-equivalence coalescing a node may describe an earlier occurrence of the same
      // placement. For the same reason moves are decoded with MoveMGFromPos rather than the
      // MoveMG property (which internally calls ParentNode.CalcPosition()).
      MGPosition rootPosMG = manager.Engine.SearchRootPosMG;
      Position rootPos = rootPosMG.ToPosition;
      GEdge playedEdge = root.EdgeForMove(result.BestMoveInfo.BestMove);
      if (playedEdge.IsNull)
      {
        return null;
      }

      GameEngineRootMoveSummary summary = new()
      {
        RootN = root.N,
        StopStatus = manager.StopStatus.ToString(),
        PlayedMoveSAN = SANOfEdge(playedEdge, in rootPosMG),
        PlayedMoveQ = (float)-playedEdge.Q,
        PlayedMoveN = playedEdge.N,
        PlayedMoveP = (float)playedEdge.P,
      };

      // Best alternative root move (best Q among the sufficiently visited moves not played)
      // and the move most favored by policy.
      int minRootChildN = MinChildN(root.N);
      float bestAlternativeQ = float.MinValue;
      float topPolicy = float.MinValue;
      for (int i = 0; i < root.NumEdgesExpanded; i++)
      {
        GEdge edge = root.ChildEdgeAtIndex(i);
        if (edge.IsNull)
        {
          continue;
        }

        if ((float)edge.P > topPolicy)
        {
          topPolicy = (float)edge.P;
          summary.TopPolicyMoveSAN = SANOfEdge(edge, in rootPosMG);
        }

        float qOurs = (float)-edge.Q;
        if (edge != playedEdge && edge.N >= minRootChildN && qOurs > bestAlternativeQ)
        {
          bestAlternativeQ = qOurs;
          summary.HaveBestAlternative = true;
          summary.BestAlternativeSAN = SANOfEdge(edge, in rootPosMG);
          summary.BestAlternativeQ = qOurs;
          summary.BestAlternativeN = edge.N;
          summary.BestAlternativeP = (float)edge.P;
        }
      }

      // Opponent's best reply to the move played. A reply's Q is already stated from our
      // perspective, and the opponent picks the reply minimizing it.
      GNode playedNode = playedEdge.ChildNode;
      if (!playedNode.IsNull && playedNode.NumEdgesExpanded > 0)
      {
        MGPosition playedPos = rootPosMG;
        playedPos.MakeMove(playedEdge.MoveMGFromPos(in rootPosMG));
        int minReplyN = MinChildN(playedNode.N);
        float bestReplyQ = float.MaxValue;
        for (int i = 0; i < playedNode.NumEdgesExpanded; i++)
        {
          GEdge replyEdge = playedNode.ChildEdgeAtIndex(i);
          if (!replyEdge.IsNull && replyEdge.N >= minReplyN && (float)replyEdge.Q < bestReplyQ)
          {
            bestReplyQ = (float)replyEdge.Q;
            summary.HaveBestReply = true;
            summary.BestReplySAN = SANOfEdge(replyEdge, in playedPos);
            summary.BestReplyQ = bestReplyQ;
          }
        }
      }

      return summary;
    }
    catch (Exception)
    {
      // Purely diagnostic; never allow a failure here to disturb play.
      return null;
    }

    static int MinChildN(int parentN) => Math.Max(ROOT_SUMMARY_MIN_CHILD_N, (int)(ROOT_SUMMARY_MIN_CHILD_FRAC_N * parentN));

    static string SANOfEdge(GEdge edge, in MGPosition posMG)
      => MGMoveConverter.ToMove(edge.MoveMGFromPos(in posMG)).ToSAN(posMG.ToPosition);
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

    ResolveProbeSourceIfNeeded();
    Search.ResolvedProbeSource = resolvedProbeSource;
    Search.Search(Evaluators, reuseGraph, SelectWorkerPools, PathVisitPools, ReusableNNBatches,
                  SelectParams, SearchParams, GameLimitManager,
                  curPositionAndMoves, searchLimit, verbose, lastSearchStartTime,
                  gameMoveHistory, callback, null, isFirstMoveOfGame,
                  MoveImmediateIfOnlyOneMove, forcedMove: forcedMove,
                  fixedSearchLimit: FixedSearchLimit,
                  gameStartingTimeLimitSeconds: gameStartingTimeLimitSeconds);
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
  public List<VerboseMoveStat> GetVerboseMoveStats(BestMoveInfoMCGS bestMoveInfo)
  {
    if (Search == null)
    {
      throw new Exception("GetVerboseMoveStats cannot return search statistics because no search has run yet.");
    }

    return VerboseMoveStatsFromMCGSNode.BuildStats(Search.Manager, bestMoveInfo);
  }



  /// <summary>
  /// Probe source (ParamsProbeGraft.ProbeSourceID) resolved ONCE PER ENGINE and reused by every
  /// search of this engine's games, so that a registry factory returning fresh instances yields
  /// one probe farm (workers, queue, transposition table) per engine: concurrent games share
  /// nothing. Disposed with the engine (see CeresAlphaBetaProbeSource.InstanceFactory).
  /// </summary>
  Ceres.Chess.Probing.IProbeSource resolvedProbeSource;

  void ResolveProbeSourceIfNeeded()
  {
    if (resolvedProbeSource != null
     || SearchParams?.ProbeGraft == null
     || !SearchParams.ProbeGraft.AnyConsumerEnabled
     || string.IsNullOrEmpty(SearchParams.ProbeGraft.ProbeSourceID))
    {
      return;
    }
    if (!Ceres.MCGS.Search.ProbeGraft.ProbeSourceRegistry.TryCreate(SearchParams.ProbeGraft.ProbeSourceID, out resolvedProbeSource))
    {
      throw new Exception($"ParamsProbeGraft.ProbeSourceID '{SearchParams.ProbeGraft.ProbeSourceID}' is not registered with ProbeSourceRegistry.");
    }
  }


  /// <summary>
  /// Diposes underlying search engine.
  /// </summary>
  public override void Dispose()
  {
    try
    {
      // The final game of a run is never followed by a ResetGame, so terminate it here.
      MiniLogFinishOpenGame();
      miniLog?.Close();
    }
    catch (Exception)
    {
      // Ignore: never let minilog teardown disrupt disposal.
    }
    miniLog = null;

    Search?.Manager.Engine.Graph.Dispose();
    Search?.Manager?.Dispose();
    Search = null;

    (resolvedProbeSource as IDisposable)?.Dispose();   // this engine's own probe farm
    resolvedProbeSource = null;
    Evaluators?.Dispose();
    Evaluators = null;
    SelectWorkerPools[0]?.Dispose();
    SelectWorkerPools[1]?.Dispose();
    PathVisitPools[0] = null;
    PathVisitPools[1] = null;
    ReusableNNBatches[0]?.Shutdown();
    ReusableNNBatches[1]?.Shutdown();
    ReusableNNBatches[0] = null;
    ReusableNNBatches[1] = null;
  }


  #region Diagnostic minilog

  /// <summary>
  /// Enables and initializes the diagnostic minilog with an explicit file name and the search
  /// limit assigned to this engine (used to populate the header). This is the unconditional
  /// enabler used by the tournament: calling it activates logging regardless of EmitMiniLog.
  /// Intended to be called once, before the first search. A null file name is ignored.
  /// </summary>
  /// <param name="fileName">full path of the minilog file to write</param>
  /// <param name="assignedSearchLimit">search limit assigned to this engine (recorded in the header)</param>
  /// <param name="expectedConcurrentWriters">
  /// number of concurrent game threads whose engine instances share this file (1 = exclusive).
  /// Greater than one causes whole games to be staged and appended atomically, so that games played
  /// concurrently do not interleave within the single aggregated file.
  /// </param>
  public void InitMiniLog(string fileName, SearchLimit assignedSearchLimit, int expectedConcurrentWriters = 1)
  {
    if (fileName == null || miniLog != null)
    {
      return;
    }

    miniLog = MCGSMiniLog.Acquire(fileName, expectedConcurrentWriters);
    miniLog.WriteHeader(ID, EvaluatorDef, assignedSearchLimit, SearchParams, SelectParams);

    // Tournament path enables logging here (regardless of EmitMiniLog); make sure the limit manager
    // also captures its per-move allocation reasoning for inclusion in the log.
    if (GameLimitManager != null)
    {
      GameLimitManager.CaptureDiagnostics = true;
    }
  }


  /// <summary>
  /// Appends a (preformatted) per-game result footer block to the minilog, if active.
  /// Supplied by the tournament because the engine cannot reference tournament types.
  /// </summary>
  public void MiniLogWriteGameResult(string footerText)
  {
    if (miniLog == null)
    {
      return;
    }

    // A game that logged no moves has no separator yet; emit it so the footer cannot be
    // misattributed to the preceding game's section by the (positional) HTML renderer.
    MiniLogEnsureGameStarted();

    // The time-usage summary must precede the footer: the footer is the call that
    // completes/flushes the staged game block.
    MiniLogAppendTimeUsageSection();

    miniLog.AppendGameResultFooter(footerText);
    miniLogFooterWritten = true;
  }


  /// <summary>
  /// Accumulates this move's time-usage data (allocation vs. consumption, stop reason)
  /// into the per-game statistics behind the minilog "=== TIME USAGE ===" section.
  /// Only moves searched under a positive time-typed per-move limit are counted.
  /// </summary>
  void UpdateMiniLogTimeUsage(GameEngineSearchResultCeresMCGS result)
  {
    MCGSManager manager = result.Search.Manager;
    SearchLimit limInit = manager.SearchLimitInitial;
    double elapsed = manager.TimeElapsedTotalSeconds;
    if (limInit == null || !SearchLimit.TypeIsTimeLimit(limInit.Type)
      || !(limInit.Value > 0) || double.IsNaN(elapsed))
    {
      return;
    }

    MiniLogTimeUsageStats t = miniLogTimeUsage;
    t.NumTimedMoves++;
    t.SumAllocSec += limInit.Value;
    t.SumElapsedSec += elapsed;
    t.BudgetFracsPct.Add(100.0 * elapsed / limInit.Value);

    switch (manager.StopStatus)
    {
      case MCGSManager.SearchStopStatus.FutilityPrunedAllMoves: t.StopFutility++; break;
      case MCGSManager.SearchStopStatus.TimeLimitReached: t.StopTimeLimit++; break;
      case MCGSManager.SearchStopStatus.OnlyOneLegalMove: t.StopOnlyMove++; break;
      case MCGSManager.SearchStopStatus.TablebaseImmediateMove: t.StopTablebase++; break;
      case MCGSManager.SearchStopStatus.Instamove: t.StopInstamove++; break;
      default: t.StopOther++; break;
    }

    // Moves consuming less than the increment leave the game clock higher than before the move.
    float increment = manager.LastGameLimitInputs?.IncrementSelf ?? 0;
    if (increment > 0 && elapsed < increment)
    {
      t.NumMovesUnderIncrement++;
    }

    if (TournamentClockRemainingSeconds is float clockAtMoveStart)
    {
      float postMoveClock = clockAtMoveStart - (float)elapsed;
      if (float.IsNaN(t.MinPostMoveClock) || postMoveClock < t.MinPostMoveClock)
      {
        t.MinPostMoveClock = postMoveClock;
      }
    }

    if (miniLogMovesThisGame == 0)
    {
      t.FirstMoveElapsed = elapsed;
    }
  }


  /// <summary>
  /// Emits the per-game "=== TIME USAGE ===" summary section to the minilog and resets the
  /// accumulator. No-op when no minilog is active or the game had no timed moves (e.g. pure
  /// node-limit play). Must be called before the game result footer is appended (only the
  /// footer completes/flushes the staged game block). Never throws.
  /// </summary>
  void MiniLogAppendTimeUsageSection()
  {
    MiniLogTimeUsageStats t = miniLogTimeUsage;
    if (miniLog == null || t.NumTimedMoves == 0)
    {
      return;
    }

    try
    {
      CultureInfo ci = CultureInfo.InvariantCulture;

      List<double> sortedFracs = new(t.BudgetFracsPct);
      sortedFracs.Sort();
      double mean = 0;
      foreach (double frac in sortedFracs)
      {
        mean += frac;
      }
      mean /= sortedFracs.Count;
      static double Percentile(List<double> sorted, double p)
        => sorted[(int)Math.Round(p * (sorted.Count - 1))];

      string nl = System.Environment.NewLine;
      StringBuilder sb = new();
      sb.Append("NumTimedMoves=").Append(t.NumTimedMoves.ToString(ci))
        .Append(" SumAllocSec=").Append(FormatNumber(t.SumAllocSec, "F2", ci))
        .Append(" SumElapsedSec=").Append(FormatNumber(t.SumElapsedSec, "F2", ci))
        .Append(" UnusedAllocSec=").Append(FormatNumber(t.SumAllocSec - t.SumElapsedSec, "F2", ci)).Append(nl);
      sb.Append("MeanBudgetFrac=").Append(FormatNumber(mean, "F1", ci))
        .Append("% P25BudgetFrac=").Append(FormatNumber(Percentile(sortedFracs, 0.25), "F1", ci))
        .Append("% P50BudgetFrac=").Append(FormatNumber(Percentile(sortedFracs, 0.50), "F1", ci))
        .Append("% P75BudgetFrac=").Append(FormatNumber(Percentile(sortedFracs, 0.75), "F1", ci)).Append("%").Append(nl);
      sb.Append("StopFutility=").Append(t.StopFutility.ToString(ci))
        .Append(" StopTimeLimit=").Append(t.StopTimeLimit.ToString(ci))
        .Append(" StopOnlyMove=").Append(t.StopOnlyMove.ToString(ci))
        .Append(" StopTablebase=").Append(t.StopTablebase.ToString(ci))
        .Append(" StopInstamove=").Append(t.StopInstamove.ToString(ci))
        .Append(" StopOther=").Append(t.StopOther.ToString(ci)).Append(nl);
      sb.Append("NumMovesUnderIncrement=").Append(t.NumMovesUnderIncrement.ToString(ci))
        .Append(" MinPostMoveClock=").Append(FormatNumber(t.MinPostMoveClock, "F2", ci))
        .Append(" FirstMoveElapsed=").Append(FormatNumber(t.FirstMoveElapsed, "F2", ci));

      miniLog.AppendTimeUsageSection(sb.ToString());
    }
    catch (Exception exc)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Minilog write failed: " + exc.Message);
    }
    finally
    {
      t.Reset();
    }
  }


  /// <summary>
  /// Writes the pending "=== NEW GAME ===" separator for the game in progress, if not yet written.
  /// Called immediately before emitting any content belonging to that game.
  /// </summary>
  void MiniLogEnsureGameStarted()
  {
    if (miniLog != null && miniLogSeparatorPending)
    {
      miniLog.WriteNewGameSeparator(CurrentGameID);
      miniLogSeparatorPending = false;
    }
  }


  /// <summary>
  /// Writes a minimal result footer for the game in progress if nothing else has, so that every
  /// game section in the log is terminated the same way regardless of how the engine is driven.
  /// The engine itself never learns the game outcome (the tournament does, and supplies a full
  /// footer via MiniLogWriteGameResult), so the synthetic footer records Result=Unknown.
  /// Called when a game ends: at the next ResetGame, and at disposal for the final game.
  /// No-op if a footer was already written, or if the game logged no moves.
  /// </summary>
  void MiniLogFinishOpenGame()
  {
    if (miniLog == null || miniLogFooterWritten || miniLogMovesThisGame == 0)
    {
      return;
    }

    try
    {
      // Emit any accumulated time-usage summary before the synthetic footer completes the game.
      MiniLogAppendTimeUsageSection();

      miniLog.AppendGameResultFooter(
          "=== GAME RESULT ===" + System.Environment.NewLine
        + $"ThisEngine={ID} Result=Unknown Reason=NoResultRecorded" + System.Environment.NewLine
        + $"GameID={CurrentGameID ?? "(unnamed)"}" + System.Environment.NewLine
        + $"MovesLogged={miniLogMovesThisGame}" + System.Environment.NewLine
        + "=== END GAME RESULT ===");
      miniLogFooterWritten = true;
    }
    catch (Exception exc)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Minilog write failed: " + exc.Message);
    }
  }


  /// <summary>
  /// True if a diagnostic minilog is currently being written by this engine.
  /// </summary>
  public bool IsMiniLogActive => miniLog != null;


  /// <summary>
  /// True if this engine created its minilog file, rather than attaching to a file already opened
  /// by the corresponding engine of another concurrent game thread.
  /// </summary>
  public bool MiniLogIsPrimaryWriter => miniLog?.IsPrimaryWriter ?? false;


  /// <summary>
  /// Full path of the active diagnostic minilog file, or null if none.
  /// </summary>
  public string MiniLogFileName => miniLog?.FileName;


  /// <summary>
  /// Appends a (preformatted) blunder-diagnostics block to the minilog, if active. Supplied by the
  /// tournament (which performs blunder detection) because the engine cannot reference tournament types.
  /// </summary>
  public void MiniLogAppendBlunder(string blunderText)
  {
    if (miniLog == null)
    {
      return;
    }

    MiniLogEnsureGameStarted();
    miniLog.AppendBlunderSection(blunderText);
  }


  /// <summary>
  /// Derives a default minilog file name for standalone (non-tournament) use.
  /// </summary>
  private string AutoMiniLogFileName()
  {
    string dir = CeresUserSettingsManager.Settings.DirCeresOutput ?? ".";
    string safeID = string.IsNullOrEmpty(ID) ? "ceres" : ID;
    return Path.Combine(dir, "ceres_" + safeID + "_" + DateTime.Now.Ticks + ".minilog.txt");
  }


  /// <summary>
  /// Builds the compact one-line per-move diagnostic record (scalar fields followed by the
  /// sorted candidate-move table) for the minilog.
  /// </summary>
  private string BuildMiniLogMoveLine(GameEngineSearchResultCeresMCGS result, BestMoveInfoMCGS bestMoveInfo)
  {
    MCGSSearch search = result.Search;
    MCGSManager manager = search.Manager;
    GNode root = search.SearchRootNode;
    MGPosition rootMG = root.CalcPosition();
    Position rootPos = rootMG.ToPosition;

    int rootN = root.N;
    long storeN = root.GraphStore.NodesStore.NumUsedNodes;
    long nnEvals = root.Graph.NNPositionEvaluationsCount;
    float? timeRem = TournamentClockRemainingSeconds;
    float? oppTimeRem = TournamentClockRemainingOpponentSeconds;
    float limInit = manager.SearchLimitInitial == null ? float.NaN : manager.SearchLimitInitial.Value;
    double elapsed = manager.TimeElapsedTotalSeconds;

    // Fraction of the per-move allocated budget consumed (limit-type-aware): for time limits this is
    // elapsed/LimInit; for node limits it is nodes-searched-this-move/LimInit (NumNodesVisitedThisSearch
    // excludes reused-tree nodes, unlike RootN); otherwise not meaningful.
    SearchLimitType limitType = manager.SearchLimitInitial == null
                              ? SearchLimitType.NodesPerMove
                              : manager.SearchLimitInitial.Type;
    double budgetFrac;
    if (!(limInit > 0))
    {
      budgetFrac = double.NaN;
    }
    else if (limitType == SearchLimitType.SecondsPerMove || limitType == SearchLimitType.SecondsForAllMoves)
    {
      budgetFrac = 100.0 * elapsed / limInit;
    }
    else if (limitType == SearchLimitType.NodesPerMove || limitType == SearchLimitType.NodesForAllMoves
             || limitType == SearchLimitType.NodesPerTree)
    {
      budgetFrac = 100.0 * manager.NumNodesVisitedThisSearch / limInit;
    }
    else
    {
      budgetFrac = double.NaN;
    }

    float nps = manager.EstimatedNPS;
    double eps = elapsed > 0 ? manager.NumEvalsThisSearch / elapsed : double.NaN;
    double busyFrac = (!double.IsNaN(manager.TimeDeviceBackendWaitSeconds) && elapsed > 0)
                      ? manager.TimeDeviceBackendWaitSeconds / elapsed
                      : double.NaN;
    float avgDepth = manager.AvgDepth;
    int selDepth = manager.MaxDepth;

    CultureInfo ci = CultureInfo.InvariantCulture;
    StringBuilder sb = new();
    sb.Append("FEN=\"").Append(rootPos.FEN).Append("\", ");
    sb.Append("RootN=").Append(rootN.ToString(ci)).Append(", ");
    sb.Append("StoreN=").Append(storeN.ToString(ci)).Append(", ");
    sb.Append("NNEvals=").Append(nnEvals.ToString(ci)).Append(", ");
    sb.Append("TimeRem=").Append(FormatSeconds(timeRem)).Append(", ");
    sb.Append("OppTimeRem=").Append(FormatSeconds(oppTimeRem)).Append(", ");
    sb.Append("LimInit=").Append(FormatNumber(limInit, "F2", ci)).Append(", ");
    sb.Append("Elapsed=").Append(FormatNumber(elapsed, "F3", ci)).Append(", ");
    sb.Append("BudgetFrac=").Append(FormatNumber(budgetFrac, "F1", ci)).Append("%, ");
    sb.Append("NPS=").Append(FormatNumber(nps, "F0", ci)).Append(", ");
    sb.Append("EPS=").Append(FormatNumber(eps, "F0", ci)).Append(", ");
    sb.Append("BackendBusy=").Append(FormatNumber(busyFrac, "F3", ci)).Append(", ");
    sb.Append("Depth=").Append(FormatNumber(avgDepth, "F2", ci)).Append(", ");
    sb.Append("SelDepth=").Append(selDepth.ToString(ci));
    // Mechanism that selected a played move differing from the most-visited (top-N) move, if any.
    if (!string.IsNullOrEmpty(bestMoveInfo.SelectionNote))
    {
      sb.Append(", Sel=").Append(bestMoveInfo.SelectionNote);
    }
    sb.Append(", Stop=").Append(manager.StopStatus.ToString());
    sb.Append(" | ");
    sb.Append(BuildMiniLogCandidateTable(root, in rootMG, in rootPos, rootN, bestMoveInfo, ci));

    return sb.ToString();
  }


  /// <summary>
  /// Builds the candidate-move table appended to each move line: each root edge as
  /// "(SAN, visit%, Q)" sorted by visits descending, with '*' prefixing the played move.
  /// Reads only edge-level data so terminal/decisive edges (which have no child node) are
  /// handled correctly.
  /// </summary>
  private string BuildMiniLogCandidateTable(GNode root, in MGPosition rootMG, in Position rootPos,
                                            int rootN, BestMoveInfoMCGS bestMoveInfo, CultureInfo ci)
  {
    StringBuilder sb = new();
    bool playedMarked = false;
    int denom = Math.Max(1, rootN);

    GEdge[] edges = root.NumEdgesExpanded == 0
                  ? Array.Empty<GEdge>()
                  : root.EdgesSorted(e => -(double)e.N - 1e-4 * (float)e.P);

    foreach (GEdge edge in edges)
    {
      if (!edge.IsExpanded)
      {
        continue;
      }

      MGMove mg = edge.MoveMGFromPos(in rootMG);
      string san = SANForMove(mg, in rootPos);
      double visitPct = 100.0 * edge.N / denom;
      double q = -edge.Q; // convert from child perspective to side-to-move perspective

      bool played = (!bestMoveInfo.BestMoveEdge.IsNull && edge == bestMoveInfo.BestMoveEdge)
                    || mg == bestMoveInfo.BestMove;
      if (played)
      {
        playedMarked = true;
      }

      if (sb.Length > 0)
      {
        sb.Append(", ");
      }
      if (played)
      {
        sb.Append('*');
      }
      sb.Append('(').Append(san).Append(", ")
        .Append(visitPct.ToString("F2", ci)).Append("%, ")
        .Append(q.ToString("F3", ci)).Append(')');
    }

    // If the played move was not among the enumerated edges (e.g. tablebase / forced / immediate
    // move, or an empty edge set), prepend a synthetic entry so it is always represented.
    if (!playedMarked)
    {
      string san = SANForMove(bestMoveInfo.BestMove, in rootPos);
      string token = "*(" + san + ", n/a, " + bestMoveInfo.QOfBest.ToString("F3", ci) + ")";
      return sb.Length > 0 ? token + ", " + sb.ToString() : token;
    }

    return sb.ToString();
  }


  /// <summary>
  /// Returns the SAN ("Qe8") for a move, falling back to coordinate notation if SAN generation fails.
  /// </summary>
  private static string SANForMove(MGMove mgMove, in Position pos)
  {
    try
    {
      return MGMoveConverter.ToMove(mgMove).ToSAN(in pos);
    }
    catch (Exception)
    {
      return mgMove.MoveStr(MGMoveNotationStyle.Coordinates);
    }
  }


  /// <summary>
  /// Formats a clock value as "174s", or "n/a" when absent/NaN.
  /// </summary>
  private static string FormatSeconds(float? seconds)
  {
    if (!seconds.HasValue || float.IsNaN(seconds.Value))
    {
      return "n/a";
    }
    return seconds.Value.ToString("F2", CultureInfo.InvariantCulture) + "s";
  }


  /// <summary>
  /// Formats a numeric value with the given format, or "n/a" when NaN/infinite.
  /// </summary>
  private static string FormatNumber(double value, string format, CultureInfo ci)
  {
    if (double.IsNaN(value) || double.IsInfinity(value))
    {
      return "n/a";
    }
    return value.ToString(format, ci);
  }

  #endregion


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
