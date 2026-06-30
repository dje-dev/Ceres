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
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
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


  public bool DisposeGraphAfterSearch;

  /// <summary>
  /// Optional fixed search limit known at engine creation time.
  /// When specified, allows optimizations for small searches.
  /// </summary>
  public readonly SearchLimit FixedSearchLimit;

  /// <summary>
  /// If true this engine emits a per-tournament diagnostic "move log" file (default false).
  /// The tournament path enables logging explicitly via InitGameLog; this flag additionally
  /// drives standalone (non-tournament) lazy initialization on the first search.
  /// </summary>
  public readonly bool EmitGameLog;

  /// <summary>
  /// Active diagnostic move-log writer (null unless logging is enabled).
  /// </summary>
  MCGSGameMoveLog gameLog;

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
  /// <param name="emitGameLog">if a per-tournament diagnostic move-log file should be written</param>
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
                                      bool emitGameLog = false) : base(id, processorGroupID)
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
    EmitGameLog = emitGameLog;

    // If a diagnostic move-log will be emitted, have the limit manager capture its per-move
    // allocation reasoning so it can be recorded near each move header (see BuildGameLogMoveLine).
    if (GameLimitManager != null && emitGameLog)
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

    gameLog?.WriteNewGameSeparator(gameID);
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

    // Emit the compact per-move diagnostic game-log line (if enabled). For standalone (non-tournament)
    // use, lazily initialize on the first search using an auto-derived file name. The write is wrapped
    // so a logging failure can never disrupt the game.
    if (gameLog == null && EmitGameLog)
    {
      InitGameLog(AutoGameLogFileName(), FixedSearchLimit ?? searchLimit);
    }
    if (gameLog != null)
    {
      try
      {
        // Emit the limits-manager allocation reasoning (if captured) immediately before the move
        // line it governs, so it sits next to that move header (mirrors the inline blunder block).
        gameLog.AppendLimitsSection(result.Search.Manager.LastGameLimitOutputs?.DiagnosticText);
        gameLog.WriteMoveLine(BuildGameLogMoveLine(result, bestMoveInfo));
      }
      catch (Exception exc)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Game move log write failed: " + exc.Message);
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
  public List<VerboseMoveStat> GetVerboseMoveStats(BestMoveInfoMCGS bestMoveInfo)
  {
    if (Search == null)
    {
      throw new Exception("GetVerboseMoveStats cannot return search statistics because no search has run yet.");
    }

    return VerboseMoveStatsFromMCGSNode.BuildStats(Search.Manager, bestMoveInfo);
  }



  /// <summary>
  /// Diposes underlying search engine.
  /// </summary>
  public override void Dispose()
  {
    try
    {
      gameLog?.Close();
    }
    catch (Exception)
    {
      // Ignore: never let move-log teardown disrupt disposal.
    }
    gameLog = null;

    Search?.Manager.Engine.Graph.Dispose();
    Search?.Manager?.Dispose();
    Search = null;
    Evaluators?.Dispose();
    Evaluators = null;
    SelectWorkerPools[0]?.Dispose();
    SelectWorkerPools[1]?.Dispose();
  }


  #region Diagnostic game move log

  /// <summary>
  /// Enables and initializes the diagnostic move-log with an explicit file name and the search
  /// limit assigned to this engine (used to populate the header). This is the unconditional
  /// enabler used by the tournament: calling it activates logging regardless of EmitGameLog.
  /// Intended to be called once, before the first search. A null file name is ignored.
  /// </summary>
  public void InitGameLog(string fileName, SearchLimit assignedSearchLimit)
  {
    if (fileName == null || gameLog != null)
    {
      return;
    }

    gameLog = new MCGSGameMoveLog(fileName);
    gameLog.WriteHeader(ID, EvaluatorDef, assignedSearchLimit, SearchParams, SelectParams);

    // Tournament path enables logging here (regardless of EmitGameLog); make sure the limit manager
    // also captures its per-move allocation reasoning for inclusion in the log.
    if (GameLimitManager != null)
    {
      GameLimitManager.CaptureDiagnostics = true;
    }
  }


  /// <summary>
  /// Appends a (preformatted) per-game result footer block to the move-log, if active.
  /// Supplied by the tournament because the engine cannot reference tournament types.
  /// </summary>
  public void GameLogWriteGameResult(string footerText)
  {
    gameLog?.AppendGameResultFooter(footerText);
  }


  /// <summary>
  /// True if a diagnostic move-log is currently being written by this engine.
  /// </summary>
  public bool IsGameLogActive => gameLog != null;


  /// <summary>
  /// Full path of the active diagnostic move-log file, or null if none.
  /// </summary>
  public string GameLogFileName => gameLog?.FileName;


  /// <summary>
  /// Appends a (preformatted) blunder-diagnostics block to the move-log, if active. Supplied by the
  /// tournament (which performs blunder detection) because the engine cannot reference tournament types.
  /// </summary>
  public void GameLogAppendBlunder(string blunderText)
  {
    gameLog?.AppendBlunderSection(blunderText);
  }


  /// <summary>
  /// Derives a default move-log file name for standalone (non-tournament) use.
  /// </summary>
  private string AutoGameLogFileName()
  {
    string dir = CeresUserSettingsManager.Settings.DirCeresOutput ?? ".";
    string safeID = string.IsNullOrEmpty(ID) ? "ceres" : ID;
    return Path.Combine(dir, "ceres_" + safeID + "_" + DateTime.Now.Ticks + ".movelog.txt");
  }


  /// <summary>
  /// Builds the compact one-line per-move diagnostic record (scalar fields followed by the
  /// sorted candidate-move table) for the move log.
  /// </summary>
  private string BuildGameLogMoveLine(GameEngineSearchResultCeresMCGS result, BestMoveInfoMCGS bestMoveInfo)
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
    sb.Append(" | ");
    sb.Append(BuildGameLogCandidateTable(root, in rootMG, in rootPos, rootN, bestMoveInfo, ci));

    return sb.ToString();
  }


  /// <summary>
  /// Builds the candidate-move table appended to each move line: each root edge as
  /// "(SAN, visit%, Q)" sorted by visits descending, with '*' prefixing the played move.
  /// Reads only edge-level data so terminal/decisive edges (which have no child node) are
  /// handled correctly.
  /// </summary>
  private string BuildGameLogCandidateTable(GNode root, in MGPosition rootMG, in Position rootPos,
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
