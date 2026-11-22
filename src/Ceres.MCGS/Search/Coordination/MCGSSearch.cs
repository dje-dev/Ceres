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
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Phases.Evaluation;

using static Ceres.MCGS.Search.Coordination.MCGSManager;
using static Ceres.MCGS.Search.Phases.MCGSSelect;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Entry point for launching a search and capturing the results.

/// Note that in most situations instead the class GameEngineCeresInProcess 
/// is preferrable as the entry point for launching searches.
/// </summary>
public partial class MCGSSearch
{
  /// <summary>
  /// Optional delegate that registers to receive informational messages that should be logged.
  /// </summary>
  /// <param name="infoMessage"></param>
  public delegate void MCGSInfoLogger(string infoMessage);

  /// <summary>
  /// The underlying serach manager.
  /// </summary>
  public MCGSManager Manager { get; internal set; }

  /// <summary>
  /// Selected best move from last search.
  /// </summary>
  public MGMove BestMove { get; private set; }

  /// <summary>
  /// Node within the graph from which the search starts.
  /// </summary>
  public GNode SearchRootNode => Manager.Engine.SearchRootNode;

  /// <summary>
  /// N of the SearchRootNode when at beginning of search.
  /// </summary>
  public int StartSearchN { get; private set; }

  /// <summary>
  /// Total number of searches conducted.
  /// </summary>
  public static int SearchCount { get; internal set; }


  #region Graph reuse related

  /// <summary>
  /// The number of times a search from this tree
  /// has been satisfied out of tree reuse (no actual search).
  /// </summary>
  public int CountSearchContinuations { get; private set; }


  /// <summary>
  /// Optional delegate that registers to receive informational messages that should be logged.
  /// </summary>
  /// <param name="search"></param>
  /// <param name="infoMessage"></param>
  public readonly MCGSInfoLogger InfoLogger;

  #endregion


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="infoLogger"></param>
  public MCGSSearch(MCGSInfoLogger infoLogger = null)
  {
    InfoLogger = infoLogger;
  }

  /// <summary>
  /// Runs a new search.
  /// </summary>
  /// <param name="nnEvaluators"></param>
  /// <param name="graphToPossiblyReuse"></param>
  /// <param name="selectWorkerPools"></param>
  /// <param name="pathVisitPools">caller-owned per-iterator path visit slot pools, reused across searches</param>
  /// <param name="reusableNNBatches">caller-owned per-evaluator neural network input batches, reused across searches</param>
  /// <param name="paramsSelect"></param>
  /// <param name="paramsSearch"></param>
  /// <param name="limitManager"></param>
  /// <param name="priorMoves"></param>
  /// <param name="searchLimit"></param>
  /// <param name="verbose"></param>
  /// <param name="startTime"></param>
  /// <param name="gameMoveHistory"></param>
  /// <param name="progressCallback"></param>
  /// <param name="isFirstMoveOfGame"></param>
  /// <param name="fixedSearchLimit"></param>
  /// <param name="gameStartingTimeLimitSeconds">starting time control of the current game (seconds),
  /// or null if unknown/not time-based; used only to resolve the default move overhead</param>
  /// <summary>
  /// Optional pre-resolved probe source (set by the owning engine so the farm's lifetime is
  /// per engine, not per search); when null the coordinator resolves ProbeSourceID itself.
  /// </summary>
  public Ceres.Chess.Probing.IProbeSource ResolvedProbeSource;


  public void Search(NNEvaluatorSet nnEvaluators,
                     Graph graphToPossiblyReuse,
                     WorkerPool<ExtendPathsWorkerInfo>[] selectWorkerPools,
                     ArraySegmentPool<MCGSPathVisit>[] pathVisitPools,
                     EncodedPositionBatchFlat[] reusableNNBatches,
                     ParamsSelect paramsSelect,
                     ParamsSearch paramsSearch,
                     IManagerGameLimit limitManager,
                     PositionWithHistory priorMoves,
                     SearchLimit searchLimit,
                     bool verbose,
                     DateTime startTime,
                     List<GameMoveStat> gameMoveHistory,
                     MCGSProgressCallback progressCallback = null,
                     PositionEvalCache positionEvalCache = null,
                     bool isFirstMoveOfGame = false,
                     bool moveImmediateIfOnlyOneMove = false,
                     MGMove forcedMove = default,
                     SearchLimit fixedSearchLimit = null,
                     float? gameStartingTimeLimitSeconds = null)
  {
    if (searchLimit == null)
    {
      throw new ArgumentNullException(nameof(searchLimit));
    }

    if (searchLimit.SearchCanBeExpanded && !MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC)
    {
      throw new Exception("STORAGE_USE_INCREMENTAL_ALLOC must be true when SearchCanBeExpanded.");
    }

    if (!MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC && !searchLimit.IsNodesLimit)
    {
      throw new Exception("SearchLimit must be NodesPerMove or NodesPerGame when STORAGE_USE_INCREMENTAL_ALLOC is false");
    }

    paramsSearch.Validate();
    paramsSelect.Validate();
    paramsSelect.ValidateAgainst(paramsSearch);

    // TODO: clean this up?
    string tablebasePaths = paramsSearch.EnableTablebases ? paramsSearch.TablebasePaths : null;
    bool forceNoTablebaseTerminals = EvaluatorSyzygy.PosIsTablebaseWinWithNoDTZAvailable(tablebasePaths, priorMoves);

#if NOT
In MCGS version we build Sygyzy evaluator but tell it to instead return
neural network eval because WDL and not DTZ available
good example position: 8/5k2/8/6P1/8/7P/pBp1K3/8 b - - 1 51


    else
    {
      node.InfoRef.EvalResultAuxilliary = (FP16)result.V;
      return default;
    }

This isn't currently easy in the MCGS engine.
As a workaround, EvaluatorSygyzy will just return as if no hit.

#endif

#if NOT
// Have to disable this to avoid overflows
// The problem is probably that the sizing here doesn't account for the fact
// that the graph can grow thru reuse very large.
// (Superseded by the live store sizing below; the move overhead reserve that used to be
//  computed here is now applied to the per-move limit, see AdjustedPerMoveSearchLimit.)

    int maxNodes;
    if (!searchLimit.SearchCanBeExpanded && searchLimit.IsNodesLimit)
    {
      maxNodes = (int)(searchLimit.Value + searchLimit.ValueIncrement + 5000);
    }
    else
    {
      // In this mode, we are just reserving virtual address space
      // from a very large pool (e.g. 256TB for Windows).
      // Therefore it is safe to reserve a very large block.
      if (searchLimit.MaxTreeNodes != null)
      {
        // Reseve somewhat more storage than the maximum requested tree nodes
        // if the search can be expenaded because during tree rewrite
        // a preparatory step (MaterializeNodesWithNonRetainedTranspositionRoots)
        // will initially make the store larger (before it is subsequently compacted).
        double NODES_BUFFER_MULTIPLIER = searchLimit.SearchCanBeExpanded ? 1.2 : 1.0;
        long maxNodesLong = (long)(NODES_BUFFER_MULTIPLIER * searchLimit.MaxTreeNodes.Value) + 100_000;
        maxNodes = (int)Math.Min(maxNodesLong, int.MaxValue - 100_000);
      }
      else
      {
        maxNodes = GraphStore.MAX_NODES;
      }
    }
#endif

    // Preserve any cross-graph evaluation-reuse provider (e.g. opponent graph reuse) so it can be
    // re-established on a freshly created graph if the prior graph is abandoned below.
    Func<Graph> priorReuseGraphProvider = graphToPossiblyReuse?.ReuseGraphProvider;

    // Try to reuse the prior graph
    Graph graphToReuse = GraphReuseManager.TryReuseGraph(paramsSearch, priorMoves, graphToPossiblyReuse,
                                                        out GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                                        out List<GraphRootToSearchRootNodeInfo> searchRootPathFromGraphRoot);

    // Resolve the move overhead in effect for this game (a fixed value for the whole game,
    // determined by the starting time control rather than the dwindling remaining clock).
    float moveOverheadSeconds = EffectiveMoveOverheadSeconds(paramsSearch, gameStartingTimeLimitSeconds);

    SearchLimit searchLimitToUse;
    ManagerGameLimitInputs gameLimitsInputs = null;
    ManagerGameLimitOutputs gameLimitsOutputs = null;
    if (!searchLimit.IsPerGameLimit)
    {
      searchLimitToUse = searchLimit;
    }
    else
    {
      SearchLimitType targetType = searchLimit.IsTimeLimit ? SearchLimitType.SecondsPerMove
                                                           : SearchLimitType.NodesPerMove;

      int searchRootNodeN = searchRootNodeInfo == default ? 0 : searchRootNodeInfo.ChildNode.N;
      double searchRootNodeQ = searchRootNodeInfo == default ? 0 : searchRootNodeInfo.ChildNode.Q;
      gameLimitsInputs = new(priorMoves.FinalPosition,
                           paramsSearch, gameMoveHistory,
                           targetType, searchRootNodeN, (float)searchRootNodeQ,
                           ClockAfterMoveOverheadReserve(searchLimit, moveOverheadSeconds), searchLimit.ValueIncrement,
                           searchLimit.MaxTreeNodes, searchLimit.MaxTreeVisits,
                           float.NaN, float.NaN,
                           maxMovesToGo: searchLimit.MaxMovesToGo,
                           isFirstMoveOfGame: isFirstMoveOfGame, paramsSearch.EnableQuickMoves);

      gameLimitsOutputs = limitManager.ComputeMoveAllocation(gameLimitsInputs);
      searchLimitToUse = gameLimitsOutputs.LimitTarget;
    }

    // Hold back the configured move overhead (UCI MoveOverheadMs) from the time actually searched.
    // For a true per-move limit (e.g. "go movetime") the overhead is reserved from this move's
    // budget, since the move is expected back within that budget. For a per-game clock the reserve
    // was instead taken once from the remaining time fed to the allocation above (a standing buffer
    // of about MoveOverheadSeconds maintained on the clock, costing each move only overhead/movesToGo).
    searchLimitToUse = AdjustedPerMoveSearchLimit(searchLimitToUse, moveOverheadSeconds,
                                                  applyMoveOverhead: !searchLimit.IsPerGameLimit);


    List<MGMove> searchMovesTablebaseRestricted = null;
    if (searchLimit.SearchMoves != null)
    {
      Position startPos = priorMoves.FinalPosition;
      foreach (Move move in searchLimit.SearchMoves)
      {
        searchMovesTablebaseRestricted.Add(MGMoveConverter.MGMoveFromPosAndMove(startPos, move));
      }
    }

    Manager = new(nnEvaluators,
                  paramsSearch, paramsSelect,
                  searchLimitToUse, limitManager, startTime,
                  gameMoveHistory, isFirstMoveOfGame,
                  forceNoTablebaseTerminals,
                  searchMovesTablebaseRestricted, priorMoves.FinalPosition.IsWhite,
                  fixedSearchLimit, reusableNNBatches,
                  resolvedProbeSource: ResolvedProbeSource)
    {
      LastGameLimitInputs = gameLimitsInputs,
      LastGameLimitOutputs = gameLimitsOutputs
    };



    // The graph stores were sized (and fixed) when the graph was created and cannot subsequently
    // grow. Estimate the node capacity the upcoming search requires so that PrepareGraphToUse can
    // PROMOTE (copy into a larger store, preserving all evaluations) a store which is too small
    // (e.g. an interactive session issuing "go nodes 1" and then "go nodes 100000").
    long minStoreNodesNeeded = 0;
    int promoteTargetMaxNodes = 0;
    if (graphToReuse != null)
    {
      int nodesInGraph = graphToReuse.Store.NodesStore.NumTotalNodes;
      int estNodesThisSearch = searchLimitToUse.EstNumSearchNodes(nodesInGraph, 1 + (int)Manager.NNEvaluator0.EstNPSBatch);

      // The estimate is explicitly not a hard upper bound, so require considerable headroom.
      const float SAFETY_MARGIN = 2.0f;
      minStoreNodesNeeded = nodesInGraph + (long)(SAFETY_MARGIN * estNodesThisSearch) + 10_000;
      promoteTargetMaxNodes = FullTierMaxNodes(paramsSearch);
    }

    // Process graph rewrite if needed (handles store capacity, memory pressure, low ratio triggers)
    Graph graphToUse = GraphReuseManager.PrepareGraphToUse(graphToReuse, searchRootNodeInfo, ref searchRootPathFromGraphRoot,
                                                           paramsSearch, searchLimitToUse, priorMoves,
                                                           minStoreNodesNeeded, promoteTargetMaxNodes);

    // If a reusable graph was abandoned, the search will run from a fresh (empty) graph.
    // The allocation above used the (now-discarded) reuse candidate's root N, which with QuickMoves
    // enabled shrinks the time budget on the assumption of a warm tree
    // (see ManagerGameLimitCeresMCGS.MapFractionGraphReusedToShrinkageMultiplier). Recompute with a
    // cold-start input (root N = 0) so the from-scratch search is not starved of time. Guarded on
    // gameLimitsInputs.RootN (a plain int) rather than the now-disposed graph's node counts.
    if (graphToUse == null && gameLimitsInputs != null && gameLimitsInputs.RootN > 0)
    {
      SearchLimitType coldTargetType = searchLimit.IsTimeLimit ? SearchLimitType.SecondsPerMove
                                                               : SearchLimitType.NodesPerMove;
      ManagerGameLimitInputs coldInputs = new(priorMoves.FinalPosition,
                           paramsSearch, gameMoveHistory,
                           coldTargetType, 0, 0f,
                           ClockAfterMoveOverheadReserve(searchLimit, moveOverheadSeconds), searchLimit.ValueIncrement,
                           searchLimit.MaxTreeNodes, searchLimit.MaxTreeVisits,
                           float.NaN, float.NaN,
                           maxMovesToGo: searchLimit.MaxMovesToGo,
                           isFirstMoveOfGame: isFirstMoveOfGame, quickMoveEnabled: paramsSearch.EnableQuickMoves);

      // applyEarlySmoothing:false — the warm allocation already advanced the per-game early-smoothing
      // window for this move; don't double-count it on this recomputation.
      ManagerGameLimitOutputs coldOutputs = limitManager.ComputeMoveAllocation(coldInputs, applyEarlySmoothing: false);

      if (MCGSParamsFixed.GRAPH_REWRITE_DUMP_REUSE_DIAGNOSTICS)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
          $"Graph abandoned: recomputed cold-start move allocation {searchLimitToUse} -> {coldOutputs.LimitTarget} "
        + $"(was sized for warm root N={gameLimitsInputs.RootN:N0}).");
      }

      searchLimitToUse = AdjustedPerMoveSearchLimit(coldOutputs.LimitTarget, moveOverheadSeconds,
                                                    applyMoveOverhead: false);
      Manager.OverrideSearchLimit(searchLimitToUse, coldInputs, coldOutputs);
    }

    // Create new graph if needed (either no prior graph, or prior graph was abandoned)
    if (graphToUse == null)
    {
      // Size from BOTH the raw incoming limit (for per-game limits, the game clock) and the
      // resolved per-move limit actually driving this search, taking the larger: sizing from the
      // raw limit alone is not guaranteed to accommodate the per-move allocation (each term is
      // internally clamped to the full tier, so the max is safe).
      int maxNodesInt = Math.Max(GraphStoreSizeNodes(searchLimit, fixedSearchLimit, paramsSearch),
                                 GraphStoreSizeNodes(searchLimitToUse, fixedSearchLimit, paramsSearch));

      if (MCGSParamsFixed.GRAPH_REWRITE_DUMP_REUSE_DIAGNOSTICS)
      {
        Console.WriteLine($"Graph store sized at {maxNodesInt:N0} nodes "
                        + $"({(maxNodesInt >= FullTierMaxNodes(paramsSearch) ? "full tier" : "small tier")}) "
                        + $"for limit {searchLimitToUse}; total reserved VA now "
                        + $"{(GraphStore.TotalReservedVirtualBytes + (long)(maxNodesInt * 1100L)) / (1024.0 * 1024.0 * 1024.0):F0} GB across "
                        + $"{GraphStore.TotalNumAllocated - GraphStore.TotalNumDisposed + 1} live stores");
      }

      bool hasAction = Manager.NNEvaluator0.HasAction;

#if !ACTION_ENABLED
      // Without ACTION_ENABLED the per-child action buffer is never populated, so ActionHead FPU
      // would silently collapse to a flat constant. Fail loudly instead. Enable with -p:ActionEnabled=true.
      if (Manager.ParamsSelect.FPUMode == ParamsSelect.FPUType.ActionHead
       || Manager.ParamsSelect.FPUModeAtRoot == ParamsSelect.FPUType.ActionHead)
      {
        throw new Exception("FPUType.ActionHead (e.g. via UCI EnableActionHead) requires a build with ACTION_ENABLED (-p:ActionEnabled=true).");
      }
#endif

      if ((Manager.ParamsSelect.FPUMode == ParamsSelect.FPUType.ActionHead
        || Manager.ParamsSelect.FPUModeAtRoot == ParamsSelect.FPUType.ActionHead)
       && !hasAction)
      {
        throw new Exception("FPUType.ActionHead requires a neural network with an action head output.");
      }

      // Right-size the transposition dictionaries to the realistic expected number of distinct
      // positions for this search budget, instead of the worst-case node-buffer reservation (maxNodes).
      int dictionarySizeHint = EstimateInitialDictionaryCapacity(searchLimitToUse, Manager.ParamsSearch, 1 + (int)Manager.NNEvaluator0.EstNPSBatch);
      //      ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"dict= {(float)dictionarySizeHint/1_000_000}  graph={maxNodesInt/1_000_000.0}");

      graphToUse = new(maxNodesInt, hasAction,
                       Manager.ParamsSearch.EnableState,
                       Manager.ParamsSearch.EnableGraph,
                       Manager.ParamsSearch.PathTranspositionMode == PathMode.PositionEquivalence,
                       MCGSParamsFixed.TryEnableLargePages,
                       Manager.ParamsSearch.EnablePseudoTranspositionBlending,
                       priorMoves,
                       Manager.ParamsSearch.TestFlag,
                       Manager.ParamsSearch.MaintainSiblingSets,
                       dictionarySizeHint);

      // Re-establish the cross-graph evaluation-reuse provider that was attached to the abandoned graph
      // (the one-shot PossiblyInitializeForOpponentGraphReuse will not re-run for this new graph).
      graphToUse.ReuseGraphProvider = priorReuseGraphProvider;
    }

    // Per-graph (= per-engine) probe-stamp weight: see Graph.ProbeStampKappa. Set here, after the
    // graph is definitively bound (PrepareGraphToUse may return null for an abandoned reuse, in
    // which case a fresh graph was constructed just above).
    graphToUse.ProbeStampKappa = paramsSelect.TPS_ProbeStampKappa;

    Manager.Engine = new MCGSEngine(Manager, selectWorkerPools, graphToUse, searchRootPathFromGraphRoot == null ? []
                                                                                              : [.. searchRootPathFromGraphRoot],
                                    pathVisitPools);

#if DEBUG
    // Verify the search root actually represents the current position before searching it.
    MCGSRootConsistencyCheck.Validate(Manager.Engine, priorMoves.FinalPosition, "search start", out _);
#endif    

    StartSearchN = SearchRootNode.N;
    (MGMove bestMove, BestMoveInfoMCGS moveInfo) = DoSearch(Manager, verbose, progressCallback,
                                                            moveImmediateIfOnlyOneMove, forcedMove);
    BestMove = bestMove;
  }


  /// <summary>
  /// Assumed lower bound on average COMMITTED bytes per node across all graph data structures
  /// (average probably more typically 400). Used to convert the MaxMemoryBytes budget into a
  /// node-count cap. Note this bounds committed memory; the reserve-only virtual address space
  /// per node is larger (~1 KB) but nearly free (see MCGSParamsFixed.SMALL_TIER_MAX_STORE_NODES).
  /// </summary>
  const long MIN_BYTES_PER_NODE = 300;

  /// <summary>
  /// Returns the "full tier" graph store size in nodes: the largest store this engine instance is
  /// permitted, independent of any particular search limit. This is the reservation used for all
  /// searches other than modest node-limited ones, and the target size when a small-tier store is
  /// promoted (see GraphReuseManager). Bounded by the data-structure/RAM cap (GraphStore.MAX_NODES),
  /// the user-configurable ParamsSearch.MaxNodes, and the committed-memory budget (MaxMemoryBytes).
  /// </summary>
  internal static int FullTierMaxNodes(ParamsSearch paramsSearch)
  {
    long maxNodes = Math.Min(GraphStore.MAX_NODES, paramsSearch.MaxNodes);

    if (paramsSearch.MaxMemoryBytes > 0)
    {
      maxNodes = Math.Min(maxNodes, paramsSearch.MaxMemoryBytes / MIN_BYTES_PER_NODE);
    }

    return (int)Math.Max(1, maxNodes);
  }


  /// <summary>
  /// Returns the number of nodes for which the stores of a new graph should be reserved
  /// to accommodate a search with the specified limit (allowing also for the graph
  /// possibly being reused over many subsequent moves).
  ///
  /// Two-tier policy. Reservation is reserve-only/commit-on-demand on both Linux and Windows, so
  /// its only real cost is virtual address space (~1 KB/node of a ~128 TB per-process budget).
  /// Therefore most searches simply reserve the FULL tier (FullTierMaxNodes) - eliminating any
  /// possibility of the store proving too small mid-game. Only a modest node-based limit takes the
  /// SMALL tier (the limit-derived whole-game estimate), so that many concurrent engines (e.g. a
  /// research harness running ~100 in-process training matches) do not each reserve ~1 TB of
  /// address space. A small-tier store that later proves too small is PROMOTED by copy, never
  /// abandoned (see GraphReuseManager.PrepareGraphToUse). Time-based limits never take the small
  /// tier: a mid-game promotion copy would burn clock time.
  /// </summary>
  /// <param name="searchLimit">Search limit for which the graph is being sized.</param>
  /// <param name="fixedSearchLimit">Optionally a fixed (per-move) limit known to apply to all moves.</param>
  /// <param name="paramsSearch">Search parameters (used to read graph reuse and memory settings).</param>
  internal static int GraphStoreSizeNodes(SearchLimit searchLimit, SearchLimit fixedSearchLimit, ParamsSearch paramsSearch)
  {
    long fullTierNodes = FullTierMaxNodes(paramsSearch);

    // Estimate the whole-game node requirement from the search limit.
    // Note that if GraphReuseRewriteEnabled then we can be more aggressive in allowing more nodes
    // because we will rewrite the graph to reduce if approaches the limit.
    // TODO: possibly unify this with code already in SearchLimit
    long MAX_MOVES_PER_GRAPH = paramsSearch.GraphReuseEnabled ? (paramsSearch.GraphReuseRewriteEnabled ? 150L : 400L) : 5L;
    const long MAX_NODES_PER_SECOND = 500_000L;
    long maxNodes = searchLimit.Type switch
    {
      SearchLimitType.BestValueMove => 1,
      SearchLimitType.BestActionMove => 1,

      SearchLimitType.NodesPerMove => (long)(MAX_MOVES_PER_GRAPH * (float)(searchLimit.Value + searchLimit.ValueIncrement)),
      SearchLimitType.NodesForAllMoves => (long)(searchLimit.Value + searchLimit.ValueIncrement * MAX_MOVES_PER_GRAPH),

      SearchLimitType.SecondsPerMove => (long)((searchLimit.Value + searchLimit.ValueIncrement) * MAX_NODES_PER_SECOND * MAX_MOVES_PER_GRAPH),
      SearchLimitType.SecondsForAllMoves => (long)((searchLimit.Value + (searchLimit.ValueIncrement * MAX_MOVES_PER_GRAPH)) * MAX_NODES_PER_SECOND),
      _ => fullTierNodes
    };

    maxNodes = Math.Min(fullTierNodes, maxNodes);

    // Possibly apply tighter constraint based on fixedSearchLimit
    if (fixedSearchLimit != null && fixedSearchLimit.IsNodesLimit)
    {
      long fixedMax = MAX_MOVES_PER_GRAPH * (long)(fixedSearchLimit.Value + fixedSearchLimit.ValueIncrement);
      maxNodes = Math.Min(maxNodes, fixedMax);
    }

    // Small tier only for node-based limits with a modest whole-game estimate (see summary above).
    bool limitIsNodesBased = searchLimit.IsNodesLimit
                          || (fixedSearchLimit != null && fixedSearchLimit.IsNodesLimit);
    bool smallTier = limitIsNodesBased && maxNodes <= MCGSParamsFixed.SMALL_TIER_MAX_STORE_NODES;

    // With EnableState the full-tier jump is suppressed: AllStateVectors is an eagerly COMMITTED
    // managed allocation of 8 bytes per reserved node (GraphStore constructor), which at full-tier
    // sizes would commit multiple GB up front. The legacy limit-derived size is used instead.
    if (smallTier || paramsSearch.EnableState)
    {
      return (int)Math.Min(maxNodes + 1000, fullTierNodes);
    }

    return (int)fullTierNodes;
  }


  /// <summary>
  /// Estimates a sensible initial capacity (in entries) for the per-graph transposition dictionaries,
  /// based on the search budget. This lets the dictionary be sized up front to roughly the number of
  /// distinct positions it will hold, avoiding a cascade of incremental resizes (directory doublings and
  /// bucket pre-allocation with the extendible map, or stop-the-world rehashes under the legacy
  /// ConcurrentDictionary) as a long search grows the table from a small default to tens of millions of
  /// entries while holding locks (a stall that idles the GPU).
  ///
  /// A tiny budget (e.g. a single node) yields a tiny hint; a large budget (e.g. a long tournament time
  /// control arriving via UCI) yields a larger one. The returned value is clamped here and again,
  /// defensively, in Graph.Initialize.
  /// </summary>
  /// <param name="limit">The per-move search limit actually being used (per-game limits have already
  /// been mapped to a per-move target by the limits manager).</param>
  /// <param name="paramsSearch">Search parameters (used to read graph-reuse settings).</param>
  /// <param name="estimatedEPS">Estimated maximum evaluations per second from this backend.</param>
  static int EstimateInitialDictionaryCapacity(SearchLimit limit, ParamsSearch paramsSearch, int estimatedEPS)
  {

    // Expected nodes added by this move's search (fresh graph => 0 initial nodes).
    long perMoveNodes = Math.Max(1, limit.EstNumSearchNodes(0, estimatedEPS));

    // The graph (and its dictionary) persists across the multiple moves that share a reused graph, so
    // anticipate some cross-move accumulation of distinct positions. Without reuse a single search adds
    // only modestly more positions than its final tree size.
    double accumFactor = paramsSearch.GraphReuseEnabled
                           ? MCGSParamsFixed.DICTIONARY_SIZE_HINT_REUSE_ACCUM_FACTOR
                           : 1.25;

    long estEntries = (long)(perMoveNodes * accumFactor);

    return (int)Math.Clamp(estEntries, MCGSParamsFixed.DICTIONARY_SIZE_HINT_MIN, MCGSParamsFixed.DICTIONARY_SIZE_HINT_MAX);
  }


  /// <summary>
  /// Returns the move overhead (seconds) in effect for a game, given the configured value and the
  /// starting time control of the game.
  ///
  /// The configured value (ParamsSearch.MoveOverheadSeconds) is used as-is unless it is still at its
  /// default, in which case a game played under a short time control (a time-based search limit with
  /// a starting value below MOVE_OVERHEAD_SHORT_TIME_CONTROL_THRESHOLD_SECONDS) instead receives the
  /// smaller MOVE_OVERHEAD_SECONDS_SHORT_TIME_CONTROL, since the full default reserve would consume
  /// an excessive fraction of such a game's time. An explicitly configured overhead (e.g. set by a
  /// GUI via UCI MoveOverheadMs) is never overridden.
  ///
  /// The test is against the starting time control of the game (not the remaining clock), so the
  /// overhead in effect is constant across all moves of a game.
  /// </summary>
  /// <param name="paramsSearch"></param>
  /// <param name="gameStartingTimeLimitSeconds">starting value of the game's time-based search limit,
  /// or null if unknown or the limit is not time-based</param>
  /// <returns></returns>
  static float EffectiveMoveOverheadSeconds(ParamsSearch paramsSearch, float? gameStartingTimeLimitSeconds)
  {
    bool isConfiguredValueDefault = paramsSearch.MoveOverheadSeconds == ParamsSearch.MOVE_OVERHEAD_SECONDS_DEFAULT;
    bool isShortTimeControl = gameStartingTimeLimitSeconds is float startingSeconds
                           && startingSeconds > 0
                           && startingSeconds < ParamsSearch.MOVE_OVERHEAD_SHORT_TIME_CONTROL_THRESHOLD_SECONDS;

    // If we have a starting time limit, carve out very small fraction as extra padding
    // (insurance against long GC pauses possible with long games).
    const float FRAC_START_TIME_PADDING = 0.002f;
    const float MAX_EXTRA_PADDING_SECONDS = 10f;  
    float extraPaddingSeconds = gameStartingTimeLimitSeconds is float ? ((float)gameStartingTimeLimitSeconds * FRAC_START_TIME_PADDING) : 0;
    extraPaddingSeconds = Math.Min(extraPaddingSeconds, MAX_EXTRA_PADDING_SECONDS);

    return isConfiguredValueDefault && isShortTimeControl ? ParamsSearch.MOVE_OVERHEAD_SECONDS_SHORT_TIME_CONTROL
                                                          : (paramsSearch.MoveOverheadSeconds + extraPaddingSeconds);
  }


  /// <summary>
  /// Returns the specified per-move search limit adjusted for actual execution:
  ///
  ///   - if applyMoveOverhead, the move overhead in effect (see EffectiveMoveOverheadSeconds)
  ///     is held back from a time limit, so that the
  ///     time spent emitting the move and any GUI/network latency do not push the move past its
  ///     allotted time. The reserve is capped at half of the allotted time so that a short search
  ///     (e.g. a "go movetime 100" analysis probe, where the overhead exceeds the whole budget) is
  ///     shortened proportionally rather than collapsed to nothing.
  ///     This applies only to true per-move limits; a limit allocated from a per-game clock has the
  ///     overhead reserved once on the clock instead (see ClockAfterMoveOverheadReserve).
  ///
  ///   - any MaxTreeNodes configured in Ceres.json is installed if the limit does not already
  ///     carry an explicit value (the UCI layer applies this setting itself; this covers the
  ///     in-process entry points, such as tournaments and test suites, which do not).
  ///     Note that the store's own capacity is enforced independently of this setting
  ///     (see MCGSManager.CalcSearchStopStatus).
  /// </summary>
  /// <param name="limit">The per-move limit about to be searched.</param>
  /// <param name="moveOverheadSeconds">The move overhead in effect (see EffectiveMoveOverheadSeconds).</param>
  /// <param name="applyMoveOverhead">If the move overhead should be held back from this limit
  /// (false when the limit came from a per-game clock which was already reserved against).</param>
  /// <returns></returns>
  static SearchLimit AdjustedPerMoveSearchLimit(SearchLimit limit, float moveOverheadSeconds,
                                                bool applyMoveOverhead)
  {
    if (limit.MaxTreeNodes is null && CeresUserSettingsManager.Settings.MaxTreeNodes is not null)
    {
      limit = limit with { MaxTreeNodes = CeresUserSettingsManager.Settings.MaxTreeNodes };
    }

    if (!applyMoveOverhead || !limit.IsTimeLimit || moveOverheadSeconds <= 0 || limit.Value <= 0)
    {
      return limit;
    }

    const float MIN_SEARCH_SECONDS = 0.01f;
    const float MAX_FRACTION_RESERVED = 0.5f;

    float overhead = Math.Min(moveOverheadSeconds, MAX_FRACTION_RESERVED * limit.Value);
    return limit with { Value = Math.Max(MIN_SEARCH_SECONDS, limit.Value - overhead) };
  }


  /// <summary>
  /// Returns the remaining game clock time of the specified per-game limit less the
  /// move overhead in effect (see EffectiveMoveOverheadSeconds).
  ///
  /// Reserving once from the clock (rather than from every per-move allocation) maintains a
  /// standing buffer of about the move overhead against move emission and GUI/network latency.
  /// </summary>
  /// <param name="searchLimit">A per-game limit whose Value is the remaining clock time.</param>
  /// <param name="moveOverheadSeconds">The move overhead in effect (see EffectiveMoveOverheadSeconds).</param>
  /// <returns></returns>
  static float ClockAfterMoveOverheadReserve(SearchLimit searchLimit, float moveOverheadSeconds)
  {
    if (!searchLimit.IsTimeLimit || moveOverheadSeconds <= 0 || searchLimit.Value <= 0)
    {
      return searchLimit.Value;
    }

    const float MAX_FRACTION_RESERVED = 0.5f;

    float reserve = Math.Min(moveOverheadSeconds, MAX_FRACTION_RESERVED * searchLimit.Value);
    return searchLimit.Value - reserve;
  }
}
