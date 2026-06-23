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
using System.Diagnostics;

using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Manages graph reuse decisions for MCGS search. When a prior graph can be carried forward to the
/// current position, decides whether to:
///   - REUSE   it unchanged (search descends from the search root), or
///   - EXTRACT the subgraph still reachable from the search root into a fresh, smaller graph, or
///   - ABANDON it and search from scratch.
///
/// The decision follows a simple memory / utilization / time policy (see <see cref="DecideAction"/>).
/// </summary>
public static class GraphReuseManager
{
  #region Constants

  /// <summary>
  /// A reused graph whose root has fewer than this many visits (N) is "trivially small":
  /// it is always reused as-is (never extracted or abandoned).
  /// </summary>
  public const int TRIVIALLY_SMALL_ROOT_N = 1_000_000;

  /// <summary>
  /// Heuristic extraction speed (reachable nodes copied per second) used to estimate extraction cost:
  /// estimated seconds to extract = reachableNodeCount / EXTRACT_NODES_PER_SEC.
  /// Calibrated from observed parallel-extraction throughput (~1M nodes/sec).
  /// </summary>
  public const double EXTRACT_NODES_PER_SEC = 1_000_000.0;

  /// <summary>
  /// Under high memory pressure, extraction is only worthwhile if it frees a meaningful fraction of
  /// the graph. Extract only when the reachable (retained) fraction is at or below this ceiling;
  /// above it, extraction barely shrinks the graph, so we reuse it (or abandon when memory is
  /// critical). Without this, a graph that is ~entirely reachable would be "extracted" at ~100%
  /// retention — paying the full copy cost to free almost nothing.
  /// </summary>
  public const double HIGH_MEM_EXTRACT_MAX_RETENTION = 0.50;

  #endregion

  #region Action

  /// <summary>
  /// The action chosen for a reusable graph.
  /// </summary>
  private enum GraphReuseAction
  {
    /// <summary>Keep the existing graph unchanged (search descends from the search root).</summary>
    Reuse,

    /// <summary>Copy the subgraph reachable from the search root into a fresh, smaller graph.</summary>
    Extract,

    /// <summary>Discard the graph; the next search starts from scratch.</summary>
    Abandon
  }

  /// <summary>
  /// Outcome of <see cref="DecideAction"/>: the chosen action, an explanation (null for plain reuse),
  /// and instrumentation from the reachability check. <see cref="ReachableNodes"/> is -1 when the
  /// decision short-circuited without running the BFS (trivially-small or low-memory graph).
  /// </summary>
  private readonly record struct ReuseDecision(GraphReuseAction Action, string Reason,
                                               double ReachabilityCheckSeconds, int ReachableNodes,
                                               GraphExtractor.ReachableSet Reachable);

  #endregion

  #region Graph Reuse Decision

  /// <summary>
  /// Determines if a prior graph can be reused for the current search.
  /// Returns the graph if reusable, null otherwise (disposing the graph if not reusable).
  /// </summary>
  public static Graph TryReuseGraph(ParamsSearch paramsSearch,
                                    PositionWithHistory priorMoves,
                                    Graph graphToPossiblyReuse,
                                    out GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                    out List<GraphRootToSearchRootNodeInfo> searchRootPathFromGraphRoot)
  {
    searchRootNodeInfo = default;
    searchRootPathFromGraphRoot = null;

    // No graph to reuse
    if (graphToPossiblyReuse == null)
    {
      return null;
    }

    // Graph reuse disabled or no valid continuation
    if (!paramsSearch.GraphReuseEnabled
     || !graphToPossiblyReuse.Store.PositionHistory.HasContinuationOf(priorMoves))
    {
      graphToPossiblyReuse.Dispose();
      return null;
    }

    // Try to find path from graph root to search root
    searchRootPathFromGraphRoot = graphToPossiblyReuse.GuardedFindPathAlongPositionWithHistory(priorMoves);
    if (searchRootPathFromGraphRoot == null)
    {
      // History prefix mismatch (or path lookup failed); cannot reuse.
      graphToPossiblyReuse.Dispose();
      return null;
    }

    // Identical position: the new search root IS the graph root (path has no moves below the root).
    // Reuse the whole graph as-is. MCGSEngine sets SearchRootNode = graph root when the path is empty
    // (exactly as for a freshly created graph), so leaving the path empty resumes the existing root.
    // Only searchRootNodeInfo.ChildNode is consumed downstream in this case (graph-rewrite decision and
    // per-game time allocation); the remaining fields are unused here and passed as default.
    if (searchRootPathFromGraphRoot.Count == 0)
    {
      GNode rootNode = graphToPossiblyReuse.GraphRootNode;
      if (!rootNode.IsEvaluated)
      {
        graphToPossiblyReuse.Dispose();
        searchRootPathFromGraphRoot = null;
        return null;
      }

      searchRootNodeInfo = new GraphRootToSearchRootNodeInfo(rootNode, default, default, default, default, false);
      graphToPossiblyReuse.SetSearchRootNode(rootNode.Index);
      return graphToPossiblyReuse;
    }

    if (searchRootPathFromGraphRoot.Count < 2)
    {
      // Search root is exactly one ply below the graph root: retain existing conservative behavior (no reuse).
      graphToPossiblyReuse.Dispose();
      searchRootPathFromGraphRoot = null;
      return null;
    }

    searchRootNodeInfo = searchRootPathFromGraphRoot[^1];

    // Search root node must be evaluated
    if (!searchRootNodeInfo.ChildNode.IsEvaluated)
    {
      graphToPossiblyReuse.Dispose();
      searchRootNodeInfo = default;
      searchRootPathFromGraphRoot = null;
      return null;
    }

    // Graph is reusable
    graphToPossiblyReuse.SetSearchRootNode(searchRootNodeInfo.ChildNode.Index);
    return graphToPossiblyReuse;
  }


  /// <summary>
  /// Decides what to do with a reusable graph (reuse / extract / abandon) and carries out the
  /// decision (disposing the old graph when it is replaced or abandoned). Returns the graph to use
  /// for the upcoming search: the original graph (reuse), a freshly-extracted smaller graph, or
  /// null (abandon, or no graph to process).
  /// </summary>
  /// <param name="graph">The reusable graph (its search root is already set), or null.</param>
  /// <param name="searchRootNodeInfo">Info about the search root node.</param>
  /// <param name="searchRootPathFromGraphRoot">Path graph-root → search-root; cleared if the graph is replaced/abandoned.</param>
  /// <param name="paramsSearch">Search parameters (for the memory budget).</param>
  /// <param name="searchLimit">Current search limit (for the time budget).</param>
  /// <param name="priorMoves">Position+history for the new search root.</param>
  public static Graph ProcessGraphRewrite(Graph graph,
                                          GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                          ref List<GraphRootToSearchRootNodeInfo> searchRootPathFromGraphRoot,
                                          ParamsSearch paramsSearch,
                                          SearchLimit searchLimit,
                                          PositionWithHistory priorMoves)
  {
    if (graph == null)
    {
      return null;
    }

    ReuseDecision decision = DecideAction(graph, searchRootNodeInfo, paramsSearch, searchLimit);
    string reason = decision.Reason;

    if (decision.Action == GraphReuseAction.Reuse)
    {
      MaybeLogReachabilityCheck("REUSE", decision, graph);
      return graph;
    }

    if (decision.Action == GraphReuseAction.Extract)
    {
      // Reuse the reachable set the decision already enumerated — single BFS, no re-enumeration.
      GraphExtractor.ExtractResult ex = GraphExtractor.ExtractFromReachable(
        graph, searchRootNodeInfo.ChildNode.Index, priorMoves, decision.Reachable, decision.ReachabilityCheckSeconds);
      if (ex.Succeeded)
      {
        LogExtraction(ex, searchRootNodeInfo.ChildNode.N, reason);

        // Switch over to the new (smaller) graph and dispose the old one. Returning a different
        // Graph object is contract-compatible with the caller (it installs whatever is returned);
        // the search root is now the graph root, so clear the path.
        graph.Dispose();
        searchRootPathFromGraphRoot = null;
        return ex.NewGraph;
      }

      // Extraction unexpectedly failed: fall through to abandonment rather than carry an over-budget graph.
      reason = "extraction did not complete; " + reason;
    }

    // Abandon (decided, or extraction fell through).
    LogAbandon(graph, searchRootNodeInfo, reason);
    MaybeLogReachabilityCheck("ABANDON", decision, graph);
    graph.Dispose();
    searchRootPathFromGraphRoot = null;
    return null;
  }


  /// <summary>
  /// Decides whether to reuse, extract, or abandon a reusable graph, following a simple
  /// memory / utilization / time policy. Returns the action and, for non-reuse actions, a short
  /// human-readable explanation (null for reuse).
  ///
  /// Definitions:
  ///   reachableNodes     = nodes reachable from the search root (exactly what extraction would retain)
  ///   reachableFraction  = reachableNodes / totalNodes         (how much of the graph actually survives)
  ///   memoryUsedFraction = process private bytes / configured max memory
  ///   timeToExtract      = reachableNodes / EXTRACT_NODES_PER_SEC      (estimated seconds)
  ///   availableTime      = seconds available for the move (MaxValue if the search is not time-based)
  ///
  /// reachableNodes is the ACTUAL count (a bounded BFS via <see cref="GraphExtractor.EnumerateReachable"/>,
  /// whose result is reused by extraction so there is only one BFS), not the visit-based
  /// searchRoot.N / graphRoot.N proxy: in transposition-heavy graphs (especially PositionEquivalence
  /// mode) most distinct positions remain reachable from the new root via cross-edges even when only a
  /// small fraction of VISITS flowed through it, so the visit ratio badly under-predicts both retention
  /// and copy cost.
  ///
  /// Policy:
  ///   - trivially small graph                                              -> reuse
  ///   - memory used &lt; 20%                                               -> reuse
  ///   - memory used &gt; 70% (high): extract if it frees enough (reachableFraction &le;
  ///         HIGH_MEM_EXTRACT_MAX_RETENTION) AND fits time (timeToExtract &le; 80% avail);
  ///         else reuse, unless memory is critical (&gt; 90%) in which case abandon to free it
  ///   - reachableFraction &lt; 30% and timeToExtract &lt; 25% avail         -> extract
  ///   - reachableFraction &lt; 10% and timeToExtract &lt; 50% avail         -> extract
  ///   - otherwise                                                          -> reuse
  /// </summary>
  private static ReuseDecision DecideAction(
    Graph graph,
    GraphRootToSearchRootNodeInfo searchRootNodeInfo,
    ParamsSearch paramsSearch,
    SearchLimit searchLimit)
  {
    // Trivially small graph: always reuse (no BFS).
    if (graph.GraphRootNode.N < TRIVIALLY_SMALL_ROOT_N)
    {
      return new ReuseDecision(GraphReuseAction.Reuse, null, 0.0, -1, default);
    }

    // Plenty of memory free: reuse (no BFS). Memory is probed only past the trivially-small short-circuit.
    double memoryUsedFraction = CurrentMemoryUsedFraction(paramsSearch, out long memoryUsedBytes);
    if (memoryUsedFraction < 0.20)
    {
      return new ReuseDecision(GraphReuseAction.Reuse, null, 0.0, -1, default);
    }

    double availableTime = searchLimit.IsTimeLimit ? searchLimit.Value : double.MaxValue;
    int totalNodes = graph.Store.NodesStore.NumUsedNodes;
    bool highMemory = memoryUsedFraction > 0.70;

    // Enumerate the nodes actually reachable from the search root (== what extraction would retain).
    // This single BFS is reused by extraction (no second enumeration). It is early-aborted at the
    // largest count that could still yield an extract, so its cost is bounded by the move's time
    // budget and the retention ceiling:
    //   retention ceiling: high mem -> HIGH_MEM_EXTRACT_MAX_RETENTION, moderate -> 0.30
    //   time factor:       high mem -> 0.80,                           moderate -> 0.50 (loosest gate)
    // For non-time-based searches the available*rate term is infinite and only the fraction cap binds.
    double fracCeiling = highMemory ? HIGH_MEM_EXTRACT_MAX_RETENTION : 0.30;
    double timeFactor = highMemory ? 0.80 : 0.50;
    double fracCapNodes = fracCeiling * totalNodes;
    double timeCapNodes = availableTime >= double.MaxValue
      ? double.MaxValue
      : timeFactor * availableTime * EXTRACT_NODES_PER_SEC;
    int abortAbove = (int)Math.Min((double)int.MaxValue, Math.Min(fracCapNodes, timeCapNodes));

    Stopwatch sw = Stopwatch.StartNew();
    GraphExtractor.ReachableSet reachable = GraphExtractor.EnumerateReachable(graph, searchRootNodeInfo.ChildNode.Index, abortAbove);
    double reachabilitySeconds = sw.Elapsed.TotalSeconds;

    int reachableNodes = reachable.Count;
    double reachableFraction = totalNodes > 0 ? (double)reachableNodes / totalNodes : 0.0;
    double timeToExtract = reachableNodes / EXTRACT_NODES_PER_SEC;

    if (highMemory)
    {
      // Extract only if it frees a meaningful fraction AND fits the time budget.
      if (reachableFraction <= HIGH_MEM_EXTRACT_MAX_RETENTION && timeToExtract <= 0.80 * availableTime)
      {
        return new ReuseDecision(GraphReuseAction.Extract,
          $"high memory {memoryUsedFraction:P0} ({MemoryGBString(memoryUsedBytes, paramsSearch.MaxMemoryBytes)}); "
        + $"{reachableNodes:N0} reachable ({reachableFraction:P0}, ~{timeToExtract:F1}s)",
          reachabilitySeconds, reachableNodes, reachable);
      }

      // Can't usefully shrink: free the memory only when it is critical (> 90%); otherwise keep the graph.
      return memoryUsedFraction > 0.90
        ? new ReuseDecision(GraphReuseAction.Abandon,
            $"critical memory {memoryUsedFraction:P0} ({MemoryGBString(memoryUsedBytes, paramsSearch.MaxMemoryBytes)}); "
          + $"extraction won't help ({reachableFraction:P0} reachable)",
            reachabilitySeconds, reachableNodes, reachable)
        : new ReuseDecision(GraphReuseAction.Reuse, null, reachabilitySeconds, reachableNodes, reachable);
    }

    // Moderate memory (0.20 - 0.70): extract only when most of the graph is stale and extraction is cheap.
    if (reachableFraction < 0.30 && timeToExtract < 0.25 * availableTime)
    {
      return new ReuseDecision(GraphReuseAction.Extract,
        $"only {reachableFraction:P0} of nodes reachable; extraction quick (~{timeToExtract:F1}s)",
        reachabilitySeconds, reachableNodes, reachable);
    }

    if (reachableFraction < 0.10 && timeToExtract < 0.50 * availableTime)
    {
      return new ReuseDecision(GraphReuseAction.Extract,
        $"only {reachableFraction:P0} of nodes reachable; extraction affordable (~{timeToExtract:F1}s)",
        reachabilitySeconds, reachableNodes, reachable);
    }

    // Otherwise: reuse.
    return new ReuseDecision(GraphReuseAction.Reuse, null, reachabilitySeconds, reachableNodes, reachable);
  }

#endregion

  #region Memory + Logging

  /// <summary>
  /// Current process memory usage as a fraction of the configured maximum
  /// (private bytes / MaxMemoryBytes). A gen-0 collection is forced first so the
  /// reading is not inflated by short-lived garbage.
  /// </summary>
  private static double CurrentMemoryUsedFraction(ParamsSearch paramsSearch, out long usedBytes)
  {
    GC.Collect(0);
    HardwareManager.ProcessMemoryInfo memInfo = HardwareManager.GetProcessMemoryInfo();
    usedBytes = memInfo.PrivateBytes;
    return usedBytes / (double)paramsSearch.MaxMemoryBytes;
  }


  /// <summary>
  /// Formats a used/max memory pair as "used/max GB" (binary GiB, matching how MaxMemoryBytes is configured).
  /// </summary>
  private static string MemoryGBString(long usedBytes, long maxBytes)
  {
    const double BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0;
    return $"{usedBytes / BYTES_PER_GB:F1}/{maxBytes / BYTES_PER_GB:F1} GB";
  }


  /// <summary>
  /// Logs (unconditionally, in yellow) a one-line summary of a graph EXTRACT decision and its stats.
  /// When GraphExtractor.LOG_EXTRACT_PHASE_TIMINGS is enabled, also logs a per-phase timing breakdown.
  /// </summary>
  private static void LogExtraction(GraphExtractor.ExtractResult ex, int searchRootN, string reason)
  {
    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
      $"Graph EXTRACT: {reason}; searchRootN={searchRootN:N0}, "
    + $"{ex.NodesBefore:N0} -> {ex.NodesAfter:N0} nodes ({ex.RetentionFraction:P1}), {ex.ElapsedSeconds * 1000:F0}ms");

    if (GraphExtractor.LOG_EXTRACT_PHASE_TIMINGS)
    {
      GraphRewriter.FinalizePhaseTimings f = ex.Finalize;
      // bfs = the single reachability BFS (shared by the decision and extraction; no longer doubled).
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        $"  EXTRACT timings (ms): bfs={ex.EnumerateSeconds * 1000:F1}, "
      + $"copy={ex.CopySeconds * 1000:F1} [{(ex.UsedParallelCopy ? "parallel" : "serial")}], "
      + $"finalize={ex.FinalizeSeconds * 1000:F1} "
      + $"(p5_parents={f.Phase5Parents * 1000:F1}, p5a_N/Q/D={f.Phase5aNodeN * 1000:F1}, "
      + $"p5b_root={f.Phase5bRoot * 1000:F1}, p6_dicts={f.Phase6Dicts * 1000:F1}, "
      + $"p6c_siblings={f.Phase6cSiblings * 1000:F1}, p7_cleanup={f.Phase7Cleanup * 1000:F1})");
    }
  }


  /// <summary>
  /// When phase timing is enabled (GraphExtractor.LOG_EXTRACT_PHASE_TIMINGS), logs the time spent in
  /// the reachability BFS check for a non-extract decision (reuse / abandon) — the case where that BFS
  /// cost is otherwise invisible. No-op when timing is disabled or no BFS ran (trivially-small or
  /// low-memory short-circuit, indicated by ReachableNodes &lt; 0).
  /// </summary>
  private static void MaybeLogReachabilityCheck(string outcome, ReuseDecision decision, Graph graph)
  {
    if (!GraphExtractor.LOG_EXTRACT_PHASE_TIMINGS || decision.ReachableNodes < 0)
    {
      return;
    }

    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
      $"  reachability check ({outcome}): {decision.ReachabilityCheckSeconds * 1000:F1}ms "
    + $"(reachable {decision.ReachableNodes:N0} of {graph.Store.NodesStore.NumUsedNodes:N0})");
  }


  /// <summary>
  /// Logs (unconditionally, in yellow) a one-line explanation of a graph ABANDON decision.
  /// </summary>
  private static void LogAbandon(Graph graph, GraphRootToSearchRootNodeInfo searchRootNodeInfo, string reason)
  {
    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
      $"Graph ABANDON: {reason} "
    + $"(graphRootN={graph.GraphRootNode.N:N0}, searchRootN={searchRootNodeInfo.ChildNode.N:N0})");
  }

  #endregion
}
