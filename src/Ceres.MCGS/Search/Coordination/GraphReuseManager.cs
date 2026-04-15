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
/// Manages graph reuse and rewrite decisions for MCGS search.
/// Encapsulates logic for determining when to reuse, rewrite, or abandon a graph.
/// </summary>
public static class GraphReuseManager
{
  #region Constants

  /// <summary>
  /// Minimum number of nodes before considering graph rewrite.
  /// </summary>
  public const int SMALL_GRAPH_THRESHOLD = 1_000_000;

  /// <summary>
  /// Memory usage fraction (relative to MaxMemory allocated to Ceres)
  /// above which rewrite is triggered.
  /// </summary>
  public const float MEMORY_PRESSURE_THRESHOLD = 0.90f;

  /// <summary>
  /// Ratio of search root N to total nodes below which rewrite is triggered.
  /// This is done because typically this indicates that much of the
  /// graph is not relevant to the current search root.
  /// </summary>
  public const float REWRITE_RATIO_THRESHOLD = 0.05f;

  /// <summary>
  /// Estimated reachability fraction (from sampling) below which the graph
  /// is abandoned outright, even if rewrite is enabled. The estimate is an
  /// upper bound, so the true reachable fraction is at most this value.
  /// </summary>
  public const float LOW_REACHABILITY_ABANDON_THRESHOLD = 0.15f;

  /// <summary>
  /// Maximum retention fraction for full rewrite.
  /// We can't retain a very high fraction because:
  ///   - would defeat purpose of rewrite (reduce memory usage), and
  ///   - the rewrite time is mostly a function of the number of nodes retained, 
  ///     so high retention could lead to problematic rewrite times
  ///     (although additional logic is in place to avoid that)
  /// </summary>
  public const float FULL_REWRITE_MAX_RETENTION = 0.60f;

  /// <summary>
  /// Target retention fraction for selective rewrite.
  /// </summary>
  public const float SELECTIVE_TARGET_RETENTION = 0.40f;

  /// <summary>
  /// Maximum fraction of search time allowed for rewrite.
  /// </summary>
  public const float MAX_REWRITE_TIME_FRACTION = 0.60f;

  /// <summary>
  /// Fraction of MaxNodes capacity above which rewrite is triggered
  /// to avoid running out of node slots during continued search.
  /// </summary>
  public const float NODE_CAPACITY_PRESSURE_THRESHOLD = 0.90f;

  #endregion

  #region Rewrite Time Estimator

  /// <summary>
  /// Adaptive estimator for graph rewrite time based on running averages with priors.
  /// Uses a pseudo-count weighted average approach where prior beliefs are treated
  /// as initial observations that get diluted as real observations accumulate.
  /// Thread-safe via locking.
  /// </summary>
  public static class RewriteTimeEstimator
  {
    // Prior pseudo-observations (equivalent to 1 observation each)
    const float PRIOR_SECONDS_PER_MILLION_RETAINED = 3.0f;
    const float PRIOR_SECONDS_PER_MILLION_STARTING = 0.050f;
    const float PRIOR_WEIGHT = 1.0f;

    // Running statistics
    static readonly object _lock = new();
    static float _sumRetainedNodesMillions = PRIOR_WEIGHT;
    static float _sumRetainedTimeSeconds = PRIOR_WEIGHT * PRIOR_SECONDS_PER_MILLION_RETAINED;
    static float _sumStartingNodesMillions = PRIOR_WEIGHT;
    static float _sumStartingOverheadSeconds = PRIOR_WEIGHT * PRIOR_SECONDS_PER_MILLION_STARTING;
    static int _observationCount = 0;

    /// <summary>
    /// Current estimate of seconds per million retained nodes.
    /// </summary>
    public static float SecondsPerMillionRetainedNodes
    {
      get
      {
        lock (_lock)
        {
          return _sumRetainedTimeSeconds / _sumRetainedNodesMillions;
        }
      }
    }

    /// <summary>
    /// Current estimate of overhead seconds per million starting nodes.
    /// </summary>
    public static float SecondsPerMillionStartingNodes
    {
      get
      {
        lock (_lock)
        {
          return _sumStartingOverheadSeconds / _sumStartingNodesMillions;
        }
      }
    }

    /// <summary>
    /// Number of observations recorded (excluding the prior pseudo-observation).
    /// </summary>
    public static int ObservationCount
    {
      get
      {
        lock (_lock)
        {
          return _observationCount;
        }
      }
    }

    /// <summary>
    /// Estimates the time required for a rewrite operation.
    /// </summary>
    /// <param name="startingNodes">Number of nodes in the graph before rewrite.</param>
    /// <param name="expectedRetainedNodes">Expected number of nodes after rewrite.</param>
    /// <returns>Estimated time in seconds.</returns>
    public static float EstimateRewriteTime(int startingNodes, int expectedRetainedNodes)
    {
      float startingMillions = startingNodes / 1_000_000f;
      float retainedMillions = expectedRetainedNodes / 1_000_000f;

      lock (_lock)
      {
        float overheadRate = _sumStartingOverheadSeconds / _sumStartingNodesMillions;
        float retentionRate = _sumRetainedTimeSeconds / _sumRetainedNodesMillions;
        return (startingMillions * overheadRate) + (retainedMillions * retentionRate);
      }
    }

    /// <summary>
    /// Calculates the maximum retention fraction that fits within a time budget.
    /// </summary>
    /// <param name="startingNodes">Number of nodes in the graph.</param>
    /// <param name="maxTimeSeconds">Maximum allowed time for rewrite.</param>
    /// <returns>Maximum retention fraction (0 to 1), or 0 if insufficient time for overhead.</returns>
    public static float MaxRetentionForTimeBudget(int startingNodes, float maxTimeSeconds)
    {
      float startingMillions = startingNodes / 1_000_000f;

      lock (_lock)
      {
        float overheadRate = _sumStartingOverheadSeconds / _sumStartingNodesMillions;
        float retentionRate = _sumRetainedTimeSeconds / _sumRetainedNodesMillions;

        float overheadSeconds = startingMillions * overheadRate;
        float availableTimeForRetention = maxTimeSeconds - overheadSeconds;

        if (availableTimeForRetention <= 0)
        {
          return 0;
        }

        // retainedMillions * retentionRate <= availableTimeForRetention
        float maxRetainedMillions = availableTimeForRetention / retentionRate;
        float maxRetainedNodes = maxRetainedMillions * 1_000_000f;
        return Math.Min(1.0f, maxRetainedNodes / startingNodes);
      }
    }

    /// <summary>
    /// Records an observation from a completed rewrite operation.
    /// Attributes the total elapsed time proportionally to overhead and retention
    /// based on current estimates, then updates the running averages.
    /// </summary>
    /// <param name="startingNodes">Number of nodes before rewrite.</param>
    /// <param name="retainedNodes">Number of nodes after rewrite.</param>
    /// <param name="elapsedSeconds">Total time taken for the rewrite.</param>
    public static void RecordObservation(int startingNodes, int retainedNodes, float elapsedSeconds)
    {
      if (elapsedSeconds <= 0 || startingNodes <= 0)
      {
        return;
      }

      float startingMillions = startingNodes / 1_000_000f;
      float retainedMillions = retainedNodes / 1_000_000f;

      lock (_lock)
      {
        // Attribute actual time proportionally based on current estimates
        float currentOverheadRate = _sumStartingOverheadSeconds / _sumStartingNodesMillions;
        float currentRetentionRate = _sumRetainedTimeSeconds / _sumRetainedNodesMillions;

        float estimatedOverhead = startingMillions * currentOverheadRate;
        float estimatedRetention = retainedMillions * currentRetentionRate;
        float totalEstimated = estimatedOverhead + estimatedRetention;

        float actualOverhead, actualRetention;
        if (totalEstimated > 0)
        {
          // Proportional attribution
          actualOverhead = elapsedSeconds * (estimatedOverhead / totalEstimated);
          actualRetention = elapsedSeconds * (estimatedRetention / totalEstimated);
        }
        else
        {
          // Fallback: split evenly
          actualOverhead = elapsedSeconds * 0.5f;
          actualRetention = elapsedSeconds * 0.5f;
        }

        // Update running totals
        _sumStartingNodesMillions += startingMillions;
        _sumStartingOverheadSeconds += actualOverhead;
        _sumRetainedNodesMillions += retainedMillions;
        _sumRetainedTimeSeconds += actualRetention;
        _observationCount++;
      }
    }

    /// <summary>
    /// Returns a formatted string with current estimates.
    /// </summary>
    public static string GetEstimatesString()
    {
      lock (_lock)
      {
        float retentionRate = _sumRetainedTimeSeconds / _sumRetainedNodesMillions;
        float overheadRate = _sumStartingOverheadSeconds / _sumStartingNodesMillions;
        return $"retentionRate={retentionRate:F3}s/M, overheadRate={overheadRate:F4}s/M, observations={_observationCount}";
      }
    }

    /// <summary>
    /// Resets the estimator to its initial state (priors only).
    /// </summary>
    public static void Reset()
    {
      lock (_lock)
      {
        _sumRetainedNodesMillions = PRIOR_WEIGHT;
        _sumRetainedTimeSeconds = PRIOR_WEIGHT * PRIOR_SECONDS_PER_MILLION_RETAINED;
        _sumStartingNodesMillions = PRIOR_WEIGHT;
        _sumStartingOverheadSeconds = PRIOR_WEIGHT * PRIOR_SECONDS_PER_MILLION_STARTING;
        _observationCount = 0;
      }
    }
  }

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
    if (searchRootPathFromGraphRoot == null || searchRootPathFromGraphRoot.Count < 2)
    {
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
  /// Processes a reusable graph, potentially performing a rewrite if triggered by
  /// memory pressure or low retention ratio. Returns the processed graph (which may
  /// be the original graph, a rewritten graph, or null if abandoned).
  /// </summary>
  /// <param name="graph">The graph to potentially rewrite.</param>
  /// <param name="searchRootNodeInfo">Info about the search root node.</param>
  /// <param name="searchRootPathFromGraphRoot">Path from graph root to search root (modified if rewrite occurs).</param>
  /// <param name="paramsSearch">Search parameters.</param>
  /// <param name="searchLimit">Current search limit (for time budget calculations).</param>
  /// <param name="priorMoves">Position history for the new search.</param>
  /// <returns>The graph to use (may be null if abandoned).</returns>
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

    int numNodesUsed = graph.Store.NodesStore.NumUsedNodes;
    int searchRootN = searchRootNodeInfo.ChildNode.N;
    float ratioSearchRootToTotal = (float)searchRootN / numNodesUsed;

    // Only consider rewrite for large graphs
    if (numNodesUsed < SMALL_GRAPH_THRESHOLD)
    {
      return graph;
    }

    // --- Graph rewrite/abandon decision logic ---
    // Three standard triggers: memory pressure, node capacity, low search-root ratio.
    // Additionally, a sampled reachability estimate is always computed for large graphs.
    // If estimated reachable fraction < LOW_REACHABILITY_ABANDON_THRESHOLD (upper bound),
    // the graph is abandoned outright even if rewrite is enabled, since rewriting a
    // nearly-unreachable graph is not worth the overhead.
    GC.Collect(0);
    HardwareManager.ProcessMemoryInfo memInfo = HardwareManager.GetProcessMemoryInfo();
    float memFraction = (float)(memInfo.PrivateBytes / (double)paramsSearch.MaxMemoryBytes);

    bool memoryPressure = memFraction >= MEMORY_PRESSURE_THRESHOLD;
    bool lowRatio = ratioSearchRootToTotal < REWRITE_RATIO_THRESHOLD;
    float nodeCapacityFraction = (float)numNodesUsed / graph.Store.MaxNodes;
    bool nodeCapacityPressure = nodeCapacityFraction >= NODE_CAPACITY_PRESSURE_THRESHOLD;
    bool standardTrigger = memoryPressure || nodeCapacityPressure || lowRatio;

    string rewriteReason = memoryPressure ? "memory pressure"
                         : nodeCapacityPressure ? "node capacity pressure"
                         : lowRatio ? "low ratio"
                         : null;

    // Always estimate reachability when graph is large enough to consider.
    float reachabilityFraction = EstimateReachability(graph, searchRootNodeInfo, priorMoves);
    bool lowReachability = reachabilityFraction < LOW_REACHABILITY_ABANDON_THRESHOLD;

    if (lowReachability)
    {
      // Low reachability forces abandonment regardless of rewrite setting.
      string reason = standardTrigger
        ? $"{rewriteReason} + low reachability ({reachabilityFraction:P1})"
        : $"low reachability ({reachabilityFraction:P1})";

      LogGraphAbandonment(graph, searchRootNodeInfo, searchRootN, numNodesUsed,
                          ratioSearchRootToTotal, memFraction, reason, priorMoves);

      graph.Dispose();
      searchRootPathFromGraphRoot = null;
      return null;
    }

    if (!standardTrigger)
    {
      return graph;
    }

    // Standard trigger fired with sufficient reachability.
    if (!paramsSearch.GraphReuseRewriteEnabled)
    {
      // Rewrite would have been triggered but feature is disabled.
      LogGraphAbandonment(graph, searchRootNodeInfo, searchRootN, numNodesUsed,
                          ratioSearchRootToTotal, memFraction, rewriteReason, priorMoves);

      graph.Dispose();
      searchRootPathFromGraphRoot = null;
      return null;
    }

    // Perform rewrite
    return PerformGraphRewrite(graph, searchRootNodeInfo, ref searchRootPathFromGraphRoot,
                               searchLimit, priorMoves, searchRootN, numNodesUsed,
                               ratioSearchRootToTotal, memFraction, rewriteReason);
  }

  /// <summary>
  /// Performs the actual graph rewrite operation.
  /// </summary>
  private static Graph PerformGraphRewrite(Graph graph,
                                           GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                           ref List<GraphRootToSearchRootNodeInfo> searchRootPathFromGraphRoot,
                                           SearchLimit searchLimit,
                                           PositionWithHistory priorMoves,
                                           int searchRootN,
                                           int numNodesUsed,
                                           float ratioSearchRootToTotal,
                                           float memFraction,
                                           string rewriteReason)
  {
    NodeIndex newRootIndex = searchRootNodeInfo.ChildNode.Index;

    // Calculate time budget constraint if using time-based search limit
    float fullRewriteMaxRetention = FULL_REWRITE_MAX_RETENTION;
    float selectiveTargetRetention = SELECTIVE_TARGET_RETENTION;
    string timeBudgetInfo = "";

    if (searchLimit.IsTimeLimit)
    {
      float maxRewriteTimeSeconds = searchLimit.Value * MAX_REWRITE_TIME_FRACTION;
      float maxRetentionFromTimeBudget = RewriteTimeEstimator.MaxRetentionForTimeBudget(numNodesUsed, maxRewriteTimeSeconds);

      if (maxRetentionFromTimeBudget <= 0)
      {
        // Not enough time even for overhead - skip rewrite entirely
        float estimatedOverhead = (numNodesUsed / 1_000_000f) * RewriteTimeEstimator.SecondsPerMillionStartingNodes;
        Log($"SKIPPING REWRITE (insufficient time budget): searchRootN={searchRootN:N0}, " +
            $"maxRewriteTime={maxRewriteTimeSeconds:F2}s, estimatedOverhead={estimatedOverhead:F2}s, " +
            $"[{RewriteTimeEstimator.GetEstimatesString()}]", red: true);
        return graph;
      }

      // Clamp retention targets to fit within time budget
      if (maxRetentionFromTimeBudget < fullRewriteMaxRetention)
      {
        fullRewriteMaxRetention = Math.Max(0.05f, maxRetentionFromTimeBudget);
        timeBudgetInfo = $", timeBudgetCappedFullRetention={fullRewriteMaxRetention:P1}";
      }
      if (maxRetentionFromTimeBudget < selectiveTargetRetention)
      {
        selectiveTargetRetention = Math.Max(0.05f, maxRetentionFromTimeBudget);
        timeBudgetInfo += $", timeBudgetCappedSelectiveRetention={selectiveTargetRetention:P1}";
      }
    }

    // Try full rewrite first with max retention threshold (possibly time-budget adjusted)
    GraphRewriter.RewriteResult result = GraphRewriter.MakeChildNewRoot(
      graph, newRootIndex, priorMoves,
      maxRetentionFraction: fullRewriteMaxRetention);

    // If full rewrite was declined, fall back to selective rewrite
    if (result.Outcome != GraphRewriter.RewriteOutcome.Rewritten)
    {
      result = GraphRewriter.MakeChildNewRootSelective(
        graph, newRootIndex, priorMoves,
        targetRetentionFraction: selectiveTargetRetention);
    }

    // Record observation for adaptive estimation
    if (result.Outcome == GraphRewriter.RewriteOutcome.Rewritten
     || result.Outcome == GraphRewriter.RewriteOutcome.RewrittenSelective)
    {
      RewriteTimeEstimator.RecordObservation(result.NodesBeforeRewrite, result.NodesAfterRewrite, (float)result.ElapsedSeconds);
    }

    // Log the outcome with current estimates
    LogRewriteResult(result, searchRootN, rewriteReason, memFraction, timeBudgetInfo);

    if (result.Outcome == GraphRewriter.RewriteOutcome.Rewritten
     || result.Outcome == GraphRewriter.RewriteOutcome.RewrittenSelective)
    {
      // After rewrite, the search root is now at the graph root index.
      searchRootPathFromGraphRoot = null;
    }

    return graph;
  }

  #endregion

  #region Logging

  /// <summary>
  /// Logs a message if diagnostics are enabled.
  /// </summary>
  private static void Log(string message, bool red = false)
  {
    if (MCGSParamsFixed.GRAPH_REWRITE_DUMP_REUSE_DIAGNOSTICS)
    {
      ConsoleUtils.WriteLineColored(red ? ConsoleColor.Red : ConsoleColor.Yellow, message);
    }
  }

  /// <summary>
  /// Estimates the fraction of graph nodes reachable from the search root
  /// using a fast sampling approach. Returns an upper bound.
  /// </summary>
  private static float EstimateReachability(Graph graph,
                                            GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                            PositionWithHistory priorMoves)
  {
    NodeIndex newRootIndex = searchRootNodeInfo.ChildNode.Index;
    GraphRewriter.RewriteResult estimateResult = GraphRewriter.MakeChildNewRoot(
      graph, newRootIndex, priorMoves,
      maxRetentionFraction: 0.001f,
      maxSampledReachabilityFraction: 0.001f);
    return estimateResult.RetentionFraction;
  }

  /// <summary>
  /// Logs graph abandonment information.
  /// </summary>
  private static void LogGraphAbandonment(Graph graph,
                                          GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                          int searchRootN,
                                          int numNodesUsed,
                                          float ratioSearchRootToTotal,
                                          float memFraction,
                                          string rewriteReason,
                                          PositionWithHistory priorMoves)
  {
    Log($"ABANDONING GRAPH: searchRootN={searchRootN:N0}, " +
        $"reason={rewriteReason}, numNodesUsed={numNodesUsed:N0}, " +
        $"ratio={ratioSearchRootToTotal:P1}, memFraction={memFraction:P1}", red: true);
  }

  /// <summary>
  /// Logs the result of a rewrite operation including current speed estimates.
  /// </summary>
  private static void LogRewriteResult(GraphRewriter.RewriteResult result,
                                       int searchRootN,
                                       string rewriteReason,
                                       float memFraction,
                                       string timeBudgetInfo)
  {
    double memReductionMB = (result.ManagedMemoryAtStart - result.ManagedMemoryAtEnd) / (1024.0 * 1024.0);

    // Calculate actual speed metrics for this rewrite
    float actualRetainedMillions = result.NodesAfterRewrite / 1_000_000f;
    float actualStartingMillions = result.NodesBeforeRewrite / 1_000_000f;
    string actualSpeedStr = "";
    if (result.ElapsedSeconds > 0 && actualRetainedMillions > 0)
    {
      // Note: This is the combined rate, not separated overhead vs retention
      float combinedRate = (float)result.ElapsedSeconds / actualRetainedMillions;
      actualSpeedStr = $", thisRewrite={combinedRate:F3}s/M";
    }

    string msg = $"Graph rewrite: searchRootN={searchRootN:N0}, reason={rewriteReason}, outcome={result.Outcome}, " +
      $"{result.NodesBeforeRewrite:N0} -> {result.NodesAfterRewrite:N0} nodes " +
      $"({100.0 * result.NodesAfterRewrite / result.NodesBeforeRewrite:F1}%), " +
      $"{result.ElapsedSeconds * 1000:F0}ms, mem delta {memReductionMB:+0.0;-0.0;0}MB{timeBudgetInfo}{actualSpeedStr}, " +
      $"[{RewriteTimeEstimator.GetEstimatesString()}]";
    Log(msg);
  }

  #endregion
}
