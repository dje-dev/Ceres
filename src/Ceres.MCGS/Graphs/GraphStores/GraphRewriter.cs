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
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs.Enumerators;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GParents;

#endregion

namespace Ceres.MCGS.Graphs.GraphStores;

/// <summary>
/// Compacts a Graph in-place after the root moves deeper (our move + opponent's move).
/// Removes nodes unreachable from the new root, rebuilds parent store and dictionaries.
/// Must be called between searches (no concurrent search activity).
///
/// IMPACT ON PLAYING STRENGTH (summary of considerations):
///
///   Positive factors:
///     1. Preserved neural network evaluations - retained nodes keep their NN values
///        and policy priors, avoiding costly re-evaluation from scratch [LARGE]
///     2. Smaller graph improves search speed - compacted dictionary and contiguous
///        node store improve cache locality; selection starts at true root rather
///        than traversing a graph-root-to-search-root prefix [MODERATE]
///     3. Reduced GC pressure - smaller live object set reduces garbage collector
///        scan times and pause frequency for subsequent search [MODERATE]
///
///   Negative factors:
///     4. Sibling (pseudo-transposition) information approximation - SiblingsQFrac/SiblingsQ
///        are cleared during edge N scaling, then reconstructed from the rebuilt transposition
///        dictionaries (Phase 6c); reconstruction is a single-pass approximation since sibling
///        Q values may not yet reflect their own sibling blending [SMALL]
///     5. Edge N discretization - proportional scaling with min-1 clamping introduces
///        rounding artifacts; NDrawByRepetition scaling can shift draw fractions;
///        RPO-regularized Q is replaced by pure MCTS Q until search resumes [SMALL]
///     6. Rewrite latency - synchronous operation blocks all search threads;
///        early-exit mechanisms (probe, sampling, BFS retention check) are critical
///        for avoiding wasted time on unproductive rewrites [VARIABLE]
///     7. Deferred node materialization - reachable deferred nodes must allocate
///        and copy policy data, slightly increasing edge header store usage [SMALL]
///
///   Net effect is strongly positive at medium-to-long time controls, where preserved
///   evaluations dominate. At fast time controls, rewrite latency can erode the benefit
///   unless early-exit guards effectively filter out unproductive rewrites.
/// </summary>
public static unsafe class GraphRewriter
{
  /// <summary>
  /// If true, validates that for each node, edges 0 through NumEdgesExpanded-1
  /// all have expanded headers. Throws an exception if this invariant is violated.
  /// </summary>
  const bool VALIDATE_EDGES_EXPANDED_IN_ORDER = false;

  /// <summary>
  /// If true, performs inline validation checks during rewrite phases to catch inconsistencies
  /// </summary>
  /// <summary>
  /// Controls ALL validation passes in the rewriter. Set to false for production performance.
  /// When true: adds ~10 validation passes (hash checks, edge-order checks, Q consistency,
  /// overlap detection, child-index bounds) that together add ~10-15% overhead.
  /// When false: all validation is skipped; only the core algorithm runs.
  /// </summary>
  const bool VALIDATE_INLINE = false;

  /// <summary>
  /// If true, Phase 6b uses level-synchronous parallel BFS for position+sequence
  /// dictionary rebuild. Each BFS level's children are processed via Parallel.For.
  /// </summary>
  const bool PARALLEL_BFS = true;

  /// <summary>
  /// Minimum frontier size to trigger parallel processing in level-synchronous BFS.
  /// Smaller frontiers are processed serially to avoid thread pool overhead.
  /// </summary>
  const int PARALLEL_BFS_THRESHOLD = 256;

  /// <summary>
  /// Per-phase timing breakdown for a rewrite operation (all values in seconds).
  /// </summary>
  public record PhaseTimings(double Phase1BFS, double Phase0Materialize, double Phase2CompactNodes,
                              double Phase3CompactEdgeHeaders, double Phase4CompactEdges,
                              double Phase5RebuildParents, double Phase5aRecalcN,
                              double Phase5bSetupRoot,
                              double Phase6RebuildDicts, double Phase6aStandalone, double Phase6bPosSeq,
                              double Phase6cSiblings,
                              double Phase7Cleanup);

  /// <summary>
  /// Outcome of a MakeChildNewRoot call.
  /// </summary>
  public enum RewriteOutcome
  {
    /// <summary>
    /// Rewrite completed successfully; graph was compacted.
    /// </summary>
    Rewritten,

    /// <summary>
    /// Declined: bounded probe from new root found an edge back to the old root,
    /// proving ~100% retention without running the full BFS.
    /// </summary>
    DeclinedRootReachable,

    /// <summary>
    /// Declined: full BFS determined the retention fraction exceeds the threshold,
    /// so compaction would be all cost with insufficient shrinkage.
    /// </summary>
    DeclinedInsufficientShrinkage,

    /// <summary>
    /// Declined: random sampling of IsPossiblyReachableFrom estimates with 95% confidence
    /// that the retention fraction would exceed maxSampledReachabilityFraction.
    /// Avoids the full BFS cost. The sampled fraction is heuristic (upper bound on actual
    /// retention) so this is a probabilistic estimate, not a guarantee.
    /// </summary>
    DeclinedEstimatedHighRetention,

    /// <summary>
    /// Declined: random sampling of IsPossiblyReachableFrom proves with 95% confidence
    /// that actual retention is below minSampledReachabilityFraction. Since IsPossiblyReachableFrom
    /// is a necessary condition (no false negatives), the sampled fraction is a rigorous
    /// upper bound on actual BFS retention — so if even this upper bound (with confidence
    /// margin) falls below the threshold, actual retention is provably below it.
    /// The graph has too little reusable content to justify the rewrite cost.
    /// </summary>
    DeclinedEstimatedLowRetention,

    /// <summary>
    /// Selective rewrite completed: edge-N threshold pruning removed low-value
    /// subgraphs while preserving the high-N backbone, even though the old root
    /// was reachable from the new root (which would normally cause ~100% retention).
    /// </summary>
    RewrittenSelective
  }

  /// <summary>
  /// Statistics returned by MakeChildNewRoot describing the rewrite operation.
  /// </summary>
  public record RewriteResult(int NodesBeforeRewrite, int NodesAfterRewrite,
                               int EdgeHeaderBlocksBefore, int EdgeHeaderBlocksAfter,
                               int EdgeBlocksBefore, int EdgeBlocksAfter,
                               double ElapsedSeconds, PhaseTimings Timings,
                               long ManagedMemoryAtStart, long ManagedMemoryAtEnd,
                               int NumDeferredMaterialized,
                               RewriteOutcome Outcome = RewriteOutcome.Rewritten,
                               float RetentionFraction = 0);


  /// <summary>
  /// Validates that for a given node, edges 0 through NumEdgesExpanded-1 all have expanded headers.
  /// Throws an exception if the invariant is violated (i.e., edges are not expanded in order).
  /// </summary>
  /// <param name="graph">The graph containing the node.</param>
  /// <param name="nodeIdx">The index of the node to validate.</param>
  /// <param name="phase">A description of which phase is performing the validation (for error messages).</param>
  static void ValidateEdgesExpandedInOrder(Graph graph, int nodeIdx, string phase)
  {
    ref GNodeStruct nodeRef = ref graph.NodesBufferOS[nodeIdx];
    int numExpanded = nodeRef.NumEdgesExpanded;

    if (numExpanded == 0)
    {
      return; // No edges to validate.
    }

    // Skip nodes without materialized edge headers.
    if (nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull ||
        nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
    {
      return;
    }

    int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
    Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, nodeRef.NumPolicyMoves);

    for (int i = 0; i < numExpanded; i++)
    {
      if (!headers[i].IsExpanded)
      {
        throw new Exception($"GraphRewriter ({phase}): Node {nodeIdx} has NumEdgesExpanded={numExpanded} " +
                            $"but edge header at index {i} is NOT expanded. " +
                            $"This violates the invariant that edges must be expanded in order 0, 1, 2, ..., NumEdgesExpanded-1.");
      }
    }
  }


  /// <summary>
  /// Validates that all retained nodes have edges expanded in order.
  /// Checks both headers (IsExpanded flag) and edge data (Type != Uninitialized).
  /// Throws if any node has an unexpanded edge within NumEdgesExpanded.
  /// </summary>
  static void ValidateAllEdgesExpandedInOrder(Graph graph, int numRetained, string phase)
  {
    int numTotalRetained = numRetained + 1;
    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];
      int numExpanded = nodeRef.NumEdgesExpanded;
      if (numExpanded == 0) continue;
      if (nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull ||
          nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
        continue;

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, nodeRef.NumPolicyMoves);

      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          throw new Exception($"GraphRewriter ({phase}): Node {newIdx} edge {i}/{numExpanded} " +
                              $"header NOT expanded (NumPolicy={nodeRef.NumPolicyMoves}).");
        }

        // Also validate the edge data at the pointed-to block.
        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

        if (edge.Type == GEdgeStruct.EdgeType.Uninitialized)
        {
          throw new Exception($"GraphRewriter ({phase}): Node {newIdx} edge {i}/{numExpanded} " +
                              $"header expanded (block={edgeBlock}) but edge data Uninitialized " +
                              $"(offsetInBlock={offsetInBlock}, NumPolicy={nodeRef.NumPolicyMoves}).");
        }
      }
    }
  }


  /// <summary>
  /// Compacts the graph so that newRootIndex becomes the new graph root at index 1.
  /// All unreachable nodes are removed and all stores are compacted in-place.
  /// </summary>
  public static RewriteResult MakeChildNewRoot(Graph graph, NodeIndex newRootIndex,
                                                PositionWithHistory newPriorMoves,
                                                float maxRetentionFraction,
                                                float minSampledReachabilityFraction = 0.0f,
                                                float maxSampledReachabilityFraction = 1.0f)
  {
    long memBefore = GC.GetTotalMemory(false);
    Stopwatch sw = Stopwatch.StartNew();

    if (newRootIndex.Index == GraphStore.ROOT_NODE_INDEX)
    {
      int n = graph.NodesStore.NumTotalNodes;
      PhaseTimings emptyTimings = new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      return new RewriteResult(n, n, 0, 0, 0, 0, 0, emptyTimings, memBefore, memBefore, 0);
    }

    Debug.Assert(!newRootIndex.IsNull);
    Debug.Assert(newRootIndex.Index > 0 && newRootIndex.Index < graph.NodesStore.NumTotalNodes);

    // Set rewriting flag to bypass lock assertions during exclusive graph access.
    graph.Store.IsRewriting = true;
    RewriteResult rewriteStats = MakeChildNewRootCore(graph, newRootIndex, newPriorMoves,
                                                       maxRetentionFraction, minSampledReachabilityFraction,
                                                       maxSampledReachabilityFraction, memBefore, sw);
    graph.Store.IsRewriting = false;
    return rewriteStats;
  }


  /// <summary>
  /// Selectively prunes a graph using edge-N threshold filtering.
  /// Used when the old root is reachable from the new root (making standard rewrite
  /// retain ~100% of nodes). Only follows edges with N >= threshold during BFS,
  /// severing low-N connections and pruning peripheral subgraphs.
  /// </summary>
  public static RewriteResult MakeChildNewRootSelective(Graph graph, NodeIndex newRootIndex,
                                                         PositionWithHistory newPriorMoves,
                                                         float targetRetentionFraction)
  {
    long memBefore = GC.GetTotalMemory(false);
    Stopwatch sw = Stopwatch.StartNew();

    if (newRootIndex.Index == GraphStore.ROOT_NODE_INDEX)
    {
      int n = graph.NodesStore.NumTotalNodes;
      PhaseTimings emptyTimings = new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      return new RewriteResult(n, n, 0, 0, 0, 0, 0, emptyTimings, memBefore, memBefore, 0);
    }

    Debug.Assert(!newRootIndex.IsNull);
    Debug.Assert(newRootIndex.Index > 0 && newRootIndex.Index < graph.NodesStore.NumTotalNodes);

    graph.Store.IsRewriting = true;
    RewriteResult result = MakeChildNewRootSelectiveCore(graph, newRootIndex, newPriorMoves,
                                                          targetRetentionFraction, memBefore, sw);
    graph.Store.IsRewriting = false;
    return result;
  }


  /// <summary>
  /// Core implementation of selective graph pruning.
  /// Computes an edge-N threshold targeting the desired retention fraction,
  /// then runs the standard rewrite phases with selective BFS and edge voiding.
  /// </summary>
  static RewriteResult MakeChildNewRootSelectiveCore(Graph graph, NodeIndex newRootIndex,
                                                      PositionWithHistory newPriorMoves,
                                                      float targetRetentionFraction,
                                                      long memBefore, Stopwatch sw)
  {
    int numNodes = graph.NodesStore.NumTotalNodes;
    int edgeHeaderBlocksBefore = graph.EdgeHeadersStore.NextFreeBlockIndex;
    int edgeBlocksBefore = graph.EdgesStore.nextFreeBlockIndex;

    // Compute edge-N threshold targeting the desired retention.
    int threshold = ComputeEdgeNThresholdForTargetRetention(graph, numNodes, targetRetentionFraction);

    // Phase 1: BFS with edge-N threshold.
    // Retry with increasing threshold if retention is too high (>90%).
    double t0 = sw.Elapsed.TotalSeconds;
    BitArray reachable = null;
    int numReachable = 0;
    float retentionFraction = 0;

    for (int attempt = 0; attempt < 3; attempt++)
    {
      reachable = Phase1FindReachableNodes(graph, newRootIndex, numNodes, out numReachable, minEdgeN: threshold);
      retentionFraction = (float)numReachable / (numNodes - 1);

      if (retentionFraction <= 0.90f || attempt == 2)
      {
        break;
      }

      // Too many survivors — double the threshold and retry.
      threshold = Math.Max(threshold * 2, threshold + 1);
    }

    double t1 = sw.Elapsed.TotalSeconds;

    // Phase 0: Materialize deferred policy copies.
    int numDeferred = Phase0MaterializeDeferredPolicyCopies(graph, numNodes, reachable);
    double t2 = sw.Elapsed.TotalSeconds;

    // Initialize scratch buffers (lazily created, grows on demand).
    int numTotalRetained = numReachable + 1;
    graph.RewriterScratchBuffers ??= new GraphRewriterScratchBuffers();
    GraphRewriterScratchBuffers scratch = graph.RewriterScratchBuffers;
    scratch.EnsureCapacity(numNodes, numTotalRetained);

    // Phase 2: Compact node store.
    Span<int> oldToNew = scratch.OldToNew(numNodes);
    Phase2CompactNodes(graph, reachable, newRootIndex, numNodes, numReachable, oldToNew);
    double t3 = sw.Elapsed.TotalSeconds;

    // Phase 3: Compact edge headers.
    Phase3CompactEdgeHeaders(graph, numReachable);
    double t4 = sw.Elapsed.TotalSeconds;

    // Phase 4: Compact edges, tolerating pruned children (voiding their edges).
    Phase4CompactEdges(graph, numReachable, scratch.OldToNewPtr, numNodes, toleratePrunedChildren: true);
    double t5 = sw.Elapsed.TotalSeconds;

    // Phase 4b: Compact voided edges within each node.
    Phase4bCompactPrunedEdges(graph, numReachable);
    double t4b = sw.Elapsed.TotalSeconds;

    // Phase 5: Rebuild parent store.
    Phase5RebuildParentStore(graph, numReachable, scratch);
    double t6 = sw.Elapsed.TotalSeconds;

    // Phase 5a: Recalculate N and Q.
    Phase5aRecalculateNodeN(graph, numReachable, scratch.GeneralAPtr);
    double t6a = sw.Elapsed.TotalSeconds;

    // Phase 5b: Setup root state.
    Phase5bSetupRootState(graph, newPriorMoves);
    double t7 = sw.Elapsed.TotalSeconds;

    // Phase 6: Rebuild dictionaries.
    Phase6RebuildDictionaries(graph, numReachable, scratch, out double phase6aTime, out double phase6bTime);
    double t8 = sw.Elapsed.TotalSeconds;

    // Phase 6c: Reconstruct sibling contributions.
    Phase6cPossiblyReconstructSiblingContributions(graph, numReachable);

    double t8c = sw.Elapsed.TotalSeconds;

    // Phase 7: Final cleanup.
    Phase7FinalCleanup(graph);
    sw.Stop();
    double t9 = sw.Elapsed.TotalSeconds;

    long memAfter = GC.GetTotalMemory(false);

    // Combine Phase 4 + 4b time into the Phase4 timing slot.
    PhaseTimings timings = new(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4b - t4, t6 - t4b, t6a - t6,
                               t7 - t6a, t8 - t7, phase6aTime, phase6bTime, t8c - t8, t9 - t8c);
    return new RewriteResult(numNodes, graph.NodesStore.NumTotalNodes,
                             edgeHeaderBlocksBefore, graph.EdgeHeadersStore.NextFreeBlockIndex,
                             edgeBlocksBefore, graph.EdgesStore.nextFreeBlockIndex,
                             t9 - t0, timings, memBefore, memAfter, numDeferred,
                             Outcome: RewriteOutcome.RewrittenSelective, RetentionFraction: retentionFraction);
  }


  /// <summary>
  /// Computes an edge-N threshold that targets the specified retention fraction.
  /// Builds a histogram of edge.N values across all expanded child edges, then
  /// walks from highest N downward, accumulating distinct reachable child nodes
  /// until the target count is reached.
  /// </summary>
  static int ComputeEdgeNThresholdForTargetRetention(Graph graph, int numNodes, float targetRetention)
  {
    int targetCount = (int)((numNodes - 1) * targetRetention);
    if (targetCount <= 1)
    {
      return int.MaxValue; // Only root survives.
    }

    // Collect all edge.N values for expanded child edges.
    // Use a dictionary for sparse histogram (edge.N can range widely).
    Dictionary<int, int> histogram = new();
    int maxEdgeN = 0;

    for (int i = 1; i < numNodes; i++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[i];
      int numExpanded = nodeRef.NumEdgesExpanded;
      if (numExpanded == 0
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, nodeRef.NumPolicyMoves);

      int lastEdgeBlock = -1;
      Span<GEdgeStruct> cachedEdgeSpan = default;
      for (int j = 0; j < numExpanded; j++)
      {
        if (!headers[j].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers[j].EdgeStoreBlockIndex;
        if (edgeBlock != lastEdgeBlock)
        {
          cachedEdgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
          lastEdgeBlock = edgeBlock;
        }
        int offsetInBlock = j % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref cachedEdgeSpan[offsetInBlock];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull && edge.N > 0)
        {
          int n = edge.N;
          if (histogram.TryGetValue(n, out int count))
          {
            histogram[n] = count + 1;
          }
          else
          {
            histogram[n] = 1;
          }
          if (n > maxEdgeN)
          {
            maxEdgeN = n;
          }
        }
      }
    }

    if (histogram.Count == 0)
    {
      return 1; // No edges found; threshold 1 will include everything.
    }

    // Sort histogram keys descending and walk from highest N downward.
    int[] sortedKeys = new int[histogram.Count];
    histogram.Keys.CopyTo(sortedKeys, 0);
    Array.Sort(sortedKeys);
    Array.Reverse(sortedKeys);

    // Accumulate edge count from highest N downward.
    // The threshold is the N value at which we've accumulated enough edges
    // to approximately reach the target node count.
    // (Each edge represents one connection; many edges can point to the same child,
    // but as a heuristic the edge count approximates distinct reachable nodes.)
    int accumulated = 0;
    int bestThreshold = 1;
    for (int k = 0; k < sortedKeys.Length; k++)
    {
      int edgeN = sortedKeys[k];
      accumulated += histogram[edgeN];
      if (accumulated >= targetCount)
      {
        bestThreshold = edgeN;
        break;
      }
      bestThreshold = edgeN;
    }

    // Ensure threshold is at least 1 to avoid including zero-N edges.
    return Math.Max(1, bestThreshold);
  }


  /// <summary>
  /// Core implementation of graph rewrite after the IsRewriting flag has been set.
  /// </summary>
  static RewriteResult MakeChildNewRootCore(Graph graph, NodeIndex newRootIndex,
                                            PositionWithHistory newPriorMoves,
                                            float maxRetentionFraction,
                                            float minSampledReachabilityFraction,
                                            float maxSampledReachabilityFraction,
                                            long memBefore, Stopwatch sw)
  {

    int numNodes = graph.NodesStore.NumTotalNodes;
    int edgeHeaderBlocksBefore = graph.EdgeHeadersStore.NextFreeBlockIndex;
    int edgeBlocksBefore = graph.EdgesStore.nextFreeBlockIndex;

    // DIAGNOSTIC: Check for pre-existing Q corruption before any rewrite phase.
    if (VALIDATE_INLINE)
    {
      for (int i = 1; i < numNodes; i++)
      {
        ref GNodeStruct preNode = ref graph.NodesBufferOS[i];
        if (preNode.N > 0 && !preNode.Terminal.IsTerminal() && Math.Abs(preNode.Q) > 2.0)
        {
          Console.WriteLine(
            $"  *** PRE-REWRITE Q corruption: node {i} Q={preNode.Q} N={preNode.N} " +
            $"NumExpanded={preNode.NumEdgesExpanded} NumPolicy={preNode.NumPolicyMoves} " +
            $"SibQFrac={preNode.SiblingsQFrac:F4} SibQ={preNode.SiblingsQ:F4} " +
            $"WinP={preNode.WinP} LossP={preNode.LossP} " +
            $"isNewRoot={i == newRootIndex.Index}");
        }
      }
    }

    // Quick probe: bounded BFS from new root to detect if old root is reachable.
    // If so, retention will be ~100% (old root is a hub connecting all subgraphs) → skip rewrite.
    const int PROBE_LIMIT = 10_000;
    bool anyHighRetentionCheck = maxRetentionFraction < 1.0f || maxSampledReachabilityFraction < 1.0f;
    bool anySamplingCheck = maxSampledReachabilityFraction < 1.0f || minSampledReachabilityFraction > 0.0f;

    if (anyHighRetentionCheck)
    {
      double tProbeStart = sw.Elapsed.TotalSeconds;
      bool oldRootReachable = ProbeOldRootReachable(graph, newRootIndex.Index, PROBE_LIMIT);
      double tProbeEnd = sw.Elapsed.TotalSeconds;

      if (oldRootReachable)
      {
        PhaseTimings probeTimings = new(tProbeEnd - tProbeStart, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        return new RewriteResult(numNodes, numNodes, 0, 0, 0, 0, tProbeEnd - tProbeStart, probeTimings,
                                 memBefore, memBefore, 0,
                                 Outcome: RewriteOutcome.DeclinedRootReachable, RetentionFraction: 1.0f);
      }
    }

    // Sample random nodes with IsPossiblyReachableFrom to estimate retention bounds.
    if (anySamplingCheck)
    {
      const int SAMPLE_COUNT = 10_000;
      double tSampleStart = sw.Elapsed.TotalSeconds;
      float estimatedRetention = SampleReachabilityFraction(graph, newRootIndex.Index, numNodes, SAMPLE_COUNT,
                                                             out int actualSampleCount);
      double tSampleEnd = sw.Elapsed.TotalSeconds;
      double sampleElapsed = tSampleEnd - tSampleStart;

      if (actualSampleCount >= 100)
      {
        float se = MathF.Sqrt(estimatedRetention * (1 - estimatedRetention) / actualSampleCount);

        // Low retention check: upper 95% CI bound < minSampledReachabilityFraction → provably too few survivors.
        // This is rigorous: IsPossiblyReachableFrom is a necessary condition, so the sampled
        // fraction is an upper bound on actual BFS retention.
        if (minSampledReachabilityFraction > 0.0f)
        {
          float upperBound95 = estimatedRetention + 1.96f * se;
          if (upperBound95 < minSampledReachabilityFraction)
          {
            PhaseTimings sampleTimings = new(sampleElapsed, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            return new RewriteResult(numNodes, numNodes, 0, 0, 0, 0, sampleElapsed, sampleTimings,
                                     memBefore, memBefore, 0,
                                     Outcome: RewriteOutcome.DeclinedEstimatedLowRetention,
                                     RetentionFraction: estimatedRetention);
          }
        }

        // High retention check: lower 95% CI bound > maxSampledReachabilityFraction → likely too many survivors.
        // This is heuristic: the sampled fraction is an upper bound on actual retention, so exceeding
        // the threshold doesn't guarantee actual retention exceeds it, but correlates strongly in practice.
        if (maxSampledReachabilityFraction < 1.0f)
        {
          float lowerBound95 = estimatedRetention - 1.96f * se;
          if (lowerBound95 > maxSampledReachabilityFraction)
          {
            PhaseTimings sampleTimings = new(sampleElapsed, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            return new RewriteResult(numNodes, numNodes, 0, 0, 0, 0, sampleElapsed, sampleTimings,
                                     memBefore, memBefore, 0,
                                     Outcome: RewriteOutcome.DeclinedEstimatedHighRetention,
                                     RetentionFraction: estimatedRetention);
          }
        }
      }
    }

    // Phase 1: BFS to find reachable nodes (runs before Phase 0 since
    // deferred nodes are always leaf nodes with no expanded edges).
    double t0 = sw.Elapsed.TotalSeconds;
    BitArray reachable = Phase1FindReachableNodes(graph, newRootIndex, numNodes, out int numReachable);
    double t1 = sw.Elapsed.TotalSeconds;

    // Abort early if retention fraction exceeds threshold (rewrite would be all cost, no benefit).
    float retentionFraction = (float)numReachable / (numNodes - 1); // -1 for null node at index 0
    if (retentionFraction > maxRetentionFraction)
    {
      PhaseTimings bfsOnlyTimings = new(t1 - t0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      return new RewriteResult(numNodes, numNodes, 0, 0, 0, 0, t1 - t0, bfsOnlyTimings,
                               memBefore, memBefore, 0,
                               Outcome: RewriteOutcome.DeclinedInsufficientShrinkage,
                               RetentionFraction: retentionFraction);
    }

    // Phase 0: Materialize deferred policy copies (only reachable nodes).
    int numDeferred = Phase0MaterializeDeferredPolicyCopies(graph, numNodes, reachable);
    double t2 = sw.Elapsed.TotalSeconds;

    // Initialize scratch buffers (lazily created, grows on demand).
    int numTotalRetained = numReachable + 1;
    graph.RewriterScratchBuffers ??= new GraphRewriterScratchBuffers();
    GraphRewriterScratchBuffers scratch = graph.RewriterScratchBuffers;
    scratch.EnsureCapacity(numNodes, numTotalRetained);

    // Snapshot hashes before compaction for validation.
    ulong[] hashSnapshot = null;
    if (VALIDATE_INLINE)
    {
      hashSnapshot = new ulong[numNodes];
      for (int i = 1; i < numNodes; i++)
      {
        if (reachable.Get(i))
        {
          hashSnapshot[i] = graph.NodesBufferOS[i].HashStandalone.Hash;
        }
      }
    }

    // Phase 2: Compact node store and build index mapping.
    Span<int> oldToNew = scratch.OldToNew(numNodes);
    Phase2CompactNodes(graph, reachable, newRootIndex, numNodes, numReachable, oldToNew);
    double t3 = sw.Elapsed.TotalSeconds;

    if (VALIDATE_INLINE)
    {
      ValidateAllEdgesExpandedInOrder(graph, numReachable, "after-Phase2");

      // Verify node hashes survived compaction.
      for (int oldIdx = 1; oldIdx < numNodes; oldIdx++)
      {
        int newIdx = oldToNew[oldIdx];
        if (newIdx > 0)
        {
          ulong actual = graph.NodesBufferOS[newIdx].HashStandalone.Hash;
          ulong expected = hashSnapshot[oldIdx];
          if (actual != expected)
          {
            throw new InvalidOperationException(
              $"Phase2 node hash corruption: old[{oldIdx}]→new[{newIdx}] " +
              $"expected={expected:X16} actual={actual:X16} " +
              $"newRootIndex={newRootIndex.Index} numReachable={numReachable}");
          }
        }
      }
    }

    // Phase 3: Compact edge headers store.
    Phase3CompactEdgeHeaders(graph, numReachable);
    double t4 = sw.Elapsed.TotalSeconds;

    if (VALIDATE_INLINE)
    {
      ValidateAllEdgesExpandedInOrder(graph, numReachable, "after-Phase3");
    }

    // Phase 4: Compact edge store and remap child node indices.
    Phase4CompactEdges(graph, numReachable, scratch.OldToNewPtr, numNodes);
    double t5 = sw.Elapsed.TotalSeconds;

    if (VALIDATE_INLINE)
    {
      ValidateAllEdgesExpandedInOrder(graph, numReachable, "after-Phase4");
    }

    // Phase 5: Rebuild parent store from edges.
    // Also accumulates incoming edge.N per child for Phase 5a (avoids a
    // second BFS and the expensive SumIncomingParentEdgeN reverse lookups).
    Span<int> incomingN = scratch.GeneralA(numTotalRetained);
    Phase5RebuildParentStore(graph, numReachable, scratch);
    double t6 = sw.Elapsed.TotalSeconds;

    // Phase 5a: Recalculate node N values using pre-accumulated incoming N.
    // Uses sequential scan instead of BFS + parent-store reverse lookups.
    Phase5aRecalculateNodeN(graph, numReachable, scratch.GeneralAPtr);

    // Validate Q consistency: recompute Q from edges and check for out-of-range values.
    if (VALIDATE_INLINE)
    {
      ValidateEdgeQConsistency(graph, numReachable);
    }

    if (VALIDATE_INLINE)
    {
      ValidateAllEdgesExpandedInOrder(graph, numReachable, "after-Phase5a");
    }
    double t6a = sw.Elapsed.TotalSeconds;

    // Phase 5b: Set root flags and update cached pointers/history.
    Phase5bSetupRootState(graph, newPriorMoves);
    double t7 = sw.Elapsed.TotalSeconds;

    if (VALIDATE_INLINE)
    {
      ValidateAllEdgesExpandedInOrder(graph, numReachable, "after-Phase5b");
    }

    // Phase 6: Rebuild dictionaries and index sets.
    Phase6RebuildDictionaries(graph, numReachable, scratch, out double phase6aTime, out double phase6bTime);
    double t8 = sw.Elapsed.TotalSeconds;

    // Phase 6c: Reconstruct sibling (pseudo-transposition) contributions.
    // Now that Phase 6 has rebuilt the transposition dictionaries, we can
    // look up siblings and restore blended Q values for all retained nodes.
    Phase6cPossiblyReconstructSiblingContributions(graph, numReachable);

    double t8c = sw.Elapsed.TotalSeconds;

    // Phase 7: Final cleanup (counters, resize, validate).
    Phase7FinalCleanup(graph);

    sw.Stop();
    double t9 = sw.Elapsed.TotalSeconds;

    long memAfter = GC.GetTotalMemory(false);

    PhaseTimings timings = new(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6a - t6,
                               t7 - t6a, t8 - t7, phase6aTime, phase6bTime, t8c - t8, t9 - t8c);
    return new RewriteResult(numNodes, graph.NodesStore.NumTotalNodes,
                             edgeHeaderBlocksBefore, graph.EdgeHeadersStore.NextFreeBlockIndex,
                             edgeBlocksBefore, graph.EdgesStore.nextFreeBlockIndex,
                             t9 - t0, timings, memBefore, memAfter, numDeferred);
  }


  /// <summary>
  /// Phase 0: Materialize deferred policy copies on reachable nodes only.
  /// Deferred nodes are always leaf nodes (no expanded edges), so Phase 1 BFS
  /// can safely run first to determine reachability.
  /// No locking needed — graph is quiescent during rewrite (exclusive access).
  /// Returns the number of deferred nodes that were materialized.
  /// </summary>
  static int Phase0MaterializeDeferredPolicyCopies(Graph graph, int numNodes, BitArray reachable)
  {
    int count = 0;
    for (int i = 1; i < numNodes; i++)
    {
      if (!reachable.Get(i))
      {
        continue;
      }

      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[i];
      if (nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        // Follow the deferral chain to find the actual source with policy data.
        GNode sourceNode = graph[nodeRef.edgeHeaderBlockIndexOrDeferredNode.NodeIndex];
        while (sourceNode.NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
        {
          sourceNode = graph[sourceNode.NodeRef.edgeHeaderBlockIndexOrDeferredNode.NodeIndex];
        }

        nodeRef.edgeHeaderBlockIndexOrDeferredNode.Clear();
        Graph.AllocateAndCopyPolicyValues(sourceNode, graph[i]);
        count++;
      }
    }

    return count;
  }


  /// <summary>
  /// Bounded BFS probe from the new root to detect if the old root (index 1) is reachable.
  /// If any edge within the first maxProbeNodes nodes points back to ROOT_NODE_INDEX,
  /// the old root is reachable, which cascades to ~100% retention (the old root is a hub
  /// connecting all sibling subgraphs). Returns true if old root found reachable.
  /// </summary>
  static bool ProbeOldRootReachable(Graph graph, int newRootIndex, int maxProbeNodes)
  {
    HashSet<int> visited = new HashSet<int>(maxProbeNodes);
    Queue<int> queue = new Queue<int>();

    visited.Add(newRootIndex);
    queue.Enqueue(newRootIndex);
    int probed = 0;

    while (queue.Count > 0 && probed < maxProbeNodes)
    {
      int nodeIdx = queue.Dequeue();
      probed++;

      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[nodeIdx];

      if (nodeRef.NumPolicyMoves == 0 || nodeRef.NumEdgesExpanded == 0)
      {
        continue;
      }

      if (nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull)
      {
        continue;
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      int numExpanded = nodeRef.NumEdgesExpanded;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlockIndex = headers[i].EdgeStoreBlockIndex;
        Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlockIndex);
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int childIdx = edge.ChildNodeIndex.Index;

          if (childIdx == GraphStore.ROOT_NODE_INDEX)
          {
            return true; // Old root reachable → ~100% retention certain.
          }

          if (visited.Add(childIdx))
          {
            queue.Enqueue(childIdx);
          }
        }
      }
    }

    return false;
  }


  /// <summary>
  /// Samples random nodes and checks IsPossiblyReachableFrom to estimate an upper bound
  /// on actual BFS retention. Returns the fraction of sampled nodes that pass the heuristic.
  /// The actual sample count is returned via out parameter (may be less than maxSamples
  /// if the graph is very small).
  /// </summary>
  static float SampleReachabilityFraction(Graph graph, int newRootIndex, int numNodes,
                                           int maxSamples, out int actualSampleCount)
  {
    int populationSize = numNodes - 2; // exclude null node (0) and new root
    actualSampleCount = Math.Min(maxSamples, populationSize);
    if (actualSampleCount < 100)
    {
      return 1.0f; // too few nodes to sample, assume worst case
    }

    // Pre-read the new root's reachability fields.
    byte rootNumPieces = graph.NodesBufferOS[newRootIndex].NumPieces;
    byte rootNumRank2Pawns = graph.NodesBufferOS[newRootIndex].NumRank2Pawns;

    // Generate random sample indices (avoid 0 and newRootIndex).
    int[] sampleIndices = new int[actualSampleCount];
    Random rand = new(42);
    for (int s = 0; s < actualSampleCount; s++)
    {
      int idx;
      do { idx = 1 + rand.Next(numNodes - 1); } while (idx == newRootIndex);
      sampleIndices[s] = idx;
    }

    // Check IsPossiblyReachableFrom in parallel.
    int reachableCount = 0;
    Parallel.ForEach(
      Partitioner.Create(0, actualSampleCount),
      () => 0,
      (range, _, localCount) =>
      {
        for (int i = range.Item1; i < range.Item2; i++)
        {
          GNodeStruct node = graph.NodesBufferOS[sampleIndices[i]];
          if (node.NumPieces <= rootNumPieces && node.NumRank2Pawns <= rootNumRank2Pawns)
          {
            localCount++;
          }
        }
        return localCount;
      },
      localCount => Interlocked.Add(ref reachableCount, localCount));

    return (float)reachableCount / actualSampleCount;
  }


  /// <summary>
  /// Phase 1: BFS from new root following child edges to find all reachable nodes.
  /// Returns a BitArray marking reachable nodes and the count of reachable nodes.
  /// </summary>
  static BitArray Phase1FindReachableNodes(Graph graph, NodeIndex newRootIndex, int numNodes,
                                            out int numReachable, int minEdgeN = 0)
  {
    BitArray reachable = new BitArray(numNodes);
    Queue<int> queue = new Queue<int>(numNodes / 2);

    reachable.Set(newRootIndex.Index, true);
    queue.Enqueue(newRootIndex.Index);
    int count = 1;

    while (queue.Count > 0)
    {
      int nodeIdx = queue.Dequeue();
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[nodeIdx];

      if (nodeRef.NumPolicyMoves == 0 || nodeRef.NumEdgesExpanded == 0)
      {
        continue;
      }

      // Should not happen after Phase 0, but be safe.
      if (nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      if (nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull)
      {
        continue;
      }

      // Validate that edges are expanded in order (if enabled).
      if (VALIDATE_EDGES_EXPANDED_IN_ORDER && VALIDATE_INLINE)
      {
        ValidateEdgesExpandedInOrder(graph, nodeIdx, "Phase1-BFS");
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      int numExpanded = nodeRef.NumEdgesExpanded;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

      int lastEdgeBlock = -1;
      Span<GEdgeStruct> cachedEdgeSpan = default;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlockIndex = headers[i].EdgeStoreBlockIndex;
        if (edgeBlockIndex != lastEdgeBlock)
        {
          cachedEdgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlockIndex);
          lastEdgeBlock = edgeBlockIndex;
        }
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref cachedEdgeSpan[offsetInBlock];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge
         && !edge.ChildNodeIndex.IsNull
         && edge.N >= minEdgeN)
        {
          int childIdx = edge.ChildNodeIndex.Index;
          if (!reachable.Get(childIdx))
          {
            reachable.Set(childIdx, true);
            queue.Enqueue(childIdx);
            count++;
          }
        }
      }
    }

    numReachable = count;
    return reachable;
  }


  /// <summary>
  /// Phase 2: Compact nodes into contiguous slots, placing new root at index 1.
  /// Populates the caller-provided oldToNew mapping (must be pre-zeroed, sized numNodes).
  /// </summary>
  static void Phase2CompactNodes(Graph graph, BitArray reachable, NodeIndex newRootIndex,
                                  int numNodes, int numReachable, Span<int> oldToNew)
  {
    // Pre-assign: new root must land at index 1 (ROOT_NODE_INDEX).
    oldToNew[newRootIndex.Index] = GraphStore.ROOT_NODE_INDEX;

    // Assign new indices for remaining reachable nodes.
    int nextFree = 2; // 0 = null, 1 = root
    for (int i = 1; i < numNodes; i++)
    {
      if (i != newRootIndex.Index && reachable.Get(i))
      {
        oldToNew[i] = nextFree++;
      }
    }

    Debug.Assert(nextFree == numReachable + 1); // +1 for the null node at index 0

    // Save the old root's data before overwriting index 1.
    // When the old root is reachable from the new root (e.g. via transpositions),
    // the copy below would destroy its data. We need the original for its new slot.
    GNodeStruct savedOldRoot = graph.NodesBufferOS[GraphStore.ROOT_NODE_INDEX];

    // Copy root node to slot 1 (if not already there).
    if (newRootIndex.Index != GraphStore.ROOT_NODE_INDEX)
    {
      graph.NodesBufferOS[GraphStore.ROOT_NODE_INDEX] = graph.NodesBufferOS[newRootIndex.Index];
    }

    // Nodes in [1, newRootIndex-1]: when all predecessors are reachable,
    // newIdx = oldIdx+1 (shifted up). But when some predecessors are pruned,
    // newIdx < oldIdx (shifted down). The shift direction is monotonic:
    // once nodes start shifting down, all subsequent nodes also shift down.
    // Use reverse pass for up-shifted nodes, forward pass for down-shifted.
    //
    // Pass A: REVERSE for up-shifted nodes (newIdx > oldIdx).
    for (int oldIdx = newRootIndex.Index - 1; oldIdx >= 1; oldIdx--)
    {
      int newIdx = oldToNew[oldIdx];
      if (newIdx <= 0 || newIdx == GraphStore.ROOT_NODE_INDEX || newIdx <= oldIdx)
      {
        continue; // Pruned, root, same-position, or down-shifted (handled in Pass B).
      }
      // Index 1 was overwritten by the new root — use saved copy.
      if (oldIdx == GraphStore.ROOT_NODE_INDEX)
      {
        graph.NodesBufferOS[newIdx] = savedOldRoot;
      }
      else
      {
        graph.NodesBufferOS[newIdx] = graph.NodesBufferOS[oldIdx];
      }
    }

    // Pass B: FORWARD for down-shifted nodes (newIdx < oldIdx) in [1, newRootIndex-1].
    for (int oldIdx = 1; oldIdx < newRootIndex.Index; oldIdx++)
    {
      int newIdx = oldToNew[oldIdx];
      if (newIdx <= 0 || newIdx == GraphStore.ROOT_NODE_INDEX || newIdx >= oldIdx)
      {
        continue; // Pruned, root, same-position, or up-shifted (handled in Pass A).
      }
      // oldIdx > 1 here (node 1 always shifts up to 2+ or is pruned).
      graph.NodesBufferOS[newIdx] = graph.NodesBufferOS[oldIdx];
    }

    // Pass C: FORWARD for nodes above newRootIndex (always newIdx <= oldIdx).
    for (int oldIdx = newRootIndex.Index + 1; oldIdx < numNodes; oldIdx++)
    {
      int newIdx = oldToNew[oldIdx];
      if (newIdx <= 0 || newIdx == GraphStore.ROOT_NODE_INDEX)
      {
        continue;
      }
      if (newIdx != oldIdx)
      {
        graph.NodesBufferOS[newIdx] = graph.NodesBufferOS[oldIdx];
      }
    }

    // Bulk-zero freed node slots to prevent stale data when space is reused.
    long clearCount = numNodes - nextFree;
    if (clearCount > 0)
    {
      graph.NodesStore.MemoryBufferOSStore.Clear(nextFree, clearCount);
    }

    // Update the store's next free index.
    graph.NodesStore.nextFreeIndex = nextFree;

    // Handle state vectors if present.
    if (graph.Store.HasState && graph.Store.AllStateVectors != null)
    {
      Half[][] oldStates = graph.Store.AllStateVectors;
      Half[][] newStates = new Half[oldStates.Length][];

      for (int oldIdx = 1; oldIdx < numNodes; oldIdx++)
      {
        int newIdx = oldToNew[oldIdx];
        if (newIdx > 0 && oldIdx < oldStates.Length && oldStates[oldIdx] != null)
        {
          newStates[newIdx] = oldStates[oldIdx];
        }
      }

      graph.Store.AllStateVectors = newStates;
      graph.NodesStore.AllStates = newStates;
    }
  }


  /// <summary>
  /// Helper record for Phase 3: tracks which edge header blocks need to be relocated.
  /// </summary>
  readonly record struct EdgeHeaderBlockFixup(int OldBlockIndex, int NodeIndex, byte NumPolicyMoves)
    : IComparable<EdgeHeaderBlockFixup>
  {
    public int CompareTo(EdgeHeaderBlockFixup other) => OldBlockIndex.CompareTo(other.OldBlockIndex);
  }


  /// <summary>
  /// Phase 3: Compact edge headers store. For each retained node with materialized
  /// edge headers, relocate the blocks forward (sorted by old block index for safe in-place copy).
  /// </summary>
  static void Phase3CompactEdgeHeaders(Graph graph, int numRetained)
  {
    List<EdgeHeaderBlockFixup> fixups = new List<EdgeHeaderBlockFixup>(numRetained);

    int numTotalRetained = numRetained + 1; // +1 for null node at 0
    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];
      if (nodeRef.NumPolicyMoves > 0
       && !nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
       && !nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        int oldBlock = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
        fixups.Add(new EdgeHeaderBlockFixup(oldBlock, newIdx, nodeRef.NumPolicyMoves));
      }
    }

    // Sort by old block index ascending to guarantee newBlock <= oldBlock for safe in-place copy.
    fixups.Sort();

    int nextFreeBlock = 1; // block 0 is reserved
    GEdgeHeadersStore edgeHeadersStore = graph.EdgeHeadersStore;

    for (int f = 0; f < fixups.Count; f++)
    {
      EdgeHeaderBlockFixup fixup = fixups[f];
      int numBlocks = (int)((fixup.NumPolicyMoves + GEdgeHeadersStore.NUM_EDGE_HEADERS_PER_BLOCK - 1)
                            / GEdgeHeadersStore.NUM_EDGE_HEADERS_PER_BLOCK);

      if (nextFreeBlock != fixup.OldBlockIndex)
      {
        edgeHeadersStore.CopyEntries(fixup.OldBlockIndex, nextFreeBlock, fixup.NumPolicyMoves);
      }

      // Update the node's edge header block index to point to the new location.
      graph.NodesBufferOS[fixup.NodeIndex].edgeHeaderBlockIndexOrDeferredNode =
        new EdgeHeaderBlockIndexOrNodeIndex(nextFreeBlock);

      nextFreeBlock += numBlocks;
    }

    // Bulk-zero freed edge header blocks to prevent stale data when space is reused.
    int oldNextFreeHeaderBlock = edgeHeadersStore.NextFreeBlockIndex;
    edgeHeadersStore.NextFreeBlockIndex = nextFreeBlock;
    long headerClearStart = (long)nextFreeBlock * GEdgeHeadersStore.NUM_EDGE_HEADERS_PER_BLOCK;
    long headerClearCount = (long)(oldNextFreeHeaderBlock - nextFreeBlock) * GEdgeHeadersStore.NUM_EDGE_HEADERS_PER_BLOCK;
    if (headerClearCount > 0)
    {
      edgeHeadersStore.MemoryBufferOSStore.Clear(headerClearStart, headerClearCount);
    }
  }


  /// <summary>
  /// Helper record for Phase 4: tracks which edge blocks need to be relocated.
  /// </summary>
  readonly record struct EdgeBlockFixup(int OldEdgeBlock, int OwnerNodeIndex, int HeaderIndex)
    : IComparable<EdgeBlockFixup>
  {
    public int CompareTo(EdgeBlockFixup other) => OldEdgeBlock.CompareTo(other.OldEdgeBlock);
  }


  /// <summary>
  /// Phase 4: Compact edge store blocks and remap ChildNodeIndex using oldToNew mapping.
  /// Also updates edge headers to point to new edge block locations.
  /// </summary>
  static void Phase4CompactEdges(Graph graph, int numRetained, int* oldToNew, int oldToNewLength,
                                  bool toleratePrunedChildren = false)
  {
    // First pass: collect all distinct edge blocks referenced by retained nodes.
    // Edge blocks are unique per node; duplicates only occur within consecutive
    // edges sharing a block (4 edges/block), so a per-node lastBlock check suffices.
    List<EdgeBlockFixup> fixups = new List<EdgeBlockFixup>(numRetained);

    int numTotalRetained = numRetained + 1;
    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];
      int numExpanded = nodeRef.NumEdgesExpanded;
      if (numExpanded == 0
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

      int lastBlock = -1;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        if (edgeBlock != lastBlock)
        {
          fixups.Add(new EdgeBlockFixup(edgeBlock, newIdx, i));
          lastBlock = edgeBlock;
        }
      }
    }

    // Sort by old edge block index ascending for safe in-place copy.
    fixups.Sort();

    // Detect duplicate edge blocks (two different nodes referencing the same block).
    // When duplicates exist, new block indices shift beyond the 1:1 mapping with old
    // blocks, causing newBlock > oldBlock for some fixups. This means later fixups'
    // source blocks can be overwritten by earlier fixups' copies.
    bool hasDuplicates = false;
    for (int f = 1; f < fixups.Count; f++)
    {
      if (fixups[f].OldEdgeBlock == fixups[f - 1].OldEdgeBlock)
      {
        hasDuplicates = true;
        break;
      }
    }

    // Per-fixup new block tracking (not a global map, since two fixups can
    // share the same OldEdgeBlock when a previous rewrite left two nodes
    // pointing to the same block; a Dictionary would overwrite the first mapping).
    int[] fixupNewBlocks = new int[fixups.Count];
    GEdgeStore edgesStore = graph.EdgesStore;
    int nextFreeEdgeBlock = 1; // block 0 reserved

    // When duplicates exist, save all source block data upfront to prevent
    // the in-place copy loop from overwriting later fixups' source blocks.
    // Each block has NUM_EDGES_PER_BLOCK edges. We flatten into a single array.
    GEdgeStruct[] savedBlocks = null;
    if (hasDuplicates)
    {
      savedBlocks = new GEdgeStruct[fixups.Count * GEdgeStore.NUM_EDGES_PER_BLOCK];
      for (int f = 0; f < fixups.Count; f++)
      {
        Span<GEdgeStruct> src = edgesStore.SpanAtBlockIndex(fixups[f].OldEdgeBlock);
        int baseIdx = f * GEdgeStore.NUM_EDGES_PER_BLOCK;
        for (int e = 0; e < GEdgeStore.NUM_EDGES_PER_BLOCK; e++)
        {
          savedBlocks[baseIdx + e] = src[e];
        }
      }
    }

    for (int f = 0; f < fixups.Count; f++)
    {
      int oldBlock = fixups[f].OldEdgeBlock;
      int newBlock = nextFreeEdgeBlock;
      fixupNewBlocks[f] = newBlock;

      // Copy block from old to new position.
      // When duplicates exist, read from saved copy to avoid stale data.
      if (savedBlocks != null)
      {
        Span<GEdgeStruct> dstSpan = edgesStore.SpanAtBlockIndex(newBlock);
        int baseIdx = f * GEdgeStore.NUM_EDGES_PER_BLOCK;
        for (int e = 0; e < GEdgeStore.NUM_EDGES_PER_BLOCK; e++)
        {
          dstSpan[e] = savedBlocks[baseIdx + e];
        }
      }
      else if (newBlock != oldBlock)
      {
        Span<GEdgeStruct> srcSpan = edgesStore.SpanAtBlockIndex(oldBlock);
        Span<GEdgeStruct> dstSpan = edgesStore.SpanAtBlockIndex(newBlock);
        srcSpan.CopyTo(dstSpan);
      }

      // Remap ChildNodeIndex for valid edges; zero stale trailing slots.
      // Determine how many edges in this block are within the owning node's expanded range.
      ref GNodeStruct ownerNodeRef = ref graph.NodesBufferOS[fixups[f].OwnerNodeIndex];
      int ownerNumExpanded = ownerNodeRef.NumEdgesExpanded;
      int headerIdx = fixups[f].HeaderIndex;
      int validEdgesInBlock = Math.Min(GEdgeStore.NUM_EDGES_PER_BLOCK, ownerNumExpanded - headerIdx);

      Span<GEdgeStruct> blockSpan = edgesStore.SpanAtBlockIndex(newBlock);
      for (int e = 0; e < GEdgeStore.NUM_EDGES_PER_BLOCK; e++)
      {
        if (e >= validEdgesInBlock)
        {
          // Trailing slot beyond the node's expanded edges — zero it.
          blockSpan[e] = default;
          continue;
        }

        ref GEdgeStruct edge = ref blockSpan[e];
        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int oldChildIdx = edge.ChildNodeIndex.Index;
          Debug.Assert(oldChildIdx < oldToNewLength,
            $"Edge ChildNodeIndex {oldChildIdx} exceeds node count {oldToNewLength} " +
            $"(owner node {fixups[f].OwnerNodeIndex}, edge slot {e}, block old={oldBlock} new={newBlock})");
          if (oldChildIdx >= oldToNewLength)
          {
            // Stale data from a prior graph state — zero the slot.
            blockSpan[e] = default;
            continue;
          }
          int newChildIdx = oldToNew[oldChildIdx];
          if (newChildIdx == 0 && toleratePrunedChildren)
          {
            // Selective pruning: child was below edge-N threshold.
            // Void the edge; Phase 4b will compact holes afterward.
            // Preserve P and Move so Phase 4b can revert header to unexpanded state.
            FP16 savedP = blockSpan[e].P;
            EncodedMove savedMove = blockSpan[e].Move;
            blockSpan[e] = default;
            blockSpan[e].P = savedP;
            blockSpan[e].Move = savedMove;
            continue;
          }
          Debug.Assert(newChildIdx > 0, $"Child node {oldChildIdx} of reachable parent must itself be reachable");
          edge.ChildNodeIndex = new NodeIndex(newChildIdx);
        }
      }

      nextFreeEdgeBlock++;
    }

    // Bulk-zero freed edge blocks to prevent stale data when space is reused.
    // MemoryBufferOSStore is MemoryBufferOS<GEdgeStructBlocked> where each element
    // is already one block of NUM_EDGES_PER_BLOCK edges, so index in block units directly.
    int oldNextFreeEdgeBlock = edgesStore.nextFreeBlockIndex;
    edgesStore.nextFreeBlockIndex = nextFreeEdgeBlock;

    long edgeClearStart = (long)nextFreeEdgeBlock;
    long edgeClearItems = (long)(oldNextFreeEdgeBlock - nextFreeEdgeBlock);
    if (edgeClearItems > 0)
    {
      edgesStore.MemoryBufferOSStore.Clear(edgeClearStart, edgeClearItems);
    }

    // Second pass: update edge headers per-fixup (not via global map).
    // Each fixup updates only its owner node's headers, ensuring that
    // even if two nodes shared the same old block, they get separate new blocks.
    for (int f = 0; f < fixups.Count; f++)
    {
      int ownerIdx = fixups[f].OwnerNodeIndex;
      int headerIdx = fixups[f].HeaderIndex;
      int oldBlock = fixups[f].OldEdgeBlock;
      int newBlock = fixupNewBlocks[f];

      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[ownerIdx];
      int numExpanded = nodeRef.NumEdgesExpanded;
      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

      // Update headers in this block group (up to NUM_EDGES_PER_BLOCK from headerIndex).
      int headerEnd = Math.Min(headerIdx + GEdgeStore.NUM_EDGES_PER_BLOCK, numExpanded);
      for (int h = headerIdx; h < headerEnd; h++)
      {
        if (headers[h].IsExpanded && headers[h].EdgeStoreBlockIndex == oldBlock)
        {
          headers[h].ForceSetEdgeBlockIndex(newBlock);
        }
      }
    }

    // Third pass: zero trailing unused slots in each node's last edge block.
    // After block compaction and remapping, trailing slots may retain stale data
    // from prior allocations (e.g. previous rewrite cycles). The invariant requires
    // that slots beyond the last expanded edge in a block have default ChildNodeIndex.
    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];
      int numExpanded = nodeRef.NumEdgesExpanded;
      if (numExpanded == 0
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int lastEdgeOffset = (numExpanded - 1) % GEdgeStore.NUM_EDGES_PER_BLOCK;
      if (lastEdgeOffset == GEdgeStore.NUM_EDGES_PER_BLOCK - 1)
      {
        continue; // Block fully used, no trailing slots.
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

      int lastExpandedIdx = numExpanded - 1;
      if (!headers[lastExpandedIdx].IsExpanded)
      {
        continue;
      }

      int lastEdgeBlock = headers[lastExpandedIdx].EdgeStoreBlockIndex;
      Span<GEdgeStruct> blockSpan = edgesStore.SpanAtBlockIndex(lastEdgeBlock);
      for (int e = lastEdgeOffset + 1; e < GEdgeStore.NUM_EDGES_PER_BLOCK; e++)
      {
        blockSpan[e] = default;
      }
    }
  }


  /// <summary>
  /// Phase 4b: Compact voided edges within each retained node's expanded range.
  /// After selective pruning, Phase 4 voids edges to pruned children (Type = Uninitialized),
  /// creating holes in positions 0..NumEdgesExpanded-1. The search engine requires
  /// contiguous expanded edges, so this pass compacts them.
  ///
  /// For each node with voided edges:
  /// 1. Collect surviving edge data into a temp buffer.
  /// 2. Write them back to contiguous positions 0..validCount-1, reusing existing blocks.
  /// 3. Shift unexpanded headers down to fill the gap.
  /// 4. Update NumEdgesExpanded and NumPolicyMoves.
  ///
  /// Block assignment: position j lives at offset (j % 4) within the block originally
  /// assigned to group (j / 4). All positions 0..numExpanded-1 have expanded headers
  /// with valid block indices from Phase 4, so the block for any target group is always available.
  /// </summary>
  static void Phase4bCompactPrunedEdges(Graph graph, int numRetained)
  {
    int numTotalRetained = numRetained + 1;
    GEdgeStore edgesStore = graph.EdgesStore;

    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];
      int numExpanded = nodeRef.NumEdgesExpanded;
      if (numExpanded == 0
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      int numPolicy = nodeRef.NumPolicyMoves;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, numPolicy);

      // Quick check: any voided edges?
      bool hasVoided = false;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }
        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        Span<GEdgeStruct> blockSpan = edgesStore.SpanAtBlockIndex(edgeBlock);
        if (blockSpan[offsetInBlock].Type == GEdgeStruct.EdgeType.Uninitialized)
        {
          hasVoided = true;
          break;
        }
      }

      if (!hasVoided)
      {
        continue;
      }

      // Collect surviving expanded edges and reverted (pruned) edge info.
      GEdgeStruct[] survivingEdges = new GEdgeStruct[numExpanded];
      GEdgeHeaderStruct[] revertedHeaders = new GEdgeHeaderStruct[numExpanded];
      int validCount = 0;
      int revertedCount = 0;

      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }
        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        Span<GEdgeStruct> blockSpan = edgesStore.SpanAtBlockIndex(edgeBlock);
        if (blockSpan[offsetInBlock].Type != GEdgeStruct.EdgeType.Uninitialized)
        {
          survivingEdges[validCount] = blockSpan[offsetInBlock];
          validCount++;
        }
        else
        {
          // Voided edge: recover P and Move from edge body (preserved by Phase 4),
          // and ActionV/ActionU from header (offsets 4-7, not overlapped by edgeBlockIndex).
#if ACTION_ENABLED
          revertedHeaders[revertedCount] = new GEdgeHeaderStruct(blockSpan[offsetInBlock].Move,
            blockSpan[offsetInBlock].P, headers[i].ActionV, headers[i].ActionU);
#else
          revertedHeaders[revertedCount] = new GEdgeHeaderStruct(blockSpan[offsetInBlock].Move,
            blockSpan[offsetInBlock].P, FP16.NaN, FP16.NaN);
#endif
          revertedCount++;
        }
      }

      // Save unexpanded headers (positions numExpanded..numPolicy-1) before they get overwritten.
      int numUnexpanded = numPolicy - numExpanded;
      GEdgeHeaderStruct[] unexpandedHeaders = null;
      if (numUnexpanded > 0)
      {
        unexpandedHeaders = new GEdgeHeaderStruct[numUnexpanded];
        for (int i = 0; i < numUnexpanded; i++)
        {
          unexpandedHeaders[i] = headers[numExpanded + i];
        }
      }

      // Save block indices for each original group before we overwrite headers.
      // Group g uses the block at header position g*NUM_EDGES_PER_BLOCK.
      int numOriginalGroups = (numExpanded + GEdgeStore.NUM_EDGES_PER_BLOCK - 1) / GEdgeStore.NUM_EDGES_PER_BLOCK;
      int[] groupBlocks = new int[numOriginalGroups];
      for (int g = 0; g < numOriginalGroups; g++)
      {
        groupBlocks[g] = headers[g * GEdgeStore.NUM_EDGES_PER_BLOCK].EdgeStoreBlockIndex;
      }

      // Write surviving edges to contiguous positions 0..validCount-1.
      for (int j = 0; j < validCount; j++)
      {
        int targetGroup = j / GEdgeStore.NUM_EDGES_PER_BLOCK;
        int targetOffset = j % GEdgeStore.NUM_EDGES_PER_BLOCK;
        int targetBlockIndex = groupBlocks[targetGroup];

        Span<GEdgeStruct> targetBlock = edgesStore.SpanAtBlockIndex(targetBlockIndex);
        targetBlock[targetOffset] = survivingEdges[j];

        headers[j].ForceSetEdgeBlockIndex(targetBlockIndex);
      }

      // Zero trailing edge slots in the last surviving block.
      if (validCount > 0)
      {
        int lastGroup = (validCount - 1) / GEdgeStore.NUM_EDGES_PER_BLOCK;
        int lastBlockIndex = groupBlocks[lastGroup];
        Span<GEdgeStruct> lastBlock = edgesStore.SpanAtBlockIndex(lastBlockIndex);
        int lastUsedOffset = (validCount - 1) % GEdgeStore.NUM_EDGES_PER_BLOCK;
        for (int e = lastUsedOffset + 1; e < GEdgeStore.NUM_EDGES_PER_BLOCK; e++)
        {
          lastBlock[e] = default;
        }
      }

      // Write reverted and original unexpanded headers (order doesn't matter here;
      // the second pass below will sort all unexpanded edges by P descending).
      int writePos = validCount;
      for (int i = 0; i < revertedCount; i++)
      {
        headers[writePos++] = revertedHeaders[i];
      }
      if (numUnexpanded > 0)
      {
        for (int i = 0; i < numUnexpanded; i++)
        {
          headers[writePos++] = unexpandedHeaders[i];
        }
      }

      // NumPolicyMoves is unchanged: validCount + revertedCount + numUnexpanded == numPolicy.
      // Only NumEdgesExpanded changes to reflect the compacted surviving expanded edges.
      Debug.Assert(writePos == numPolicy,
        $"Header count mismatch: wrote {writePos} but numPolicy={numPolicy}");
      nodeRef.NumEdgesExpanded = (byte)validCount;
    }

    // Second pass: restore non-ascending P ordering for unexpanded edges of ALL retained nodes.
    // The search engine's move ordering (CheckMoveOrderRearrangeAtIndex) may have rearranged
    // unexpanded edges by Q during search. After a selective rewrite the engine expects
    // unexpanded edges sorted by P descending, so restore that invariant.
    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef2 = ref graph.NodesBufferOS[newIdx];
      int numExp = nodeRef2.NumEdgesExpanded;
      int numPol = nodeRef2.NumPolicyMoves;
      int numUnexp = numPol - numExp;
      if (numUnexp <= 1
       || nodeRef2.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef2.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int hdrBlockIdx = nodeRef2.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> hdrs = graph.EdgeHeadersStore.SpanAtBlockIndex(hdrBlockIdx, numPol);

      // Insertion sort on hdrs[numExp..numPol-1] by P descending.
      for (int i = numExp + 1; i < numPol; i++)
      {
        GEdgeHeaderStruct key = hdrs[i];
        float keyP = (float)key.P;
        int j = i - 1;
        while (j >= numExp && (float)hdrs[j].P < keyP)
        {
          hdrs[j + 1] = hdrs[j];
          j--;
        }
        hdrs[j + 1] = key;
      }
    }
  }


  /// <summary>
  /// Toggle between single-pass BFS (creates parent edges during traversal) and
  /// three-pass approach (BFS discovery, then graph parents, then non-graph parents).
  /// 
  /// Performance considerations for large graphs (e.g., 20M nodes):
  /// 
  /// SINGLE_PASS_BFS (true):
  ///   - Single BFS traversal with deferred list for non-graph edges
  ///   - Memory: bool[N] (1 byte/node) + List of deferred edges (8 bytes each)
  ///   - Access pattern: BFS order is inherently cache-unfriendly for large graphs
  ///     because child nodes are scattered in memory. However, we touch each node
  ///     exactly once during discovery.
  ///   - The deferred list iteration at the end is sequential and cache-friendly.
  ///   - Pro: Eliminates one full graph traversal.
  ///   - Con: List allocation/growth for deferred edges; final foreach has random
  ///     memory access pattern when calling CreateParentEdge.
  /// 
  /// THREE_PASS (false):
  ///   - Pass 1 (BFS): Discover graph parents, store in int[N] array
  ///   - Pass 2: Sequential scan creating graph-parent edges
  ///   - Pass 3: Sequential scan creating non-graph parent edges
  ///   - Memory: int[N] (4 bytes/node) + BFS queue
  ///   - Access pattern: Passes 2 and 3 are sequential index scans (cache-friendly
  ///     for the index iteration itself), but each node's edge data is still
  ///     scattered in memory.
  ///   
  /// For 20M nodes at ~100ns DRAM latency:
  ///   - Both approaches are dominated by random memory access to edge structures.
  ///   - The BFS queue in both cases causes random access as nodes are visited.
  ///   - The three-pass approach has slightly more predictable memory access in
  ///     passes 2-3 (sequential index iteration), but the edge/header lookups
  ///     inside each iteration are still random.
  ///   - Single-pass saves one complete traversal (~20M node visits) but adds
  ///     list overhead for transposition edges.
  ///   
  /// Expected: Single-pass is likely ~10-20% faster on large graphs because:
  ///   1. One fewer complete graph traversal (saves ~20M random accesses)
  ///   2. Deferred list is typically small (transposition edges are a fraction of total)
  ///   3. Modern CPUs handle list growth efficiently with geometric allocation
  /// 
  /// The three-pass approach may be preferable if:
  ///   - Transposition rate is very high (>30% of edges are non-graph)
  ///   - Memory pressure is critical (int[] vs bool[] + List)
  ///   - Debugging/profiling benefits from separated phases
  /// </summary>
  const bool SINGLE_PASS_BFS = true;

  /// <summary>
  /// Phase 5: Rebuild parent store from scratch by walking all retained node edges.
  /// Uses BFS to discover graph parents first, ensuring position 0 in each node's parent list
  /// is always a BFS-graph parent (guaranteed cycle-free path to root).
  /// </summary>
  /// <remarks>
  /// The graph-parent invariant guarantees that CalcPosition (and any root-ascent logic) following
  /// position-0 parents will have O(depth) worst-case traversal with no possibility of cycles.
  /// </remarks>
  static void Phase5RebuildParentStore(Graph graph, int numRetained, GraphRewriterScratchBuffers scratch)
  {
    GParentsStore parentsStore = graph.ParentsStore;
    int numTotalRetained = numRetained + 1; // +1 for null node at 0

    // --- Clear existing parent data ---
    // Clear all previously-used detail segments to prevent stale data from
    // corrupting the parent chain rebuild. Segments are recycled (NextFreeBlockIndex
    // is reset to 1) but the block memory retains old entries, causing IsFull/IsLink
    // misreads in CreateParentEdge when stale non-zero values remain.
    int oldNextFree = parentsStore.DetailSegments.NextFreeBlockIndex;
    long clearCount = oldNextFree - 1;
    if (clearCount > 0)
    {
      parentsStore.DetailSegments.MemoryBufferOSStore.Clear(1, clearCount);
    }
    parentsStore.DetailSegments.NextFreeBlockIndex = 1;

    // Clear ParentsHeader for all retained nodes (including null node at 0).
    for (int i = 0; i < numTotalRetained; i++)
    {
      graph.NodesBufferOS[i].ParentsHeader = default;
    }

    int* incomingNPtr = scratch.GeneralAPtr;
    if (SINGLE_PASS_BFS)
    {
      Phase5RebuildParentStoreSinglePass(graph, parentsStore, numTotalRetained, incomingNPtr, scratch);
    }
    else
    {
      Phase5RebuildParentStoreThreePass(graph, parentsStore, numTotalRetained, incomingNPtr, scratch);
    }

#if DEBUG
    // Verify graph-parent invariant: following position-0 parents from every node reaches root.
    for (int nodeIdx = 2; nodeIdx < numTotalRetained; nodeIdx++)
    {
      int current = nodeIdx;
      int steps = 0;
      while (current != GraphStore.ROOT_NODE_INDEX)
      {
        Debug.Assert(steps++ < numTotalRetained, $"Graph-parent cycle detected at node {nodeIdx}");
        GParentsHeader hdr = graph.NodesBufferOS[current].ParentsHeader;
        Debug.Assert(!hdr.IsEmpty, $"Node {current} has no parent on graph-parent path from {nodeIdx}");
        current = hdr.IsDirectEntry
          ? hdr.AsDirectParentNodeIndex.Index
          : parentsStore.DetailSegments.SegmentRef(hdr.AsSegmentLinkIndex).Entries[0].AsDirectParentNodeIndex.Index;
      }
    }
#endif
  }


  /// <summary>
  /// Single-pass implementation: BFS traversal that creates parent edges during discovery.
  /// Graph-parent edges are created immediately when a node is first discovered (position 0).
  /// Non-graph edges (transpositions) are deferred to a list and created after BFS completes.
  /// </summary>
  static void Phase5RebuildParentStoreSinglePass(Graph graph, GParentsStore parentsStore, int numTotalRetained,
                                                  int* incomingN, GraphRewriterScratchBuffers scratch)
  {
    // Visited buffer already zeroed by EnsureCapacity; use int 0/1.
    int* visited = scratch.VisitedPtr;
    visited[GraphStore.ROOT_NODE_INDEX] = 1;

    // Deferred non-graph edges: (parentIdx, childIdx) pairs to create after BFS completes.
    List<(int parentIdx, int childIdx)> deferredEdges = new();

    Queue<int> bfsQueue = new(numTotalRetained / 2);
    bfsQueue.Enqueue(GraphStore.ROOT_NODE_INDEX);

    while (bfsQueue.Count > 0)
    {
      int parentIdx = bfsQueue.Dequeue();
      ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];
      int numExpanded = parentNode.NumEdgesExpanded;
      if (numExpanded == 0
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(
        headerBlockIdx, (int)parentNode.NumPolicyMoves);

      int lastEdgeBlock = -1;
      Span<GEdgeStruct> cachedEdgeSpan = default;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded) continue;
        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        if (edgeBlock != lastEdgeBlock)
        {
          cachedEdgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
          lastEdgeBlock = edgeBlock;
        }
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref cachedEdgeSpan[offsetInBlock];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int childIdx = edge.ChildNodeIndex.Index;
          incomingN[childIdx] += edge.N;
          if (visited[childIdx] == 0)
          {
            // First discovery: this is the graph parent — create immediately (position 0).
            visited[childIdx] = 1;
            bfsQueue.Enqueue(childIdx);
            parentsStore.CreateParentEdge(new NodeIndex(parentIdx), new NodeIndex(childIdx));
          }
          else
          {
            // Already visited: this is a cross-edge (transposition parent) — defer creation.
            deferredEdges.Add((parentIdx, childIdx));
          }
        }
      }
    }

    // Create deferred non-graph parent edges (appended after position 0).
    foreach ((int parentIdx, int childIdx) in deferredEdges)
    {
      parentsStore.CreateParentEdge(new NodeIndex(parentIdx), new NodeIndex(childIdx));
    }
  }


  /// <summary>
  /// Three-pass implementation: 
  ///   1. BFS to discover graph parents (stored in auxiliary array)
  ///   2. Sequential pass to create graph-parent edges (position 0)
  ///   3. Sequential pass to create non-graph parent edges
  /// </summary>
  static void Phase5RebuildParentStoreThreePass(Graph graph, GParentsStore parentsStore, int numTotalRetained,
                                                 int* incomingN, GraphRewriterScratchBuffers scratch)
  {
    // --- BFS to discover graph parents ---
    // This ensures that each node's first parent (position 0) lies on a cycle-free path to the root.
    // Reuse Visited buffer (already zeroed) for bfsGraphParent: 0 = no graph parent.
    Span<int> bfsGraphParent = numTotalRetained <= 1024
      ? stackalloc int[numTotalRetained]
      : scratch.Visited(numTotalRetained);
    if (numTotalRetained > 1024)
    {
      // Visited was already zeroed by EnsureCapacity, no extra clear needed.
    }
    else
    {
      bfsGraphParent.Clear();
    }

    Queue<int> bfsQueue = new(numTotalRetained / 2);
    bfsQueue.Enqueue(GraphStore.ROOT_NODE_INDEX);

    while (bfsQueue.Count > 0)
    {
      int parentIdx = bfsQueue.Dequeue();
      ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];
      int numExpanded = parentNode.NumEdgesExpanded;
      if (numExpanded == 0
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(
        headerBlockIdx, (int)parentNode.NumPolicyMoves);

      int lastEdgeBlock = -1;
      Span<GEdgeStruct> cachedEdgeSpan = default;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded) continue;
        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        if (edgeBlock != lastEdgeBlock)
        {
          cachedEdgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
          lastEdgeBlock = edgeBlock;
        }
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref cachedEdgeSpan[offsetInBlock];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int childIdx = edge.ChildNodeIndex.Index;
          incomingN[childIdx] += edge.N;
          if (bfsGraphParent[childIdx] == 0 && childIdx != GraphStore.ROOT_NODE_INDEX)
          {
            bfsGraphParent[childIdx] = parentIdx;
            bfsQueue.Enqueue(childIdx);
          }
        }
      }
    }

    // --- Pass 1: create graph-parent edges (position 0 for each node) ---
    // This establishes the graph-parent invariant by ensuring the first parent edge created
    // for each node is the one discovered via BFS (guaranteed cycle-free path to root).
    for (int childIdx = 2; childIdx < numTotalRetained; childIdx++)  // skip null(0) and root(1)
    {
      int graphParent = bfsGraphParent[childIdx];
      Debug.Assert(graphParent > 0, $"Node {childIdx} has no BFS graph parent");
      parentsStore.CreateParentEdge(new NodeIndex(graphParent), new NodeIndex(childIdx));
    }

    // --- Pass 2: create remaining (non-graph) parent edges ---
    // These are transposition parents that provide alternative paths but are not on the graph.
    for (int parentIdx = 1; parentIdx < numTotalRetained; parentIdx++)
    {
      ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];
      int numExpanded = parentNode.NumEdgesExpanded;
      if (numExpanded == 0
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(
        headerBlockIdx, (int)parentNode.NumPolicyMoves);

      int lastEdgeBlock3 = -1;
      Span<GEdgeStruct> cachedEdgeSpan3 = default;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded) continue;
        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        if (edgeBlock != lastEdgeBlock3)
        {
          cachedEdgeSpan3 = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
          lastEdgeBlock3 = edgeBlock;
        }
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref cachedEdgeSpan3[offsetInBlock];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int childIdx = edge.ChildNodeIndex.Index;
          if (parentIdx != bfsGraphParent[childIdx])  // skip graph parent (already created in Pass 1)
          {
            parentsStore.CreateParentEdge(new NodeIndex(parentIdx), new NodeIndex(childIdx));
          }
        }
      }
    }
  }


  /// <summary>
  /// Phase 5a: Recalculate N values for all retained nodes and scale outgoing edges.
  /// After pruning, some parent edges are removed but node N values still reflect
  /// visits from those pruned paths. Uses pre-accumulated incomingN from Phase 5's BFS
  /// instead of a separate BFS + SumIncomingParentEdgeN reverse lookups.
  ///   1. For each node: set N = incomingN[node] (pre-accumulated in Phase 5)
  ///   2. Scale outgoing child edge N values proportionally to maintain the invariant
  ///      that N = 1 + sum(outgoing edge N), i.e. sum(outgoing) = N - 1.
  /// The fixup passes then correct any over-estimation from using original (pre-scaling)
  /// edge.N values by iteratively capping edge.N to child.N until convergence.
  /// </summary>
  static void Phase5aRecalculateNodeN(Graph graph, int numRetained, int* incomingN)
  {
    int numTotalRetained = numRetained + 1;

    // Process root first: its N = sum(child edge N) + 1 (no incoming edges after rewrite).
    {
      ref GNodeStruct rootRef = ref graph.NodesBufferOS[GraphStore.ROOT_NODE_INDEX];
      if (!rootRef.Terminal.IsTerminal())
      {
        int sumChildN = SumChildEdgeN(graph, ref rootRef);
        rootRef.N = sumChildN + 1;
      }
    }

    // Process all non-root nodes using pre-accumulated incomingN.
    // No BFS needed: incomingN[childIdx] was accumulated during Phase 5's BFS
    // from original (pre-scaling) edge.N values.
    // Each node reads its own incomingN[childIdx] and writes only to its own
    // edges/fields, so this pass is embarrassingly parallel.
    const int PARALLEL_N_SCALING_THRESHOLD = 4096;
    if (numTotalRetained >= PARALLEL_N_SCALING_THRESHOLD)
    {
      Parallel.For(2, numTotalRetained, childIdx =>
      {
        Phase5aScaleNodeN(graph, childIdx, incomingN);
      });
    }
    else
    {
      for (int childIdx = 2; childIdx < numTotalRetained; childIdx++)
      {
        Phase5aScaleNodeN(graph, childIdx, incomingN);
      }
    }

    // Fixup pass: cap edge.N to child.N for all edges, recompute node.N.
    // In a DAG with transpositions, a higher-index node can have an edge to a
    // lower-index node. Processing high-to-low means the lower node's N hasn't
    // been finalized yet when the cap is applied. After the lower node's N
    // decreases, the cap becomes stale. Repeat until convergence.
    // The number of passes needed equals the longest back-edge chain in the DAG;
    // 3 is insufficient for large graphs (100K+ nodes). Since all values are
    // non-negative integers that can only decrease, convergence is guaranteed.
    for (int pass = 0; pass < 100; pass++)
    {
      bool anyChange = false;
      for (int i = numTotalRetained - 1; i >= GraphStore.ROOT_NODE_INDEX; i--)
      {
        ref GNodeStruct nodeRef = ref graph.NodesBufferOS[i];
        if (nodeRef.Terminal.IsTerminal())
        {
          continue;
        }

        int numExp = nodeRef.NumEdgesExpanded;
        if (numExp == 0
         || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
         || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
        {
          continue;
        }

        int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
        Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

        int fixedSum = 0;
        int lastEdgeBlock = -1;
        Span<GEdgeStruct> cachedEdgeSpan = default;
        for (int j = 0; j < numExp; j++)
        {
          if (!headers[j].IsExpanded)
          {
            continue;
          }

          int edgeBlock = headers[j].EdgeStoreBlockIndex;
          if (edgeBlock != lastEdgeBlock)
          {
            cachedEdgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
            lastEdgeBlock = edgeBlock;
          }
          int offsetInBlock = j % GEdgeStore.NUM_EDGES_PER_BLOCK;
          ref GEdgeStruct edge = ref cachedEdgeSpan[offsetInBlock];

          if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
          {
            int childN = graph.NodesBufferOS[edge.ChildNodeIndex.Index].N;
            if (edge.N > childN)
            {
              edge.N = childN;
              anyChange = true;
            }
          }

          // Cap NDrawByRepetition to edge.N after the edge.N cap.
          if (edge.NDrawByRepetition > edge.N)
          {
            edge.NDrawByRepetition = edge.N;
            anyChange = true;
          }

          fixedSum += edge.N;
        }

        int newN = fixedSum + 1;
        if (nodeRef.N != newN)
        {
          nodeRef.N = newN;
          anyChange = true;
        }
      }

      if (!anyChange)
      {
        break;
      }
    }

    // Q and D recomputation pass: bottom-up, compute Q and D from edges.
    // After fixup convergence, edge.N values are final. This pass computes
    // W = V + sum(-edge.Q * edge.N) and sets QPure = W/N.
    // D = (DrawP + sum(childD * (edge.N - NDrawByRepetition) + 1.0 * NDrawByRepetition)) / N
    // Also clears SiblingsQFrac/SiblingsQ (rebuilt in Phase 6c if enabled).
    for (int i = numTotalRetained - 1; i >= GraphStore.ROOT_NODE_INDEX; i--)
    {
      ref GNodeStruct nodeRef2 = ref graph.NodesBufferOS[i];
      if (nodeRef2.Terminal.IsTerminal() || nodeRef2.N == 0)
      {
        continue;
      }

      int numExp2 = nodeRef2.NumEdgesExpanded;
      if (numExp2 == 0
       || nodeRef2.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef2.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      // Compute W = V + sum(-edge.Q * edge.N) while refreshing edge.QChild.
      // Also compute D = DrawP + sum(childD * nNonRep + 1.0 * nDrawByRep) for all edges.
      // Bottom-up order ensures child Q and D values are already recomputed.
      float v = (float)nodeRef2.WinP - (float)nodeRef2.LossP;
      double w = v;
      double dSum = nodeRef2.DrawP; // self contribution (1 visit for initial eval)
      int headerBlockIdx2 = nodeRef2.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers2 = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx2, (int)nodeRef2.NumPolicyMoves);
      int lastEdgeBlock2 = -1;
      Span<GEdgeStruct> cachedEdgeSpan2 = default;
      for (int j = 0; j < numExp2; j++)
      {
        if (!headers2[j].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers2[j].EdgeStoreBlockIndex;
        if (edgeBlock != lastEdgeBlock2)
        {
          cachedEdgeSpan2 = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
          lastEdgeBlock2 = edgeBlock;
        }
        int offsetInBlock = j % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref cachedEdgeSpan2[offsetInBlock];
        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          double childQ = graph.NodesBufferOS[edge.ChildNodeIndex.Index].Q;
          double childD = graph.NodesBufferOS[edge.ChildNodeIndex.Index].D;
          if (!double.IsNaN(childQ))
          {
            edge.QChild = childQ;
            edge.IsStale = false;
          }
          if (edge.N > 0)
          {
            w += -edge.Q * edge.N;
            // D contribution: childD for non-repetition visits, 1.0 for draw-by-repetition visits
            int nNonRep = edge.N - edge.NDrawByRepetition;
            dSum += childD * nNonRep + 1.0 * edge.NDrawByRepetition;
          }
        }
        else if (edge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn && edge.N > 0)
        {
          // Terminal drawn edges contribute D=1.0 for all visits
          dSum += 1.0 * edge.N;
        }
        // TerminalEdgeDecisive: D=0, no contribution to dSum
      }

      if (VALIDATE_INLINE && graph.Store.UsesPositionEquivalenceMode)
      {
        if (nodeRef2.SiblingsQFrac != 0)
        {
          throw new InvalidOperationException($"Node {i} has siblings Q in use, not expected in PositionEquivalenceMode");
        }
      }

      // For N=1 nodes (all edge.N=0 after capping), Q must equal V and D must equal DrawP.
      if (nodeRef2.N == 1)
      {
        nodeRef2.Q = v;
        nodeRef2.D = nodeRef2.DrawP;
        nodeRef2.SiblingsQFrac = 0;
        nodeRef2.SiblingsQ = 0;
      }
      else if (nodeRef2.CheckmateKnownToExistAmongChildren)
      {
        // Checkmate-known nodes keep Q fixed as a win; D=0 (no draws possible).
        nodeRef2.D = 0;
        nodeRef2.SiblingsQFrac = 0;
        nodeRef2.SiblingsQ = 0;
      }
      else
      {
        double qPure = w / nodeRef2.N;
        nodeRef2.Q = qPure;
        nodeRef2.D = dSum / nodeRef2.N;
        nodeRef2.SiblingsQFrac = 0;
        nodeRef2.SiblingsQ = 0;

        // If draw is known to exist among children, Q must be non-negative.
        if (nodeRef2.DrawKnownToExistAmongChildren && nodeRef2.Q < 0)
        {
          nodeRef2.Q = 0;
        }
      }
    }
  }


  /// <summary>
  /// Scales a single non-root node's N and outgoing edges based on pre-accumulated incomingN.
  /// Thread-safe: reads incomingN[childIdx] (immutable) and writes only to this node's own fields/edges.
  /// </summary>
  static void Phase5aScaleNodeN(Graph graph, int childIdx, int* incomingN)
  {
    ref GNodeStruct childNode = ref graph.NodesBufferOS[childIdx];

    // Terminal nodes: their N represents visits to the terminal state.
    // They have no outgoing edges, so just set from accumulated incoming.
    if (childNode.Terminal.IsTerminal())
    {
      childNode.N = incomingN[childIdx];
      return;
    }

    int sumIncomingN = incomingN[childIdx];

    // Calculate current sum of outgoing child edge N values.
    int sumOutgoingN = SumChildEdgeN(graph, ref childNode);

    // Leaf nodes (no expanded edges): N = 1 (invariant: N = 1 + sum(outgoing) = 1).
    // Terminal nodes were handled above, so this is a non-terminal unexpanded node.
    if (sumOutgoingN == 0)
    {
      // Check if the node was never evaluated.
      // Such nodes can exist due to draw by repetition when in PositionEquivalence mode.
      // Set N = 0 to mark it as unvisited, preserving the invariant that N > 0 implies evaluated.
      if (!childNode.IsEvaluated)
      {
        childNode.N = 0;
        childNode.SetQNaN();
      }
      else
      {
        // Evaluated leaf node (no expanded edges or all edges pruned).
        // N=1 represents the single evaluation visit.
        childNode.N = 1;
        // For leaf nodes with N=1, Q must equal V and D must equal DrawP (the NN evaluation).
        // The Q/D recomputation pass skips leaf nodes, so we must set Q and D here.
        // Clear sibling fields to prevent ComputeQPure() from extracting a
        // wrong QPure when the search resumes backup through this node.
        childNode.Q = (float)childNode.WinP - (float)childNode.LossP;
        childNode.D = childNode.DrawP;
        childNode.SiblingsQFrac = 0;
        childNode.SiblingsQ = 0;
      }
      return;
    }

    // Scale outgoing edges proportionally.
    // Target: sum(outgoing) = sumIncomingN - 1 (so that N = 1 + sum(outgoing)).
    int targetOutgoing = sumIncomingN - 1;
    if (sumOutgoingN != targetOutgoing && sumOutgoingN > 0)
    {
      int actualOutgoing = ScaleOutgoingEdges(graph, ref childNode, targetOutgoing, sumOutgoingN);

      // Use the actual outgoing sum (may exceed target due to min-1 edge clamping)
      // to maintain the invariant N = 1 + sum(outgoing).
      childNode.N = actualOutgoing + 1;
    }
    else
    {
      childNode.N = sumIncomingN;
    }
  }


  /// <summary>
  /// Calculates the sum of N values for all outgoing child edges of a node.
  /// </summary>
  static int SumChildEdgeN(Graph graph, ref GNodeStruct nodeRef)
  {
    int numExpanded = nodeRef.NumEdgesExpanded;
    if (numExpanded == 0
     || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
     || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
    {
      return 0;
    }

    int sum = 0;
    int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
    Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

    for (int i = 0; i < numExpanded; i++)
    {
      if (!headers[i].IsExpanded)
      {
        continue;
      }

      int edgeBlock = headers[i].EdgeStoreBlockIndex;
      Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
      int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
      sum += edgeSpan[offsetInBlock].N;
    }

    return sum;
  }


  /// <summary>
  /// Calculates the sum of N values for all incoming parent edges to a node.
  /// </summary>
  static int SumIncomingParentEdgeN(Graph graph, int nodeIdx)
  {
    int sum = 0;
    ref GNodeStruct nodeRef = ref graph.NodesBufferOS[nodeIdx];

    // Use the ParentsHeader to iterate through parents.
    GParentsHeader parentsHeader = nodeRef.ParentsHeader;
    if (parentsHeader.IsEmpty)
    {
      return 0;
    }

    // Single parent case (direct entry).
    if (parentsHeader.IsDirectEntry)
    {
      NodeIndex parentIdx = parentsHeader.AsDirectParentNodeIndex;
      if (!parentIdx.IsNull)
      {
        sum += GetEdgeNFromParentToChild(graph, parentIdx.Index, nodeIdx);
      }
      return sum;
    }

    // Multiple parents: use the ParentIndexEnumerator.
    ParentIndexEnumerator enumerator = graph.ParentsStore.NodeParentsInfo(new NodeIndex(nodeIdx)).GetEnumerator();
    while (enumerator.MoveNext())
    {
      int parentIdxValue = enumerator.Current;
      sum += GetEdgeNFromParentToChild(graph, parentIdxValue, nodeIdx);
    }

    return sum;
  }


  /// <summary>
  /// Gets the N value of the edge from a specific parent to a specific child.
  /// </summary>
  static int GetEdgeNFromParentToChild(Graph graph, int parentIdx, int childIdx)
  {
    ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];

    int numExpanded = parentNode.NumEdgesExpanded;
    if (numExpanded == 0
     || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
     || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
    {
      return 0;
    }

    int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
    Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)parentNode.NumPolicyMoves);

    for (int i = 0; i < numExpanded; i++)
    {
      if (!headers[i].IsExpanded)
      {
        continue;
      }

      int edgeBlock = headers[i].EdgeStoreBlockIndex;
      Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
      int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
      ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

      if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && edge.ChildNodeIndex.Index == childIdx)
      {
        return edge.N;
      }
    }

    return 0;
  }


  /// <summary>
  /// Scales all outgoing child edge N values proportionally so their sum approximates targetSum.
  /// Uses rounding with remainder distribution. Each expanded edge is clamped to a minimum
  /// N of 1 to prevent creating N=0 child nodes (expanded edges have at least 1 visit).
  /// Returns the actual sum of outgoing edge N values after scaling.
  /// </summary>
  static int ScaleOutgoingEdges(Graph graph, ref GNodeStruct nodeRef, int targetSum, int currentSum)
  {
    if (currentSum == 0 || targetSum < 0)
    {
      return 0;
    }

    int numExpanded = nodeRef.NumEdgesExpanded;
    int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
    Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)nodeRef.NumPolicyMoves);

    double scaleFactor = (double)targetSum / currentSum;

    // First pass: compute scaled values with min-1 clamping and track remainders.
    Span<int> edgeIndices = stackalloc int[numExpanded];
    Span<int> scaledValues = stackalloc int[numExpanded];
    Span<double> remainders = stackalloc double[numExpanded];
    int edgeCount = 0;
    int scaledSum = 0;

    for (int i = 0; i < numExpanded; i++)
    {
      if (!headers[i].IsExpanded)
      {
        continue;
      }

      int edgeBlock = headers[i].EdgeStoreBlockIndex;
      Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
      int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
      ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

      double exactScaled = edge.N * scaleFactor;
      int rounded = Math.Max(1, (int)exactScaled);
      double remainder = exactScaled - (int)exactScaled;

      edgeIndices[edgeCount] = i;
      scaledValues[edgeCount] = rounded;
      remainders[edgeCount] = remainder;
      scaledSum += rounded;
      edgeCount++;
    }

    // Second pass: distribute remainder to edges with largest fractional parts
    // only if the sum is still below the target.
    int deficit = targetSum - scaledSum;
    if (deficit > 0)
    {
      for (int d = 0; d < deficit && d < edgeCount; d++)
      {
        int maxIdx = -1;
        double maxRemainder = -1;
        for (int j = 0; j < edgeCount; j++)
        {
          if (remainders[j] > maxRemainder)
          {
            maxRemainder = remainders[j];
            maxIdx = j;
          }
        }
        if (maxIdx >= 0)
        {
          scaledValues[maxIdx]++;
          scaledSum++;
          remainders[maxIdx] = -1;
        }
      }
    }

    // Third pass: apply scaled values to edges and scale NDrawByRepetition proportionally.
    int applyIdx = 0;
    for (int i = 0; i < numExpanded && applyIdx < edgeCount; i++)
    {
      if (!headers[i].IsExpanded)
      {
        continue;
      }

      int edgeBlock = headers[i].EdgeStoreBlockIndex;
      Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
      int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
      ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];
      int oldN = edge.N;
      int newN = scaledValues[applyIdx];
      edge.N = newN;

      // Scale NDrawByRepetition by the same proportion as N, capped to newN.
      if (edge.NDrawByRepetition > 0)
      {
        int scaledNDraw = oldN > 0
          ? Math.Max(0, (int)Math.Round((double)edge.NDrawByRepetition * newN / oldN))
          : 0;
        edge.NDrawByRepetition = Math.Min(scaledNDraw, newN);
      }

      applyIdx++;
    }

    return scaledSum;
  }


  /// <summary>
  /// Phase 5b: Set root node flags, update cached graph pointers, and update
  /// PositionHistory/HistoryHashes. Must run before Phase 6 because CalcPosition()
  /// relies on IsGraphRoot and PriorPositionsMG being correct.
  /// </summary>
  static unsafe void Phase5bSetupRootState(Graph graph, PositionWithHistory newPriorMoves)
  {
    // Clear IsSearchRoot and IsGraphRoot from all retained nodes to prevent
    // stale flags after compaction (the old root's copy retains these flags).
    int numTotal = graph.NodesStore.NumTotalNodes;
    for (int i = 1; i < numTotal; i++)
    {
      ref GNodeStruct n = ref graph.NodesBufferOS[i];
      n.miscFields.IsGraphRoot = false;
      n.miscFields.IsSearchRoot = false;
    }

    // Set root node flags.
    ref GNodeStruct rootNode = ref graph.NodesBufferOS[GraphStore.ROOT_NODE_INDEX];
    rootNode.miscFields.IsGraphRoot = true;
    rootNode.miscFields.IsSearchRoot = true;
    rootNode.ParentsHeader = default; // Root has no parents.

    // Update Graph cached pointers.
    graph.NodesRootNodePtr = graph.NodePtr(new NodeIndex(GraphStore.ROOT_NODE_INDEX));
    graph.GraphRootNode = graph[GraphStore.ROOT_NODE_INDEX];

    // Reset priorSearchRoot before calling SetSearchRootNode to avoid stale assert.
    graph.priorSearchRoot = default;
    graph.SetSearchRootNode(new NodeIndex(GraphStore.ROOT_NODE_INDEX));

    // Update PositionHistory and HistoryHashes (needed for CalcPosition root position).
    graph.Store.SetPriorMovesForRewrite(newPriorMoves);
  }


  /// <summary>
  /// Phase 6c: Reconstruct sibling (pseudo-transposition) contributions for all retained nodes.
  /// After Phase 6 rebuilds the transposition dictionaries, this pass looks up each node's
  /// siblings and restores the SiblingsQFrac/SiblingsQ fields that Phase 5a cleared to 0.
  /// Uses the existing ResetNodeQUsingNewQPure(qPure, refreshSiblingContribution: true) path
  /// which calls CalcPseudotranspositionContribution internally.
  /// </summary>
  static void Phase6cPossiblyReconstructSiblingContributions(Graph graph, int numRetained)
  {
    if (graph.Store.UsesPositionEquivalenceMode)
    {
      // Siblings not applicable/used in PositionEquivalence mode.
      // Noting to do.
    }
    else
    {
      int numTotalRetained = numRetained + 1;
      for (int i = GraphStore.ROOT_NODE_INDEX; i < numTotalRetained; i++)
      {
        ref GNodeStruct nodeRef = ref graph.NodesBufferOS[i];
        if (nodeRef.Terminal.IsTerminal() || nodeRef.N <= 1 || nodeRef.NumEdgesExpanded == 0
         || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
         || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
        {
          continue;
        }

        // nodeRef.Q is currently QPure (sibling fields are 0 from Phase 5a).
        GNode gnode = new GNode(graph, new NodeIndex(i));
        gnode.ResetNodeQUsingNewQPure(nodeRef.Q, refreshSiblingContribution: true);
      }
    }
  }


  /// <summary>
  /// Phase 6: Rebuild transposition dictionaries and NodeIndexSet store.
  /// The standalone and position+sequence dict rebuilds run in parallel
  /// since they write to independent dictionaries and read disjoint node fields.
  /// </summary>
  static void Phase6RebuildDictionaries(Graph graph, int numRetained, GraphRewriterScratchBuffers scratch,
                                         out double standaloneTime, out double posSeqTime)
  {
    int numTotalRetained = numRetained + 1;

    // Reset NodeIndexSetStore.
    graph.NodeIndexSetStore.nextFreeIndex = GNodeIndexSetStore.FIRST_ALLOCATED_INDEX;

    // Create fresh dictionaries upfront.
    const int DICTIONARY_CONCURRENCY = 16;
    graph.transpositionsPosStandalone =
      new Ceres.Base.DataTypes.ConcurrentDictionaryExtendible<PosHash64WithMove50AndReps, GNodeIndexSetIndex>(DICTIONARY_CONCURRENCY, numRetained);

    bool needPosSeqDict = graph.GraphEnabled
                       && !(Graph.SINGLE_DICTIONARY_POSITION_MODE && graph.Store.UsesPositionEquivalenceMode);

    if (needPosSeqDict)
    {
      graph.transpositionPositionAndSequence =
        new Ceres.Base.DataTypes.ConcurrentDictionaryExtendible<PosHash96MultisetFinalized, int>(DICTIONARY_CONCURRENCY, numRetained);
    }
    else
    {
      graph.transpositionPositionAndSequence = null;
    }

    // Run both dict rebuilds in parallel when the 96-bit dict is needed.
    double posSeqElapsed = 0;
    if (needPosSeqDict)
    {
      Task posSeqTask = Task.Run(() =>
      {
        Stopwatch psw = Stopwatch.StartNew();
        Phase6RebuildPosAndSequenceDict(graph, numTotalRetained, scratch);
        posSeqElapsed = psw.Elapsed.TotalSeconds;
      });

      Stopwatch ssw = Stopwatch.StartNew();
      Phase6RebuildStandaloneDictParallel(graph, numTotalRetained, scratch);
      standaloneTime = ssw.Elapsed.TotalSeconds;

      posSeqTask.Wait();
      posSeqTime = posSeqElapsed;
    }
    else
    {
      Stopwatch ssw = Stopwatch.StartNew();
      Phase6RebuildStandaloneDictParallel(graph, numTotalRetained, scratch);
      standaloneTime = ssw.Elapsed.TotalSeconds;
      posSeqTime = 0;
    }

    // Ensure underlying memory is committed for current usage.
    graph.NodeIndexSetStore.sets.InsureAllocated(graph.NodeIndexSetStore.NumTotalSets);
  }


  /// <summary>
  /// Minimum number of retained nodes to trigger parallel standalone dict rebuild.
  /// Below this threshold, sequential scan is faster (avoids thread pool overhead).
  /// </summary>
  const int PARALLEL_STANDALONE_DICT_THRESHOLD = 4096;

  /// <summary>
  /// Rebuilds the standalone transposition dictionary by scanning all retained nodes.
  /// Uses a two-pass approach for parallelism:
  ///   Pass 1 (parallel): TryAdd for unique keys — handles the vast majority of nodes.
  ///   Pass 2 (sequential): Handle collisions that need NodeIndexSet allocation.
  /// </summary>
  static void Phase6RebuildStandaloneDictParallel(Graph graph, int numTotalRetained, GraphRewriterScratchBuffers scratch)
  {
    if (numTotalRetained < PARALLEL_STANDALONE_DICT_THRESHOLD)
    {
      Phase6RebuildStandaloneDictSequential(graph, numTotalRetained);
      return;
    }

    // Pass 1: Parallel unique insertions via TryAdd.
    // Nodes that collide (TryAdd returns false) are collected for pass 2.
    // Limit parallelism to avoid starving Phase 6b's concurrent parallel BFS.
    // Reuse GeneralA buffer for collision nodes (Phase 5 incomingN is done).
    int* collisionNodesPtr = scratch.GeneralAPtr;
    int collisionCount = 0;

    int maxDop = Math.Max(2, System.Environment.ProcessorCount / 3);
    Parallel.For(1, numTotalRetained, new ParallelOptions { MaxDegreeOfParallelism = maxDop }, newIdx =>
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];

      if (nodeRef.miscFields.HasRepetitions)
      {
        return;
      }

      if ((int)nodeRef.miscFields.Move50Category > (int)Move50CategoryEnum.From76Thru90)
      {
        return;
      }

      PosHash64 hash64 = nodeRef.HashStandalone;
      PosHash64WithMove50AndReps key =
        MGPositionHashing.Hash64WithMove50AndRepsAdded(hash64, 0, nodeRef.miscFields.Move50Category);

      bool added = graph.transpositionsPosStandalone.TryAdd(key, GNodeIndexSetIndex.FromDirectNodeIndex(newIdx));
      if (!added)
      {
        int slot = Interlocked.Increment(ref collisionCount) - 1;
        collisionNodesPtr[slot] = newIdx;
      }
    });

    // Pass 2: Sequential collision handling (NodeIndexSet creation/extension).
    for (int c = 0; c < collisionCount; c++)
    {
      int newIdx = collisionNodesPtr[c];
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];

      PosHash64 hash64 = nodeRef.HashStandalone;
      PosHash64WithMove50AndReps key =
        MGPositionHashing.Hash64WithMove50AndRepsAdded(hash64, 0, nodeRef.miscFields.Move50Category);

      NodeIndex thisNodeIndex = new NodeIndex(newIdx);

      bool got = graph.transpositionsPosStandalone.TryGetValue(key, out GNodeIndexSetIndex existingEntry);
      if (!got || existingEntry.IsNull)
      {
        graph.transpositionsPosStandalone[key] = GNodeIndexSetIndex.FromDirectNodeIndex(newIdx);
      }
      else if (existingEntry.IsDirectNodeIndex)
      {
        int existingNodeIdx = existingEntry.DirectNodeIndex;
        int newSetIndex = graph.NodeIndexSetStore.AllocateNext();
        NodeIndexSet siblingSet = new();
        siblingSet.Add(new NodeIndex(existingNodeIdx), true);
        siblingSet.Add(thisNodeIndex, true);
        graph.NodeIndexSetStore.sets[newSetIndex] = siblingSet;
        graph.transpositionsPosStandalone[key] = GNodeIndexSetIndex.FromNodeSetIndex(newSetIndex);
      }
      else
      {
        int setIndex = existingEntry.NodeSetIndex;
        NodeIndexSet siblingSet = graph.NodeIndexSetStore.sets[setIndex];
        siblingSet.Add(thisNodeIndex, true);
        graph.NodeIndexSetStore.sets[setIndex] = siblingSet;
      }
    }
  }


  /// <summary>
  /// Sequential fallback for small graphs where parallel overhead exceeds the benefit.
  /// </summary>
  static void Phase6RebuildStandaloneDictSequential(Graph graph, int numTotalRetained)
  {
    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];

      if (nodeRef.miscFields.HasRepetitions)
      {
        continue;
      }

      if ((int)nodeRef.miscFields.Move50Category > (int)Move50CategoryEnum.From76Thru90)
      {
        continue;
      }

      PosHash64 hash64 = nodeRef.HashStandalone;
      PosHash64WithMove50AndReps key =
        MGPositionHashing.Hash64WithMove50AndRepsAdded(hash64, 0, nodeRef.miscFields.Move50Category);

      NodeIndex thisNodeIndex = new NodeIndex(newIdx);

      bool got = graph.transpositionsPosStandalone.TryGetValue(key, out GNodeIndexSetIndex existingEntry);
      if (!got || existingEntry.IsNull)
      {
        graph.transpositionsPosStandalone[key] = GNodeIndexSetIndex.FromDirectNodeIndex(newIdx);
      }
      else if (existingEntry.IsDirectNodeIndex)
      {
        int existingNodeIdx = existingEntry.DirectNodeIndex;
        int newSetIndex = graph.NodeIndexSetStore.AllocateNext();
        NodeIndexSet siblingSet = new();
        siblingSet.Add(new NodeIndex(existingNodeIdx), true);
        siblingSet.Add(thisNodeIndex, true);
        graph.NodeIndexSetStore.sets[newSetIndex] = siblingSet;
        graph.transpositionsPosStandalone[key] = GNodeIndexSetIndex.FromNodeSetIndex(newSetIndex);
      }
      else
      {
        int setIndex = existingEntry.NodeSetIndex;
        NodeIndexSet siblingSet = graph.NodeIndexSetStore.sets[setIndex];
        siblingSet.Add(thisNodeIndex, true);
        graph.NodeIndexSetStore.sets[setIndex] = siblingSet;
      }
    }
  }


  /// <summary>
  /// Rebuilds the standalone transposition dictionary into a custom
  /// ConcurrentDictionaryExtendible (for benchmarking purposes).
  /// Replicates the same scan and dict operations (TryGetValue + indexer set)
  /// but skips NodeIndexSetStore allocation since that cost is dict-independent.
  /// </summary>
  static void Phase6RebuildStandaloneDictCustom(Graph graph, int numTotalRetained,
                                                  ConcurrentDictionaryExtendible<PosHash64WithMove50AndReps, GNodeIndexSetIndex> customDict)
  {
    for (int newIdx = 1; newIdx < numTotalRetained; newIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[newIdx];

      if (nodeRef.miscFields.HasRepetitions)
      {
        continue;
      }

      if ((int)nodeRef.miscFields.Move50Category > (int)Move50CategoryEnum.From76Thru90)
      {
        continue;
      }

      PosHash64 hash64 = nodeRef.HashStandalone;
      PosHash64WithMove50AndReps key =
        MGPositionHashing.Hash64WithMove50AndRepsAdded(hash64, 0, nodeRef.miscFields.Move50Category);

      bool got = customDict.TryGetValue(key, out GNodeIndexSetIndex existingEntry);
      if (!got || existingEntry.IsNull)
      {
        customDict[key] = GNodeIndexSetIndex.FromDirectNodeIndex(newIdx);
      }
      else if (existingEntry.IsDirectNodeIndex)
      {
        // In the real rebuild this creates a NodeIndexSet; here we just update the dict
        // entry to a set-style index to replicate the same dict write pattern.
        customDict[key] = GNodeIndexSetIndex.FromNodeSetIndex(newIdx);
      }
      else
      {
        // In the real rebuild this adds to an existing set and doesn't update the dict.
        // We replicate the same no-dict-write path.
      }
    }
  }


  /// <summary>
  /// Rebuilds the position+sequence transposition dictionary via BFS from root.
  /// Positions are computed incrementally (parent + move) instead of via CalcPosition(),
  /// which also allows proper detection of irreversible moves to reset the running hash.
  /// Dispatches to parallel or serial BFS based on PARALLEL_BFS flag.
  /// </summary>
  static void Phase6RebuildPosAndSequenceDict(Graph graph, int numTotalRetained, GraphRewriterScratchBuffers scratch)
  {
    // RunningHashes already zeroed by EnsureCapacity (default = empty hash state).
    // Positions buffer written before read during BFS — no zeroing needed.
    MGPosition* positionsPtr = scratch.PositionsPtr;
    PosHash96MultisetRunning* runningHashesPtr = scratch.RunningHashesPtr;

    // Initialize root: position from the history, running hash from HistoryHashes.
    GNode rootNode = graph[GraphStore.ROOT_NODE_INDEX];
    MGPosition rootPos = rootNode.CalcPosition();
    positionsPtr[GraphStore.ROOT_NODE_INDEX] = rootPos;

    PosHash96MultisetRunning rootRunningHash = graph.Store.HistoryHashes.RunningHashAtEnd;
    runningHashesPtr[GraphStore.ROOT_NODE_INDEX] = rootRunningHash;

    // Register root node's finalized hash.
    PosHash96 rootHash96 = MGPositionHashing.Hash96(in rootPos);
    PosHash96MultisetFinalized rootFinalized = rootRunningHash.Finalized(rootHash96);
    graph.transpositionPositionAndSequence[rootFinalized] = GraphStore.ROOT_NODE_INDEX;

    if (PARALLEL_BFS)
    {
      Phase6PosSeqParallelBFS(graph, numTotalRetained, scratch);
    }
    else
    {
      Phase6PosSeqSerialBFS(graph, numTotalRetained, scratch);
    }
  }


  /// <summary>
  /// Rebuilds the position+sequence transposition dictionary into a custom
  /// ConcurrentDictionaryExtendible (for benchmarking purposes).
  /// Uses the same BFS logic as Phase6RebuildPosAndSequenceDict.
  /// </summary>
  static void Phase6RebuildPosAndSequenceDictCustom(Graph graph, int numTotalRetained,
                                                     ConcurrentDictionaryExtendible<PosHash96MultisetFinalized, int> customDict)
  {
    PosHash96MultisetRunning[] runningHashes = new PosHash96MultisetRunning[numTotalRetained];
    MGPosition[] positions = new MGPosition[numTotalRetained];

    GNode rootNode = graph[GraphStore.ROOT_NODE_INDEX];
    MGPosition rootPos = rootNode.CalcPosition();
    positions[GraphStore.ROOT_NODE_INDEX] = rootPos;

    PosHash96MultisetRunning rootRunningHash = graph.Store.HistoryHashes.RunningHashAtEnd;
    runningHashes[GraphStore.ROOT_NODE_INDEX] = rootRunningHash;

    PosHash96 rootHash96 = MGPositionHashing.Hash96(in rootPos);
    PosHash96MultisetFinalized rootFinalized = rootRunningHash.Finalized(rootHash96);
    customDict[rootFinalized] = GraphStore.ROOT_NODE_INDEX;

    if (PARALLEL_BFS)
    {
      Phase6PosSeqParallelBFSCustom(graph, numTotalRetained, positions, runningHashes, customDict);
    }
    else
    {
      Phase6PosSeqSerialBFSCustom(graph, numTotalRetained, positions, runningHashes, customDict);
    }
  }


  /// <summary>
  /// Serial BFS for custom dict benchmark.
  /// </summary>
  static void Phase6PosSeqSerialBFSCustom(Graph graph, int numTotalRetained,
                                            MGPosition[] positions,
                                            PosHash96MultisetRunning[] runningHashes,
                                            ConcurrentDictionaryExtendible<PosHash96MultisetFinalized, int> customDict)
  {
    bool[] visited = new bool[numTotalRetained];
    visited[GraphStore.ROOT_NODE_INDEX] = true;
    Queue<int> queue = new Queue<int>();
    queue.Enqueue(GraphStore.ROOT_NODE_INDEX);

    while (queue.Count > 0)
    {
      int parentIdx = queue.Dequeue();
      ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];

      int numExpanded = parentNode.NumEdgesExpanded;
      if (numExpanded == 0
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      MGPosition parentPos = positions[parentIdx];
      PosHash96 parentHash96 = MGPositionHashing.Hash96(in parentPos);
      PosHash96MultisetRunning runningAfterParent = runningHashes[parentIdx];
      runningAfterParent.Add(parentHash96);

      int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)parentNode.NumPolicyMoves);

      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

        if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeIndex.IsNull)
        {
          continue;
        }

        int childIdx = edge.ChildNodeIndex.Index;
        if (visited[childIdx])
        {
          continue;
        }

        visited[childIdx] = true;

        EncodedMove encodedMove = edge.Move;
        MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encodedMove, in parentPos);
        MGPosition childPos = parentPos;
        childPos.MakeMove(mgMove);
        positions[childIdx] = childPos;

        bool isIrreversible = parentPos.IsIrreversibleMove(mgMove, in childPos);
        PosHash96MultisetRunning childRunning = isIrreversible ? default : runningAfterParent;
        runningHashes[childIdx] = childRunning;

        PosHash96 childHash96 = MGPositionHashing.Hash96(in childPos);
        PosHash96MultisetFinalized childFinalized = childRunning.Finalized(childHash96);
        customDict.TryAdd(childFinalized, childIdx);

        queue.Enqueue(childIdx);
      }
    }
  }


  /// <summary>
  /// Parallel BFS for custom dict benchmark.
  /// </summary>
  static void Phase6PosSeqParallelBFSCustom(Graph graph, int numTotalRetained,
                                              MGPosition[] positions,
                                              PosHash96MultisetRunning[] runningHashes,
                                              ConcurrentDictionaryExtendible<PosHash96MultisetFinalized, int> customDict)
  {
    int[] visited = new int[numTotalRetained];
    visited[GraphStore.ROOT_NODE_INDEX] = 1;

    int[] frontierA = new int[numTotalRetained];
    int[] frontierB = new int[numTotalRetained];
    frontierA[0] = GraphStore.ROOT_NODE_INDEX;
    int frontierCount = 1;
    int[] currentFrontier = frontierA;
    int[] nextFrontier = frontierB;
    int[] nextCountBox = new int[1];

    while (frontierCount > 0)
    {
      nextCountBox[0] = 0;

      if (frontierCount >= PARALLEL_BFS_THRESHOLD)
      {
        Parallel.For(0, frontierCount, parentSlot =>
        {
          ExpandParentForPosSeqCustom(graph, currentFrontier[parentSlot],
                                      positions, runningHashes, visited,
                                      nextFrontier, nextCountBox, customDict);
        });
      }
      else
      {
        for (int s = 0; s < frontierCount; s++)
        {
          ExpandParentForPosSeqCustom(graph, currentFrontier[s],
                                      positions, runningHashes, visited,
                                      nextFrontier, nextCountBox, customDict);
        }
      }

      (currentFrontier, nextFrontier) = (nextFrontier, currentFrontier);
      frontierCount = nextCountBox[0];
    }
  }


  /// <summary>
  /// Expands a single parent for the custom dict parallel BFS benchmark.
  /// </summary>
  static void ExpandParentForPosSeqCustom(Graph graph, int parentIdx,
                                            MGPosition[] positions,
                                            PosHash96MultisetRunning[] runningHashes,
                                            int[] visited, int[] nextFrontier, int[] nextCountBox,
                                            ConcurrentDictionaryExtendible<PosHash96MultisetFinalized, int> customDict)
  {
    ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];

    int numExpanded = parentNode.NumEdgesExpanded;
    if (numExpanded == 0
     || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
     || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
    {
      return;
    }

    MGPosition parentPos = positions[parentIdx];
    PosHash96 parentHash96 = MGPositionHashing.Hash96(in parentPos);
    PosHash96MultisetRunning runningAfterParent = runningHashes[parentIdx];
    runningAfterParent.Add(parentHash96);

    int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
    Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)parentNode.NumPolicyMoves);

    for (int i = 0; i < numExpanded; i++)
    {
      if (!headers[i].IsExpanded)
      {
        continue;
      }

      int edgeBlock = headers[i].EdgeStoreBlockIndex;
      Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
      int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
      ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeIndex.IsNull)
      {
        continue;
      }

      int childIdx = edge.ChildNodeIndex.Index;

      if (Interlocked.CompareExchange(ref visited[childIdx], 1, 0) != 0)
      {
        continue;
      }

      EncodedMove encodedMove = edge.Move;
      MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encodedMove, in parentPos);
      MGPosition childPos = parentPos;
      childPos.MakeMove(mgMove);
      positions[childIdx] = childPos;

      bool isIrreversible = parentPos.IsIrreversibleMove(mgMove, in childPos);
      PosHash96MultisetRunning childRunning = isIrreversible ? default : runningAfterParent;
      runningHashes[childIdx] = childRunning;

      PosHash96 childHash96 = MGPositionHashing.Hash96(in childPos);
      PosHash96MultisetFinalized childFinalized = childRunning.Finalized(childHash96);
      customDict.TryAdd(childFinalized, childIdx);

      int slot = Interlocked.Increment(ref nextCountBox[0]) - 1;
      nextFrontier[slot] = childIdx;
    }
  }


  /// <summary>
  /// Serial Queue-based BFS for position+sequence dict rebuild.
  /// </summary>
  static void Phase6PosSeqSerialBFS(Graph graph, int numTotalRetained, GraphRewriterScratchBuffers scratch)
  {
    // Clear visited for reuse (may have been used by Phase 5).
    scratch.ClearVisited(numTotalRetained);
    int* visited = scratch.VisitedPtr;
    MGPosition* positions = scratch.PositionsPtr;
    PosHash96MultisetRunning* runningHashes = scratch.RunningHashesPtr;

    visited[GraphStore.ROOT_NODE_INDEX] = 1;
    Queue<int> queue = new Queue<int>();
    queue.Enqueue(GraphStore.ROOT_NODE_INDEX);

    while (queue.Count > 0)
    {
      int parentIdx = queue.Dequeue();
      ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];

      int numExpanded = parentNode.NumEdgesExpanded;
      if (numExpanded == 0
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      MGPosition parentPos = positions[parentIdx];
      PosHash96 parentHash96 = MGPositionHashing.Hash96(in parentPos);
      PosHash96MultisetRunning runningAfterParent = runningHashes[parentIdx];
      runningAfterParent.Add(parentHash96);

      int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)parentNode.NumPolicyMoves);

      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

        if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeIndex.IsNull)
        {
          continue;
        }

        int childIdx = edge.ChildNodeIndex.Index;
        if (visited[childIdx] != 0)
        {
          continue;
        }

        visited[childIdx] = 1;

        EncodedMove encodedMove = edge.Move;
        MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encodedMove, in parentPos);
        MGPosition childPos = parentPos;
        childPos.MakeMove(mgMove);
        positions[childIdx] = childPos;

        bool isIrreversible = parentPos.IsIrreversibleMove(mgMove, in childPos);
        PosHash96MultisetRunning childRunning = isIrreversible ? default : runningAfterParent;
        runningHashes[childIdx] = childRunning;

        PosHash96 childHash96 = MGPositionHashing.Hash96(in childPos);
        PosHash96MultisetFinalized childFinalized = childRunning.Finalized(childHash96);
        graph.transpositionPositionAndSequence.TryAdd(childFinalized, childIdx);

        queue.Enqueue(childIdx);
      }
    }
  }


  /// <summary>
  /// Level-synchronous parallel BFS for position+sequence dict rebuild.
  /// Each BFS level's children are processed via Parallel.For for large frontiers,
  /// parallelizing the expensive position computation, hashing, and dict registration.
  /// </summary>
  static void Phase6PosSeqParallelBFS(Graph graph, int numTotalRetained, GraphRewriterScratchBuffers scratch)
  {
    // Clear visited for reuse (may have been used by Phase 5).
    scratch.ClearVisited(numTotalRetained);
    int* visited = scratch.VisitedPtr;
    MGPosition* positions = scratch.PositionsPtr;
    PosHash96MultisetRunning* runningHashes = scratch.RunningHashesPtr;
    int* frontierAPtr = scratch.FrontierAPtr;
    int* frontierBPtr = scratch.FrontierBPtr;

    visited[GraphStore.ROOT_NODE_INDEX] = 1;

    // Double-buffered frontiers pre-allocated to max size.
    frontierAPtr[0] = GraphStore.ROOT_NODE_INDEX;
    int frontierCount = 1;
    int* currentFrontier = frontierAPtr;
    int* nextFrontier = frontierBPtr;

    while (frontierCount > 0)
    {
      scratch.NextCount = 0;

      if (frontierCount >= PARALLEL_BFS_THRESHOLD)
      {
        Parallel.For(0, frontierCount, parentSlot =>
        {
          ExpandParentForPosSeq(graph, currentFrontier[parentSlot],
                                positions, runningHashes, visited,
                                nextFrontier, scratch);
        });
      }
      else
      {
        for (int s = 0; s < frontierCount; s++)
        {
          ExpandParentForPosSeq(graph, currentFrontier[s],
                                positions, runningHashes, visited,
                                nextFrontier, scratch);
        }
      }

      // Swap frontiers.
      int* temp = currentFrontier;
      currentFrontier = nextFrontier;
      nextFrontier = temp;
      frontierCount = scratch.NextCount;
    }
  }


  /// <summary>
  /// Expands a single parent node's children for the position+sequence dict BFS.
  /// Computes child positions incrementally, registers in the transposition dictionary,
  /// and adds newly visited children to the next frontier via atomic operations.
  /// </summary>
  static void ExpandParentForPosSeq(Graph graph, int parentIdx,
                                     MGPosition* positions,
                                     PosHash96MultisetRunning* runningHashes,
                                     int* visited, int* nextFrontier,
                                     GraphRewriterScratchBuffers scratch)
  {
    ref GNodeStruct parentNode = ref graph.NodesBufferOS[parentIdx];

    int numExpanded = parentNode.NumEdgesExpanded;
    if (numExpanded == 0
     || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNull
     || parentNode.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
    {
      return;
    }

    MGPosition parentPos = positions[parentIdx];

    // Verify parent BFS-computed position matches stored hash.
    if (VALIDATE_INLINE)
    {
      PosHash64 bfsHash = MGPositionHashing.Hash64(in parentPos);
      if (bfsHash.Hash != parentNode.HashStandalone.Hash)
      {
        throw new InvalidOperationException(
          $"Phase6 BFS position mismatch at parent={parentIdx}: " +
          $"bfsHash={bfsHash.Hash:X16} stored={parentNode.HashStandalone.Hash:X16} " +
          $"bfsPos={parentPos.ToPosition.FEN} N={parentNode.N} numExpanded={numExpanded}");
      }
    }

    PosHash96 parentHash96 = MGPositionHashing.Hash96(in parentPos);
    PosHash96MultisetRunning runningAfterParent = runningHashes[parentIdx];
    runningAfterParent.Add(parentHash96);

    int headerBlockIdx = parentNode.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
    Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, (int)parentNode.NumPolicyMoves);

    for (int i = 0; i < numExpanded; i++)
    {
      if (!headers[i].IsExpanded)
      {
        continue;
      }

      int edgeBlock = headers[i].EdgeStoreBlockIndex;
      Span<GEdgeStruct> edgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
      int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
      ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeIndex.IsNull)
      {
        continue;
      }

      int childIdx = edge.ChildNodeIndex.Index;

      // Atomically claim this child (safe for both serial and parallel paths).
      if (Interlocked.CompareExchange(ref Unsafe.AsRef<int>(visited + childIdx), 1, 0) != 0)
      {
        continue;
      }

      // Compute child position incrementally from parent.
      EncodedMove encodedMove = edge.Move;
      MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encodedMove, in parentPos);
      MGPosition childPos = parentPos;
      childPos.MakeMove(mgMove);

      // Verify child BFS-computed position matches stored hash.
      if (VALIDATE_INLINE)
      {
        ref GNodeStruct childNode2 = ref graph.NodesBufferOS[childIdx];
        PosHash64 childBfsHash = MGPositionHashing.Hash64(in childPos);
        if (childBfsHash.Hash != childNode2.HashStandalone.Hash)
        {
          throw new InvalidOperationException(
            $"Phase6 BFS child position mismatch: parent={parentIdx} child={childIdx} edge[{i}] " +
            $"move={encodedMove} parentPos={parentPos.ToPosition.FEN} childPos={childPos.ToPosition.FEN} " +
            $"childBfsHash={childBfsHash.Hash:X16} childStoredHash={childNode2.HashStandalone.Hash:X16} " +
            $"parentBfsHashMatch={(MGPositionHashing.Hash64(in parentPos).Hash == parentNode.HashStandalone.Hash)} " +
            $"parentN={parentNode.N} childN={childNode2.N}");
        }
      }

      positions[childIdx] = childPos;

      // Reset running hash on irreversible moves.
      bool isIrreversible = parentPos.IsIrreversibleMove(mgMove, in childPos);
      PosHash96MultisetRunning childRunning = isIrreversible ? default : runningAfterParent;
      runningHashes[childIdx] = childRunning;

      // Finalize and register in dictionary.
      PosHash96 childHash96 = MGPositionHashing.Hash96(in childPos);
      PosHash96MultisetFinalized childFinalized = childRunning.Finalized(childHash96);
      graph.transpositionPositionAndSequence.TryAdd(childFinalized, childIdx);

      int slot = Interlocked.Increment(ref scratch.NextCount) - 1;
      nextFrontier[slot] = childIdx;
    }
  }


  /// <summary>
  /// Phase 7: Final cleanup - reset counters, resize stores, and validate.
  /// Root flags, cached pointers, and history were already set in Phase 5b.
  /// </summary>
  static void Phase7FinalCleanup(Graph graph)
  {
    // Reset NN statistics counters.
    graph.NNPositionEvaluationsCount = 0;
    graph.NNBatchesCount = 0;
    graph.NNBatchSizeMax = 0;
    graph.NumLinksToExistingNodes = 0;

    // Release committed but unused memory.
    graph.Store.ResizeToCurrent();

    int numRetained = graph.Store.NodesStore.NumUsedNodes;

    // Validate no overlapping edge header blocks between nodes.
    // Sort-based O(N log N) approach: collect ranges, sort by start, check consecutive pairs.
    if (VALIDATE_INLINE)
    {
      List<(int start, int count, int nodeIdx)> headerRanges = new(numRetained);
      for (int i = 1; i <= numRetained; i++)
      {
        ref GNodeStruct ni = ref graph.NodesBufferOS[i];
        if (ni.NumPolicyMoves == 0 || ni.edgeHeaderBlockIndexOrDeferredNode.IsNull
         || ni.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
        {
          continue;
        }
        int start = ni.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
        int count = graph.EdgeHeadersStore.NumBlocksReservedForNumItems((int)ni.NumPolicyMoves);
        headerRanges.Add((start, count, i));
      }

      headerRanges.Sort((a, b) => a.start.CompareTo(b.start));

      for (int i = 1; i < headerRanges.Count; i++)
      {
        (int prevStart, int prevCount, int prevNode) = headerRanges[i - 1];
        (int curStart, int curCount, int curNode) = headerRanges[i];
        if (curStart < prevStart + prevCount)
        {
          throw new Exception(
            $"OVERLAPPING EDGE HEADERS: node {prevNode} blocks [{prevStart}..{prevStart + prevCount}) " +
            $"vs node {curNode} blocks [{curStart}..{curStart + curCount})");
        }
      }
    }

    // Validate all edge ChildNodeIndex values are within bounds.
    // This catches stale edge data from prior graph states that
    // survived rewrite cleanup (e.g., trailing slots not properly zeroed).
    if (VALIDATE_INLINE)
    {
      ValidateEdgeChildIndices(graph, numRetained);
    }

    // In debug mode, validate internal consistency.
#if DEBUG
    graph.Validate(nodesInFlightExpectedZero: true, dumpIfFails: true);
#endif
  }


  /// <summary>
  /// Validates that all expanded edges of retained nodes have ChildNodeIndex
  /// within the valid range [1, NumTotalNodes). Also checks that trailing
  /// edge slots (beyond NumEdgesExpanded) in each block are zeroed.
  /// </summary>
  static void ValidateEdgeChildIndices(Graph graph, int numRetained)
  {
    int numTotalNodes = graph.NodesStore.NumTotalNodes;
    int numTotalRetained = numRetained + 1;
    GEdgeStore edgesStore = graph.EdgesStore;

    for (int nodeIdx = 1; nodeIdx < numTotalRetained; nodeIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[nodeIdx];
      int numExpanded = nodeRef.NumEdgesExpanded;
      if (numExpanded == 0
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, nodeRef.NumPolicyMoves);

      int lastBlock = -1;
      int edgeIndexInBlock = 0;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        if (edgeBlock != lastBlock)
        {
          // Entering a new block — check trailing slots of previous block.
          if (lastBlock >= 0 && edgeIndexInBlock < GEdgeStore.NUM_EDGES_PER_BLOCK)
          {
            Span<GEdgeStruct> prevBlock = edgesStore.SpanAtBlockIndex(lastBlock);
            for (int t = edgeIndexInBlock; t < GEdgeStore.NUM_EDGES_PER_BLOCK; t++)
            {
              if (prevBlock[t].Type == GEdgeStruct.EdgeType.ChildEdge && !prevBlock[t].ChildNodeIndex.IsNull)
              {
                throw new Exception(
                  $"STALE TRAILING EDGE: node {nodeIdx} block {lastBlock} slot {t} " +
                  $"has ChildNodeIndex={prevBlock[t].ChildNodeIndex.Index} but is beyond expanded range");
              }
            }
          }
          lastBlock = edgeBlock;
          edgeIndexInBlock = 0;
        }

        // Validate the expanded edge's ChildNodeIndex.
        Span<GEdgeStruct> block = edgesStore.SpanAtBlockIndex(edgeBlock);
        ref GEdgeStruct edge = ref block[edgeIndexInBlock];
        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int childIdx = edge.ChildNodeIndex.Index;
          if (childIdx < 1 || childIdx >= numTotalNodes)
          {
            throw new Exception(
              $"INVALID EDGE CHILD: node {nodeIdx} edge {i} block {edgeBlock} slot {edgeIndexInBlock} " +
              $"ChildNodeIndex={childIdx} but NumTotalNodes={numTotalNodes}");
          }
        }
        edgeIndexInBlock++;
      }

      // Check trailing slots of the last block.
      if (lastBlock >= 0 && edgeIndexInBlock < GEdgeStore.NUM_EDGES_PER_BLOCK)
      {
        Span<GEdgeStruct> lastBlockSpan = edgesStore.SpanAtBlockIndex(lastBlock);
        for (int t = edgeIndexInBlock; t < GEdgeStore.NUM_EDGES_PER_BLOCK; t++)
        {
          if (lastBlockSpan[t].Type == GEdgeStruct.EdgeType.ChildEdge && !lastBlockSpan[t].ChildNodeIndex.IsNull)
          {
            throw new Exception(
              $"STALE TRAILING EDGE: node {nodeIdx} block {lastBlock} slot {t} " +
              $"has ChildNodeIndex={lastBlockSpan[t].ChildNodeIndex.Index} but is beyond expanded range");
          }
        }
      }
    }
  }


  /// <summary>
  /// Validates Q consistency for all retained nodes after rewrite.
  /// For each non-terminal node with expanded edges, recomputes Q from edges:
  ///   recomputedQ = (-sum(edge.Q * edge.N) + V) / (sum(edge.N) + 1)
  /// and checks that |recomputedQ| &lt;= 2.0.
  /// Also detects edges where NDrawByRepetition > N (invalid after scaling).
  /// </summary>
  static void ValidateEdgeQConsistency(Graph graph, int numRetained)
  {
    int numTotalRetained = numRetained + 1;
    GEdgeStore edgesStore = graph.EdgesStore;

    for (int nodeIdx = 1; nodeIdx < numTotalRetained; nodeIdx++)
    {
      ref GNodeStruct nodeRef = ref graph.NodesBufferOS[nodeIdx];

      if (nodeRef.Terminal.IsTerminal())
      {
        continue;
      }

      int numExpanded = nodeRef.NumEdgesExpanded;
      if (numExpanded == 0
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      // Check stored Q on the node itself.
      if (Math.Abs(nodeRef.Q) > 2.0)
      {
        throw new Exception(
          $"Q INTEGRITY: node {nodeIdx} has stored Q={nodeRef.Q} (|Q|>2). " +
          $"N={nodeRef.N} NumExpanded={numExpanded}");
      }

      int headerBlockIdx = nodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, nodeRef.NumPolicyMoves);

      double sumEdgeW = 0;
      int sumEdgeN = 0;

      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        Span<GEdgeStruct> edgeSpan = edgesStore.SpanAtBlockIndex(edgeBlock);
        int offsetInBlock = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        ref GEdgeStruct edge = ref edgeSpan[offsetInBlock];

        // Check NDrawByRepetition > N (would corrupt edge.Q computation).
        if (edge.NDrawByRepetition > edge.N)
        {
          throw new Exception(
            $"Q INTEGRITY: node {nodeIdx} edge {i} has NDrawByRepetition={edge.NDrawByRepetition} > N={edge.N}. " +
            $"QChild={edge.QChild} edge.Q={edge.Q} Move={edge.Move}");
        }

        sumEdgeW += edge.Q * edge.N;
        sumEdgeN += edge.N;
      }

      // Recompute Q from edges: QPure = (-sumEdgeW + V) / (sumEdgeN + 1).
      double nodeV = (float)nodeRef.WinP - (float)nodeRef.LossP;
      int sumN = sumEdgeN + 1;
      double recomputedQ = (-sumEdgeW + nodeV) / sumN;

      if (Math.Abs(recomputedQ) > 2.0)
      {
        throw new Exception(
          $"Q INTEGRITY: node {nodeIdx} recomputed Q={recomputedQ} (|Q|>2). " +
          $"V={nodeV} sumEdgeW={sumEdgeW} sumEdgeN={sumEdgeN} sumN={sumN} " +
          $"storedQ={nodeRef.Q} N={nodeRef.N} NumExpanded={numExpanded}");
      }
    }
  }
}
