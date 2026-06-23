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
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Graphs.GraphStores;

/// <summary>
/// Builds a brand-new <see cref="Graph"/> containing only the subgraph reachable from a new search
/// root, by COPYING the reachable nodes/edges into a fresh store. This is an alternative to the
/// in-place <see cref="GraphRewriter"/> for the case where the reachable subgraph is SMALL relative
/// to the whole graph.
///
/// WHY A SEPARATE PRIMITIVE
///   The in-place rewrite is O(total old nodes): Phase 2 moves/clears every node slot of the old
///   graph regardless of how few survive. Extraction is O(reachable): it visits only the reachable
///   nodes and their edges, then disposes the old graph wholesale (an O(1) native free). When
///   retention is low, extraction is far cheaper and additionally lets us KEEP low-reachability
///   graphs that would otherwise be abandoned (discarding all their neural-network evaluations).
///
/// HOW CORRECTNESS IS SHARED, NOT REIMPLEMENTED
///   The extractor only reproduces the compacted node+edge LAYOUT that the rewriter's Phases 2-4
///   produce (nodes at contiguous indices [1..numReachable], expanded edges packed into fresh edge
///   blocks with child indices remapped). It then calls <see cref="GraphRewriter.FinalizeAfterCopy"/>
///   which runs the EXACT SAME tail phases as the in-place rewrite (parent-store rebuild, N/Q/D
///   recomputation, root/history setup, dictionary and sibling reconstruction, cleanup + DEBUG
///   Validate). All the subtle derived-state logic is therefore identical code, not a second copy.
///
/// TWO-STEP API (single BFS)
///   <see cref="EnumerateReachable"/> performs the one BFS over reachable nodes (with an early-abort
///   cap) and returns a <see cref="ReachableSet"/>. The reuse decision uses its count to decide, and
///   — if it decides to extract — passes the SAME set to <see cref="ExtractFromReachable"/> so the
///   subgraph is enumerated only once (no redundant probe-then-enumerate BFS).
///
/// PARALLELISM
///   The per-node copy (struct + edge headers + edges) is embarrassingly parallel: every node writes
///   only to its own node slot and its own freshly-allocated, disjoint header/edge blocks; the source
///   graph and the old->new index map are read-only; and block allocation is thread-safe (Interlocked
///   index bump with lock-free fast path). It is parallelized via <see cref="USE_PARALLEL_COPY"/>.
///
/// PRECONDITIONS: the graph must be quiescent (between searches: no in-flight visits, nothing
/// locked), and the root must be an evaluated node in the source graph. Works for both
/// PositionEquivalence ("Position") and PositionAndHistory modes.
/// </summary>
public static unsafe class GraphExtractor
{
  /// <summary>
  /// If true, the per-node copy loop runs in parallel (for reachable sets at or above
  /// <see cref="PARALLEL_COPY_MIN_NODES"/>). Set false to force the serial copy path
  /// (useful for A/B timing or debugging).
  /// </summary>
  const bool USE_PARALLEL_COPY = true;

  /// <summary>
  /// Minimum reachable-node count below which the copy is done serially even when
  /// <see cref="USE_PARALLEL_COPY"/> is true (avoids Parallel.For overhead on tiny graphs).
  /// </summary>
  const int PARALLEL_COPY_MIN_NODES = 2048;

  /// <summary>
  /// If true, the per-phase timing breakdown (BFS / copy / finalize sub-phases) is logged
  /// in yellow after each extraction (and the reachability BFS time for reuse/abandon decisions).
  /// Off by default; turn on to profile where the time goes.
  /// </summary>
  internal const bool LOG_EXTRACT_PHASE_TIMINGS = false;


  /// <summary>
  /// Result of <see cref="EnumerateReachable"/>: the nodes reachable from a root, in BFS order.
  /// <see cref="NewToOld"/>[k] is the old node index that becomes new index (k+1); it is null when
  /// the enumeration aborted early (<see cref="Aborted"/> == true), in which case only
  /// <see cref="Count"/> (a lower bound greater than the abort threshold) is meaningful.
  /// </summary>
  public readonly record struct ReachableSet(int Count, List<int> NewToOld, bool Aborted);


  /// <summary>
  /// Statistics describing an extraction operation (and the resulting graph), including a
  /// per-phase timing breakdown (all times in seconds). <see cref="EnumerateSeconds"/> is the
  /// single reachability BFS (performed by the caller's <see cref="EnumerateReachable"/> and passed in).
  /// </summary>
  public record ExtractResult(Graph NewGraph, int NodesBefore, int NodesAfter,
                              double ElapsedSeconds, float RetentionFraction, bool Succeeded,
                              double EnumerateSeconds, double CopySeconds, double FinalizeSeconds,
                              GraphRewriter.FinalizePhaseTimings Finalize, bool UsedParallelCopy);


  /// <summary>
  /// Enumerates (via one BFS, following expanded child edges) the nodes reachable from
  /// <paramref name="rootIndex"/> — exactly the set <see cref="ExtractFromReachable"/> would retain.
  /// Assigns new contiguous indices in BFS order (parent &lt; child for tree edges, which keeps the
  /// downstream Phase 5a single-pass Q/D recomputation accurate). Mirrors
  /// GraphRewriter.Phase1FindReachableNodes.
  ///
  /// Stops early and returns <see cref="ReachableSet.Aborted"/> == true (with NewToOld == null) once
  /// the running count exceeds <paramref name="abortAbove"/>, so cost is bounded to
  /// O(min(reachable, abortAbove)) when the caller only needs a thresholded answer (reuse/abandon).
  /// Uses a cheap BitArray for the visited set; the full old->new index map (a large int[]) is built
  /// only later, in <see cref="ExtractFromReachable"/>, i.e. only when we actually extract.
  /// </summary>
  public static ReachableSet EnumerateReachable(Graph graph, NodeIndex rootIndex, int abortAbove = int.MaxValue)
  {
    int numNodes = graph.NodesStore.NumTotalNodes;
    BitArray visited = new(numNodes);
    List<int> newToOld = new(Math.Min(graph.NodesStore.NumUsedNodes, 1024));
    Queue<int> queue = new();

    visited.Set(rootIndex.Index, true);
    newToOld.Add(rootIndex.Index);
    queue.Enqueue(rootIndex.Index);

    while (queue.Count > 0)
    {
      int oldIdx = queue.Dequeue();
      ref GNodeStruct nr = ref graph.NodesBufferOS[oldIdx];

      int numExpanded = nr.NumEdgesExpanded;
      if (numExpanded == 0
       || nr.NumPolicyMoves == 0
       || nr.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nr.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = nr.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = graph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, nr.NumPolicyMoves);

      int lastEdgeBlock = -1;
      Span<GEdgeStruct> cachedEdgeSpan = default;
      for (int i = 0; i < numExpanded; i++)
      {
        if (!headers[i].IsExpanded)
        {
          continue;
        }

        int edgeBlock = headers[i].EdgeStoreBlockIndex;
        if (edgeBlock != lastEdgeBlock)
        {
          cachedEdgeSpan = graph.EdgesStore.SpanAtBlockIndex(edgeBlock);
          lastEdgeBlock = edgeBlock;
        }
        ref GEdgeStruct edge = ref cachedEdgeSpan[i % GEdgeStore.NUM_EDGES_PER_BLOCK];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int childIdx = edge.ChildNodeIndex.Index;
          if (!visited.Get(childIdx))
          {
            visited.Set(childIdx, true);
            newToOld.Add(childIdx);
            if (newToOld.Count > abortAbove)
            {
              // Beyond the threshold the caller cares about; drop the (now useless) ordering list.
              return new ReachableSet(newToOld.Count, null, true);
            }
            queue.Enqueue(childIdx);
          }
        }
      }
    }

    return new ReachableSet(newToOld.Count, newToOld, false);
  }


  /// <summary>
  /// Convenience wrapper: enumerate the full reachable set and extract it. Used by callers that do not
  /// run their own decision/enumeration. The reuse decision instead calls <see cref="EnumerateReachable"/>
  /// and <see cref="ExtractFromReachable"/> directly so the BFS is shared with the decision.
  /// <paramref name="maxNodesToExtract"/> is RESERVED — not enforced in v1 (full reachable set is copied).
  /// </summary>
  public static ExtractResult TryExtract(Graph oldGraph, NodeIndex newRootIndex,
                                         PositionWithHistory priorMoves,
                                         int maxNodesToExtract = int.MaxValue)
  {
    Stopwatch sw = Stopwatch.StartNew();
    ReachableSet reachable = EnumerateReachable(oldGraph, newRootIndex, int.MaxValue);
    double enumerateSeconds = sw.Elapsed.TotalSeconds;
    return ExtractFromReachable(oldGraph, newRootIndex, priorMoves, reachable, enumerateSeconds);
  }


  /// <summary>
  /// Builds a new Graph containing the already-enumerated reachable subgraph. On success the returned
  /// <see cref="ExtractResult.NewGraph"/> is a fully finalized graph whose graph root (index 1) is the
  /// former search root; the CALLER disposes <paramref name="oldGraph"/> after switching over.
  /// </summary>
  /// <param name="oldGraph">Source graph (read only; not modified).</param>
  /// <param name="newRootIndex">Index in <paramref name="oldGraph"/> of the node that becomes the new root.</param>
  /// <param name="priorMoves">Position+history for the new root (initializes root state/hashes).</param>
  /// <param name="reachable">The reachable set from <see cref="EnumerateReachable"/> (must not be aborted).</param>
  /// <param name="enumerateSeconds">Time already spent enumerating (for reporting in the result).</param>
  public static ExtractResult ExtractFromReachable(Graph oldGraph, NodeIndex newRootIndex,
                                                   PositionWithHistory priorMoves,
                                                   ReachableSet reachable, double enumerateSeconds)
  {
    int numUsedOld = oldGraph.NodesStore.NumUsedNodes;
    int numReachable = reachable.Count;
    List<int> newToOld = reachable.NewToOld;

    if (reachable.Aborted || newToOld == null || numReachable <= 0)
    {
      // Caller should only extract a complete (non-aborted) set; decline defensively.
      return new ExtractResult(null, numUsedOld, 0, enumerateSeconds, 0f, false,
                               enumerateSeconds, 0, 0, null, false);
    }

    int numNodesOld = oldGraph.NodesStore.NumTotalNodes;   // includes null node at index 0

    Stopwatch swPhase = Stopwatch.StartNew();

    // ---- Phase E2/E3/E4: construct twin graph and copy the reachable subgraph ----
    // Build the old->new index map from the BFS order. This is the only large (int[numNodesOld])
    // allocation, and it happens only here on the extract path (the reuse/abandon decision used the
    // cheap BitArray inside EnumerateReachable and never reaches this point).
    int[] oldToNew = new int[numNodesOld];                 // 0 == not reachable
    for (int k = 0; k < numReachable; k++)
    {
      oldToNew[newToOld[k]] = k + 1;
    }

    // We carry the prior graph's transposition dictionaries over to the new graph below (and Phase 6
    // of FinalizeAfterCopy refills them in place), so the new graph's own constructor-built dictionaries
    // are immediately discarded. Pass a minimal dictionarySizeHint (clamped up to the 16K floor in
    // Graph.Initialize -> a ~128-bucket throwaway) rather than numReachable: the only use of the
    // constructor dictionary is the root-node registration done during construction, which is never
    // read before Phase 6 rebuilds the dictionaries from scratch.
    Graph newGraph = new(maxNodes: oldGraph.Store.MaxNodes,
                         hasAction: oldGraph.Store.HasAction,
                         hasState: oldGraph.Store.HasState,
                         graphEnabled: oldGraph.Store.GraphEnabled,
                         coalescedMode: oldGraph.Store.UsesPositionEquivalenceMode,
                         tryEnableLargePages: MCGSParamsFixed.TryEnableLargePages,
                         nodesWithOneVisitMayHaveDifferentQ: oldGraph.NodesWithOneVisitMayHaveDifferentQ,
                         priorHistory: priorMoves,
                         testFlag: oldGraph.TestFlag,
                         maintainSiblingSets: oldGraph.MaintainSiblingSets,
                         dictionarySizeHint: 1);

    // Carry over the prior graph's transposition dictionary ALLOCATION. Extraction remaps every node
    // index, so the entry VALUES (node indices / GNodeIndexSetIndex) cannot be reused -- but the
    // allocation can: hand the prior (larger, already-grown) dictionaries to the new graph so Phase 6
    // refills them in place with zero bucket splits and future-growth headroom, instead of allocating
    // fresh. Safe to assign the stale-content dictionaries now: no rewrite phase before Phase 6 (nor
    // the copy loop below) reads them, and Phase 6 clears each one before refilling. The old graph is
    // disposed by the caller after we return; Dispose only nulls these (now-null) fields and frees the
    // native store, so the managed dictionaries live on, owned by the new graph.
    newGraph.transpositionsPosStandalone = oldGraph.transpositionsPosStandalone;
    newGraph.transpositionPositionAndSequence = oldGraph.transpositionPositionAndSequence;
    oldGraph.transpositionsPosStandalone = null;
    oldGraph.transpositionPositionAndSequence = null;

    // Commit node-store pages for indices [0..numReachable] before any direct write. The copy loop
    // writes nodes directly (not via AllocateNext) so the node buffer never grows during the loop,
    // keeping ref-to-node-slot stable and the loop free of node-store synchronization.
    newGraph.NodesStore.MemoryBufferOSStore.InsureAllocated(numReachable + 1);

    bool hasState = oldGraph.Store.HasState && oldGraph.Store.AllStateVectors != null;

    bool useParallel = USE_PARALLEL_COPY && numReachable >= PARALLEL_COPY_MIN_NODES;
    if (useParallel)
    {
      // Each node writes disjoint memory (its own node slot + freshly-allocated header/edge blocks);
      // block allocation is thread-safe; oldToNew and the source graph are read-only.
      Parallel.For(0, numReachable, k => CopyNode(oldGraph, newGraph, newToOld[k], k + 1, oldToNew, hasState));
    }
    else
    {
      for (int k = 0; k < numReachable; k++)
      {
        CopyNode(oldGraph, newGraph, newToOld[k], k + 1, oldToNew, hasState);
      }
    }

    // Establish the used-node count for the new graph (null node 0 + nodes [1..numReachable]).
    newGraph.NodesStore.nextFreeIndex = numReachable + 1;
    if (hasState)
    {
      newGraph.NodesStore.AllStates = newGraph.Store.AllStateVectors;
    }

    double copySeconds = swPhase.Elapsed.TotalSeconds;

    // ---- Phase E5: rebuild all derived state via the shared rewriter tail phases ----
    swPhase.Restart();
    GraphRewriter.FinalizePhaseTimings finalizeTimings = GraphRewriter.FinalizeAfterCopy(newGraph, numReachable, priorMoves);
    double finalizeSeconds = swPhase.Elapsed.TotalSeconds;

    // Carry over the cross-graph evaluation-reuse provider (opponent graph reuse). The MCGSSearch
    // re-attach site only runs on the from-scratch branch, so we must propagate it here.
    newGraph.ReuseGraphProvider = oldGraph.ReuseGraphProvider;

    float retention = numUsedOld > 0 ? (float)numReachable / numUsedOld : 0f;
    double elapsedSeconds = enumerateSeconds + copySeconds + finalizeSeconds;
    return new ExtractResult(newGraph, numUsedOld, numReachable, elapsedSeconds, retention, true,
                             enumerateSeconds, copySeconds, finalizeSeconds, finalizeTimings, useParallel);
  }


  /// <summary>
  /// Copies a single reachable node (its struct, optional state vector, and policy/edges) from the
  /// old graph into the new graph at the given new index. Thread-safe for concurrent invocation on
  /// DISTINCT (oldIdx, newIdx) pairs: it writes only to <paramref name="newIdx"/>'s node slot and to
  /// header/edge blocks it allocates itself (allocation is thread-safe), and reads only read-only
  /// state (the source graph and <paramref name="oldToNew"/>).
  /// </summary>
  static void CopyNode(Graph oldGraph, Graph newGraph, int oldIdx, int newIdx, int[] oldToNew, bool hasState)
  {
    ref GNodeStruct srcRef = ref oldGraph.NodesBufferOS[oldIdx];

    // Whole-struct copy (carries N/Q/D/WinP/LossP/Hash/Terminal/pieces/uncertainty/MRaw/miscFields).
    newGraph.NodesBufferOS[newIdx] = srcRef;
    ref GNodeStruct dstRef = ref newGraph.NodesBufferOS[newIdx];

    // Parent linkage is rebuilt from scratch in Phase 5; root/search flags are reset in Phase 5b.
    dstRef.ParentsHeader = default;

    // Carry over the per-node state vector reference, if any.
    if (hasState && oldIdx < oldGraph.Store.AllStateVectors.Length)
    {
      newGraph.Store.AllStateVectors[newIdx] = oldGraph.Store.AllStateVectors[oldIdx];
    }

    int nPol = srcRef.NumPolicyMoves;

    if (srcRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
    {
      // Deferred-policy leaf: its policy lives on another (materialized) node in the OLD graph.
      // Follow the deferral chain to the source and materialize the policy into the new node.
      GNode source = oldGraph[srcRef.edgeHeaderBlockIndexOrDeferredNode.NodeIndex];
      while (source.NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        source = oldGraph[source.NodeRef.edgeHeaderBlockIndexOrDeferredNode.NodeIndex];
      }

      dstRef.edgeHeaderBlockIndexOrDeferredNode.Clear();
      dstRef.NumPolicyMoves = 0;               // AllocateAndCopyPolicyValues requires these preconditions
      dstRef.NumEdgesExpanded = 0;             // deferred nodes are always leaves

      GNode dstNode = newGraph[newIdx];
      using (new NodeLockBlock(dstNode))       // AllocatedEdgeHeaders asserts the target node is locked
      {
        Graph.AllocateAndCopyPolicyValues(source, dstNode);
      }
    }
    else if (nPol == 0)
    {
      // No policy (e.g. terminal or not-yet-evaluated leaf): no edge headers.
      dstRef.edgeHeaderBlockIndexOrDeferredNode.Clear();
      dstRef.NumEdgesExpanded = 0;
    }
    else
    {
      // Materialized node: copy edge headers, and copy each expanded edge into fresh edge blocks.
      int numExp = srcRef.NumEdgesExpanded;
      int srcHdrBlock = srcRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> srcHeaders = oldGraph.EdgeHeadersStore.SpanAtBlockIndex(srcHdrBlock, (byte)nPol);

      int dstHdrBlock = (int)newGraph.EdgeHeadersStore.AllocateEntriesStartBlock(nPol);
      dstRef.edgeHeaderBlockIndexOrDeferredNode = new EdgeHeaderBlockIndexOrNodeIndex(dstHdrBlock);
      Span<GEdgeHeaderStruct> dstHeaders = newGraph.EdgeHeadersStore.SpanAtBlockIndex(dstHdrBlock, (byte)nPol);

      // Copy all headers verbatim. For unexpanded headers this carries (move, P) in the existing
      // non-ascending-P order; for expanded headers it carries the OLD edge-block index which is
      // corrected by ForceSetEdgeBlockIndex below.
      srcHeaders.CopyTo(dstHeaders);

      // Copy expanded edges. Edge at header index i lives at offset (i % 4) of the block for
      // group (i / 4); all four headers of a group share one edge block (matches InitializeNewEdge
      // and the layout the validator enforces). In a valid quiescent graph headers [0..numExp) are
      // a contiguous expanded prefix, so a new block is allocated exactly at each group start.
      int newEdgeBlock = 0;
      for (int i = 0; i < numExp; i++)
      {
        Debug.Assert(srcHeaders[i].IsExpanded, "Expanded prefix invariant violated in source graph");

        int offset = i % GEdgeStore.NUM_EDGES_PER_BLOCK;
        if (offset == 0)
        {
          newEdgeBlock = (int)newGraph.EdgesStore.AllocatedNewBlock();   // fresh block: trailing slots are zero
        }

        int srcEdgeBlock = srcHeaders[i].EdgeStoreBlockIndex;
        GEdgeStruct edgeCopy = oldGraph.EdgesStore.SpanAtBlockIndex(srcEdgeBlock)[offset];

        // No visits may be in flight in a quiescent graph; clear defensively.
        edgeCopy.NumInFlight0 = 0;
        edgeCopy.NumInFlight1 = 0;

        // Remap the child index. ChildEdge children are always reachable (BFS), so map > 0.
        // Terminal edges may carry a stale/unreachable child pointer (never dereferenced); null it.
        if (!edgeCopy.ChildNodeIndex.IsNull)
        {
          int oc = edgeCopy.ChildNodeIndex.Index;
          int nc = (oc > 0 && oc < oldToNew.Length) ? oldToNew[oc] : 0;
          edgeCopy.ChildNodeIndex = nc > 0 ? new NodeIndex(nc) : default;
        }

        newGraph.EdgesStore.SpanAtBlockIndex(newEdgeBlock)[offset] = edgeCopy;
        dstHeaders[i].ForceSetEdgeBlockIndex(newEdgeBlock);
      }
    }
  }
}
