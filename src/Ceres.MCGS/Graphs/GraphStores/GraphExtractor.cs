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
///   retention is low (the trigger threshold is ~25%), extraction is far cheaper and additionally
///   lets us KEEP low-reachability graphs that would otherwise be abandoned (discarding all their
///   neural-network evaluations).
///
/// HOW CORRECTNESS IS SHARED, NOT REIMPLEMENTED
///   The extractor only reproduces the compacted node+edge LAYOUT that the rewriter's Phases 2-4
///   produce (nodes at contiguous indices [1..numReachable], expanded edges packed into fresh edge
///   blocks with child indices remapped). It then calls <see cref="GraphRewriter.FinalizeAfterCopy"/>
///   which runs the EXACT SAME tail phases as the in-place rewrite (parent-store rebuild, N/Q/D
///   recomputation, root/history setup, dictionary and sibling reconstruction, cleanup + DEBUG
///   Validate). All the subtle derived-state logic is therefore identical code, not a second copy.
///
/// PARALLELISM
///   The per-node copy (struct + edge headers + edges) is embarrassingly parallel: every node writes
///   only to its own node slot and its own freshly-allocated, disjoint header/edge blocks; the source
///   graph and the old->new index map are read-only; and block allocation is thread-safe (Interlocked
///   index bump with lock-free fast path). It is parallelized via <see cref="USE_PARALLEL_COPY"/>.
///   (The reused finalize phases are independently parallel where the rewriter already makes them so —
///   e.g. dictionary rebuild — and serial elsewhere; per-phase timings are reported to expose this.)
///
/// SCOPE (v1): always copies the FULL reachable set. The caller decides whether to extract (based on
/// an estimated retention fraction). The <c>maxNodesToExtract</c> argument is reserved for a future
/// partial/truncated mode and is NOT enforced here.
///
/// PRECONDITIONS: the graph must be quiescent (between searches: no in-flight visits, nothing
/// locked), and <paramref name="newRootIndex"/> must be an evaluated node in <paramref name="oldGraph"/>.
/// Works for both PositionEquivalence ("Position") and PositionAndHistory modes (the mode is carried
/// into the new graph and the shared finalize phases branch on it as needed).
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
  /// If true, the per-phase timing breakdown (enumerate / copy / finalize sub-phases) is logged
  /// in yellow after each extraction. Off by default; turn on to profile where extraction time goes.
  /// (The timings are always populated in <see cref="ExtractResult"/> regardless of this flag.)
  /// </summary>
  internal const bool LOG_EXTRACT_PHASE_TIMINGS = false;


  /// <summary>
  /// Statistics describing an extraction operation (and the resulting graph), including a
  /// per-phase timing breakdown (all times in seconds).
  /// </summary>
  public record ExtractResult(Graph NewGraph, int NodesBefore, int NodesAfter,
                              double ElapsedSeconds, float RetentionFraction, bool Succeeded,
                              double EnumerateSeconds, double CopySeconds, double FinalizeSeconds,
                              GraphRewriter.FinalizePhaseTimings Finalize, bool UsedParallelCopy);


  /// <summary>
  /// Attempts to build a new Graph containing the subgraph reachable from <paramref name="newRootIndex"/>.
  /// On success the returned <see cref="ExtractResult.NewGraph"/> is a fully finalized graph whose
  /// graph root (index 1) is the former search root. The CALLER is responsible for disposing
  /// <paramref name="oldGraph"/> after switching over to the new graph.
  /// </summary>
  /// <param name="oldGraph">Source graph (read only; not modified).</param>
  /// <param name="newRootIndex">Index in <paramref name="oldGraph"/> of the node to become the new root.</param>
  /// <param name="priorMoves">Position+history for the new root (used to initialize root state/hashes).</param>
  /// <param name="maxNodesToExtract">RESERVED — not enforced in v1 (full reachable set is always copied).</param>
  public static ExtractResult TryExtract(Graph oldGraph, NodeIndex newRootIndex,
                                         PositionWithHistory priorMoves,
                                         int maxNodesToExtract = int.MaxValue)
  {
    Stopwatch swTotal = Stopwatch.StartNew();
    Stopwatch swPhase = Stopwatch.StartNew();

    int numNodesOld = oldGraph.NodesStore.NumTotalNodes;   // includes null node at index 0
    int numUsedOld = oldGraph.NodesStore.NumUsedNodes;

    Debug.Assert(!newRootIndex.IsNull && newRootIndex.Index > 0 && newRootIndex.Index < numNodesOld);

    // ---- Phase E1: BFS enumerate reachable nodes and assign new contiguous indices ----
    // BFS order yields parent < child for tree edges, which keeps the downstream Phase 5a
    // single-pass Q/D recomputation accurate. Mirrors GraphRewriter.Phase1FindReachableNodes,
    // but also produces the old->new index map and an ordered list of old indices.
    // (Kept serial: it is read-only fan-out whose cost is modest relative to the copy; a parallel
    // level-synchronous BFS is possible later if profiling shows it dominates.)
    int[] oldToNew = new int[numNodesOld];                 // 0 == not reachable
    List<int> newToOld = new(Math.Min(numUsedOld, 1024));  // newToOld[k] -> old index of new index (k+1)

    Queue<int> queue = new();
    oldToNew[newRootIndex.Index] = 1;
    newToOld.Add(newRootIndex.Index);
    queue.Enqueue(newRootIndex.Index);

    while (queue.Count > 0)
    {
      int oldIdx = queue.Dequeue();
      ref GNodeStruct nr = ref oldGraph.NodesBufferOS[oldIdx];

      int numExpanded = nr.NumEdgesExpanded;
      if (numExpanded == 0
       || nr.NumPolicyMoves == 0
       || nr.edgeHeaderBlockIndexOrDeferredNode.IsNull
       || nr.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        continue;
      }

      int headerBlockIdx = nr.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
      Span<GEdgeHeaderStruct> headers = oldGraph.EdgeHeadersStore.SpanAtBlockIndex(headerBlockIdx, nr.NumPolicyMoves);

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
          cachedEdgeSpan = oldGraph.EdgesStore.SpanAtBlockIndex(edgeBlock);
          lastEdgeBlock = edgeBlock;
        }
        ref GEdgeStruct edge = ref cachedEdgeSpan[i % GEdgeStore.NUM_EDGES_PER_BLOCK];

        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge && !edge.ChildNodeIndex.IsNull)
        {
          int childIdx = edge.ChildNodeIndex.Index;
          if (oldToNew[childIdx] == 0)
          {
            oldToNew[childIdx] = newToOld.Count + 1;
            newToOld.Add(childIdx);
            queue.Enqueue(childIdx);
          }
        }
      }
    }

    int numReachable = newToOld.Count;
    double enumerateSeconds = swPhase.Elapsed.TotalSeconds;

    if (numReachable <= 0)
    {
      // Should not happen (root is always reachable); decline defensively.
      swTotal.Stop();
      return new ExtractResult(null, numUsedOld, 0, swTotal.Elapsed.TotalSeconds, 0f, false,
                               enumerateSeconds, 0, 0, null, false);
    }

    // ---- Phase E2/E3/E4: construct twin graph and copy the reachable subgraph ----
    swPhase.Restart();

    Graph newGraph = new(maxNodes: oldGraph.Store.MaxNodes,
                         hasAction: oldGraph.Store.HasAction,
                         hasState: oldGraph.Store.HasState,
                         graphEnabled: oldGraph.Store.GraphEnabled,
                         coalescedMode: oldGraph.Store.UsesPositionEquivalenceMode,
                         tryEnableLargePages: MCGSParamsFixed.TryEnableLargePages,
                         nodesWithOneVisitMayHaveDifferentQ: oldGraph.NodesWithOneVisitMayHaveDifferentQ,
                         priorHistory: priorMoves,
                         testFlag: oldGraph.TestFlag);

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

    swTotal.Stop();

    float retention = numUsedOld > 0 ? (float)numReachable / numUsedOld : 0f;
    return new ExtractResult(newGraph, numUsedOld, numReachable, swTotal.Elapsed.TotalSeconds, retention, true,
                             enumerateSeconds, copySeconds, finalizeSeconds, finalizeTimings, useParallel);
  }


  /// <summary>
  /// Counts the nodes reachable from <paramref name="rootIndex"/> by following expanded child edges —
  /// i.e. exactly the set <see cref="TryExtract"/> would retain (this is the same traversal as Phase E1,
  /// which itself mirrors GraphRewriter.Phase1FindReachableNodes). Stops early once the running count
  /// exceeds <paramref name="abortAbove"/> (returning abortAbove + 1), so cost is bounded to
  /// O(min(reachable, abortAbove)) when the caller only needs a thresholded answer.
  ///
  /// Used by the reuse decision to base extract/abandon on the ACTUAL reachable-node count rather than
  /// the (visit-based) searchRoot.N proxy, which badly under-predicts retention in transposition-heavy
  /// graphs (e.g. PositionEquivalence mode). NOTE: keep this traversal in sync with Phase E1 in
  /// <see cref="TryExtract"/> so the count equals what extraction actually retains.
  /// </summary>
  public static int CountReachableNodes(Graph graph, NodeIndex rootIndex, int abortAbove = int.MaxValue)
  {
    int numNodes = graph.NodesStore.NumTotalNodes;
    BitArray visited = new(numNodes);
    Queue<int> queue = new();

    visited.Set(rootIndex.Index, true);
    queue.Enqueue(rootIndex.Index);
    int count = 1;

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
            if (++count > abortAbove)
            {
              return count;   // exceeded the threshold the caller cares about
            }
            queue.Enqueue(childIdx);
          }
        }
      }
    }

    return count;
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
