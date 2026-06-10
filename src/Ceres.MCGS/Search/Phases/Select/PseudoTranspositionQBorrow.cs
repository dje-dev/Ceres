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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Phases;

/// <summary>
/// PTQB (pseudo-transposition Q-borrowing): when a new node is created whose position
/// matches an existing node reached via a different history (a "pseudo-twin"), borrow the
/// twin's search-refined Q as part of the new node's initial value instead of relying only
/// on the twin's raw NN value V.
///
/// The borrow is applied by initializing the new node's SiblingsQ/SiblingsQFrac fields
/// (the same mechanism used by pseudo-transposition blending), so:
///   - the first backup (which calls ResetNodeQUsingNewQPure with the copied V) produces
///     stored Q = lambda * twinQ + (1 - lambda) * V with exact-inversion bookkeeping,
///     and backs that blended Q up the path with no backup-code changes;
///   - subsequent PTB select-phase refreshes replace the creation-time weight with the
///     standard excess-N blend, providing natural decay.
///
/// History-safety verification (the GHI conditions):
///   - "the twin's subgraph contains a repetition/50-move result valid only for ITS history"
///     is checked in O(1) via the transitively maintained HistorySensitiveSubgraph flag;
///   - "a twin subgraph position is a repetition under OUR history" is checked by a bounded
///     fat-edges-first walk of the twin's subgraph against the set of positions in our own
///     reversible run (path since last irreversible move, plus the reversible run of the
///     spine/prehistory); the borrow weight is discounted by the visit mass not covered
///     by the walk, and aborted entirely on any overlap.
/// </summary>
internal static class PseudoTranspositionQBorrow
{
  /// <summary>
  /// Diagnostic counters, kept in per-thread cells with plain (non-atomic) increments on
  /// thread-private cache lines and aggregated only when the stats string is built.
  /// Shared Interlocked counters here would cost a cross-socket locked operation per
  /// new-node expansion at high transposition rates.
  /// </summary>
  private sealed class StatsCell
  {
    public long NumBorrowsApplied;
    public long NumBorrowsAbortedOverlap;
    public long NumBorrowsSkippedNoDonor;
    public long NumFastPathFullClean;
    public long NumWalks;
    public long NumWalkNodesExamined;
    public long NumDonorBusy;
    public double SumLambdaApplied;
  }

  [ThreadStatic]
  static StatsCell statsCellForThread;

  static readonly ConcurrentQueue<StatsCell> allStatsCells = new();

  static StatsCell StatsCellThisThread
  {
    get
    {
      StatsCell cell = statsCellForThread;
      if (cell == null)
      {
        cell = new StatsCell();
        statsCellForThread = cell;
        allStatsCells.Enqueue(cell);
      }
      return cell;
    }
  }


  internal static string StatsString
  {
    get
    {
      // Aggregate over per-thread cells (reads of aligned long/double are atomic on x64;
      // tearing-free but possibly slightly stale, acceptable for diagnostics).
      long applied = 0, aborted = 0, noDonor = 0, fastPath = 0, walks = 0, walkNodes = 0, donorBusy = 0;
      double sumLambda = 0;
      foreach (StatsCell cell in allStatsCells)
      {
        applied += cell.NumBorrowsApplied;
        aborted += cell.NumBorrowsAbortedOverlap;
        noDonor += cell.NumBorrowsSkippedNoDonor;
        fastPath += cell.NumFastPathFullClean;
        walks += cell.NumWalks;
        walkNodes += cell.NumWalkNodesExamined;
        donorBusy += cell.NumDonorBusy;
        sumLambda += cell.SumLambdaApplied;
      }

      return $"PTQB applied={applied:N0} (avgLambda="
           + $"{(applied == 0 ? 0 : sumLambda / applied):F3}) "
           + $"blockedContamination={aborted:N0} noDonor={noDonor:N0} "
           + $"fastPathFullClean={fastPath:N0} donorBusy={donorBusy:N0} walks={walks:N0} "
           + $"avgWalkNodes={(walks == 0 ? 0 : (double)walkNodes / walks):F1}";
    }
  }


  /// <summary>
  /// Interval between periodic console dumps of StatsString (while the feature is active).
  /// </summary>
  const long STATS_DUMP_INTERVAL_MS = 3 * 60 * 1000;

  // Initialized at class load so the first dump arrives one full interval into use.
  static long lastStatsDumpTimeMS = System.Environment.TickCount64;

  /// <summary>
  /// Dumps StatsString to the console (in yellow) if at least STATS_DUMP_INTERVAL_MS has
  /// elapsed since the last dump. Called from TryApply, so dumps occur only while the
  /// feature is enabled and actually being exercised. The CompareExchange guarantees a
  /// single dump per interval under concurrent callers.
  /// (The hot-path read is a plain Volatile.Read: an Interlocked.Read would be a bus-locked
  /// RMW taking the cache line exclusive on every call from every thread.)
  /// </summary>
  static void PossiblyDumpStats()
  {
    long nowMS = System.Environment.TickCount64;
    long last = Volatile.Read(ref lastStatsDumpTimeMS);
    if (nowMS - last >= STATS_DUMP_INTERVAL_MS
     && Interlocked.CompareExchange(ref lastStatsDumpTimeMS, nowMS, last) == last)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, StatsString);
    }
  }


  /// <summary>
  /// Attempts to apply a verified Q-borrow to a newly created node.
  /// Caller must have verified ParamsSearch.EnablePseudoTranspositionQBorrow
  /// (whose validation implies PTB enabled, graph mode, and history mode)
  /// and that the new position is not a duplicate within our history.
  /// </summary>
  /// <param name="engine">The engine (for graph and history context).</param>
  /// <param name="path">The current path; its leaf visit is the newly created node.</param>
  /// <param name="newNode">The newly created (N=0, value-copied) node.</param>
  /// <param name="childPos">The new node's position.</param>
  /// <param name="childHash64">Standalone hash of the new node's position.</param>
  /// <param name="moveToNewNodeIrreversible">If the move into the new node was irreversible
  /// (enables the full-clean fast path which skips the verification walk).</param>
  internal static void TryApply(MCGSEngine engine, MCGSPath path, GNode newNode,
                                in MGPosition childPos, PosHash64 childHash64,
                                bool moveToNewNodeIrreversible)
  {
    ParamsSearch paramsSearch = engine.Manager.ParamsSearch;
    Debug.Assert(paramsSearch.EnablePseudoTranspositionQBorrow);

    if (childPos.Rule50Count > paramsSearch.PseudoTranspositionQBorrowMaxRule50)
    {
      // Near the 50-move horizon the twin's statistics (possibly accumulated under a much
      // lower rule50 count) become unreliable for our context.
      return;
    }

    PossiblyDumpStats();

    // Locate the donor: the maximum-N eligible member of the pseudo-transposition set
    // for this position (which may differ from the arbitrary first member used as the
    // source of the value/policy copy). Eligibility already excludes nodes with N=0
    // (including the just-registered newNode itself), known draw children, repetition
    // context and high Move50 category.
    Graph graph = engine.Graph;
    PosHash64WithMove50AndReps standaloneKey = MGPositionHashing.Hash64WithMove50AndRepsAdded(
      childHash64, childPos.RepetitionCount, childPos.Move50Category);
    GNode donor = NodeIndexSet.MaxNSiblingNode(graph, standaloneKey);

    if (donor.IsNull
     || !donor.IsEvaluated
     || donor.N < paramsSearch.PseudoTranspositionQBorrowMinTwinN)
    {
      // NOTE: a TAINTED donor (HistorySensitiveSubgraph) is deliberately NOT excluded here:
      // the verification walk descends into the taint and localizes its mass, so a donor
      // with (say) a 3-visit repetition terminal inside a 20,000-visit subgraph remains
      // usable with a proportional discount rather than being rejected outright.
      // (When PseudoTranspositionBlendingRequiresCleanSubgraph is enabled, eligibility in
      //  MaxNSiblingNode pre-excludes tainted donors before we ever see them - that flag
      //  trades this refinement for an O(1) screen.)
      StatsCellThisThread.NumBorrowsSkippedNoDonor++;
      return;
    }

    long cleanN;
    long contaminatedN;

    if (moveToNewNodeIrreversible && !donor.HistorySensitiveSubgraph)
    {
      // FAST PATH: the donor's full mass is provably clean, no walk needed.
      //   - The move into the new node was irreversible, so OUR reversible run is empty:
      //     no donor-subgraph position can be a repetition under our history.
      //   - The donor is untainted, and taint floods to ALL ancestors of any
      //     repetition/rule50 terminal, so no reachable descendant is tainted either:
      //     the walk's only other contamination sources (tainted-parent drawn terminals,
      //     unlocalizable taint on lock failure) cannot fire.
      // The walk could therefore only ever UNDER-measure cleanN here (budget/lock
      // artifacts leaving unverified tail mass); cleanN = donor.N is the exact answer.
      // PTBVerifiedCleanFrac is deliberately NOT written: the encoded "never verified"
      // default already decodes to 1.0 (the consumer no-ops), and skipping the write
      // preserves the raw byte's diagnostic distinction.
      // (A stale read of a concurrently-flooding taint flag is tolerated, exactly as the
      // walk tolerates racy edge.N reads; the missed mass is bounded by the few visits
      // of a just-created repetition terminal, and the PTB refresh decays the borrow.)
      cleanN = donor.N;
      contaminatedN = 0;
      StatsCellThisThread.NumFastPathFullClean++;
    }
    else
    {
      // Build the set of standalone hashes of OUR reversible run (positions which a future
      // line could repeat). Empty when the move into the new node was irreversible.
      // The new node's own (junction) position is deliberately excluded: it occurs once in
      // both contexts, and any RE-occurrence inside the donor's subgraph is handled by the
      // walk (for the donor it was a repetition of its own root position, hence tainted).
      HashSet<PosHash64> ourReversibleRunHashes = BuildOurReversibleRunHashes(engine, path, graph);

      // Verification walk over the donor's subgraph (fat edges first, memoized, budget-bounded),
      // classifying visit mass into clean / contaminated / unverified buckets.
      if (!TryVerifyAndMeasure(graph, donor, ourReversibleRunHashes,
                               paramsSearch.PseudoTranspositionQBorrowMaxVerifyNodes,
                               out cleanN, out contaminatedN))
      {
        // Donor busy (locked by another thread); no verification information obtained.
        StatsCellThisThread.NumDonorBusy++;
        return;
      }

      // Contamination RATE is judged over the verified mass only (clean + contaminated):
      // the fat-first walk verifies the dominant mass, and unverified tail mass should not
      // be presumed dirty. Incidental contamination perturbs the donor Q linearly in its
      // mass fraction; rates above the threshold instead signal structural repetition
      // dynamics where the donor's whole estimate (including its policy) diverged from our
      // context, so the node is blocked entirely and permanently.
      long verifiedN = cleanN + contaminatedN;
      double contaminationRate = verifiedN <= 0 ? 0 : (double)contaminatedN / verifiedN;
      bool blocked = contaminationRate > paramsSearch.PseudoTranspositionQBorrowMaxContaminationFraction;

      // Persist the verdict for the node's lifetime (read by the PTB sibling refresh as a
      // multiplicative cap on future blending) - but only when backed by sufficient verified
      // mass, since with only a handful of verified visits the contamination rate is
      // dominated by small-sample noise and a permanent verdict is not warranted.
      // The verdict is path-invariant in history mode: every path into this node shares the
      // same multiset since the last irreversible move, so it holds for all future visits.
      // (Single-byte write; atomic without a lock.)
      if (verifiedN >= paramsSearch.PseudoTranspositionQBorrowMinVerifiedNForCap)
      {
        newNode.NodeRef.SetPTBVerifiedCleanFrac(blocked ? 0 : (1.0 - contaminationRate));
      }

      if (blocked)
      {
        StatsCellThisThread.NumBorrowsAbortedOverlap++;
        return;
      }
    }

    // Immediate borrow weight:
    //   - judged against the donor's FULL mass (unverified tail counts against the
    //     snapshot borrow, unlike the persistent rate-based cap above), and
    //   - shrunk by the donor's statistical confidence donorN / (donorN + k), so that a
    //     near-leaf donor (whose Q carries little information beyond V, possibly itself
    //     dominated by an earlier borrow) receives proportionally little weight and
    //     high-weight borrow cascades through chains of small nodes cannot form.
    double donorConfidence = donor.N / (donor.N + (double)paramsSearch.PseudoTranspositionQBorrowShrinkageK);
    double lambda = paramsSearch.PseudoTranspositionQBorrowMaxWeight
                  * ((double)cleanN / donor.N)
                  * donorConfidence;
    if (lambda < (1.0 / 255.0))
    {
      // Would quantize to zero weight.
      return;
    }

    // Install the borrow via the sibling-blend fields. Do NOT write Q here: the node has
    // N=0 and its first backup (ResetNodeQUsingNewQPure with the copied V) reads these
    // stored fields and produces the blended Q with correct exact-inversion bookkeeping.
    // Non-blocking lock for uniformity with the walk (the caller holds the expansion
    // parent's lock); on contention simply skip the borrow.
    if (!newNode.TryAcquireLock())
    {
      return;
    }
    try
    {
      newNode.NodeRef.SiblingsQ = donor.Q;
      newNode.NodeRef.SiblingsQFrac = lambda;
    }
    finally
    {
      newNode.ReleaseLock();
    }

    StatsCell statsCell = StatsCellThisThread;
    statsCell.NumBorrowsApplied++;
    statsCell.SumLambdaApplied += lambda;
  }


  /// <summary>
  /// Collects the standalone hashes of all positions in our reversible run:
  /// path positions strictly above the new leaf back to (and including) the first position
  /// after the most recent irreversible move, continuing through the search root,
  /// the graph-root-to-search-root spine, and game prehistory with the same cutoff.
  /// Mirrors the enumeration structure of MCGSPath.HashFoundInHistoryOrPrehistory.
  /// </summary>
  // Reusable per-thread collections for TryApply (cleared at the start of each use, so an
  // exception cannot leave stale state visible to the next call). Capacities stay small and
  // bounded: the run-hash set by the reversible run length, the walk structures by
  // PseudoTranspositionQBorrowMaxVerifyNodes. Pooling these eliminates 3-4 heap allocations
  // per newly created node (performed while holding the expansion parent's lock).
  [ThreadStatic] static HashSet<PosHash64> cachedRunHashes;
  [ThreadStatic] static PriorityQueue<(int ChildIndex, int EdgeN), int> cachedFrontier;
  [ThreadStatic] static HashSet<int> cachedVisited;
  [ThreadStatic] static HashSet<int> cachedContaminated;


  private static HashSet<PosHash64> BuildOurReversibleRunHashes(MCGSEngine engine, MCGSPath path, Graph graph)
  {
    HashSet<PosHash64> hashes = cachedRunHashes ??= new();
    hashes.Clear();

    // Path visits, leaf to root. The first (leaf) visit is the new node itself: skip its
    // hash but honor its irreversibility flag (if the move INTO the new node was
    // irreversible, no earlier position can ever repeat below it).
    bool isLeafVisit = true;
    foreach (MCGSPathVisitMember visitPair in path.PathVisitsLeafToRoot)
    {
      ref readonly MCGSPathVisit visitRef = ref visitPair.PathVisitRef;

      if (!isLeafVisit)
      {
        hashes.Add(visitRef.ChildNodeHashStandalone64);
      }
      isLeafVisit = false;

      if (visitRef.MoveIrreverisible)
      {
        return hashes;
      }
    }

    // Search root itself (the path visits cover children only).
    hashes.Add(engine.SearchRootNode.HashStandalone);

    // Spine from search root down toward graph root, stopping at an irreversible move.
    // A position reached BY an irreversible move is itself still repeatable (it lies on
    // the new side of the boundary), so include its hash before cutting off — matching
    // MCGSPath.HashFoundInGraphRootPathOrPrehistory.
    GraphRootToSearchRootNodeInfo[] spine = engine.SearchRootPathFromGraphRoot;
    if (spine != null)
    {
      for (int i = spine.Length - 1; i >= 0; i--)
      {
        hashes.Add(spine[i].ChildHashStandalone64);
        if (spine[i].MoveToChildIrreversible)
        {
          return hashes;
        }
      }
    }

    // Prehistory, newest backward, stopping at the most recent irreversible move.
    var historyHashes = graph.Store.HistoryHashes;
    var prehistory = historyHashes.PriorPositionsHashes64;
    bool[] prehistoryMoveIrreversible = historyHashes.MoveAfterPositionWasIrreversible;
    for (int i = prehistory.Length - 1; i >= 0; i--)
    {
      hashes.Add(prehistory[i]);
      if (i == 0 || prehistoryMoveIrreversible[i - 1])
      {
        break;
      }
    }

    return hashes;
  }


  /// <summary>
  /// Walks the donor's subgraph fat-edges-first (bounded by maxVerifyNodes, memoized over
  /// the DAG), classifying the donor's visit mass into three buckets:
  ///   CLEAN        - walked mass through positions that neither repeat our history nor
  ///                  belong to localized history-sensitive results;
  ///   CONTAMINATED - in-edge mass of positions repeating OUR history (lines that for us
  ///                  end as draws), repetition/50-move terminal draw mass localized under
  ///                  tainted nodes, and the mass of taint that could not be localized;
  ///   UNVERIFIED   - the remaining frontier when the budget is exhausted (also nodes
  ///                  whose lock could not be obtained).
  /// cleanN = donor.N - unverifiedN - contaminatedN (frontier subtraction; conservative
  /// under DAG sharing, where duplicate in-edges may leave residual frontier mass).
  ///
  /// Tainted nodes are DESCENDED INTO rather than rejected: the taint flag floods all
  /// ancestors of a repetition/rule50 terminal, so following it downward localizes the
  /// contamination to the actual drawn-terminal mass, to whatever resolution the budget
  /// allows. (Drawn terminal edges under an UNtainted parent are guaranteed history-free -
  /// stalemate/material/tablebase - and count as clean.)
  ///
  /// CONCURRENCY: unlike the existing whole-graph walks (GraphRewriter, DRP revaluation)
  /// which run on a quiescent graph, this walk runs during the active select phase while
  /// other threads may be expanding edges of the very nodes being examined. A node's edge
  /// headers/blocks are only stable under that node's lock, so each node's edges are
  /// enumerated only after a NON-BLOCKING TryAcquireLock (blocking here could deadlock,
  /// since the caller holds the expansion parent's lock and lock ordering with other
  /// select threads is unconstrained). On TryAcquire failure the node's subtree is treated
  /// as unverified (or as contaminated, if the node is tainted and thus unlocalizable).
  ///
  /// Returns false (no information) only if the donor itself could not be examined.
  /// </summary>
  private static bool TryVerifyAndMeasure(Graph graph, GNode donor,
                                          HashSet<PosHash64> ourReversibleRunHashes,
                                          int maxVerifyNodes,
                                          out long cleanN, out long contaminatedN)
  {
    long donorN = donor.N;
    Debug.Assert(donorN > 0);

    cleanN = 0;
    contaminatedN = 0;

    // Max-heap by edge.N (PriorityQueue is a min-heap, so negate).
    PriorityQueue<(int ChildIndex, int EdgeN), int> frontier = cachedFrontier ??= new();
    frontier.Clear();
    HashSet<int> visited = cachedVisited ??= new();
    visited.Clear();
    visited.Add(donor.Index.Index);
    HashSet<int> contaminatedNodes = cachedContaminated ??= new();
    contaminatedNodes.Clear();
    long frontierN = 0;
    long localContaminatedN = 0;

    if (!TryEnqueueChildEdgesLocked(donor, donor.HistorySensitiveSubgraph,
                                    frontier, ref frontierN, ref localContaminatedN))
    {
      // Donor busy (locked by another thread); no verification information.
      return false;
    }

    int budget = maxVerifyNodes;
    int numNodesExamined = 0;
    while (frontier.Count > 0 && budget > 0)
    {
      (int childIndex, int edgeN) = frontier.Dequeue();
      frontierN -= edgeN;

      if (!visited.Add(childIndex))
      {
        // Duplicate in-edge (DAG): the node was already classified. Mass through an edge
        // into a contaminated node is itself contaminated; otherwise it was absorbed into
        // the walked (clean-by-subtraction) region.
        if (contaminatedNodes.Contains(childIndex))
        {
          localContaminatedN += edgeN;
        }
        continue;
      }
      budget--;
      numNodesExamined++;

      GNode node = graph[childIndex];

      if (ourReversibleRunHashes.Count > 0 && ourReversibleRunHashes.Contains(node.HashStandalone))
      {
        // This position would be a repetition under OUR history (but was not under the
        // donor's): lines through it are live play for the donor but end as draws for us.
        // Classify the whole in-edge mass as contaminated; nothing below is reachable
        // for us, so do not descend.
        localContaminatedN += edgeN;
        contaminatedNodes.Add(childIndex);
        continue;
      }

      bool nodeTainted = node.HistorySensitiveSubgraph;
      if (!TryEnqueueChildEdgesLocked(node, nodeTainted, frontier, ref frontierN, ref localContaminatedN))
      {
        if (nodeTainted)
        {
          // Contamination exists somewhere below but cannot be localized now: classify
          // the whole subtree mass as contaminated (conservative).
          localContaminatedN += edgeN;
          contaminatedNodes.Add(childIndex);
        }
        else
        {
          // Clean node whose subtree cannot be examined now: unverified.
          frontierN += edgeN;
        }
      }
    }

    contaminatedN = localContaminatedN;
    long unverifiedN = frontierN;
    cleanN = donorN - unverifiedN - contaminatedN;
    if (cleanN < 0)
    {
      // Possible under racy reads of moving edge counts; clamp.
      cleanN = 0;
    }

    StatsCell statsCell = StatsCellThisThread;
    statsCell.NumWalks++;
    statsCell.NumWalkNodesExamined += numNodesExamined;
    return true;
  }


  /// <summary>
  /// Enqueues the expanded child edges of a node onto the walk frontier,
  /// enumerating them only under the node's lock (non-blocking attempt).
  /// Returns false if the lock could not be acquired (edges not examined).
  /// Nodes with no materialized children (unexpanded, or with a pending deferred
  /// policy copy whose header field holds a node index rather than an edge block)
  /// are leaf-like: nothing to enqueue, their mass counts as covered.
  ///
  /// When the parent is TAINTED, the mass of its drawn terminal edges is added to
  /// contaminatedN: the taint indicates a repetition/rule50 terminal beneath this node,
  /// and drawn terminals are indistinguishable by kind on the edge itself, so all drawn
  /// terminal mass under a tainted parent is conservatively treated as history-sensitive.
  /// (Under an untainted parent, drawn terminals are guaranteed history-free and clean.)
  /// </summary>
  private static bool TryEnqueueChildEdgesLocked(GNode node, bool nodeTainted,
                                                 PriorityQueue<(int ChildIndex, int EdgeN), int> frontier,
                                                 ref long frontierN, ref long contaminatedN)
  {
    if (!node.TryAcquireLock())
    {
      return false;
    }

    try
    {
      if (node.NumEdgesExpanded == 0 || node.IsPendingPolicyCopy)
      {
        return true;
      }

      foreach (GEdge edge in node.ChildEdgesExpanded)
      {
        if (edge.Type != GEdgeStruct.EdgeType.ChildEdge)
        {
          if (nodeTainted && edge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn)
          {
            contaminatedN += edge.N;
          }
          continue;
        }

        GNode child = edge.ChildNode;
        int edgeN = edge.N;
        if (child.IsNull || edgeN <= 0)
        {
          // Skip edges with no settled visits.
          continue;
        }

        frontier.Enqueue((child.Index.Index, edgeN), -edgeN);
        frontierN += edgeN;
      }

      return true;
    }
    finally
    {
      node.ReleaseLock();
    }
  }


}
