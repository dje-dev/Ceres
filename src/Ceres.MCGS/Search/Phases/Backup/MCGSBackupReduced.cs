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
using System.Diagnostics;
using System.Threading;

using Ceres.Base.DataTypes;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Strategies;

#endregion

namespace Ceres.MCGS.Search.Phases.Backup;


/// <summary>
/// Applies updates to graph (Q, N, etc. for both edges and nodes) arising from selected visits.
/// A "reduction" algorithm is used to coalesce values from multiple paths in a graph:
///
/// During select each path visit is armed with the number of visits attempted through it
/// (MCGSPathVisit.NumVisitsAttemptedPendingBackup, incremented again if later paths split off
/// through the same visit). Backup walks each path leaf to root and at every level
/// (under the parent node's lock) subtracts its attempted visits from that counter:
///   - if the counter remains positive, other paths have yet to pass through this visit;
///     the current path deposits its contribution (visit counts plus leaf-value moments)
///     into the visit's Accumulator and stops, leaving the ancestor updates to them;
///   - the path which brings the counter to zero merges (and clears) the accumulator into
///     its own contribution and continues upward carrying the combined update.
/// The per-parent NodeLockBlock serializes the counter decrement and the accumulator access,
/// so the final arrival always observes all earlier deposits. Edge in-flight counters are
/// likewise decremented exactly once per attempted visit (deferred ones by the final carrier).
/// </summary>
public partial class MCGSBackup
{
  public void BackupReduced(MCGSSelectBackupStrategyBase strategy, MCGSPath path, int iteratorID)
  {
    ref MCGSPathVisit leafVisitRef = ref path.LeafVisitRef;

    int numVisitsAttempted = leafVisitRef.NumVisitsAttempted;
    int numVisitsAccepted = leafVisitRef.NumVisitsAccepted.Value;

    BackupValue initialBackupValue = (numVisitsAccepted > 0) ? InitialBackupValueForPath(path)
                                                             : default;

    // Per-visit repetition-draw fraction at the leaf, propagated into each path node's
    // RepDrawFraction running average (additive: internal levels re-read the child node's
    // CURRENT fraction and add it for the new visits, rather than threading the leaf value).
    // NOTE: D used to be plumbed this same additive way; it is now delta-corrected on internal
    //       edges (see STEP 4 below), so R no longer exactly mirrors D's update.
    //   - coalesce-mode draw by repetition: 1 (the visit itself is a repetition draw);
    //   - terminal drawn edge carrying the NDrawByRepetition kind sentinel
    //     (repetition/50-move draw): 1; history-free drawn terminals: 0;
    //   - node-terminated paths (NN eval, transposition links/copies, auto-extended):
    //     the leaf node's own current fraction (0 for fresh nodes; meaningful when
    //     linking into an existing rep-heavy subgraph).
    double leafR = 0;
    if (numVisitsAccepted > 0)
    {
      GEdge leafEdge = leafVisitRef.ParentChildEdge;
      if (path.TerminationReason == MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode)
      {
        leafR = 1;
      }
      else if (leafEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        leafR = leafEdge.ChildNode.RepDrawFraction;
      }
      else if (leafEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn && leafEdge.NDrawByRepetition > 0)
      {
        leafR = 1;
      }
    }

    bool haveProcessedLeaf = false;

    // D of the node updated in the PREVIOUS loop iteration, captured BEFORE its update.
    // Because the loop walks leaf->root, that node is the CHILD of the edge processed in the
    // next iteration, so this value is the child's pre-update ("last pushed") D. It plays the
    // role that the stored edge.QChild cache plays for Q's delta-correcting backup, but is
    // reconstructed from the walk order instead of being stored on the edge (so no growth of
    // the graph data structures). NaN until the first node is updated.
    double childDBeforeBackup = double.NaN;

    // Leaf-value volatility tracking: thread the first two moments (sum and sum-of-squares) of the
    // leaf value backed up along this path up through the SAME accumulator rails that carry the
    // visit counts, so each node can maintain an exact windowed dispersion estimate of the leaf
    // values it sees (rather than only ever seeing the aggregate Q-movement delta, which dilutes
    // and cancels that dispersion). Gated by the (default off) ParamsSearch flag.
    bool trackVol = trackLeafValueVolatility
                 && numVisitsAccepted > 0
                 && !double.IsNaN(initialBackupValue.V);
    // sV: running sum of leaf values in the CURRENT edge's child perspective (negated once per
    // level as we ascend); sV2: sum of squares (perspective-invariant). Both mirror numVisitsAccepted.
    double sV = trackVol ? initialBackupValue.V * numVisitsAccepted : 0;
    double sV2 = trackVol ? initialBackupValue.V * initialBackupValue.V * numVisitsAccepted : 0;

    // Perform backup, starting from the leaf and working toward root.
    foreach (MCGSPathVisitMember visitPair in path.PathVisitsLastBackedUpToRoot)
    {
      ref MCGSPathVisit visitPathRef = ref visitPair.PathVisitRef;
      GEdge visitEdge = visitPathRef.ParentChildEdge;

      // Prefetch the edge structure to reduce memory latency when accessed later
      visitEdge.Prefetch();

      GNode parentNode = visitEdge.ParentNode;

      using (new NodeLockBlock(parentNode))
      {
        // Check if this is a synchronization edge where multiple paths converge.
        // If other paths have yet to arrive here, our contribution has been deposited
        // into the accumulator and the last of them will finish the work.
        if (!TryContinueThroughMergePoint(ref visitPathRef, haveProcessedLeaf,
                                          ref numVisitsAttempted, ref numVisitsAccepted,
                                          ref sV, ref sV2))
        {
          return;
        }

        // Possibly initiate prefetch of our parent (for speed)
        if (!parentNode.IsSearchRoot)
        {
          PrefetchChild(parentNode, visitPathRef.IndexOfChildInParent);
        }

        // STEP 1: capture the W previously contributed upward by this edge before this update.
        int visitEdgeN = visitEdge.N;
        double priorEdgeW = visitEdgeN == 0 ? 0 : visitEdgeN * visitEdge.Q;
        Debug.Assert(numVisitsAccepted <= short.MaxValue);
        visitPathRef.NumVisitsAccepted = (short)numVisitsAccepted;

        // STEP 2: update the edge: (a) increase N by numVisitsAccepted, and (b) reset Q to be same as child.
        if (numVisitsAccepted > 0)
        {
          BackupVisitToEdge(strategy, path, visitEdge, haveProcessedLeaf, numVisitsAccepted, in initialBackupValue);
        }

        // STEP 3: decrement NInFlight.
        Debug.Assert(numVisitsAttempted <= short.MaxValue);
        GNodeStruct.UpdateEdgeNInFlightForIterator(visitEdge, iteratorID, (short)-numVisitsAttempted);

        // STEP 4: update parent node fields.
        if (numVisitsAccepted > 0)
        {
          // Capture the W contributed by this edge after the update.
          // Use the difference from prior W contribution to determine update magnitude,
          // thereby capturing the impact of all changes to child Q since this path last visited
          // (it may have happened that other transposition paths had caused interim updates to child Q).
          // N.B. edge W is in the child's perspective, so the parent's W moves by the
          //      negated change (prior minus new).
          double newEdgeW = visitEdge.N == 0 ? 0 : visitEdge.N * visitEdge.Q;
          double parentPerspectiveDeltaW = priorEdgeW - newEdgeW;

          // Snapshot this node's D before it is updated below. The next (parent) iteration reads
          // it as the pre-update child D for its delta-correcting D backup (this node is the
          // child of the edge processed next). Captured before ANY mutation of parentNode.
          double parentDBeforeUpdate = parentNode.D;

          if (!haveProcessedLeaf && visitEdge.N > 0 && visitEdge.Q <= -1)
          {
            if (!parentNode.CheckmateKnownToExistAmongChildren) // only do first time
            {
              parentNode.UpdateNodeForProvenChildLoss((FP16)(float)-visitEdge.Q, 0);
            }
            parentNode.NodeRef.N += numVisitsAccepted;
          }
          else
          {
            // Compute the parent-node deltas for this edge's contribution. The repetition-draw
            // fraction (deltaR) and the leaf edge keep the additive running-average plumbing; the
            // D update for internal edges is delta-correcting (see below).
            double newParentDeltaD;
            double childR;
            if (haveProcessedLeaf)
            {
              GNode childNode = visitEdge.ChildNode;
              double childD = childNode.D;
              childR = childNode.RepDrawFraction;

              // Delta-correcting D backup, mirroring the priorEdgeW/newEdgeW telescoping used for
              // Q: re-credit ALL of this edge's visits at the child's CURRENT D (not just the new
              // ones), so drift in the child's D since prior visits is healed. The child's
              // pre-update D (childDBeforeBackup) stands in for the not-stored edge.QChild analog.
              // The edge's draw-by-repetition mass is constant for a ChildEdge during backup and
              // contributes +rep to both the old and new contribution, so it cancels in the delta.
              //   contrib = (edge.N - edge.NDrawByRepetition) * child.D   (+ rep, cancels)
              // Exact for tree structure; shared (multi-parent) edges are corrected only while
              // on-path -- a bounded approximation (off-path parents are not updated here).
              int edgeRep = visitEdge.NDrawByRepetition;
              double childDOld = double.IsNaN(childDBeforeBackup) ? childD : childDBeforeBackup;
              newParentDeltaD = ((visitEdge.N - edgeRep) * childD) - ((visitEdgeN - edgeRep) * childDOld);
            }
            else
            {
              // Leaf edge: first push of this leaf's value (or a true childless leaf whose D is
              // constant), so the additive form is already exact.
              newParentDeltaD = initialBackupValue.D * numVisitsAccepted;
              childR = leafR;
            }

            double newParentDeltaR = childR * numVisitsAccepted;

            // Fold the batch of leaf values backed up through this node into its volatility
            // estimate, measured about the node's pre-update pure Q. The node-perspective leaf
            // sum is -sV. Skipped when checkmate is known (Q is pinned, dispersion meaningless).
            if (trackVol && !parentNode.CheckmateKnownToExistAmongChildren)
            {
              double qRef = parentNode.N > 0 ? parentNode.ComputeQPure() : (-sV / numVisitsAccepted);
              parentNode.NodeRef.LeafValueVolatility.AddBatch(qRef, -sV, sV2, numVisitsAccepted);
            }

            strategy.BackupToNode(parentNode, numVisitsAccepted, parentPerspectiveDeltaW, newParentDeltaD, newParentDeltaR);
          }

          // Carry this node's pre-update D to the next (parent) iteration, where this node is the
          // child of the edge being backed up (set in both branches above, including proven-loss).
          childDBeforeBackup = parentDBeforeUpdate;
        }

        // Ascend one ply: flip the leaf-sum sign so sV stays in the next edge's child perspective.
        sV = -sV;

        haveProcessedLeaf = true;
      }
    }
  }


  /// <summary>
  /// Handles merge-point coordination at a visit potentially shared by multiple paths
  /// (must be called while holding the parent node lock, which serializes the
  /// pending counter decrement and the accumulator access).
  ///
  /// Decrements the pending-backup counter armed during select. If other paths have yet
  /// to pass through this visit, deposits this path's contribution into the visit's
  /// accumulator and returns false (the final arrival completes the work). Otherwise
  /// merges (and clears) any contributions deposited by earlier arrivals into the
  /// ref arguments and returns true.
  /// </summary>
  private static bool TryContinueThroughMergePoint(ref MCGSPathVisit visitPathRef, bool haveProcessedLeaf,
                                                   ref int numVisitsAttempted, ref int numVisitsAccepted,
                                                   ref double sV, ref double sV2)
  {
    if (!haveProcessedLeaf)
    {
      // Leaf visits are never shared between paths, just decrement.
      Interlocked.Add(ref visitPathRef.NumVisitsAttemptedPendingBackup, -numVisitsAttempted);
      return true;
    }

    int newNumPendingBackup = Interlocked.Add(ref visitPathRef.NumVisitsAttemptedPendingBackup, -numVisitsAttempted);
    Debug.Assert(numVisitsAccepted >= 0);

    if (newNumPendingBackup > 0)
    {
      // There are more paths yet to be processed that will pass thru this visit.
      // We'll exit and let them eventually finish the work.
      // But first we have to update the accumulator for the child.
      visitPathRef.Accumulator.DoAdd(numVisitsAttempted, numVisitsAccepted, sV, sV2);
      return false;
    }

    if (visitPathRef.Accumulator.NumVisitsAttempted > 0)
    {
      // Other paths already visited this child and accumulated updates.
      // We first apply our own update and then reload from accumulator
      // so we see all accumulated values.
      ref MCGSBackupAccumulator inFlightAccumulator = ref visitPathRef.Accumulator;
      inFlightAccumulator.DoAdd(numVisitsAttempted, numVisitsAccepted, sV, sV2);

      numVisitsAttempted = inFlightAccumulator.NumVisitsAttempted;
      numVisitsAccepted = inFlightAccumulator.NumVisitsAccepted;
      sV = inFlightAccumulator.SumV;
      sV2 = inFlightAccumulator.SumV2;

      // Clear after consumption so that a protocol violation (the counter reaching zero
      // twice due to a select-side arming bug) cannot silently re-merge stale contributions.
      inFlightAccumulator = default;
    }

    return true;
  }


  /// <summary>
  /// Applies the edge update for one visit: increments edge N by the accepted visits
  /// and refreshes the edge's cached child Q (for the leaf edge using the initial backup
  /// value, with special handling for draw-by-repetition terminations).
  /// </summary>
  private static void BackupVisitToEdge(MCGSSelectBackupStrategyBase strategy, MCGSPath path, GEdge visitEdge,
                                        bool haveProcessedLeaf, int numVisitsAccepted, in BackupValue initialBackupValue)
  {
    if (haveProcessedLeaf)
    {
      GNode childNode = visitEdge.ChildNode;
      strategy.BackupToEdge(visitEdge, numVisitsAccepted, childNode.Q, childNode.D, visitEdge.ChildNodeHasDrawKnownToExist);
      return;
    }

    // Leaf edge. On a draw-by-repetition, pass ChildNode.Q (NOT initialBackupValue.V == 0)
    // as the cached QChild so the edge faithfully mirrors the child node's Q; edge.Q still
    // dilutes to the draw value via NDrawByRepetition. Writing 0 here clobbers QChild and
    // forces edge.Q = 0 until the next non-rep visit -- and for an edge whose child is the
    // search root (whose position is in every path's history) there is NEVER a non-rep
    // visit, so QChild would stay stuck at 0 forever while the child's true Q is nonzero.
    // (Matches the internal-edge case above, which already passes ChildNode.Q.)
    bool isRepDraw = path.TerminationReason == MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode;
    double newQChildForLeaf = isRepDraw ? visitEdge.ChildNode.Q : initialBackupValue.V;
    if (isRepDraw && double.IsNaN(newQChildForLeaf))
    {
      // Unevaluated draw-by-repetition placeholder child (Q is the NaN "unevaluated" sentinel):
      // its value is a draw (0), never NaN. Passing NaN here would store edge.QChild = NaN and
      // make deltaQ = NaN (silently disabling off-path propagation) and feed NaN*0 into the
      // edge.Q dilution. A search-root child has a real Q (not NaN) and is therefore unaffected.
      newQChildForLeaf = 0;
    }

    strategy.BackupToEdge(visitEdge, numVisitsAccepted, newQChildForLeaf, initialBackupValue.D, false);
    if (isRepDraw)
    {
      // The NDrawByRepetition for the edge immediately leading to the draw is incremented
      // in tandem with the primary update (but is not itself backed up to ancestors).
      visitEdge.IncrementNDrawRepetition(numVisitsAccepted);
    }
  }
}
