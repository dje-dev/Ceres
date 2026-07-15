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
using Ceres.Chess;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PUCT;
using Ceres.MCGS.Search.TPS;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Shared kernel that recomputes a single node's Q from its children using exactly the rule the
/// engine's backup phase applies. Used by both the full-graph diagnostic (BottomUpQRecalculator)
/// and the selective amortized propagator (SelectiveQPropagator) so the recompute logic - including
/// the CB-GPUCT branch and the edge guards - lives in exactly one place.
/// </summary>
internal static class QRecomputeHelper
{
  /// <summary>
  /// Returns whether a node's Q is meaningful to recompute. Nodes whose Q is undefined, fixed, or
  /// trivial are excluded: orphaned (graph reuse), not yet evaluated, never visited, leaf nodes with
  /// no expanded children (Q is just V), terminal nodes, and proven-checkmate nodes (Q pinned).
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  internal static bool IsEligibleForRecompute(GNode node)
  {
    return !(node.IsOldGeneration
          || !node.IsEvaluated
          || node.N == 0
          || node.NumEdgesExpanded == 0
          || node.Terminal.IsTerminal()
          || node.CheckmateKnownToExistAmongChildren);
  }


  /// <summary>
  /// Refreshes the cached child-Q on each (visited) outgoing edge from the child node's Q, then
  /// recomputes and stores this node's Q, returning the new value.
  ///
  /// When <paramref name="snapshotOrEmpty"/> is non-empty, child Q is read from that snapshot
  /// (indexed by node index) - used by the parallel full-graph sweep, where reading live child Q
  /// would race with concurrent in-place writes. When it is empty, child Q is read live (correct for
  /// single-threaded callers running in the quiescent post-backup region), and the edge's IsStale
  /// flag is cleared since the edge is now exactly current.
  ///
  /// The recompute mirrors the engine's backup rule:
  ///   - regularized backup active : the TPS tempered posterior (TPSScoreCalc.ComputeVBar).
  ///   - otherwise                 : pure Q = (sum over children of -edge.Q * edge.N + V) / N.
  /// Edges with N == 0 are skipped (no contribution; also avoids a NaN * 0). The result is
  /// idempotent: recomputing a node whose children are unchanged yields the same Q.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="snapshotOrEmpty">Snapshot of node Q by index, or empty to read child Q live.</param>
  /// <param name="paramsSelect"></param>
  /// <param name="regularizedBackupActive">ParamsSelect.RegularizedBackupActive (hoisted by callers).</param>
  /// <returns>The newly stored Q value.</returns>
  internal static double RecomputeNodeQ(GNode node,
                                        ReadOnlySpan<double> snapshotOrEmpty,
                                        ParamsSelect paramsSelect,
                                        bool regularizedBackupActive)
  {
    bool useSnapshot = !snapshotOrEmpty.IsEmpty;
    int numExpanded = node.NumEdgesExpanded;

    double sumChildW = 0;
    for (int childIndex = 0; childIndex < numExpanded; childIndex++)
    {
      GEdge edge = node.ChildEdgeAtIndex(childIndex);
      if (edge.N == 0)
      {
        // No backed-up visits along this edge: contributes nothing and is not refreshed
        // (guards against multiplying a possibly-NaN child Q by zero).
        continue;
      }

      if (edge.Type == GEdgeStruct.EdgeType.ChildEdge
       && !edge.ChildNodeIndex.IsNull
       && !edge.ChildNodeHasDrawKnownToExist)
      {
        if (useSnapshot)
        {
          edge.QChild = snapshotOrEmpty[edge.ChildNodeIndex.Index];
        }
        else
        {
          // Live read: the edge becomes exactly current, so it is no longer stale.
          edge.QChild = edge.ChildNode.Q;
          edge.IsStale = false;
        }
      }

      // Child Q is stored from the child's perspective, so negate it for the parent.
      // edge.Q (not edge.QChild) already dilutes for draw-by-repetition visits.
      sumChildW += -edge.Q * edge.N;
    }

    if (regularizedBackupActive)
    {
      // Mirror MCGSStrategyPUCT.BackupToNode under the TPS backup: store the
      // tempered-posterior value recomputed from the now-current child stats.
      double vBar = TPSScoreCalc.ComputeVBar(node, paramsSelect);
      node.NodeRef.Q = vBar;
      return vBar;
    }
    else
    {
      // Mirror the standard pure-Q backup. refreshSiblingContribution is left false so the existing
      // pseudo-transposition sibling blend is preserved and only the pure component is recomputed.
      double sumWChildrenAndSelf = sumChildW + node.NodeRef.V;
      node.ResetQUsingSumWChildrenAndSelf(sumWChildrenAndSelf, refreshSiblingContribution: false);
      return node.Q;
    }
  }


  /// <summary>
  /// Recomputes and stores this node's draw probability D exactly from its children, using the same
  /// aggregation rule as GNode.ComputeDFromChildren (keep the two in sync): each visited child edge
  /// contributes its draw mass - child.D for ordinary visits, 1 for draw-by-repetition visits,
  /// 1 for a drawn terminal edge or a child with a known available draw, 0 for a decisive terminal -
  /// plus the node's own DrawP for its initial self-eval, divided by N.
  ///
  /// Mirrors <see cref="RecomputeNodeQ"/>'s snapshot contract: when <paramref name="snapshotOrEmpty"/>
  /// is non-empty, child D is read from that snapshot (indexed by node index) for the parallel
  /// full-graph sweep, where reading live child D would race in-place writes; when empty, child D is
  /// read live (correct for single-threaded callers in the quiescent post-backup region). D is
  /// display-only, so unlike Q this maintains no per-edge cache and clears no stale flag.
  /// Intended to be called alongside RecomputeNodeQ on the same (eligible) nodes.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="snapshotOrEmpty">Snapshot of node D by index, or empty to read child D live.</param>
  /// <returns>The newly stored D value.</returns>
  internal static double RecomputeNodeD(GNode node, ReadOnlySpan<double> snapshotOrEmpty)
  {
    int n = node.N;
    if (n <= 1)
    {
      // No (visited) children to aggregate: D is just the node's own NN draw probability.
      double dSelf = node.DrawP;
      node.NodeRef.D = dSelf;
      return dSelf;
    }

    bool useSnapshot = !snapshotOrEmpty.IsEmpty;
    int numExpanded = node.NumEdgesExpanded;

    double dSum = node.DrawP; // self contribution (1 visit for initial eval)
    for (int childIndex = 0; childIndex < numExpanded; childIndex++)
    {
      GEdge edge = node.ChildEdgeAtIndex(childIndex);
      if (edge.N == 0)
      {
        continue;
      }

      if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        if (edge.ChildNodeIndex.IsNull)
        {
          continue;
        }

        if (edge.ChildNodeHasDrawKnownToExist)
        {
          dSum += 1.0 * edge.N;
        }
        else
        {
          double childD = useSnapshot ? snapshotOrEmpty[edge.ChildNodeIndex.Index] : edge.ChildNode.D;
          int nNonRep = edge.N - edge.NDrawByRepetition;
          dSum += childD * nNonRep + 1.0 * edge.NDrawByRepetition;
        }
      }
      else if (edge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn)
      {
        dSum += 1.0 * edge.N;
      }
      // TerminalEdgeDecisive: D = 0, no contribution.
    }

    double newD = dSum / n;
    node.NodeRef.D = newD;
    return newD;
  }
}
