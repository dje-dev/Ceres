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

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Repetition draw-by-repetition reconciliation and queries on the search-root subtree:
/// reclassifying coalesced edges whose move now completes a repetition as full draws
/// (ReconcileDrawByRepetitions), and the helper used by best-move selection to detect when a move
/// hands the opponent a repetition draw (DrawByRepetitionExistsAtChildEdgeAmongExpandedChildren).
/// </summary>
public partial struct GNode
{



  /// <summary>
  /// Statistics describing a ReconcileDrawByRepetitions pass (for diagnostics / yellow console output
  /// when reconciling deeper than the search root's direct children).
  /// </summary>
  internal struct RepetitionReconcileStats
  {
    /// <summary>Distinct nodes visited by the reconciliation walk.</summary>
    public int NodesWalked;
    /// <summary>Edges reclassified from (partly) non-draw to full draw-by-repetition.</summary>
    public int EdgesReconciled;
    /// <summary>Total visit mass reclassified (sum over reconciled edges of N - prior NDrawByRepetition).</summary>
    public long VisitMassReconciled;
    /// <summary>Deepest ply below the search root reached by the walk.</summary>
    public int MaxDepthReached;
    /// <summary>True if the walk was halted early by the node-count safety cap.</summary>
    public bool HitNodeCap;
  }


  /// <summary>
  /// Reconciles the draw-by-repetition classification of expanded child edges, from this (search-root)
  /// node down to <paramref name="maxDepthPlies"/> plies, against the CURRENT repetition context (the
  /// graph-root -> search-root spine plus prehistory).
  ///
  /// Why this is needed (PositionEquivalence mode only): nodes are coalesced by board position, so the
  /// same board recurring at different points in the game maps to ONE node which is reused across moves
  /// via graph reuse. An edge whose move completes a repetition NOW may have accumulated many non-draw
  /// visits earlier in the game when that same board was reached WITHOUT the repeating history. Those
  /// stale visits dilute the edge's aggregate Q (GEdgeStruct.Q dilutes child Q by the non-draw fraction
  /// N-NDrawByRepetition), so the forced-draw move is not valued as a draw and the engine can walk into
  /// (or fail to claim) a repetition with its eval frozen.
  ///
  /// Only HISTORY-LEVEL repetitions are reconciled: an edge is converted only when its child board is
  /// present in the spine/prehistory. Because the spine+prehistory is a common prefix of EVERY search
  /// path, such a board is a draw regardless of the in-tree path that reached it - so reclassifying the
  /// whole edge (NDrawByRepetition = N => edge.Q == 0) is exact even for shared (multi-parent) nodes and
  /// at any depth. Purely in-tree repetitions (board repeats an in-tree ancestor only) are path-dependent
  /// and are intentionally left to live selection's per-path detection. The test is identical to the one
  /// live selection uses at the search root (MCGSPath.HashFoundInHistoryOrPrehistory reduces to exactly
  /// HashFoundInGraphRootPathOrPrehistory there), so the reclassification is stable rather than re-diluted.
  ///
  /// Node Q values are refreshed bottom-up: a node is recomputed from its children after its subtree is
  /// processed, so a converted edge (or a changed descendant) propagates up to this node. Forward-only:
  /// edges that are not currently history repetitions are left untouched.
  /// </summary>
  /// <param name="nodesGraphToSearchRoot">Path from graph root to search root for spine/prehistory checking.</param>
  /// <param name="maxDepthPlies">How many plies below the search root to reconcile (1 = direct children only).</param>
  /// <returns>Statistics about the pass.</returns>
  internal RepetitionReconcileStats ReconcileDrawByRepetitions(ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot,
                                                               int maxDepthPlies)
  {
    RepetitionReconcileStats stats = default;
    if (maxDepthPlies < 1)
    {
      return stats;
    }

    // Visited: nodes already processed (coalesced DAG => process each once; history-rep status is
    // path-independent so a single pass suffices). Changed: subset whose Q was refreshed (so a node
    // reached via a second parent learns its child changed without reprocessing it).
    HashSet<int> visited = new();
    HashSet<int> changed = new();
    ReconcileRecursive(this, nodesGraphToSearchRoot, 1, maxDepthPlies, NodeWalkCap(maxDepthPlies, N), visited, changed, ref stats);
    return stats;
  }


  // The reconciliation walk visits the distinct expanded nodes within maxDepth plies of the search root,
  // which is bounded by (expanded branching)^maxDepth. Expanded branching is far below the ~35 legal-move
  // average (only visited edges are walked, expansion thins with depth, transpositions coalesce) and lower
  // still in the repetition-prone/low-material positions where the gate lets this run (empirically ~6-15).
  // So the safety cap is scaled with depth from a GENEROUS branching estimate: it is a pure backstop that
  // essentially never truncates real work (a typical level-4 walk is a few thousand to tens of thousands of
  // nodes, well under the level-4 cap of 25^4 ~= 390K) but bounds the pathological high-branching case far
  // more tightly at shallow depths than a single flat ceiling would. The cap is additionally limited to 20%
  // of the root's visit count so the walk can never consume a large fraction of the (reused) tree.
  // RepetitionReconcileStats.HitNodeCap reports the rare case the cap actually bites.
  private const int RECONCILE_BRANCH_ESTIMATE = 25;
  private const int RECONCILE_MAX_NODES_ABSOLUTE = 20_000_000;

  private static int NodeWalkCap(int maxDepthPlies, int rootN)
  {
    long cap = 1;
    for (int i = 0; i < maxDepthPlies; i++)
    {
      cap *= RECONCILE_BRANCH_ESTIMATE;
      if (cap >= RECONCILE_MAX_NODES_ABSOLUTE)
      {
        cap = RECONCILE_MAX_NODES_ABSOLUTE;
        break;
      }
    }

    // Never walk more than 20% of the root's visits (bounds the walk relative to the reused tree size).
    return (int)Math.Min(cap, rootN / 5);
  }

  private bool ReconcileRecursive(GNode node,
                                  ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot,
                                  int depth, int maxDepthPlies, int nodeWalkCap,
                                  HashSet<int> visited, HashSet<int> changed,
                                  ref RepetitionReconcileStats stats)
  {
    int idx = node.Index.Index;
    if (!visited.Add(idx))
    {
      // Already fully processed via another parent; report its cached changed status.
      return changed.Contains(idx);
    }

    stats.NodesWalked++;
    if (depth > stats.MaxDepthReached)
    {
      stats.MaxDepthReached = depth;
    }

    bool anyChanged = false;

    foreach (GEdge edge in node.ChildEdgesExpanded)
    {
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        continue;
      }

      // Descend first (post-order) so a child's Q is refreshed before this node is recomputed.
      if (depth < maxDepthPlies && !edge.ChildNode.IsNull && stats.NodesWalked < nodeWalkCap)
      {
        if (ReconcileRecursive(edge.ChildNode, nodesGraphToSearchRoot, depth + 1, maxDepthPlies, nodeWalkCap, visited, changed, ref stats))
        {
          anyChanged = true;
        }
      }
      else if (depth < maxDepthPlies && stats.NodesWalked >= nodeWalkCap)
      {
        stats.HitNodeCap = true;
      }

      // Convert this edge if its child board is a history-level repetition not yet fully classified.
      // (Terminal edges and N==0 edges carry nothing to dilute and are skipped by the guard below.)
      if (edge.N > 0 && edge.NDrawByRepetition < edge.N)
      {
        bool haveSeenRepetition = false;
        if (MCGSPath.HashFoundInGraphRootPathOrPrehistory(Graph, nodesGraphToSearchRoot,
                                                          edge.ChildNode.HashStandalone, ref haveSeenRepetition))
        {
          stats.VisitMassReconciled += edge.N - edge.NDrawByRepetition;
          edge.NDrawByRepetition = edge.N;
          stats.EdgesReconciled++;
          anyChanged = true;
        }
      }
    }

    if (anyChanged)
    {
      // A child edge value collapsed to a draw (or a descendant changed): refresh this node's Q.
      node.ResetNodeQFromChildren(false);
      changed.Add(idx);
    }

    return anyChanged;
  }


  /// <summary>
  /// Checks if any expanded child edges from the child node at the given edge
  /// result in a draw by repetition (i.e., the position after making the move
  /// matches a position already seen in the current path or prehistory).
  /// </summary>
  /// <param name="edge">The edge whose child node's children are to be checked.</param>
  /// <param name="nodesGraphToSearchRoot">Path from graph root to search root for prehistory checking.</param>
  /// <param name="pathHashes">Set of position hashes representing the current path to root.</param>
  /// <returns>True if any grandchild position is a repetition of a position in the path or prehistory.</returns>
  public static bool DrawByRepetitionExistsAtChildEdgeAmongExpandedChildren(GNode parentNode,
                                                                            GEdge edgeIgnore,
                                                                            ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot,
                                                                            HashSet<PosHash64> pathHashes)
  {
    foreach (GEdge childEdge in parentNode.ChildEdgesExpanded)
    {
      if (childEdge != edgeIgnore
       && childEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        PosHash64 grandchildHash = childEdge.ChildNode.HashStandalone;
        bool haveSeenRepetition = false;
        bool isDrawByRepetition = (pathHashes != null && pathHashes.Contains(grandchildHash))
                               || MCGSPath.HashFoundInGraphRootPathOrPrehistory(parentNode.Graph, nodesGraphToSearchRoot,
                                                                                grandchildHash, ref haveSeenRepetition);
        if (isDrawByRepetition)
        {
          return true;
        }
      }
    }

    return false;
  }
}
