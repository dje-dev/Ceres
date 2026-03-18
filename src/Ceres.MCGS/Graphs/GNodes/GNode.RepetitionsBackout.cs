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
/// Logic for removing invalidated draw-by-repetition visits from nodes/edges.
/// 
/// When the search root changes, some draw-by-repetition visits may become invalidated
/// because the positions they repeated against are no longer in the path to the new root.
/// 
/// The update logic depends on whether the position is still a draw-by-repetition (DBR)
/// and whether we can safely propagate N decrements upward (requires single-parent path to root):
/// 
///   Condition      | Action                              | Only if canPropagateUpward | Update ancestor N
///   ---------------|-------------------------------------|----------------------------|------------------
///   Now a DBR      | NDrawByRepetition = N               | true                       | no
///   Now not DBR    | NDrawByRepetition = 0, N -= removed | true                       | yes
/// 
/// When canPropagateUpward is false (multi-parent node exists somewhere above in the path),
/// we skip all updates because we cannot correctly apportion the removed N across multiple parent edges.
/// </summary>
public partial struct GNode
{
  /// <summary>
  /// Attempts to remove invalidated draw-by-repetition visits
  /// from this node's edges up to the specified depth.
  /// 
  /// Repetitions can become invalidated when after the search root
  /// is changed some of the now truncated-paths should no longer
  /// have their draw-by-repetition visits counted.
  /// </summary>
  /// <param name="depthToProcess"></param>
  /// <param name="nodesGraphToSearchRoot"></param>
  /// <returns></returns>
  internal void RemoveInvalidatedDrawByRepetitionsFromNodeEdges(int depthToProcess, ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot)
  {
    HashSet<PosHash64> pathHashes = [HashStandalone];
    RemoveInvalidatedDrawByRepetitionsFromNodeEdges(this, nodesGraphToSearchRoot, depthToProcess, pathHashes, canPropagateUpward: true);
  }


  private int RemoveInvalidatedDrawByRepetitionsFromNodeEdges(GNode node,
                                                              ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot,
                                                              int depthToProcess, HashSet<PosHash64> pathHashes,
                                                              bool canPropagateUpward)
  {
    bool madeChangesAnyEdge = false;
    int totalRemovedFromChildren = 0;

    foreach (GEdge edge in node.ChildEdgesExpanded)
    {
      if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        // Recurse to deeper levels first (process deepest first)
        if (depthToProcess > 1)
        {
          GNode childNode = edge.ChildNode;

          // If the child node has multiple parents, we cannot safely propagate
          // removals upward through this edge because we don't know what fraction
          // of the child's removed visits came through this specific edge vs other parent edges.
          bool childHasMultipleParents = childNode.NumParentsMoreThanOne;
          bool canPropagateFromChild = canPropagateUpward && !childHasMultipleParents;

          pathHashes.Add(childNode.HashStandalone);
          int removedBelow = RemoveInvalidatedDrawByRepetitionsFromNodeEdges(childNode, nodesGraphToSearchRoot, depthToProcess - 1, pathHashes, canPropagateFromChild);
          if (removedBelow > 0 && canPropagateFromChild)
          {
            madeChangesAnyEdge = true;
            totalRemovedFromChildren += removedBelow;

            // Propagate the decrement up: this edge's N should be reduced by the amount removed from the subtree below.
            Debug.Assert(edge.N >= removedBelow);
            edge.N -= removedBelow;
          }
          pathHashes.Remove(childNode.HashStandalone);
        }

        // Then process this level (direct repetition on this edge)
        if (edge.NDrawByRepetition > 0)
        {
          int removedFromThisEdge = RemoveInvalidatedDrawByRepetitionFromEdge(edge, nodesGraphToSearchRoot, pathHashes, canPropagateUpward);
          if (removedFromThisEdge > 0)
          {
            madeChangesAnyEdge = true;
            totalRemovedFromChildren += removedFromThisEdge;
          }
        }
      }
    }

    if (madeChangesAnyEdge && canPropagateUpward)
    {
      // Decrement this node's N by the total amount removed from all child edges.
      // Note: NDrawByRepetition is only set on the leaf edge (the edge immediately
      // leading to the draw by repetition) and is NOT propagated up the path during backup.
      // Therefore we only decrement N here, not NDrawByRepetition.
      Debug.Assert(node.NodeRef.N >= totalRemovedFromChildren);
      node.NodeRef.N -= totalRemovedFromChildren;

      // Trigger immediate recalc/reset of Q since some were removed.
      node.ResetNodeQFromChildren(false);
    }

    return totalRemovedFromChildren;
  }


  private int RemoveInvalidatedDrawByRepetitionFromEdge(GEdge edge,
                                                        ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot,
                                                        HashSet<PosHash64> pathHashes,
                                                        bool canPropagateUpward)
  {
    // Skip if we cannot propagate upward (multiple parents somewhere above).
    if (!canPropagateUpward)
    {
      return 0;
    }

    bool haveSeenRepetition = false;
    bool isDrawByRepetition = pathHashes.Contains(edge.ChildNode.HashStandalone)
                           || MCGSPath.HashFoundInGraphRootPathOrPrehistory(Graph, nodesGraphToSearchRoot,
                                                                            edge.ChildNode.HashStandalone, ref haveSeenRepetition);
    if (isDrawByRepetition)
    {
//if (edge.NDrawByRepetition != edge.N) Console.Write("+");

      // Convert all to be draw by repetition (no visits removed, just reclassified, no ancestor N update)
      edge.NDrawByRepetition = edge.N;
      return 0;
    }
    else
    {
      int nBeingRemoved = edge.NDrawByRepetition;

//if (nBeingRemoved > 0) Console.Write("-");

      // Update the edge
      Debug.Assert(edge.N >= nBeingRemoved);

      edge.NDrawByRepetition -= nBeingRemoved;
      edge.N -= nBeingRemoved;

      return nBeingRemoved;
    }
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
