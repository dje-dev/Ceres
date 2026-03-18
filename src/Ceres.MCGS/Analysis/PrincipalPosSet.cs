#region Using directives

using System.Collections.Generic;
using System;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Analysis;

/// <summary>
/// Helper methods for extracting principal positions from MCGS search graphs.
/// </summary>
public class PrincipalPosSet
{
  /// <summary>
  /// The starting position used to build this set.
  /// </summary>
  public MGPosition StartPosition { get; }

  /// <summary>
  /// The root node from which the search started.
  /// </summary>
  public GNode StartSearchNode { get; }

  /// <summary>
  /// Minimum number of visits required for best child.
  /// </summary>
  public int MinVisits { get; }

  /// <summary>
  /// Maximum absolute Q difference from root (for filtering).
  /// </summary>
  public float MaxAbsQDiff { get; }

  /// <summary>
  /// The collected principal positions meeting the criteria.
  /// </summary>
  public List<PrincipalPos> Members { get; }

  private PrincipalPosSet(MGPosition startPosition, GNode startSearchNode, int minVisits, float maxAbsQDiff, List<PrincipalPos> members)
  {
    StartPosition = startPosition;
    StartSearchNode = startSearchNode;
    MinVisits = minVisits;
    MaxAbsQDiff = maxAbsQDiff;
    Members = members;
  }

  /// <summary>
  /// Collects all principal positions where the best child (by visit count) has fewer than minVisits.
  /// Descends from the start node and follows every path until reaching a node whose
  /// best child has fewer than minVisits, at which point that parent node is added 
  /// to the collection along with the complete path from root.
  /// </summary>
  /// <param name="startPosition">The chess position at the root node</param>
  /// <param name="startSearchNode">The root node to start from</param>
  /// <param name="minVisits">Minimum number of visits required for best child (e.g. 100)</param>
  /// <param name="maxAbsQDiff">Maximum absolute Q difference from root (filters leaf nodes)</param>
  /// <returns>A PrincipalPositionSet containing positions meeting the criteria</returns>
  public static PrincipalPosSet CollectNodesAboveVisitThreshold(MGPosition startPosition, GNode startSearchNode, int minVisits = 100, float maxAbsQDiff = float.MaxValue)
  {
    List<PrincipalPos> collectedPositions = new();
    Dictionary<PosHash64, int> visitedIndices = new();
    List<(GNode, MGPosition)> currentPath = new();

    CollectNodesRecursive(startPosition, startSearchNode, startSearchNode, minVisits, collectedPositions, visitedIndices, currentPath, maxAbsQDiff);

    return new PrincipalPosSet(startPosition, startSearchNode, minVisits, maxAbsQDiff, collectedPositions);
  }

  private static void CollectNodesRecursive(MGPosition position, GNode startSearchNode, GNode node, int minVisits,
                                           List<PrincipalPos> collectedPositions,
                                           Dictionary<PosHash64, int> visitedIndices,
                                           List<(GNode, MGPosition)> currentPath,
                                           float maxAbsQDiff)
  {
    // Track occurrences of this node
    if (visitedIndices.ContainsKey(node.HashStandalone))
    {
      visitedIndices[node.HashStandalone]++;
      return;
    }

    visitedIndices[node.HashStandalone] = 1;

    // Add current node to the path
    currentPath.Add((node, position));

    // If this node has no expanded children, skip it
    if (node.NumEdgesExpanded == 0)
    {
      currentPath.RemoveAt(currentPath.Count - 1);
      return;
    }

    // Find the child with the highest visit count
    int maxChildVisits = 0;
    foreach (GEdge childEdge in node.ChildEdgesExpanded)
    {
      if (childEdge.ChildNode.N > maxChildVisits)
      {
        maxChildVisits = childEdge.ChildNode.N;
      }
    }

    // If the best child has fewer than minVisits, collect this parent node and stop
    if (maxChildVisits < minVisits && node != startSearchNode)
    {
      // Calculate Q difference from root to check if it meets the maxAbsQDiff filter
      int depth = currentPath.Count;
      bool isRootPlayerPerspective = (depth % 2) == 1;
      double qFromRootPerspective = isRootPlayerPerspective ? node.Q : -node.Q;
      double qDiff = qFromRootPerspective - startSearchNode.Q;
      
      // Only add if Q difference is within acceptable range
      if (Math.Abs(qDiff) <= maxAbsQDiff)
      {
        // Create a copy of the current path for this principal position
        List<(GNode, MGPosition)> pathCopy = new(currentPath);
        int numOccurrences = visitedIndices[node.HashStandalone];
        PrincipalPos principalPosition = new(node, position, pathCopy, numOccurrences);
        collectedPositions.Add(principalPosition);
      }
      
      currentPath.RemoveAt(currentPath.Count - 1);
      return; // Stop descent along this path
    }

    // Otherwise, continue descending to all children
    foreach (GEdge childEdge in node.ChildEdgesExpanded)
    {
      GNode childNode = childEdge.ChildNode;

      MGPosition childPosition = position;
      MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(childEdge.Move, in position);
      childPosition.MakeMove(mgMove);
      if (childNode.N > minVisits)
      {
        CollectNodesRecursive(childPosition, startSearchNode, childNode, minVisits, collectedPositions, visitedIndices, currentPath, maxAbsQDiff);
      }
    }

    // Remove current node from the path when backtracking
    currentPath.RemoveAt(currentPath.Count - 1);
  }
}
