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
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GEdgeHeaders;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Partial class containing computed metrics and statistics for GNode.
/// </summary>
public readonly partial struct GNode
{
  /// <summary>
  /// Calculates the Shannon entropy of the policy prior distribution.
  /// Entropy = -sum(p_i * log(p_i)) where p_i are the policy prior probabilities.
  /// Higher entropy indicates more uncertainty/spread in the policy.
  /// </summary>
  public readonly double PolicyPriorEntropy
  {
    get
    {
      double entropy = 0;
      int numMoves = NumPolicyMoves;
      
      for (int i = 0; i < numMoves; i++)
      {
        GEdgeHeaderStruct edgeHeader = ChildEdgeHeaderAtIndex(i);
        double p = edgeHeader.P;
        
        // Only include non-zero probabilities (0 * log(0) is defined as 0)
        if (p > 0)
        {
          entropy -= p * Math.Log(p);
        }
      }
      
      return entropy;
    }
  }


  /// <summary>
  /// Calculates the Shannon entropy based on the actual visit distribution.
  /// Entropy = -sum(p_i * log(p_i)) where p_i are the empirical visit probabilities.
  /// Uses visit counts from visited edges, normalized by (N - 1) to exclude the root visit.
  /// Higher entropy indicates more exploration spread across moves.
  /// </summary>
  public readonly double PolicyVisitsEntropy
  {
    get
    {
      double entropy = 0;
      int totalVisits = N - 1; // Exclude the root visit itself
      
      if (totalVisits <= 0)
      {
        return 0;
      }

      using (new NodeLockBlock(this))
      {
        int numExpandedEdges = NumEdgesExpanded;

        for (int i = 0; i < numExpandedEdges; i++)
        {
          GEdge edge = ChildEdgeAtIndex(i);
          int edgeN = edge.N;

          if (edgeN > 0)
          {
          double p = (double)edgeN / totalVisits;
            entropy -= p * Math.Log(p);
          }
        }

        return entropy;
      }
    }
  }


  /// <summary>
  /// Descends the principal variation (following the child edge with highest N at each step)
  /// and returns the number of plies until an irreversible move is encountered.
  /// An irreversible move is one that resets the 50-move counter (pawn move or capture).
  /// </summary>
  /// <returns>The number of plies until an irreversible move, or null if no irreversible move is found in the PV.</returns>
  public readonly int? PlyUntilPVIsIrreversibleMove() => PlyUntilPVIsIrreversibleMoveWithMove().ply;


  /// <summary>
  /// Descends the principal variation (following the child edge with highest N at each step)
  /// and returns the number of plies until an irreversible move is encountered along with the move itself.
  /// An irreversible move is one that resets the 50-move counter (pawn move or capture).
  /// </summary>
  /// <returns>A tuple containing the number of plies until an irreversible move (or null if none found) and the irreversible move edge (or null if none found).</returns>
  public readonly (int? ply, GEdge? irreversibleEdge) PlyUntilPVIsIrreversibleMoveWithMove()
  {
    int plyCount = 0;
    GNode currentNode = this;

    // Track visited node indices to detect cycles (draw by repetition).
    HashSet<int> visitedNodeIndices = new();

    while (currentNode.NumEdgesExpanded > 0)
    {
      // In PositionEquivalence mode, paths can lead back to the same node.
      // When a repeated node is encountered, stop since this indicates a draw by repetition.
      if (!visitedNodeIndices.Add(currentNode.Index.Index))
      {
        break;
      }

      // Find the child edge with the highest N (principal variation move).
      GEdge bestEdge = currentNode.EdgeWithMaxValue(edge => edge.N);

      if (bestEdge.IsNull || bestEdge.N == 0)
      {
        break;
      }

      plyCount++;

      // Check if this move is irreversible (resets the 50-move counter).
      if (bestEdge.MoveMG.ResetsMove50Count)
      {
        return (plyCount, bestEdge);
      }

      // Descend to the child node.
      currentNode = bestEdge.ChildNode;

      // Safeguard against infinite loops (should not normally be reached
      // since cycle detection above should catch repetitions first).
      if (plyCount > 500)
      {
        break;
      }
    }

    // No irreversible move found in the PV.
    return (null, null);
  }
}
