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

using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdgeHeaders;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Various support methods for GNode relating to move (re-)ordering.
/// </summary>
public readonly partial struct GNode : IComparable<GNode>, IEquatable<GNode>
{
  /// <summary>
  /// Absolute mimimum Q superiority required to consider a move reorder.
  /// </summary>
  const double MIN_Q_SUPERIORITY = 0.05;

  /// <summary>
  /// Baseline minimum Q superiority required to consider a move reorder
  /// for a single node, but scaled down by sqrt(N).
  /// </summary>
  const double MIN_Q_SUPERIORITY_FOR_1_NODE = 0.20;

  /// <summary>
  /// Returns the GNode having the maximum Q value among sibling nodes.
  /// </summary>
  /// <param name="nodePos"></param>
  /// <param name="thisPolicyMove"></param>
  /// <returns></returns>
  public GNode MaxSiblingNodeAfterChildMove(in MGPosition nodePos, EncodedMove thisPolicyMove)
  {
    // Generate position after top policy move.
    MGMove topPolicyMoveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(thisPolicyMove, in nodePos);
    MGPosition childPosMG = nodePos;
    childPosMG.MakeMove(topPolicyMoveMG);

    PosHash64WithMove50AndReps hash64WithMoveAndReps;
    hash64WithMoveAndReps = MGPositionHashing.Hash64WithMove50AndRepsAdded(MGPositionHashing.Hash64(in childPosMG),
                                                                            childPosMG.RepetitionCount,
                                                                            childPosMG.Move50Category);

    GNode maxNode = NodeIndexSet.MaxNSiblingNode(Graph, hash64WithMoveAndReps);
    return maxNode;
  }


  /// <summary>
  /// Returns the GNode having the maximum Q value among sibling nodes.
  /// </summary>
  /// <param name="nodePos"></param>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  /// <exception cref="Exception"></exception>
  public GNode MaxSiblingNodeAfterChildMove(in MGPosition nodePos, int childIndex)
  {
    Debug.Assert(IsLocked);

    GEdgeHeaderStruct moveInfo = EdgeHeadersSpan[childIndex];

    if (!moveInfo.IsExpanded) // has not been converted concurrently to edge        
    {
      // Generate position after top policy move.
      MGMove topPolicyMoveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(moveInfo.Move, in nodePos);
      MGPosition childPosMG = nodePos;
      childPosMG.MakeMove(topPolicyMoveMG);

      PosHash64WithMove50AndReps hash64WithMoveAndReps;
      hash64WithMoveAndReps = MGPositionHashing.Hash64WithMove50AndRepsAdded(MGPositionHashing.Hash64(in childPosMG),
                                                                              childPosMG.RepetitionCount,
                                                                              childPosMG.Move50Category);

      GNode maxNode = NodeIndexSet.MaxNSiblingNode(Graph, hash64WithMoveAndReps);
      return maxNode;
    }
    else
    {
      throw new Exception("Not implemented: MaxSiblingNodeAfterChildMove when top move is expanded to edge");
    }
  }


  /// <summary>
  /// Scans forward from a specified child index to see if any of the next N child moves
  /// should be swapped to earlier position based on value (Q).
  /// </summary>
  /// <param name="position"></param>
  /// <param name="startIndex"></param>
  /// <param name="numLookForward"></param>
  /// <param name="minPolicyFraction"></param>
  public void CheckMoveOrderRearrangeAtIndex(in MGPosition position, int startIndex, int numLookForward, double minPolicyFraction)
  {
    if (startIndex >= NumPolicyMoves)
    {
      return;
    }

    // Find the value and index of the node having the best MaxSiblingNodeAfterChildMove
    // by scanning to the right a specified maximum number of positions.
    GNode bestNodeAtStartIndex = MaxSiblingNodeAfterChildMove(in position, startIndex);

    if (!bestNodeAtStartIndex.IsNull)
    {
      int bestIndex = startIndex;
      double bestQ = bestNodeAtStartIndex.Q;

      Span<GEdgeHeaderStruct> edgeHeaders = EdgeHeadersSpan;
      double startPolicy = (double)edgeHeaders[startIndex].P;

      for (int i = startIndex + 1; i <= Math.Min(NumPolicyMoves - 1, startIndex + numLookForward); i++)
      {
        double candidatePolicy = (double)edgeHeaders[i].P;
        double policyRatio = candidatePolicy / startPolicy;

        if (policyRatio < minPolicyFraction)
        {
          break;
        }

        GNode otherNode = MaxSiblingNodeAfterChildMove(in position, i);
        if (!otherNode.IsNull)
        {
          // Check that meets minimum criterion for N.
          double qDifference = bestNodeAtStartIndex.Q - otherNode.Q;

          double minQSuperiority = Math.Max(MIN_Q_SUPERIORITY, MIN_Q_SUPERIORITY_FOR_1_NODE / Math.Sqrt(Math.Abs(otherNode.N)));
          if (qDifference > minQSuperiority
           && otherNode.Q < bestQ)
          {
            bestQ = otherNode.Q;
            bestIndex = i;
          }

        }
      }
      // If some child to right has a value better than the known value at startIndex, swap them.
      if (bestIndex != startIndex)
      {
        SwapChildEdgeHeaders(startIndex, bestIndex);
        
        // Recursively try to do more
        CheckMoveOrderRearrangeAtIndex(in position, startIndex + 1, numLookForward - 1, minPolicyFraction);
      }
    }
  }


  /// <summary>
  /// Swap the edge headers of two child edges.
  /// </summary>
  /// <param name="child0Index"></param>
  /// <param name="child1Index"></param>
  public void SwapChildEdgeHeaders(int child0Index, int child1Index)
  {
    // Swapping involving already expanded edges not allowed
    // because data structures may have already taken a dependency on the child indices.
    Debug.Assert(child0Index >= NumEdgesExpanded);
    Debug.Assert(child1Index >= NumEdgesExpanded);

    Span<GEdgeHeaderStruct> edgeHeaders = EdgeHeadersSpan;
    (edgeHeaders[child0Index], edgeHeaders[child1Index]) = (edgeHeaders[child1Index], edgeHeaders[child0Index]);
  }

}

