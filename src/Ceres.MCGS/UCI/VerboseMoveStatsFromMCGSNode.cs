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
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.UCI;

/// <summary>
/// Static helper methods which construct VerboseMoveStat 
/// from specified MCGS search root node.
/// </summary>
public static class VerboseMoveStatsFromMCGSNode
{
  public static List<VerboseMoveStat> BuildStats(MCGSManager searchManager, BestMoveInfoMCGS bestMoveInfo)
  {
    GNode searchRootNode = searchManager.Engine.SearchRootNode;

    // Create dummy parent stats.
    VerboseMoveStats statsParent = new(searchManager.RootMGPos.ToPosition, null, 0, 0, 0, null);
    List<VerboseMoveStat> stats = [];

    if (searchRootNode.NumEdgesExpanded == 0 && searchRootNode.EdgeHeadersSpan.Length > 0) //go nodes 1
    {
      for (int childIndex=searchRootNode.NumEdgesExpanded;childIndex<searchRootNode.NumPolicyMoves;childIndex++)
      {
        EncodedMove move = searchRootNode.ChildEdgeHeaderAtIndex(childIndex).Move;
        MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, searchRootNode.CalcPosition());

        if (mgMove == bestMoveInfo.BestMove)
        {
          VerboseMoveStat stat = BuildStatNoSearch(statsParent, searchRootNode, bestMoveInfo);
          VerboseMoveStat stat1 = BuildStatNotExpanded(statsParent, searchRootNode, childIndex);
          stat1.Q = stat.Q;
          stats.Add(stat1);
        }
        else
        {

          VerboseMoveStat stat = BuildStatNotExpanded(statsParent, searchRootNode, childIndex);
          stats.Add(stat);
        }

      }
      stats.Reverse();
    }
    else
    {
      // First process policy moves not yet expanded
      for (int childIndex = searchRootNode.NumEdgesExpanded; childIndex < searchRootNode.NumPolicyMoves; childIndex++)
      {
        EncodedMove move = searchRootNode.ChildEdgeHeaderAtIndex(childIndex).Move;
        MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, searchRootNode.CalcPosition());
        VerboseMoveStat stat = BuildStatNotExpanded(statsParent, searchRootNode, childIndex);
        stats.Add(stat);          
      }
      stats.Reverse(); // reverse to have the lowest probability first
      

      // Now process moves expanded in order of visit count.
      GEdge[] sortedN = searchRootNode.NumEdgesExpanded == 0 ? null : searchRootNode.EdgesSorted(node => node.N + 0.0001f * node.P);

      if (sortedN != null)
      {
        foreach (GEdge edge in sortedN)
        {
          if (edge.IsExpanded && edge != bestMoveInfo.BestMoveEdge)
          {
            stats.Add(BuildStatExpanded(statsParent, edge, edge.ChildNode, false));
          }
        }
      }

      // Save the best move for last.
      if (searchRootNode.N > 1) // otherwise unexpanded and will have been output above
      {
        if (bestMoveInfo.BestMoveEdge.IsNull)
        {
          stats.Add(BuildStatNoSearch(statsParent, searchRootNode, bestMoveInfo));
        }
        else
        {
          stats.Add(BuildStatExpanded(statsParent, bestMoveInfo.BestMoveEdge, bestMoveInfo.BestMoveEdge.ChildNode, false));
        }
      }
    }
    // Finally, output the search root node.
    stats.Add(BuildStatExpanded(statsParent, default, searchRootNode, true));
    return stats;
  }


  static VerboseMoveStat BuildStatNoSearch(VerboseMoveStats statsParent, GNode parent, BestMoveInfoMCGS info)
  {
    VerboseMoveStat stat = new(statsParent, null);
    stat.MoveString = info.BestMove.MoveStr(MGMoveNotationStyle.Coordinates);
    float v = parent.V;
    stat.WL = v;
    stat.Q = new EncodedEvalLogistic(v);
    stat.V = new EncodedEvalLogistic(v);
    stat.D = v == 0 ? 1 : 0;
    stat.UncertaintyValue = parent.UncertaintyValue;
    stat.UncertaintyPolicy = parent.UncertaintyPolicy;

    return stat;
  }


  static VerboseMoveStat BuildStatNotExpanded(VerboseMoveStats statsParent, GNode node, int childIndex)
  {
    GEdgeHeaderStruct edgeHeader = node.EdgeHeadersSpan[childIndex];
    VerboseMoveStat stat = new(statsParent, null);

    MGMove move = MGMoveConverter.ToMGMove(node.CalcPosition(), edgeHeader.Move);
    stat.MoveString = move.MoveStr(MGMoveNotationStyle.Coordinates);
    stat.MoveCode = edgeHeader.Move.IndexNeuralNet;
    stat.P = edgeHeader.P * 100.0f;
    stat.UncertaintyValue = float.NaN;
    stat.UncertaintyPolicy = float.NaN;
    return stat;
  }


  static VerboseMoveStat BuildStatExpanded(VerboseMoveStats parent, GEdge edge, GNode node, bool isSearchRoot)
  {
    VerboseMoveStat stat = new(parent, null);
    float multiplier = isSearchRoot ? 1 : -1;

    MGMove move = default;
    int nnIndex = -1;
    int n;
    if (!edge.IsNull)
    {
      MGPosition position = edge.ParentNode.CalcPosition();
      move = MGMoveConverter.ToMGMove(in position, edge.Move);
      nnIndex = edge.Move.IndexNeuralNet;
      n = edge.N;
    }
    else
    {
      n = node.N;
    }

    stat.MoveString = isSearchRoot ? "node" : move.MoveStr(MGMoveNotationStyle.Coordinates);
    stat.MoveCode = nnIndex;// isSearchRoot ? 20 : node.Parent.ChildAtIndexInfo(node.IndexInParentsChildren).move.IndexNeuralNet;
    stat.VisitCount = n;
    stat.P = isSearchRoot ? 100 : edge.P * 100.0f;

    stat.D = node.NodeRef.DrawP;
    stat.WL = (node.IsSearchRoot && node.N == 0) ? node.V : (float)node.Q * multiplier;
    stat.Q = new EncodedEvalLogistic((float)node.Q * multiplier);
    stat.M = node.NodeRef.M;
    stat.V = new EncodedEvalLogistic((float)node.V * multiplier);
    stat.U = node.UncertaintyValue;
    stat.UncertaintyValue = node.UncertaintyValue;
    stat.UncertaintyPolicy = node.UncertaintyPolicy;

    return stat;
  }

}

