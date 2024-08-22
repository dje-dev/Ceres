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
using Ceres.Chess.MoveGen;
using Ceres.MCTS.MTCSNodes;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.LC0.Positions;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.Managers.Uncertainty;

#endregion

namespace Ceres.Features.UCI
{
  /// <summary>
  /// Static helper methods which construct LC0VerbosesMoveStats 
  /// from specified MCTS search root node.
  /// </summary>
  public static class VerboseMoveStatsFromMCTSNode
  {
    public static List<VerboseMoveStat> BuildStats(MCTSNode searchRootNode)
    {
      // Create dummy parent stats.
      VerboseMoveStats statsParent = new VerboseMoveStats(searchRootNode.Annotation.Pos, null, 0, 0, 0, null);

      List<VerboseMoveStat> stats = new List<VerboseMoveStat>();

      BestMoveInfo best = searchRootNode.BestMoveInfo(false);

      // First process policy moves not yet expanded
      // starting from last one (lowest probability).
      for (int i = searchRootNode.NumPolicyMoves - 1; i >= 0; i--)
      {
        (MCTSNode node, EncodedMove move, FP16 p) info = searchRootNode.ChildAtIndexInfo(i);
        if (info.node.IsNull)
        {
          VerboseMoveStat stat = BuildStatNotExpanded(statsParent, searchRootNode, i);
          stats.Add(stat);
        }
      }

      // Now process moves expanded in order of visit count.
      MCTSNode[] sortedN = searchRootNode.ChildrenSorted(s => (float)s.N + 0.0001f * s.P);
      foreach (MCTSNode node in sortedN)
      {
        if (node != best.BestMoveNode)
        {
          stats.Add(BuildStatExpanded(statsParent, node, false));
        }
      }

      // Save the best move for last.
      if (best.BestMoveNode.IsNull)
      {
        if (searchRootNode.N > 1) // otherwise unexpanded and will have been output above
        {
          stats.Add(BuildStatNoSearch(statsParent, searchRootNode, best));
        }
      }
      else
      {
        stats.Add(BuildStatExpanded(statsParent, best.BestMoveNode, false));
      }

      // Finally, output the search root node.
      stats.Add(BuildStatExpanded(statsParent, searchRootNode, true));

      return stats;
    }


    static VerboseMoveStat BuildStatNoSearch(VerboseMoveStats statsParent, MCTSNode parent, BestMoveInfo info)
    {
      VerboseMoveStat stat = new VerboseMoveStat(statsParent, null);
      stat.MoveString = info.BestMove.MoveStr(MGMoveNotationStyle.LC0Coordinate);
      float v = parent.V;
      stat.WL = v;
      stat.Q = new EncodedEvalLogistic(v);
      stat.V = new EncodedEvalLogistic(v);
      stat.D = v == 0 ? 1 : 0;
      return stat;
    }

    static VerboseMoveStat BuildStatNotExpanded(VerboseMoveStats statsParent, MCTSNode node, int childIndex)
    {
      MCTSNodeStructChild child = node.ChildAtIndexRef(childIndex);
      VerboseMoveStat stat = new VerboseMoveStat(statsParent, null);
      stat.MoveString = node.Annotation.Pos.MiscInfo.SideToMove == Chess.SideType.White ? child.Move.ToString() : child.Move.Flipped.ToString();
      stat.MoveCode = child.Move.IndexNeuralNet;
      stat.P = child.P * 100.0f;
      stat.U = node.ChildU(childIndex);
      return stat;
    }


    static VerboseMoveStat BuildStatExpanded(VerboseMoveStats parent, MCTSNode node, bool isSearchRoot)
    {
      VerboseMoveStat stat = new VerboseMoveStat(parent, null);
      float multiplier = isSearchRoot ? 1 : -1;

      node.Annotate();

      stat.MoveString = isSearchRoot ? "node" : node.Annotation.PriorMoveMG.MoveStr(MGMoveNotationStyle.LC0Coordinate);
      stat.MoveCode = isSearchRoot ? 20 : node.Parent.ChildAtIndexInfo(node.IndexInParentsChildren).move.IndexNeuralNet;
      stat.VisitCount = node.N;
      stat.P = isSearchRoot ? 100 : (node.P * 100.0f);
      stat.D = node.StructRef.DrawP;
      stat.WL = (node.IsRoot && node.N == 0) ? node.V : (float)node.Q * multiplier;
      stat.Q = new EncodedEvalLogistic((float)node.Q * multiplier);
      stat.M = node.MPosition;
      stat.V = new EncodedEvalLogistic((float)node.V * multiplier);
      stat.U = isSearchRoot ? 0 : node.Parent.ChildU(node.IndexInParentsChildren);
#if LEGACY_UNCERTAINTY
      stat.Uncertainty = MCTSNodeUncertaintyAccumulator.GetUncertainty(node.N, float.NaN);
#else
      stat.UncertaintyValue = node.UncertaintyVPosition;
#endif

      return stat;
    }

  }
}