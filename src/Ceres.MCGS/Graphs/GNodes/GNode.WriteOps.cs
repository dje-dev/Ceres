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

using Ceres.Base.DataTypes;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;
public readonly partial struct GNode : IComparable<GNode>, IEquatable<GNode>
{
  /// <summary>
  /// Forcibly resets state to be that of a drawn position.
  /// </summary>
  internal void ResetToDraw()
  {
    throw new Exception("Needs remediation. For example, consult MCGSParamsFixed.ENABLE_DRAW_KNOWN_TO_EXIST and also propagate to edge");
#if NOT
    ref GNodeStruct nodeRef = ref NodeRef;

    nodeRef.WinP = FP16.Zero;
    nodeRef.LossP = FP16.Zero;
    nodeRef.Terminal = GameResult.Draw;
    nodeRef.M = 1;

    nodeRef.UncertaintyValue = FP16.Zero;
    nodeRef.UncertaintyPolicy = FP16.Zero;

    nodeRef.Q = 0;
    nodeRef.DSum = 0;
    
    if (!IsGraphRoot)
    {
      foreach (GNode parent in Parents)
      {
        parent.NodeRef.DrawKnownToExistAmongChildren = true;

        GEdge edge = parent.ChildFromGPosition(this);

        edge.N = nodeRef.N;
        edge.Q = 0;

#if ACTION_ENABLED
        edge.ActionV = FP16.Zero;
        edge.ActionU = FP16.Zero;
#endif
        edge.UncertaintyV = FP16.Zero;
        edge.UncertaintyP = FP16.Zero;
      }
    }
#endif
  }


  /// <summary>
  /// Processes a node which has been determined to be a proven loss.
  /// Propagates this upward to the parent since parent's best move 
  /// is now obviously this one.
  /// </summary>
  /// <param name="lossP"></param>
  /// <param name="m"></param>
  internal void UpdateNodeForProvenChildLoss(FP16 lossP, FP16 m)
  {
    // This checkmate will obviously be chosen by the opponent
    // Therefore propagate the result up to the opponent as a victory,
    // overriding such that the Q for that node reflects the certain loss
    NodeRef.WinP = lossP;
    NodeRef.LossP = 0;
    NodeRef.Q = lossP;
    NodeRef.SiblingsQFrac = 0;
    NodeRef.D = 0;
    NodeRef.M = (byte)MathF.Round(m + 1, 0);
    NodeRef.CheckmateKnownToExistAmongChildren = true;
    NodeRef.UncertaintyPolicy = 0;
    NodeRef.UncertaintyValue = 0;

#if NOT
    // TODO: Does it make sense to restore the below? 
    //       But maybe only for the edge not currently being updated?
    //       Make sure any update to edge is propagated up to node immediately so update not lost.
    // VisitTo leading to parent should also be updated.
    if (!IsGraphRoot)
    {
      foreach (GEdge parentEdge in ParentEdges)
      {
        parentEdge.Q = lossP;

        MCGSSelectBackupStrategyBase strategy;
        if (parentEdge.ParentNode.Graph.TestFlag)
        {
          strategy.BackupToNode(parentEdge.ParentNode, 0, parentEdge.N * priorWContrib, priorDContrib, siblingBlendingInUse);
        }
#if ACTION_ENABLED
throw new NotImplementedException("If we restore this method, review code below. Also review sign of Q and lossP for correctness!");
        visitTo.ActionV = lossP;
        visitTo.ActionU = FP16.Zero;
#endif

        parentEdge.UncertaintyV = FP16.Zero;
        parentEdge.UncertaintyP = FP16.Zero;
      }
    }
#endif
  }
}
