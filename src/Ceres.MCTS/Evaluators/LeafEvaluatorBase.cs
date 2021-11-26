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
using System.Threading;
using Ceres.Base.DataTypes;
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Abstract base class for objects that can evaluate nodes, i.e.
  /// compute value and policy outputs for a given encoded position.
  /// </summary>
  public abstract class LeafEvaluatorBase
  {
    public enum LeafEvaluatorMode
    {
      /// <summary>
      /// Normal mode where TryEvluator returns 
      /// complete result as a LeafEvluationResult if found.
      /// </summary>
      ReturnEvaluationResult,

      /// <summary>
      /// Alternate mode where default is always returned by TryEvaluate,
      /// but the MCTSNode.
      /// </summary>
      SetAuxilliaryEval
    }

    /// <summary>
    /// Determines how result of successful leaf evaluation are returned.
    /// </summary>
    public LeafEvaluatorMode Mode { internal set; get; } = LeafEvaluatorMode.ReturnEvaluationResult;

    /// <summary>
    /// Attempts to evaluate node immediately and returns if successful (else default).
    /// </summary>
    /// <param name="node"></param>
    protected abstract LeafEvaluationResult DoTryEvaluate(MCTSNode node);

    /// <summary>
    /// BatchPostprocess is called at the end of gathering a batch of leaf nodes.
    /// It is guaranteed that no other operations are concurrently active at this time.
    /// </summary>
    public virtual void BatchPostprocess()
    {
    }

    /// <summary>
    /// Attempts to evaluate node immediately and returns if successful (else default).
    /// </summary>
    /// <param name="node"></param>
    public  LeafEvaluationResult TryEvaluate(MCTSNode node)
    {
      LeafEvaluationResult result = DoTryEvaluate(node);
      if (result.IsNull)
      {
        return default;
      } 
      else if (Mode == LeafEvaluatorMode.ReturnEvaluationResult)
      {
        return result;
      }
      else
      {
        node.InfoRef.EvalResultAuxilliary = (FP16)result.V;
        return default;
      }
    }

  }

}
