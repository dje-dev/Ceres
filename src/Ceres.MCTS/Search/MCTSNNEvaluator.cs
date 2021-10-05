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
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Evaluators;
using Ceres.Base.DataTypes;
using Ceres.MCTS.Iteration;

#endregion

namespace Ceres.MCTS.Search
{
  public class MCTSNNEvaluator
  {
    public readonly LeafEvaluatorNN Evaluator;

    LeafEvaluatorNN.EvalResultTarget resultTarget;

    public ListBounded<MCTSNode> Evaluate(MCTSIterator context, ListBounded<MCTSNode> nodes)
    {
      nodes[0].Context.NumNNBatches++;
      nodes[0].Context.NumNNNodes+= nodes.Count;

      
      if (resultTarget == LeafEvaluatorNN.EvalResultTarget.PrimaryEvalResult)
      {
        Debug.Assert(nodes[0].EvalResult.IsNull); // null evaluator indicates should have been sent here
      }
      else if (resultTarget == LeafEvaluatorNN.EvalResultTarget.SecondaryEvalResult)
      {
        //Debug.Assert(nodes[0].EvalResultSecondary.IsNull); // null evaluator indicates should have been sent here
      }

      Evaluator.BatchGenerate(context, nodes.AsSpan, resultTarget);

      return nodes;
    }


    public MCTSNNEvaluator(LeafEvaluatorNN evaluator, bool targetPrimary)
    {
      Evaluator = evaluator;
      resultTarget = targetPrimary ? LeafEvaluatorNN.EvalResultTarget.PrimaryEvalResult : LeafEvaluatorNN.EvalResultTarget.SecondaryEvalResult;

    }
  }
}

