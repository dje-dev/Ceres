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
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// An evalutor which for each batch can dynamically choose which 
  /// of several executors to use for a given batch.
  /// </summary>
  public class NNEvaluatorDynamic : NNEvaluatorCompound
  {
    /// <summary>
    /// The supplied Func which specifies which evaluator
    /// should be used for a given batch.
    /// </summary>
    public readonly Func<IEncodedPositionBatchFlat, int> DynamicEvaluatorIndexPredicate;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="dynamicEvaluatorIndexPredicate"></param>
    public NNEvaluatorDynamic(NNEvaluator[] evaluators, 
                            Func<IEncodedPositionBatchFlat, int> dynamicEvaluatorIndexPredicate = null) 
      : base(evaluators)
    {
      DynamicEvaluatorIndexPredicate = dynamicEvaluatorIndexPredicate;
    }


    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => MinBatchSizeAmongAllEvaluators;


    /// <summary>
    /// Virtual method that evaluates batch into internal buffers.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, 
                                                                   bool retrieveSupplementalResults = false)
    {
      int index;
      if (DynamicEvaluatorIndexPredicate != null)
        index = DynamicEvaluatorIndexPredicate(positions);
      else
        index = positions.PreferredEvaluatorIndex;

      return Evaluators[index].EvaluateIntoBuffers(positions, retrieveSupplementalResults);
    }

  }
}
