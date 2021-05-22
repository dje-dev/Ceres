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
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// An evalutor which for each batch can dynamically choose which 
  /// of several executors to use for a given position.
  /// </summary>
  public class NNEvaluatorDynamicByPos : NNEvaluatorCompound
  {
    /// <summary>
    /// A supplied function which is called for each position
    /// and passed the evaluations from each sub-evaluator,
    /// and returns index of the preferred evaluator for that position.
    public delegate int DynamicEvaluatorIndexPredicate(NNPositionEvaluationBatchMember[] batchResults);

    /// </summary>
    public readonly DynamicEvaluatorIndexPredicate DynamicEvaluatorChooser;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="dynamicEvaluatorIndexPredicate"></param>
    public NNEvaluatorDynamicByPos(NNEvaluator[] evaluators,
                                   DynamicEvaluatorIndexPredicate dynamicEvaluatorIndexPredicate)
      : base(evaluators)
    {
      if (dynamicEvaluatorIndexPredicate == null)
      {
        throw new ArgumentNullException(nameof(dynamicEvaluatorIndexPredicate));
      }

      DynamicEvaluatorChooser = dynamicEvaluatorIndexPredicate;
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
      if (retrieveSupplementalResults)
      {
        throw new NotImplementedException();
      }

      // Run all of the evaluators and get resulting batch results.
      // TODO: consider making this parallel
      IPositionEvaluationBatch[] batches = new IPositionEvaluationBatch[this.Evaluators.Length];
      for (int i = 0; i < batches.Length; i++)
      {
        batches[i] = Evaluators[i].EvaluateIntoBuffers(positions, retrieveSupplementalResults);
      }

      int numPos = positions.NumPos;

      // Create new arrays of the chosen values.
      CompressedPolicyVector[] policies = new CompressedPolicyVector[numPos];
      FP16[] w = new FP16[numPos];
      FP16[] l = new FP16[numPos];
      FP16[] m = HasM ? new FP16[numPos] : null;


      // For each position call the supplied delegate to choose the preferred evaluator.
      NNPositionEvaluationBatchMember[] values = new NNPositionEvaluationBatchMember[Evaluators.Length];
      for (int posNum = 0; posNum < numPos; posNum++)
      {
        // Construct the array of results from each evaluator to pass to the supplied delegate.
        for (int evaluatorNum = 0; evaluatorNum < Evaluators.Length; evaluatorNum++)
        {
          values[evaluatorNum] = new NNPositionEvaluationBatchMember(batches[evaluatorNum], posNum);
        }

        // Decide which evaluator to use.
        int index = DynamicEvaluatorChooser(values);
        if (index < 0 || index > Evaluators.Length)
        {
          throw new Exception($"Returned index {index} out of range");
        }

        // Extract out the evaluation results from the preferred evaluator.

        bool chosenEvaluatorIsWDL = Evaluators[index].IsWDL;
        if (!IsWDL && chosenEvaluatorIsWDL)
        {
          // Need to downgrade representation from the WDL evaluator
          // to make it expressed in same way as would be by an non-WDL evaluator.
          w[posNum] = batches[index].GetWinP(posNum) - batches[index].GetLossP(posNum);
          l[posNum] = 0;
        }
        else
        {
          w[posNum] = batches[index].GetWinP(posNum);
          l[posNum] = batches[index].GetLossP(posNum);
        }

        if (HasM)
        {
          m[posNum] = batches[index].GetM(posNum);
        }
        var policyInfo = batches[index].GetPolicy(posNum);
        policies[posNum] = policyInfo.policies.Span[policyInfo.index];
      }

      // Construct an output batch, choosing desired evaluator for each position
      PositionEvaluationBatch batch = new(IsWDL, HasM, positions.NumPos, policies, w, l, m, null, default, false);

      return batch;
    }

  }
}


