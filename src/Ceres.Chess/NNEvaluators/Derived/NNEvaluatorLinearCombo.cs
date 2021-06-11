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
using System.Linq;
using System.Threading.Tasks;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Subclass of NNEvaluatorCompound which implements a weighted
  /// average combination of output heads (possibly each with a different weight).
  /// </summary>
  public class NNEvaluatorLinearCombo :  NNEvaluatorCompound
  {
    public delegate float[] WeightsOverrideDelegate(in Position pos);

    public readonly float[] WeightsValue;
    public readonly float[] WeightsPolicy;
    public readonly float[] WeightsM;

    /// <summary>
    /// Optional delegate called for each position to determine weights to use for value head for that position.
    /// </summary>
    public readonly WeightsOverrideDelegate WeightsValueOverrideFunc;

    /// <summary>
    /// Optional delegate called for each position to determine weights to use for MLH head for that position.
    /// </summary>
    public readonly WeightsOverrideDelegate WeightsMOverrideFunc;

    /// <summary>
    /// Optional delegate called for each position to determine weights to use for policy head for that position.
    /// </summary>
    public readonly WeightsOverrideDelegate WeightsPolicyOverrideFunc;

    protected IPositionEvaluationBatch[] subResults;

    /// <summary>
    /// Constructor (for case where weights are the same across the output heads).
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="weights"></param>
    public NNEvaluatorLinearCombo(NNEvaluator[] evaluators, IList<float> weights = null) : base(evaluators)
    {
      float[] weightsArray;
      if (weights == null)
      {
        // Default equal weight
        weightsArray = MathUtils.Uniform(evaluators.Length);
      }
      else
      {
        if (weights.Count != evaluators.Length) throw new ArgumentException("Number of weights specified does not match number of evaluators");
        weightsArray = weights.ToArray();
      }

      WeightsValue = WeightsPolicy = WeightsM = weightsArray;
    }

    /// <summary>
    /// Constructor (for case where weights are different across the output heads).
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="weightsValue"></param>
    /// <param name="weightsPolicy"></param>
    /// <param name="weightsM"></param>
    /// <param name="weightsValueOverrideFunc"></param>
    /// <param name="weightsMOverrideFunc"></param>
    /// <param name="weightsPolicyOverrideFunc"></param>
    public NNEvaluatorLinearCombo(NNEvaluator[] evaluators, 
                                IList<float> weightsValue, 
                                IList<float> weightsPolicy, 
                                IList<float> weightsM,
                                WeightsOverrideDelegate weightsValueOverrideFunc = null,
                                WeightsOverrideDelegate weightsMOverrideFunc = null,
                                WeightsOverrideDelegate weightsPolicyOverrideFunc = null) : base(evaluators)
    {
      WeightsValue  = weightsValue  != null ? weightsValue.ToArray()  : MathUtils.Uniform(evaluators.Length);
      WeightsPolicy = weightsPolicy != null ? weightsPolicy.ToArray() : MathUtils.Uniform(evaluators.Length);
      WeightsM      = weightsM      != null ? weightsM.ToArray()      : MathUtils.Uniform(evaluators.Length);

      WeightsValueOverrideFunc = weightsValueOverrideFunc;
      WeightsMOverrideFunc = weightsMOverrideFunc;
      WeightsPolicyOverrideFunc = weightsPolicyOverrideFunc;
    }

    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => MinBatchSizeAmongAllEvaluators;



    object execLockObj = new object();


    /// <summary>
    /// Implementation of virtual method that actually evaluates a batch.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      lock (execLockObj)
      {
        subResults = new IPositionEvaluationBatch[Evaluators.Length];

        // Ask all constituent evaluators to evaluate this batch (in parallel)
        Parallel.For(0, Evaluators.Length,
          delegate (int i)
          {
            subResults[i] = (IPositionEvaluationBatch)Evaluators[i].EvaluateIntoBuffers(positions, retrieveSupplementalResults);
          });

        if (retrieveSupplementalResults) throw new NotImplementedException();
        float[] valueHeadConvFlat = null;

        // Extract the combined policies
        CompressedPolicyVector[] policies = ExtractComboPolicies(positions);

        // Compute average value result
        FP16[] w = null;
        FP16[] l = null;
        FP16[] m = null;

        // TODO: also compute and pass on the averaged Activations
        Memory<NNEvaluatorResultActivations> activations = new Memory<NNEvaluatorResultActivations>();

        w = WeightsValueOverrideFunc == null ? AverageFP16(positions.NumPos, subResults, (e, i) => e.GetWinP(i), WeightsValue)
                                             : AverageFP16(positions.NumPos, subResults, (e, i) => e.GetWinP(i), WeightsValueOverrideFunc, positions);

        if (IsWDL)
        {
          l = WeightsValueOverrideFunc == null ? AverageFP16(positions.NumPos, subResults, (e, i) => e.GetLossP(i), WeightsValue)
                                               : AverageFP16(positions.NumPos, subResults, (e, i) => e.GetLossP(i), WeightsValueOverrideFunc, positions);
        }

        if (HasM)
        { 
          m = WeightsValueOverrideFunc == null ? AverageFP16(positions.NumPos, subResults, (e, i) => e.GetM(i), WeightsM)
                                               : AverageFP16(positions.NumPos, subResults, (e, i) => e.GetM(i), WeightsMOverrideFunc, positions);
        }

        TimingStats stats = new TimingStats();
        return new PositionEvaluationBatch(IsWDL, HasM, positions.NumPos, policies, w, l, m, activations, stats);
      }
    }


    private CompressedPolicyVector[] ExtractComboPolicies(IEncodedPositionBatchFlat positions)
    {
      Span<float> policyAverages = stackalloc float[EncodedPolicyVector.POLICY_VECTOR_LENGTH];

      // Compute average policy result for all positions
      CompressedPolicyVector[] policies = new CompressedPolicyVector[positions.NumPos];
      for (int posIndex = 0; posIndex < positions.NumPos; posIndex++)
      {
        policyAverages.Clear();

        float[] weights = WeightsPolicyOverrideFunc == null ? WeightsPolicy 
                                                            : WeightsPolicyOverrideFunc(MGChessPositionConverter.PositionFromMGChessPosition(in positions.Positions[posIndex]));

        for (int evaluatorIndex = 0; evaluatorIndex < Evaluators.Length; evaluatorIndex++)
        {
          (Memory<CompressedPolicyVector> policiesArray, int policyIndex) = subResults[evaluatorIndex].GetPolicy(posIndex);
          CompressedPolicyVector thesePolicies = policiesArray.Span[policyIndex];
          foreach ((EncodedMove move, float probability) moveInfo in thesePolicies.ProbabilitySummary())
          {
            if (moveInfo.move.RawValue == CompressedPolicyVector.SPECIAL_VALUE_RANDOM_NARROW ||
                moveInfo.move.RawValue == CompressedPolicyVector.SPECIAL_VALUE_RANDOM_WIDE)
            {
              throw new NotImplementedException("Mixing NNEvaluatorLinearCombo and random evaluator probably not yet supported");
            }

            float thisContribution = weights[evaluatorIndex] * moveInfo.probability;
            policyAverages[moveInfo.move.IndexNeuralNet] += thisContribution;
          }
        }

        CompressedPolicyVector.Initialize(ref policies[posIndex], policyAverages, false);
      }

      return policies;
    }

    #region Utility methods


    // --------------------------------------------------------------------------------------------
    static FP16[] AverageFP16(int numPos, IPositionEvaluationBatch[] batches,
                              Func<IPositionEvaluationBatch, int, float> getValueFunc, float[] weights)
    {
      FP16[] ret = new FP16[numPos];
      for (int i = 0; i < numPos; i++)
      {
        for (int evaluatorIndex = 0; evaluatorIndex < batches.Length; evaluatorIndex++)
        {
          ret[i] += (FP16)(weights[evaluatorIndex] * getValueFunc(batches[evaluatorIndex], i));
        }
      }

      return ret;
    }

    // --------------------------------------------------------------------------------------------
    static FP16[] AverageFP16(int numPos, IPositionEvaluationBatch[] batches,
                              Func<IPositionEvaluationBatch, int, float> getValueFunc, 
                              WeightsOverrideDelegate weightFunc, IEncodedPositionBatchFlat positions)
    {
      FP16[] ret = new FP16[numPos];
      for (int i = 0; i < numPos; i++)
      {
        for (int evaluatorIndex = 0; evaluatorIndex < batches.Length; evaluatorIndex++)
        {
          float[] weight = weightFunc(MGChessPositionConverter.PositionFromMGChessPosition(in positions.Positions[i]));
          ret[i] += (FP16)(weight[evaluatorIndex] * getValueFunc(batches[evaluatorIndex], i));
        }
      }
      return ret;
    }

    #endregion
  }
}
