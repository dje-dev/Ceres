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

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using System;
using System.Collections.Generic;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// NNEvaluator used for diagnostics purposes which compares
  /// the output of two or more evaluators for (near) equality
  /// (for example, when computed by different backend implementations).
  /// </summary>
  public class NNEvaluatorCompare : NNEvaluatorLinearCombo
  {
    public static bool VERBOSE = false;

    public enum CompareMode
    {
      /// <summary>
      /// No supplemental statistics.
      /// </summary>
      None, 

      /// <summary>
      /// Basic supplemental statistics.
      /// </summary>
      Full,

      /// <summary>
      /// Supplemental statistics with comparison against a third reference evaluator.
      /// </summary>
      FullWithReferenceEvaluator
    }

    /// <summary>
    /// Type of statistics or reference checking conducted.
    /// </summary>
    public readonly CompareMode Mode;

    /// <summary>
    /// The reference evaluator to use (in FullWithReferenceMode).
    /// </summary>
    public readonly NNEvaluator ReferenceEvaluator;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="stats"></param>
    /// <param name="referenceEvaluator">evaluator used only in FullWithReferenceEvaluator returning what are believed most accurate values</param>
    /// <param name="evaluators"></param>
    public NNEvaluatorCompare(CompareMode stats, NNEvaluator referenceEvaluator, params NNEvaluator[] evaluators)
      : base(evaluators, null)
    {
      if (evaluators.Length < 2)
      {
        throw new Exception("At least two evaluators must be provided with NNEvaluatorCompare.");
      }

      if (stats == CompareMode.FullWithReferenceEvaluator && referenceEvaluator == null)
      {
        throw new Exception("Reference evaluator must be non-null in FullWithReferenceEvaluator mode.");
      }

      Mode = stats;
      ReferenceEvaluator = referenceEvaluator;
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    public NNEvaluatorCompare(params NNEvaluator[] evaluators)
      : this(CompareMode.None, null, evaluators)
    {
    }


    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => MinBatchSizeAmongAllEvaluators;


    int countPolicyErr = 0;
    int countValueErr = 0;
    float sumPolicy0Err = 0;
    float sumPolicy1Err = 0;
    float sumValue0Err = 0;
    float sumValue1Err = 0;

    /// <summary>
    /// Evaluates specified batch into internal buffers.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      IPositionEvaluationBatch result = base.DoEvaluateIntoBuffers(positions, retrieveSupplementalResults);

      static string ErrMagnitudes(float eval0, float eval1, float evalRef, 
                                  ref float sumErr0, ref float sumErr1, ref int count)
      {
        float err0 = MathF.Abs(evalRef - eval0);
        float err1 = MathF.Abs(evalRef - eval1);

        count++;
        sumErr0 += err0;
        sumErr1 += err1;

        return $" err0:{err0, 7:F4} err1: {err1, 7:F4}  avg0: {sumErr0/count, 7:F4}  avg1: {sumErr1 / count,7:F4}";
      }

      int numVOK = 0;
      int numPolicyOK = 0;
      float maxPolicyDiff = 0;
      for (int i = 0; i < positions.NumPos; i++)
      {
        float v0 = subResults[0].GetWinP(i) - subResults[0].GetLossP(i);
        float v1 = subResults[1].GetWinP(i) - subResults[1].GetLossP(i);

        // Check W/D/L
        if (float.IsNaN(v0) || float.IsNaN(v1) || MathF.Abs(v0 - v1) > 0.02)
        {
          string refString = "";
          if (Mode == CompareMode.FullWithReferenceEvaluator)
          {
            NNEvaluatorResult[] refResult = ReferenceEvaluator.EvaluateBatch(positions.GetSubBatchSlice(i, 1));
            refString = ErrMagnitudes(v0, v1, refResult[0].V, ref sumValue0Err, ref sumValue1Err, ref countValueErr);
          }
          Console.WriteLine($"NNEvaluatorCompare V discrepancy at POS: {i,6:F0}  {v0,7:F3} {v1,7:F3} {refString}");
        }
        else
        {
          numVOK++;
        }
        
        (Memory<CompressedPolicyVector> policiesArray0, int policyIndex0) = subResults[0].GetPolicy(i);
        CompressedPolicyVector thesePolicies0 = policiesArray0.Span[policyIndex0];
        (Memory<CompressedPolicyVector> policiesArray1, int policyIndex1) = subResults[1].GetPolicy(i);
        CompressedPolicyVector thesePolicies1 = policiesArray1.Span[policyIndex1];

        float[] policies0 = thesePolicies0.DecodedAndNormalized;
        float[] policies1 = thesePolicies1.DecodedAndNormalized;
        float maxDiff = 0;
        
        for (int p=0;p<policies0.Length;p++)
        {
          float diff = MathF.Abs(policies0[p] - policies1[p]);
          float tolerance = Math.Max(0.03f, 0.07f * MathF.Abs(policies0[p] + policies1[p] * 0.5f));
          if ( float.IsNaN(policies0[p]) 
            || float.IsNaN(policies1[p]) 
            || (diff > maxDiff && (diff > tolerance)))
          {
            string refString = "";
            if (Mode == CompareMode.FullWithReferenceEvaluator)
            {
              NNEvaluatorResult[] refResult = ReferenceEvaluator.EvaluateBatch(positions.GetSubBatchSlice(i, 1));
              refString = ErrMagnitudes(policies0[p], policies1[p], refResult[0].Policy.DecodedAndNormalized[p],
                                        ref sumPolicy0Err, ref sumPolicy1Err, ref countPolicyErr);
            }

            if (maxDiff == 0) Console.WriteLine("NNEvaluatorCompare policy discrepancies ");
            maxDiff = policies0[p] - policies1[p];
            Console.WriteLine($"  POS: {i,6:F0}   {p,6} {policies0[p], 6:F3} { policies1[p], 6:F3} {refString}");
          }
        }

        if (maxDiff == 0)
          numPolicyOK++;
        else if (maxDiff > maxPolicyDiff)
          maxPolicyDiff = maxDiff;
      }

      if (VERBOSE)
      {
        Console.WriteLine();
        Console.WriteLine($"{numVOK} of {positions.NumPos} had approximately equal W/D/L scores between the first two WFEvalNetCompare");
        Console.WriteLine($"{numPolicyOK} of {positions.NumPos} had all policies good, worse significant difference {maxPolicyDiff}");
      }

      return result;
    }


    protected override void DoShutdown()
    {
      base.DoShutdown();
    }

  }
}
