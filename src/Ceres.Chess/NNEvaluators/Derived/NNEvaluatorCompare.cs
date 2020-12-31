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

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    public NNEvaluatorCompare(params NNEvaluator[] evaluators)
      : base(evaluators, null)
    {
    }


    /// <summary>
    /// Evaluates specified batch into internal buffers.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public override IPositionEvaluationBatch EvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      IPositionEvaluationBatch result = base.EvaluateIntoBuffers(positions, retrieveSupplementalResults);

      int numVOK = 0;
      int numPolicyOK = 0;
      float maxPolicyDiff = 0;
      for (int i = 0; i < positions.NumPos; i++)
      {
        float v0 = subResults[0].GetWinP(i) - subResults[0].GetLossP(i);
        float v1 = subResults[1].GetWinP(i) - subResults[1].GetLossP(i);

        // Check W/D/L
        if (MathF.Abs(v0 - v1) > 0.02)
          Console.WriteLine($"WFEvalNetCompare V discrepancy: {i,6:F0} {v0,7:F3} {v1,7:F3}");
        else
          numVOK++;
        
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
          if (diff > maxDiff && (diff > tolerance))
          {
            if (maxDiff == 0) Console.WriteLine("WFEvalNetCompare policy discrepancies:");
            maxDiff = policies0[p] - policies1[p];
            Console.WriteLine($"  {p,6} {policies0[p], 6:F3} { policies1[p], 6:F3}");
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
