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

using Ceres.Base;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// NNEvaluator sublcass which returns random value and policy evaluations
  /// (use for diagnostic and benchmarking purposes).
  /// 
  /// The values generated are pseudorandom in the sense that
  ///   - they embed no chess knowledge, but
  ///   - they are deterministic, always yielding the same value and policy for a given policy
  /// </summary>
  public class NNEvaluatorRandom : NNEvaluator
  {
    public enum RandomType 
    { 
      /// <summary>
      /// The policy is relatively (diffuse distribution)
      /// </summary>
      WidePolicy, 

      /// <summary>
      /// The policy is relatively wide (most mass concentrated on a small number of moves)
      /// </summary>
      NarrowPolicy
    };


    readonly bool isWDL;
    public override bool IsWDL => isWDL;
    public override bool HasM => false;

    public readonly RandomType Type;

    public NNEvaluatorRandom(RandomType randomType, bool isWDL)
    {
      this.isWDL = isWDL;
      Type = randomType;
    }


    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => 4096;


    static int HashInRange<T>(Span<T> items, int startIndex, int length)
    {
      int hash = 0;
      for (int i = startIndex; i < startIndex + length; i++)
        hash += items[i].GetHashCode();
      return hash;
    }


    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      TimingStats timingStats = new TimingStats();
      using (new TimingBlock("EvalBatch", timingStats, TimingBlock.LoggingType.None))
      {
        CompressedPolicyVector[] policies = new CompressedPolicyVector[positions.NumPos];

        FP16[] w = new FP16[positions.NumPos];
        FP16[] l = IsWDL ? new FP16[positions.NumPos] : null;

        for (int i = 0; i < positions.NumPos; i++)
        {
          int hashPos = HashInRange(positions.PosPlaneBitmaps, i * EncodedPositionWithHistory.NUM_PLANES_TOTAL, EncodedPositionWithHistory.NUM_PLANES_TOTAL);
          if (hashPos < 0) hashPos *= -1;
          //hashPos = Math.Abs(hashPos) ^ 172854;

          // Generate value
          if (IsWDL)
          {
            GenerateRandValue(hashPos, ref w[i], ref l[i]);
          }
          else
          {
            FP16 dummyL = 0;
            GenerateRandValue(hashPos, ref w[i], ref dummyL);
          }

          // Initialize policies. Mark them as requests to be random
          // (the actual randomization will be done during search, when we have the set of legal moves handy)
          // TODO: if the batch also contains Positions already, we could do the assignment now
          CompressedPolicyVector.InitializeAsRandom(ref policies[i], Type == RandomType.WidePolicy);
        }

        return new PositionEvaluationBatch(IsWDL, HasM, positions.NumPos, policies, w, l, null, null, timingStats);
      }
    }


    /// <summary>
    /// Generates a pseudorandom value score in range [-1.0, 1.0]
    /// </summary>
    /// <param name="hashPos"></param>
    /// <param name="w">win probability</param>
    /// <param name="l">loss probability (only if WDL)</param>
    void GenerateRandValue(int hashPos, ref FP16 w, ref FP16 l)
    {
      const int DIV1 = 1826743;
      float q = Math.Abs(((float)(hashPos % DIV1)) / (float)DIV1);

      // Force the q to typically have values closer to 0 by squaring
      q *= q;

      if (IsWDL)
      {
        const int DIV2 = 782743;
        float hashPosInRange2 = Math.Abs(((float)(hashPos % DIV2)) / (float)DIV2);

        w = (FP16)q;
        float maxD = 1.0f - Math.Abs(q);
        float d = hashPosInRange2 * maxD;
        l = (FP16)(1.0f - w - d);
      }
      else
      {
        w = (FP16)q;
        l = 0;
      }

    }

    /// <summary>
    /// If this evaluator produces the same output as another specified evaluator.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <returns></returns>
    public override bool IsEquivalentTo(NNEvaluator evaluator)
    {
      return evaluator is NNEvaluatorRandom 
        && ((NNEvaluatorRandom)evaluator).Type == Type;
    }

    protected override void DoShutdown()
    {
    }


  }
}
