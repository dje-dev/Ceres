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

using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using System;
using System.Collections.Generic;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Sublcass of NNEvaluatorCompound which possibly splits batches
  /// across multiple sequential batches to same evaluator,
  /// targeting specific optimal batch size.
  /// </summary>
  public class NNEvaluatorSubBatchedWithTarget : NNEvaluator
  {
    /// <summary>
    /// Underlying evaluator.
    /// </summary>
    public readonly NNEvaluator Evaluator;

    /// <summary>
    /// Batch size at which maximum NPS is seen.
    /// </summary>
    public int MaximumNPSSubBatchSize;

    /// <summary>
    /// Batch size above which NPS suddenly degrades markedly (optional),
    /// or batch size which is not to be exceeded for other reasons 
    /// (e.g. to reduce GPU memory requirements).
    /// </summary>
    public int maxSubBatchSize;


    /// <summary>
    /// 
    /// </summary>
    /// <param name="evaluator"></param>
    /// <param name="maximumNPSSubBatchSize"></param>
    /// <param name="maximumSubBatchSize">Batch size above which NPS suddenly degrades markedly (optional),
    /// or batch size which is not to be exceeded for other reasons 
    /// (e.g. to reduce GPU memory requirements)</param>
    public NNEvaluatorSubBatchedWithTarget(NNEvaluator evaluator, int maximumNPSSubBatchSize, int? maximumSubBatchSize = null)
    {
      Console.WriteLine("NNEvaluatorSplitTargetBatchSize");
      Evaluator = evaluator;
      MaximumNPSSubBatchSize = maximumNPSSubBatchSize;
      maxSubBatchSize = Math.Max(maximumSubBatchSize ?? maximumNPSSubBatchSize, MaximumNPSSubBatchSize);

      // Make sure neither batch size exceeds capabilities of underlying evaluator.
      maxSubBatchSize = Math.Min(maxSubBatchSize, Evaluator.MaxBatchSize);
      MaximumNPSSubBatchSize = Math.Min(MaximumNPSSubBatchSize, Evaluator.MaxBatchSize);
    }

    public override bool IsWDL => Evaluator.IsWDL;

    public override bool HasM => Evaluator.HasM;

    public override int MaxBatchSize => int.MaxValue;

    public override bool PolicyReturnedSameOrderMoveList => false; // batching disturbs move list order
    public override InputTypes InputsRequired => Evaluator.InputsRequired;
    public override void CalcStatistics(bool computeBreaks, float maxSeconds = 1)
    {
      Evaluator.CalcStatistics(computeBreaks, maxSeconds);
    }

    public override float EstNPSBatch => Evaluator.EstNPSBatch;
    public override float EstNPSSingleton => Evaluator.EstNPSSingleton;
    public override bool IsEquivalentTo(NNEvaluator evaluator)
    {
      return Evaluator.IsEquivalentTo(evaluator);
    }


    static int NumBatchesAtSize(int numPos, int batchSize) 
      => (numPos % batchSize == 0) ? (numPos / batchSize) : 1 + (numPos / batchSize);
    
    void BuildSizes(int numPos, List<int> subBatchSizes)
    {
      if (NumBatchesAtSize(numPos, maxSubBatchSize)
        < NumBatchesAtSize(numPos, MaximumNPSSubBatchSize))
      {
        // At least one sub-batch must be more than MaximumNPSSubBatchSize,
        // so go ahead and send as oversized (MaximumNPSSubBatchSize)
        int numThisBatch = Math.Min(maxSubBatchSize, numPos);

        subBatchSizes.Add(numThisBatch);

        // Recursively continue reducing based on how many remain.
        BuildSizes(numPos - numThisBatch, subBatchSizes);
      }
      else
      {
        // Just use multiples of MaximumNPSSubBatchSize,
        // no benefit to using any larger.
        BuildBatchSizes(numPos, subBatchSizes, MaximumNPSSubBatchSize);
      }

    }

    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (positions.NumPos < (MaximumNPSSubBatchSize * 165) / 100)
      {
        // Too small to profitably split across multiple devices
        return Evaluator.EvaluateIntoBuffers(positions, retrieveSupplementalResults);
      }
      else
      {
        // Determine an efficient partitioning.
        // of the positions into batches
        List<int> subBatchSizes = new();
        BuildSizes(positions.NumPos, subBatchSizes);

        //Dump(positions.NumPos, subBatchSizes);

        // Evaluate all the sub-batches.
        IPositionEvaluationBatch[] results = new IPositionEvaluationBatch[subBatchSizes.Count];
        int numDone = 0;
        for (int i = 0; i < results.Length; i++)
        {
          int thisBatchSize = subBatchSizes[i];

          IEncodedPositionBatchFlat thisSubBatch = positions.GetSubBatchSlice(numDone, thisBatchSize);

          // Extract the batch result into a copy so not overwritten by Evaluator.
          IPositionEvaluationBatch nnBatch = Evaluator.EvaluateIntoBuffers(thisSubBatch, retrieveSupplementalResults);
          PositionEvaluationBatch nnBatchD = nnBatch as PositionEvaluationBatch;
          if (nnBatchD == null)
          {
            throw new NotImplementedException("NNEvaluatorSubBatchedWithTarget requires Evaluator to return PositionEvaluationBatch");
          }
          results[i] = new PositionEvaluationBatch(nnBatchD.IsWDL, nnBatchD.HasM, nnBatchD.NumPos, nnBatchD.Policies,
                                                   nnBatchD.W, nnBatchD.L, nnBatchD.M, nnBatchD.Activations, nnBatchD.Stats, true);
          numDone += thisBatchSize;
        }

        return new PositionsEvaluationBatchMerged(results, subBatchSizes.ToArray());
      }

    }

    private static void BuildBatchSizes(int numPos, List<int> subBatchSizes, int subBatchSize)
    {
      int numLeft = numPos;

      // If the last batch would be very small,
      // redistribute some from prior batch into this one to make more similar
      int ADJ = subBatchSize / 2;
      while (numLeft > 0)
      {
        if (numLeft < ADJ && subBatchSizes.Count > 0)
        {
          subBatchSizes[^1] -= ADJ;
          numLeft += ADJ;
        }

        int thisBatchSize = Math.Min(subBatchSize, numLeft);
        subBatchSizes.Add(thisBatchSize);
        numLeft -= thisBatchSize;
      }
    }

    private static void Dump(int numPos, List<int> subBatchSizes)
    {
      Console.Write(numPos + " --> ");
      foreach (var i in subBatchSizes)
      {
        Console.Write(i + " ");
      }
      Console.WriteLine();
    }

    protected override void DoShutdown() => Evaluator.Shutdown();
  }
}
