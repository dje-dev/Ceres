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
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataType;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Sublcass of NNEvaluatorCompound which splits batches
  /// across multiple evaluators using a specified distribution.
  /// </summary>
  public class NNEvaluatorSplit : NNEvaluatorCompound
  {
    /// <summary>
    /// Target fractions (summing to 1.0) to allocate to each evaluator.
    /// The fractions are optimally proportional to evaluator performance.
    /// </summary>
    public readonly float[] PreferredFractions;

    /// <summary>
    /// More efficient mode which returns a WFEvaluationBatchMerged
    /// which is just a merged view over multiple batches (without copying data)
    /// </summary>
    public readonly bool UseMergedBatch;

    /// <summary>
    /// Minimum number of positions before splitting begins.
    /// </summary>
    public readonly int MinSplitSize;

    int indexPerferredEvalator;


    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize
    {
      get
      {
        int maxBatchSizeMostRestrictiveEvaluator = int.MaxValue;
        for (int i=0; i<Evaluators.Length; i++)
        {
          int maxThisEvaluator = Evaluators[i].MaxBatchSize;

          if (maxThisEvaluator < maxBatchSizeMostRestrictiveEvaluator)
          {
            maxBatchSizeMostRestrictiveEvaluator = maxThisEvaluator;
          }
        }
        
        return maxBatchSizeMostRestrictiveEvaluator;
      }
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="preferredFractions"></param>
    /// <param name="minSplitSize"></param>
    /// <param name="useMergedBatch"></param>
    public NNEvaluatorSplit(NNEvaluator[] evaluators, 
                            float[] preferredFractions = null, 
                            int minSplitSize = 48, // TODO: make this smarter (based on NPS)
                            bool useMergedBatch = true)
      : base(evaluators)
    {
      if (preferredFractions != null && preferredFractions.Length != evaluators.Length)
      {
        throw new ArgumentException($"Number of preferred fractions {preferredFractions.Length} does not match number of evaluators {evaluators.Length}");
      }

      if (preferredFractions == null)
      {
        preferredFractions = MathUtils.Uniform(evaluators.Length);
      }

      MinSplitSize = minSplitSize;
      UseMergedBatch = useMergedBatch;

      if (preferredFractions == null)
      {
        // Default assumption is equality
        int numEvaluators = evaluators.Length;
        PreferredFractions = new float[numEvaluators];
        for (int i = 0; i < numEvaluators; i++)
          PreferredFractions[i] = 1.0f / numEvaluators;
      }
      else
        PreferredFractions = preferredFractions;

      indexPerferredEvalator = ArrayUtils.IndexOfElementWithMaxValue(PreferredFractions, PreferredFractions.Length);
    }


    /// <summary>
    /// Implementation of virtual method to actually evaluate the batch.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (retrieveSupplementalResults) throw new NotImplementedException();

      if (positions.NumPos <= MinSplitSize)
      {
        // Too small to profitably split across multiple devices
        return Evaluators[indexPerferredEvalator].EvaluateIntoBuffers(positions, retrieveSupplementalResults);
      }
      else
      {
        // TODO: someday we could use the idea already used in LZTrainingPositionServerBatchSlice
        //       and construct custom WFEvaluationBatch which are just using approrpiate Memory slices
        //       Need to create a new constructor for WFEvaluationBatch
        IPositionEvaluationBatch[] results = new IPositionEvaluationBatch[Evaluators.Length];

        Task[] tasks = new Task[Evaluators.Length];
        int[] subBatchSizes = new int[Evaluators.Length];
        for (int i = 0; i < Evaluators.Length; i++)
        {
          int capI = i;
          IEncodedPositionBatchFlat thisSubBatch = GetSubBatch(positions, PreferredFractions, capI);
          subBatchSizes[capI] = thisSubBatch.NumPos;
          tasks[i] = Task.Run(() => results[capI] = Evaluators[capI].EvaluateIntoBuffers(thisSubBatch, retrieveSupplementalResults));
        }
        Task.WaitAll(tasks);

        if (UseMergedBatch)
        {
          return new PositionsEvaluationBatchMerged(results, subBatchSizes);
        }
        else
        {
          CompressedPolicyVector[] policies = new CompressedPolicyVector[positions.NumPos];
          FP16[] w = new FP16[positions.NumPos];
          FP16[] l = new FP16[positions.NumPos];
          FP16[] m = new FP16[positions.NumPos];

          bool isWDL = results[0].IsWDL;
          bool hasM = results[0].HasM;

          int nextPosIndex = 0;
          for (int i = 0; i < Evaluators.Length; i++)
          {
            PositionEvaluationBatch resultI = (PositionEvaluationBatch)results[i];
            int thisNumPos = resultI.NumPos;

            resultI.Policies.CopyTo(new Memory<CompressedPolicyVector>(policies).Slice(nextPosIndex, thisNumPos));
            resultI.W.CopyTo(new Memory<FP16>(w).Slice(nextPosIndex, thisNumPos));

            if (isWDL)
            {
              resultI.L.CopyTo(new Memory<FP16>(l).Slice(nextPosIndex, thisNumPos));
              resultI.M.CopyTo(new Memory<FP16>(m).Slice(nextPosIndex, thisNumPos));
            }

            nextPosIndex += thisNumPos;
          }

          TimingStats stats = new TimingStats();
          return new PositionEvaluationBatch(isWDL, hasM, positions.NumPos, policies, w, l, m, null, stats);
        }

      }


      static float[] ToCumulative(float[] values)
      {
        float[] ret = new float[values.Length];
        float acc = 0;
        for (int i = 0; i < ret.Length; i++)
        {
          ret[i] = acc;
          acc += values[i];
        }

        if (Math.Abs(acc - 1.0f) > 0.001) throw new Exception("Fractions should sum to 1.0");

        return ret;
      }


      IEncodedPositionBatchFlat GetSubBatch(IEncodedPositionBatchFlat fullBatch, float[] splitFracs, int thisSplitIndex)
      {
        float[] cums = ToCumulative(splitFracs);

        int StartIndex(int i) => (int)(fullBatch.NumPos * cums[i]);

        int start = StartIndex(thisSplitIndex);
        int end;

        bool isLastSplit = thisSplitIndex == splitFracs.Length - 1;
        if (isLastSplit)
          end = fullBatch.NumPos;
        else
          end = StartIndex(thisSplitIndex + 1);

        int length = end - start;

        return fullBatch.GetSubBatchSlice(start, length);
      }
    }

  }
}
