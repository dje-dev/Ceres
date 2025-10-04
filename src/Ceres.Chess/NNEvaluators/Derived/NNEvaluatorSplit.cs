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
using System.Buffers;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataType;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using Chess.Ceres.NNEvaluators;

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
    /// Minimum number of positions before splitting begins.
    /// </summary>
    public readonly int MinSplitSize;

    /// <summary>
    /// Allocator that manages the splitting of positions into batches across evaluators.
    /// </summary>
    public ItemsInBucketsAllocator EvaluatorAllocator;


    int indexPerferredEvalator;


    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize
    {
      get
      {
        int maxBatchSizeMostRestrictiveEvaluator = int.MaxValue;
        for (int i = 0; i < Evaluators.Length; i++)
        {
          int maxThisEvaluator = Evaluators[i].MaxBatchSize;

          if (maxThisEvaluator < maxBatchSizeMostRestrictiveEvaluator)
          {
            maxBatchSizeMostRestrictiveEvaluator = maxThisEvaluator;
          }
        }

        bool allFractionsSame = PreferredFractions.All(f => f == PreferredFractions.First());
        if (allFractionsSame)
        {
          // Safe to assume we can spread larger batches evenly across all evaluators
          return Evaluators.Length * maxBatchSizeMostRestrictiveEvaluator;
        }
        else
        {
          // TODO: This is too conservative. Someday use the PreferredFractions
          //       to make a less restrictive estimate on allowable batch size.
          return maxBatchSizeMostRestrictiveEvaluator;
        }
      }
    }


    /// <summary>
    /// Switches the evaluator allocator to a specified (potentially shared) one.
    /// </summary>
    /// <param name="allocator"></param>
    public void SetAllocator(ItemsInBucketsAllocator allocator)
    {
      Debug.Assert(allocator.BucketCount == allocator.BucketCount);

      EvaluatorAllocator = allocator;
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="preferredFractions"></param>
    /// <param name="minSplitSize"></param>
    public NNEvaluatorSplit(NNEvaluator[] evaluators,
                            float[] preferredFractions = null,
                            int minSplitSize = 32) // TODO: make this smarter (based on NPS)
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

      if (preferredFractions == null)
      {
        // Default assumption is equality
        int numEvaluators = evaluators.Length;
        preferredFractions = new float[numEvaluators];
        for (int i = 0; i < numEvaluators; i++)
        {
          preferredFractions[i] = 1.0f / numEvaluators;
        }
      }

      // By default use our own private allocator.
      EvaluatorAllocator = new ItemsInBucketsAllocator(preferredFractions);
      PreferredFractions = preferredFractions;

      indexPerferredEvalator = ArrayUtils.IndexOfElementWithMaxValue(PreferredFractions, PreferredFractions.Length);
    }


    /// <summary>
    /// Implementation of virtual method to actually evaluate the batch.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (retrieveSupplementalResults) throw new NotImplementedException();

      // Determine how many evaluators to actually use for this batch.
      // Never use more than positions.NumPos / 32 (integer division) evaluators.
      const int MIN_EVALUTOR_POSITIONS = 32;
      int maxAllowedBySize = Math.Max(1, (int)Math.Round((float)positions.NumPos / MIN_EVALUTOR_POSITIONS, 0)); // safeguard
      int evaluatorsToUse = Math.Min(Evaluators.Length, maxAllowedBySize);

      // If after capping we only use one, just evaluate with the preferred (fastest) evaluator directly.
      if (evaluatorsToUse == 1)
      {
        return Evaluators[indexPerferredEvalator].EvaluateIntoBuffers(positions, retrieveSupplementalResults);
      }

      // Compute the allocation of positions across evaluators.
      int[] allocations = EvaluatorAllocator.Allocate(evaluatorsToUse, positions.NumPos);

      // Allocate arrays sized only for the evaluators we will actually use.
      IPositionEvaluationBatch[] results = new IPositionEvaluationBatch[evaluatorsToUse];
      Task[] tasks = new Task[evaluatorsToUse];
      int[] subBatchSizes = new int[evaluatorsToUse];

      // TODO: make this cleaner, create a virtual method at NNEvaluator
      bool usePreallocatedFixedBuffer = (Evaluators[0] is not NNEvaluatorONNX)
                                      || ((NNEvaluatorONNX)Evaluators[0]).Type == NNBackends.ONNXRuntime.ONNXNetExecutor.NetTypeEnum.LC0;
      Half[] flatValuesBuffer = null;
      if (usePreallocatedFixedBuffer)
      {
        // Allocate a temporary buffer for the flat values and force evaluation using this buffer.
        int bufferLength = EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES * positions.NumPos * 64;
        flatValuesBuffer = ArrayPool<Half>.Shared.Rent(bufferLength);
        Memory<Half> forceInitialize = positions.ValuesFlatFromPlanes(flatValuesBuffer, nhwc: false, scale50MoveCounter: false);
      }

      // Dispatch the sub-batches across the evaluators.
      int totalAllocated = 0;
      for (int i = 0; i < evaluatorsToUse; i++)
      {
        int numAllocated = allocations[i];
        if (allocations[i] > 0)
        {
          int capI = i;
          IEncodedPositionBatchFlat thisSubBatch = positions.GetSubBatchSlice(totalAllocated, numAllocated);
          subBatchSizes[capI] = numAllocated;
          tasks[i] = Task.Run(() => results[capI] = Evaluators[capI].EvaluateIntoBuffers(thisSubBatch, retrieveSupplementalResults));
          totalAllocated += allocations[i];
        }
      }

      Task.WaitAll(tasks);

      EvaluatorAllocator.Deallocate(allocations);

      if (usePreallocatedFixedBuffer)
      {
        // Release flat values buffer.
        ArrayPool<Half>.Shared.Return(flatValuesBuffer);
      }

      /// More efficient mode which is just a merged view over multiple batches 
      /// (without copying data)
      const bool USE_MERGED_BATCH_VIEW = true;

      if (USE_MERGED_BATCH_VIEW)
      {
        return new PositionsEvaluationBatchMerged(results, subBatchSizes);
      }
      else
      {
        bool isWDL = results[0].IsWDL;
        bool hasM = results[0].HasM;
        bool hasUncertaintyV = results[0].HasUncertaintyV;
        bool hasUncertaintyP = results[0].HasUncertaintyP;
        bool hasValueSecondary = results[0].HasValueSecondary;
        bool hasAction = results[0].HasAction;
        bool hasState = results[0].HasState;


        CompressedPolicyVector[] policies = new CompressedPolicyVector[positions.NumPos];
        FP16[] w = new FP16[positions.NumPos];
        FP16[] l = new FP16[positions.NumPos];
        FP16[] w2 = HasValueSecondary ? new FP16[positions.NumPos] : null;
        FP16[] l2 = HasValueSecondary ? new FP16[positions.NumPos] : null;
        FP16[] m = hasM ? new FP16[positions.NumPos] : null;
        FP16[] uncertaintyV = hasUncertaintyV ? new FP16[positions.NumPos] : null;
        FP16[] uncertaintyP = hasUncertaintyP ? new FP16[positions.NumPos] : null;
        CompressedActionVector[] actions = HasAction ? new CompressedActionVector[positions.NumPos] : null;
        Half[][] states = HasState ? new Half[positions.NumPos][] : null;

        int nextPosIndex = 0;
        for (int i = 0; i < evaluatorsToUse; i++)
        {
          PositionEvaluationBatch resultI = (PositionEvaluationBatch)results[i];
          int thisNumPos = resultI.NumPos;

          resultI.Policies.CopyTo(new Memory<CompressedPolicyVector>(policies).Slice(nextPosIndex, thisNumPos));
          resultI.W.CopyTo(new Memory<FP16>(w).Slice(nextPosIndex, thisNumPos));

          if (hasAction)
          {
            resultI.Actions.CopyTo(new Memory<CompressedActionVector>(actions).Slice(nextPosIndex, thisNumPos));
          }

          if (hasState)
          {
            resultI.States.CopyTo(new Memory<Half[]>(states).Slice(nextPosIndex, thisNumPos));
          }

          if (isWDL)
          {
            resultI.L.CopyTo(new Memory<FP16>(l).Slice(nextPosIndex, thisNumPos));
          }

          if (hasValueSecondary)
          {
            resultI.W2.CopyTo(new Memory<FP16>(w2).Slice(nextPosIndex, thisNumPos));

            if (isWDL)
            {
              resultI.L2.CopyTo(new Memory<FP16>(l2).Slice(nextPosIndex, thisNumPos));
            }
          }

          if (hasM)
          {
            resultI.M.CopyTo(new Memory<FP16>(m).Slice(nextPosIndex, thisNumPos));
          }

          if (HasUncertaintyV)
          {
            resultI.UncertaintyV.CopyTo(new Memory<FP16>(uncertaintyV).Slice(nextPosIndex, thisNumPos));
          }

          if (HasUncertaintyP)
          {
            resultI.UncertaintyP.CopyTo(new Memory<FP16>(uncertaintyP).Slice(nextPosIndex, thisNumPos));
          }

          nextPosIndex += thisNumPos;
        }

        TimingStats stats = new TimingStats();
        return new PositionEvaluationBatch(isWDL, hasM, hasUncertaintyV, hasUncertaintyP, hasAction, hasValueSecondary, hasState,
                                           positions.NumPos,
                                           policies, actions, w, l, w2, l2, m, uncertaintyV, uncertaintyP, states, null, stats);
      }

    }

  }
}
