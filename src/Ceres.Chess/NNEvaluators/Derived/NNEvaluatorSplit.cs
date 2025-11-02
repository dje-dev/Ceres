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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

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
    /// Minimum number of positions before splitting begins.
    /// </summary>
    public readonly int MinSplitSize;

    /// <summary>
    /// Allocator that manages the splitting of positions into batches across evaluators.
    /// </summary>
    public ItemsInBucketsAllocator EvaluatorAllocator;


    int indexPerferredEvalator;

    /// <summary>
    /// Work item for dedicated evaluator threads.
    /// </summary>
    private sealed class EvaluatorWorkItem
    {
      public IEncodedPositionBatchFlat SubBatch;
      public bool RetrieveSupplementalResults;
      public IPositionEvaluationBatch Result;
      public ManualResetEventSlim CompletionSignal;

      public void Reset()
      {
        SubBatch = null;
        Result = null;
        RetrieveSupplementalResults = false;
        CompletionSignal.Reset();
      }
    }

    /// <summary>
    /// Dedicated worker threads for each evaluator
    /// (do not rely upon thread pool to minimize latency).
    /// </summary>
    private Thread[] workerThreads;

    /// <summary>
    /// Work queues for each evaluator thread.
    /// </summary>
    private BlockingCollection<EvaluatorWorkItem>[] workQueues;

    /// <summary>
    /// Pre-allocated work item pools.
    /// </summary>
    private EvaluatorWorkItem[] workItemPool;

    /// <summary>
    /// Cancellation token source for shutdown.
    /// </summary>
    private CancellationTokenSource shutdownTokenSource;

    /// <summary>
    /// Pre-allocated arrays to avoid allocations during evaluation.
    /// </summary>
    private IPositionEvaluationBatch[] resultsBuffer;
    private int[] subBatchSizesBuffer;
    private int[] evaluatorIndicesBuffer;


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


    const int DEFAULT_MIN_SPLIT_SIZE = 18;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="preferredFractions"></param>
    /// <param name="minSplitSize"></param>
    public NNEvaluatorSplit(NNEvaluator[] evaluators,
                           float[] preferredFractions = null,
                           int minSplitSize = DEFAULT_MIN_SPLIT_SIZE) // TODO: make this smarter (based on NPS)
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

      // Initialize dedicated worker threads
      InitializeWorkerThreads();

      // Pre-allocate fixed-length buffers to avoid ArrayPool allocations during evaluation
      resultsBuffer = new IPositionEvaluationBatch[Evaluators.Length];
      subBatchSizesBuffer = new int[Evaluators.Length];
      evaluatorIndicesBuffer = new int[Evaluators.Length];
    }


    /// <summary>
    /// Initializes dedicated worker threads for each evaluator.
    /// </summary>
    private void InitializeWorkerThreads()
    {
      shutdownTokenSource = new CancellationTokenSource();
      workerThreads = new Thread[Evaluators.Length];
      workQueues = new BlockingCollection<EvaluatorWorkItem>[Evaluators.Length];

      // Pre-allocate one work item per evaluator (reused across calls)
      workItemPool = new EvaluatorWorkItem[Evaluators.Length];

      for (int i = 0; i < Evaluators.Length; i++)
      {
        int evaluatorIndex = i;
        workQueues[i] = new BlockingCollection<EvaluatorWorkItem>(boundedCapacity: 1);

        // Pre-allocate and reuse work item
        workItemPool[i] = new EvaluatorWorkItem
        {
          CompletionSignal = new ManualResetEventSlim(false)
        };

        workerThreads[i] = new Thread(() => EvaluatorWorkerThread(evaluatorIndex))
        {
          IsBackground = true,
          Name = $"NNEvaluatorSplit-Worker-{evaluatorIndex}",
          Priority = ThreadPriority.AboveNormal
        };
        workerThreads[i].Start();
      }
    }


    /// <summary>
    /// Worker thread that processes evaluation requests for a specific evaluator.
    /// </summary>
    /// <param name="evaluatorIndex"></param>
    private void EvaluatorWorkerThread(int evaluatorIndex)
    {
      try
      {
        foreach (EvaluatorWorkItem workItem in workQueues[evaluatorIndex].GetConsumingEnumerable(shutdownTokenSource.Token))
        {
          try
          {
            workItem.Result = Evaluators[evaluatorIndex].EvaluateIntoBuffers(workItem.SubBatch, workItem.RetrieveSupplementalResults);
          }
          catch (Exception ex)
          {
            // Store exception in result or handle appropriately
            Console.Error.WriteLine($"Error in evaluator {evaluatorIndex}: {ex}");
            throw;
          }
          finally
          {
            workItem.CompletionSignal.Set();
          }
        }
      }
      catch (OperationCanceledException)
      {
        // Normal shutdown
      }
    }


    /// <summary>
    /// Implementation of virtual method to actually evaluate the batch.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (retrieveSupplementalResults)
      {
        throw new NotImplementedException();
      }

      // Determine how many evaluators to actually use for this batch.
      // Never use more than Round(positions.NumPos / MinSplitSize) evaluators.
      int maxAllowedBySize = Math.Max(1, (int)Math.Round((float)positions.NumPos / MinSplitSize, 0)); // safeguard
      int evaluatorsToUse = Math.Min(Evaluators.Length, maxAllowedBySize);

#if NOT
      // If after capping we only use one, just evaluate with the preferred (fastest) evaluator directly.
      if (evaluatorsToUse == 1)
      {
        return Evaluators[indexPerferredEvalator].EvaluateIntoBuffers(positions, retrieveSupplementalResults);
}
#endif

      // Compute the allocation of positions across evaluators.
      int[] allocations = EvaluatorAllocator.Allocate(evaluatorsToUse, positions.NumPos);

      IPositionEvaluationBatch[] results = resultsBuffer;
      int[] subBatchSizes = subBatchSizesBuffer;
      int[] evaluatorIndices = evaluatorIndicesBuffer;

      // Dispatch the sub-batches across the evaluators.
      int positionsAllocated = 0;
      int evaluatorsAllocated = 0;
      for (int i = 0; i < Evaluators.Length; i++)
      {
        int numAllocated = allocations[i];
        if (numAllocated > 0)
        {
          IEncodedPositionBatchFlat thisSubBatch = positions.GetSubBatchSlice(positionsAllocated, numAllocated);

          // Reuse pre-allocated work item instead of creating new one
          EvaluatorWorkItem workItem = workItemPool[i];
          workItem.Reset();
          workItem.SubBatch = thisSubBatch;
          workItem.RetrieveSupplementalResults = retrieveSupplementalResults;

          subBatchSizes[evaluatorsAllocated] = numAllocated;
          evaluatorIndices[evaluatorsAllocated] = i;

          // Submit work to dedicated thread
          workQueues[i].Add(workItem);

          evaluatorsAllocated++;
          positionsAllocated += numAllocated;
        }
      }

      // Wait for all work items to complete.
      for (int i = 0; i < evaluatorsAllocated; i++)
      {
        int evaluatorIndex = evaluatorIndicesBuffer[i];
        EvaluatorWorkItem workItem = workItemPool[evaluatorIndex];
        workItem.CompletionSignal.Wait();
        resultsBuffer[i] = workItem.Result;
      }

      EvaluatorAllocator.Deallocate(allocations);

      /// More efficient mode which is just a merged view over multiple batches 
      /// (without copying data)
      const bool USE_MERGED_BATCH_VIEW = true;

      if (USE_MERGED_BATCH_VIEW)
      {
        return new PositionsEvaluationBatchMerged(resultsBuffer, subBatchSizesBuffer);
      }
      else
      {
        bool isWDL = resultsBuffer[0].IsWDL;
        bool hasM = resultsBuffer[0].HasM;
        bool hasUncertaintyV = resultsBuffer[0].HasUncertaintyV;
        bool hasUncertaintyP = resultsBuffer[0].HasUncertaintyP;
        bool hasValueSecondary = resultsBuffer[0].HasValueSecondary;
        bool hasAction = resultsBuffer[0].HasAction;
        bool hasState = resultsBuffer[0].HasState;


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
          PositionEvaluationBatch resultI = (PositionEvaluationBatch)resultsBuffer[i];
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
                                           positions.NumPos, policies, actions, w, l, w2, l2, m, uncertaintyV, uncertaintyP, states, null, stats);
      }
    }


    /// <summary>
    /// Shutdown worker threads.
    /// </summary>
    protected override void DoShutdown()
    {
      if (shutdownTokenSource != null)
      {
        shutdownTokenSource.Cancel();

        // Complete all work queues to unblock worker threads
        foreach (BlockingCollection<EvaluatorWorkItem> queue in workQueues)
        {
          queue.CompleteAdding();
        }

        // Wait for all threads to finish
        foreach (Thread thread in workerThreads)
        {
          if (thread != null && thread.IsAlive)
          {
            thread.Join(millisecondsTimeout: 10);
          }
        }

        // Dispose resources
        foreach (BlockingCollection<EvaluatorWorkItem> queue in workQueues)
        {
          queue.Dispose();
        }

        // Dispose work items (specifically the ManualResetEventSlim)
        foreach (EvaluatorWorkItem workItem in workItemPool)
        {
          workItem.CompletionSignal.Dispose();
        }

        shutdownTokenSource.Dispose();
      }

      base.DoShutdown();
    }
  }
}
