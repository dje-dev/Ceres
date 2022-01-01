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
using System.Diagnostics;
using System.Threading.Tasks;

using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNBackends.CUDA;
using Ceres.Chess.NNEvaluators.CUDA;

#endregion

namespace Ceres.Chess.NNEvaluators
{

  /// <summary>
  /// An NNEvaluator with a specified maximum batch size,
  /// that can split up batches into multiple sequential subbatches 
  /// sent to a specified underlying evaluator,
  /// optionally overlapping processing on GPU with preparation of inputs and outputs
  /// to improve throughput.
  /// </summary>
  public class NNEvaluatorSubBatchedWithTarget : NNEvaluator
  {
    /// <summary>
    /// Underlying evaluator.
    /// </summary>
    public readonly NNEvaluator Evaluator;

    /// <summary>
    /// Maximum batch size sent to device at one time,
    /// ideally a value which is a local 
    /// nodes per second maximum for the device.
    /// </summary>
    public int MaxSubBatchSize;

    /// <summary>
    /// Optimal batch size sent to device at one time,
    /// ideally a value which is a local 
    /// nodes per second maximum for the device.
    /// </summary>
    public int OptimalSubBatchSize;

    /// <summary>
    /// If asynchronous processing should be attempted to improve performance
    /// (overlap GPU processing and pre/post processing steps).
    /// </summary>
    const bool TRY_RUN_EVALUATOR_ASYNC = true;
    const bool ASYNC_PREPARE_INPUTS = true;

    bool runEvaluatorAsync;

    List<(int batchSize, int[] partitionBatchSizes)> optimalBatchSizePartitions;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <param name="maxSubBatchSize"></param>
    public NNEvaluatorSubBatchedWithTarget(NNEvaluator evaluator, int maxSubBatchSize, int optimalSubBatchSize,
                                           List<(int batchSize, int[] partitionBatchSizes)> optimalBatchSizePartitions)
    {
//      Console.WriteLine($"NNEvaluatorSubBatchedWithTarget {maxSubBatchSize}");

      Debug.Assert(evaluator is not NNEvaluatorSubBatchedWithTarget);

      Evaluator = evaluator;

      // Make sure neither batch size exceeds capabilities of underlying evaluator.
      MaxSubBatchSize = Math.Min(maxSubBatchSize, Evaluator.MaxBatchSize);
      OptimalSubBatchSize = optimalSubBatchSize;

      this.optimalBatchSizePartitions = optimalBatchSizePartitions;
      runEvaluatorAsync = TRY_RUN_EVALUATOR_ASYNC && Evaluator is NNEvaluatorCUDA;

      if (runEvaluatorAsync)
      {
        // Optimize for the very common case of batch size exactly maximumNPSSubBatchSize.
        NNEvaluatorCUDA evalCUDA = Evaluator as NNEvaluatorCUDA;
        evalCUDA.Evaluator.SetCommonBatchSize(maxSubBatchSize);
      }
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



    void BuildSizes(int numPos, List<int> subBatchSizes)
    {
      if (this.optimalBatchSizePartitions != null)
      {
        BuildSizesWithPredefinedSizes(numPos, subBatchSizes);
        return;
      }

      int half = MaxSubBatchSize / 2;

      int left = numPos;
      while (left > 0)
      {
        if (left > MaxSubBatchSize)
        {
          if (left < OptimalSubBatchSize * 3 && OptimalSubBatchSize <= MaxSubBatchSize / 2)
          {
            subBatchSizes.Add(OptimalSubBatchSize);
            left -= OptimalSubBatchSize;
          }
          else
          {
            subBatchSizes.Add(MaxSubBatchSize);
            left -= MaxSubBatchSize;
          }
        }
        else if (left > (MaxSubBatchSize * 80) / 100)
        {
          // Make sure the 3 batch sizes are somewhat equal in size.
          subBatchSizes.Add(left/2);
          left -= left/2;
        }
        else
        {
          subBatchSizes.Add(left);
          left = 0;
        }
      }



      
    }

    void BuildSizesWithPredefinedSizes(int numPos, List<int> subBatchSizes)
    {
      List<int> predefined = CheckPredefinedPartition(numPos);
      if (predefined != null)
      {
        subBatchSizes.AddRange(predefined);
        return;
      }

      // Put the smallest batch first so that batch at end is of full size,
      // which is more efficient because the last batch can stay in situ
      // in the underlying Evaluator buffers.
      int remainder = numPos % MaxSubBatchSize;
      if (remainder > 0)
      {
        subBatchSizes.Add(remainder);
        numPos -= remainder;
      }

      while (numPos > 0)
      {
        List<int> pre = CheckPredefinedPartition(numPos);
        if (pre != null)
        {
          subBatchSizes.AddRange(pre);
          return;
        }

        subBatchSizes.Add(MaxSubBatchSize);
        numPos -= MaxSubBatchSize;
      }
    }


    List<PositionEvaluationBatch> subBatches;

    /// <summary>
    /// Returns (creating if necessary) a PositionEvaluationBatch to be used
    /// to hold the evaluation results from the subbatch with specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    PositionEvaluationBatch GetOutputBatchBuffered(int index, bool retrieveSupplementalResults)
    {
      if (subBatches == null)
      {
         subBatches = new List<PositionEvaluationBatch>();
      }

      if (index < subBatches.Count)
      {
        return subBatches[index];
      }
      else if (index == subBatches.Count)
      {
        PositionEvaluationBatch batch = new PositionEvaluationBatch(IsWDL, HasM, MaxSubBatchSize, retrieveSupplementalResults);
        subBatches.Add(batch);
        return batch;
      }
      else
      {
        throw new Exception("GetBatch expected batches fetched in order.");
      }
    }

    IEncodedPositionBatchFlat priorPositions = null;
    short[] priorMoveIndices = null;
    short[] priorNumMoves = null;

    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      priorPositions = null;

      if (positions.NumPos < (MaxSubBatchSize * 80) / 100)
      {
        // Batch size fits into one subbatch, just dispatch to underlying evaluator.
        (Evaluator as NNEvaluatorCUDA).Evaluator.AsyncMode = false;
        return Evaluator.EvaluateIntoBuffers(positions, retrieveSupplementalResults);
      }
      else
      {
        // Determine an efficient partitioning.
        // of the positions into batches
        List<int> subBatchSizes = new();
        BuildSizes(positions.NumPos, subBatchSizes);

//        Dump(positions.NumPos, subBatchSizes);

        Task evaluateTask = null;

        NNEvaluatorCUDA evaluatorCUDA = Evaluator as NNEvaluatorCUDA;

        // Evaluate all the sub-batches.
        IPositionEvaluationBatch[] results = new IPositionEvaluationBatch[subBatchSizes.Count];
        int numDone = 0;
        for (int i = 0; i < results.Length; i++)
        {
          int thisBatchSize = subBatchSizes[i];

          IEncodedPositionBatchFlat thesePositions = positions.GetSubBatchSlice(numDone, thisBatchSize);

          if (runEvaluatorAsync)
          {
            (Evaluator as NNEvaluatorCUDA).Evaluator.AsyncMode = true;

            // Launch the evaluator to start execution on device.
            // Note that if not the first set of positions, then they have already been prepared
            // on the device and we pass null to indicate this.
            IEncodedPositionBatchFlat positionsToLaunchWith = (ASYNC_PREPARE_INPUTS && i > 0) ? null : thesePositions;
            int numPositions = positionsToLaunchWith != null ? positionsToLaunchWith.NumPos : subBatchSizes[i];
            evaluateTask = evaluatorCUDA.LaunchEvaluateBatchAsync(positionsToLaunchWith, numPositions, retrieveSupplementalResults);

            // Process and copy results from last (if any) into a result stored here.
            if (priorPositions != null)
            {
              IPositionEvaluationBatch iSourceBuffer = evaluatorCUDA.GetLastAsyncBatchResult(priorPositions, priorNumMoves, priorMoveIndices, retrieveSupplementalResults, false);
              CopyResultsIntoLocalBuffer(retrieveSupplementalResults, results, i - 1, iSourceBuffer);
              // results[i - 1] = evaluatorCUDA.GetLastAsyncBatchResult(priorPositions, priorNumMoves, priorMoveIndices, retrieveSupplementalResults, true);
            }

            // Prepare next set of positions, if any.
            bool processNextInputsAsync = ASYNC_PREPARE_INPUTS && i < results.Length - 1;
            if (processNextInputsAsync)
            {
              // Wait until evaluator is done using the input from the current batch being processed.
              (Evaluator as NNEvaluatorCUDA).Evaluator.InputsCopyToDeviceFinished.Wait();

              SaveCopyOfPriorBatchInputs(evaluatorCUDA, thisBatchSize, thesePositions);

              // Get next subbatch and prepare.
              IEncodedPositionBatchFlat nextPositions = positions.GetSubBatchSlice(numDone + thisBatchSize, subBatchSizes[i + 1]);
              (Evaluator as NNEvaluatorCUDA).PrepareInputPositions(nextPositions);
            }

            // Now signal to the Evaluator that it's ok to overwrite results.
            evaluatorCUDA.Evaluator.SetOkToFillResults();
            evaluateTask?.Wait();

            if (!processNextInputsAsync)
            {
              SaveCopyOfPriorBatchInputs(evaluatorCUDA, thisBatchSize, thesePositions);
            }
          }
          else
          {
            // Extract the batch result into a copy so not overwritten by Evaluator.
            IPositionEvaluationBatch nnBatch = Evaluator.EvaluateIntoBuffers(thesePositions, retrieveSupplementalResults);
            if (i == results.Length - 1)
            {
              // No need to make copy since this is last buffer to be evaluated
              // and we can leave the results in situ in the evaluator buffers.
              results[i] = nnBatch;
            }
            else
            {
              CopyResultsIntoLocalBuffer(retrieveSupplementalResults, results, i, nnBatch);
            }
#if NOT
            PositionEvaluationBatch nnBatchD = nnBatch as PositionEvaluationBatch;
            if (nnBatchD == null)
            {
              throw new NotImplementedException("NNEvaluatorSubBatchedWithTarget requires Evaluator to return PositionEvaluationBatch");
            }
            results[i] = new PositionEvaluationBatch(nnBatchD.IsWDL, nnBatchD.HasM, nnBatchD.NumPos, nnBatchD.Policies,
                                                     nnBatchD.W, nnBatchD.L, nnBatchD.M, nnBatchD.Activations, nnBatchD.Stats, true);
#endif
          }
          numDone += thisBatchSize;
        }

        if (runEvaluatorAsync && priorPositions != null)
        {
          // Do not request copy here (last argument) since for last batch
          // we can just leave the result values in situ in the evaluator because
          // there is no subsequent subbatch to be processed which would overwrite them.
          results[^1] = evaluatorCUDA.GetLastAsyncBatchResult(priorPositions, priorNumMoves, priorMoveIndices, retrieveSupplementalResults, false);

          // Reset evaluator back to default synchronous mode.
          (Evaluator as NNEvaluatorCUDA).Evaluator.AsyncMode = false;
        }

        return new PositionsEvaluationBatchMerged(results, subBatchSizes.ToArray());
      }

    }

    private void SaveCopyOfPriorBatchInputs(NNEvaluatorCUDA evaluatorCUDA, int thisBatchSize, IEncodedPositionBatchFlat thesePositions)
    {
      // Save copies of variables pertaining to these input positions that
      // will be overwritten as part of subsequent evaluation
      priorNumMoves = evaluatorCUDA.Evaluator.inputOutput.InputNumMovesUsed.AsSpan().Slice(0, thisBatchSize).ToArray();
      priorMoveIndices = evaluatorCUDA.Evaluator.inputOutput.InputMoveIndices.AsSpan().Slice(0, thisBatchSize * NNBackendInputOutput.MAX_MOVES).ToArray();
      priorPositions = thesePositions;
    }

    private void CopyResultsIntoLocalBuffer(bool retrieveSupplementalResults, IPositionEvaluationBatch[] results, int i, IPositionEvaluationBatch iSourceBuffer)
    {
      PositionEvaluationBatch batchCopyBuffer = GetOutputBatchBuffered(i, retrieveSupplementalResults);
      if (iSourceBuffer is PositionEvaluationBatch sourceBuffer)
      {
        batchCopyBuffer.CopyFrom(sourceBuffer);
        results[i] = batchCopyBuffer;
      }
      else
      {
        throw new Exception("NNEvaluatorSubBatchedWithTarget underlying Evaluator required to return PositionEvaluationBatch");
      }
    }

    static List<int> BuildPartition(params int[] b)
    {
      List<int> ret = new List<int>();
      for (int i = 0; i < b.Length; i++)
      {
        ret.Add(b[i]);
      }
      return ret;
    }
    List<int> CheckPredefinedPartition(int bs)
    {
      if (optimalBatchSizePartitions != null)
      {
        foreach (var v in optimalBatchSizePartitions)
        {
          if (v.batchSize == bs)
          {
            return BuildPartition(v.partitionBatchSizes);
          }
        }
      }

      return null;
#if not
      if (bs == 160) return BuildPartition(64, 96);
      if (bs == 176) return BuildPartition(84, 92);
      if (bs == 192) return BuildPartition(96, 96);
      if (bs == 208) return BuildPartition(100, 108);
      if (bs == 224) return BuildPartition(96, 128);
      if (bs == 240) return BuildPartition(92, 148);
      if (bs == 256) return BuildPartition(108, 148);
      if (bs == 272) return BuildPartition(92, 180);
      if (bs == 288) return BuildPartition(96, 192);
      if (bs == 304) return BuildPartition(148, 156);
      if (bs == 320) return BuildPartition(128, 192);
      if (bs == 336) return BuildPartition(148, 188);
      if (bs == 352) return BuildPartition(160, 192);
      if (bs == 368) return BuildPartition(180, 188);
      if (bs == 384) return BuildPartition(192, 192);
      return null;
#endif
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
