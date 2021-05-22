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
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

using System.Collections.Concurrent;
using System.Collections.Generic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  // <summary>
  /// Batching NN evaluator which pools possibly mulitple batch requests 
  /// (from different threads) into a combo batch, dispatching to GPU only when either :
  ///   - the aggregate batch becomes large enough, or
  ///   - a certain quantum of time has elapsed since we started collecting for the batch
  ///   
  /// TODO: Someday possible have mechanism that allows big batches to be processed inline
  ///       which is more efficient and avoids the possibility of buffer overflow 
  ///       and obviates the reduced maximum batch size set AdjustForPooled.
  /// </summary>
  public class NNEvaluatorPooled : NNEvaluatorCompound
  {
    /// <summary>
    /// Threshold number of nodes at which level (or above) the batch will be dispatched.
    /// </summary>
    public readonly int BatchSizeThreshold;

    /// <summary>
    /// Threshold maximum waiting time (in milliseconds) before which any pending nodes will be dispatched.
    /// </summary>
    public readonly float BatchDelayMS;

    /// <summary>
    /// If the supplemental NN layers should be retrieved.
    /// </summary>
    public readonly bool RetrieveSupplementalResults = false;


    #region Internal data 

    /// <summary>
    /// Time at which the last batch was dispatched
    /// </summary>
    DateTime lastBatchTime = DateTime.Now;

    /// <summary>
    /// The dispatch task is continually checking if the conditions
    /// are met that we should dispatch a batch.
    /// </summary>
    Task dispatchTask;

    /// <summary>
    /// Each of possibly many evaluators has an associated task which continuously
    /// loops, trying to extract and dispatch batches from the work queue.
    /// </summary>
    List<Task> evaluatorTasks;

    #region Cancellation

    /// <summary>
    /// If cancellation is pending
    /// </summary>
    CancellationTokenSource cancelSource;

    /// <summary>
    /// The cancellation token.
    /// </summary>
    CancellationToken cancelToken;

    /// <summary>
    /// If the evaluator has been cancelled.
    /// </summary>
    bool haveCancelled = false;

    #endregion

    /// <summary>
    /// Queue of batches which are ready to be executed (work queue).
    /// </summary>
    BlockingCollection<NNEvaluatorPoolBatch> pendingPooledBatches;

    /// <summary>
    /// The current pooled batch which is being in process of pooling positions.
    /// </summary>
    NNEvaluatorPoolBatch currentPooledBatch;

    /// <summary>
    /// Object used for sychronization.
    /// </summary>
    readonly object lockObj = new object();

    #endregion

    const int DEFAULT_BATCH_THRESHOLD = 64;
    const int DEFAULT_DELAY_MS = 3;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="evaluators"></param>
    public NNEvaluatorPooled(NNEvaluator[] evaluators, 
                             int batchSizeThreshold = DEFAULT_BATCH_THRESHOLD, 
                             float batchDelayMilliseconds = DEFAULT_DELAY_MS, 
                             bool retrieveSupplementalResults = false) : base(evaluators)
    {
      int MAX_BATCH_SIZE_THRESHOLD = MaxBatchSize / 2;
      if (batchSizeThreshold > MAX_BATCH_SIZE_THRESHOLD)
      {
        // Don't allow the batch size threshold to be too close to
        // the absolute batch size threshold to avoid overflow in the final pooling.
        throw new ArgumentOutOfRangeException(nameof(batchSizeThreshold), $"NNEvaluatorPooled batchSizeThreshold is too large, maximum value: {MAX_BATCH_SIZE_THRESHOLD}");
      }

      // Verify the evaluators not the same object but are equivalent.
      for (int i=0;i<evaluators.Length;i++)
      {
        for (int j = 0; j < evaluators.Length; j++)
        {
          if (i != j && object.ReferenceEquals(evaluators[i], evaluators[j]))
            throw new ArgumentException("The evaluators passed to NNEvaluatorPooled must not be duplicated");
        }

        if (!evaluators[0].IsEquivalentTo(evaluators[i]))
          throw new Exception("Cannot combine evaluators with different output in same NNEvaluatorsMultiBatched");
      }

      BatchSizeThreshold = batchSizeThreshold;
      BatchDelayMS = batchDelayMilliseconds;
      RetrieveSupplementalResults = retrieveSupplementalResults;

      pendingPooledBatches = new BlockingCollection<NNEvaluatorPoolBatch>();
      cancelSource = new CancellationTokenSource();
      cancelToken = cancelSource.Token;

      // Create and launch task for each evaluator.
      int index = 0;
      evaluatorTasks = new List<Task>();
      foreach (NNEvaluator evaluator in evaluators)
      {
        int thisIndex = index;

        evaluatorTasks.Add(Task<int>.Run(() => EvaluatorThreadMethod(thisIndex)));
        index++;
      }

      // Create an initial empty batch.
      currentPooledBatch = new NNEvaluatorPoolBatch();

      // Finally, launch the dispatch task which watches when
      // pooled batches have filled sufficiently to be launched.
      dispatchTask = Task.Run(DispatchTaskProcessor, cancelToken);
    }


    /// <summary>
    /// Constructor for the special case where there is only one evaluator.
    /// </summary>
    /// <param name="evaluators"></param>
    /// <param name="batchSizeThreshold"></param>
    /// <param name="batchDelayMilliseconds"></param>
    /// <param name="retrieveSupplementalResults"></param>
    public NNEvaluatorPooled(NNEvaluator evaluators,
                             int batchSizeThreshold = DEFAULT_BATCH_THRESHOLD,
                             float batchDelayMilliseconds = DEFAULT_DELAY_MS,
                             bool retrieveSupplementalResults = false) 
      : this (new NNEvaluator[] { evaluators }, batchSizeThreshold, batchDelayMilliseconds, 
               retrieveSupplementalResults)
    {
    }

    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => MinBatchSizeAmongAllEvaluators;



    /// <summary>
    /// The method executed by each worker thread.
    /// </summary>
    /// <param name="index"></param>
    void EvaluatorThreadMethod(int index)
    {
      try
      {
        RunEvaluatorThreadLoop(index);
      }
      catch (Exception exc)
      {
        Console.WriteLine("Exception in pooled batch evaluation " + exc);
        Environment.Exit(3);
      }
    }


    /// <summary>
    /// The worker thread method that continuously lopps, 
    /// processing pooled batches as they arrive queue.
    /// Each thread uses just one of the evaluators exclusively.
    /// </summary>
    /// <param name="index"></param>
    void RunEvaluatorThreadLoop(int index)
    {
      // Retrieve the evaluator associated with this thread.
      NNEvaluator thisEvaluator = Evaluators[index];

      // Loop continuously until cancellation requested.
      while (!cancelToken.IsCancellationRequested)
      {
        // Try to get a pending batch that this thread can process.
        foreach (NNEvaluatorPoolBatch batchToProcess in pendingPooledBatches.GetConsumingEnumerable(cancelToken))
        { 
          // Process the batch.
          batchToProcess.ProcessPooledBatch(thisEvaluator, RetrieveSupplementalResults);

          // Release any threads that were waiting for this pooled batch.
          batchToProcess.batchesDoneEvent.Set();
        }
      }

    }


    ~NNEvaluatorPooled()
    {
      DoShutdown();
    }


    protected override void DoShutdown()
    {
      if (!haveCancelled)
      {
        base.Shutdown();
        cancelSource?.Cancel();
        haveCancelled = true;
      }
    }


    /// <summary>
    /// Implements virtual method to evaluate a specified batch.
    /// This may block for some time before executing,
    /// waiting for more additions to be made to the pooled batch.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (retrieveSupplementalResults != RetrieveSupplementalResults) 
        throw new Exception("Internal error: Requested unexpected retrieveSupplementalResults");


      // Launch if the current batch already exceeds threshold number of positions
      // to avoid overflow and also because there is little benefit to accumulate more.
      while (true)
      {
        lock (lockObj)
        {
          int currentPendingPositions = currentPooledBatch.NumPendingPositions;
          if (currentPendingPositions > DEFAULT_BATCH_THRESHOLD)
          {
            Launch();
          }
          else
          {
            break;
          }
        }
      }
      

      int batchIndex;
      NNEvaluatorPoolBatch poolBatch;
      lock (lockObj)
      {
        poolBatch = currentPooledBatch;// grab local copy of this since it may change upon next set of batches
        batchIndex = poolBatch.pendingBatches.Count;
        poolBatch.pendingBatches.Add(positions);

//        if (positions.Positions == null)
//          throw new Exception("missing **********************");
      }

      // Wait until we are signalled that this pooled batch has completed processing
      poolBatch.batchesDoneEvent.Wait();

      Debug.Assert(!float.IsNaN(poolBatch.completedBatches[batchIndex].GetV(0)));

      // Now that the batch has finished, return just the sub-batch that was requested in this call.
      return poolBatch.completedBatches[batchIndex];
    }


    #region Main thread method

    /// <summary>
    /// Launches the current batch.
    /// </summary>
    void Launch()
    {
      pendingPooledBatches.Add(currentPooledBatch);
      currentPooledBatch = new NNEvaluatorPoolBatch();
      lastBatchTime = DateTime.Now;
    }


    /// <summary>
    /// Launches the current set of batches if the launch-conditions are met.
    /// </summary>
    void LaunchIfShould()
    {
      lock (lockObj)
      {
        // Check if we should launch have accumulated enough positins (or waiited long enough)
        if (ShouldLaunch())
        {
          Launch();
        }
      }
    }


    public void DispatchTaskProcessor()
    {
      // Repeated until cancelled.
      while (!cancelToken.IsCancellationRequested)
      {
        LaunchIfShould();

        // Delay a short while before checking again.
        const int DELAY_MS = DEFAULT_DELAY_MS / 2;
        Thread.Sleep(DELAY_MS);
      }
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Determines if it is time to stop aggregating and
    /// actually evaluate the current aggregated bartches.
    /// </summary>
    /// <returns></returns>
    bool ShouldLaunch()
    {
      int numPending = currentPooledBatch.NumPendingPositions;

      bool batchLargeEnough = numPending >= BatchSizeThreshold;
      bool haveWaitedLongEnough = (DateTime.Now - lastBatchTime).TotalSeconds >= (0.001f * BatchDelayMS);

      if (false)
      {
        if (batchLargeEnough)
          Console.WriteLine($"launch multibatch because large enough {numPending} versus threshold {BatchSizeThreshold}");
        else if (haveWaitedLongEnough)
          Console.WriteLine($"launch multibatch because time up,  {(DateTime.Now - lastBatchTime).TotalSeconds} versus threshold {(0.001f * BatchDelayMS)}");
      }

      return batchLargeEnough || (haveWaitedLongEnough && numPending > 0);
    }
    



  #endregion

  }
}
