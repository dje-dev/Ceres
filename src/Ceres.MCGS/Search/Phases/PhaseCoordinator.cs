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

#endregion

namespace Ceres.MCGS.Search.Phases;

/// <summary>
/// Coordinates phase synchronization (select vs backup) for two iterators.
/// Includes mutual exclusion and ordering conditions:
///   - mutual exclusion: Select and Backup phases must not overlap
///   - ordering: Backup(batch k) allowed to start only after Evaluate(batch j) finishes for all j &lt; k.
/// </summary>
internal sealed class PhaseCoordinator
{
  const bool DEBUG_OUTPUT = false;

  public const bool ENABLE_CONCURRENT_EVALUATE_WITH_BACKUP_SELECT = true; 

  public const bool LAST_EVALUATE_MUST_FINISH_BEFORE_NEXT_BACKUP_STARTS = false;

  /// <summary>
  /// Lock object for mutual exclusion between Select and Backup phases.
  /// </summary>
  private readonly Lock selectOrBackupExclusionLock = new();

  
  public volatile int NumActive = 0;

  /// <summary>
  /// Tracks the highest consecutive batch index for which evaluation has completed.
  /// Batches must complete evaluation in sequential order to advance this counter.
  /// Used to enforce ordering: Backup(k) waits for Evaluate(k-1) to complete.
  /// </summary>
  private long lastSequentialCompletedEvaluateBatch = -1;

  /// <summary>
  /// Tracks a single batch that completed out of order (if any).
  /// Since we have only 2 iterators, at most 1 batch can be out of order at a time.
  /// -1 means no out-of-order batch is pending.
  /// </summary>
  private long outOfOrderCompletedBatch = -1;

  /// <summary>
  /// Lock for updating and waiting on evaluate completion tracking.
  /// </summary>
  private readonly object evaluateCompletionLock = new();


  [Conditional("DEBUG")]
  void DebugWriteLn(string str)
  {
    if (DEBUG_OUTPUT)
    {
      Console.WriteLine(str);
    } 
  }


  private void Enter()
  {
    selectOrBackupExclusionLock.Enter();
    Interlocked.Increment(ref NumActive);
  }

  private void Exit()
  {
    Interlocked.Decrement(ref NumActive);
    selectOrBackupExclusionLock.Exit();
  }

  /// <summary>
  /// Marks the beginning of the selection phase for the specified batch index.
  /// Enforces mutual exclusion with Backup phase.
  /// </summary>
  /// <param name="iteratorID"></param>
  /// <param name="batchIndex"></param>
  public void EnterSelect(int iteratorID, int batchIndex)
  {
    Enter();
    DebugWriteLn("EnteredSelect iterator:" + iteratorID + "  batch:" + batchIndex);
  }



  /// <summary>
  /// Marks the beginning of the evaluation phase for the specified batch index.
  /// Tracks active evaluations to enforce ordering with Backup phase.
  /// </summary>
  public void EnterEvaluate(int iteratorID, int batchIndex)
  {
    if (!ENABLE_CONCURRENT_EVALUATE_WITH_BACKUP_SELECT)
    {
      Enter();
    }
    DebugWriteLn("EnteredEvaluate iterator:" + iteratorID + "  batch:" + batchIndex);
  }


  /// <summary>
  /// Marks the end of the evaluation phase for the specified batch index.
  /// Updates tracking to allow subsequent Backup phases to proceed.
  /// </summary>
  public void ExitEvaluate(int iteratorID, int batchIndex)
  {
    if (!ENABLE_CONCURRENT_EVALUATE_WITH_BACKUP_SELECT)
    {
      DebugWriteLn("ExitingEvaluate iterator:" + iteratorID + "  batch:" + batchIndex);
      Exit();
    }
    else if (LAST_EVALUATE_MUST_FINISH_BEFORE_NEXT_BACKUP_STARTS)
    {
      // Update the completion tracker and notify waiting backups
      lock (evaluateCompletionLock)
      {
        // Check if this is the next expected sequential batch
        if (batchIndex == lastSequentialCompletedEvaluateBatch + 1)
        {
          // Advance the sequential counter
          lastSequentialCompletedEvaluateBatch = batchIndex;
          
          // Check if the out-of-order batch can now be processed
          if (outOfOrderCompletedBatch == lastSequentialCompletedEvaluateBatch + 1)
          {
            lastSequentialCompletedEvaluateBatch = outOfOrderCompletedBatch;
            outOfOrderCompletedBatch = -1; // Clear it
          }
          
          DebugWriteLn($"ExitingEvaluate iterator:{iteratorID} batch:{batchIndex} (updated lastSequential:{lastSequentialCompletedEvaluateBatch})");
          
          // Wake up any threads waiting for evaluations to complete
          Monitor.PulseAll(evaluateCompletionLock);
        }
        else if (batchIndex > lastSequentialCompletedEvaluateBatch + 1)
        {
          // This batch completed out of order - track it
          // With 2 iterators, we should only have at most 1 out-of-order batch at a time
          outOfOrderCompletedBatch = batchIndex;
          DebugWriteLn($"ExitingEvaluate iterator:{iteratorID} batch:{batchIndex} (out of order, stored as pending, lastSequential:{lastSequentialCompletedEvaluateBatch})");
        }
        else
        {
          // This batch is older than what we've already processed - shouldn't happen
          DebugWriteLn($"ExitingEvaluate iterator:{iteratorID} batch:{batchIndex} (WARNING: already processed, lastSequential:{lastSequentialCompletedEvaluateBatch})");
        }
      }
    }
  }


  /// <summary>
  /// Marks the end of the selection phase for the specified batch index and releases the associated lock.
  /// </summary>
  /// <param name="batchIndex"></param>
  public void ExitSelect(int iteratorID, int batchIndex)
  {
    DebugWriteLn("ExitingSelect iterator:" + iteratorID + "  batch:" + batchIndex);
    Exit();
  }



  /// <summary>
  /// Marks the beginning of the backup phase for the specified batch index.
  /// Enforces two restrictions:
  /// 1. Mutual exclusion with Select phase (via Enter())
  /// 2. Waits for all Evaluate(j) where j &lt; batchIndex to complete
  /// </summary>
  public void EnterBackup(int iteratorID, int batchIndex)
  {
    if (LAST_EVALUATE_MUST_FINISH_BEFORE_NEXT_BACKUP_STARTS)
    {
      // Wait for all evaluations of batches j < batchIndex to complete sequentially
      lock (evaluateCompletionLock)
      {
        while (lastSequentialCompletedEvaluateBatch < batchIndex - 1)
        {
          DebugWriteLn($"WaitingForEvaluate iterator:{iteratorID} batch:{batchIndex} (waiting for batch {batchIndex - 1} to complete, current: {lastSequentialCompletedEvaluateBatch})");
          Monitor.Wait(evaluateCompletionLock);
        }
      }
    }

    // Enter mutual exclusion with Select
    Enter();
    DebugWriteLn("EnteredBackup iterator:" + iteratorID + "  batch:" + batchIndex); 
  }


  /// <summary>
  /// Signals that the backup operation for the specified batch index
  /// has completed and releases the associated lock.
  /// </summary>
  public void ExitBackup(int iteratorID, int batchIndex)
  {
    DebugWriteLn("ExitingBackup iterator:" + iteratorID + "  batch:" + batchIndex);
    Exit();
  }
}
