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
  /// If true, backups are forced to occur in batch-selection order ("no crossing"): the batch
  /// selected first (lowest batch index) backs up first. Batches advance a shared turn counter
  /// so that a later batch waits (before acquiring the select/backup exclusion lock) until all
  /// earlier batches have completed (or skipped) their backup. Batch indices are contiguous per
  /// search (Engine.nextBatchID, reset at each search start alongside ResetBackupOrder()).
  /// Set from ParamsSearchExecution.AllowOutOfOrderBatches (inverse) in the MCGSEngine constructor.
  /// </summary>
  public bool EnforceInOrderBackup = true;

  readonly object backupOrderLock = new();
  int nextBackupTurn;

  /// <summary>Resets the in-order-backup turn counter. Call at the start of each search.</summary>
  public void ResetBackupOrder()
  {
    lock (backupOrderLock)
    {
      nextBackupTurn = 0;
      Monitor.PulseAll(backupOrderLock);
    }
  }

  /// <summary>
  /// Blocks until it is batchIndex's turn to back up (i.e. all earlier batches have finished or
  /// skipped their backup). Must be called BEFORE acquiring the select/backup exclusion lock so a
  /// later batch never holds that lock while waiting on an earlier batch that needs it.
  /// </summary>
  public void EnterBackupOrder(int batchIndex)
  {
    if (!EnforceInOrderBackup)
    {
      return;
    }
    lock (backupOrderLock)
    {
      while (nextBackupTurn != batchIndex)
      {
        Monitor.Wait(backupOrderLock);
      }
    }
  }

  /// <summary>Advances the backup turn past batchIndex, releasing the next batch in order.</summary>
  public void ExitBackupOrder(int batchIndex)
  {
    if (!EnforceInOrderBackup)
    {
      return;
    }
    lock (backupOrderLock)
    {
      nextBackupTurn = batchIndex + 1;
      Monitor.PulseAll(backupOrderLock);
    }
  }


  // ---- Always-on instrumentation: fraction of backups that occur out of selection order ----
  // (a "crossing" = a batch backs up while an earlier-selected batch has not yet done so). This is
  // measured regardless of EnforceInOrderBackup: with it on the fraction is ~0 (by construction);
  // with it off this reveals how much crossing naturally occurs (i.e. what enforcement prevents).
  readonly object orderStatsLock = new();
  int statsNextInOrder;
  readonly System.Collections.Generic.HashSet<int> statsCompletedAhead = new();
  long statsTotalBackups;
  long statsOutOfOrderBackups;

  public void ResetBackupOrderStats()
  {
    lock (orderStatsLock)
    {
      statsNextInOrder = 0;
      statsCompletedAhead.Clear();
      statsTotalBackups = 0;
      statsOutOfOrderBackups = 0;
    }
  }

  /// <summary>
  /// Records that batchIndex has reached the backup stage (didBackup=true for a real backup,
  /// false for a 0-path batch that performs no backup but still occupies a slot in the order).
  /// Counts a real backup as out-of-order if any earlier-indexed batch has not yet reached here.
  /// </summary>
  public void RecordBackupOrder(int batchIndex, bool didBackup)
  {
    lock (orderStatsLock)
    {
      bool inOrder = batchIndex == statsNextInOrder;
      if (inOrder)
      {
        statsNextInOrder++;
        while (statsCompletedAhead.Remove(statsNextInOrder))
        {
          statsNextInOrder++;
        }
      }
      else
      {
        statsCompletedAhead.Add(batchIndex);
      }

      if (didBackup)
      {
        statsTotalBackups++;
        if (!inOrder)
        {
          statsOutOfOrderBackups++;
        }
      }
    }
  }

  public long BackupTotalCount
  {
    get
    {
      lock (orderStatsLock)
      {
        return statsTotalBackups;
      }
    }
  }

  public long BackupOutOfOrderCount
  {
    get
    {
      lock (orderStatsLock)
      {
        return statsOutOfOrderBackups;
      }
    }
  }

  public double BackupOutOfOrderFraction
  {
    get
    {
      lock (orderStatsLock)
      {
        return statsTotalBackups > 0 ? (double)statsOutOfOrderBackups / statsTotalBackups : 0;
      }
    }
  }

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


  // Per-search phase instrumentation (Stopwatch ticks), summed across both iterators and all
  // batches of THIS search. These are INSTANCE fields (one PhaseCoordinator per search) so that
  // concurrent searches running in the same process do not cross-increment or mutually reset each
  // other's counters. Each phase total includes time waiting on the select/backup exclusion lock
  // (exclWait). Also tracks the per-phase max and a duration histogram for the CPU phases
  // (select/backup) so the residual-idle question can distinguish mean cost vs tail (worst-case).
  internal long ExclWaitSelectTicks;
  internal long ExclWaitBackupTicks;
  internal long PhaseTicksSelect;
  internal long PhaseTicksEval;
  internal long PhaseTicksBackup;
  internal long PhaseBatchCount;
  internal long PhaseMaxSelectTicks;
  internal long PhaseMaxBackupTicks;
  static readonly double[] PHASE_BUCKET_MS = { 0.25, 0.5, 1, 2, 4, 8, 16, 32, double.PositiveInfinity };
  readonly long[] PhaseHistSelect = new long[PHASE_BUCKET_MS.Length];
  readonly long[] PhaseHistBackup = new long[PHASE_BUCKET_MS.Length];

  /// <summary>
  /// Zeroes all phase-timing counters. Call once at the true start of each search.
  /// </summary>
  internal void ResetPhaseTiming()
  {
    ExclWaitSelectTicks = 0;
    ExclWaitBackupTicks = 0;
    PhaseTicksSelect = 0;
    PhaseTicksEval = 0;
    PhaseTicksBackup = 0;
    PhaseBatchCount = 0;
    PhaseMaxSelectTicks = 0;
    PhaseMaxBackupTicks = 0;
    System.Array.Clear(PhaseHistSelect);
    System.Array.Clear(PhaseHistBackup);
  }

  /// <summary>
  /// Records the select-phase duration (Stopwatch ticks) for one batch.
  /// </summary>
  internal void RecordSelectPhase(long ticks)
  {
    Interlocked.Add(ref PhaseTicksSelect, ticks);
    RecordPhaseHist(ticks, PhaseHistSelect, ref PhaseMaxSelectTicks);
  }

  /// <summary>
  /// Records the evaluate-phase duration (Stopwatch ticks) for one batch.
  /// </summary>
  internal void RecordEvalPhase(long ticks)
  {
    Interlocked.Add(ref PhaseTicksEval, ticks);
  }

  /// <summary>
  /// Records the backup-phase duration (Stopwatch ticks) for one batch and counts the batch.
  /// </summary>
  internal void RecordBackupPhase(long ticks)
  {
    Interlocked.Add(ref PhaseTicksBackup, ticks);
    RecordPhaseHist(ticks, PhaseHistBackup, ref PhaseMaxBackupTicks);
    Interlocked.Increment(ref PhaseBatchCount);
  }

  void RecordPhaseHist(long ticks, long[] hist, ref long maxField)
  {
    long cur;
    while ((cur = Volatile.Read(ref maxField)) < ticks
        && Interlocked.CompareExchange(ref maxField, ticks, cur) != cur)
    {
    }

    double ms = ticks * 1000.0 / Stopwatch.Frequency;
    for (int b = 0; b < PHASE_BUCKET_MS.Length; b++)
    {
      if (ms <= PHASE_BUCKET_MS[b])
      {
        Interlocked.Increment(ref hist[b]);
        break;
      }
    }
  }

  static string HistString(long[] hist, int[] colWidth)
  {
    var sb = new System.Text.StringBuilder();
    for (int b = 0; b < PHASE_BUCKET_MS.Length; b++)
    {
      string hi = double.IsInfinity(PHASE_BUCKET_MS[b]) ? "inf" : PHASE_BUCKET_MS[b].ToString("0.##");
      if (b > 0)
      {
        sb.Append("  "); // two spaces between buckets for readability
      }
      sb.Append($"<={hi}:{hist[b].ToString().PadLeft(colWidth[b])}");
    }
    return sb.ToString();
  }

  /// <summary>
  /// Computes the per-bucket count-field width (widest count across both histograms) so that
  /// the select and backup rows printed on adjacent lines align vertically column-by-column.
  /// </summary>
  static int[] HistColWidths(long[] histA, long[] histB)
  {
    int[] widths = new int[PHASE_BUCKET_MS.Length];
    for (int b = 0; b < PHASE_BUCKET_MS.Length; b++)
    {
      widths[b] = System.Math.Max(histA[b].ToString().Length, histB[b].ToString().Length);
    }
    return widths;
  }

  /// <summary>
  /// Human-readable per-batch summary of this search's phase timing.
  /// </summary>
  internal string PhaseTimingSummary()
  {
    long n = Volatile.Read(ref PhaseBatchCount);
    if (n == 0)
    {
      return "(no batches)";
    }

    double f = Stopwatch.Frequency;
    double sel = PhaseTicksSelect * 1000.0 / f, ev = PhaseTicksEval * 1000.0 / f, bk = PhaseTicksBackup * 1000.0 / f;
    double waitSel = ExclWaitSelectTicks * 1000.0 / f;
    double waitBak = ExclWaitBackupTicks * 1000.0 / f;
    double maxSel = PhaseMaxSelectTicks * 1000.0 / f, maxBak = PhaseMaxBackupTicks * 1000.0 / f;
    int[] histColWidth = HistColWidths(PhaseHistSelect, PhaseHistBackup);
    return $"batches={n:N0} per-batch[ms] select={sel / n:F3} eval={ev / n:F3} backup={bk / n:F3} "
         + $"exclWait(sel/bak)={waitSel / n:F3}/{waitBak / n:F3} max(sel/bak)={maxSel:F2}/{maxBak:F2}ms"
         + $"\n                            selectHist[ms] {HistString(PhaseHistSelect, histColWidth)}"
         + $"\n                            backupHist[ms] {HistString(PhaseHistBackup, histColWidth)}";
  }

  private void Enter(bool isBackup = false)
  {
    long t0 = Stopwatch.GetTimestamp();
    selectOrBackupExclusionLock.Enter();
    long waited = Stopwatch.GetTimestamp() - t0;
    if (isBackup)
    {
      Interlocked.Add(ref ExclWaitBackupTicks, waited);
    }
    else
    {
      Interlocked.Add(ref ExclWaitSelectTicks, waited);
    }
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
    // Enforce in-order backup BEFORE taking the exclusion lock (avoids holding it while waiting
    // on an earlier batch which itself needs the exclusion lock to back up).
    EnterBackupOrder(batchIndex);

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
    Enter(isBackup: true);
    // Record backup order under the exclusion lock (so the observed order is exactly the serialized
    // backup order). With EnforceInOrderBackup this will be in-order; otherwise it measures the
    // natural crossing rate.
    RecordBackupOrder(batchIndex, didBackup: true);
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
    ExitBackupOrder(batchIndex);
  }
}
