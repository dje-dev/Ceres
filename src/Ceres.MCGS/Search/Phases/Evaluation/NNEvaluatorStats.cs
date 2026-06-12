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

#endregion

namespace Ceres.MCGS.Search.Phases.Evaluation;

/// <summary>
/// Cumulative low-overhead statistics for an MCGSEvaluatorNeuralNet instance.
/// Updated once per batch (no per-position atomics) by the single iterator thread
/// which owns the evaluator; safe to read from other threads for reporting.
/// </summary>
public struct NNEvaluatorStats
{
  /// <summary>
  /// Total number of NN batches evaluated.
  /// </summary>
  public long NumBatches;

  /// <summary>
  /// Total number of positions evaluated across all batches.
  /// </summary>
  public long NumPositions;

  /// <summary>
  /// Smallest batch size seen.
  /// </summary>
  public int MinBatchSize;

  /// <summary>
  /// Largest batch size seen.
  /// </summary>
  public int MaxBatchSize;

  /// <summary>
  /// Number of positions whose available history was shorter than the
  /// number of history planes (so fill-in or zeroed planes were used).
  /// </summary>
  public long NumPositionsWithHistoryFillIn;

  /// <summary>
  /// Accumulated Stopwatch ticks spent encoding positions into the batch (SetBatch).
  /// </summary>
  public long EncodeTicks;

  /// <summary>
  /// Accumulated Stopwatch ticks spent in NN inference (EvaluateIntoBuffers).
  /// </summary>
  public long InferenceTicks;

  /// <summary>
  /// Accumulated Stopwatch ticks spent installing results into nodes (RetrieveResults).
  /// </summary>
  public long WriteBackTicks;


  public readonly double MeanBatchSize => NumBatches == 0 ? 0 : (double)NumPositions / NumBatches;

  public readonly double EncodeSeconds => (double)EncodeTicks / Stopwatch.Frequency;

  public readonly double InferenceSeconds => (double)InferenceTicks / Stopwatch.Frequency;

  public readonly double WriteBackSeconds => (double)WriteBackTicks / Stopwatch.Frequency;


  /// <summary>
  /// Registers a completed batch.
  /// </summary>
  /// <param name="batchSize"></param>
  /// <param name="numShortHistory">number of positions in batch with incomplete history planes</param>
  public void RegisterBatch(int batchSize, int numShortHistory)
  {
    NumBatches++;
    NumPositions += batchSize;
    MinBatchSize = NumBatches == 1 ? batchSize : Math.Min(MinBatchSize, batchSize);
    MaxBatchSize = Math.Max(MaxBatchSize, batchSize);
    NumPositionsWithHistoryFillIn += numShortHistory;
  }


  /// <summary>
  /// Resets all statistics to zero.
  /// </summary>
  public void Reset() => this = default;


  /// <summary>
  /// Returns string summary.
  /// </summary>
  /// <returns></returns>
  public override readonly string ToString()
  {
    return $"batches {NumBatches:N0}  positions {NumPositions:N0}  "
         + $"size min/mean/max {MinBatchSize}/{MeanBatchSize:F1}/{MaxBatchSize}  "
         + $"encode {EncodeSeconds:F2}s  infer {InferenceSeconds:F2}s  apply {WriteBackSeconds:F2}s  "
         + $"short history {NumPositionsWithHistoryFillIn:N0}";
  }
}
