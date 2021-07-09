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

using Ceres.Base.Benchmarking;
using Ceres.Base.Math;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Code for benchmarking NNEvaluators, and also for
  /// determing their batch size breaks 
  /// (peaks of perforamnce based on batch size which varies depending on GPU model).
  /// </summary>
  public static class NNEvaluatorBenchmark
  {
    /// <summary>
    /// Constructs a test batch of specified size.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <param name="count"></param>
    /// <param name="fen"></param>
    /// <returns></returns>
    public static EncodedPositionBatchFlat MakeTestBatch(NNEvaluator evaluator, int count, string fen = null)
    {
      EncodedPositionBatchFlat batch;

      if (fen == null) fen = Position.StartPosition.FEN;
      Position rawPos = Position.FromFEN(fen);
      MGPosition mgPos = MGPosition.FromPosition(rawPos);

      EncodedPositionWithHistory position = EncodedPositionWithHistory.FromFEN(fen);
      EncodedPositionWithHistory[] positions = new EncodedPositionWithHistory[count];
      Array.Fill(positions, position);


      bool hasPositions = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Positions);
      bool hasMoves = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves);
      bool hasHashes = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes);
      bool hasBoards = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Boards);

      batch = new EncodedPositionBatchFlat(positions, count, hasPositions);

      if (fen != null)
      {
        if (hasPositions) batch.Positions = new MGPosition[count];
        if (hasHashes) batch.PositionHashes = new ulong[count];
        if (hasMoves) batch.Moves = new MGMoveList[count];

        for (int i = 0; i < count; i++)
        {
          if (hasPositions) batch.Positions[i] = MGChessPositionConverter.MGChessPositionFromFEN(fen);
          if (hasHashes) batch.PositionHashes[i] = (ulong)i + (ulong)batch.Positions[i].GetHashCode();
          if (hasMoves)
          {
            MGMoveList moves = new MGMoveList();
            MGMoveGen.GenerateMoves(in mgPos, moves);
            batch.Moves[i] = moves;
          }
        }
      }
      return batch;
    }


    public static void Warmup(NNEvaluator evaluator)
    {
      evaluator.EvaluateIntoBuffers(MakeTestBatch(evaluator, 1), false);
    }

    static EncodedPositionBatchFlat batch1;
    static EncodedPositionBatchFlat batchBig;


    /// <summary>
    /// Estimates performance of evaluating either single positions or batches.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <param name="computeBreaks"></param>
    /// <param name="bigBatchSize"></param>
    /// <param name="estimateSingletons"></param>
    /// <param name="numWarmups"></param>
    /// <returns></returns>
    public static (float NPSSingletons, float NPSBigBatch, int[] Breaks) EstNPS(NNEvaluator evaluator, bool computeBreaks = false,
                                                                                int bigBatchSize = 256, bool estimateSingletons = true,
                                                                                int numWarmups = 1)
    {
      if (batch1 == null)
      {
        batchBig = MakeTestBatch(evaluator, bigBatchSize);
        batch1 = MakeTestBatch(evaluator, 1);
      }

      IPositionEvaluationBatch result;

      // Run numerous batches to "warm up" the GPU (make sure in full power state).
      for (int i = 0; i < numWarmups; i++)
      {
        for (int j=0;j<50;j++) evaluator.EvaluateIntoBuffers(batch1, false);
        result = evaluator.EvaluateIntoBuffers(batchBig, false);
        for (int j = 0; j < 50; j++) evaluator.EvaluateIntoBuffers(batch1, false);
      }

      float npsSingletons = float.NaN;
      if (estimateSingletons)
      {
        // Singletons
        const int NUM_SINGLETONS = 20;
        TimingStats statsSingletons = new TimingStats();
        float accumulatedTimeSingletons = 0;
        for (int i = 0; i < NUM_SINGLETONS; i++)
        {
          using (new TimingBlock(statsSingletons, TimingBlock.LoggingType.None))
          {
            result = evaluator.EvaluateIntoBuffers(batch1, false);
            accumulatedTimeSingletons += (float)statsSingletons.ElapsedTimeSecs;
          }
        }
        npsSingletons = NUM_SINGLETONS / accumulatedTimeSingletons;
      }

      // Big batch
      TimingStats statsBig = new TimingStats();
      using (new TimingBlock(statsBig, TimingBlock.LoggingType.None))
      {
        // To make sure we defeat any possible caching in place,
        // randomize the batch in some trivial way
        result = evaluator.EvaluateIntoBuffers(batchBig, false);
      }

      float npsBatchBig = bigBatchSize / (float)statsBig.ElapsedTimeSecs;

      int[] breaks = computeBreaks ? FindBreaks(evaluator, 48, 432, 0) : null;

      return (npsSingletons, npsBatchBig, breaks);
    }


    static float GetNPSEstimate(NNEvaluator evaluator, int numEstimates, int batchSize,
                                   float abortIfGreaterThanNPS, float latencyAdjustmentSecs)
    {
      float worst = 0;
      float[] samples = new float[numEstimates];
      for (int i = 0; i < numEstimates; i++)
      {
        float nps = NPSAtBatchSize(evaluator, batchSize + 1, latencyAdjustmentSecs);
        if (nps > abortIfGreaterThanNPS) return nps;
        if (nps < worst) worst = nps;
        samples[i] = nps;
      }

      // Remove single worst observation (assume noise)
      int count = 0;
      float[] outlierAdjustedSamples = new float[numEstimates - 1];
      for (int i = 0; i < outlierAdjustedSamples.Length; i++)
        if (samples[i] != worst)
          outlierAdjustedSamples[count++] = samples[i];

      return (float)StatUtils.Average(outlierAdjustedSamples);
    }


    /// <summary>
    /// Iteratively searches for break points in batch sizing performance.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <param name="minBatchSize"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="latencyAdjustmentSecs"></param>
    /// <returns></returns>
    static int[] FindBreaks(NNEvaluator evaluator, int minBatchSize, int maxBatchSize, float latencyAdjustmentSecs)
    {
      List<int> breaks = new List<int>();

      const int SKIP_COUNT = 8;
      const int SKIP_COUNT_AFTER_BREAK = 32;
      float lastNPS = 0;
      int? lastBreakPoint = null;
      for (int batchSize = minBatchSize - SKIP_COUNT; batchSize <= maxBatchSize; batchSize += SKIP_COUNT)
      {
        // Get estimated NPS (several times, to adjust for noise)
        int NUM_TRIES = 8;
        float avgNPS = GetNPSEstimate(evaluator, NUM_TRIES, batchSize, lastNPS, latencyAdjustmentSecs);
        if (lastNPS != 0)
        {
          float fraction = (float)avgNPS / (float)lastNPS;
          const float BREAK_MAX_FRACTION = 0.95f;
          bool isBreak = fraction <= BREAK_MAX_FRACTION;

          if (isBreak)
          {
            breaks.Add(batchSize);
            lastBreakPoint = batchSize;
            batchSize += SKIP_COUNT_AFTER_BREAK - SKIP_COUNT;
          }
        }
        lastNPS = avgNPS;

        // If we haven't seen any breaks within 120, for efficiency assume we won't see any more
        if (lastBreakPoint is not null && (batchSize - lastBreakPoint > 120)) break;
      }

      return breaks.ToArray();
    }


    /// <summary>
    /// Determines the nodes per second achieved at a specified batch size.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <param name="batchSize"></param>
    /// <param name="latencyAdjustmentSecs"></param>
    /// <returns></returns>
    static float NPSAtBatchSize(NNEvaluator evaluator, int batchSize, float latencyAdjustmentSecs)
    {
      TimingStats statsBig = new TimingStats();
      EncodedPositionBatchFlat positions = MakeTestBatch(evaluator, batchSize);
      using (new TimingBlock(statsBig, TimingBlock.LoggingType.None))
      {
        IPositionEvaluationBatch result = evaluator.EvaluateIntoBuffers(positions, false);
      }

      float npsBatchBig = batchSize / ((float)statsBig.ElapsedTimeSecs - latencyAdjustmentSecs);
      return npsBatchBig;
    }


    public static float[] GetBigBatchNPSFractions(NNEvaluator[] evaluators)
    {
      (float, float, int[])[] stats = new (float, float, int[])[evaluators.Length];
      float totalBigBatch = 0;
      for (int i = 0; i < evaluators.Length; i++)
      {
        stats[i] = NNEvaluatorBenchmark.EstNPS(evaluators[i], false);
        totalBigBatch += stats[i].Item2;
      }

      float[] fracs = new float[evaluators.Length];
      for (int i = 0; i < evaluators.Length; i++)
        fracs[i] = stats[i].Item2 / totalBigBatch;

      return fracs;
    }

  }
}
