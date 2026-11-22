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
using System.Linq;
using System.Threading.Tasks;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Latency-aware static balancing for NNEvaluatorSplit across heterogeneous
  /// sub-evaluators (e.g. a local GPU plus one or more remote hosts).
  ///
  /// 1. Calibration (once, at construction): each sub-evaluator is timed end-to-end
  ///    (including network round trip, serialization and compression for remote evaluators)
  ///    over a ladder of batch sizes, using realistic diverse positions, with all sub-evaluators
  ///    measured concurrently (as they run during search). This yields a monotone time table t_i(n).
  ///
  /// 2. Planning: for every batch size N, choose the allocation n_i (sum = N, n_i <= MaxBatchSize_i)
  ///    minimizing the makespan max_i t_i(n_i) + SplitPenaltyMs * (#evaluators used - 1).
  ///    Because t_i includes per-call latency, small batches stay on the fastest/lowest-latency
  ///    evaluator and remote hosts only receive work once a batch is large enough to pay for it.
  ///    All plans are precomputed, so the per-batch cost at run time is a table lookup.
  ///
  /// No recalibration is performed while running (each server is assumed to serve a single client).
  /// </summary>
  public sealed class NNEvaluatorSplitBalancer
  {
    /// <summary>
    /// Batch sizes measured during calibration (clipped to each evaluator's maximum).
    /// </summary>
    static readonly int[] CALIBRATION_SIZES = BuildLadder();

    /// <summary>
    /// Dense ladder (TensorRT time is a step function of batch size due to padding to engine profile sizes,
    /// so sizes are measured densely: every 8 up to 256, then 16 to 512, 32 to 1024, 64 to 2048, 256 to 8192).
    /// </summary>
    static int[] BuildLadder()
    {
      List<int> sizes = new() { 1, 2, 4 };
      for (int n = 8; n <= 256; n += 8) sizes.Add(n);
      for (int n = 272; n <= 512; n += 16) sizes.Add(n);
      for (int n = 544; n <= 1024; n += 32) sizes.Add(n);
      for (int n = 1088; n <= 2048; n += 64) sizes.Add(n);
      for (int n = 2304; n <= 8192; n += 256) sizes.Add(n);
      return sizes.ToArray();
    }

    /// <summary>
    /// Fixed cost charged per additional sub-evaluator used for a batch (dispatch/merge overhead).
    /// </summary>
    public const float DEFAULT_SPLIT_PENALTY_MS = 0.05f;

    /// <summary>
    /// Maximum number of sub-evaluators for which all subsets are considered when planning.
    /// </summary>
    const int MAX_EVALUATORS_SUBSET_SEARCH = 8;

    public readonly string[] Names;
    public readonly int[] MaxBatchSizes;

    /// <summary>
    /// Measured (batch size, milliseconds) points per evaluator.
    /// </summary>
    public readonly (int n, float ms)[][] Measurements;

    /// <summary>
    /// Monotone interpolated time tables: Times[i][n] = estimated ms to evaluate n positions on evaluator i.
    /// </summary>
    public readonly float[][] Times;

    /// <summary>
    /// Plans[N][i] = number of positions given to evaluator i for a batch of N positions.
    /// </summary>
    public readonly int[][] Plans;

    /// <summary>
    /// PredictedMs[N] = predicted makespan (ms) of the plan for N positions (excluding split penalty).
    /// </summary>
    public readonly float[] PredictedMs;

    public readonly float SplitPenaltyMs;
    public readonly double CalibrationSeconds;

    public int MaxTotalBatchSize => Plans.Length - 1;


    NNEvaluatorSplitBalancer(string[] names, int[] maxBatchSizes, (int, float)[][] measurements,
                             float splitPenaltyMs, double calibrationSeconds)
    {
      Names = names;
      MaxBatchSizes = maxBatchSizes;
      Measurements = measurements;
      SplitPenaltyMs = splitPenaltyMs;
      CalibrationSeconds = calibrationSeconds;

      Times = new float[names.Length][];
      for (int i = 0; i < names.Length; i++)
      {
        Times[i] = BuildTimeTable(measurements[i], maxBatchSizes[i]);
      }

      (Plans, PredictedMs) = BuildPlans(Times, splitPenaltyMs);
    }


    /// <summary>
    /// Creates a balancer from explicit measurements (no evaluation performed); useful for testing.
    /// </summary>
    public static NNEvaluatorSplitBalancer FromMeasurements(string[] names, int[] maxBatchSizes, (int, float)[][] measurements,
                                                            float splitPenaltyMs = DEFAULT_SPLIT_PENALTY_MS)
      => new(names, maxBatchSizes, measurements, splitPenaltyMs, 0);


    /// <summary>
    /// Calibrates the given evaluators (concurrently) and builds the balancing plans.
    /// </summary>
    public static NNEvaluatorSplitBalancer Calibrate(NNEvaluator[] evaluators, string[] names,
                                                     float splitPenaltyMs = DEFAULT_SPLIT_PENALTY_MS)
    {
      Stopwatch sw = Stopwatch.StartNew();
      int[] maxBatchSizes = evaluators.Select(e => e.MaxBatchSize).ToArray();
      (int, float)[][] measurements = new (int, float)[evaluators.Length][];

      Parallel.For(0, evaluators.Length, new ParallelOptions() { MaxDegreeOfParallelism = evaluators.Length },
                   i => measurements[i] = CalibrateOne(evaluators[i], maxBatchSizes[i]));

      return new NNEvaluatorSplitBalancer(names, maxBatchSizes, measurements, splitPenaltyMs, sw.Elapsed.TotalSeconds);
    }


    /// <summary>
    /// Measures one evaluator over the calibration ladder.
    /// </summary>
    static (int, float)[] CalibrateOne(NNEvaluator evaluator, int maxBatchSize)
    {
      int[] sizes = CALIBRATION_SIZES.Where(s => s < maxBatchSize).Append(maxBatchSize).Distinct().ToArray();
      EncodedPositionBatchFlat fullBatch = MakeDiverseBatch(evaluator, maxBatchSize);

      // Warmup: touch every size once (TensorRT engines may lazily set up profiles / CUDA graphs).
      for (int rep = 0; rep < 2; rep++)
      {
        evaluator.EvaluateIntoBuffers(fullBatch.GetSubBatchSlice(0, 1));
      }
      foreach (int n in sizes)
      {
        evaluator.EvaluateIntoBuffers(fullBatch.GetSubBatchSlice(0, n));
      }

      List<(int, float)> ret = new(sizes.Length);
      foreach (int n in sizes)
      {
        IEncodedPositionBatchFlat batch = fullBatch.GetSubBatchSlice(0, n);
        int runs = n <= 256 ? 5 : (n <= 1024 ? 4 : 3);
        float[] t = new float[runs];
        for (int r = 0; r < runs; r++)
        {
          long t0 = Stopwatch.GetTimestamp();
          evaluator.EvaluateIntoBuffers(batch);
          t[r] = (float)Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        Array.Sort(t);
        ret.Add((n, t[runs / 2])); // median
      }
      return ret.ToArray();
    }


    /// <summary>
    /// Builds a monotone (non-decreasing) time table for n = 0..maxBatchSize using conservative step
    /// interpolation: t(n) = measured time of the smallest measured size >= n.
    ///
    /// Linear interpolation is wrong for TensorRT-style backends whose time is a step function of batch size
    /// (n just above an engine profile size is padded to the next profile). The step estimate is an upper
    /// bound for monotone backends, and has the useful property that the planner's capacity for any time
    /// limit always lands on a measured size (the right end of a plateau), where the estimate is exact.
    /// </summary>
    static float[] BuildTimeTable((int n, float ms)[] points, int maxBatchSize)
    {
      (int n, float ms)[] pts = points.OrderBy(p => p.n).ToArray();
      float[] t = new float[maxBatchSize + 1];
      t[0] = 0;
      int j = 0;
      for (int n = 1; n <= maxBatchSize; n++)
      {
        while (j < pts.Length - 1 && pts[j].n < n)
        {
          j++;
        }
        t[n] = pts[j].ms;
      }

      // Enforce monotonicity: evaluating n positions never costs more than evaluating m >= n positions.
      for (int n = maxBatchSize - 1; n >= 1; n--)
      {
        t[n] = MathF.Min(t[n], t[n + 1]);
      }
      return t;
    }


    /// <summary>
    /// Largest n <= max with t[n] <= limitMs (t is non-decreasing).
    /// </summary>
    static int Capacity(float[] t, float limitMs)
    {
      int lo = 0, hi = t.Length - 1;
      while (lo < hi)
      {
        int mid = (lo + hi + 1) / 2;
        if (t[mid] <= limitMs)
        {
          lo = mid;
        }
        else
        {
          hi = mid - 1;
        }
      }
      return lo;
    }


    /// <summary>
    /// Computes the optimal allocation for every total batch size.
    /// </summary>
    static (int[][] plans, float[] predicted) BuildPlans(float[][] times, float splitPenaltyMs)
    {
      int k = times.Length;
      int maxTotal = times.Sum(t => t.Length - 1);
      int[][] plans = new int[maxTotal + 1][];
      float[] predicted = new float[maxTotal + 1];
      plans[0] = new int[k];

      // Candidate subsets (bit masks) of evaluators.
      List<int> subsets = new();
      if (k <= MAX_EVALUATORS_SUBSET_SEARCH)
      {
        for (int mask = 1; mask < (1 << k); mask++)
        {
          subsets.Add(mask);
        }
      }
      else
      {
        subsets.Add((1 << k) - 1); // all
        for (int i = 0; i < k; i++)
        {
          subsets.Add(1 << i);
        }
      }

      // Candidate makespans: every distinct table value (the optimum is always one of them).
      float[] candidates = times.SelectMany(t => t.Skip(1)).Distinct().OrderBy(v => v).ToArray();

      for (int N = 1; N <= maxTotal; N++)
      {
        float bestScore = float.MaxValue;
        float bestMakespan = 0;
        int[] best = null;

        foreach (int mask in subsets)
        {
          int maxInSubset = 0;
          int count = 0;
          for (int i = 0; i < k; i++)
          {
            if ((mask & (1 << i)) != 0)
            {
              maxInSubset += times[i].Length - 1;
              count++;
            }
          }
          if (maxInSubset < N)
          {
            continue;
          }

          // Smallest candidate makespan T such that the subset can cover N positions within T.
          int lo = 0, hi = candidates.Length - 1;
          while (lo < hi)
          {
            int mid = (lo + hi) / 2;
            if (SubsetCapacity(times, mask, candidates[mid]) >= N)
            {
              hi = mid;
            }
            else
            {
              lo = mid + 1;
            }
          }
          float T = candidates[lo];

          float score = T + splitPenaltyMs * (count - 1);
          if (score < bestScore - 1e-6f)
          {
            int[] alloc = new int[k];
            for (int i = 0; i < k; i++)
            {
              alloc[i] = (mask & (1 << i)) != 0 ? Capacity(times[i], T) : 0;
            }
            float makespan = TrimExcess(times, alloc, N);
            score = makespan + splitPenaltyMs * (alloc.Count(a => a > 0) - 1);
            if (score < bestScore - 1e-6f)
            {
              bestScore = score;
              bestMakespan = makespan;
              best = alloc;
            }
          }
        }

        plans[N] = best;
        predicted[N] = bestMakespan;
      }

      return (plans, predicted);
    }


    static int SubsetCapacity(float[][] times, int mask, float limitMs)
    {
      int sum = 0;
      for (int i = 0; i < times.Length; i++)
      {
        if ((mask & (1 << i)) != 0)
        {
          sum += Capacity(times[i], limitMs);
        }
      }
      return sum;
    }


    /// <summary>
    /// Removes the excess over N, one position at a time from the evaluator currently finishing last.
    /// Returns the resulting makespan.
    /// </summary>
    static float TrimExcess(float[][] times, int[] alloc, int N)
    {
      int excess = alloc.Sum() - N;
      while (excess > 0)
      {
        int worst = -1;
        for (int i = 0; i < alloc.Length; i++)
        {
          if (alloc[i] > 0 && (worst < 0 || times[i][alloc[i]] > times[worst][alloc[worst]]
                                          || (times[i][alloc[i]] == times[worst][alloc[worst]] && alloc[i] > alloc[worst])))
          {
            worst = i;
          }
        }
        alloc[worst]--;
        excess--;
      }

      float makespan = 0;
      for (int i = 0; i < alloc.Length; i++)
      {
        if (alloc[i] > 0)
        {
          makespan = MathF.Max(makespan, times[i][alloc[i]]);
        }
      }
      return makespan;
    }


    /// <summary>
    /// Returns a human-readable summary of calibration and representative plans.
    /// </summary>
    public string Summary()
    {
      System.Text.StringBuilder sb = new();
      sb.AppendLine($"[BalancedSplit] calibrated {Names.Length} evaluators in {CalibrationSeconds:F1}s (split penalty {SplitPenaltyMs}ms)");
      int[] showSizes = { 1, 16, 64, 128, 256, 512, 1024, 2048 };
      for (int i = 0; i < Names.Length; i++)
      {
        string pts = string.Join("  ", showSizes.Where(s => s <= MaxBatchSizes[i]).Select(s => $"{s}:{Times[i][s]:F2}"));
        sb.AppendLine($"[BalancedSplit]   #{i} {Names[i]}  max={MaxBatchSizes[i]}  ms@n  {pts}");
      }
      foreach (int n in new[] { 1, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072 })
      {
        if (n <= MaxTotalBatchSize)
        {
          sb.AppendLine($"[BalancedSplit]   N={n,5}  plan=[{string.Join(", ", Plans[n].Select(a => $"{a,4}"))}]  predicted {PredictedMs[n]:F2}ms  ({n / PredictedMs[n] * 1000,8:N0} nps)");
        }
      }
      return sb.ToString();
    }


    #region Diverse calibration positions

    /// <summary>
    /// Creates a batch of distinct, realistic positions (random playouts from the start position,
    /// with full move history) so that serialization/compression costs of remote evaluators
    /// are representative of real search batches (not artificially compressible repeated positions).
    /// </summary>
    public static EncodedPositionBatchFlat MakeDiverseBatch(NNEvaluator evaluator, int count, int seed = 12345)
    {
      Random rand = new(seed);
      EncodedPositionWithHistory[] positions = new EncodedPositionWithHistory[count];
      MGPosition[] finalPositions = new MGPosition[count];

      for (int i = 0; i < count; i++)
      {
        List<Position> history = new() { Position.StartPosition };
        MGPosition mg = Position.StartPosition.ToMGPosition;
        int plies = rand.Next(0, 80);
        for (int p = 0; p < plies; p++)
        {
          MGMoveList moves = new();
          MGMoveGen.GenerateMoves(in mg, moves);
          if (moves.NumMovesUsed == 0)
          {
            break;
          }
          mg.MakeMove(moves.MovesArray[rand.Next(moves.NumMovesUsed)]);
          history.Add(mg.ToPosition);
        }

        // Avoid positions with no legal moves (policy undefined).
        MGMoveList finalMoves = new();
        MGMoveGen.GenerateMoves(in mg, finalMoves);
        if (finalMoves.NumMovesUsed == 0 && history.Count > 1)
        {
          history.RemoveAt(history.Count - 1);
          mg = history[^1].ToMGPosition;
        }

        Position[] last = history.Skip(Math.Max(0, history.Count - EncodedPositionBoards.NUM_MOVES_HISTORY)).ToArray();
        positions[i] = new EncodedPositionWithHistory();
        positions[i].SetFromSequentialPositions(last, true);
        finalPositions[i] = mg;
      }

      bool hasPositions = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Positions);
      bool hasMoves = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves);
      bool hasHashes = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes);

      EncodedPositionBatchFlat batch = new(positions, count, hasPositions);
      if (hasPositions) batch.Positions = new MGPosition[count];
      if (hasHashes) batch.PositionHashes = new ulong[count];
      if (hasMoves) batch.Moves = new MGMoveList[count];

      for (int i = 0; i < count; i++)
      {
        MGPosition thisPos = finalPositions[i];
        if (hasPositions) batch.Positions[i] = thisPos;
        if (hasHashes) batch.PositionHashes[i] = (ulong)i + (ulong)thisPos.GetHashCode();
        if (hasMoves)
        {
          MGMoveList moves = new();
          MGMoveGen.GenerateMoves(in thisPos, moves);
          batch.Moves[i] = moves;
        }
      }
      return batch;
    }

    #endregion
  }
}
