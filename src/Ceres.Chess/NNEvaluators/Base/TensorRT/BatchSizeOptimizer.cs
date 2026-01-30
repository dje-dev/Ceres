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
using System.Linq;
using System.Threading.Tasks;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Optimizes a set of anchor batch sizes to maximize average throughput
/// across all batch sizes in the typical operating range.
///
/// Each batch size can be adjusted by at most ±10% (1/10th its value).
/// All batch sizes must be even numbers.
///
/// Algorithm: Parallel hill-climbing that evaluates throughput at sample
/// points and selects configurations with highest total throughput.
/// </summary>
public static class BatchSizeOptimizer
{
  const int NUM_ITERATIONS = 50;

  /// <summary>
  /// Result of the optimization.
  /// </summary>
  public record OptimizationResult(
    int[] OriginalSizes,
    int[] OptimizedSizes,
    float OriginalScore,
    float OptimizedScore,
    float ImprovementPercent
  );

  /// <summary>
  /// Optimizes batch sizes to minimize throughput cliffs in the execution time curve.
  /// Uses parallel evaluation across all CPU cores for fast execution (< 1 second).
  /// </summary>
  /// <param name="originalSizes">Original anchor batch sizes (must be sorted ascending).</param>
  /// <param name="originalTimingsPerGPU">Measured execution times [gpu][sizeIndex] in milliseconds.</param>
  /// <param name="numGPUs">Number of GPUs to consider.</param>
  /// <param name="maxBatch">Maximum batch size to evaluate (1..maxBatch).</param>
  /// <param name="maxIterations">Maximum hill-climbing iterations (default 20 for speed).</param>
  /// <returns>Optimization result with original and optimized sizes.</returns>
  public static OptimizationResult Optimize(
    int[] originalSizes,
    float[][] originalTimingsPerGPU,
    int numGPUs,
    int maxBatch)
  {
    if (originalSizes == null || originalSizes.Length == 0)
    {
      throw new ArgumentException("originalSizes cannot be null or empty");
    }
    if (originalTimingsPerGPU == null || originalTimingsPerGPU.Length < numGPUs)
    {
      throw new ArgumentException("originalTimingsPerGPU must have timings for all GPUs");
    }

    int n = originalSizes.Length;

    // Compute bounds: each size can vary by ±10%, rounded to even numbers
    int[] minBounds = new int[n];
    int[] maxBounds = new int[n];
    for (int i = 0; i < n; i++)
    {
      int delta = Math.Max(2, originalSizes[i] / 10);
      minBounds[i] = Math.Max(2, originalSizes[i] - delta);
      maxBounds[i] = originalSizes[i] + delta;
      // Round bounds to even
      if (minBounds[i] % 2 != 0) minBounds[i]++;
      if (maxBounds[i] % 2 != 0) maxBounds[i]--;
    }

    // Evaluate original configuration
    float originalScore = EvaluateSolution(originalSizes, originalSizes, originalTimingsPerGPU, numGPUs, maxBatch);

    // Start with original sizes
    int[] best = (int[])originalSizes.Clone();
    float bestScore = originalScore;

    // Parallel hill climbing: evaluate all single-dimension moves in parallel
    // Delta percentages of base batch size, rounded to nearest multiple of 2
    int[] deltaPercents = { -10, -7, -4, -2, 2, 4, 7, 10 };

    for (int iter = 0; iter < NUM_ITERATIONS; iter++)
    {
      // Generate all valid candidate moves
      List<(int idx, int delta, int[] candidate)> candidates = new();
      for (int i = 0; i < n; i++)
      {
        foreach (int pct in deltaPercents)
        {
          // Compute delta as percentage of base batch size, rounded to nearest multiple of 2
          int rawDelta = (int)Math.Round(best[i] * pct / 100.0);
          int d = ((rawDelta + (rawDelta >= 0 ? 1 : -1)) / 2) * 2; // Round to nearest even
          if (d == 0) d = pct > 0 ? 2 : -2; // Ensure non-zero delta
          int newVal = best[i] + d;

          // All batch sizes must be even
          if (newVal % 2 != 0)
          {
            newVal += 1;
          }

          // Check bounds
          if (newVal < minBounds[i] || newVal > maxBounds[i])
          {
            continue;
          }

          // Check ordering (sizes must remain strictly increasing with gap >= 2)
          if (i > 0 && newVal <= best[i - 1] + 1)
          {
            continue;
          }
          if (i < n - 1 && newVal >= best[i + 1] - 1)
          {
            continue;
          }

          int[] candidate = (int[])best.Clone();
          candidate[i] = newVal;
          candidates.Add((i, d, candidate));
        }
      }

      if (candidates.Count == 0)
      {
        break;
      }

      // Evaluate all candidates in parallel
      float[] scores = new float[candidates.Count];
      Parallel.For(0, candidates.Count, j =>
      {
        scores[j] = EvaluateSolution(candidates[j].candidate, originalSizes, originalTimingsPerGPU, numGPUs, maxBatch);
      });

      // Find best candidate
      int bestIdx = -1;
      float bestCandidateScore = bestScore;
      for (int j = 0; j < candidates.Count; j++)
      {
        if (scores[j] < bestCandidateScore - 0.0001f)
        {
          bestCandidateScore = scores[j];
          bestIdx = j;
        }
      }

      if (bestIdx < 0)
      {
        break; // Converged
      }

      best = candidates[bestIdx].candidate;
      bestScore = bestCandidateScore;
    }

    float improvement = originalScore > 0 ? (originalScore - bestScore) / originalScore * 100 : 0;

    return new OptimizationResult(originalSizes, best, originalScore, bestScore, improvement);
  }

  /// <summary>
  /// Evaluates a candidate batch size configuration by computing the execution time curve
  /// and scoring based on gaps and throughput drops.
  /// </summary>
  private static float EvaluateSolution(
    int[] candidateSizes,
    int[] originalSizes,
    float[][] originalTimingsPerGPU,
    int numGPUs,
    int maxBatch)
  {
    int n = candidateSizes.Length;

    // Interpolate timings for candidate sizes based on original measurements
    float[][] timings = new float[numGPUs][];
    for (int g = 0; g < numGPUs; g++)
    {
      timings[g] = new float[n];
      for (int i = 0; i < n; i++)
      {
        timings[g][i] = InterpolateTiming(candidateSizes[i], originalSizes, originalTimingsPerGPU[g]);
      }
    }

    // Generate sample points focused on critical regions
    List<int> samplePoints = GenerateSamplePoints(candidateSizes, maxBatch);

    // Compute execution times at sample points using BatchScheduler
    float[] times = new float[samplePoints.Count];
    for (int i = 0; i < samplePoints.Count; i++)
    {
      int k = samplePoints[i];
      BatchScheduler.ScheduleResult result = BatchScheduler.Schedule(numGPUs, candidateSizes, timings, k);
      times[i] = result.TotalEstimatedMs;
    }

    // Compute gap score
    return ComputeGapScore(samplePoints, times);
  }

  /// <summary>
  /// Generates sample batch sizes for throughput evaluation.
  /// Uses dense sampling to accurately measure total throughput impact.
  /// </summary>
  private static List<int> GenerateSamplePoints(int[] batchSizes, int maxBatch)
  {
    // This provides good coverage while keeping evaluation fast
    List<int> points = new();
    int limit = Math.Min(maxBatch, batchSizes[^1] * 4);
    for (int k = 16; k <= limit; k += 1)
    {
      points.Add(k);
    }
    return points;
  }

  /// <summary>
  /// Computes score based on total throughput (positions/ms).
  /// Lower score = higher throughput = better.
  /// </summary>
  private static float ComputeGapScore(List<int> samplePoints, float[] times)
  {
    // Pure throughput optimization: sum of nps at each sample point
    // Return negative so lower score = higher throughput
    float totalNps = 0f;
    for (int i = 0; i < times.Length; i++)
    {
      totalNps += samplePoints[i] / Math.Max(0.001f, times[i]) * 1000f;
    }
    return -totalNps;
  }

  /// <summary>
  /// Interpolates the expected timing for a batch size based on measured timings
  /// at the original anchor sizes. Uses linear interpolation between brackets
  /// and linear extrapolation outside the measured range.
  /// </summary>
  private static float InterpolateTiming(int size, int[] originalSizes, float[] originalTimings)
  {
    if (originalSizes.Length == 0)
    {
      return 1f;
    }

    // Find bracketing indices
    int lowerIdx = -1;
    int upperIdx = -1;
    for (int i = 0; i < originalSizes.Length; i++)
    {
      if (originalSizes[i] <= size)
      {
        lowerIdx = i;
      }
      if (originalSizes[i] >= size && upperIdx < 0)
      {
        upperIdx = i;
      }
    }

    // Extrapolate below
    if (lowerIdx < 0)
    {
      float rate = originalTimings[0] / originalSizes[0];
      return rate * size;
    }

    // Extrapolate above
    if (upperIdx < 0)
    {
      float rate = originalTimings[^1] / originalSizes[^1];
      return rate * size;
    }

    // Exact match
    if (lowerIdx == upperIdx)
    {
      return originalTimings[lowerIdx];
    }

    // Linear interpolation
    float t = (float)(size - originalSizes[lowerIdx]) / (originalSizes[upperIdx] - originalSizes[lowerIdx]);
    return originalTimings[lowerIdx] + t * (originalTimings[upperIdx] - originalTimings[lowerIdx]);
  }

  /// <summary>
  /// Runs optimization and prints a concise summary of results.
  /// Designed to run in under 1 second.
  /// </summary>
  public static void AnalyzeAndOptimize(
    int[] originalSizes,
    float[][] timingsPerGPU,
    int numGPUs,
    int maxBatch)
  {
    Console.WriteLine("Optimizing batch sizes...");

    // Run optimization
    OptimizationResult result = Optimize(originalSizes, timingsPerGPU, numGPUs, maxBatch);

    // Check if sizes actually changed
    bool changed = false;
    for (int i = 0; i < originalSizes.Length; i++)
    {
      if (originalSizes[i] != result.OptimizedSizes[i])
      {
        changed = true;
        break;
      }
    }

    if (changed)
    {
      Console.WriteLine($"  Original:  [{string.Join(", ", result.OriginalSizes)}]");
      Console.WriteLine($"  Optimized: [{string.Join(", ", result.OptimizedSizes)}]");
      // Score is negative total throughput, so improvement means more negative (higher throughput)
      float origThroughput = -result.OriginalScore / 1000f;
      float optThroughput = -result.OptimizedScore / 1000f;
      float pctImprove = origThroughput > 0 ? (optThroughput - origThroughput) / origThroughput * 100f : 0f;
      Console.WriteLine($"  Throughput: {origThroughput:F0}k -> {optThroughput:F0}k nps ({pctImprove:+0.0;-0.0}%)");
    }
    else
    {
      Console.WriteLine($"  Batch sizes are near-optimal.");
    }
  }
}
