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
using System.Linq;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Optimal batch scheduling for single or multi-GPU inference.
/// Minimizes execution time while considering padding overhead.
/// Uses execution time data per GPU to make informed decisions.
/// </summary>
public static class BatchScheduler
{
  /// <summary>
  /// Per-batch overhead in milliseconds. This accounts for kernel launch, memory copy setup,
  /// and other fixed costs per inference call. Estimated from typical CUDA/TensorRT workloads.
  /// </summary>
  public const float PER_BATCH_OVERHEAD_MS = 0.5f;

  /// <summary>
  /// Result of the scheduling algorithm.
  /// </summary>
  public record ScheduleResult(
    int[][] ExecutionPlan,  // Outer: GPU index, Inner: batch sizes to execute in order
    int MaxTimeMS,          // Maximum execution time across all GPUs
    int TotalPositions,     // Total positions including padding
    int Padding             // Wasted padding positions
  );

  /// <summary>
  /// Schedules batch execution across multiple GPUs optimally.
  /// Uses bounded recursive search over engine-size multisets with LPT assignment and greedy-seeded pruning.
  /// </summary>
  /// <param name="numGPUs">Number of available GPUs</param>
  /// <param name="engineSizes">Available engine batch sizes (e.g., [8, 20, 32, 64, 96, 192])</param>
  /// <param name="executionTimesPerGPU">Execution time in ms for each engine size, per GPU [gpu][engineIdx]</param>
  /// <param name="targetBatch">Total positions to evaluate</param>
  /// <param name="preferredGPU">GPU to assign longest-executing batches to</param>
  /// <returns>Optimal execution plan</returns>
  public static ScheduleResult Schedule(int numGPUs,
                                        int[] engineSizes, float[][] executionTimesPerGPU,
                                        int targetBatch, int preferredGPU = 0)
  {
    if (targetBatch <= 0 || numGPUs <= 0)
    {
      int[][] emptyPlan = new int[Math.Max(numGPUs, 0)][];
      for (int i = 0; i < emptyPlan.Length; i++)
      {
        emptyPlan[i] = [];
      }
      return new ScheduleResult(emptyPlan, 0, 0, 0);
    }

    preferredGPU = Math.Clamp(preferredGPU, 0, numGPUs - 1);
    int n = engineSizes.Length;

    // Sort engines descending by size, tracking original indices for per-GPU time lookup
    Span<int> sizes = stackalloc int[n];
    Span<int> origIndices = stackalloc int[n];
    for (int i = 0; i < n; i++)
    {
      sizes[i] = engineSizes[i];
      origIndices[i] = i;
    }
    for (int i = 0; i < n - 1; i++)
    {
      for (int j = i + 1; j < n; j++)
      {
        if (sizes[j] > sizes[i])
        {
          (sizes[i], sizes[j]) = (sizes[j], sizes[i]);
          (origIndices[i], origIndices[j]) = (origIndices[j], origIndices[i]);
        }
      }
    }

    // Build flat per-GPU times reordered to sorted engine order: allGPUTimes[gpu * n + sortedIdx]
    // Also compute per-engine minimum time across GPUs (for pruning lower bound)
    Span<float> allGPUTimes = stackalloc float[numGPUs * n];
    Span<float> minGPUTimes = stackalloc float[n];
    float globalMinTime = float.MaxValue;
    for (int e = 0; e < n; e++)
    {
      int origIdx = origIndices[e];
      float minT = float.MaxValue;
      for (int g = 0; g < numGPUs; g++)
      {
        float t = executionTimesPerGPU[g][origIdx];
        allGPUTimes[g * n + e] = t;
        if (t < minT)
        {
          minT = t;
        }
      }
      minGPUTimes[e] = minT;
      if (minT < globalMinTime)
      {
        globalMinTime = minT;
      }
    }

    int maxBatches = Math.Min((targetBatch + sizes[n - 1] - 1) / sizes[n - 1], 24);

    Span<int> bestPlan = stackalloc int[maxBatches];
    int bestLen = 0;
    float bestMakespan = float.MaxValue;
    int bestPadding = int.MaxValue;

    // Seed with greedy solution (largest-first) so pruning is effective from the start
    {
      Span<int> greedy = stackalloc int[maxBatches];
      int remaining = targetBatch;
      int len = 0;
      while (remaining > 0 && len < maxBatches)
      {
        int chosen = n - 1;
        for (int i = 0; i < n; i++)
        {
          if (sizes[i] <= remaining)
          {
            chosen = i;
            break;
          }
        }
        greedy[len] = chosen;
        remaining -= sizes[chosen];
        len++;
      }
      bestLen = len;
      greedy.Slice(0, len).CopyTo(bestPlan);
      bestPadding = Math.Max(0, -remaining);
      bestMakespan = ComputeLPTMakespan(bestPlan, bestLen, allGPUTimes, numGPUs, n, preferredGPU);
    }

    // Exhaustive search with pruning
    Span<int> current = stackalloc int[maxBatches];
    SearchBatchesMultiGPU(sizes, n, allGPUTimes, minGPUTimes, numGPUs, preferredGPU,
                          targetBatch, current, 0, 0, 0f, globalMinTime,
                          bestPlan, ref bestLen, ref bestMakespan, ref bestPadding, maxBatches);

    // Reconstruct per-GPU assignment from best plan using LPT
    int[][] executionPlan = BuildLPTAssignment(bestPlan, bestLen, sizes, allGPUTimes, numGPUs, n, preferredGPU);

    int totalPos = 0;
    for (int i = 0; i < bestLen; i++)
    {
      totalPos += sizes[bestPlan[i]];
    }

    return new ScheduleResult(executionPlan, (int)Math.Ceiling(bestMakespan), totalPos, totalPos - targetBatch);
  }


  /// <summary>
  /// Single-GPU scheduling: finds the batch sequence minimizing total inference time.
  /// Uses bounded recursive search over engine-size multisets with greedy-seeded pruning.
  /// No managed allocations except the returned result array.
  /// </summary>
  public static int[] ScheduleSingleGPU(ReadOnlySpan<int> engineSizes, ReadOnlySpan<float> executionTimes, int targetBatch)
  {
    if (targetBatch <= 0 || engineSizes.Length == 0)
    {
      return [];
    }

    int n = engineSizes.Length;

    // Sort engines descending by size (n <= 12, simple bubble sort)
    Span<int> sizes = stackalloc int[n];
    Span<float> times = stackalloc float[n];
    for (int i = 0; i < n; i++)
    {
      sizes[i] = engineSizes[i];
      times[i] = executionTimes[i];
    }
    for (int i = 0; i < n - 1; i++)
    {
      for (int j = i + 1; j < n; j++)
      {
        if (sizes[j] > sizes[i])
        {
          (sizes[i], sizes[j]) = (sizes[j], sizes[i]);
          (times[i], times[j]) = (times[j], times[i]);
        }
      }
    }

    // Minimum execution time across all engines (for pruning lower bound)
    float minEngineTime = times[0];
    for (int i = 1; i < n; i++)
    {
      if (times[i] < minEngineTime)
      {
        minEngineTime = times[i];
      }
    }

    // Max batches: ceil(target / smallest engine), capped to limit stack and search depth
    int maxBatches = Math.Min((targetBatch + sizes[n - 1] - 1) / sizes[n - 1], 24);

    Span<int> bestPlan = stackalloc int[maxBatches];
    int bestLen = 0;
    float bestCost = float.MaxValue;
    int bestPadding = int.MaxValue;

    // Seed with greedy solution (largest-first) so pruning is effective from the start
    {
      Span<int> greedy = stackalloc int[maxBatches];
      int remaining = targetBatch;
      int len = 0;
      float cost = 0;
      while (remaining > 0 && len < maxBatches)
      {
        // Largest engine that fits, or smallest engine if none fit
        int chosen = n - 1;
        for (int i = 0; i < n; i++)
        {
          if (sizes[i] <= remaining)
          {
            chosen = i;
            break;
          }
        }
        greedy[len] = chosen;
        cost += times[chosen];
        remaining -= sizes[chosen];
        len++;
      }
      bestCost = cost + len * PER_BATCH_OVERHEAD_MS;
      bestPadding = Math.Max(0, -remaining);
      bestLen = len;
      greedy.Slice(0, len).CopyTo(bestPlan);
    }

    // Exhaustive search with pruning (generates multisets via non-decreasing engine index)
    Span<int> current = stackalloc int[maxBatches];
    SearchBatches(sizes, times, n, targetBatch, current, 0, 0, 0f, minEngineTime,
                  bestPlan, ref bestLen, ref bestCost, ref bestPadding, maxBatches);

    // Build result array (only managed allocation)
    int[] result = new int[bestLen];
    for (int i = 0; i < bestLen; i++)
    {
      result[i] = sizes[bestPlan[i]];
    }
    return result;
  }


  /// <summary>
  /// Recursive search for optimal batch combination.
  /// Generates multisets by requiring engine index >= startIdx (avoids duplicates).
  /// Prunes branches whose lower-bound cost exceeds best known cost.
  /// </summary>
  private static void SearchBatches(
    ReadOnlySpan<int> sizes, ReadOnlySpan<float> times, int n,
    int remaining, Span<int> current, int depth, int startIdx,
    float costSoFar, float minEngineTime,
    Span<int> bestPlan, ref int bestLen, ref float bestCost, ref int bestPadding,
    int maxDepth)
  {
    if (remaining <= 0)
    {
      // Valid plan: engine sizes sum >= target
      float totalCost = costSoFar + depth * PER_BATCH_OVERHEAD_MS;
      int padding = -remaining;

      // Accept if clearly better time, or similar time (within 3%) with less padding
      if (totalCost < bestCost * 0.97f || (totalCost <= bestCost * 1.03f && padding < bestPadding))
      {
        if (totalCost < bestCost)
        {
          bestCost = totalCost;
        }
        bestPadding = padding;
        bestLen = depth;
        current.Slice(0, depth).CopyTo(bestPlan);
      }
      return;
    }

    if (depth >= maxDepth)
    {
      return;
    }

    // Lower bound: need at least ceil(remaining / largestAvailable) more batches,
    // each costing at least minEngineTime + PER_BATCH_OVERHEAD_MS
    int largestAvailable = sizes[startIdx];
    int minBatchesNeeded = (remaining + largestAvailable - 1) / largestAvailable;
    float lowerBound = costSoFar + (depth + minBatchesNeeded) * PER_BATCH_OVERHEAD_MS
                       + minBatchesNeeded * minEngineTime;
    if (lowerBound > bestCost * 1.03f)
    {
      return;
    }

    for (int i = startIdx; i < n; i++)
    {
      current[depth] = i;
      SearchBatches(sizes, times, n, remaining - sizes[i], current, depth + 1, i,
                    costSoFar + times[i], minEngineTime,
                    bestPlan, ref bestLen, ref bestCost, ref bestPadding, maxDepth);
    }
  }

  /// <summary>
  /// Recursive search for optimal multi-GPU batch combination.
  /// Generates multisets by requiring engine index >= startIdx (avoids duplicates).
  /// Evaluates each valid plan by LPT makespan; prunes using total-work / numGPUs lower bound.
  /// </summary>
  private static void SearchBatchesMultiGPU(
    ReadOnlySpan<int> sizes, int n,
    ReadOnlySpan<float> allGPUTimes, ReadOnlySpan<float> minGPUTimes, int numGPUs, int preferredGPU,
    int remaining, Span<int> current, int depth, int startIdx,
    float minCostSoFar, float globalMinTime,
    Span<int> bestPlan, ref int bestLen, ref float bestMakespan, ref int bestPadding,
    int maxDepth)
  {
    if (remaining <= 0)
    {
      float makespan = ComputeLPTMakespan(current, depth, allGPUTimes, numGPUs, n, preferredGPU);
      int padding = -remaining;

      if (makespan < bestMakespan * 0.97f || (makespan <= bestMakespan * 1.03f && padding < bestPadding))
      {
        if (makespan < bestMakespan)
        {
          bestMakespan = makespan;
        }
        bestPadding = padding;
        bestLen = depth;
        current.Slice(0, depth).CopyTo(bestPlan);
      }
      return;
    }

    if (depth >= maxDepth)
    {
      return;
    }

    // Lower bound: total minimum work / numGPUs assumes perfect load balance
    int largestAvailable = sizes[startIdx];
    int minBatchesNeeded = (remaining + largestAvailable - 1) / largestAvailable;
    int totalBatches = depth + minBatchesNeeded;
    float lowerBound = (minCostSoFar + minBatchesNeeded * globalMinTime
                        + totalBatches * PER_BATCH_OVERHEAD_MS) / numGPUs;
    if (lowerBound > bestMakespan * 1.03f)
    {
      return;
    }

    for (int i = startIdx; i < n; i++)
    {
      current[depth] = i;
      SearchBatchesMultiGPU(sizes, n, allGPUTimes, minGPUTimes, numGPUs, preferredGPU,
                            remaining - sizes[i], current, depth + 1, i,
                            minCostSoFar + minGPUTimes[i], globalMinTime,
                            bestPlan, ref bestLen, ref bestMakespan, ref bestPadding, maxDepth);
    }
  }


  /// <summary>
  /// Computes makespan using LPT (Longest Processing Time first) assignment.
  /// First batch goes to preferredGPU, remaining batches go to least-loaded GPU.
  /// Each GPU's load includes per-batch overhead.
  /// </summary>
  private static float ComputeLPTMakespan(
    ReadOnlySpan<int> batchIndices, int batchCount,
    ReadOnlySpan<float> allGPUTimes, int numGPUs, int n, int preferredGPU)
  {
    if (batchCount == 0)
    {
      return 0;
    }

    // Sort batches descending by preferred GPU time for LPT ordering
    Span<int> sorted = stackalloc int[batchCount];
    batchIndices.Slice(0, batchCount).CopyTo(sorted);
    for (int a = 0; a < batchCount - 1; a++)
    {
      for (int b = a + 1; b < batchCount; b++)
      {
        if (allGPUTimes[preferredGPU * n + sorted[b]] > allGPUTimes[preferredGPU * n + sorted[a]])
        {
          (sorted[a], sorted[b]) = (sorted[b], sorted[a]);
        }
      }
    }

    // Track per-GPU execution time and batch count
    Span<float> gpuLoads = stackalloc float[numGPUs];
    Span<int> gpuCounts = stackalloc int[numGPUs];
    gpuLoads.Clear();
    gpuCounts.Clear();

    // First batch to preferred GPU
    gpuLoads[preferredGPU] = allGPUTimes[preferredGPU * n + sorted[0]];
    gpuCounts[preferredGPU] = 1;

    // Remaining batches: assign to GPU with lowest effective load
    for (int b = 1; b < batchCount; b++)
    {
      int targetGPU = 0;
      float minLoad = gpuLoads[0] + gpuCounts[0] * PER_BATCH_OVERHEAD_MS;
      for (int g = 1; g < numGPUs; g++)
      {
        float load = gpuLoads[g] + gpuCounts[g] * PER_BATCH_OVERHEAD_MS;
        if (load < minLoad)
        {
          targetGPU = g;
          minLoad = load;
        }
      }
      gpuLoads[targetGPU] += allGPUTimes[targetGPU * n + sorted[b]];
      gpuCounts[targetGPU]++;
    }

    // Makespan = max GPU time (execution + overhead)
    float makespan = 0;
    for (int g = 0; g < numGPUs; g++)
    {
      float gpuTime = gpuLoads[g] + gpuCounts[g] * PER_BATCH_OVERHEAD_MS;
      if (gpuTime > makespan)
      {
        makespan = gpuTime;
      }
    }
    return makespan;
  }


  /// <summary>
  /// Builds the per-GPU execution plan from a batch plan using LPT assignment.
  /// Returns int[numGPUs][] with each inner array containing engine batch sizes for that GPU.
  /// </summary>
  private static int[][] BuildLPTAssignment(
    ReadOnlySpan<int> batchIndices, int batchCount,
    ReadOnlySpan<int> sizes, ReadOnlySpan<float> allGPUTimes, int numGPUs, int n, int preferredGPU)
  {
    if (batchCount == 0)
    {
      int[][] empty = new int[numGPUs][];
      for (int g = 0; g < numGPUs; g++)
      {
        empty[g] = [];
      }
      return empty;
    }

    // Sort batches descending by preferred GPU time (same ordering as ComputeLPTMakespan)
    Span<int> sorted = stackalloc int[batchCount];
    batchIndices.Slice(0, batchCount).CopyTo(sorted);
    for (int a = 0; a < batchCount - 1; a++)
    {
      for (int b = a + 1; b < batchCount; b++)
      {
        if (allGPUTimes[preferredGPU * n + sorted[b]] > allGPUTimes[preferredGPU * n + sorted[a]])
        {
          (sorted[a], sorted[b]) = (sorted[b], sorted[a]);
        }
      }
    }

    // LPT assignment tracking
    Span<float> gpuLoads = stackalloc float[numGPUs];
    Span<int> gpuCounts = stackalloc int[numGPUs];
    Span<int> assignments = stackalloc int[batchCount];
    gpuLoads.Clear();
    gpuCounts.Clear();

    // First batch to preferred GPU
    assignments[0] = preferredGPU;
    gpuLoads[preferredGPU] = allGPUTimes[preferredGPU * n + sorted[0]];
    gpuCounts[preferredGPU] = 1;

    for (int b = 1; b < batchCount; b++)
    {
      int targetGPU = 0;
      float minLoad = gpuLoads[0] + gpuCounts[0] * PER_BATCH_OVERHEAD_MS;
      for (int g = 1; g < numGPUs; g++)
      {
        float load = gpuLoads[g] + gpuCounts[g] * PER_BATCH_OVERHEAD_MS;
        if (load < minLoad)
        {
          targetGPU = g;
          minLoad = load;
        }
      }
      assignments[b] = targetGPU;
      gpuLoads[targetGPU] += allGPUTimes[targetGPU * n + sorted[b]];
      gpuCounts[targetGPU]++;
    }

    // Build per-GPU result arrays
    int[][] result = new int[numGPUs][];
    for (int g = 0; g < numGPUs; g++)
    {
      result[g] = new int[gpuCounts[g]];
    }

    Span<int> fillIdx = stackalloc int[numGPUs];
    fillIdx.Clear();
    for (int b = 0; b < batchCount; b++)
    {
      int g = assignments[b];
      result[g][fillIdx[g]] = sizes[sorted[b]];
      fillIdx[g]++;
    }

    return result;
  }


  /// <summary>
  /// Computes normalized speed fractions for each GPU based on execution times.
  /// Uses sum of execution times across all batch sizes as the metric (lower = faster).
  /// GPUs with identical device names have their speeds averaged before computing fractions.
  /// </summary>
  /// <param name="executionTimesPerGPU">Execution times [gpu][batchSizeIdx] in milliseconds</param>
  /// <param name="deviceNames">Device name strings for each GPU (e.g., "NVIDIA RTX PRO 6000 Blackwell")</param>
  /// <param name="numGPUs">Number of GPUs to compute fractions for (may be less than total pools)</param>
  /// <returns>Array of fractions per GPU, summing to 1.0. Faster GPUs get higher fractions.</returns>
  public static float[] ComputeSpeedNormalizedFractions(float[][] executionTimesPerGPU, string[] deviceNames, int numGPUs)
  {
    if (executionTimesPerGPU == null || numGPUs <= 0)
    {
      return null;
    }

    // Compute sum of execution times for each GPU (lower = faster)
    float[] totalTimes = new float[numGPUs];
    for (int gpu = 0; gpu < numGPUs; gpu++)
    {
      totalTimes[gpu] = executionTimesPerGPU[gpu].Sum();
    }

    // Average times for GPUs with identical device names (O(n²) but n is tiny, typically < 8)
    if (deviceNames != null && deviceNames.Length >= numGPUs)
    {
      for (int gpu = 0; gpu < numGPUs; gpu++)
      {
        string name = deviceNames[gpu];
        if (name == null)
        {
          continue;
        }

        // Sum times and count for all GPUs with this name
        float sum = 0;
        int count = 0;
        for (int other = 0; other < numGPUs; other++)
        {
          if (string.Equals(name, deviceNames[other], StringComparison.Ordinal))
          {
            sum += executionTimesPerGPU[other].Sum();
            count++;
          }
        }

        // Replace with average if there are multiple identical GPUs
        if (count > 1)
        {
          totalTimes[gpu] = sum / count;
        }
      }
    }

    // Convert to inverse speed and normalize to fractions in a single pass
    float[] fractions = new float[numGPUs];
    float speedSum = 0;
    for (int gpu = 0; gpu < numGPUs; gpu++)
    {
      fractions[gpu] = totalTimes[gpu] > 0 ? 1.0f / totalTimes[gpu] : 0;
      speedSum += fractions[gpu];
    }

    if (speedSum > 0)
    {
      for (int gpu = 0; gpu < numGPUs; gpu++)
      {
        fractions[gpu] /= speedSum;
      }
    }
    else
    {
      // Fallback to equal distribution
      float equalFraction = 1.0f / numGPUs;
      for (int gpu = 0; gpu < numGPUs; gpu++)
      {
        fractions[gpu] = equalFraction;
      }
    }

    return fractions;
  }
}
