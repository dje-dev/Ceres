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
  /// Schedules batch execution across GPUs optimally.
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
      return new ScheduleResult(Enumerable.Range(0, numGPUs).Select(_ => Array.Empty<int>()).ToArray(), 0, 0, 0);
    }

    // Clamp preferredGPU to valid range
    preferredGPU = Math.Clamp(preferredGPU, 0, numGPUs - 1);

    // Build engine info using preferred GPU's times for candidate generation
    (int Size, float TimeMS)[] engines = engineSizes
      .Select((s, i) => (Size: s, TimeMS: executionTimesPerGPU[preferredGPU][i]))
      .OrderByDescending(e => e.Size)
      .ToArray();

    // Build per-GPU lookup: sizeToTime[gpu][size] = time
    Dictionary<int, float>[] sizeToTimePerGPU = new Dictionary<int, float>[numGPUs];
    for (int gpu = 0; gpu < numGPUs; gpu++)
    {
      sizeToTimePerGPU[gpu] = engineSizes
        .Select((s, i) => (Size: s, Time: executionTimesPerGPU[gpu][i]))
        .ToDictionary(x => x.Size, x => x.Time);
    }

    // Generate candidate batch combinations using multiple strategies
    List<int[]> candidates = GenerateCandidates(engines, targetBatch, numGPUs);

    if (candidates.Count == 0)
    {
      // Fallback: single largest batch
      int[] fallback = [engines[0].Size];
      candidates.Add(fallback);
    }

    // Evaluate each candidate
    List<(int[][] Plan, float MaxTime, int TotalPos)> evaluated = [];

    foreach (int[] batches in candidates)
    {
      int[][] plan = AssignToGPUs(batches, sizeToTimePerGPU, numGPUs, preferredGPU);
      float maxTime = ComputeMaxTime(plan, sizeToTimePerGPU);
      int totalPos = batches.Sum();
      evaluated.Add((plan, maxTime, totalPos));
    }

    // Find minimum maxTime
    float minMaxTime = evaluated.Min(e => e.MaxTime);

    // Filter to within 3% of optimal, then pick minimum padding
    (int[][] Plan, float MaxTime, int TotalPos) best = evaluated
      .Where(e => e.MaxTime <= minMaxTime * 1.03f)
      .OrderBy(e => e.TotalPos)
      .ThenBy(e => e.MaxTime)
      .First();

    return new ScheduleResult(best.Plan, (int)Math.Ceiling(best.MaxTime), best.TotalPos, best.TotalPos - targetBatch);
  }


  /// <summary>
  /// Simplified single-GPU scheduling that returns batch sequence.
  /// Accounts for per-batch overhead when comparing plans.
  /// </summary>
  public static int[] ScheduleSingleGPU(int[] engineSizes, float[] executionTimes, int targetBatch)
  {
    if (targetBatch <= 0)
    {
      return [];
    }

    // Build engine info sorted by size descending
    (int Size, float TimeMS)[] engines = engineSizes
      .Select((s, i) => (Size: s, TimeMS: executionTimes[i]))
      .OrderByDescending(e => e.Size)
      .ToArray();

    Dictionary<int, float> sizeToTime = engines.ToDictionary(e => e.Size, e => e.TimeMS);

    // Generate candidates
    List<int[]> candidates = GenerateCandidates(engines, targetBatch, numGPUs: 1);

    if (candidates.Count == 0)
    {
      return [engines[0].Size];
    }

    // For single GPU, optimize for: minimal time (including per-batch overhead), then minimal padding
    float minTime = float.MaxValue;
    int[] bestPlan = null;
    int bestPadding = int.MaxValue;

    foreach (int[] batches in candidates)
    {
      // Include per-batch overhead for each inference call
      float totalTime = batches.Sum(b => sizeToTime[b]) + batches.Length * PER_BATCH_OVERHEAD_MS;
      int totalPos = batches.Sum();
      int padding = totalPos - targetBatch;

      // Within 3% of best time, prefer less padding
      if (totalTime < minTime * 0.97f || (totalTime <= minTime * 1.03f && padding < bestPadding))
      {
        if (totalTime < minTime)
        {
          minTime = totalTime;
        }
        bestPlan = batches;
        bestPadding = padding;
      }
    }

    return bestPlan ?? [engines[0].Size];
  }

  /// <summary>
  /// Compute max execution time across all GPUs using per-GPU times.
  /// </summary>
  static float ComputeMaxTime(int[][] plan, Dictionary<int, float>[] sizeToTimePerGPU)
  {
    float maxTime = 0;
    for (int gpu = 0; gpu < plan.Length; gpu++)
    {
      float gpuTime = plan[gpu].Sum(b => sizeToTimePerGPU[gpu][b]);
      maxTime = Math.Max(maxTime, gpuTime);
    }
    return maxTime;
  }

  /// <summary>
  /// Generate candidate batch combinations using multiple strategies.
  /// </summary>
  static List<int[]> GenerateCandidates((int Size, float TimeMS)[] engines, int target, int numGPUs)
  {
    HashSet<string> seen = [];
    List<int[]> result = [];

    // Strategy 1: Greedy largest-first (minimal batches)
    AddCandidate(GreedyLargest(engines, target), seen, result);

    // Strategy 2: Try to find exact k batches for k = 1 to numGPUs * 6
    for (int k = 1; k <= Math.Min(numGPUs * 6, 40); k++)
    {
      int[] combo = FindKBatches(engines, target, k);
      if (combo != null)
      {
        AddCandidate(combo, seen, result);
      }
    }

    // Strategy 3: Try uniform batch sizes (good for load balancing)
    foreach ((int Size, float TimeMS) engine in engines)
    {
      int count = (target + engine.Size - 1) / engine.Size;
      if (count <= numGPUs * 4)
      {
        AddCandidate(Enumerable.Repeat(engine.Size, count).ToArray(), seen, result);
      }
    }

    // Strategy 4: Exhaustive search for small targets
    int maxEngineSize = engines.Max(e => e.Size);
    if (target <= maxEngineSize * 8)
    {
      ExhaustiveSearch(engines, target, [], 0, seen, result, maxDepth: 20);
    }

    // Strategy 5: Mixed strategies - try combinations of 2 different sizes
    for (int i = 0; i < engines.Length; i++)
    {
      for (int j = i; j < engines.Length; j++)
      {
        int[] mixed = FindMixedBatches(engines[i].Size, engines[j].Size, target);
        if (mixed != null)
        {
          AddCandidate(mixed, seen, result);
        }
      }
    }

    return result;
  }

  static void AddCandidate(int[] batches, HashSet<string> seen, List<int[]> result)
  {
    if (batches == null || batches.Length == 0) return;
    string key = string.Join(",", batches.OrderByDescending(x => x));
    if (seen.Add(key))
    {
      result.Add(batches);
    }
  }

  /// <summary>
  /// Greedy: use largest engines first until target is met.
  /// </summary>
  static int[] GreedyLargest((int Size, float TimeMS)[] engines, int target)
  {
    List<int> batches = [];
    int remaining = target;

    while (remaining > 0)
    {
      (int Size, float TimeMS) chosen = engines.FirstOrDefault(e => e.Size <= remaining);
      if (chosen == default)
      {
        chosen = engines[^1]; // Smallest engine
      }
      batches.Add(chosen.Size);
      remaining -= chosen.Size;
    }

    return [.. batches];
  }

  /// <summary>
  /// Find exactly k batches that cover target with minimal excess.
  /// </summary>
  static int[] FindKBatches((int Size, float TimeMS)[] engines, int target, int k)
  {
    if (k <= 0) return null;

    int avgSize = (target + k - 1) / k;

    (int Size, float TimeMS) baseEngine = engines
      .Where(e => e.Size >= avgSize)
      .OrderBy(e => e.Size)
      .FirstOrDefault();

    if (baseEngine == default)
    {
      baseEngine = engines[0]; // Largest
    }

    List<int> batches = Enumerable.Repeat(baseEngine.Size, k).ToList();
    int sum = baseEngine.Size * k;

    if (sum >= target)
    {
      // Try to reduce padding by using smaller engines where possible
      for (int i = 0; i < k && sum > target; i++)
      {
        foreach ((int Size, float TimeMS) smaller in engines.Where(e => e.Size < batches[i]).OrderByDescending(e => e.Size))
        {
          int newSum = sum - batches[i] + smaller.Size;
          if (newSum >= target)
          {
            sum = newSum;
            batches[i] = smaller.Size;
            break;
          }
        }
      }
      return [.. batches];
    }

    // Need to use larger engines for some slots
    for (int i = 0; i < k && sum < target; i++)
    {
      foreach ((int Size, float TimeMS) larger in engines.Where(e => e.Size > batches[i]).OrderBy(e => e.Size))
      {
        int newSum = sum - batches[i] + larger.Size;
        sum = newSum;
        batches[i] = larger.Size;
        if (sum >= target) break;
      }
      if (sum >= target) break;
    }

    return sum >= target ? [.. batches] : null;
  }

  /// <summary>
  /// Find a combination using exactly two engine sizes.
  /// </summary>
  static int[] FindMixedBatches(int size1, int size2, int target)
  {
    int bestExcess = int.MaxValue;
    int[] best = null;

    int maxA = (target / size1) + 2;
    for (int a = 0; a <= maxA; a++)
    {
      int remaining = target - a * size1;
      if (remaining <= 0)
      {
        int excess = a * size1 - target;
        if (excess < bestExcess)
        {
          bestExcess = excess;
          best = Enumerable.Repeat(size1, a).ToArray();
        }
        break;
      }

      int b = (remaining + size2 - 1) / size2;
      int total = a * size1 + b * size2;
      int excess2 = total - target;

      if (excess2 < bestExcess && a + b <= 50)
      {
        bestExcess = excess2;
        best = Enumerable.Repeat(size1, a).Concat(Enumerable.Repeat(size2, b)).ToArray();
      }
    }

    return best;
  }

  /// <summary>
  /// Exhaustive search for small targets.
  /// </summary>
  static void ExhaustiveSearch((int Size, float TimeMS)[] engines,
                               int target, List<int> current, int startIdx,
                               HashSet<string> seen, List<int[]> result, int maxDepth)
  {
    int sum = current.Sum();
    if (sum >= target)
    {
      int maxPadding = engines.Max(e => e.Size) * 2;
      if (sum - target <= maxPadding)
      {
        AddCandidate([.. current], seen, result);
      }
      return;
    }

    if (current.Count >= maxDepth) return;

    for (int i = startIdx; i < engines.Length; i++)
    {
      current.Add(engines[i].Size);
      ExhaustiveSearch(engines, target, current, i, seen, result, maxDepth);
      current.RemoveAt(current.Count - 1);
    }
  }

  /// <summary>
  /// Assign batches to GPUs. Longest-executing batches go to preferredGPU first,
  /// then remaining batches distributed via LPT to minimize makespan.
  /// </summary>
  static int[][] AssignToGPUs(int[] batches, Dictionary<int, float>[] sizeToTimePerGPU, int numGPUs, int preferredGPU)
  {
    // Sort batches by execution time on preferred GPU (descending)
    List<int> sorted = [.. batches.OrderByDescending(b => sizeToTimePerGPU[preferredGPU][b])];

    float[] gpuLoads = new float[numGPUs];
    List<int>[] gpuBatches = new List<int>[numGPUs];
    for (int i = 0; i < numGPUs; i++)
    {
      gpuBatches[i] = [];
    }

    // Assign first batch (longest) to preferred GPU
    bool firstBatch = true;

    foreach (int batch in sorted)
    {
      int targetGPU;

      if (firstBatch)
      {
        targetGPU = preferredGPU;
        firstBatch = false;
      }
      else
      {
        // LPT: assign to GPU with minimum current load
        targetGPU = 0;
        float minLoad = gpuLoads[0];
        for (int i = 1; i < numGPUs; i++)
        {
          if (gpuLoads[i] < minLoad)
          {
            targetGPU = i;
            minLoad = gpuLoads[i];
          }
        }
      }

      gpuBatches[targetGPU].Add(batch);
      gpuLoads[targetGPU] += sizeToTimePerGPU[targetGPU][batch];
    }

    return gpuBatches.Select(g => g.ToArray()).ToArray();
  }
}
