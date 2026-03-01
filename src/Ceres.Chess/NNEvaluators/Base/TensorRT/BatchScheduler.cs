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

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Optimal batch scheduling for single or multi-GPU inference.
///
/// Minimizes total estimated time (makespan + multi-GPU penalty) using
/// two-phase dynamic programming:
///   Phase 1: Per-GPU DP computes min-time batch sequences for any position count.
///   Phase 2: Cross-GPU DP with active-GPU-count tracking finds optimal position
///            distribution, balancing makespan reduction against multi-GPU penalty.
///
/// Handles heterogeneous GPUs with different timing profiles and accounts for
/// per-batch pipeline overlap credits on consecutive engine executions.
///
/// Uses stackalloc buffers for zero managed allocation.
/// </summary>
public static class BatchScheduler
{
  /// <summary>
  /// Diagnostic flag: when true, prints per-device execution plans and summary to console.
  /// </summary>
  public const bool VERBOSE_DETAILS = false;

  /// <summary>
  /// Per-additional-batch time adjustment in milliseconds. Negative values represent a pipeline
  /// credit: when the EnginePool runs consecutive batches, H2D of batch N+1 overlaps with
  /// compute of batch N, making the marginal cost of each additional batch less than its
  /// standalone execution time. Applied as max(0, numBatches - 1) * PER_BATCH_OVERHEAD_MS,
  /// so the first batch always pays its full engine time with no adjustment.
  /// </summary>
  public const float PER_BATCH_OVERHEAD_MS = 0.5f;

  /// Estimated per-GPU coordination overhead in milliseconds.
  /// Accounts for CUDA context switching (cudaSetDevice + cudaDeviceSynchronize)
  /// on dispatch and collection, plus Parallel.For thread scheduling.
  /// Used to decide optimal GPU count: multi-GPU only when engine time savings exceed this cost.
  public const float PER_GPU_FIXED_COST_MS = 0f;

  /// <summary>
  /// Result of the scheduling algorithm.
  /// </summary>
  public record ScheduleResult(
    int[][] ExecutionPlan,  // Outer: GPU index, Inner: batch sizes to execute in order
    int MaxTimeMS,          // Maximum execution time across all GPUs (ceiling of MakespanMs)
    int TotalPositions,     // Total positions including padding
    int Padding             // Wasted padding positions
  )
  {
    /// <summary>
    /// Precise makespan in milliseconds (max GPU time, not including multi-GPU penalty).
    /// </summary>
    public float MakespanMs { get; init; }

    /// <summary>
    /// Multi-GPU penalty in milliseconds: MULTI_GPU_PENALTY_MS * (numGPUsUsed - 1).
    /// </summary>
    public float MultiGpuPenaltyMs { get; init; }

    /// <summary>
    /// Total estimated time: MakespanMs + MultiGpuPenaltyMs.
    /// </summary>
    public float TotalEstimatedMs { get; init; }
  }


  /// <summary>
  /// Schedules batch execution across multiple GPUs optimally, accounting for
  /// multi-GPU coordination penalty.
  ///
  /// Algorithm overview:
  ///   Phase 1 (Per-GPU DP): For each GPU g and position count k in [0..N], compute
  ///     gpuCost[g][k] = minimum adjusted cost to cover >= k positions, where
  ///     adjusted cost = sum(timing_e - overlapCredit) per engine execution.
  ///     Actual time = gpuCost[k] + overlapCredit for k > 0, else 0.
  ///     Also records gpuChoice[g][k] and gpuBatched[g][k] for backtrace.
  ///
  ///   Phase 2 (Cross-GPU DP with GPU-count tracking):
  ///     dp[u][r] = min makespan distributing r positions across GPUs [g..G-1]
  ///     using exactly u active GPUs. Processed right-to-left with ping-pong.
  ///     For each state (u, r), considers GPU g idle (inherit dp[u][r]) or
  ///     active (binary search for optimal n in dp[u-1][r-n]).
  ///     Final selection: argmin_u (dp[u][N] + MULTI_GPU_PENALTY_MS * (u-1)).
  ///
  ///   Phase 3 (Backtrace): Recovers per-GPU position counts and batch sequences.
  ///
  /// Complexity: O(G * N * M) for Phase 1, O(G^2 * N * log N) for Phase 2.
  /// </summary>
  public static ScheduleResult Schedule(int numGPUs,
                                        int[] engineSizes, float[][] executionTimesPerGPU,
                                        int targetBatch, int preferredGPU = 0)
  {
    if (targetBatch <= 0 || numGPUs <= 0)
    {
      int[][] emptyPlan = new int[Math.Max(numGPUs, 0)][];
      for (int i = 0; i < emptyPlan.Length; i++)
      {
        emptyPlan[i] = Array.Empty<int>();
      }
      return new ScheduleResult(emptyPlan, 0, 0, 0)
      {
        MakespanMs = 0f,
        MultiGpuPenaltyMs = 0f,
        TotalEstimatedMs = 0f
      };
    }

    int M = engineSizes.Length;
    int N = targetBatch;
    int G = numGPUs;
    float overlapCredit = -PER_BATCH_OVERHEAD_MS; // positive value (0.5ms)
    int stride = N + 1;

    // -------------------------------------------------------------------
    // Phase 1: Per-GPU cost DP
    //
    // Flat layout: gpuCost[g * stride + k], gpuChoice[g * stride + k],
    //              gpuBatched[g * stride + k].
    //
    // gpuCost[g][k] = min sum of (timing_e - overlapCredit) over all engine
    //   executions needed to cover >= k positions on GPU g.
    // Actual time for k positions: gpuCost[g][k] + overlapCredit (k > 0).
    //
    // gpuBatched[g][k] = total positions executed (>= k due to padding).
    // When costs tie, prefer the engine choice with less padding.
    // -------------------------------------------------------------------
    int gpuBufSize = G * stride;
    Span<float> gpuCost = stackalloc float[gpuBufSize];
    Span<int> gpuChoice = stackalloc int[gpuBufSize];
    Span<int> gpuBatched = stackalloc int[gpuBufSize];

    for (int g = 0; g < G; g++)
    {
      int off = g * stride;
      gpuCost[off] = 0f;
      gpuChoice[off] = 0;
      gpuBatched[off] = 0;
      float[] timings = executionTimesPerGPU[g];

      for (int k = 1; k <= N; k++)
      {
        float best = float.MaxValue;
        int bestE = 0;
        int bestB = int.MaxValue;

        for (int e = 0; e < M; e++)
        {
          int prev = k - engineSizes[e];
          if (prev < 0)
          {
            prev = 0;
          }
          float c = gpuCost[off + prev] + timings[e] - overlapCredit;
          int b = gpuBatched[off + prev] + engineSizes[e];
          if (c < best - 0.001f || (c <= best + 0.001f && b < bestB))
          {
            best = c;
            bestE = e;
            bestB = b;
          }
        }

        gpuCost[off + k] = best;
        gpuChoice[off + k] = bestE;
        gpuBatched[off + k] = bestB;
      }
    }

    // -------------------------------------------------------------------
    // Phase 2: Cross-GPU distribution DP with active-GPU-count dimension
    //
    // dp[u][r] = min makespan for distributing r positions across GPUs
    //   [g..G-1] using exactly u of them active (assigned n > 0).
    //
    // Flat layout: dp[u * stride + r]. Ping-pong buffers sized (G+1)*stride.
    //
    // For each GPU g (right to left), two options per state (u, r):
    //   Idle:   dpCur[u][r] = dpNext[u][r]
    //   Active: dpCur[u][r] = min_n>0 max(ActualTime(g,n), dpNext[u-1][r-n])
    // Take the option minimizing makespan, with padding as tiebreaker.
    //
    // distChoice stores the chosen n for backtrace.
    //
    // Final: best u = argmin_u (dp[u][N] + MULTI_GPU_PENALTY_MS * (u-1)).
    // -------------------------------------------------------------------
    int uCount = G + 1; // u ranges 0..G
    int dpLayerSize = uCount * stride;
    Span<float> dpBufA = stackalloc float[dpLayerSize];
    Span<float> dpBufB = stackalloc float[dpLayerSize];
    Span<int> dpPadBufA = stackalloc int[dpLayerSize];
    Span<int> dpPadBufB = stackalloc int[dpLayerSize];

    int distSize = Math.Max(1, G - 1) * uCount * stride;
    Span<int> distChoice = stackalloc int[distSize];

    Span<float> dpNext = dpBufA;
    Span<float> dpCur = dpBufB;
    Span<int> dpPadNext = dpPadBufA;
    Span<int> dpPadCur = dpPadBufB;

    // Initialize dpNext to infinity
    for (int i = 0; i < dpLayerSize; i++)
    {
      dpNext[i] = float.MaxValue;
      dpPadNext[i] = int.MaxValue;
    }

    // Base case: last GPU (g = G-1)
    // u=0, r=0: no GPUs active, 0 positions
    dpNext[0] = 0f;
    dpPadNext[0] = 0;

    // u=1: last GPU active, covers r positions
    int lastOff = (G - 1) * stride;
    for (int r = 1; r <= N; r++)
    {
      dpNext[stride + r] = gpuCost[lastOff + r] + overlapCredit;
      dpPadNext[stride + r] = gpuBatched[lastOff + r];
    }

    // DP from GPU G-2 down to GPU 0
    for (int g = G - 2; g >= 0; g--)
    {
      int gOff = g * stride;
      int maxU = G - g; // max active GPUs among [g..G-1]
      int distBase = g * uCount * stride;

      // Initialize dpCur to infinity
      for (int i = 0; i < dpLayerSize; i++)
      {
        dpCur[i] = float.MaxValue;
        dpPadCur[i] = int.MaxValue;
      }

      for (int u = 0; u <= maxU; u++)
      {
        int uOff = u * stride;

        for (int r = 0; r <= N; r++)
        {
          float bestMs = float.MaxValue;
          int bestN = 0;
          int bestPad = int.MaxValue;

          // Option A: GPU g is idle
          float idleMs = dpNext[uOff + r];
          if (idleMs < float.MaxValue)
          {
            bestMs = idleMs;
            bestN = 0;
            bestPad = dpPadNext[uOff + r];
          }

          // Option B: GPU g is active (requires u >= 1 and r >= 1)
          if (u >= 1 && r >= 1)
          {
            int prevUOff = (u - 1) * stride;

            // Binary search for crossing: smallest n where f(n) >= h(n)
            // f(n) = ActualTime(g, n) = gpuCost[g][n] + overlapCredit
            // h(n) = dpNext[u-1][r-n]
            int lo = 1, hi = r;
            while (lo < hi)
            {
              int mid = lo + (hi - lo) / 2;
              float fMid = gpuCost[gOff + mid] + overlapCredit;
              float hMid = dpNext[prevUOff + r - mid];
              if (fMid < hMid)
              {
                lo = mid + 1;
              }
              else
              {
                hi = mid;
              }
            }

            // Check crossing region for best makespan
            float activeBestMs = float.MaxValue;
            int activeBestN = 1;
            int checkLo = Math.Max(1, lo - 1);
            int checkHi = Math.Min(r, lo + 1);
            for (int n = checkLo; n <= checkHi; n++)
            {
              float fn = gpuCost[gOff + n] + overlapCredit;
              float hn = dpNext[prevUOff + r - n];
              if (hn >= float.MaxValue)
              {
                continue;
              }
              float ms = fn > hn ? fn : hn;
              if (ms < activeBestMs)
              {
                activeBestMs = ms;
                activeBestN = n;
              }
            }

            // Padding tie-breaking: bounded scan within valid makespan range
            if (activeBestMs < float.MaxValue)
            {
              float threshold = activeBestMs + 0.001f;

              // Smallest n (>=1) where dpNext[u-1][r-n] <= threshold
              int nMinH;
              {
                int sLo = 1, sHi = r;
                while (sLo < sHi)
                {
                  int mid = sLo + (sHi - sLo) / 2;
                  float hv = dpNext[prevUOff + r - mid];
                  if (hv > threshold)
                  {
                    sLo = mid + 1;
                  }
                  else
                  {
                    sHi = mid;
                  }
                }
                nMinH = sLo;
              }

              // Largest n (<=r) where ActualTime(g,n) <= threshold
              int nMaxF;
              {
                int sLo = 1, sHi = r;
                while (sLo < sHi)
                {
                  int mid = sLo + (sHi - sLo + 1) / 2;
                  float fv = gpuCost[gOff + mid] + overlapCredit;
                  if (fv <= threshold)
                  {
                    sLo = mid;
                  }
                  else
                  {
                    sHi = mid - 1;
                  }
                }
                nMaxF = sLo;
              }

              int activeBestPad = gpuBatched[gOff + activeBestN] + dpPadNext[prevUOff + r - activeBestN];
              for (int n = nMinH; n <= nMaxF; n++)
              {
                int hPad = dpPadNext[prevUOff + r - n];
                if (hPad >= int.MaxValue)
                {
                  continue;
                }
                int pad = gpuBatched[gOff + n] + hPad;
                if (pad < activeBestPad)
                {
                  activeBestPad = pad;
                  activeBestN = n;
                }
              }

              // Compare active vs idle
              if (activeBestMs < bestMs - 0.001f)
              {
                bestMs = activeBestMs;
                bestN = activeBestN;
                bestPad = activeBestPad;
              }
              else if (activeBestMs <= bestMs + 0.001f && activeBestPad < bestPad)
              {
                bestN = activeBestN;
                bestPad = activeBestPad;
              }
            }
          }

          dpCur[uOff + r] = bestMs;
          dpPadCur[uOff + r] = bestPad;
          distChoice[distBase + uOff + r] = bestN;
        }
      }

      // Swap buffers
      Span<float> tmpF = dpNext; dpNext = dpCur; dpCur = tmpF;
      Span<int> tmpI = dpPadNext; dpPadNext = dpPadCur; dpPadCur = tmpI;
    }

    // Find optimal u: minimize makespan + penalty
    float bestTotal = float.MaxValue;
    int bestU = 1;
    for (int u = 1; u <= G; u++)
    {
      float ms = dpNext[u * stride + N];
      if (ms >= float.MaxValue)
      {
        continue;
      }
      float total = ms + PER_GPU_FIXED_COST_MS * (u - 1);
      if (total < bestTotal - 0.001f ||
          (total <= bestTotal + 0.001f && dpPadNext[u * stride + N] < dpPadNext[bestU * stride + N]))
      {
        bestTotal = total;
        bestU = u;
      }
    }

    float makespan = dpNext[bestU * stride + N];
    float penalty = PER_GPU_FIXED_COST_MS * Math.Max(0, bestU - 1);

    // -------------------------------------------------------------------
    // Phase 3: Backtrace
    // -------------------------------------------------------------------
    Span<int> assigned = stackalloc int[G];
    int remU = bestU;
    int remR = N;
    for (int g = 0; g < G - 1; g++)
    {
      int n = distChoice[g * uCount * stride + remU * stride + remR];
      assigned[g] = n;
      if (n > 0)
      {
        remU--;
      }
      remR -= n;
    }
    assigned[G - 1] = remR;

    // Reconstruct per-GPU batch sequences from gpuChoice tables
    int[][] plan = new int[G][];
    int totalPositions = 0;
    int devicesUsed = 0;

    for (int g = 0; g < G; g++)
    {
      int items = assigned[g];
      if (items <= 0)
      {
        plan[g] = Array.Empty<int>();
        continue;
      }

      devicesUsed++;
      int gOff = g * stride;

      // Count sequence length
      int seqLen = 0;
      for (int k = items; k > 0;)
      {
        seqLen++;
        k = Math.Max(0, k - engineSizes[gpuChoice[gOff + k]]);
      }

      // Fill sequence and compute total
      int[] seq = new int[seqLen];
      int batched = 0;
      int idx = 0;
      for (int k = items; k > 0;)
      {
        int e = gpuChoice[gOff + k];
        seq[idx++] = engineSizes[e];
        batched += engineSizes[e];
        k = Math.Max(0, k - engineSizes[e]);
      }

      plan[g] = seq;
      totalPositions += batched;

      if (VERBOSE_DETAILS)
      {
        float time = gpuCost[gOff + items] + overlapCredit;
        int padding = batched - items;
        Console.WriteLine($"  OPTIMIZED_PLAN [device {g}]: {items} -> [{string.Join(", ", seq)}] total={batched} padding={padding} time={time:F1}ms");
      }
    }

    if (VERBOSE_DETAILS)
    {
      Console.WriteLine($"  BATCH SIZE {N} makespan={makespan:F1}ms penalty={penalty:F1}ms total={makespan + penalty:F1}ms with {devicesUsed} GPUs");
    }

    return new ScheduleResult(plan, (int)Math.Ceiling(makespan), totalPositions, totalPositions - N)
    {
      MakespanMs = makespan,
      MultiGpuPenaltyMs = penalty,
      TotalEstimatedMs = makespan + penalty
    };
  }


  /// <summary>
  /// Single-GPU scheduling: finds the batch sequence minimizing total inference time.
  /// Uses DP over position counts with pipeline overlap credits.
  /// When concurrent=true, also evaluates concurrent cost (pairs on separate CUDA streams)
  /// for diagnostic output. The plan itself is unchanged; the caller controls execution strategy.
  /// Only managed allocation is the returned result array; working buffers use stackalloc.
  /// </summary>
  public static int[] ScheduleSingleGPU(ReadOnlySpan<int> engineSizes, ReadOnlySpan<float> executionTimes,
                                         int targetBatch, int deviceIndex = -1, bool concurrent = false)
  {
    if (targetBatch <= 0 || engineSizes.Length == 0)
    {
      return Array.Empty<int>();
    }

    int M = engineSizes.Length;
    int N = targetBatch;
    float overlapCredit = -PER_BATCH_OVERHEAD_MS;

    Span<float> cost = stackalloc float[N + 1];
    Span<int> choice = stackalloc int[N + 1];

    for (int k = 1; k <= N; k++)
    {
      float best = float.MaxValue;
      int bestE = 0;

      for (int e = 0; e < M; e++)
      {
        int prev = k - engineSizes[e];
        if (prev < 0)
        {
          prev = 0;
        }
        float c = cost[prev] + executionTimes[e] - overlapCredit;
        if (c < best)
        {
          best = c;
          bestE = e;
        }
      }

      cost[k] = best;
      choice[k] = bestE;
    }

    // Count sequence length, then fill
    int seqLen = 0;
    for (int k = N; k > 0;)
    {
      seqLen++;
      k = Math.Max(0, k - engineSizes[choice[k]]);
    }

    int[] result = new int[seqLen];
    int idx = 0;
    for (int k = N; k > 0;)
    {
      int e = choice[k];
      result[idx++] = engineSizes[e];
      k = Math.Max(0, k - engineSizes[e]);
    }

    if (VERBOSE_DETAILS)
    {
      float time = cost[N] + overlapCredit;
      int batched = 0;
      foreach (int b in result)
      {
        batched += b;
      }
      string label = deviceIndex >= 0 ? $"device {deviceIndex}" : "single GPU";
      Console.WriteLine($"  OPTIMIZED_PLAN [{label}]: {N} -> [{string.Join(", ", result)}] total={batched} padding={batched - N} time={time:F1}ms");
    }

    // When concurrent execution is enabled, evaluate concurrent cost for diagnostics.
    // The plan itself is unchanged; the caller controls execution via NUM_COMPUTE_STREAMS.
    if (concurrent && result.Length > 1 && VERBOSE_DETAILS)
    {
      float seqCost = ComputeSequentialCost(result, engineSizes, executionTimes);
      float concCost = ComputeConcurrentCost(result, engineSizes, executionTimes);
      string concLabel = deviceIndex >= 0 ? $"device {deviceIndex}" : "single GPU";
      Console.WriteLine($"  CONCURRENT_EVAL [{concLabel}]: seq={seqCost:F2}ms conc={concCost:F2}ms plan=[{string.Join(", ", result)}]");
    }

    return result;
  }


  /// <summary>
  /// Per-concurrent-pair overhead in milliseconds. Accounts for launching both H2D + compute
  /// on two streams, syncing both, then D2H. Less than 2x PER_BATCH_OVERHEAD_MS because
  /// the two launches happen in quick succession.
  /// </summary>
  public const float PER_CONCURRENT_PAIR_OVERHEAD_MS = 0.3f;


  /// <summary>
  /// Computes the estimated sequential cost for a batch plan.
  /// </summary>
  private static float ComputeSequentialCost(int[] plan, ReadOnlySpan<int> engineSizes, ReadOnlySpan<float> executionTimes)
  {
    float total = 0;
    for (int i = 0; i < plan.Length; i++)
    {
      total += LookupTime(plan[i], engineSizes, executionTimes);
    }
    if (plan.Length > 1)
    {
      total += (plan.Length - 1) * PER_BATCH_OVERHEAD_MS;
    }
    return total;
  }


  /// <summary>
  /// Computes the estimated concurrent cost for a batch plan (pairs execute concurrently).
  /// </summary>
  private static float ComputeConcurrentCost(int[] plan, ReadOnlySpan<int> engineSizes, ReadOnlySpan<float> executionTimes)
  {
    float total = 0;
    int numPairs = 0;

    for (int i = 0; i < plan.Length; i += 2)
    {
      float timeA = LookupTime(plan[i], engineSizes, executionTimes);
      if (i + 1 < plan.Length)
      {
        // Pair: concurrent execution
        float timeB = LookupTime(plan[i + 1], engineSizes, executionTimes);
        total += Math.Max(timeA, timeB);
        numPairs++;
      }
      else
      {
        // Odd batch runs alone
        total += timeA;
      }
    }

    total += numPairs * PER_CONCURRENT_PAIR_OVERHEAD_MS;
    return total;
  }


  /// <summary>
  /// Looks up the execution time for a given batch size.
  /// </summary>
  private static float LookupTime(int batchSize, ReadOnlySpan<int> engineSizes, ReadOnlySpan<float> executionTimes)
  {
    for (int i = 0; i < engineSizes.Length; i++)
    {
      if (engineSizes[i] == batchSize)
      {
        return executionTimes[i];
      }
    }
    return executionTimes[0]; // Fallback (shouldn't happen)
  }


  /// <summary>
  /// Computes normalized speed fractions for each GPU based on execution times.
  /// Uses sum of execution times across all batch sizes as the metric (lower = faster).
  /// GPUs with identical device names have their speeds averaged before computing fractions.
  /// </summary>
  public static float[] ComputeSpeedNormalizedFractions(float[][] executionTimesPerGPU, string[] deviceNames, int numGPUs)
  {
    if (executionTimesPerGPU == null || numGPUs <= 0)
    {
      return null;
    }

    Span<float> totalTimes = stackalloc float[numGPUs];
    for (int gpu = 0; gpu < numGPUs; gpu++)
    {
      float s = 0;
      float[] times = executionTimesPerGPU[gpu];
      for (int i = 0; i < times.Length; i++)
      {
        s += times[i];
      }
      totalTimes[gpu] = s;
    }

    if (deviceNames != null && deviceNames.Length >= numGPUs)
    {
      for (int gpu = 0; gpu < numGPUs; gpu++)
      {
        string name = deviceNames[gpu];
        if (name == null)
        {
          continue;
        }

        float sum = 0;
        int count = 0;
        for (int other = 0; other < numGPUs; other++)
        {
          if (string.Equals(name, deviceNames[other], StringComparison.Ordinal))
          {
            sum += totalTimes[other];
            count++;
          }
        }

        if (count > 1)
        {
          totalTimes[gpu] = sum / count;
        }
      }
    }

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
      float equalFraction = 1.0f / numGPUs;
      for (int gpu = 0; gpu < numGPUs; gpu++)
      {
        fractions[gpu] = equalFraction;
      }
    }

    return fractions;
  }
}
