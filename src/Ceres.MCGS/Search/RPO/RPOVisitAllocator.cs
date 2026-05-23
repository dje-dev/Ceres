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
using System.Buffers;

#endregion

namespace Ceres.MCGS.Search.RPO;

/// <summary>
/// Apportions a visit budget across children so that the resulting empirical
/// counts approximate the target distribution pi_bar produced by
/// RegularizedPolicyOptimum.Solve.
///
/// This is the classical largest-remainders (Hamilton) apportionment problem.
/// Two equivalent algorithms are provided:
///
///   IterativeLargestDeficit  : one-at-a-time greedy on deficit
///                              d_i = pi_bar_i * (sumN + k) - currentN_i.
///                              Matches the historical CBGPUCT inner loop exactly.
///
///   HamiltonClosedForm       : O(n log n) closed-form floor + fractional-remainder
///                              top-up.  Behaviorally identical to iterative for
///                              fixed pi_bar (up to tie-breaking).
///
/// The allocator is pure: it does not mutate currentN.  The caller is responsible
/// for folding the returned visit deltas into any in-flight counters.
/// </summary>
public static class RPOVisitAllocator
{
  private const int STACKALLOC_MAX = 64;


  /// <summary>
  /// Allocates 'budget' visits across the children.
  /// </summary>
  /// <param name="piBar">Target distribution (length n).  Must sum to (approximately) 1.</param>
  /// <param name="currentN">Current visit counts (length n), already inclusive of any in-flight terms.</param>
  /// <param name="budget">Number of visits to apportion.  Must be greater than or equal to 0.</param>
  /// <param name="visitsAddedOut">Output (length greater than or equal to n): visits added per child.  Caller may pre-zero or rely on the allocator (which only adds positive increments).</param>
  /// <param name="firstStepDeficitsOut">
  /// Optional output (may be empty).  When non-empty, receives the deficit vector
  /// at iteration 0:  pi_bar_i * (sumN + 1) - currentN_i.  Useful for diagnostics.
  /// </param>
  /// <param name="options">Algorithm selection and early-termination flag.</param>
  /// <returns>Number of visits actually placed.  May be less than budget if StopWhenAllOverQuota is true.</returns>
  public static int Allocate(ReadOnlySpan<double> piBar,
                             ReadOnlySpan<double> currentN,
                             int budget,
                             Span<short> visitsAddedOut,
                             Span<double> firstStepDeficitsOut,
                             RPOAllocationOptions options = default)
  {
    if (piBar.Length != currentN.Length)
    {
      throw new ArgumentException("piBar and currentN must have equal length.");
    }
    if (visitsAddedOut.Length < piBar.Length)
    {
      throw new ArgumentException("visitsAddedOut is shorter than piBar.");
    }
    if (!firstStepDeficitsOut.IsEmpty && firstStepDeficitsOut.Length < piBar.Length)
    {
      throw new ArgumentException("firstStepDeficitsOut is shorter than piBar.");
    }
    if (budget < 0)
    {
      throw new ArgumentOutOfRangeException(nameof(budget));
    }

    int n = piBar.Length;
    if (n == 0)
    {
      return 0;
    }

    // Always emit the first-step deficit if requested, even when budget == 0.
    if (!firstStepDeficitsOut.IsEmpty)
    {
      double sumN0 = 0.0;
      for (int i = 0; i < n; i++)
      {
        sumN0 += currentN[i];
      }
      double target0 = sumN0 + 1.0;
      for (int i = 0; i < n; i++)
      {
        firstStepDeficitsOut[i] = piBar[i] * target0 - currentN[i];
      }
    }

    if (budget == 0)
    {
      return 0;
    }

    return options.Algorithm switch
    {
      RPOAllocationAlgorithm.HamiltonClosedForm => AllocateHamilton(piBar, currentN, budget, visitsAddedOut, options.StopWhenAllOverQuota),
      _ => AllocateIterative(piBar, currentN, budget, visitsAddedOut, options.StopWhenAllOverQuota)
    };
  }


  // ----------------------------------------------------------------------------
  // Iterative largest-deficit (default; matches legacy CBGPUCTScoreCalc inner loop)
  // ----------------------------------------------------------------------------

  private static int AllocateIterative(ReadOnlySpan<double> piBar,
                                       ReadOnlySpan<double> currentN,
                                       int budget,
                                       Span<short> visitsAddedOut,
                                       bool stopWhenAllOverQuota)
  {
    int n = piBar.Length;

    double[] rentedRunningN = null;
    Span<double> runningN = n <= STACKALLOC_MAX
      ? stackalloc double[n]
      : (rentedRunningN = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);

    try
    {
      double sumN = 0.0;
      for (int i = 0; i < n; i++)
      {
        runningN[i] = currentN[i];
        sumN += currentN[i];
      }

      int placed = 0;
      while (placed < budget)
      {
        double targetTotal = sumN + 1.0;

        int bestIdx = 0;
        double bestDeficit = piBar[0] * targetTotal - runningN[0];
        for (int i = 1; i < n; i++)
        {
          double d = piBar[i] * targetTotal - runningN[i];
          if (d > bestDeficit)
          {
            bestDeficit = d;
            bestIdx = i;
          }
        }

        if (stopWhenAllOverQuota && bestDeficit < 0.0)
        {
          break;
        }

        visitsAddedOut[bestIdx] += 1;
        runningN[bestIdx] += 1.0;
        sumN += 1.0;
        placed++;
      }
      return placed;
    }
    finally
    {
      if (rentedRunningN != null)
      {
        ArrayPool<double>.Shared.Return(rentedRunningN);
      }
    }
  }


  // ----------------------------------------------------------------------------
  // Hamilton closed-form (largest remainders)
  // ----------------------------------------------------------------------------
  // For fixed pi_bar this produces the same total allocation as the iterative
  // form, up to tie-breaking.  Runs in O(n log n).
  //
  // Algorithm: target_i = pi_bar_i * (sumN_start + budget).  Add to currentN_i
  // a floor delta = max(0, floor(target_i - currentN_i)); the remaining budget
  // goes to the K children with largest fractional remainders.

  private static int AllocateHamilton(ReadOnlySpan<double> piBar,
                                      ReadOnlySpan<double> currentN,
                                      int budget,
                                      Span<short> visitsAddedOut,
                                      bool stopWhenAllOverQuota)
  {
    int n = piBar.Length;

    double[] rentedFrac = null;
    int[] rentedIdx = null;
    Span<double> fractionalRemainder = n <= STACKALLOC_MAX
      ? stackalloc double[n]
      : (rentedFrac = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);
    Span<int> indices = n <= STACKALLOC_MAX
      ? stackalloc int[n]
      : (rentedIdx = ArrayPool<int>.Shared.Rent(n)).AsSpan(0, n);

    try
    {
      double sumN = 0.0;
      for (int i = 0; i < n; i++)
      {
        sumN += currentN[i];
      }
      double totalAfter = sumN + budget;

      // Early-quit if all children would be over their quota at the start.
      if (stopWhenAllOverQuota)
      {
        double targetTotalAtFirst = sumN + 1.0;
        bool anyDeficit = false;
        for (int i = 0; i < n; i++)
        {
          if (piBar[i] * targetTotalAtFirst - currentN[i] >= 0.0)
          {
            anyDeficit = true;
            break;
          }
        }
        if (!anyDeficit)
        {
          return 0;
        }
      }

      int placed = 0;
      for (int i = 0; i < n; i++)
      {
        double rawDelta = piBar[i] * totalAfter - currentN[i];
        // Children already over quota receive zero (cannot give back visits).
        double floorDelta = Math.Floor(Math.Max(0.0, rawDelta));
        // Cap individual allocations to the remaining budget defensively.
        if (floorDelta > budget - placed)
        {
          floorDelta = budget - placed;
        }
        visitsAddedOut[i] += (short)floorDelta;
        placed += (int)floorDelta;
        fractionalRemainder[i] = rawDelta - floorDelta;
        indices[i] = i;
      }

      int remaining = budget - placed;
      if (remaining > 0)
      {
        // Partial selection sort: pick the top `remaining` by fractional remainder.
        // For chess (n <= 64), full sort is fine; for larger n we'd want a heap.
        SortIndicesByFractionalRemainderDescending(indices, fractionalRemainder);

        int upTo = Math.Min(remaining, n);
        for (int k = 0; k < upTo; k++)
        {
          int idx = indices[k];
          // In stop-when-over-quota mode, only allocate to children whose
          // first-step deficit was positive (would have been chosen iteratively).
          if (stopWhenAllOverQuota && fractionalRemainder[idx] < 0.0)
          {
            break;
          }
          visitsAddedOut[idx] += 1;
          placed++;
        }
      }

      return placed;
    }
    finally
    {
      if (rentedFrac != null) ArrayPool<double>.Shared.Return(rentedFrac);
      if (rentedIdx  != null) ArrayPool<int>.Shared.Return(rentedIdx);
    }
  }


  /// <summary>
  /// In-place sort of indices by fractionalRemainder[indices[k]] descending.
  /// Insertion sort - cheap for n up to about 64 (chess branching factor bound).
  /// </summary>
  private static void SortIndicesByFractionalRemainderDescending(Span<int> indices, ReadOnlySpan<double> fractionalRemainder)
  {
    int n = indices.Length;
    for (int i = 1; i < n; i++)
    {
      int key = indices[i];
      double keyVal = fractionalRemainder[key];
      int j = i - 1;
      while (j >= 0 && fractionalRemainder[indices[j]] < keyVal)
      {
        indices[j + 1] = indices[j];
        j--;
      }
      indices[j + 1] = key;
    }
  }
}
