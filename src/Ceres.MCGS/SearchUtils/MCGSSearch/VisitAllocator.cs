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

namespace Ceres.MCGS.Search;

public static class VisitAllocator
{

  /// <summary>
  /// Rearranges the elements in the specified span to ensure any empty slots starting at a specified index
  /// appear as leftmost as possible.
  /// </summary>
  /// <param name="visits">A span of integers representing the elements to be rearranged. The span is modified in place.</param>
  /// <param name="numAlreadyUsed">The number of elements at the beginning of the span that are already used and should remain unchanged.</param>
  static void MakeContiguous(Span<int> visits, int numAlreadyUsed)
  {
    if (numAlreadyUsed < 0 || numAlreadyUsed > visits.Length)
    {
      throw new ArgumentOutOfRangeException(nameof(numAlreadyUsed));
    }

    int emptySlotWriteIndex = numAlreadyUsed;
    for (int readIndex = numAlreadyUsed; readIndex < visits.Length; readIndex++)
    {
      if (visits[readIndex] != 0)
      {
        if (readIndex != emptySlotWriteIndex)
        {
          visits[emptySlotWriteIndex] = visits[readIndex];
        }
        emptySlotWriteIndex++;
      }
    }

    // Fill the remaining part of the span with 0s
    for (int i = emptySlotWriteIndex; i < visits.Length; i++)
    {
      visits[i] = 0;
    }
  }


  /// <summary>
  /// Allocate exactly numVisitsToAllocate fresh visits so that,
  /// once they are added element-wise to currentDistrib,
  /// the resulting fraction vector tracks optimalDistribFractions
  /// as closely as possible (greedy L1 minimisation under quantisation).
  /// </summary>
  public static Span<int> AllocateVisits(int numVisitsToAllocate,
                                         Span<int> currentDistrib,
                                         Span<float> optimalDistribFractions,
                                         int numAlreadyUsed)
  {
    if (currentDistrib.Length != optimalDistribFractions.Length)
    {
      throw new ArgumentException("Spans must have equal length.");
    }

    if (numVisitsToAllocate < 0)
    {
      throw new ArgumentOutOfRangeException(nameof(numVisitsToAllocate));
    }

    int categories = currentDistrib.Length;

    // 1.  Normalise fractions so they sum to 1 (if necessary).
    float fractionSum = 0.0f;
    for (int i = 0; i < categories; i++)
    {
      fractionSum += optimalDistribFractions[i];
    }
    if (fractionSum <= 0.0f)
    {
      throw new ArgumentException("All target fractions are zero or negative.");
    }
    float renorm = 1.0f / fractionSum;
    for (int i = 0; i < categories; i++)
    {
      optimalDistribFractions[i] *= renorm;
    }

     // 2. Prepare result buffer and running totals.
    int[] allocation = new int[categories];
    int currentTotal = 0;
    for (int i = 0; i < categories; i++)
    {
      currentTotal += currentDistrib[i];
    }

    // 3. Greedy incremental allocation:
    //    At each step, give one visit to the category whose
    //    current fraction is furthest *below* its target.
    //    This guarantees we never exceed the visit budget.
    for (int remaining = numVisitsToAllocate; remaining > 0; remaining--)
    {
      float denominatorAfterIncrement = (float)(currentTotal + numVisitsToAllocate - remaining + 1);

      float bestGap = float.NegativeInfinity;
      int bestIndex = 0;

      for (int i = 0; i < categories; i++)
      {
        int prospectiveCount = currentDistrib[i] + allocation[i];
        float currentFraction = prospectiveCount / denominatorAfterIncrement;
        float gap = optimalDistribFractions[i] - currentFraction;

        if (gap > bestGap)
        {
          bestGap = gap;
          bestIndex = i;
        }
      }

      allocation[bestIndex] += 1; // assign this visit
    }


//      MakeContiguous(allocation, numAlreadyUsed);

#if DEBUG
    // Any nodes being first visited must be contiguous starting at first non-visited
    bool zeroAtNumUsed = allocation[numAlreadyUsed] == 0;
    bool foundLaterNonzero = false; 
    for (int i = numAlreadyUsed + 1; i < allocation.Length; i++)
    {
      if (allocation[i] != 0)
      {
        foundLaterNonzero = true;
        break;
      }
    }
    if (zeroAtNumUsed && foundLaterNonzero)
    {
      throw new InvalidOperationException("Internal error: allocation not contiguous.");
    }
#endif

    return allocation;
  }
}
