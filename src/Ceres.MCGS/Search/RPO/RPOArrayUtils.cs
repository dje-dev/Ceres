#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;
using System.Runtime.CompilerServices;

namespace Ceres.MCGS.Search.RPO;

public static class RPOArrayUtils
{
  /// <summary>
  /// Selects the subset of rows where <paramref name="counts[i]"/> ? <paramref name="minCount"/>.
  /// The surviving items are copied into new, tightly-packed arrays
  /// (<c>counts</c>, <c>floats1</c>, <c>floats2</c>) and an <c>indices</c> array that
  /// tells you each element's original position.
  /// </summary>
  /// <returns>
  /// (<c>counts</c>, <c>floats1</c>, <c>floats2</c>, <c>indices</c>)
  /// </returns>
  /// <exception cref="ArgumentException">
  /// Thrown if the three spans do not share the same length.
  /// </exception>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static (int[] counts, float[] floats1, float[] floats2, int[] indices)
      SubsetGreaterThan(
          Span<int> counts,
          Span<float> floats1,
          Span<float> floats2,
          int minCount)
  {
    int len = counts.Length;
    if (floats1.Length != len || floats2.Length != len)
      throw new ArgumentException("All spans must have the same length.");

    // -------- Pass 1: determine survivor count --------
    int survivors = 0;
    for (int i = 0; i < len; ++i)
      if (counts[i] >= minCount)
        ++survivors;

    // -------- Allocate result arrays exactly once --------
    int[] outCounts = new int[survivors];
    float[] outFloats1 = new float[survivors];
    float[] outFloats2 = new float[survivors];
    int[] outIdx = new int[survivors];

    // -------- Pass 2: copy survivors --------
    int dst = 0;
    for (int src = 0; src < len; ++src)
    {
      int c = counts[src];
      if (c < minCount) continue;

      outCounts[dst] = c;
      outFloats1[dst] = floats1[src];
      outFloats2[dst] = floats2[src];
      outIdx[dst] = src;   // original index
      ++dst;
    }

    return (outCounts, outFloats1, outFloats2, outIdx);
  }
}
