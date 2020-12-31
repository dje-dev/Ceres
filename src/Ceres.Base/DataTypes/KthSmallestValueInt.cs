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

using Ceres.Base.DataType;
using System;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Implementation algorithm to efficiently select the Kth 
  /// smallest value in an array (selection algorithm)
  /// via partitioning.
  /// </summary>
  public static class KthSmallestValueInt
  {
    /// <summary>
    /// 
    /// Uses standard selection sort (based on partitioning) with time complexity O(n)
    /// </summary>
    /// <param name="values"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    public static int CalcKthSmallestValue(Span<int> values, int k)
    {
      return Partition(values, 0, values.Length - 1, k);
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="values"></param>
    /// <param name="minIndex"></param>
    /// <param name="maxIndex"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    static int Partition(Span<int> values, int minIndex, int maxIndex, int k)
    {
      if (minIndex >= maxIndex) return values[minIndex];

      int pivotValue = values[minIndex];
      int pivotIndex = minIndex;
      for (int i = minIndex + 1; i <= maxIndex; i++)
      {
        if (values[i] >= pivotValue)
          ArrayUtils.SwapStructsInSpan(values, ++minIndex, i);
      }

      ArrayUtils.SwapStructsInSpan(values, pivotIndex, minIndex);

      int numInRange = maxIndex - minIndex + 1;

      if (k < numInRange)
        return Partition(values, minIndex + 1, maxIndex, k);
      else if (k > numInRange)
        return Partition(values, pivotIndex, minIndex - 1, k - numInRange);
      else if (numInRange == k)
        return pivotValue;
      else
        throw new Exception("Internal error: Partition");
    }

  }


}
