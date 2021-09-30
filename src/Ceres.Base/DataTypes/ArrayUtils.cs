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
using System.Diagnostics;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Base.DataType
{

  public static class ArrayUtils
  {
    #region Change shape

    public static T[] To1D<T>(T[][] raw)
    {
      if (raw.Length == 0) return Array.Empty<T>();

      int countRight = raw[0].Length;
      int totalCount = countRight * raw.Length;
      T[] ret = new T[totalCount];
      int offset = 0;
      for (int i = 0; i < raw.Length; i++)
        for (int j = 0; j < countRight; j++)
          ret[offset++] = raw[i][j];

      return ret;
    }


    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="values"></param>
    /// <param name="index1"></param>
    /// <param name="index2"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SwapStructsInArray<T>(T[] values, int index1, int index2) where T : struct
    {
      T temp = values[index1];
      values[index1] = values[index2];
      values[index2] = temp;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="values"></param>
    /// <param name="index1"></param>
    /// <param name="index2"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SwapStructsInSpan<T>(Span<T> values, int index1, int index2) where T : struct
    {
      T temp = values[index1];
      values[index1] = values[index2];
      values[index2] = temp;
    }


    /// <summary>
    /// Unpacks 1D float array into array of arrays of floats.
    /// </summary>
    /// <param name="raw"></param>
    /// <param name="numOuter"></param>
    /// <returns></returns>
    public static T[][] ToArrayOfArray<T>(T[,] raw)
    {
      if (raw == null) return null;

      int numOuter = raw.GetLength(0);
      int numInner = raw.GetLength(1);

      T[][] ret = new T[numOuter][];

      for (int i = 0; i < numOuter; i++)
      {
        ret[i] = new T[numInner];
        for (int j = 0; j < numInner; j++)
          ret[i][j] = raw[i, j];
      }
      return ret;
    }


    /// <summary>
    /// Unpacks 1D float array into array of arrays of floats.
    /// </summary>
    /// <param name="raw"></param>
    /// <param name="numOuter"></param>
    /// <returns></returns>
    public static T[][] ToArrayOfArray<T>(Span<T> raw, int numInner)
    {
      if (raw == null) return null;

      if (raw.Length % numInner != 0) throw new Exception("does not evenly divide");
      int numOuter = raw.Length / numInner;

      T[][] ret = new T[numOuter][];
      int count = 0;
      for (int i = 0; i < numOuter; i++)
      {
        ret[i] = new T[numInner];
        for (int j = 0; j < numInner; j++)
          ret[i][j] = raw[count++];
      }
      return ret;
    }


    /// <summary>
    /// Unpacks 1D array into 2D array.
    /// </summary>
    /// <param name="raw"></param>
    /// <param name="numOuter"></param>
    /// <returns></returns>
    public static T[,] To2D<T>(Span<T> raw, int numInner)
    {
      if (raw == null) return null;

      if (raw.Length % numInner != 0) throw new Exception("does not evenly divide");
      int numOuter = raw.Length / numInner;

      // TO DO: make faster with memory copy
      T[,] ret = new T[numOuter, numInner];
      int count = 0;
      for (int i = 0; i < numOuter; i++)
      {
        for (int j = 0; j < numInner; j++)
          ret[i, j] = raw[count++];
      }
      return ret;
    }


    /// <summary>
    /// Unpacks 1D array into 2D array (shuffle/transposed).
    /// </summary>
    /// <param name="raw"></param>
    /// <param name="numOuter"></param>
    /// <returns></returns>
    public static T[,] To2DShuffled<T>(Span<T> raw, int numInner)
    {
      if (raw == null) return null;

      if (raw.Length % numInner != 0) throw new Exception("does not evenly divide");
      int numOuter = raw.Length / numInner;

      // TO DO: make faster with memory copy
      T[,] ret = new T[numOuter, numInner];
      int count = 0;
      for (int j = 0; j < numInner; j++)
      {
        for (int i = 0; i < numOuter; i++)
          ret[i, j] = raw[count++];
      }
      return ret;
    }


    #endregion

    /// <summary>
    /// Returns the index of the element having minimal value.
    /// 
    /// In the case of ties, the smallest index is returned.
    /// </summary>
    /// <param name="array"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int IndexOfElementWithMinValue(Span<float> array, int count)
    {
      Debug.Assert(count > 0);

      // Efficiently handle common cases with small count
      if (count == 1)
        return 0;
      else if (count == 2)
        return array[1] > array[0] ? 0 : 1;

      // For better performance, we run two sets of comparisons (odd and even)
      // at the same time (to increase instruction level parallelism)
      // This improves performance by about 40%
      int minIndex0 = 0;
      int minIndex1 = 1;
      float minValue0 = array[0];
      float minValue1 = array[1];

      int i = 2;
      while (i < count - 1)
      {
        if (array[i] < minValue0)
        {
          minValue0 = array[i];
          minIndex0 = i;
        }
        if (array[i + 1] < minValue1)
        {
          minValue1 = array[i + 1];
          minIndex1 = i + 1;
        }

        i += 2;
      }

      // Make sure maxIndex0 has the smaller index
      // (we prefer smaller indices in case of ties of values)
      if (minIndex0 > minIndex1)
      {
        int temp = minIndex0;
        minIndex0 = minIndex1;
        minIndex1 = temp;

        float tempV = minValue0;
        minValue0 = minValue1;
        minValue1 = tempV;
      }

      int lastIndex = count - 1;
      if (i == lastIndex)
      {
        float last = array[lastIndex];
        if (minValue0 <= minValue1)
          return minValue0 <= last ? minIndex0 : lastIndex;
        else
          return minValue1  <= last ? minIndex1 : lastIndex;
      }
      else
        return minValue0 <= minValue1 ? minIndex0 : minIndex1;
    }



    /// <summary>
    /// Returns the index of the element having maximal value.
    /// 
    /// In the case of ties, the smallest index is returned.
    /// </summary>
    /// <param name="array"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int IndexOfElementWithMaxValue(Span<float> array, int count)
    {
      Debug.Assert(count > 0);

      // Efficiently handle common cases with small count
      if (count == 1)
        return 0;
      else if (count == 2)
        return array[1] > array[0] ? 1 : 0;

      // For better performance, we run two sets of comparisons (odd and even)
      // at the same time (to increase instruction level parallelism)
      // This improves performance by about 40%
      int maxIndex0 = 0;
      int maxIndex1 = 1;
      float maxValue0 = array[0];
      float maxValue1 = array[1];

      int i = 2;
      while (i < count - 1)
      {
        if (array[i] > maxValue0)
        {
          maxValue0 = array[i];
          maxIndex0 = i;
        }
        if (array[i + 1] > maxValue1)
        {
          maxValue1 = array[i + 1];
          maxIndex1 = i + 1;
        }

        i += 2;
      }

      // Make sure maxIndex0 has the smaller index
      // (we prefer smaller indices in case of ties of values)
      if (maxIndex0 > maxIndex1)
      {
        int temp = maxIndex0;
        maxIndex0 = maxIndex1;
        maxIndex1 = temp;

        float tempV = maxValue0;
        maxValue0 = maxValue1;
        maxValue1 = tempV;
      }

      int lastIndex = count - 1;
      if (i == lastIndex)
      {
        float last = array[lastIndex];
        if (maxValue0 >= maxValue1)
          return maxValue0 >= last ? maxIndex0 : lastIndex;
        else
          return maxValue1 >= last ? maxIndex1 : lastIndex;
      }
      else
        return maxValue0 >= maxValue1 ? maxIndex0 : maxIndex1;
    }

  }

}
