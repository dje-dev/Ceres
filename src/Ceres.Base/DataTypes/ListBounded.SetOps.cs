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

using Ceres.Base.Misc;
using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Methods at ListBounded<T> which relate to set operations.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public partial class ListBounded<T> : IDisposable, IEnumerable<T>, IList<T> where T : IComparable<T>
  {
    /// <summary>
    /// 
    /// 
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public ListBounded<T> SubsetNotIn(ListBounded<T> other, ref T[] scratchBuffer)
    {
      if (scratchBuffer == null || scratchBuffer.Length < other.MaxLength)
        scratchBuffer = new T[other.MaxLength];

      int countThis = Count;
      int countOther = other.Count;

      // Make a copy of the other buffer and sort the copy
      Array.Copy(other.array, scratchBuffer, countOther);
      Array.Sort(scratchBuffer, 0, countOther);

      ListBounded<T> ret = new ListBounded<T>(Count);

      for (int i = 0; i < countThis; i++)
      {
        T item = this[i];

        // Search other array for this item, and add to our return list if not found
        if (Array.BinarySearch(scratchBuffer, 0, countOther, item) < 0)
          ret.Add(item);
      }
      return ret;
    }

    /// <summary>
    /// 
    /// 
    /// NOTE: this is destructive because it sorts the array sunderlying both ListBounded.
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public ListBounded<T> SubsetNotInOLD(ListBounded<T> other)
    {
      Array.Sort(array, 0, Count);
      Array.Sort(other.array, 0, other.Count);

      int len2 = other.Count;
      int pos2 = 0;

      ListBounded<T> ret = new ListBounded<T>(Count);

      for (int i = 0; i < Count; i++)
      {
        T item = this[i];

        while (pos2 < len2 && other[pos2].CompareTo(item) == -1)
          pos2++;

        if (pos2 == len2 || other[pos2].CompareTo(item) == 1)
          ret.Add(item);
      }
      return ret;
    }

    // --------------------------------------------------------------------------------------------
    /// <summary>
    /// Returns a new ListBounded which represents the union of items (including possible duplicates).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public ListBounded<T> UnionWith(ListBounded<T> other)
    {
      ListBounded<T> ret = new ListBounded<T>(Count + other.Count);

      Array.Copy(array, ret.array, Count);
      Array.Copy(other.array, 0, ret.array, Count, other.Count);

      return ret;
    }

    /// <summary>
    /// Returns a new ListBounded which represents the (sorted) union of items (excluding duplicates).
    /// 
    /// NOTE: this is destructive because it sorts the array sunderlying both ListBounded.
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public ListBounded<T> UnionDistinctWith(ListBounded<T> other)
    {
      Array.Sort(array, 0, Count);
      Array.Sort(other.array, 0, other.Count);

      int len1 = this.Count;
      int len2 = other.Count;

      int pos1 = 0;
      int pos2 = 0;

      ListBounded<T> ret = new ListBounded<T>(len1 + len2);

      while (pos1 < len1 || pos2 < len2)
      {
        if (pos1 == len1)
          ret.Add(other[pos2++]);
        else if (pos2 == len2)
          ret.Add(this[pos1++]);
        else
        {
          T try1 = this[pos1];
          T try2 = other[pos2];

          int compareResult = try1.CompareTo(try2);
          if (compareResult == 0)
          {
            ret.Add(try1);
            pos1++;
            pos2++;
          }
          else if (compareResult == -1)
          {
            ret.Add(try1);
            pos1++;
          }
          else if (compareResult == 1)
          {
            ret.Add(try2);
            pos2++;
          }
          else
            Debug.Assert(false);
        }

      }
      return ret;
    }
  }
}