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

#endregion

namespace Ceres.Base.Algorithms
{
  /// <summary>
  /// Static helper methods relating to sorting.
  /// </summary>
  public static class Sorting
  {
    /// <summary>
    /// Sorts the contents of a Span (in place) against a generic type.
    /// Runtime is slightly (circa 15%) faster than the generic version of Sort also in this class.

    /// A simple bubble short is used.
    /// 
    /// TODO: Switch to the new version that will be included in .NET Core 5.0.
    /// </summary>
    /// <param name="array"></param>
    public static void SortBubble<T>(Span<T> array) where T : IComparable<T>
    {
      bool neededSwap;
      int length = array.Length;

      do
      {
        neededSwap = false;
        for (int index = 0; index < length - 1; index++)
        {
          bool needSwap = array[index].CompareTo(array[index + 1]) == 1;

          if (needSwap)
          {
            T temp = array[index];
            array[index] = array[index + 1];
            array[index + 1] = temp;
            neededSwap = true;
          }
        }
      } while (neededSwap);
    }

    /// <summary>
    /// Sorts the contents of a Span (in place) against the concrete type int.
    /// 
    /// A simple bubble short is used.
    /// </summary>
    /// <param name="array"></param>
    public static void SortBubble(Span<int> array)
    {
      bool neededSwap;
      int length = array.Length;

      do
      {
        neededSwap = false;
        for (int index = 0; index < length - 1; index++)
        {
          bool needSwap = array[index] > array[index + 1];

          if (needSwap)
          {
            int temp = array[index];
            array[index] = array[index + 1];
            array[index + 1] = temp;
            neededSwap = true;
          }
        }
      } while (neededSwap);
    }

  }
}
