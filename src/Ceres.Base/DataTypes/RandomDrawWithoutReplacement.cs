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
using System.Threading;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Threadsafe data structure that returns randomized draws from 
  /// a set of objects without replacement.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class RandomDrawWithoutReplacement<T>
  {
    private readonly T[] objects;
    private int nextIndex;


    public RandomDrawWithoutReplacement(T[] items)
    {
      objects = items;
      Shuffle(objects);
      nextIndex = 0;
    }


    public bool TryDraw(out T item)
    {
      int index = Interlocked.Increment(ref nextIndex) - 1;
      if (index >= objects.Length)
      {
        item = default!;
        return false;
      }

      item = objects[index];
      return true;
    }


    private static void Shuffle(T[] array)
    {
      Random rng = new Random(Guid.NewGuid().GetHashCode());
      for (int i = array.Length - 1; i > 0; i--)
      {
        int j = rng.Next(i + 1);
        (array[i], array[j]) = (array[j], array[i]);
      }
    }
  }
}
