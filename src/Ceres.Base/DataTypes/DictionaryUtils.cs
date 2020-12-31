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
using System.Threading.Tasks;

#endregion

namespace Ceres.Base.DataTypes
{
  public static class DictionaryUtils
  {

    /// <summary>
    /// Modifies an IDictionary by removing to reduce size to below the specified resizeTarget
    /// where the items are removed in an order defined by a priority function.
    /// </summary>
    /// <typeparam name="K"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <param name="dict"></param>
    /// <param name="priorityFunc"></param>
    /// <param name="resizeTarget"></param>
    public static void PruneDictionary<K, T>(IDictionary<K, T> dict,
                                             Func<T, float> priorityFunc,
                                             int resizeTarget) where K : unmanaged
    {
      int dictCount = dict.Count;
      float[] keyPrioritiesForSorting = GC.AllocateUninitializedArray<float>(dictCount);
      float[] keyPriorities = GC.AllocateUninitializedArray<float>(dictCount);
      K[] keys = GC.AllocateUninitializedArray<K>(dictCount);

      int count = 0;
      foreach (KeyValuePair<K, T> kvp in dict)
      {
        // Since the dictionary might keep growing 
        // while we iterate, see if we have reached
        // end of our allocated space and just stop there
        if (count >= dictCount) break;

        float priorityValue = priorityFunc(kvp.Value); 
        keyPriorities[count] = priorityValue;
        keyPrioritiesForSorting[count] = priorityValue;

        keys[count] = kvp.Key;
        count++;
      }

      if (count < dictCount) throw new Exception("Internal error: dictionary shrunk during PruneDictionary operation.");

      int numToPrune = dictCount - resizeTarget;

      // Compute the minimum sequence number an entry must have
      // to be retained (to enforce LRU eviction)
      float cutoff = KthSmallestValueFloat.CalcKthSmallestValue(keyPrioritiesForSorting, numToPrune);

      for (int i = 0; i < count; i++)
        if (keyPriorities[i] <= cutoff)
          dict.Remove(keys[i]);
    }

  }
}
