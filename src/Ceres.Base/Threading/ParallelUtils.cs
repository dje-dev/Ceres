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

using System.Threading.Tasks;
using Ceres.Base.OperatingSystem;

#endregion

namespace Ceres.Base.Threading
{
  /// <summary>
  /// Various static helper methods for functions such as 
  /// calculating optimal number of threads for work.
  /// </summary>
  public static class ParallelUtils
  {
  
    // Reuse ParallelOptions objects if possible to reduce GC pressure
    const int MAX_CACHED_OPTIONS = 256;
    static ParallelOptions[] cachedOptions = new ParallelOptions[MAX_CACHED_OPTIONS];


    /// <summary>
    /// Returns a ParallelOptions object customized for necessary number of threads
    /// to process a set of items, assuming a specified optimal number of items per thread.
    /// </summary>
    /// <param name="numItems"></param>
    /// <param name="optimalItemsPerThread"></param>
    /// <returns></returns>
    public static ParallelOptions ParallelOptions(int numItems, int optimalItemsPerThread)
    {
      int maxThreads = CalcMaxParallelism(numItems, optimalItemsPerThread);
      if (maxThreads >= MAX_CACHED_OPTIONS)
      {
        return new ParallelOptions() { MaxDegreeOfParallelism = maxThreads };
      }
      else if (cachedOptions[maxThreads] != null)
      {
        return cachedOptions[maxThreads];
      }
      else
      {
        return cachedOptions[maxThreads] = new ParallelOptions() { MaxDegreeOfParallelism = maxThreads };
      }
    }


    public static int CalcMaxParallelism(int numItems, int optimalItemsPerThread)
    {
      if (numItems < optimalItemsPerThread + optimalItemsPerThread / 2)
      {
        return 1;
      }

      return System.Math.Min(HardwareManager.MaxAvailableProcessors, numItems / optimalItemsPerThread);
    }


    /// <summary>
    /// 
    /// TODO: Tune this further.
    /// </summary>
    /// <param name="numBatchItems"></param>
    /// <param name="targetNumItemsPerThread"></param>
    /// <returns></returns>
    static int CalcNumThreadsForBatch(int numBatchItems, int targetNumItemsPerThread)
    {
      int numProcessors = HardwareManager.MaxAvailableProcessors;

      int idealNumThreads = 1 + (numBatchItems / targetNumItemsPerThread);

      // Try to leave a small number of processors not involved in this task
      if (numProcessors <= 6)
      {
        idealNumThreads = System.Math.Min(idealNumThreads, numProcessors - 1); // If <=6 processors, leave 1 unused
      }
      else
      {
        idealNumThreads = System.Math.Min(idealNumThreads, numProcessors - 2); // if >6 processors, leave 2 unused
      }

      // Final check:  can't have less than one processor!
      if (idealNumThreads <= 1)
      {
        idealNumThreads = 1;
      }

      return idealNumThreads;
    }

  }
}
