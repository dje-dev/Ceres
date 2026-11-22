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
using System.Collections.Concurrent;
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
      => ParallelOptionsForMaxThreads(CalcMaxParallelism(numItems, optimalItemsPerThread));


    /// <summary>
    /// Returns a ParallelOptions object with the specified maximum degree of parallelism,
    /// reusing a cached instance if one is available for that thread count.
    /// </summary>
    /// <param name="maxThreads"></param>
    /// <returns></returns>
    public static ParallelOptions ParallelOptionsForMaxThreads(int maxThreads)
    {
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


    /// <summary>
    /// Executes a loop body over the half-open range [fromInclusive, toExclusive),
    /// in parallel only if the item count justifies more than one thread
    /// (see CalcMaxParallelism), otherwise serially on the calling thread.
    ///
    /// The serial path is semantically identical to running Parallel.For with a
    /// MaxDegreeOfParallelism of one, but avoids its setup cost and per-call allocations
    /// (which are substantial relative to the work when batches are small).
    /// </summary>
    /// <param name="fromInclusive"></param>
    /// <param name="toExclusive"></param>
    /// <param name="optimalItemsPerThread"></param>
    /// <param name="body"></param>
    /// <param name="maxParallelism">optionally an upper bound on the number of threads used</param>
    public static void For(int fromInclusive, int toExclusive, int optimalItemsPerThread, Action<int> body,
                           int maxParallelism = int.MaxValue)
    {
      int numItems = toExclusive - fromInclusive;
      int maxThreads = System.Math.Min(CalcMaxParallelism(numItems, optimalItemsPerThread), maxParallelism);
      if (maxThreads <= 1)
      {
        for (int i = fromInclusive; i < toExclusive; i++)
        {
          body(i);
        }
      }
      else
      {
        Parallel.For(fromInclusive, toExclusive, ParallelOptionsForMaxThreads(maxThreads), body);
      }
    }


    /// <summary>
    /// Executes a body over contiguous subranges partitioning [fromInclusive, toExclusive),
    /// in parallel only if the item count justifies more than one thread
    /// (see CalcMaxParallelism), otherwise as a single serial call on the calling thread.
    ///
    /// As with For, the serial path is semantically identical to the parallel one restricted
    /// to a single thread but avoids the setup cost and per-call allocations
    /// (notably those of the range partitioner).
    /// </summary>
    /// <param name="fromInclusive"></param>
    /// <param name="toExclusive"></param>
    /// <param name="optimalItemsPerThread"></param>
    /// <param name="rangeBody">action accepting the inclusive start and exclusive end of a subrange</param>
    /// <param name="maxParallelism">optionally an upper bound on the number of threads used</param>
    public static void ForRange(int fromInclusive, int toExclusive, int optimalItemsPerThread, Action<int, int> rangeBody,
                                int maxParallelism = int.MaxValue)
    {
      int numItems = toExclusive - fromInclusive;
      int maxThreads = System.Math.Min(CalcMaxParallelism(numItems, optimalItemsPerThread), maxParallelism);
      if (maxThreads <= 1)
      {
        if (numItems > 0)
        {
          rangeBody(fromInclusive, toExclusive);
        }
      }
      else
      {
        Parallel.ForEach(Partitioner.Create(fromInclusive, toExclusive),
                         ParallelOptionsForMaxThreads(maxThreads),
                         range => rangeBody(range.Item1, range.Item2));
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
