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

using System.Diagnostics;
using System.Threading;

#endregion

namespace Ceres.Base.Threading
{
  /// <summary>
  /// Data structure that efficiently accumulates long counts
  /// which are incremented by potentially multiple concurrent threads.
  /// </summary>
  public struct AccumulatorMultithreaded
  {
    #region Private data

    // Use multiple buckets with spacing in between so they
    // fall in separate cache lines to reduce contention.
    const int NUM_BUCKETS = 256;
    const int NUM_PER_BUCKET = 64 / sizeof(long);

    long[] accumulators;

    #endregion

    /// <summary>
    /// Returns if the accumulator has been initialized.
    /// </summary>
    public bool IsInitialized => accumulators != null;

    /// <summary>
    /// Initializes for use.
    /// </summary>
    public void Initialize()
    {
      accumulators = new long[NUM_BUCKETS * NUM_PER_BUCKET];
    }

    /// <summary>
    /// Increments the accumulator by specified amount.
    /// </summary>
    /// <param name="increment"></param>
    /// <param name="randomValue">any pseudorandom value (used to distribute across buckets)</param>
    public void Add(long increment, int randomValue)
    {
      Interlocked.Add(ref accumulators[(randomValue % NUM_BUCKETS) * NUM_PER_BUCKET], increment);
    }


    /// <summary>
    /// Returns the currently accumulated value (not threadsafe).
    /// </summary>
    public long Value
    {
      get
      {
        if (accumulators == null)
        {
          // Not yet intialized.
          return 0;
        }
        else
        {
          long acc = 0;
          for (int i = 0; i < NUM_BUCKETS; i++)
          {
            acc += accumulators[i * NUM_PER_BUCKET];
          }
          return acc;
        }
      }
    }

  }

}
