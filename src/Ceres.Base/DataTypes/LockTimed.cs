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
using System.Threading;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// Provides a mechanism for acquiring a lock,
/// optionally collecting statistics on wait times.
/// </summary>
public class LockTimed
{
  /// <summary>
  /// If true, allows multiple threads to wait for the lock concurrently and collect statistics.
  /// </summary>
  public readonly bool SupportConcurrentWaiting;

  /// <summary>
  /// The internal lock for synchronization.
  /// </summary>
  private readonly Lock lockObject = new();

  /// <summary>
  /// An extra lock used solely to update statistics.
  /// </summary>
  private readonly Lock statsLock = new();

  #region Statistics fields

  /// <summary>
  /// Count of lock acquisitions.
  /// </summary>
  private long callCount;

  /// <summary>
  /// Sum of wait times (in milliseconds).
  /// </summary>
  private double totalWait;

  /// <summary>
  /// Sum of squared wait times for stdDev calculation.
  /// </summary>
  private double totalSquaredWait;

  /// <summary>
  /// Maximum wait time (in milliseconds).
  /// </summary>
  private long maxWait;

  /// <summary>
  /// Second maximum wait time (in milliseconds).
  /// </summary>
  private long secondMaxWait;

  #endregion


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="supportConcurrentWaiting"></param>
  public LockTimed(bool supportConcurrentWaiting)
  {
    SupportConcurrentWaiting = supportConcurrentWaiting;
  }


  /// <summary>
  /// If the lock is currently held by any thread.
  /// </summary>
  public bool IsEntered => lockObject.IsHeldByCurrentThread;


  /// <summary>
  /// Acquire returns a guard that holds the lock.
  /// Use it with a using block: using (myTimedLock.Acquire()) { ... }
  /// </summary>
  /// <returns></returns>
  public LockGuard Acquire()
  {
    long startTicks = Stopwatch.GetTimestamp();
    lockObject.Enter();
    long elapsedTicks = Stopwatch.GetTimestamp() - startTicks;

    // Compute wait time in milliseconds.
    long waitMilliseconds = (elapsedTicks * 1000) / Stopwatch.Frequency;

    if (SupportConcurrentWaiting)
    {
      Interlocked.Increment(ref callCount);

      lock (statsLock)
      {
        UpdateStats(waitMilliseconds);
      }
    }
    else
    {
      callCount++;
      UpdateStats(waitMilliseconds);
    }

    return new LockGuard(lockObject);
  }


  /// <summary>
  /// Updates the statistics with a new wait time.
  /// </summary>
  private void UpdateStats(long waitMilliseconds)
  {
    totalWait += waitMilliseconds;
    totalSquaredWait += waitMilliseconds * waitMilliseconds;

    if (waitMilliseconds > maxWait)
    {
      // When a new maximum is found, the old max becomes the second max.
      secondMaxWait = maxWait;
      maxWait = waitMilliseconds;
    }
    else if (waitMilliseconds > secondMaxWait)
    {
      secondMaxWait = waitMilliseconds;
    }
  }


  /// <summary>
  /// Returns a string representation of the lock statistics.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    long count;
    double sum;
    double sumSquared;
    long max;
    long secondMax;

    lock (statsLock)
    {
      count = callCount;
      sum = totalWait;
      sumSquared = totalSquaredWait;
      max = maxWait;
      secondMax = secondMaxWait;
    }

    double avg = 0.0;
    double stdDev = 0.0;

    if (count > 0)
    {
      avg = sum / count;
      double variance = (sumSquared / count) - (avg * avg);
      stdDev = variance > 0.0 ? System.Math.Sqrt(variance) : 0.0;
    }

    return $"Count: {count}, Sum: {sum:F0} ms, Avg: {avg:F2} +/- {stdDev:F2} ms, Max: {max} ms, SecondMax: {secondMax} ms";
  }


  /// <summary>
  /// The disposable guard that releases the lock on Dispose.
  /// </summary>
  public readonly struct LockGuard : IDisposable
  {
    private readonly Lock theLock;

    public LockGuard(Lock lockObject)
    {
      theLock = lockObject;
    }

    public void Dispose()
    {
      theLock.Exit();
    }
  }
}


