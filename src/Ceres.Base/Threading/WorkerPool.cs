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
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

#endregion

namespace Ceres.Base.Threading;

/// <summary>
/// Creates and manages a pool of worker threads that process submitted work items.
/// The pool starts with a configurable number of threads and can grow on demand
/// up to a specified maximum when work saturation is detected.
/// 
/// Thread-safety: All public methods are thread-safe and support concurrent access.
/// Nested work submissions (submitting work from within a work item callback) are supported.
/// 
/// TODO: consider this class versus to ParallelItemProcessorWorkerPool
///       (verify correct shutdown logic when work items continue to arrive).
/// </summary>
/// <typeparam name="T">The type of state object passed to work item delegates.</typeparam>
public sealed class WorkerPool<T> : IDisposable
{
  private readonly BlockingCollection<(Action<T>, T)> pendingWork;
  private readonly ManualResetEventSlim drainedEvent;
  private readonly List<Thread> workers;
  private readonly object workerListLock = new object();
  private readonly CancellationTokenSource shutdownTokenSource;

  private readonly int growthIncrement;
  private readonly int maxThreads;
  private readonly string threadNamePrefix;

  private volatile bool shutdownRequested;
  private volatile bool disposed;

  private static int unexpectedExceptionLogged; // 0/1 flag to log warning only once.

  private int pendingWorkCount;          // Enqueued-but-not-finished actions.
  private int activeWorkerCount;         // Currently executing actions.
  private int peakActiveWorkerCount;     // High-water mark.
  private int createdThreadCount;        // Threads created.
  private int growthLock;                // 0/1 guard to serialize growth.


  /// <summary>
  /// Constructs a new worker pool with the specified threading parameters.
  /// </summary>
  /// <param name="initialThreads">Number of worker threads to create immediately.</param>
  /// <param name="growthIncrement">Number of threads to add when the pool grows due to saturation.</param>
  /// <param name="maximumThreads">Upper limit on total threads; defaults to unlimited if null.</param>
  /// <param name="threadNamePrefix">Prefix for worker thread names (useful for debugging).</param>
  /// <exception cref="ArgumentOutOfRangeException">Thrown when initialThreads or growthIncrement is not positive.</exception>
  public WorkerPool(int initialThreads,
                    int growthIncrement = 2,
                    int? maximumThreads = null,
                    string? threadNamePrefix = null)
  {
    ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(initialThreads, 0);
    ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(growthIncrement, 0);

    this.growthIncrement = growthIncrement;
    maxThreads = maximumThreads.HasValue ? System.Math.Max(initialThreads, maximumThreads.Value) : int.MaxValue;
    this.threadNamePrefix = threadNamePrefix ?? "WorkerPool";

    // BlockingCollection provides a blocking Take() over a concurrent queue.
    pendingWork = new();
    drainedEvent = new ManualResetEventSlim(true); // no work yet => "drained"
    workers = new List<Thread>();
    shutdownTokenSource = new CancellationTokenSource();
    StartThreads(initialThreads);
  }


  /// <summary>
  /// Enqueues a work item to be processed by a worker thread.
  /// </summary>
  /// <param name="action">The delegate to execute, receiving the work item as its parameter.</param>
  /// <param name="workItem">The state object passed to the action delegate.</param>
  /// <exception cref="ObjectDisposedException">Thrown if the pool has been disposed.</exception>
  /// <remarks>
  /// This method is thread-safe and supports nested submissions (submitting work from within
  /// a work item callback). The pool may automatically grow if saturation is detected.
  /// </remarks>
  public void SubmitWorkItem(Action<T> action, T workItem)
  {
    ObjectDisposedException.ThrowIf(disposed, this);

    // Increment pending count BEFORE resetting the event to avoid race with WaitAll.
    // The pattern: increment first, then reset ensures that if a concurrent decrement
    // reaches zero and sets the event, our reset occurs on a non-zero count state.
    int newCount = Interlocked.Increment(ref pendingWorkCount);
    if (newCount == 1)
    {
      drainedEvent.Reset();
    }

    // Unbounded add; returns immediately.
    pendingWork.Add((action, workItem));

    // If we're nearly saturated, consider growing.
    int active = Volatile.Read(ref activeWorkerCount);
    int created = Volatile.Read(ref createdThreadCount);
    if (active >= created - 2)
    {
      TryGrow();
    }
  }


  /// <summary>
  /// Blocks until all pending work (including any nested submissions) has completed.
  /// </summary>
  /// <returns>The number of milliseconds spent waiting, or 0 if no waiting was required.</returns>
  /// <exception cref="ObjectDisposedException">Thrown if the pool has been disposed.</exception>
  public int WaitAll()
  {
    ObjectDisposedException.ThrowIf(disposed, this);

    Stopwatch? sw = null;

    while (true)
    {
      if (Volatile.Read(ref pendingWorkCount) == 0)
      {
        return (int)(sw?.ElapsedMilliseconds ?? 0);
      }

      sw ??= Stopwatch.StartNew();
      drainedEvent.Wait();
    }
  }


  /// <summary>
  /// Gets the maximum number of workers observed simultaneously executing user work (high-water mark).
  /// </summary>
  public int MaxConcurrentWorkersObserved => Volatile.Read(ref peakActiveWorkerCount);

  /// <summary>
  /// Gets the current number of dedicated worker threads in this pool.
  /// This value only increases over the pool's lifetime (threads are not removed).
  /// </summary>
  public int CurrentThreadCount => Volatile.Read(ref createdThreadCount);

  /// <summary>
  /// Releases all resources and terminates worker threads.
  /// </summary>
  public void Dispose()
  {
    Dispose(true);
    GC.SuppressFinalize(this);
  }


  /// <summary>
  /// Creates and starts the specified number of worker threads.
  /// </summary>
  private void StartThreads(int count)
  {
    for (int i = 0; i < count; i++)
    {
      Thread thread = new(WorkerLoop)
      {
        IsBackground = true,
        Name = threadNamePrefix + "-" + (Volatile.Read(ref createdThreadCount) + 1).ToString()
      };

      lock (workerListLock)
      {
        workers.Add(thread);
        Interlocked.Increment(ref createdThreadCount);
      }

      thread.Start();
    }
  }


  /// <summary>Attempts to grow the pool by adding more worker threads if below maximum.</summary>
  private void TryGrow()
  {
    if (Volatile.Read(ref shutdownRequested)
     || Interlocked.Exchange(ref growthLock, 1) == 1) 
    { 
      return; 
    }

    try
    {
      int created = Volatile.Read(ref createdThreadCount);
      if (created >= maxThreads) 
      { 
        return;
      }

      int target = System.Math.Min(created + growthIncrement, maxThreads);
      int toCreate = target - created;
      if (toCreate > 0)
      {
        StartThreads(toCreate);
      }
    }
    finally
    {
      Volatile.Write(ref growthLock, 0);
    }
  }


  /// <summary>
  /// Main loop executed by each worker thread to process queued work items.
  /// </summary>
  private void WorkerLoop()
  {
    while (true)
    {
      (Action<T> action, T state) workItem;
      try
      {
        // Blocks until an item arrives or disposal cancels.
        workItem = pendingWork.Take(shutdownTokenSource.Token);
      }
      catch (OperationCanceledException)
      {
        if (shutdownRequested)
        {
          return;
        }
        continue;
      }
      catch (InvalidOperationException ex)
      {
        // BlockingCollection is marked as CompleteAdding (we don't use that here),
        // but handle defensively. Log warning on first occurrence.
        if (Interlocked.Exchange(ref unexpectedExceptionLogged, 1) == 0)
        {
          Console.WriteLine($"WARNING: WorkerPool caught unexpected InvalidOperationException (logging once):");
          Console.WriteLine(ex.ToString());
        }

        if (shutdownRequested)
        {
          return;
        }
        continue;
      }

      int nowActive = Interlocked.Increment(ref activeWorkerCount);
      UpdatePeakActiveWorkerCount(nowActive);

      try
      {
        workItem.action(workItem.state);
      }
      finally
      {
        Interlocked.Decrement(ref activeWorkerCount);

        int remaining = Interlocked.Decrement(ref pendingWorkCount);
        if (remaining == 0)
        {
          drainedEvent.Set();
        }
      }
    }
  }


  /// <summary>
  /// Atomically updates the high-water mark if the candidate exceeds current peak.
  /// </summary>
  private void UpdatePeakActiveWorkerCount(int candidate)
  {
    while (true)
    {
      int current = Volatile.Read(ref peakActiveWorkerCount);
      if (candidate <= current)
      {
        return;
      }

      if (Interlocked.CompareExchange(ref peakActiveWorkerCount, candidate, current) == current)
      {
        return;
      }
    }
  }

  /// <summary>
  /// Core disposal logic that signals shutdown, cancels pending work, and joins all worker threads.
  /// </summary>
  /// <param name="disposing">True if called from Dispose(); false if called from finalizer.</param>
  private void Dispose(bool disposing)
  {
    if (!disposing || disposed)
    {
      return;
    }

    disposed = true;
    shutdownRequested = true;

    // Signal cancellation to wake up all blocked workers.
    shutdownTokenSource.Cancel();

    // Wait for all worker threads to finish.
    List<Thread> threadsToJoin;
    lock (workerListLock)
    {
      threadsToJoin = [.. workers];
    }

    foreach (Thread thread in threadsToJoin)
    {
      if (thread.IsAlive)
      {
        thread.Join();
      }
    }

    pendingWork.Dispose();
    drainedEvent.Dispose();
    shutdownTokenSource.Dispose();
  }
}
