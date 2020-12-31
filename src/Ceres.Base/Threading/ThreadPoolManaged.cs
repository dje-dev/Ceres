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
using System.Collections.Concurrent;

#endregion

namespace Ceres.Base.Threading
{
  /// <summary>
  /// Mimics functionality of static ThreadPool, but runs over dedicated thread pool
  /// Adapted and modernized from code by Stephen Toub (stoub@microsoft.com), origionally at:
  ///   http://www.gotdotnet.com/community/usersamples/Default.aspx?query=ManagedThreadPool
  ///   
  /// TODO: Consider moving to the new native C# ThreadPool impelmentation debuting in .NET 6.
  /// </summary>
  public class ThreadPoolManaged
  {
    public readonly bool RoundRobinAffinitize;

    public readonly string PoolID;

    int roundRobinAffinitizeProcessorID;

    int GlobalThreadPoolIndex;

    bool incrementProcessorIDEachThread;

    [ThreadStatic]
    public static int ThisThreadGlobalThreadPoolIndex = 0;

    int _maxWorkerThreads;

    /// <summary>Queue of all the callbacks waiting to be executed.</summary>
    ConcurrentQueue<WaitingCallback> _waitingCallbacks;

    /// <summary>
    /// Used to signal that a worker thread is needed for processing.  Note that multiple
    /// threads may be needed simultaneously and as such we use a semaphore instead of
    /// an auto reset event.
    /// </summary>
    Semaphore _workerThreadNeeded;

    /// <summary>
    /// List of all worker threads at the disposal of the thread pool.
    /// </summary>
    Thread[] _workerThreads;

    CountdownEvent countdownThreads = new CountdownEvent(1);


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxWorkerThreads"></param>
    /// <param name="priority"></param>
    /// <param name="roundRobinAffinitize"></param>
    public ThreadPoolManaged(string poolID, int maxWorkerThreads, int globalThreadPoolIndex,
                             ThreadPriority priority = ThreadPriority.Normal,
                             bool roundRobinAffinitize = false,
                             int roundRobinAffinitizeProcessorID = 0,
                             bool incrementProcessorIDEachThread = true)
    {
      PoolID = poolID;

      _maxWorkerThreads = maxWorkerThreads;
      GlobalThreadPoolIndex = globalThreadPoolIndex;
      _waitingCallbacks = new ConcurrentQueue<WaitingCallback>();
      _workerThreads = new Thread[maxWorkerThreads];

      RoundRobinAffinitize = roundRobinAffinitize;
      this.roundRobinAffinitizeProcessorID = roundRobinAffinitizeProcessorID;
      this.incrementProcessorIDEachThread = incrementProcessorIDEachThread;

      _workerThreadNeeded = new Semaphore(0);

      for (int i = 0; i < _maxWorkerThreads; i++)
      {
        Thread newThread = new Thread(new ParameterizedThreadStart(ProcessQueuedItems));
        _workerThreads[i] = newThread;

        newThread.Name = $"ThreadPoolManaged {poolID} #{i}";
        newThread.IsBackground = true;
        newThread.Priority = priority;

        int threadIndex = roundRobinAffinitize ? i : -1;
        newThread.Start(threadIndex);
      }
    }

    public void WaitDone()
    {
      // take out initialization value of 1
      countdownThreads.Signal();

      // wait until we drain completely to zero
      countdownThreads.Wait();

      // Reset start in preparation for possible next round of work items
      countdownThreads.Reset();
    }


    ~ThreadPoolManaged()
    {
      shutdown = true;
    }

    /// <summary>
    /// Queues a user work item to the thread pool.
    /// </summary>
    /// <param name="callback">
    /// A WaitCallback representing the delegate to invoke when the thread in the 
    /// thread pool picks up the work item.
    /// </param>
    public void QueueUserWorkItem(WaitCallback callback) => QueueUserWorkItem(callback, null);


    /// <summary>
    /// Queues a user work item to the thread pool.
    /// </summary>
    /// <param name="callback">
    /// A WaitCallback representing the delegate to invoke when the thread in the 
    /// thread pool picks up the work item.
    /// </param>
    /// <param name="state">
    /// The object that is passed to the delegate when serviced from the thread pool.
    /// </param>
    public void QueueUserWorkItem(WaitCallback callback, object state)
    {
      if (!shutdown)
      {
        WaitingCallback waiting = new WaitingCallback(callback, state);

        countdownThreads.AddCount(1);
        _waitingCallbacks.Enqueue(waiting);
        _workerThreadNeeded.AddOne();
      }
    }

    /// <summary>
    /// Empties the work queue of any queued work items.
    /// </summary>
    public void EmptyQueue()
    {
      // Try to dispose of all remaining state
      foreach (WaitingCallback callback in _waitingCallbacks)
      {
        if (callback.State is IDisposable) ((IDisposable)callback.State).Dispose();
      }

      // Clear all waiting items and reset the number of worker threads currently needed
      // to be 0 (there is nothing for threads to do)
      _waitingCallbacks.Clear();
      _workerThreadNeeded.Reset(0);
    }


    /// <summary>
    /// Gets the number of threads at the disposal of the thread pool
    /// </summary>
    public int MaxThreads => _maxWorkerThreads;

    /// <summary>
    /// Gets the number of callback delegates currently waiting in the thread pool
    /// </summary>
    public int NumWaitingCallbacks => _waitingCallbacks.Count;


    volatile bool shutdown = false;
    public void Shutdown()
    {
      EmptyQueue();
      shutdown = true;
    }

    private void ProcessQueuedItems(object threadIndexWithinPool)
    {
      // Set the ThreadStatic variable indicating the which thread pool we belong to
      ThisThreadGlobalThreadPoolIndex = (int)GlobalThreadPoolIndex;

      if (RoundRobinAffinitize && GlobalThreadPoolIndex != -1)
      {
        int processorID = roundRobinAffinitizeProcessorID;
        if (this.incrementProcessorIDEachThread) processorID += (int)threadIndexWithinPool;
        throw new NotImplementedException("Internal error: AffinitizeThreadRoundRobin needs remediation");
        //Native32.AffinitizeThreadRoundRobin((uint)processorID);
      }

      while (!shutdown)
      {
        WaitingCallback callback = default;
        bool gotCallback = false;
        while (!gotCallback)
        {
          gotCallback = _waitingCallbacks.TryDequeue(out callback);

          if (!gotCallback) _workerThreadNeeded.WaitOne();
        }

        try
        {
          callback.Callback(callback.State);
        }
        catch (Exception exc)
        {
          throw new Exception("Exception in ThreadPoolManaged worker item " + exc.ToString());
        }
        finally
        {
          countdownThreads.Signal(1);
        }
      }
    }

    private struct WaitingCallback
    {
      private WaitCallback _callback;
      private object _state;

      public WaitingCallback(WaitCallback callback, object state)
      {
        _callback = callback;
        _state = state;
      }

      public WaitCallback Callback { get { return _callback; } }
      public object State { get { return _state; } }
    }

    public override string ToString()
    {
      return $"<ThreadPoolManaged {Thread.CurrentThread.ManagedThreadId}>";
    }

  }

  class Semaphore
  {
    private int _count;

    public Semaphore() : this(1) { }
    public Semaphore(int count)
    {
      if (count < 0) throw new ArgumentException("Internal error: semaphore must have a count of at least 0.", "count");
      _count = count;
    }

    public void AddOne() { V(); }
    public void WaitOne() { P(); }

    public void P()
    {
      lock (this)
      {
        while (_count <= 0) Monitor.Wait(this, Timeout.Infinite);
        _count--;
      }
    }

    public void V()
    {
      lock (this)
      {
        _count++;
        Monitor.Pulse(this);
      }
    }

    public void Reset(int count)
    {
      lock (this) { _count = count; }
    }

  }

}
