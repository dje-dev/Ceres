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
using System.Threading;

#endregion

namespace Ceres.Base.Threading
{
  /// <summary>
  /// Creates a pool of worker threads to process items
  /// and accepts items to be added to a pool of items to be processed
  /// by the worker threads.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  /// <typeparam name="S"></typeparam>
  public class ParallelItemProcessorWorkerPool<T, S> 
  {
    /// <summary>
    /// Function to create a new worker state.
    /// </summary>
    public readonly Func<S> CreateWorkerStateFunc;

    /// <summary>
    /// If true, worker states are disposed when done.
    /// </summary>
    public readonly bool DisposeStatesWhenDone;

    /// <summary>
    /// Action to invoke on each item.
    /// </summary>
    public readonly Action<T, S> Action;

    /// <summary>
    /// Number of concurrent worker threads.
    /// </summary>
    public readonly int NumWorkerThreads;

    /// <summary>
    /// Frequency with which to recreate worker states (number of action calls).
    /// </summary>
    public readonly long WorkerStateRefreshFrequency;

    public readonly int MaxPendingItems;


    private BlockingCollection<T> pendingItems;
    private Thread[] workers;
    private S[] states;

    readonly object lockObj = new();


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="numWorkerThreads">number of parallel worker threads</param>
    /// <param name="maxPendingItems">maximum number of pending items held in pending work item queue</param>
    /// <param name="createWorkerStateFunc">func that returns new state to be associated with a worker</param>
    /// <param name="itemActionFunc">funct to be called for each item added</param>
    /// <param name="disposeStatesWhenDone">if the state should be disposed at end of processing or before recreation</param>
    /// <param name="workerStateRefreshFrequency">interval between recreation of worker engine</param>
    public ParallelItemProcessorWorkerPool(int numWorkerThreads, int maxPendingItems,
                                           Func<S> createWorkerStateFunc, Action<T, S> itemActionFunc,
                                           bool disposeStatesWhenDone = true,
                                           long workerStateRefreshFrequency = 50_000)
    {
      if (numWorkerThreads <= 0 || maxPendingItems <= 0)
      {
        throw new ArgumentException("numWorkers and maxPendingItems must be > 0");
      }

      CreateWorkerStateFunc = createWorkerStateFunc;
      NumWorkerThreads = numWorkerThreads;
      Action = itemActionFunc;
      DisposeStatesWhenDone = disposeStatesWhenDone;
      WorkerStateRefreshFrequency = workerStateRefreshFrequency;
      MaxPendingItems = maxPendingItems;

      pendingItems = new BlockingCollection<T>(maxPendingItems);

      // Start all workers and create associated states.
      states = new S[numWorkerThreads];
      workers = new Thread[NumWorkerThreads];
      for (int i = 0; i < NumWorkerThreads; i++)
      {
        workers[i] = new Thread(Worker);
        workers[i].Start(i);
      }
    }


    /// <summary>
    /// Adds an item to the pool of items to be processed.
    /// </summary>
    /// <param name="item"></param>
    public void Add(T item)
    {
      pendingItems.Add(item);
    }


    /// <summary>
    /// Finishes adding items to the pool of items to be processed,
    /// waits for all workers to complete (and optionally disposes of them).
    /// </summary>
    public void DoneAdding()
    {
      // Mark the collection as not accepting any more items.
      pendingItems.CompleteAdding();

      // Wait for all worker threads to finish.
      foreach (Thread worker in workers)
      {
        worker.Join();
      }

      // Possibly dispose of all worker states.
      if (DisposeStatesWhenDone)
      {
        foreach (S state in states)
        {
          if (state is IDisposable)
          {
            ((IDisposable)state).Dispose();
          }
        }
      }

    }


    /// <summary>
    /// Worker thread function.
    /// </summary>
    /// <param name="workerIndexObj"></param>
    void Worker(object workerIndexObj)
    {
      long numCalls = 0;
      int workerIndex = (int)workerIndexObj;
      foreach (T item in pendingItems.GetConsumingEnumerable())
      {
        try
        {
          S state = states[workerIndex];

          // Create/recreate state if requested.
          if (numCalls++ % WorkerStateRefreshFrequency == 0)
          {
            // Possibly dispose old state.
            if (numCalls > 0 && DisposeStatesWhenDone && state is IDisposable)
            {
              ((IDisposable)state).Dispose();
            }

            // Recreate new state.
            lock (lockObj)
            {
              Console.WriteLine("Creating state for worker " + workerIndex);
            }
            state = states[workerIndex] = CreateWorkerStateFunc();
          }

          // Invoke action.
          Action(item, state);
        }
        catch (Exception ex)
        {
          Console.WriteLine("Exception in worker thread " + workerIndex + " " + ex);
        }
      }
    }

  }
}

  
