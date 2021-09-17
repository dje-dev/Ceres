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
using System.Threading.Tasks;

#endregion

namespace Ceres.Base.Threading
{
  /// <summary>
  /// Manages a queue into which items can be enqueued
  /// for background processing (calling specified method)
  /// on a dedicated thread.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class ItemProcessorThread<T> where T : struct
  {
    Action<T> processor;
    BlockingCollection<T> pendingItems;
    ManualResetEvent doneEvent = new ManualResetEvent(false);


    /// <summary>
    /// Constructor for a processor thread with 
    /// a pending item queue of specified maximum size.
    /// </summary>
    public ItemProcessorThread(Action<T> processor, int queueItems = 1024)
    {
      pendingItems = new(queueItems);
      this.processor = processor;
      Task workerTask = Task.Factory.StartNew(Worker);
    }


    /// <summary>
    /// Adds a specified item to the queue of items to process.
    /// </summary>
    public void Add(T t)
    {
      pendingItems.Add(t);
    }


    /// <summary>
    /// Waits until the queue of pending items to process is drained 
    /// and shuts down worker thread.
    /// </summary>
    public void WaitDone()
    {
      pendingItems.CompleteAdding();
      doneEvent.WaitOne();
    }


    #region Internals

    /// <summary>
    /// The worker method that waits and dispatches items.
    /// </summary>
    void Worker()
    {
      while (!pendingItems.IsCompleted)
      {
        T item = default;
        bool wasError = false;
        try
        {
          item = pendingItems.Take();

        }
        catch (InvalidOperationException)
        {
          wasError = true;
        }

        if (!wasError) processor(item);
      }

      doneEvent.Set();
    }

    #endregion 
  }

}
