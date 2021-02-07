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

using System.Collections.Generic;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// A Queue with a fixed maximum number of entries
  /// such that the least recently added items are removed
  /// to prvent overflow.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class QueueFixedSize<T>
  {
    /// <summary>
    /// Maximum number of items which can be present in Queue at once.
    /// </summary>
    public int MaxItems { get; init; }

    /// <summary>
    /// A snaphot of the curren items as an Array.
    /// </summary>
    public T[] Items => items.ToArray();


    /// <summary>
    /// The queue items.
    /// </summary>
    Queue<T> items;


    /// <summary>
    /// Consructor for a queue of speciifed maximum size.
    /// </summary>
    /// <param name="maxSize"></param>
    public QueueFixedSize(int maxSize)
    {
      MaxItems = maxSize;
      items = new Queue<T>(maxSize);
    }

    /// <summary>
    /// Adds specifed item to queue.
    /// </summary>
    /// <param name="item"></param>
    public void Enqueue(T item)
    {
      items.Enqueue(item);
      CheckSize();
    }

    
    /// <summary>
    /// Internal worker item that removes overflow items if necessary.
    /// </summary>
    private void CheckSize()
    {
      if (items.Count > MaxItems)
      {
        while (items.Count > MaxItems)
        {
          items.TryDequeue(out T _);
        }
      }
    }
  }
}
