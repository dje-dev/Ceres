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

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Maintains a pool (of specified maximum size) of objects of specified type
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class ObjectPool<T> where T : class
  {
    public readonly int MaxObjectsInPool;
    public readonly Func<T> Generator;

    #region Private data

    static int numCreated = 0;
    static BlockingCollection<T> poolObjects = new BlockingCollection<T>();

    #endregion

    /// <summary>
    /// Constructor (for a given maximum number if items im pool)
    /// </summary>
    /// <param name="maxItemsInPool"></param>
    public ObjectPool(Func<T> generator, int maxItemsInPool)
    {
      Generator = generator;
      MaxObjectsInPool = maxItemsInPool;
    }

    // --------------------------------------------------------------------------------------------
    /// <summary>
    /// Gets an object from the pool (blocking if necessary to wait for available object)
    /// </summary>
    /// <param name="generator"></param>
    /// <param name="maxObjects"></param>
    /// <returns></returns>
    public T GetFromPool()
    {
      // Return immediately if we have an available object
      if (poolObjects.TryTake(out T poolObj)) return poolObj;

      // Create another object if we are below the maximum
      if (numCreated < MaxObjectsInPool)
      {
        lock (poolObjects)
        {
          // Verify again now that we hold the lock
          if (numCreated < MaxObjectsInPool)
          {
            poolObjects.Add(Generator());
            Interlocked.Increment(ref numCreated);
          }
        }
      }

      // Return object from pool (blocking if necessary to wait for an available one)
      return poolObjects.Take();
    }

    /// <summary>
    /// Puts back an object into the pool.
    /// </summary>
    /// <param name="object"></param>
    public void RestoreToPool(T obj) => poolObjects.Add(obj);


    public void Shutdown(Action<T> shutdownAction)
    {
      while (poolObjects.TryTake(out T item))
        shutdownAction?.Invoke(item);
      poolObjects = null;
    }

  }
}
