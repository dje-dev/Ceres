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
using System.Linq;
using System.Collections.Concurrent;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// A pool of sequential integer IDs starting from 0 up to some maximum,
  /// with methods to allocate or restore IDs from the pool in a thread-safe way.
  /// </summary>
  public class IDPool
  {
    /// <summary>
    /// An identifying string for the pool (for diagnostic purposes).
    /// </summary>
    public readonly string PoolID;

    /// <summary>
    /// Maximum number of sequential integers starting at 0 with which pool is initialized.
    /// </summary>
    public readonly int MaxIDs;

    ConcurrentBag<int> availableIDs;

    /// <summary>
    /// Create a pool of sequential IDs starting at 0.
    /// </summary>
    public IDPool(string poolID, int maxIDs)
    {
      MaxIDs = maxIDs;
      PoolID = poolID;

      availableIDs = new ConcurrentBag<int>(Enumerable.Range(0, maxIDs));
    }

    /// <summary>
    /// Retrieve a free ID from the pool
    /// </summary>
    public int GetFreeID()
    {
      if (!availableIDs.TryTake(out int sessionID))
        throw new Exception("Maximum number IDs {MaxIDs} exceeded for ID pool {PoolID}");

      return sessionID;
    }

    /// <summary>
    /// Return an ID to the pool.
    /// </summary>
    public void ReleaseID(int id) => availableIDs.Add(id);
  }
}
