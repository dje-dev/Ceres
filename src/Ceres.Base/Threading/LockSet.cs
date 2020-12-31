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
using System.Runtime.InteropServices;
using System.Threading;

#endregion

namespace Ceres.Base.Threading
{
  /// <summary>
  /// Coordinates exlusive access to items based on their numerical index,
  /// via a set of multiple locks over which the underlying items 
  /// are distributed to try to avoid collisions and thereby increase concurrency.
  /// 
  /// Optionally either Monitor or SpinLock can be used to provide the locks.
  /// NOTE: the SpinLock version seems to suffer from rare failures, not sure if correct.
  /// </summary>
  public class LockSet
  {
    /// <summary>
    /// The number of underlying locks (and therefore max possible degree of concurrency).
    /// </summary>
    public readonly int NumLocks;

    /// <summary>
    /// If SpinLock should be used for the locks (instead of Monitor).
    /// </summary>
    public readonly bool UseSpinLock = false;

    /// <summary>
    /// Underlying locks array.
    /// </summary>
    readonly LockEntry[] locks;

    /// <summary>
    /// If locking mechamism is active.
    /// In some contexts the access is known to be single threaded 
    /// and this can be transiently set to false to improve performance.
    /// </summary>
    public bool LockingActive = true;


    /// <summary>
    /// Constructor for a set of locks of specified size, optionally using SpinLock implementation.
    /// </summary>
    /// <param name="numLocks"></param>
    /// <param name="useSpinLock"></param>
    public LockSet(int numLocks, bool useSpinLock = false)
    {
      NumLocks = numLocks;
      UseSpinLock = useSpinLock;

      locks = new LockEntry[numLocks];
      for (int i = 0; i < numLocks; i++)
        locks[i].Init(useSpinLock);
    }


    public LockSetBlock LockBlock(int itemIndex) => new LockSetBlock(this, itemIndex);


    public void Acquire(int itemIndex)
    {
      if (LockingActive)
      {
        if (UseSpinLock)
        {
          bool tookLock = false;
          locks[itemIndex % NumLocks].SpinLockObj.Enter(ref tookLock);
          Debug.Assert(tookLock);
        }
        else
          Monitor.Enter(locks[itemIndex % NumLocks].LockObj);
      }
    }


    public void Release(int itemIndex)
    {
      if (LockingActive)
      {
        if (UseSpinLock)
          locks[itemIndex % NumLocks].SpinLockObj.Exit();
        else
          Monitor.Exit(locks[itemIndex % NumLocks].LockObj);
      }
    }
  }

  /// <summary>
  /// Structure containing the lock.
  /// Padded so each lock is definitely on its own cache line to prevent false sharing.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Size = 2 * 64)]
  struct LockEntry
  {
    internal SpinLock SpinLockObj;
    internal object LockObj;

    internal void Init(bool useSpinLock)
    {
      if (useSpinLock)
        SpinLockObj = new SpinLock(false);
      else
        LockObj = new object();
    }
  }

}
