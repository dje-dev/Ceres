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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// A non-reentrant spin-lock (1 byte).
/// 
/// For debugging purposes some information about the 
/// creating thread is stored in the lock state.  
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 1)]
public struct SpinLockByte
{
  const byte VALUE_UNLOCKED = 0;
  const byte VALUE_LOCKED_NO_THREAD_TRACKING = 1;
  const byte VALUE_ILLEGAL = byte.MaxValue;

  // Valid tracked range: [2..253] inclusive (252 distinct values).
  const byte VALUE_TRACKED_MIN = 2;
  const byte VALUE_TRACKED_MAX = 253;
  const int VALUE_TRACKED_RANGE = VALUE_TRACKED_MAX - VALUE_TRACKED_MIN + 1;

  /// <summary>
  /// Internal state of the lock:
  ///   0   = free
  ///   1   = held (no thread tracking)
  ///   2..253 = held with tracked owner (bucketized CurrentManagedThreadId)
  ///   255 = illegal value (for debugging)
  /// </summary>
  internal byte state;

  /// <summary>
  /// Returns the raw state value (for diagnostics only).
  /// </summary>
  public readonly byte StateRaw => state;

#if DEBUG
  const bool CHECK_THREAD_REENTRANCY = true;
#else
  const bool CHECK_THREAD_REENTRANCY = false;
#endif

  /// <summary>
  /// Acquires the lock.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public void Acquire()
  {
    Debug.Assert(state != VALUE_ILLEGAL, "AcquireFoundIllegal");

    // Only assert on reentrancy if the lock currently appears held.
    if (CHECK_THREAD_REENTRANCY)
    {
      byte observed = Volatile.Read(ref state);
      if (observed != VALUE_UNLOCKED)
      {
        Debug.Assert(!IsKnownLockedByThisThread, "ThreadReacquire");
      }
    }

    int tid = System.Environment.CurrentManagedThreadId;
    byte stateValueToUse =
        (CHECK_THREAD_REENTRANCY && IsTrackableThreadId(tid))
          ? StateValueToUseForThreadID(tid)
          : VALUE_LOCKED_NO_THREAD_TRACKING;

    // Fast path attempt.
    if (Interlocked.CompareExchange(ref state, stateValueToUse, VALUE_UNLOCKED) == VALUE_UNLOCKED)
    {
      return;
    }

    // Contended path with backoff; still busy-waiting (no kernel wait).
    SpinWait sw = new();
    while (true)
    {
      // Spin while observed held, to reduce cache-line bouncing.
      while (Volatile.Read(ref state) != VALUE_UNLOCKED)
      {
        sw.SpinOnce(-1);
      }

      if (Interlocked.CompareExchange(ref state, stateValueToUse, VALUE_UNLOCKED) == VALUE_UNLOCKED)
      {
        return;
      }
    }
  }

  /// <summary>
  /// Maps any positive thread ID into the tracked range [2..253].
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static byte StateValueToUseForThreadID(int threadID)
  {
    Debug.Assert(threadID > 0);
    return (byte)(VALUE_TRACKED_MIN + ((threadID - 1) % VALUE_TRACKED_RANGE));
  }


  /// <summary>
  /// Returns true if thread tracking is feasible for this thread ID.
  /// All positive thread IDs can be tracked (mapped into the bucket range).
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static bool IsTrackableThreadId(int threadID) => threadID > 0;


  /// <summary>
  /// Releases the lock.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public void Release()
  {
    Debug.Assert(IsLocked, "ReleaseFoundUnlocked");
    Debug.Assert(state != VALUE_ILLEGAL, "ReleaseFoundIllegal");

    if (CHECK_THREAD_REENTRANCY)
    {
      Debug.Assert(IsPossiblyLockedByThisThread, "ReleaseFromWrongThread");
    }

    // Clear the state (unlock).
    Volatile.Write(ref state, VALUE_UNLOCKED);
    // or: Interlocked.Exchange(ref state, (byte)0);
  }


  /// <summary>
  /// Indicates whether any thread currently holds the lock.
  /// </summary>
  public bool IsLocked => Volatile.Read(ref state) != VALUE_UNLOCKED;


  #region Internal integrity support

  /// <summary>
  /// Disallow copy constructor (which would break the synchronization semantics).
  /// </summary>
  /// <param name="_"></param>
  /// <exception cref="InvalidOperationException"></exception>
  [Obsolete("Do not copy SpinLockByte. Use the original instance (field) or pass by ref.", true)]
  public SpinLockByte(SpinLockByte _) => throw new InvalidOperationException();


  public void SetIllegalValue()
  {
    Debug.Assert(!IsLocked);
    state = VALUE_ILLEGAL;
  }

  /// <summary>
  /// Conservative check used on Release assertions.
  /// </summary>
  public bool IsPossiblyLockedByThisThread
  {
    get
    {
      if (!CHECK_THREAD_REENTRANCY)
      {
        return true;
      }

      byte s = Volatile.Read(ref state);
      if (s < VALUE_TRACKED_MIN || s > VALUE_TRACKED_MAX)
      {
        return true; // not tracked, so anything is possible
      }

      int tid = System.Environment.CurrentManagedThreadId;
      return s == StateValueToUseForThreadID(tid);
    }
  }

  /// <summary>
  /// If it is known that this node's lock is held by the current thread
  /// (but it is not always possible to determine this due to hash collisions).
  /// </summary>
  public bool IsKnownLockedByThisThread
  {
    get
    {
      if (!CHECK_THREAD_REENTRANCY)
      {
        return false;
      }

      byte s = Volatile.Read(ref state);
      if (s < VALUE_TRACKED_MIN || s > VALUE_TRACKED_MAX)
      {
        return false; // outside tracked range, can't prove ownership
      }

      int tid = System.Environment.CurrentManagedThreadId;
      return s == StateValueToUseForThreadID(tid);
    }
  }

  #endregion

  /// <summary>
  /// Returns a readable representation of the lock state.
  /// </summary>
  public override string ToString() => IsLocked ? $"Locked" : "Unlocked";
}
