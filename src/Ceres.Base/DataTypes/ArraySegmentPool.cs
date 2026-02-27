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
/// Manages a pool of items that can be allocated in contiguous segments.
/// </summary>
/// <typeparam name="T">The value type of items stored in the pool.</typeparam>
public sealed class ArraySegmentPool<T> where T : struct
{
  /// <summary>
  /// Maximum total number of items that can be stored in the pool.
  /// The underlying use of InlineArray necessitates a fixed maximum number of items.
  /// This number should be chosen to prevent overflow in most scenarios.
  /// It will be larger for bigger batches and longer paths (deeper search),
  /// also considering internal fragmentation due to allocation in blocks of <see cref="GROWTH_QUANTUM"/>.
  /// 
  /// However overflowing is not fatal because the search engine monitors for 
  /// batches approaching overflow and will stop the batch early if necessary.
  /// </summary>
  public const int MAX_ITEMS = 22 * 1024;


  /// <summary>
  /// Number of items by which slot segments are grown at a time.
  /// N.B. Runtime speed is quite sensitive to this number.
  ///      For example, 6 is much better than 4 for large graphs (e.g. 10mm nodes).
  /// </summary>
  public const int GROWTH_QUANTUM = 8;


  /// <summary>
  /// The inline array used to store items.
  /// </summary>
  [InlineArray(MAX_ITEMS)]
  private struct InlineBuffer { private T item0; }


  /// <summary>
  /// Inline fixed-size array of items.
  /// </summary>
  private InlineBuffer buffer;


  /// <summary>
  /// Index of the next free slot in the pool (also equals total items allocated).
  /// Modified atomically via <see cref="Interlocked"/> operations.
  /// </summary>
  private int nextFreeIndex;


  /// <summary>
  /// Allocates a new segment of the given number of items 
  /// (rounded up to the nearest multiple of <see cref="GROWTH_QUANTUM"/>).
  /// This method is thread-safe and lock-free.
  /// </summary>
  /// <param name="itemCount">Requested number of items, or null to use <see cref="GROWTH_QUANTUM"/>.</param>
  /// <returns>A reference to the allocated segment.</returns>
  public ArraySegmentRef<T> AllocateSegment(int? itemCount)
  {
    int requestedCount = itemCount ?? GROWTH_QUANTUM;
    int capacity = RoundUp(requestedCount);

    // Use compare-exchange loop to ensure we don't allocate past MAX_ITEMS.
    // This prevents a race where multiple threads could all increment past the limit.
    int currentFree, newFree;
    do
    {
      currentFree = Volatile.Read(ref nextFreeIndex);
      newFree = currentFree + capacity;

      if (newFree > MAX_ITEMS)
      {
        throw new InvalidOperationException($"ArraySegmentPool overflow: requested {capacity} items at index {currentFree}, but max is {MAX_ITEMS}.");
      }
    }
    while (Interlocked.CompareExchange(ref nextFreeIndex, newFree, currentFree) != currentFree);

    return new ArraySegmentRef<T>(this, currentFree, capacity);
  }


  /// <summary>
  /// Returns a span representing a slice of the buffer.
  /// </summary>
  /// <param name="start">The starting index within the buffer.</param>
  /// <param name="length">The number of items in the slice.</param>
  /// <returns>A span over the specified range.</returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  internal Span<T> Slice(int start, int length)
  {
    Debug.Assert((uint)start < MAX_ITEMS && (uint)(start + length) <= MAX_ITEMS);

    ref T first = ref buffer[start];
    return MemoryMarshal.CreateSpan(ref first, length);
  }


  /// <summary>
  /// Returns a reference to the item at the given absolute index.
  /// </summary>
  /// <param name="absoluteIndex">The index within the pool's buffer.</param>
  /// <returns>A reference to the item at the specified index.</returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  internal ref T ItemAt(int absoluteIndex)
  {
    return ref buffer[absoluteIndex];
  }


  /// <summary>
  /// Returns the number of items currently allocated.
  /// </summary>
  public int Allocated => nextFreeIndex;


  /// <summary>
  /// Returns the fraction of the pool that is currently in use.
  /// </summary>
  public float FractionInUse => (float)Allocated / MAX_ITEMS;


  /// <summary>
  /// Clears the pool, releasing all allocated items.
  /// WARNING: This method is NOT thread-safe. Ensure no other threads are accessing the pool.
  /// </summary>
  /// <param name="clearMem">If true, zeros out the memory of all allocated items.</param>
  public void Clear(bool clearMem = true)
  {
    int allocated = nextFreeIndex;
    if (allocated == 0)
    {
      return;
    }

    if (clearMem)
    {
      Slice(0, allocated).Clear();
    }

    nextFreeIndex = 0;
  }


  /// <summary>
  /// Rounds up the given number to the nearest multiple of <see cref="GROWTH_QUANTUM"/>.
  /// </summary>
  /// <param name="n">The value to round up.</param>
  /// <returns>The smallest multiple of <see cref="GROWTH_QUANTUM"/> that is >= <paramref name="n"/>.</returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  internal static int RoundUp(int n)
  {
    const bool POWER_OF_TWO = (GROWTH_QUANTUM & (GROWTH_QUANTUM - 1)) == 0;

    if (POWER_OF_TWO)
    {
      return (n + GROWTH_QUANTUM - 1) & ~(GROWTH_QUANTUM - 1);
    }
    else
    {
      int q = GROWTH_QUANTUM;
      return ((n + q - 1) / q) * q;
    }
  }
}
