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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ceres.Base.DataTypes;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// Represents a reference to a segment of an array managed by an <see cref="ArraySegmentPool{T}"/>,
/// providing access to a contiguous region of elements.
/// </summary>
/// <typeparam name="T">The value type of items stored in the segment.</typeparam>
public record struct ArraySegmentRef<T> where T : struct
{
  /// <summary>
  /// The pool that owns and manages the underlying buffer.
  /// </summary>
  private readonly ArraySegmentPool<T> owningPool;

  /// <summary>
  /// Index of the first item in the segment within the parent array.
  /// </summary>
  internal int startIndex;

  /// <summary>
  /// Number of items allocated in the segment.
  /// </summary>
  internal int numItemsAllocated;


  /// <summary>
  /// Constructs a new segment reference.
  /// </summary>
  /// <param name="owningPool">The pool that owns the underlying buffer.</param>
  /// <param name="start">The starting index within the pool's buffer.</param>
  /// <param name="capacity">The number of items allocated for this segment.</param>
  internal ArraySegmentRef(ArraySegmentPool<T> owningPool, int start, int capacity)
  {
    this.owningPool = owningPool;
    startIndex = start;
    numItemsAllocated = capacity;
  }


  /// <summary>
  /// Returns the number of items used in the segment.
  /// </summary>
  public int NumItemsAllocated => numItemsAllocated;

  /// <summary>
  /// Returns the starting index of the segment within the parent array.
  /// </summary>
  public int StartIndex => startIndex;


  /// <summary>
  /// Returns a span over the allocated portion of the segment.
  /// </summary>
  public Span<T> Span
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    get => MemoryMarshal.CreateSpan(ref owningPool.ItemAt(startIndex), numItemsAllocated);
  }


  /// <summary>
  /// Gets a reference to the item at the specified index within this segment.
  /// Bounds checking is performed only in DEBUG builds.
  /// </summary>
  /// <param name="index">The zero-based index within the segment.</param>
  /// <returns>A reference to the item at the specified index.</returns>
  public ref T this[int index]
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    get
    {
#if DEBUG
      // Fast unsigned check handles both negative and too-large.
      if ((uint)index >= (uint)numItemsAllocated)
      {
        throw new ArgumentOutOfRangeException(nameof(index));
      }
#endif
      return ref owningPool.ItemAt(startIndex + index);
    }
  }


  /// <summary>
  /// Ensures that the segment has sufficient capacity for a specified number of items.
  /// If the current capacity is insufficient, allocates a new larger segment and copies existing data.
  /// Note: The old segment's memory is not reclaimed (internal fragmentation).
  /// </summary>
  /// <param name="neededItemCount">The minimum required capacity.</param>
  public void EnsureSize(int neededItemCount)
  {
    if (neededItemCount <= numItemsAllocated)
    {
      return;
    }

    // AllocateSegment handles rounding up to GROWTH_QUANTUM.
    ArraySegmentRef<T> bigger = owningPool.AllocateSegment(neededItemCount);

    // Copy existing data to the new segment.
    Span.CopyTo(bigger.Span);

    startIndex = bigger.startIndex;
    numItemsAllocated = bigger.numItemsAllocated;
  }
}
