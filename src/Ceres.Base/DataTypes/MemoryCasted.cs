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
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// Static helper class for casting Memory{TSrc} to Memory{TDest} without copying.
/// </summary>
public static class MemoryCasted
{
  /// <summary>
  /// Returns a Memory{TDest} that reinterprets the given array of TSrc as TDest.
  /// </summary>
  /// <typeparam name="TSrc"></typeparam>
  /// <typeparam name="TDest"></typeparam>
  /// <param name="array"></param>
  /// <returns></returns>
  public static Memory<TDest> AsMemory<TSrc, TDest>(TSrc[] array)
      where TSrc : unmanaged
      where TDest : unmanaged
  {
    if (array is null)
    {
      return Memory<TDest>.Empty;
    }

    if (Unsafe.SizeOf<TSrc>() != Unsafe.SizeOf<TDest>())
    {
      throw new NotSupportedException("TSrc and TDest must be the same size.");
    }

    return new ReinterpretingArrayMemoryManager<TSrc, TDest>(array).Memory;
  }


  private sealed class ReinterpretingArrayMemoryManager<TFrom, TTo> : MemoryManager<TTo>
      where TFrom : unmanaged
      where TTo : unmanaged
  {
    private readonly TFrom[] array;
    private MemoryHandle pinned;
    private bool isPinned;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="array"></param>
    public ReinterpretingArrayMemoryManager(TFrom[] array)
    {
      this.array = array ?? Array.Empty<TFrom>();
    }

    public override Span<TTo> GetSpan()
    {
      if (array.Length == 0)
      {
        return Span<TTo>.Empty;
      }

      // Reinterpret ref to first element, then create a span of same element count.
      ref TFrom srcRef = ref MemoryMarshal.GetArrayDataReference(array);
      ref TTo dstRef = ref Unsafe.As<TFrom, TTo>(ref srcRef);
      return MemoryMarshal.CreateSpan(ref dstRef, array.Length);
    }


    /// <summary>
    /// Pins the memory and returns a handle.
    /// </summary>
    /// <param name="elementIndex"></param>
    /// <returns></returns>
    public override unsafe MemoryHandle Pin(int elementIndex = 0)
    {
      if (isPinned)
      {
        throw new InvalidOperationException("Already pinned.");
      }

      if ((uint)elementIndex > (uint)array.Length)
      {
        throw new ArgumentOutOfRangeException(nameof(elementIndex));
      }

      // Pin the source array and adjust the pointer for the TTo element index.
      pinned = new Memory<TFrom>(array).Pin();
      isPinned = true;

      byte* basePtr = (byte*)pinned.Pointer;
      byte* adjPtr = basePtr + (nuint)elementIndex * (nuint)Unsafe.SizeOf<TTo>();

      // Tie lifetime to this manager so Unpin() is called on dispose.
      return new MemoryHandle(adjPtr, default, this);
    }

    /// <summary>
    /// Unpins the memory.
    /// </summary>
    public override void Unpin()
    {
      if (isPinned)
      {
        pinned.Dispose();
        isPinned = false;
      }
    }

    protected override void Dispose(bool disposing)
    {
      if (disposing && isPinned)
      {
        pinned.Dispose();
        isPinned = false;
      }
    }

  }
}
