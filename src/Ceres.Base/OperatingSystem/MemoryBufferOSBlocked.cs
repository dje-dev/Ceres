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
using System.Runtime.InteropServices;
using System.Threading;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Generic base class that manages blocks of items of an unmanaged type T.
  /// </summary>
  [Serializable]
  public class MemoryBufferOSBlocked<T> where T : unmanaged
  {
    /// <summary>
    /// Underlying OS memory buffer that stores items.
    /// </summary>
    protected readonly MemoryBufferOS<T> entries;


    /// <summary>
    /// Number of items per block.
    /// </summary>
    protected readonly int itemsPerBlock;

    /// <summary>
    /// Extra items used for padding.
    /// </summary>
    protected readonly int bufferExtraItems;

    /// <summary>
    /// Next free block index. Block 0 is reserved.
    /// </summary>
    protected int nextFreeBlockIndex = 1;

    /// <summary>
    /// Object used for locking when resizing.
    /// </summary>
    readonly object lockObj = new();


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxItems"></param>
    /// <param name="itemsPerBlock"></param>
    /// <param name="bufferExtraItems"></param>
    /// <param name="tryEnableLargePages"></param>
    /// <param name="sharedMemName"></param>
    /// <param name="useExistingSharedMem"></param>
    /// <param name="useIncrementalAlloc"></param>
    public MemoryBufferOSBlocked(long maxItems, int itemsPerBlock, int bufferExtraItems,
                                 bool tryEnableLargePages, string sharedMemName,
                                 bool useExistingSharedMem, bool useIncrementalAlloc)
    {
      if (Marshal.SizeOf<T>() * bufferExtraItems < 256 * 1024)
      {
        throw new ArgumentException("BufferExtraItems must yield size of at least 256 kbytes to avoid excessive OS calls.");
      }

      this.itemsPerBlock = itemsPerBlock;
      this.bufferExtraItems = bufferExtraItems;

      long maxItemsWithBuffer = maxItems + bufferExtraItems;
      entries = new MemoryBufferOS<T>(maxItemsWithBuffer, tryEnableLargePages, sharedMemName,
                                     useExistingSharedMem, useIncrementalAlloc);
    }


    /// <summary>
    /// Returns underlying memory buffer.
    /// </summary>
    public MemoryBufferOS<T> Entries { get => entries; }


    /// <summary>
    /// Gets a pointer to the raw underlying memory.
    /// </summary>
    public unsafe void* RawMemory => entries.RawMemory;

    /// <summary>
    /// Gets the total number of items allocated (by block).
    /// </summary>
    public long NumAllocatedItems => (long)nextFreeBlockIndex * itemsPerBlock;

    /// <summary>
    /// Copies entries from one block area to another.
    /// </summary>
    public void CopyEntries(long sourceBlockIndex, long destinationBlockIndex, int numItems)
        => entries.CopyEntries(sourceBlockIndex * itemsPerBlock,
                              destinationBlockIndex * itemsPerBlock,
                              numItems);

    /// <summary>
    /// Ensures that the buffer has allocated space for at least the specified number of items.
    /// </summary>
    public void InsureAllocated(long numItems) => entries.InsureAllocated(numItems);


    /// <summary>
    /// Resizes the underlying memory to exactly the number of currently used items.
    /// </summary>
    public void ResizeToCurrent() => ResizeToNumItems((long)nextFreeBlockIndex * itemsPerBlock);


    /// <summary>
    /// Resizes the underlying memory block to commit only the specified number of items.
    /// </summary>
    /// <exception cref="ArgumentException"></exception>
    protected void ResizeToNumItems(long numItems)
    {
      if (numItems < nextFreeBlockIndex)
      {
        throw new ArgumentException("Attempt to resize to size smaller than current number of used items.");
      }
      else if (numItems > entries.NumItemsAllocated)
      {
        throw new ArgumentException("Attempt to resize to size larger than allocated.");
      }
      entries.ResizeToNumItems(numItems);
    }

    /// <summary>
    /// Computes the number of blocks needed to hold the specified number of items.
    /// </summary>
    protected int NumBlocksReservedForNumItems(int numItems)
    {
      bool fitsExactly = numItems % itemsPerBlock == 0;
      return numItems / itemsPerBlock + (fitsExactly ? 0 : 1);
    }


    /// <summary>
    /// Allocates enough blocks to hold the given number of new items.
    /// Returns the starting block index of the allocation.
    /// </summary>
    public long AllocateEntriesStartBlock(int numItems)
    {
      int numBlocksRequired = NumBlocksReservedForNumItems(numItems);
      long newNextFreeBlockIndex = Interlocked.Add(ref nextFreeBlockIndex, numBlocksRequired);

      // Compute the new required allocation including extra padding.
      long newNumEntriesWithPadding = newNextFreeBlockIndex * itemsPerBlock + bufferExtraItems;

      // Check if we need to allocate (optimistic check without lock)
      if (entries.NumItemsAllocated < newNumEntriesWithPadding)
      {
        lock (lockObj)
        {
          // Re-check inside lock - another thread might have already allocated
          if (entries.NumItemsAllocated < newNumEntriesWithPadding)
          {
            entries.InsureAllocated(newNumEntriesWithPadding);
          }
        }
      }

      return newNextFreeBlockIndex - numBlocksRequired;
    }


    /// <summary>
    /// Returns a span covering a sequence of items starting at the given absolute item index.
    /// </summary>
    public Span<T> SpanAtIndex(long startIndex, int count)
    {
      if (count == 0)
      {
        return Span<T>.Empty;
      }
      return entries.Slice(startIndex, count);
    }

    /// <summary>
    /// Returns a span covering a sequence of items starting at the beginning of the given block.
    /// </summary>
    public Span<T> SpanAtBlockIndex(long blockIndex, int count)
    {
      if (count == 0)
      {
        return Span<T>.Empty;
      }
      return entries.Slice((blockIndex) * itemsPerBlock, count);
    }


    /// <summary>
    /// Releases storage associated with the store.
    /// </summary>
    public virtual void Deallocate()
    {
      entries.Dispose();
    }


    public override string ToString() =>
        $"<MemoryBufferOSBlocked NumAllocatedItems={NumAllocatedItems} UsedItems~{nextFreeBlockIndex * itemsPerBlock}>";
  }
}
