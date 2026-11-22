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

using Ceres.Base.OperatingSystem.Linux;
using System;
using System.Diagnostics;
using System.Threading;

#endregion

namespace Ceres.Base.OperatingSystem
{
  public static class RawMemoryManagerIncrementalLinuxStats
  {
    /// <summary>
    /// Current number of bytes allocated.
    /// </summary>
    public static long BytesCurrentlyAllocated = 0;

    /// <summary>
    /// Maximum number of bytes ever allocated at any point in time.
    /// </summary>
    public static long MaxBytesAllocated = 0;
  }

  /// <summary>
  /// Linux memory allocation class which uses incrementally allocated blocks of memory.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  internal unsafe class RawMemoryManagerIncrementalLinux<T> : IRawMemoryManagerIncremental<T> where T : unmanaged
  {
    void* rawMemoryPointer;

    void* IRawMemoryManagerIncremental<T>.RawMemoryAddress => rawMemoryPointer;

    public long NumItemsReserved { get; private set; }
    public long NumBytesReserved { get; private set; }

    long numBytesAllocated = 0;


    const long ALLOCATE_INCREMENTAL_BYTES = 1024 * 1024 * 4;
    const int PAGE_SIZE = 1024 * 2048;

    static long RoundToHugePageSize(long numBytes)
    {
      // TO DO: determine the true page size at runtime (what if possibly 1GB huge pages?) 
      const int HUGE_PAGE_SIZE = 1024 * 2048;
      return (((numBytes - 1) / HUGE_PAGE_SIZE) + 1) * HUGE_PAGE_SIZE;
    }


    /// <summary>
    /// If reservations should ask the kernel to back this range with transparent huge pages.
    ///
    /// The graph stores are multi-gigabyte and accessed in essentially random order, so with 4 KB
    /// pages most node visits can take a TLB miss whose page walk is itself a cache miss. A 2 MB
    /// page raises the reach of the same number of TLB entries by ~512x.
    ///
    /// This is only an advisory hint, distinct from MAP_HUGETLB (UseLargePages) which needs a
    /// pre-reserved hugetlbfs pool and fails when it is absent. Costs nothing where unsupported:
    /// the call is a no-op on kernels without THP and simply returns an error which is ignored,
    /// and it is never reached on non-Linux platforms (MemoryBufferOS selects the Windows manager
    /// there). Set to false to reserve with ordinary pages.
    ///
    /// CAVEAT: where /sys/kernel/mm/transparent_hugepage/defrag is "madvise" (the Ubuntu
    /// default), advised regions compact SYNCHRONOUSLY in the page fault path, so on a
    /// fragmented machine the mprotect growth below can stall mid-search. If per-move times
    /// become erratic, try: echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
    ///
    /// Verify it is taking effect during a search with:
    ///   grep AnonHugePages /proc/&lt;pid&gt;/smaps_rollup
    /// </summary>
    internal const bool USE_TRANSPARENT_HUGE_PAGES = false;


    static bool largePageAllocationEverFailed = false;

    /// <summary>
    /// Whether the successful reservation actually requested MAP_HUGETLB.
    /// Distinct from the useLargePages argument, which is only what the caller asked for:
    /// the request is skipped when a previous reservation already found the hugetlbfs pool
    /// unavailable, and is abandoned when the huge page attempt fails and we fall back.
    /// </summary>
    bool usesLargePages = false;

    public void Reserve(string sharedMemName, bool useExistingSharedMemory, long numItems, bool useLargePages)
    {
      if (rawMemoryPointer != null)
      {
        throw new Exception("Internal error: Reserve should be called only once");
      }

      if (useExistingSharedMemory)
      {
        throw new NotImplementedException("Use existing shared memory not yet implemented under Linux");
      }

      NumItemsReserved = numItems;

      // We overreserve by one PAGE_SIZE since the OS may not allow partial page allocation
      NumBytesReserved = RoundToHugePageSize(numItems * sizeof(T) + PAGE_SIZE);

      int mapFlags = LinuxAPI.MAP_NORESERVE | LinuxAPI.MAP_PRIVATE | LinuxAPI.MAP_ANONYMOUS;

      // Huge pages are requested only if the caller asked AND no earlier reservation already
      // found the hugetlbfs pool unavailable (after which every later attempt would fail too).
      bool attemptedLargePages = useLargePages && !largePageAllocationEverFailed;
      if (attemptedLargePages)
      {
        mapFlags |= LinuxAPI.MAP_HUGETLB;
      }

      IntPtr mapPtr = (IntPtr)LinuxAPI.mmap(null, NumBytesReserved, LinuxAPI.PROT_NONE, mapFlags, -1, 0);
      if (mapPtr.ToInt64() == -1)
      {
        // Only meaningful to retry when this attempt actually asked for huge pages; otherwise
        // the failure has some other cause and dropping a flag that was never set cannot help.
        if (attemptedLargePages)
        {
          // Clear the bit rather than toggling it: XOR would ADD MAP_HUGETLB back on a retry
          // of a mapping which had not requested it.
          mapFlags &= ~LinuxAPI.MAP_HUGETLB;

          mapPtr = (IntPtr)LinuxAPI.mmap(null, NumBytesReserved, LinuxAPI.PROT_NONE, mapFlags, -1, 0);
          if (mapPtr.ToInt64() != -1)
          {
            attemptedLargePages = false;
            largePageAllocationEverFailed = true;
            Console.WriteLine("NOTE: Attempt to allocate large page failed, falling back to non-large pages.");
          }
        }

        if (mapPtr.ToInt64() == -1)
        {
          throw new Exception($"Virtual memory reservation of {NumBytesReserved} bytes failed using mmap.");
        }
      }

      // Reflects the mapping which actually succeeded, not merely what the caller requested.
      usesLargePages = attemptedLargePages;

      if (USE_TRANSPARENT_HUGE_PAGES)
      {
        // Advisory only: failure (no THP support, or THP set to "never") is expected on some
        // systems and simply leaves the range on ordinary pages, so the result is ignored.
        // The reservation is already rounded to a 2 MB boundary, and the advice is a property
        // of the mapping which survives the mprotect calls that commit pages as the graph grows.
        LinuxAPI.madvise((void*)mapPtr, NumBytesReserved, LinuxAPI.MADV_HUGEPAGE);
      }

      rawMemoryPointer = (void*)mapPtr;
    }


    public void InsureAllocated(long numItems)
    {
      if (numItems > NumItemsReserved)
      {
        throw new ArgumentException($"Allocation overflow, requested {numItems} but maximum was set as {NumItemsReserved}");
      }

      long numBytesNeeded = numItems * sizeof(T) + PAGE_SIZE; // overallocate to avoid partial page access
      numBytesNeeded = RoundToHugePageSize(numBytesNeeded);

      if (numBytesNeeded > numBytesAllocated)
      {
        // Grow to at least the number of bytes actually needed, but by no less than the
        // standard incremental amount (to keep the number of mprotect syscalls modest).
        long newBytesAllocated = System.Math.Max(numBytesNeeded, numBytesAllocated + ALLOCATE_INCREMENTAL_BYTES);
        newBytesAllocated = RoundToHugePageSize(newBytesAllocated);

        // Never extend beyond the region that was reserved at construction.
        if (newBytesAllocated > NumBytesReserved)
        {
          newBytesAllocated = NumBytesReserved;
        }

        int resultCode = LinuxAPI.mprotect(rawMemoryPointer, newBytesAllocated, LinuxAPI.PROT_READ | LinuxAPI.PROT_WRITE);
        if (resultCode != 0)
        {
          throw new Exception($"Virtual memory extension to size {newBytesAllocated} failed with error {resultCode}");
        }

        long bytesNewlyCommitted = newBytesAllocated - numBytesAllocated;

        // Publish the new committed size only after mprotect has succeeded.
        Volatile.Write(ref numBytesAllocated, newBytesAllocated);

        Interlocked.Add(ref RawMemoryManagerIncrementalLinuxStats.BytesCurrentlyAllocated, bytesNewlyCommitted);
        if (RawMemoryManagerIncrementalLinuxStats.BytesCurrentlyAllocated > RawMemoryManagerIncrementalLinuxStats.MaxBytesAllocated)
        {
          RawMemoryManagerIncrementalLinuxStats.MaxBytesAllocated = RawMemoryManagerIncrementalLinuxStats.BytesCurrentlyAllocated;
        }
      }
    }

    public long NumItemsAllocated => Volatile.Read(ref numBytesAllocated) / sizeof(T);

    public void Dispose()
    {
      int resultCode = LinuxAPI.munmap(rawMemoryPointer, NumBytesReserved);
      if (resultCode != 0)
      {
        throw new Exception($"Virtual memory munmap of size {NumBytesReserved} failed with error {resultCode}");
      }

      Interlocked.Add(ref RawMemoryManagerIncrementalLinuxStats.BytesCurrentlyAllocated, -numBytesAllocated);

      numBytesAllocated = 0;
      NumBytesReserved = 0;
      NumItemsReserved = 0;
      rawMemoryPointer = null;
    }

    public void ResizeToNumItems(long numItems)
    {
      Debug.Assert(numItems <= NumItemsAllocated);

      long numBytesNeeded = numItems * sizeof(T) + PAGE_SIZE; // overallocate to avoid partial page access
      numBytesNeeded = RoundToHugePageSize(numBytesNeeded);
      numItems = numBytesNeeded / sizeof(T);

      if (numBytesNeeded < numBytesAllocated)
      {
        long freeBlocksStart = ((IntPtr)rawMemoryPointer).ToInt64() + numBytesNeeded;
        long itemsFree = NumItemsAllocated - numItems;
        long bytesFree = itemsFree * sizeof(T);

        int resultCode = LinuxAPI.mprotect((void*)freeBlocksStart, bytesFree, LinuxAPI.PROT_NONE);
        if (resultCode != 0)
        {
          throw new Exception($"Virtual memory mprotect to decommit size {bytesFree} failed with error {resultCode}");
        }

        Interlocked.Add(ref RawMemoryManagerIncrementalLinuxStats.BytesCurrentlyAllocated, -bytesFree);

        numBytesAllocated -= bytesFree;
      }
    }
  }

}
