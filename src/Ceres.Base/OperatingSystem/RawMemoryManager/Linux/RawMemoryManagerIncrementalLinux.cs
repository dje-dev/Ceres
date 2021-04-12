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

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Linux memory allocation class which uses incrementally allocated blocks of memory.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  internal unsafe class RawMemoryManagerIncrementalLinux<T> : IRawMemoryManagerIncremental<T> where T : unmanaged
  {
    void* rawMemoryPointer;

    void* IRawMemoryManagerIncremental<T>.RawMemoryAddress => rawMemoryPointer;

    long numItemsAllocated;
    long numBytesAllocated = 0;

    const long ALLOCATE_INCREMENTAL_BYTES = 1024 * 1024 * 2;
    const int PAGE_SIZE = 1024 * 2048;//4096;

    static long RoundToHugePageSize(long numBytes)
    {
      // TO DO: determine the true page size at runtime (what if possibly 1GB huge pages?) 
      const int HUGE_PAGE_SIZE = 1024 * 2048;
      return (((numBytes - 1) / HUGE_PAGE_SIZE) + 1) * HUGE_PAGE_SIZE;
    }


    public void Reserve(string sharedMemName, bool useExistingSharedMemory, long numItems, bool? useLargePages = true)
    {
      // By default we enable large pages under Linux, typically
      // the OS will fallback to non-large pages if necessary.
      useLargePages = useLargePages ?? true;

      if (rawMemoryPointer != null) throw new Exception("Internal error: Reserve should be called only once");
      if (useExistingSharedMemory) throw new NotImplementedException("Use existing shared memory not yet implemented under Linux");

      // We overreserve by one PAGE_SIZE since the OS may not allow partial page allocation
      long numBytes = numItems * sizeof(T) + PAGE_SIZE;
      numBytes = RoundToHugePageSize(numBytes);

      int mapFlags = LinuxAPI.MAP_NORESERVE | LinuxAPI.MAP_PRIVATE | LinuxAPI.MAP_ANONYMOUS;
      if (useLargePages.Value)
      {
        mapFlags |= LinuxAPI.MAP_HUGETLB;
      };

      rawMemoryPointer = (float*)LinuxAPI.mmap(null, numBytes, LinuxAPI.PROT_NONE, mapFlags, -1, 0);

      if (rawMemoryPointer == null) throw new Exception($"Virtual memory reservation of {numBytes} bytes failed using mmap");
    }


    public void InsureAllocated(long numItems)
    {
      long numBytesNeeded = numItems * sizeof(T) + PAGE_SIZE; // overallocate to avoid partial page access
      numBytesNeeded = RoundToHugePageSize(numBytesNeeded);

      if (numBytesNeeded > numBytesAllocated)
      {
        numBytesAllocated += ALLOCATE_INCREMENTAL_BYTES;
        numItemsAllocated = numItems;

        int resultCode = LinuxAPI.mprotect(rawMemoryPointer, numBytesAllocated, LinuxAPI.PROT_READ | LinuxAPI.PROT_WRITE);
        if (resultCode != 0) throw new Exception($"Virtual memory extension to size {numBytesAllocated} failed with error {resultCode}");
      }
    }

    public long NumItemsAllocated => this.numBytesAllocated / sizeof(T); // TODO: make faster?

    public void Dispose()
    {
      LinuxAPI.munmap(rawMemoryPointer, numBytesAllocated);
      numBytesAllocated = 0;
      rawMemoryPointer = null;
    }

  }

}
