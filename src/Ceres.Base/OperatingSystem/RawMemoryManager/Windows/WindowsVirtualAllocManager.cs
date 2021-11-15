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
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;

using Ceres.Base.OperatingSystem.Windows;
using Ceres.Base.Math;
using Ceres.Base.Environment;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Manages incremental allocations memory for items (with fixed size) from a reserved region.
  /// This implementatation is based on the VirtualAlloc API in Windows.
  /// 
  /// Note that callers can probabl afford to over-reserve memory 
  /// because the virtual address space on Windows has size 256TB
  /// </summary>
  [SupportedOSPlatform("windows")]
  public class WindowsVirtualAllocManager
  {
    /// <summary>
    /// Logger to which log messages sent
    /// </summary>
    public readonly IAppLogger Logger;

    /// <summary>
    /// Maximum number of items to which the table must be guaranteed to be expandable
    /// </summary>
    public long MaxItems;

    /// <summary>
    /// Size in bytes of each item
    /// </summary>
    public int ItemSizeBytes;

    /// <summary>
    /// If the large page feature is requested from the OS
    /// </summary>
    public bool LargePages;

    /// <summary>
    /// 
    /// </summary>
    public long AllocBlockSizeBytes;
    public IntPtr AllocPtr;

    long NumBlocksCommitted;
    long NumItemsCommitted;

    /// <summary>
    /// 
    /// Note that if the item size does not divide block size,
    /// this will be an underestimate (misses fractional part),
    /// causing a harmless and (very small) amount of potential overallocation
    /// </summary>
    internal long ItemsPerBlock => AllocBlockSizeBytes / ItemSizeBytes;

    static Win32SystemInfo.SYSTEM_INFO info = default;

    /// <summary>
    /// Current number of bytes allocated.
    /// </summary>
    public static long BytesCurrentlyAllocated = 0;

    /// <summary>
    /// Maximum number of bytes ever allocated at any point in time.
    /// </summary>
    public static long MaxBytesAllocated = 0;

    // If we touch an OS page with items we need to be sure
    // the whole page has been allocated, so pad if necessary
    readonly long minimumExtraItems;

    #region Public methods

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="maxItems"></param>
    /// <param name="itemSizeBytes"></param>
    /// <param name="largePages"></param>
    public WindowsVirtualAllocManager(IAppLogger logger, long maxItems, uint itemSizeBytes, bool largePages)
    {
      MaxItems = maxItems;
      ItemSizeBytes = (int)itemSizeBytes;
      LargePages = largePages;

      // Initialize system memory information if necessary
      if (info.dwNumberOfProcessors == 0)
        Win32SystemInfo.GetSystemInfo(ref info);

      //     if (info.dwPageSize % itemSizeBytes != 0)
      //       throw new Exception("Internal error: implementation limitation that page size must be a multiple of element length");

      // VirtualAlloc works at minimum granularity of 64KB
      // but for efficiency reasons we allocate even larger blocks
      // of 512KB to reduce number of calls to VirtualAlloc
      const int BLOCK_SIZE_BYTES = 2 * 1024 * 1024;
      Debug.Assert(BLOCK_SIZE_BYTES % info.dwPageSize == 0);

      long requiredMaxBytes = MathUtils.RoundedUp(maxItems * itemSizeBytes, info.dwPageSize);
      if (requiredMaxBytes < BLOCK_SIZE_BYTES)
      {
        AllocBlockSizeBytes = (int)requiredMaxBytes;
      }
      else
      {
        AllocBlockSizeBytes = BLOCK_SIZE_BYTES;
      }

      minimumExtraItems = System.Math.Min((int)info.dwPageSize, (int)info.dwPageSize / ItemSizeBytes);
    }

    public void Release()
    {
      Logger?.LogInfo("Memory", $"Release all memory", (int)AllocPtr);

      if (!Win32.VirtualFree(AllocPtr, IntPtr.Zero, Win32.MEM_RELEASE))
      {
        if (!System.Environment.HasShutdownStarted)
        {
          throw new Exception("VirtualFree", new Win32Exception(Marshal.GetLastWin32Error()));
        }
      }

      long bytesReleased = (long)NumBlocksCommitted * (long)AllocBlockSizeBytes;
      // BAD PERFORMANCE  GC.RemoveMemoryPressure(bytesReleased);
      Interlocked.Add(ref BytesCurrentlyAllocated, -bytesReleased);
    }

    public IntPtr ReserveAndAllocateFirstBlock()
    {
      int allocationType = Win32.MEM_RESERVE;
      if (LargePages)
      {
        allocationType |= Win32SystemInfo.MEM_LARGE_PAGES;
      }

      long maxTotalSizeBytes = MathUtils.RoundedUp((MaxItems + minimumExtraItems) * ItemSizeBytes, AllocBlockSizeBytes);
      AllocPtr = Win32.VirtualAlloc(IntPtr.Zero, (IntPtr)maxTotalSizeBytes, allocationType, Win32.PAGE_NOACCESS);
      if (AllocPtr.ToInt64() == 0)
      {
        throw new Win32Exception($"Failure in memory reserve (VirtualAlloc) of size {maxTotalSizeBytes} with Windows error {Marshal.GetLastWin32Error()}");
      }

      // Allocate first block
      InsureItemsAllocated(1);

      return AllocPtr;
    }

    public void InsureItemsAllocated(long numItems)
    {
      if (numItems > MaxItems)
      {
        throw new ArgumentException($"Allocation overflow, requested {numItems} but maximum was set as {MaxItems}");
      }

      long numItemsDeficient = (numItems + minimumExtraItems) - NumItemsCommitted;

      if (numItemsDeficient > 0)
      {
        long itemsToAllocate = MathUtils.RoundedUp(numItemsDeficient, ItemsPerBlock);
        long blocksToAllocate = itemsToAllocate / ItemsPerBlock;
        AllocateMoreBlocks(blocksToAllocate);
      }
    }

    public void ResizeToNumItems(long numItems)
    {
      long itemsToAllocate = MathUtils.RoundedUp(numItems, ItemsPerBlock);
      long blocksToAllocate = itemsToAllocate / ItemsPerBlock;
      ResizeToNumBlocks(blocksToAllocate);
    }


    public long NumItemsAllocated => NumItemsCommitted;

    #endregion

    #region Internal helpers

    void AllocateMoreBlocks(long numBlocks)
    {
      int allocationType = Win32.MEM_COMMIT;
      if (LargePages) allocationType |= Win32SystemInfo.MEM_LARGE_PAGES;

      IntPtr newBlocksStart = (IntPtr)(AllocPtr.ToInt64() + (NumBlocksCommitted * AllocBlockSizeBytes));
      long bytesAlloc = numBlocks * AllocBlockSizeBytes;
      IntPtr newBlockAdr = Win32.VirtualAlloc(newBlocksStart, (IntPtr)bytesAlloc, allocationType, Win32.PAGE_READWRITE);
      if (newBlockAdr.ToInt64() == 0)
      {
        throw new Exception($"Failure in memory incremental allocation (VirtualAlloc) of size {AllocBlockSizeBytes} with Windows error {Marshal.GetLastWin32Error()}");
      }

      // BAD PERFORMANCE  GC.AddMemoryPressure(bytesAlloc);
      Interlocked.Add(ref BytesCurrentlyAllocated, bytesAlloc);

      if (BytesCurrentlyAllocated > MaxBytesAllocated)
      {
        MaxBytesAllocated = BytesCurrentlyAllocated;
      }

      NumItemsCommitted += numBlocks * ItemsPerBlock;
      NumBlocksCommitted += numBlocks;
    }


    void ResizeToNumBlocks(long numBlocks)
    {
      Debug.Assert(numBlocks <= NumBlocksCommitted);

      // In order to keep a buffer of committed space above,
      // resize to one more than requested.
      numBlocks++;

      if (numBlocks < NumBlocksCommitted)
      {
        IntPtr newBlocksStart = (IntPtr)(AllocPtr.ToInt64() + (numBlocks * AllocBlockSizeBytes));
        long numBlocksDecommit = NumBlocksCommitted - numBlocks;
        long bytesFree = numBlocksDecommit * AllocBlockSizeBytes;
        bool successFree = Win32.VirtualFree(newBlocksStart, (IntPtr)bytesFree, Win32.MEM_DECOMMIT);
        if (!successFree)
        {
          throw new Exception($"Failure in memory incremental release (VirtualFree) of size {bytesFree} with Windows error {Marshal.GetLastWin32Error()}");
        }

        // BAD PERFORMANCE  GC.AddMemoryPressure(bytesAlloc);
        Interlocked.Add(ref BytesCurrentlyAllocated, -bytesFree);

        NumBlocksCommitted -= numBlocksDecommit;
        NumItemsCommitted -= numBlocksDecommit * ItemsPerBlock;
      }
    }

    #endregion
  }
}
