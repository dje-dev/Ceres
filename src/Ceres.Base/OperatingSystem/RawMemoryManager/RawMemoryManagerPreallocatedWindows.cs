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

using Ceres.Base.Math;
using Ceres.Base.OperatingSystem.Windows;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Operating system memory allocation class which uses preallocated block of memory.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  internal unsafe class WRawMemoryManagerPreallocatedWindows<T> : IRawMemoryManagerIncremental<T> where T : unmanaged
  {
    #region Internal data

    public long NumAllocatedItems;

    IntPtr allocBytesSize;

    internal void* rawMemoryAddress;
    IntPtr allocPointer = default;

    Win32SharedMappedMemory mmf;

    #endregion

    void* IRawMemoryManagerIncremental<T>.RawMemoryAddress => rawMemoryAddress;

    internal WRawMemoryManagerPreallocatedWindows()
    {
    }

    void IRawMemoryManagerIncremental<T>.Reserve(string sharedMemName, bool useExistingSharedMemory, long numItems, bool largePages)
    {
      NumAllocatedItems = numItems;
      if (numItems > int.MaxValue - 1) throw new ArgumentOutOfRangeException(nameof(numItems), "numItems must be < int.MaxValue");

      RawAlloc((int)numItems, sharedMemName, useExistingSharedMemory, largePages);
      rawMemoryAddress = (void*)allocPointer;
    }

    void IRawMemoryManagerIncremental<T>.InsureAllocated(long numItems)
    {

    }

    long IRawMemoryManagerIncremental<T>.NumItemsAllocated => NumAllocatedItems;

    void IRawMemoryManagerIncremental<T>.Dispose()
    {
      if (allocPointer != default)
      {
        if (mmf != null)
        {
          // Just dispose our connection to memory mapped file
          mmf.Dispose();
        }
        else
        {
          if (!Win32.VirtualFree(allocPointer, IntPtr.Zero, Win32.MEM_RELEASE))
            if (!System.Environment.HasShutdownStarted)
              throw new Exception("VirtualFree", new Win32Exception(Marshal.GetLastWin32Error()));

          // BAD PERFORMANCE GC.RemoveMemoryPressure((long)allocBytesSize);
        }
        allocPointer = default;
      }
    }

    void RawAlloc(int numItems, string sharedMemorySegmentName, bool useExistingSharedMemory, bool? largePages)
    {
      // Default value is false on Windows
      largePages = largePages ?? false;

      long LARGE_PAGE_SIZE = Win32SharedMappedMemory.LargePageSize;

      // Determine actual size for allocation (padded to round up to full large page size)
      long allocSizePadded = (numItems * Marshal.SizeOf(typeof(T)));
      if (largePages.Value) allocSizePadded = MathUtils.RoundedUp(allocSizePadded, LARGE_PAGE_SIZE);

      allocBytesSize = (IntPtr)allocSizePadded;

      if (sharedMemorySegmentName != null)
      {
        mmf = new Win32SharedMappedMemory(sharedMemorySegmentName, useExistingSharedMemory, (uint)allocSizePadded, largePages.Value);
        allocPointer = mmf.MemoryStartPtr;
      }
      else
      {
        // Note that VirtualAlloc is guaranteed to zero memory before returning
        if (largePages.Value)
        {
          // Acquire special priveleges required to allocate large pages
          Win32AcquirePrivilege.VerifyCouldAcquireSeLockPrivilege();

          int allocationType = Win32.MEM_RESERVE | Win32.MEM_COMMIT | Win32SystemInfo.MEM_LARGE_PAGES;
          allocPointer = Win32.VirtualAlloc(IntPtr.Zero, allocBytesSize, allocationType, Win32.PAGE_READWRITE);
          //  allocPointer = Win32.VirtualAllocExNuma(Process.GetCurrentProcess().Handle, IntPtr.Zero, allocBytesSize, allocationType, Win32.PAGE_READWRITE, 0); 
        }
        else
        {
          allocPointer = Win32.VirtualAlloc(IntPtr.Zero, allocBytesSize, Win32.MEM_RESERVE | Win32.MEM_COMMIT, Win32.PAGE_READWRITE);
          // allocPointer = Win32.VirtualAllocExNuma(Process.GetCurrentProcess().Handle, IntPtr.Zero, allocBytesSize, Win32.MEM_RESERVE | Win32.MEM_COMMIT, Win32.PAGE_READWRITE, 0);
        }

        // BAD PERFORMANCE GC.AddMemoryPressure((long)allocBytesSize);

        if (allocPointer == IntPtr.Zero)
          throw new Exception($"VirtualAlloc failed (Size={allocSizePadded} LargePages={largePages})", new Win32Exception(Marshal.GetLastWin32Error()));
      }

    }

    public void ResizeToNumItems(long numItems)
    {
      throw new NotImplementedException();
    }
  }
}
