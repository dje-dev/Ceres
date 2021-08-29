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

using Ceres.Base.Environment;
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Windows memory allocation class which uses incrementally allocated blocks of memory.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  [SupportedOSPlatform("windows")]
  internal unsafe class RawMemoryManagerIncrementalWindows<T> : IRawMemoryManagerIncremental<T> where T : unmanaged
  {
    #region Internal data

    WindowsVirtualAllocManager allocManager;

    internal void* rawMemoryAddress;

    #endregion

    void* IRawMemoryManagerIncremental<T>.RawMemoryAddress => rawMemoryAddress;

    internal RawMemoryManagerIncrementalWindows()
    {
    }

    void IRawMemoryManagerIncremental<T>.Reserve(string sharedMemName, bool useExistingSharedMemory, long numItems, bool largePages)
    {
      if (largePages)
      {
        throw new Exception("Large pages not supported with RawMemoryManagerIncremental");
      }

      if (sharedMemName != null || useExistingSharedMemory)
      {
        throw new Exception("RawMemoryManagerIncremental does not support use of shared memory segments");
      }

      IAppLogger logger = null; //CeresLogger.Logger
      allocManager = new WindowsVirtualAllocManager(logger, numItems, (uint)Marshal.SizeOf<T>(), false);
      rawMemoryAddress = (void*)allocManager.ReserveAndAllocateFirstBlock();
    }

    void IRawMemoryManagerIncremental<T>.InsureAllocated(long numItems) => allocManager.InsureItemsAllocated(numItems);

    void IRawMemoryManagerIncremental<T>.ResizeToNumItems(long numItems) => allocManager.ResizeToNumItems(numItems);

    long IRawMemoryManagerIncremental<T>.NumItemsAllocated => allocManager.NumItemsAllocated;

    void IRawMemoryManagerIncremental<T>.Dispose() => allocManager.Release();
  }

}
