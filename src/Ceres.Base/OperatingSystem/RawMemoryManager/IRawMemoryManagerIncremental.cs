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

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Interface implemented by classes that support incremental memory allocation.
  /// 
  /// Note that these methods do not need to protect against concurrent execution
  /// since it is assumed that the caller will have already protected against that.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public unsafe interface IRawMemoryManagerIncremental<T> where T : unmanaged
  {
    internal void* RawMemoryAddress { get; }

    void Reserve(string sharedMemName, bool useExistingSharedMemory, long numItems, bool largePages);
    void InsureAllocated(long numItems);

    void ResizeToNumItems(long numItems);

    long NumItemsAllocated { get; }

    void Dispose();
  }
}