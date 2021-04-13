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

using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Data structure providing array-like indexed access to 
  /// members which are unmanaged structures.
  /// 
  /// The underlying storage is sequential in memory, pinned to a fixed address, and 
  /// automatically expanding in size to accomodate at least the index 
  /// of the largest item referenced so far.
  /// 
  /// TODO: Consider adding functinality to link to another MemoryOSBuffer such 
  ///       that when the other buffer grows in size this object also does.
  ///       That would obviate the InsureAllocated calls in GetRef.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class ArrayStructsIncrementalPinned<T> where T : unmanaged
  {
    IRawMemoryManagerIncremental<T> rawMemoryManager;

    public ArrayStructsIncrementalPinned(int maxItems, bool useLargePages)
    {
      if (SoftwareManager.IsLinux)
        rawMemoryManager = new RawMemoryManagerIncrementalLinux<T>();
      else
        rawMemoryManager = new RawMemoryManagerIncrementalWindows<T>();

      rawMemoryManager.Reserve(null, false, maxItems, useLargePages);
    }   


    public unsafe ref T GetRef(int index)
    {
      rawMemoryManager.InsureAllocated(index + 1);
      return ref Unsafe.Add(ref Unsafe.AsRef<T>(rawMemoryManager.RawMemoryAddress), index);
    }

  }
}