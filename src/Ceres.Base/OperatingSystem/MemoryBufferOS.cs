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
using System.Diagnostics;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// A buffer of memory allocated from the operating system (Windows)
  /// which is interpreted as an array of structs of generic unmanaged type T.
  /// 
  /// The underlying memory can either be:
  ///   - directly allocated by this class, or
  ///   - mapped in form another process
  ///   
  /// Optional large page memory is supported, with limitations:
  ///   - running process must have privilege for allocation large page memory,
  ///   - prone to failure if memory is too fragmented
  ///   
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public unsafe class MemoryBufferOS<T> : IDisposable where T : unmanaged
  {
    public readonly bool UseExistingSharedMemory;

    /// <summary>
    /// Number of items requested to be allocated
    /// </summary>
    public readonly long MaxItems;

    /// <summary>
    /// If memory should be allocated only incrementally as needed
    /// (by extending the allocated region of the large block reserved at initialization)
    /// </summary>
    public readonly bool UseIncrementalAlloc;

    // TODO: restore GC pressure (?)
    //bool gcPressureWasAdded;

    void* rawMemoryAddress;


    IRawMemoryManagerIncremental<T> rawMemoryManager;

    #region Constructor

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxItems"></param>
    /// <param name="largePages"></param>
    /// <param name="sharedMemoryName"></param>
    /// <param name="useExistingSharedMemory"></param>
    /// <param name="useIncrementalAlloc"></param>
    public MemoryBufferOS(long maxItems, bool largePages, string sharedMemoryName,
                          bool useExistingSharedMemory, bool useIncrementalAlloc)
    {
      MaxItems = maxItems;
      UseExistingSharedMemory = useExistingSharedMemory;
      UseIncrementalAlloc = useIncrementalAlloc;

      // Make name unique to this process if we are not sharing
      if (sharedMemoryName != null && !useExistingSharedMemory)
      {
        sharedMemoryName += "_" + Process.GetCurrentProcess().Id.ToString();
      }

      Allocate(sharedMemoryName, useExistingSharedMemory, maxItems, largePages);
    }

    #endregion

    #region Raw memory allocation/deallocation

    /// <summary>
    /// Allocates the memory buffer.
    /// </summary>
    /// <param name="sharedMemName"></param>
    /// <param name="useExistingSharedMemory"></param>
    /// <param name="numItems"></param>
    /// <param name="largePages"></param>
    /// <exception cref="Exception"></exception>
    void Allocate(string sharedMemName, bool useExistingSharedMemory, long numItems, bool largePages)
    {
      if (SoftwareManager.IsLinux)
      {
        if (UseIncrementalAlloc)
        {
          rawMemoryManager = new RawMemoryManagerIncrementalLinux<T>() as IRawMemoryManagerIncremental<T>;
        }
        else
        {
          throw new Exception("Only UseIncrementalAlloc mode supported under Linux");
        }
      }
      else
      {
        if (UseIncrementalAlloc)
        {
          rawMemoryManager = new RawMemoryManagerIncrementalWindows<T>() as IRawMemoryManagerIncremental<T>;
        }
        else
        {
          rawMemoryManager = new WRawMemoryManagerPreallocatedWindows<T>() as IRawMemoryManagerIncremental<T>;
        }
      }

      rawMemoryManager.Reserve(sharedMemName, useExistingSharedMemory, numItems, largePages);

      rawMemoryAddress = rawMemoryManager.RawMemoryAddress;
    }

    /// <summary>
    /// Insures that the specified number of items are allocated and avaliable for use.
    /// </summary>
    /// <param name="numItems"></param>
    public void InsureAllocated(long numItems) => rawMemoryManager.InsureAllocated(numItems);


    /// <summary>
    /// Resizes the buffer to the specified number of items.
    /// </summary>
    /// <param name="numItems"></param>
    public void ResizeToNumItems(long numItems) => rawMemoryManager.ResizeToNumItems(numItems);


    /// <summary>
    /// Returns the number of items allocated.
    /// </summary>
    public long NumItemsAllocated => rawMemoryManager.NumItemsAllocated;

    /// <summary>
    /// Releases the memory buffer.
    /// </summary>
    public void Dispose() => rawMemoryManager.Dispose();


    #endregion

    #region Access helpers

    /// <summary>
    /// Returns a pointer to the raw memory (which is guaranteed fixed within virtual address space).
    /// </summary>
    public void* RawMemory => rawMemoryAddress;


    /// <summary>
    /// Returns the total number of items in the buffer.
    /// </summary>
    public long Length => MaxItems;


    /// <summary>
    /// Indexer (using a long).
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public ref T this[long index]
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
      get => ref ((T*)rawMemoryAddress)[index];
    }


    /// <summary>
    /// Copies specified number of entries from one location to another.
    /// </summary>
    /// <param name="sourceIndex"></param>
    /// <param name="destinationIndex"></param>
    /// <param name="numEntries"></param>
    public void CopyEntries(long sourceIndex, long destinationIndex, int numEntries)
    {
      byte* start = (byte*)rawMemoryAddress;
      byte* source = start + sourceIndex * sizeof(T);
      byte* dest = start + destinationIndex * sizeof(T);

      long numBytes = sizeof(T) * numEntries;
      Debug.Assert(numBytes < int.MaxValue);

      // N.B. memory ranges may overlap so don't use (for example) Unsafe.CopyBlock.
      Buffer.MemoryCopy(source, dest, (uint)numBytes, (uint)numBytes);
    }


    /// <summary>
    /// Clears a section of the buffer to zeros.
    /// </summary>
    /// <param name="startIndex"></param>
    /// <param name="length"></param>
    public void Clear(long startIndex, long length)
    {
      int MAX_PER_BLOCK = int.MaxValue / sizeof(T);

      while (length > 0)
      {
        long numToClear = length < MAX_PER_BLOCK ? length : MAX_PER_BLOCK;
        ClearSmallBlock(startIndex, (int)numToClear);

        startIndex += numToClear;
        length -= numToClear;
      }
    }


    /// <summary>
    /// Clears a block of memory starting from a specified index for a given length by setting the bytes to zero.
    /// </summary>
    /// <param name="startIndex">Indicates the starting position in memory from which the clearing operation begins.</param>
    /// <param name="length">Specifies the number of elements to clear from the starting position.</param>
    void ClearSmallBlock(long startIndex, long length)
    {
      long sizeBytes = sizeof(T) * length;
      Debug.Assert(sizeBytes < uint.MaxValue);
      Debug.Assert(startIndex + length < uint.MaxValue);

      byte* ptr = (byte*)rawMemoryAddress + startIndex * sizeof(T);
      Unsafe.InitBlockUnaligned(ptr, 0, (uint)sizeBytes);
    }


    /// <summary>
    /// Returns a Span<T> view of the memory.
    /// </summary>
    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    public Span<T> Span
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        Debug.Assert(MaxItems <= int.MaxValue); // the Length property of a Span<T> is an int

        return new Span<T>(rawMemoryAddress, (int)MaxItems);
      }
    }


    /// <summary>
    /// Returns a Span<T> view of a slice of the memory.
    /// </summary>
    /// <param name="startIndex"></param>
    /// <param name="length"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> Slice(long startIndex, long length)
    {
      Debug.Assert(length < int.MaxValue); // TODO: possibly this should be allowed if the length of the slice does not exceed int.MaxValue
      Debug.Assert(startIndex + length <= NumItemsAllocated);

      return new Span<T>((T*)rawMemoryAddress + startIndex, (int)length);
    }

    #endregion
  }
}
