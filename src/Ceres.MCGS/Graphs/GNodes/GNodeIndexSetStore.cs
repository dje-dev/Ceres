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
using System.Threading;
using Ceres.Base.OperatingSystem;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Manages raw storage of NodeIndexSet structs in a single contiguous array of structures.
/// </summary>
public partial class GNodeIndexSetStore
{
  /// <summary>
  /// Maximum number of NodeIndexSet structs which this store is configured to hold.
  /// </summary>
  public int MaxSets { init; get; }

  /// <summary>
  /// All NodeIndexSet structs stored in a single resizable memory array located at a fixed address.
  /// </summary>
  public MemoryBufferOS<NodeIndexSet> sets;


  [DebuggerBrowsable(DebuggerBrowsableState.Never)]
  public Span<NodeIndexSet> Span => sets.Span;

  internal const int FIRST_ALLOCATED_INDEX = 1;

  readonly Lock lockObj = new();

  /// <summary>
  /// Returns the number of sets allocated so far.
  /// </summary>
  public int NumUsedSets => nextFreeIndex - FIRST_ALLOCATED_INDEX;

  /// <summary>
  /// Returns the number of sets in use (all allocated sets plus one unused root node at beginning).
  /// </summary>
  public int NumTotalSets => nextFreeIndex; // includes reserved null entry at 0

  /// <summary>
  /// The index indicating the next free set slot.
  /// </summary>
  internal int nextFreeIndex = FIRST_ALLOCATED_INDEX; // Index 0 reserved, indicates null set

  /// <summary>
  /// Parent store to which this sets store belongs.
  /// </summary>
  public readonly GraphStore ParentStore;

  /// <summary>
  /// Address of the first set in the store.
  /// This is guaranteed to be a fixed address, so we can use it to calculate the offset of any set.
  /// </summary>
  readonly long addressSetZero;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="parentStore">Parent graph store</param>
  /// <param name="numSets">Maximum number of sets to store</param>
  /// <param name="useIncrementalAlloc">If incremental allocation should be used</param>
  /// <param name="largePages">If large pages should be used</param>
  /// <param name="useExistingSharedMem">If shared memory should be used</param>
  public unsafe GNodeIndexSetStore(GraphStore parentStore, int numSets, 
                                  bool useIncrementalAlloc,
                                  bool largePages, bool useExistingSharedMem)
  {
    ParentStore = parentStore;
    MaxSets = numSets;

    parentStore.DebugLogInfo($"GNodeIndexSetStore with {numSets} sets, large pages: {largePages}, shared memory: {useExistingSharedMem}");

    string memorySegmentName = useExistingSharedMem ? "CeresSharedNodeIndexSets" : null;

    sets = new MemoryBufferOS<NodeIndexSet>(numSets + BUFFER_SETS, largePages, memorySegmentName, useExistingSharedMem, useIncrementalAlloc);

    addressSetZero = (long)(IntPtr)Unsafe.AsPointer(ref sets[0]);
  }


  /// <summary>
  /// Underlying memory buffer.
  /// </summary>
  public MemoryBufferOS<NodeIndexSet> MemoryBufferOSStore => sets;


  /// <summary>
  /// Releases the sets store.
  /// </summary>
  public void Deallocate() => sets.Dispose();

  /// <summary>
  /// Allocates and returns the next available set index.
  /// </summary>
  /// <returns>Index of the allocated set</returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public int AllocateNext()
  {
    // Take next available (lock-free)
    int gotIndex = Interlocked.Increment(ref nextFreeIndex) - 1;

    // Check for overflow (with page buffer)
    if (sets.NumItemsAllocated <= gotIndex + BUFFER_SETS)
    {
      lock (lockObj)
      {
        sets.InsureAllocated(gotIndex + BUFFER_SETS);
      }
    }

    return gotIndex;
  }


  /// <summary>
  /// Resizes memory store to exactly fit current used space.
  /// </summary>
  public void ResizeToCurrent() => ResizeToNumSets(NumTotalSets);

  /// <summary>
  /// Resizes underlying memory block to commit only specified number of items.
  /// </summary>
  /// <param name="numSets">Number of sets to resize to</param>
  /// <exception cref="Exception">If requested size is invalid</exception>
  void ResizeToNumSets(int numSets)
  {
    if (numSets < NumTotalSets)
    {
      throw new ArgumentException("Attempt to resize GNodeIndexSetStore to size smaller than current number of used sets.");
    }
    else if (numSets > sets.NumItemsAllocated)
    {
      throw new ArgumentException("Attempt to resize GNodeIndexSetStore to size larger than current.");
    }

    sets.ResizeToNumItems(numSets);
  }


  /// <summary>
  /// Overallocate sufficiently to make sure allocation reaches to end of a (possibly large) page
  /// </summary>
  static int BUFFER_SETS => (2048 * 1024) / Unsafe.SizeOf<NodeIndexSet>();


  /// <summary>
  /// Returns a string summary of the object.
  /// </summary>
  /// <returns>String representation</returns>
  public override string ToString()
  {
    return $"<GNodeIndexSetStore MaxSets={MaxSets} UsedSets={NumUsedSets}>";
  }  
}