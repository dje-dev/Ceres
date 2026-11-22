#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;
using System.Diagnostics;
using Ceres.Base.OperatingSystem;


#endregion

namespace Ceres.MCGS.Graphs.GParents;

/// <summary>
/// Store of segments containing parent details for nodes with multiple parents.
/// Now implemented on top of MemoryBufferOSBlocked to manage segments.
/// Each buffer ELEMENT is one complete segment: GParentsDetailsStruct is itself a fixed-size
/// struct of MAX_ENTRIES_PER_SEGMENT entries. Therefore the blocking factor is 1 (one item
/// per allocation), and segment indices are item indices.
/// This version uses incremental allocation and an overallocation factor to reduce OS calls.
/// </summary>
internal unsafe class GParentsDetailStore : MemoryBufferOSBlocked<GParentsDetailsStruct>
{
  /// <summary>
  /// Number of items (segments) per allocation block. Must be 1: the element type is already a
  /// whole segment, and SegmentRef addresses segments by ITEM index.
  public const int SEGMENT_BLOCK_SIZE = 1;

  /// <summary>
  /// Extra segments reserved beyond the expected maximum (also used by the base class as the
  /// incremental-commit padding granularity; at 32 bytes/segment this is ~5 MB).
  /// </summary>
  const int SEGMENTS_EXTRA = 160 * 1024;

  /// <summary>
  /// Cached pointer to segment 0. The underlying MemoryBufferOS reserves its full virtual
  /// range once and only commits pages incrementally, so this address is fixed for the
  /// life of the store.
  /// </summary>
  readonly GParentsDetailsStruct* segmentsBasePtr;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="initialMaxSegments">The expected maximum number of segments.</param>
  /// <param name="tryEnableLargePages">Flag to try enabling large pages.</param>
  internal GParentsDetailStore(long initialMaxSegments, bool tryEnableLargePages)
      : base(initialMaxSegments,
             SEGMENT_BLOCK_SIZE,
             SEGMENTS_EXTRA,
             tryEnableLargePages,
             null,   // shared memory name not used
             false,  // useExistingSharedMem
             true)   // useIncrementalAlloc turned on
  {
    segmentsBasePtr = (GParentsDetailsStruct*)entries.RawMemory;
  }


  /// <summary>
  /// Underlying memory buffer.
  /// </summary>
  public MemoryBufferOS<GParentsDetailsStruct> MemoryBufferOSStore => Entries;


  /// <summary>
  /// Resets the next free block index.
  /// </summary>
  internal new int NextFreeBlockIndex
  {
    get => nextFreeBlockIndex.Value;
    set => nextFreeBlockIndex.Value = value;
  }


  /// <summary>
  /// Allocates a new segment and returns its index.
  /// Note that index 0 is reserved (never allocated).
  /// </summary>
  internal int AllocateSegment()
  {
    // Allocate one segment (one item; with the blocking factor of 1 the returned block index
    // is identical to the item index used by SegmentRef).
    return (int)AllocateEntriesStartBlock(1);
  }


  /// <summary>
  /// Returns a reference to the segment at the specified index.
  /// This provides similar functionality to the original SegmentRef.
  /// </summary>
  /// <param name="index">The segment index (nonzero).</param>
  /// <returns>Reference to the allocated segment.</returns>
  internal ref GParentsDetailsStruct SegmentRef(int index)
  {
    // With a blocking factor of 1 the segment index is the item index into the buffer.
    Debug.Assert(index > 0 && index < NumAllocatedItems);
    return ref segmentsBasePtr[index];
  }


  /// <summary>
  /// Dumps the entries of the segment to the console.
  /// Follows any "follow" pointer if the last entry is negative.
  /// </summary>
  /// <param name="index">Segment index to dump.</param>
  internal void DumpSegmentsToConsole(int index)
  {
    ref GParentsDetailsStruct segment = ref SegmentRef(index);
    for (int i = 0; i < GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT; i++)
    {
      Console.WriteLine($"Segment {index} entry {i} = {segment.Entries[i]}");
    }

    // If the last entry is negative, it indicates a follow pointer.
    if (segment.Entries[GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1].IsLink)
    {
      Console.WriteLine("follow");
      DumpSegmentsToConsole(segment.Entries[GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1].AsSegmentLinkIndex);
    }
  }
}
