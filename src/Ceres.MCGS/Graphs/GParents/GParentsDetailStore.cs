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
using Ceres.Base.OperatingSystem;


#endregion

namespace Ceres.MCGS.Graphs.GParents;

/// <summary>
/// Store of segments containing parent details for nodes with multiple parents.
/// Now implemented on top of MemoryBufferOSBlocked to manage segments.
/// Each allocated segment spans a block of memory of fixed size,
/// determined by GParentDetailsStruct.MAX_ENTRIES_PER_SEGMENT.
/// This version uses incremental allocation and an overallocation factor to reduce OS calls.
/// </summary>
internal class GParentsDetailStore : MemoryBufferOSBlocked<GParentsDetailsStruct>
{
  /// <summary>
  /// The block (segment) size is defined as the maximum number of entries per segment.
  /// </summary>
  public const int SEGMENT_BLOCK_SIZE = GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT;

  /// <summary>
  /// Extra segments allocated beyond the expected maximum (optional, here set to zero).
  /// </summary>
  const int SEGMENTS_EXTRA = 160 * 1024;


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
    get => nextFreeBlockIndex;
    set => nextFreeBlockIndex = value;
  }


  /// <summary>
  /// Allocates a new segment and returns its index.
  /// Note that index 0 is reserved (never allocated).
  /// </summary>
  internal int AllocateSegment()
  {
    // Allocate one segment (i.e. one block of SEGMENT_BLOCK_SIZE items)
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
    // Use the base method SpanAtIndex to get a span for one segment and return the first element by reference.
    return ref SpanAtIndex(index, 1)[0];
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
