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
using Ceres.Base.OperatingSystem;
using Ceres.Chess.EncodedPositions;

using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.Enumerators;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GParents;

/// <summary>
/// A table structure to store parent-child relationships for MCTS nodes.
/// 
/// A root table is laid out with one 4-byte value for each sequential node in store.
/// </summary>
public partial class GParentsStore
{
  /// <summary>
  /// Graph store to which this visit store belongs.
  /// </summary>
  public GraphStore ParentStore { init; get; }

  /// <summary>
  /// Table of segments to contain linked list of parent information for nodes having multiple parents.
  /// </summary>
  internal GParentsDetailStore DetailSegments { init; get; }


  MemoryBufferOS<GNodeStruct> nodes;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="maxParents"></param>
  public GParentsStore(GraphStore parentStore, long maxParents, bool positionEquivalenceMode, bool tryEnableLargePages)
  {
    ParentStore = parentStore;
    nodes = ParentStore.NodesStore.nodes;

    parentStore.DebugLogInfo($"GParentsStore with maxParents={maxParents}, tryeEnableLargePages={tryEnableLargePages}");

    // Divide number of multiple parents by number of parents per segment (excluding the slot used as the link in linked list)
    long parentsMultiplier = positionEquivalenceMode ? 8 : 1; // In coalesced mode many more nodes will converge on same node
    long setNumDetailSegments = (parentsMultiplier * maxParents) / (GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1);
      
    DetailSegments = new GParentsDetailStore(setNumDetailSegments, tryEnableLargePages); 
  }


  /// <summary>
  /// Underlying memory buffer.
  /// </summary>
  public MemoryBufferOS<GParentsDetailsStruct> MemoryBufferOSStore => DetailSegments.MemoryBufferOSStore;


  public void GetParentsNodeIndices(NodeIndex nodeIndex, Span<int> parents)
  {
    // GFIX: for now use this slow version, eventually replace all calls with use of the enumerator.
    innerGetParentsForChild(nodeIndex, parents);

#if DEBUG
    // Cross check the results against what comes back from the Enumerator.
    Span<int> childSpan = stackalloc int[parents.Length];
    int count = 0;
    foreach (int parentIndex in NodeParentsInfo(nodeIndex))
    {
      childSpan[count] = parentIndex;
      if (childSpan[count] != parents[count])
      {
        Debug.Assert(childSpan[count] == parents[count]);
      }
      count++;
    }
    Debug.Assert(count == parents.Length || parents[count] == -1);
#endif
  }


  /// <summary>
  /// Populates a span with the parent indices for the specified child index
  /// (with a -1 terminator at the end).
  /// </summary>
  /// <param name="childIndex"></param>
  /// <param name="parents"></param>
  /// <exception cref="Exception"></exception>
  private void innerGetParentsForChild(NodeIndex childIndex, Span<int> parents)
  {
    GParentsHeader headerPointer = nodes[childIndex.Index].ParentsHeader;

    if (headerPointer.IsEmpty)
    {
      parents[0] = -1;
      return;
    }

    // Single entry inline.
    if (headerPointer.IsDirectEntry)
    {
      parents[0] = headerPointer.AsDirectParentNodeIndex.Index;
      if (parents.Length > 1)
      {
        parents[1] = -1;
      }
    }
    else
    {
      GParentsDetailsStruct segment = DetailSegments.SegmentRef(headerPointer.AsSegmentLinkIndex);
      segment.GetParents(DetailSegments, parents, 0);
    }
  }


  #region Enumeration

  internal ParentIndicesEnumerable NodeParentsInfo(NodeIndex nodeIndex)
  {
    return new ParentIndicesEnumerable(this, nodeIndex);
  }

  #endregion

  /// <summary>
  /// Deallocates resources associated with detail segments.
  /// usage.
  /// </summary>
  public void Deallocate() => DetailSegments.Deallocate();


  /// <summary>
  /// Returns if a child exists for the specified index.
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  public bool ParentHasChild(NodeIndex parentIndex, NodeIndex childIndex)
  {
    // TODO: This could be made more efficient by enumerating the children without creating a Span.
    Span<int> parentNodeIndices = stackalloc int[EncodedPolicyVectorCompressed.MAX_MOVES];
    GetParentsNodeIndices(childIndex, parentNodeIndices);
    for (int j = 0; j < parentNodeIndices.Length && parentNodeIndices[j] != -1; j++)
    {
      if (parentNodeIndices[j] == parentIndex.Index)
      {
        return true;
      }
    }

    return false;
  }


  /// <summary>
  /// Dumps full list of parents for a specified entry.
  /// </summary>
  /// <param name="index"></param>
  public void DumpForEntry(NodeIndex index)
  {
    GParentsHeader header = nodes[index.Index].ParentsHeader;

    if (header.IsEmpty)
    {
      Console.WriteLine($"Entry {index} has no parents");
    }
    else if (header.IsDirectEntry)
    {
      Console.WriteLine($"Entry {index} has single parent {header}");
    }
    else
    {
      ref GParentsDetailsStruct segment = ref DetailSegments.SegmentRef(header.AsSegmentLinkIndex);
      Console.WriteLine($"Entry {index} has multiple parents in segment {segment}");
    }
  }


  /// <summary>
  /// Creates a parent-child relationship between two nodes in the graph.
  /// </summary>
  /// <remarks>Establishes a parent edge for the specified child node:
  ///   - if does not already have any parent entries, the parent is directly assigned
  ///   - If already has one or more parent entries, stores in a GParentDetailsStruct
  ///     (possibly as part of a linked list).
  /// <param name="parentIndex">index of the parent node</param>
  /// <param name="childIndex">index of the child node</param>
  public void CreateParentEdge(NodeIndex parentIndex, NodeIndex childIndex)
  {
    Debug.Assert(ParentStore.IsRewriting || nodes[childIndex.index].LockRef.IsLocked);

    Debug.Assert(childIndex.Index > 0); // Zero has a special reserved meaning in these data structures

    ref GParentsHeader parentsHeaderPointer = ref nodes[childIndex.Index].ParentsHeader;
    if (parentsHeaderPointer.IsEmpty)
    {
      // This will be the first VisitFrom.
      // For now, just reuse the slot to point to this single parent,
      // making this a direct entry.
      parentsHeaderPointer = parentIndex.Index;
    }
    else if (parentsHeaderPointer.IsDirectEntry)
    {
      // Was pointing to a single parent, but now need multiple entries.
      // Convert single direct entry into pointer to a segment in extra segments.
      // The original tree parent is preserved at position 0 (Entries[0]) to maintain
      // the tree-parent invariant for cycle-free CalcPosition traversal.
      int segmentIndex = DetailSegments.AllocateSegment();

      ref GParentsDetailsStruct newSegment = ref DetailSegments.SegmentRef(segmentIndex);
      newSegment.Entries[0] = parentsHeaderPointer;
      newSegment.Entries[1] = parentIndex.Index;

      // Verify tree-parent invariant: original parent preserved at position 0 after transition.
      Debug.Assert(newSegment.Entries[0].AsDirectParentNodeIndex.Index == parentsHeaderPointer.AsDirectParentNodeIndex.Index,
        "Tree parent must be preserved at position 0 during single-to-multi parent transition");

      parentsHeaderPointer.SetToSegmentLink(segmentIndex);
    }
    else // if (curValue.IsLink)
    {
      // Add this new entry onto existing segment (or extend if necessary).
      ref GParentsDetailsStruct segment = ref DetailSegments.SegmentRef(parentsHeaderPointer.AsSegmentLinkIndex);
      while (true)
      {
        if (segment.IsFull)
        {
          if (segment.Entries[GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1].IsLink)
          {
            segment = ref DetailSegments.SegmentRef(segment.Entries[GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1].AsSegmentLinkIndex);
          }
          else
          {
            int segmentIndex = DetailSegments.AllocateSegment();
            ref GParentsDetailsStruct newSegment = ref DetailSegments.SegmentRef(segmentIndex);

            // Move the last entry from the old segment to the new segment
            newSegment.Entries[0] = segment.Entries[GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1].AsDirectParentNodeIndex.Index;

            // Update the old segment to point to the new segment (via the last entry)
            segment.Entries[GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1].SetToSegmentLink(segmentIndex);

            // Finally, add ourself to the new segment
            newSegment.Entries[1] = parentIndex.Index;
            break;
          }
        }
        else
        {
          segment.AddEntry(parentIndex.Index);
          break;
        }
      }
    }
  }
}
