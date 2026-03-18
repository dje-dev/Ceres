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

using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GParents;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

/// <summary>
/// Helper enumerator class to efficiently (allocation-free) enumerate parent indices of a child node.
/// </summary>
public ref struct ParentIndexEnumerator : IEnumerator<int>
{
  /// <summary>
  /// Underlying store containing all the VisitsFrom information.
  /// </summary>
  private readonly GParentsDetailStore parentsDetailStore;

  /// <summary>
  /// In the special case where there is only a single parent recorded directly in the root store,
  /// this value will be set to the index of that parent (and none others will be enumerated).
  /// </summary>
  private readonly int pendingSingleVisitFrom;

  /// <summary>
  /// The current value to be returned from Current().
  /// </summary>
  private int current;

  /// <summary>
  /// The currently segment processed segment.
  /// </summary>
  ref GParentsDetailsStruct thisSegment;

  /// <summary>
  /// The index of the next entry to be returned (in the current segment being processed)
  /// or -1 no more entries to process.
  /// </summary>
  int nextSegmentIndex;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="parentsStore"></param>
  /// <param name="nodeIndex"></param>
  public ParentIndexEnumerator(GParentsStore parentsStore, NodeIndex nodeIndex)
  {
    parentsDetailStore = parentsStore.DetailSegments;
    nextSegmentIndex = 0; // Start before the first element
    pendingSingleVisitFrom = -1;

    GParentsHeader parentHeaderPointer = parentsStore.ParentStore.NodesStore.nodes[nodeIndex.Index].ParentsHeader;

    if (parentHeaderPointer.IsDirectEntry) // Single entry inline.
    {
      pendingSingleVisitFrom = parentHeaderPointer.AsDirectParentNodeIndex.Index;
    }
    else if (parentHeaderPointer.IsEmpty)
    {
      nextSegmentIndex = -1;
    }
    else
    {
      thisSegment = ref parentsDetailStore.SegmentRef(parentHeaderPointer.AsSegmentLinkIndex);
    }
  }


  /// <summary>
  /// Marks the current operation as complete by setting the next segment index to an invalid state.  
  /// </summary>
  /// processing.</remarks>
  public void SetDone() => nextSegmentIndex = -1;


  /// <summary>
  /// Advances the enumerator to the next element in the collection.
  /// </summary>
  /// <returns></returns>
  public bool MoveNext()
  {
    if (nextSegmentIndex == -1)
    {
      return false;
    }

    if (pendingSingleVisitFrom != -1)
    {
      // Just one parent stored inline in the main store.
      current = pendingSingleVisitFrom;
      nextSegmentIndex = -1;
      return true;
    }

    // Processing from a segment.
    int lastIndex = GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT - 1;
    Debug.Assert(lastIndex >= 0, "MAX_ENTRIES_PER_SEGMENT must be >= 1.");

    // If on last entry of current segment,
    // potentially follow a link to the next segment.
    if (nextSegmentIndex == lastIndex)
    {
      if (thisSegment.Entries[lastIndex].IsLink)
      {
        thisSegment = ref this.parentsDetailStore.SegmentRef(thisSegment.Entries[lastIndex].AsSegmentLinkIndex);
        nextSegmentIndex = 0;
      }
    }

    if (thisSegment.Entries[nextSegmentIndex].IsDirectEntry)
    {
      current = thisSegment.Entries[nextSegmentIndex].AsDirectParentNodeIndex.Index;
      nextSegmentIndex = (nextSegmentIndex == lastIndex) ? -1 : nextSegmentIndex + 1;
      return true;
    }
    else if (thisSegment.Entries[nextSegmentIndex].IsEmpty)
    {
      nextSegmentIndex = -1;
      return false;
    }
    else
    {
      throw new NotImplementedException("Internal error: unexpected segment entry type");
    }
  }


  public void Reset()
  {
    throw new NotImplementedException();
  }


  public void Dispose()
  {
  }

  public int Current => current;

  object IEnumerator.Current => Current;
}
