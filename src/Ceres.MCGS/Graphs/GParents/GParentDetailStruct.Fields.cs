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
using System.Runtime.InteropServices;

#endregion

namespace Ceres.MCGS.Graphs.GParents;

  /// <summary>
  /// An inline array of a fixed number of MCTSNodeParentInfo.
  /// </summary>
  [System.Runtime.CompilerServices.InlineArray(GParentsDetailsStruct.MAX_ENTRIES_PER_SEGMENT)]
  struct MCTSNodeParentsListSegmentValueInline
  {
    GParentsHeader Entry;
  }

/// <summary>
/// Struct containing a fixed number of MCGSNodeParentInfo entries
/// used to store parent-child relationships for MCGS nodes.
/// 
/// Multiple segments can be chained together as a linked list
/// via the last entry in the segment (serving as a pointer to the next segment).
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = MAX_ENTRIES_PER_SEGMENT * sizeof(int))]
public record struct GParentsDetailsStruct
{
  /// <summary>
  /// Up to 8 entries per segment, or entries and a link to next segment.
  /// Tests showed that using 8 instead of 4 yielded greatly improved performance,
  /// saving time in CreateParentEdge method (following chained linked list entries).
  /// </summary>
  internal const int MAX_ENTRIES_PER_SEGMENT = 8;

  /// <summary>
  /// The entries in this segment.
  /// </summary>
  internal MCTSNodeParentsListSegmentValueInline Entries;

  /// <summary>
  /// If all entries are currently in use.
  /// </summary>
  internal readonly bool IsFull => !Entries[MAX_ENTRIES_PER_SEGMENT - 1].IsEmpty;


  /// <summary>
  /// Sets the parent index for the specified index.
  /// </summary>
  /// <param name="index"></param>
  /// <param name="parentIndex"></param>
  internal void SetParent(int index, int parentIndex) => Entries[index] = parentIndex;


  /// <summary>
  /// Gets all the parents for this segment and appends them to the provided span.
  /// 
  /// TODO: probably convert this into an allocation-free enumerator (such as the VisitFromStoreEnumerator).
  /// </summary>
  /// <param name="parentsTable"></param>
  /// <param name="parents"></param>
  /// <param name="numParentsAlreadyInSpan"></param>
  /// <returns>number of retrieved parents</returns>
  internal int GetParents(GParentsDetailStore parentsTable, Span<int> parents, int numParentsAlreadyInSpan)
  {
    Debug.Assert(MAX_ENTRIES_PER_SEGMENT >= 1);
    Debug.Assert(Entries[0].IsDirectEntry);

    int lastIndex = MAX_ENTRIES_PER_SEGMENT - 1;

    GParentsDetailsStruct segment = this;

    while (true)
    {
      // Consume direct entries in 0..lastIndex-1
      for (int i = 0; i < lastIndex; i++)
      {
        if (segment.Entries[i].IsDirectEntry)
        {
          parents[numParentsAlreadyInSpan++] = segment.Entries[i].AsDirectParentNodeIndex.Index;
        }
        else
        {
          // Must be empty; links only allowed in the last slot.
          if (segment.Entries[i].IsEmpty)
          {
            if (numParentsAlreadyInSpan < parents.Length)
            {
              parents[numParentsAlreadyInSpan] = -1;
            }
            return numParentsAlreadyInSpan;
          }

          throw new NotImplementedException("Internal error: unexpected segment entry type before last slot.");
        }
      }

      // Handle the last slot specially (may be a link)
      GParentsHeader last = segment.Entries[lastIndex];

      if (last.IsDirectEntry)
      {
        parents[numParentsAlreadyInSpan++] = last.AsDirectParentNodeIndex.Index;

        if (numParentsAlreadyInSpan < parents.Length)
        {
          parents[numParentsAlreadyInSpan] = -1;
        }
        return numParentsAlreadyInSpan;
      }

      if (last.IsLink)
      {
        // Follow to next segment and continue.
        segment = parentsTable.SegmentRef(last.AsSegmentLinkIndex);
        continue;
      }

      if (last.IsEmpty)
      {
        if (numParentsAlreadyInSpan < parents.Length)
        {
          parents[numParentsAlreadyInSpan] = -1;
        }
        return numParentsAlreadyInSpan;
      }

      throw new NotImplementedException("Internal error: unexpected segment entry type in last slot.");
    }
  }


  /// <summary>
  /// Adds a new parent index to the segment.
  /// </summary>
  /// <param name="parentIndex"></param>
  internal void AddEntry(int parentIndex)
  {
    Debug.Assert(MAX_ENTRIES_PER_SEGMENT == 8);
    Debug.Assert(Entries[7].IsEmpty);

    // For improved speed we exploit the fact that
    // entries are filled in order
    // (allowing a bisection search for an empty slot).

    // Do a 3-step binary search over indices 1..7.
    if (Entries[4].IsEmpty)
    {
      // First empty is in [1..4]
      if (Entries[2].IsEmpty)
      {
        // First empty is in [1..2]
        if (Entries[1].IsEmpty)
        {
          Entries[1] = parentIndex;
          return;
        }
        else
        {
          Entries[2] = parentIndex;
          return;
        }
      }
      else
      {
        // First empty is in [3..4]
        if (Entries[3].IsEmpty)
        {
          Entries[3] = parentIndex;
          return;
        }
        else
        {
          Entries[4] = parentIndex;
          return;
        }
      }
    }
    else
    {
      // First empty is in [5..7]
      if (Entries[6].IsEmpty)
      {
        // First empty is in [5..6]
        if (Entries[5].IsEmpty)
        {
          Entries[5] = parentIndex;
          return;
        }
        else
        {
          Entries[6] = parentIndex;
          return;
        }
      }
      else
      {
        // 0..6 are full; 7 must be empty (we already ruled out full segment).
        Entries[7] = parentIndex;
        return;
      }
    }
  }


  /// <summary>
  /// Returns string representation of the segment.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<GParentDetailStruct {Entries[0]}, {Entries[1]}, {Entries[2]}, {Entries[3]}>";
  }

}
