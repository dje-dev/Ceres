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
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Graphs.GParents;

/// <summary>
/// An entry providing information on the parent(s) of a node.
/// 
/// Each entry can be one of three types:
///  - zero indicating either the node is not yet allocated or has no parents (i.e. the root)
///  - positive integer representing the node index (plus one) of the single parent for this node
///  - negative integer representing the negated index of the starting entry in VisitFromDetailStruct
///    for the case of multiple parents
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = sizeof(int))]
internal record struct GParentsHeader
{
  // We want default struct value of 0 to indicate "unused"
  // but 0 is a valid parent (the root node). Therefore internally store numbers offset from zero.
  const int START_VALUE = 1;

  /// <summary>
  /// The entry of one of the three possible types.
  /// </summary>
  int Entry;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="rawValue"></param>
  private GParentsHeader(int rawValue) => Entry = rawValue + 1;


  /// <summary>
  /// Conversion operator from int.
  /// </summary>
  /// <param name="value"></param>
  public static implicit operator GParentsHeader(int value)
  {
    Debug.Assert(value >= 0, "MCTSNodeParentInfo must be non-negative");
    return new GParentsHeader(value);
  }


  /// <summary>
  /// Sets the entry to be an index into the segment table.
  /// </summary>
  /// <param name="segmentIndex"></param>
  internal void SetToSegmentLink(int segmentIndex) => Entry = -(segmentIndex + 1);


  /// <summary>
  /// Returns if the entry is empty (i.e. not yet allocated).
  /// </summary>
  internal readonly bool IsEmpty => Entry == 0;


  /// <summary>
  /// Returns if the entry is a link to a segment of possibly multiple parent entries.
  /// </summary>
  internal readonly bool IsLink => Entry < 0;


  /// <summary>
  /// Returns if the entry is a direct entry to a parent (in the common case when that is the only parent).
  /// </summary>
  internal readonly bool IsDirectEntry => Entry > 0;


  /// <summary>
  /// Accessor for the (common) case that there is only a single direct parent.
  /// </summary>
  internal readonly NodeIndex AsDirectParentNodeIndex
  {
    get
    {
      Debug.Assert(IsDirectEntry);
      return new NodeIndex((int)Entry - 1);
    }
  }


  /// <summary>
  /// The index of the first segment (if segments exist to capture multiple parents).
  /// </summary>
  internal readonly int AsSegmentLinkIndex
  {
    get
    {
      Debug.Assert(Entry < 0);
      return -Entry - 1;
    }
  }


  /// <summary>
  /// Returns string representation of the entry.
  /// </summary>
  /// <returns></returns>
  public override string ToString() => IsEmpty
    ? "Empty"
    : IsDirectEntry
        ? $"<GPHS direct {AsDirectParentNodeIndex.Index}>"
        : $"<GPHS link {AsSegmentLinkIndex}>";
}
