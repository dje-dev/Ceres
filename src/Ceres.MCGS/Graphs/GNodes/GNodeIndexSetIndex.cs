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

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// A lightweight index reference that can either:
/// 1. Store a direct reference to a single NodeIndex (when Value is negative)
/// 2. Reference a NodeIndexSet stored in the GNodeIndexSetStore (when Value is positive)
/// 
/// This dual interpretation optimizes the common case of having only one node in the set
/// by avoiding the allocation of a NodeIndexSet.
/// </summary>
public readonly record struct GNodeIndexSetIndex
{
  /// <summary>
  /// Raw internal value with dual interpretation:
  /// - If negative: -(Value+1) represents a direct NodeIndex
  /// - If zero: Represents a null/invalid reference
  /// - If positive: Represents an index in the GNodeIndexSetStore
  /// </summary>
  public readonly int Value;

  /// <summary>
  /// Returns if the index is null (not pointing to a valid NodeIndexSet or NodeIndex).
  /// </summary>
  public bool IsNull => Value == 0;

  /// <summary>
  /// Returns if this contains a direct NodeIndex rather than referencing a NodeIndexSet.
  /// </summary>
  public bool IsDirectNodeIndex => Value < 0;

  /// <summary>
  /// Returns the stored NodeIndex if this is a direct node index reference.
  /// Only valid when IsDirectNodeIndex is true.
  /// </summary>
  public int DirectNodeIndex => IsDirectNodeIndex ? -(Value + 1) : throw new InvalidOperationException("Not a direct node index");

  /// <summary>
  /// Returns the index into the GNodeIndexSetStore.
  /// Only valid when IsDirectNodeIndex is false and IsNull is false.
  /// </summary>
  public int NodeSetIndex => !IsDirectNodeIndex && !IsNull ? Value : throw new InvalidOperationException("Not a node set index");


  /// <summary>
  /// Creates a GNodeIndexSetIndex that references a NodeIndexSet in the store.
  /// </summary>
  /// <param name="nodeSetIndex">The index in the GNodeIndexSetStore.</param>
  public static GNodeIndexSetIndex FromNodeSetIndex(int nodeSetIndex) => new(nodeSetIndex);
 

  /// <summary>
  /// Creates a GNodeIndexSetIndex that directly stores a single NodeIndex.
  /// </summary>
  /// <param name="directNodeIndex">The node index to store directly.</param>
  public static GNodeIndexSetIndex FromDirectNodeIndex(int directNodeIndex) => new(-(directNodeIndex + 1));


  /// <summary>
  /// Private constructor for internal use.
  /// </summary>
  /// <param name="value">Raw internal value.</param>
  private GNodeIndexSetIndex(int value) => Value = value;


  /// <summary>
  /// Returns a string representation of the index.
  /// </summary>
  /// <returns>A string representation of the index.</returns>
  public override string ToString()
  {
    if (IsNull)
    {
      return "<GNodeIndexSetIndex Null>";
    }
    else if (IsDirectNodeIndex)
    {
      return $"<GNodeIndexSetIndex DirectNodeIdx={DirectNodeIndex}>";
    }
    else
    {
      return $"<GNodeIndexSetIndex NodeSetIdx={Value}>";
    }
  }
}
