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

namespace Ceres.MCGS.Graphs.GEdgeHeaders;

/// <summary>
/// Wrapper for an int which represents either:
///   - index into the edge header store (for a fully materialized policy information)
///   - (if deferred) index of a node from which the policy information can be copied if/when needed
///  Encoding within a single int is accomplished by negating the value if represents a deferred node index.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
[Serializable]
internal record struct EdgeHeaderBlockIndexOrNodeIndex
{
  private int rawValue;

  /// <summary>
  /// Returns if this is a deferred node index.
  /// </summary>
  public readonly bool IsNodeIndex => rawValue < 0;

  /// <summary>
  /// If uninitialized.
  /// </summary>
  public readonly bool IsNull => rawValue == 0;


  /// <summary>
  /// Constructor for a block index into the edge header store.
  /// </summary>
  /// <param name="blockIndexIntoEdgeHeaderStore"></param>
  public EdgeHeaderBlockIndexOrNodeIndex(int blockIndexIntoEdgeHeaderStore)
  {
    Debug.Assert(blockIndexIntoEdgeHeaderStore > 0);
    rawValue = blockIndexIntoEdgeHeaderStore;
  }


  /// <summary>
  /// Constructor for a deferred node index.
  /// </summary>
  /// <param name="deferredNodeIndex"></param>
  public EdgeHeaderBlockIndexOrNodeIndex(NodeIndex deferredNodeIndex) 
    => rawValue = -deferredNodeIndex.Index;


  /// <summary>
  /// Resets back to uninitialized state.
  /// </summary>
  public void Clear() => rawValue = 0;


  /// <summary>
  /// Returns the block index into the edge header store if this is a materialized.
  /// </summary>
  public readonly int BlockIndexIntoEdgeHeaderStore
  {
    get
    {
      Debug.Assert(rawValue >= 0);
      return rawValue;
    }
  }

  /// <summary>
  /// Returns the index of the node to be used for sourcing policy information if this is a deferred node index.
  /// </summary>
  public readonly NodeIndex NodeIndex
  {
    get
    {
      Debug.Assert(rawValue < 0);
      return new NodeIndex(-rawValue);
    }
  }

  /// <summary>
  /// Returns string representation.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
    => IsNodeIndex ? $"<EdgeHeaderBlockIndexOrNode (Node pending copy): #{NodeIndex}>" 
                   : $"<EdgeHeaderBlockIndexOrNode BlockIndex: {BlockIndexIntoEdgeHeaderStore}>";
}
