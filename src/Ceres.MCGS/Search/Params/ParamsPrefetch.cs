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

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Set of parameters used to control the prefetching behavior.
/// 
/// Many of the parameters are specified as an array of per-depth values,
/// these corresponding to the depths starting at 0 (root).
/// </summary>
[Serializable]
public record ParamsPrefetch
{
  /// <summary>
  /// Constructor.
  /// </summary>
  public ParamsPrefetch()
  {
  }

  /// <summary>
  /// Maximum depth at which prefetching will take place (root = depth 0).
  /// </summary>
  public int NumDepthLevels { init; get; } = int.MaxValue;

  /// <summary>
  /// Maximum total number of nodes to prefetch.
  /// </summary>
  public int MaxNumNodes { init; get; } = int.MaxValue;

  /// <summary>
  /// Optional maximum number of nodes to prefetch at each level.
  /// </summary>
  public int[] MaxNodesPerDepth { init; get; } = null;

  /// <summary>
  /// Maximum width of the tree to prefetch at each depth.
  /// </summary>
  public int MaxWidth { init; get; } = int.MaxValue;

  /// <summary>
  /// Optional minimum absolute policy percentage required for a child 
  /// to be considered for prefetching at each depth.
  /// </summary>
  public float[] MinAbsolutePolicyPctPerDepth { init; get; }

  /// <summary>
  /// Optional maximum gap between policy percentage of candidate node 
  /// and best policy node at each depth.
  /// </summary>
  public float[] MaxProbabilityPctGapFromBestPerDepth { init; get; }


  /// <summary>
  /// If children should be possibly rearranged after prefetch
  /// to put most attractive children (using V) earlier in the child sequence.
  /// </summary>
  public bool PrefetchResortChildrenUsingV = false;

}
