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

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

/// <summary>
/// Enumerator over parent edges of a given child node.
/// </summary>
public ref struct ParentEdgesEnumerator
{
  /// <summary>
  /// Parent graph.
  /// </summary>
  public readonly Graph graph;

  /// <summary>
  /// Index of the child node whose parent edges are being enumerated.
  /// </summary>
  public readonly NodeIndex ChildIndex;

  /// <summary>
  /// Enumerator over parent indices.
  /// </summary>
  ParentIndexEnumerator innerEnumerator;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="childIndex"></param>
  public ParentEdgesEnumerator(Graph graph, NodeIndex childIndex)
  {
    innerEnumerator = graph.Store.ParentsStore.NodeParentsInfo(childIndex).GetEnumerator();

    this.graph = graph;
    this.ChildIndex = childIndex;
    Current = default;
  }

  public GEdge Current { get; private set; }


  public bool MoveNext()
  {
    if (!innerEnumerator.MoveNext())
    {
      Current = default;
      return false;
    }

    // Retrieve parent for this child.
    NodeIndex parentNodeIndex = new(innerEnumerator.Current);
    GNode parentNode = graph[parentNodeIndex];

    // Find index of this child within parent's child edges.
    int indexInParent = parentNode.IndexOfChildInChildEdges(ChildIndex);
    if (indexInParent == -1)
    {
      throw new Exception("ParentEdgesEnumerator: Parent not found in child's edges");
    }

    // Set Current to the edge connecting the parent to this child.
    Current = parentNode.ChildEdgeAtIndex(indexInParent);

    return true;
  }
}
