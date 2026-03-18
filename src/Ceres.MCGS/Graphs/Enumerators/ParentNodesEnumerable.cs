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
using System.Collections;
using System.Collections.Generic;

using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

/// <summary>
/// Enumerable over the parent nodes of a specified node.
/// </summary>
public ref struct ParentNodesEnumerable : IEnumerable<GNode>
{
  /// <summary>
  /// Parent graph.
  /// </summary>
  public readonly Graph Graph;

  /// <summary>
  /// Index of the child node whose parents are being enumerated.
  /// </summary>
  public readonly NodeIndex ChildIndex;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="childIndex"></param>
  public ParentNodesEnumerable(Graph graph, NodeIndex childIndex)
  {
    Graph = graph;
    ChildIndex = childIndex;
  }

  public ParentNodesEnumerator GetEnumerator() => new ParentNodesEnumerator(Graph, ChildIndex);

  IEnumerator<GNode> IEnumerable<GNode>.GetEnumerator() => throw new NotImplementedException();

  IEnumerator IEnumerable.GetEnumerator() => throw new NotImplementedException();
}
