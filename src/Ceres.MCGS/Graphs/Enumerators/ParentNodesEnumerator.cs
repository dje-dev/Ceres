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
using Ceres.MCGS.Graphs.GParents;

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

/// <summary>
/// Enumerator over parent nodes of a given child node.
/// </summary>
public ref struct ParentNodesEnumerator : IEnumerator<GNode>
{
  /// <summary>
  /// Parent graph.
  /// </summary>
  public Graph Graph;

  /// <summary>
  /// Provides access to the underlying enumerator used for parent index traversal.
  /// </summary>
  ParentIndexEnumerator inner;

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="childIndex"></param>
  public ParentNodesEnumerator(Graph graph, NodeIndex childIndex)
  {
    Graph = graph;
    inner = new ParentIndicesEnumerable(graph.Store.ParentsStore, childIndex).GetEnumerator();
  }

  public GNode Current => new GNode(Graph, new NodeIndex(inner.Current));

  object IEnumerator.Current => new GNode(Graph, new NodeIndex(inner.Current));

  public bool MoveNext() => inner.MoveNext();

  public void Dispose() { }

  public void Reset() =>  throw new NotImplementedException();
}
