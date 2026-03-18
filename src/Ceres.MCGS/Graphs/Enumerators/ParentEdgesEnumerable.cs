#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

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

using System.Diagnostics;
using Ceres.MCGS.Graphs.GNodes;


#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

/// <summary>
/// Enumerable over the parent edges of a specified node.
/// </summary>
public ref struct ParentEdgesEnumerable
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
  /// <param name="childNodeIndex"></param>
  public ParentEdgesEnumerable(Graph graph, NodeIndex childNodeIndex)
  {
    Debug.Assert(childNodeIndex != default);

    this.Graph = graph;
    this.ChildIndex = childNodeIndex;
  }

  public ParentEdgesEnumerator GetEnumerator() => new ParentEdgesEnumerator(Graph, ChildIndex);
}
