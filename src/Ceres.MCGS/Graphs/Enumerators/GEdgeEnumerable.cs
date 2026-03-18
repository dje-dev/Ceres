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

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

/// <summary>
/// Represents an enumerable collection of edges associated with a specific node in a graph.
/// </summary>
public readonly ref struct GEdgeEnumerable
{
  /// <summary>
  /// Node 
  /// </summary>
  public GNode Node { get; init; }

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="mode"></param>
  public GEdgeEnumerable(GNode node) => Node = node;
    
  public readonly GEdgeEnumerator GetEnumerator() => new(Node);
}
