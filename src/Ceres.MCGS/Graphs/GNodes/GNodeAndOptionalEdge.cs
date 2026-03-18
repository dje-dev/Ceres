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
using Ceres.MCGS.Graphs.GEdges;
#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Represents a parent node an optionally an edge connecting to a child node.
/// The edge may be undefined in the case of a leaf node
/// </summary>
/// <param name="ParentNode"></param>
/// <param name="Edge"></param>
public readonly record struct GNodeAndOptionalEdge
{
  /// <summary>
  /// The node (which is the parent of the edge if provided).
  /// </summary>
  public readonly GNode ParentNode;

  
  /// <summary>
  /// Optional edge connecting to a child.
  /// </summary>
  public readonly GEdge Edge;


  /// <summary>
  /// Constructor when both node and edge are available.
  /// </summary>
  /// <param name="parentNode"></param>
  /// <param name="edge"></param>
  public GNodeAndOptionalEdge(GNode parentNode, GEdge edge)
  {
    Debug.Assert(!parentNode.IsNull);
    Debug.Assert(edge.IsNull || edge.ParentNode == parentNode);

    ParentNode = parentNode;
    Edge = edge;
  }

  /// <summary>
  /// Constructor when edge is not available.
  /// </summary>
  /// <param name="parentNode"></param>
  public GNodeAndOptionalEdge(GNode parentNode)
  {
    Debug.Assert(!parentNode.IsNull);
    ParentNode = parentNode;
  }


  /// <summary>
  /// Returns if the parent node is null.
  /// </summary>
  public readonly bool IsNull => ParentNode.IsNull;


  /// <summary>
  /// Returns if the edge is available.
  /// </summary>
  public readonly bool HasEdge => !Edge.IsNull;


  /// <summary>
  /// Returns string representation of the node and optionally the edge.
  /// </summary>
  /// <returns></returns>
  public override string ToString() =>  $"<GNodeAndOptionalEdge {ParentNode} {(HasEdge ? Edge.ToString() : "<no edge>")}>";  
}

