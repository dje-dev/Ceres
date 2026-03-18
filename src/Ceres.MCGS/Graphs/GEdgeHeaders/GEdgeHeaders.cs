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
using Ceres.MCGS.Graphs.GNodes;


#endregion

namespace Ceres.MCGS.Graphs.GEdgeHeaders;

/// <summary>
/// Provides a read-only view of the edge header structures associated with a node.
/// </summary>
public readonly ref struct GNodeEdgeHeaders
{
  /// <summary>
  /// The parent node to which these edge headers belong.
  /// </summary>
  public readonly GNode Parent;

  /// <summary>
  /// Returns a span over the collection of edge header structures.
  /// </summary>
  public readonly Span<GEdgeHeaderStruct> HeaderStructsSpan;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="parent"></param>
  /// <param name="headerStructsSpan"></param>
  internal GNodeEdgeHeaders(GNode parent, Span<GEdgeHeaderStruct> headerStructsSpan)
  {
    Parent = parent;
    HeaderStructsSpan = headerStructsSpan;
  }


  /// <summary>
  /// Constructor for span over the already initialized pointer to entries (via BlockIndexIntoEdgeHeaderStore).
  /// </summary>
  /// <param name="parent"></param>
  public GNodeEdgeHeaders(GNode parent)
  {
    Parent = parent;
    HeaderStructsSpan = parent.Graph.EdgeHeadersStore.SpanAtBlockIndex(Parent.BlockIndexIntoEdgeHeaderStore, Parent.NodeRef.NumPolicyMoves);
  }


  /// <summary>
  /// Indexer to access  reference to individual edge header structures.
  /// </summary>
  /// <param name="index"></param>
  /// <returns></returns>
  public ref readonly GEdgeHeaderStruct this[int index] => ref HeaderStructsSpan[index]; 
}
