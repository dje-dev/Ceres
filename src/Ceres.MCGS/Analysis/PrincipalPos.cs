#region Using directives

using System.Collections.Generic;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GNodes;

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

namespace Ceres.MCGS.Analysis;

/// <summary>
/// Represents a principal position in the search tree, including the leaf node,
/// its position, and the complete path from root to leaf.
/// </summary>
public class PrincipalPos
{
  /// <summary>
  /// The leaf node at the end of the principal variation.
  /// </summary>
  public GNode LeafNode { get; init; }

  /// <summary>
  /// The chess position at the leaf node.
  /// </summary>
  public MGPosition LeafPosition { get; init; }

  /// <summary>
  /// The sequence of nodes and positions from root to leaf.
  /// </summary>
  public List<(GNode Node, MGPosition Position)> PathFromRoot { get; init; }

  /// <summary>
  /// Number of times this position was encountered during collection.
  /// </summary>
  public int NumOccurrences { get; init; }


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="leafNode"></param>
  /// <param name="leafPosition"></param>
  /// <param name="pathFromRoot"></param>
  /// <param name="numOccurrences"></param>
  public PrincipalPos(GNode leafNode, MGPosition leafPosition, List<(GNode, MGPosition)> pathFromRoot, int numOccurrences)
  {
    LeafNode = leafNode;
    LeafPosition = leafPosition;
    PathFromRoot = pathFromRoot;
    NumOccurrences = numOccurrences;
  }
}
