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

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

public ref struct GEdgeWithIndexEnumerator
{
  private readonly GNode Node;
  private int currentIndex = -1;
  private readonly int Count;

  public GEdgeWithIndexEnumerator(GNode node)
  {
    Node = node;
    Count = node.NumEdgesExpanded;
  }

  public bool MoveNext()
  {
    currentIndex++;
    return currentIndex < Count;
  }

  public readonly (GEdge Edge, int IndexChildInParent) Current => (Node.ChildEdgeAtIndex(currentIndex), currentIndex);
}
