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

using Ceres.Base.DataType.Trees;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Methods relating to traversal of graph.
/// </summary>
public partial struct GNodeStruct
{
  /// <summary>
  /// Delegate for visiting nodes during traversal.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="depth"></param>
  /// <returns></returns>
  public delegate bool VisitorFunc(ref GNodeStruct node, int depth);

  /// <summary>
  /// Delegate for visiting nodes during sequential traversal (returning the node and its index).
  /// </summary>
  /// <param name="node"></param>
  /// <param name="index"></param>
  /// <returns></returns>
  public delegate bool VisitorSequentialFunc(ref GNodeStruct node, NodeIndex index);


  /// <summary>
  /// Traverses all nodes rooted at this node, calling specified delegate for each,
  /// using a specified sequencing.
  /// </summary>
  /// <param name="store"></param>
  /// <param name="visitorFunc"></param>
  /// <param name="traversalType"></param>
  public void Traverse(GraphStore store, VisitorFunc visitorFunc, TreeTraversalType traversalType)
  {
    if (traversalType == TreeTraversalType.Unspecified
     || traversalType == TreeTraversalType.Sequential)
    {
      DoTraverseSequential(store, visitorFunc);
    }
    else
    {
      DoTraverse(store, visitorFunc, traversalType, 0);
    }
  }


  /// <summary>
  /// Traverses node sequentially in order of creation (returning the node and its index).
  /// This traversal will typically be much faster because it is cache friendly.
  /// </summary>
  /// <param name="visitorFunc"></param>
  public void TraverseSequential(GraphStore store, VisitorSequentialFunc visitorFunc)
  {
    ref GNodeStruct node = ref this;

    int numNodes = store.NodesStore.NumTotalNodes;

    for (int i = 1; i < numNodes; i++)
    {
      ref GNodeStruct thisNodeRef = ref store.NodesStore.nodes[i];
      if (!visitorFunc(ref thisNodeRef, new NodeIndex(i)))
      {
        return;
      }
    }
  }


  /// <summary>
  /// Traverses node sequentially in order of creation.
  /// This traversal will typically be much faster because it is cache friendly.
  /// </summary>
  /// <param name="visitorFunc"></param>
  void DoTraverseSequential(GraphStore store, VisitorFunc visitorFunc)
  {
    ref GNodeStruct node = ref this;

    // TODO: consider moving this into the MCTSNodeStructStorage class to keep implementation details in there
    int numNodes = store.NodesStore.NumTotalNodes;

    for (int i = 1; i < numNodes; i++)
    {
      ref GNodeStruct thisNodeRef = ref store.NodesStore.nodes[i];

      // TODO: Depth is not available
      if (!visitorFunc(ref store.NodesStore.nodes[i], -1))
      {
        return;
      }
    }
  }


  /// <summary>
  /// Traverses the current node in the graph using the specified traversal strategy and invokes the visitor function
  /// for this node.
  /// </summary>
  /// <param name="store">The graph store containing the nodes and edges to be traversed.</param>
  /// <param name="visitorFunc">A delegate that is called for the current node during traversal. The function receives a reference to the node and
  /// the current depth, and should return <see langword="true"/> to continue traversal or <see langword="false"/> to
  /// stop.</param>
  /// <param name="traversalType">The traversal strategy to use for visiting nodes, such as breadth-first or depth-first.</param>
  /// <param name="depth">The current depth of traversal, typically representing the distance from the root node.</param>
  /// <returns>true if the visitor function returns true for the current node; otherwise, false.</returns>
  bool DoTraverse(GraphStore store, VisitorFunc visitorFunc, TreeTraversalType traversalType, int depth)
  {
    ref GNodeStruct node = ref this;

    if (traversalType == TreeTraversalType.BreadthFirst)
    {
      if (!visitorFunc(ref node, depth))
      {
        return false;
      }
    }

    if (traversalType == TreeTraversalType.DepthFirst)
    {
      if (!visitorFunc(ref node, depth))
      {
        return false;
      }
    }

    return true;
  }
}
