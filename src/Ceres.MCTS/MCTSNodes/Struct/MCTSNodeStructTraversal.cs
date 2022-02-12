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
using Ceres.MCTS.MTCSNodes.Storage;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Methods relating to traverso of tree.
  /// </summary>
  public partial struct MCTSNodeStruct
  {
    public delegate bool VisitorFunc(ref MCTSNodeStruct node, int depth);
    public delegate bool VisitorSequentialFunc(ref MCTSNodeStruct node, MCTSNodeStructIndex index);


    /// <summary>
    /// Traverses all nodes rooted at this node, calling specified delegate for each,
    /// using a specified sequencing.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="visitorFunc"></param>
    /// <param name="traversalType"></param>
    public void Traverse(MCTSNodeStore store, VisitorFunc visitorFunc, TreeTraversalType traversalType)
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


    /// Traverses node sequentially in order of creation (returning the node and its index).
    /// This traversal will typically be much faster because it is cache friendly.
    /// 
    /// 
    /// </summary>
    /// <param name="visitorFunc"></param>
    public void TraverseSequential(MCTSNodeStore store, VisitorSequentialFunc visitorFunc)
    {
      ref MCTSNodeStruct node = ref this;

      // TODO: consider moving this into the MCTSNodeStructStorage class to keep implementation details in there
      int numNodes = store.Nodes.NumTotalNodes;

      for (int i = 1; i < numNodes; i++)
      {
        ref MCTSNodeStruct thisNodeRef = ref store.Nodes.nodes[i];
        if (!thisNodeRef.Detached)
        {
          if (!visitorFunc(ref thisNodeRef, new MCTSNodeStructIndex(i)))
          {
            return;
          }
        }
      }
    }


    /// <summary>
    /// Traverses node sequentially in order of creation.
    /// This traversal will typically be much faster because it is cache friendly.
    /// </summary>
    /// <param name="visitorFunc"></param>
    void DoTraverseSequential(MCTSNodeStore store, VisitorFunc visitorFunc)
    {
      ref MCTSNodeStruct node = ref this;

      // TODO: consider moving this into the MCTSNodeStructStorage class to keep implementation details in there
      int numNodes = store.Nodes.NumTotalNodes;

      for (int i = 1; i < numNodes; i++)
      {
        ref MCTSNodeStruct thisNodeRef = ref store.Nodes.nodes[i];
        if (!thisNodeRef.Detached)
        {
          // Depth is not available
          if (!visitorFunc(ref store.Nodes.nodes[i], -1))
          {
            return;
          }
        }
      }
    }


    bool DoTraverse(MCTSNodeStore store, VisitorFunc visitorFunc, TreeTraversalType traversalType, int depth)
    {
      ref MCTSNodeStruct node = ref this;

      if (traversalType == TreeTraversalType.BreadthFirst)
      {
        if (!visitorFunc(ref node, depth))
        {
          return false;
        }
      }

      if (!node.IsTranspositionLinked)
      {
        int numExpanded = node.NumChildrenExpanded;
        for (int i = 0; i < numExpanded; i++)
        {
          node.ChildAtIndex(i).ChildRef(store).DoTraverse(store, visitorFunc, traversalType, depth +1);
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
}
