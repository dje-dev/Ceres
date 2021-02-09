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
    public delegate bool VisitorFunc(ref MCTSNodeStruct node);
    public delegate bool VisitorSequentialFunc(ref MCTSNodeStruct node, MCTSNodeStructIndex index);


    public void Traverse(MCTSNodeStore store, VisitorFunc visitorFunc, TreeTraversalType traversalType)
    {
      if (traversalType == TreeTraversalType.Unspecified
       || traversalType == TreeTraversalType.Sequential)
        DoTraverseSequential(store, visitorFunc);
      else
        DoTraverse(store, visitorFunc, traversalType);
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
            return;
        }
      }

    }


    /// <summary>
    /// Traverses node sequentially in order of creation.
    /// This traversal will typically be much faster because it is cache friendly.
    /// 
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
          if (!visitorFunc(ref store.Nodes.nodes[i]))
            return;
        }
      }
    }


    bool DoTraverse(MCTSNodeStore store, VisitorFunc visitorFunc, TreeTraversalType traversalType)
    {
      ref MCTSNodeStruct node = ref this;

      if (traversalType == TreeTraversalType.BreadthFirst)
        if (!visitorFunc(ref node))
          return false;

      if (!node.IsTranspositionLinked)
      {
        // For efficiency, we track number of children already visited and 
        // abort visiting children when we know there are no more unvisited
        int numChildrenVisited = 0;
        int numChildren = node.N - 1;
        int numPolicyMoves = node.NumPolicyMoves;

        for (int i = 0; i < numPolicyMoves; i++)
        {
          MCTSNodeStructChild child = node.ChildAtIndex(i);
          if (child.IsExpanded)
          {
            ref MCTSNodeStruct childRef = ref child.ChildRef;
            childRef.DoTraverse(store, visitorFunc, traversalType);

            // Update statistcs and check if we can early abort
            numChildrenVisited += childRef.N;
            if (numChildrenVisited == numChildren) break;
          }
        }
      }

      if (traversalType == TreeTraversalType.DepthFirst)
        if (!visitorFunc(ref node))
          return false;

      return true;
    }

  }
}
