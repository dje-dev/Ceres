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

using System.Diagnostics;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  public partial class MCTSNodeStructStorage
  {
    /// <summary>
    /// Swaps 
    /// </summary>
    /// <param name="store"></param>
    /// <param name="i1"></param>
    /// <param name="i2"></param>
    public static void SwapNodePositions(MCTSNodeStore store, MCTSNodeStructIndex i1, MCTSNodeStructIndex i2)
    {
      //      if (nodes[i1.Index].ParentIndex != i2 && nodes[i2.Index].ParentIndex != i1)
      //         Console.WriteLine("Internal error: not supported");

      if (i1 != i2)
      {
        Span<MCTSNodeStruct> nodes = store.Nodes.nodes.Span;

        // Swap references from parents to these children
        ModifyParentsChildRef(store, i1, i2);
        ModifyParentsChildRef(store, i2, i1);

        // Swap the parent references of any children in both
        ModifyChildrensParentRef(store, nodes, ref nodes[i1.Index], i2);
        ModifyChildrensParentRef(store, nodes, ref nodes[i2.Index], i1);

        // Swap nodes themselves
        MCTSNodeStruct temp = nodes[i1.Index];
        nodes[i1.Index] = nodes[i2.Index];
        nodes[i2.Index] = temp;
      }
    }


    /// <summary>
    /// Moves a node to another location (strictly to a lower index) 
    /// in the node store, updating children linked data structures as needed.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="nodes"></param>
    /// <param name="from"></param>
    /// <param name="to"></param>
    public static void MoveNodeDown(MCTSNodeStore store, Span<MCTSNodeStruct> nodes,
                                    MCTSNodeStructIndex from, MCTSNodeStructIndex to)
    {
      // Node order must be preserved, this is assumed/required
      // several places in the code.
      //
      // For example, tree rebuilding is done in situ
      // and nodes are shifted strictly down to
      // prevent overwriting data yet to be rewritten.
      Debug.Assert(to.Index <= from.Index);
      if (from != to)
      {
        Debug.Assert(!nodes[from.Index].IsRoot);

        ref MCTSNodeStruct fromNodeRef = ref nodes[from.Index];

        // Swap references from parents to these children
        ref MCTSNodeStruct parent = ref nodes[fromNodeRef.ParentRef.Index.Index];
        parent.ModifyExpandedChildIndex(store, from, to);

        // Swap the parent references of any children in both
        ModifyChildrensParentRef(store, nodes, ref fromNodeRef, to);

        // Swap nodes themselves
        nodes[to.Index] = nodes[from.Index];
      }
    }


    /// <summary>
    /// Iterate over the children of the parent of "from" to find "from" 
    /// and change that child index to point to the new index "to" of that child
    /// </summary>
    /// <param name="store"></param>
    /// <param name="from"></param>
    /// <param name="to"></param>
    public static void ModifyParentsChildRef(MCTSNodeStore store, MCTSNodeStructIndex from, MCTSNodeStructIndex to)
    {
      if (!store.Nodes.nodes[from.Index].IsRoot)
      {
        ref MCTSNodeStruct parent = ref store.Nodes.nodes[from.Index].ParentRef;
        parent.ModifyExpandedChildIndex(store, from, to);
      }
    }


    static void ModifyChildrensParentRef(MCTSNodeStore store, Span<MCTSNodeStruct> rawNodes,
                                         ref MCTSNodeStruct node, MCTSNodeStructIndex newParentIndex)
    {
      if (!node.IsTranspositionLinked && node.NumChildrenExpanded > 0)
      {
        Span<MCTSNodeStructChild> children = node.ChildrenFromStore(store);
        int numChildrenExpanded = node.NumChildrenExpanded;
        for (int i = 0; i < numChildrenExpanded; i++)
        {
          rawNodes[children[i].ChildIndex.Index].ParentIndex = newParentIndex;
        }
      }
    }

  }

}