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

using Ceres.Base.DataTypes;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Storage;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  public partial struct MCTSNodeStruct
  {
    public static MCTSNode CloneChild(MCTSNode sourceParent, MCTSNode targetParent, int childIndex)
    {
      Debug.Assert(childIndex < sourceParent.NumChildrenExpanded);
      Debug.Assert(childIndex == targetParent.NumChildrenExpanded); // must expand strictly in order by index

      MCTSTree tree = sourceParent.Tree;

      MCTSNode sourceChild = sourceParent.ChildAtIndex(0);

      lock (sourceParent)
      {
        lock (sourceChild)
        {
          ref readonly MCTSNodeStruct sourceParentRef = ref sourceParent.Ref;
          ref readonly MCTSNodeStruct sourceChildRef = ref sourceChild.Ref;

          ref MCTSNodeStruct targetParentRef = ref targetParent.Ref;

          // Create target child and copy children (as unexpanded).
          targetParent.Ref.CopyUnexpandedChildrenFromOtherNode(tree, new MCTSNodeStructIndex(sourceParent.Index));

          MCTSNode targetChild = targetParent.CreateChild(childIndex);
          ref MCTSNodeStruct targetChildRef = ref targetChild.Ref;


          // TODO: avoid ChildAtIndex to avoid dictionarylookup?
          targetChildRef.CopyUnexpandedChildrenFromOtherNode(tree, new MCTSNodeStructIndex(sourceParent.ChildAtIndex(0).Index));

          targetChildRef.numChildrenVisited = 0;
          targetChildRef.NumChildrenExpanded = 0;

          Debug.Assert(!double.IsNaN(sourceParent.W));
          Debug.Assert(!double.IsNaN(sourceChildRef.W));

          targetChildRef.N = 1;
          targetChildRef.Terminal = sourceChildRef.Terminal;
          targetChildRef.W = sourceChildRef.V;
          targetChildRef.ZobristHash = sourceChildRef.ZobristHash;
          targetChildRef.MPosition = sourceChildRef.MPosition;
          targetChildRef.mSum = sourceChildRef.MPosition;
          targetChildRef.dSum = 1.0f - sourceChildRef.WinP - sourceChildRef.LossP;
          targetChildRef.WinP = sourceChildRef.WinP;
          targetChildRef.LossP = sourceChildRef.LossP;

          //      targetChildRef.P = sourceChildRef.P;
          //      targetChildRef.ZobristHash = sourceChildRef.ZobristHash;

          /* 
          *N
          *W
          P
          NInFlight
          NInFlight2
          *WinP
          *LossP
          *mSum
          *dSum
          ParentIndex
          childStartBlockIndex
          PriorMove
          *ZobristHash
          NumPolicyMoves
          numChildrenVisited
          NumChildrenExpanded
          Unused1
          CacheIndex
          VSecondary
          Uncertainty


          MiscFields
          *  Terminal
            DrawKnownToExistAmongChildren
          *  MPosition

           */
          return targetChild;
        }
      }
    }


    #region Deep clone

    /// <summary>
    /// Clones a subtree rooted at "source" onto this node,
    /// but only as many nodes in the subtre as specified by "numNodesToClone" (visited in order of creation)
    /// </summary>
    /// <param name="store"></param>
    /// <param name="transpositionDictionary"></param>
    /// <param name="source"></param>
    /// <param name="numNodesToClone"></param>
    public void CloneSubtree(MCTSNodeStore store,
                             ConcurrentDictionary<int, MCTSNodeTranspositionVisitor> transpositionDictionary,
                             ref MCTSNodeStruct source, int numNodesToClone)
    {
      Debug.Assert(source.N >= numNodesToClone);
      int count = 0;

      MCTSNodeStructIndex lastIndex = default;
      MCTSNodeSequentialVisitor visitor = new MCTSNodeSequentialVisitor(store, source.Index);
      foreach (MCTSNodeStructIndex childNodeIndex in visitor.Iterate)
      {
        if (++count == numNodesToClone)
        {
          lastIndex = childNodeIndex;
          break;
        }
      }

      CloneSubtree(store, transpositionDictionary, ref source, lastIndex);
    }


    /// <summary>
    /// Clones a full subtree (transposition equivalent).
    /// </summary>
    /// <param name="store"></param>
    /// <param name="activeTranspositionVisitors"></param>
    /// <param name="source"></param>
    /// <param name="highestIndexToKeep"></param>
    public void CloneSubtree(MCTSNodeStore store,
                             ConcurrentDictionary<int, MCTSNodeTranspositionVisitor> activeTranspositionVisitors,
                             ref MCTSNodeStruct source, MCTSNodeStructIndex highestIndexToKeep)
    {
      throw new NotImplementedException("Needs remediation due to change to use allocate children in blocks rather than indivdually, also more fields need cloning");

      Debug.Assert(NumChildrenExpanded == 0);
      Debug.Assert(!source.Detached);

      // Copy the node itself (except preserve the parent, N, and NInFlight)
      MCTSNodeStructIndex originalParent = this.ParentIndex;
      int originalN = this.N;
      short originalNInFlight = this.NInFlight;
      short originalNInFlight2 = this.NInFlight2;
      this = source;
      this.ParentIndex = originalParent;
      this.N = originalN;
      this.NInFlight = originalNInFlight;
      this.NInFlight2 = originalNInFlight2;

      if (source.IsTranspositionLinked)
      {
        if (source.NumNodesTranspositionExtracted > 1)
        {
          // This was a linked node, with active visitation in progress
          // Clone the visitor and add to the table so we are now another visitor 
          // that starts out in the same place as the source
          MCTSNodeTranspositionVisitor visitor = activeTranspositionVisitors[source.Index.Index];
          MCTSNodeTranspositionVisitor clone = visitor.Clone() as MCTSNodeTranspositionVisitor;
          activeTranspositionVisitors[Index.Index] = clone;
          return;
        }
        else
        {
          // This is a linked node, but there was only a single V copied,
          // nothing more to do
          return;
        }
      }


      int clonedN = 1;
      int clonedNumChildrenVisited = 0;
      int clonedNumChildrenExpanded = 0;
      float clonedSumPVisited = 0f;
      double clonedSumW = V;

      // Allocate and copy over all the children
      if (source.NumPolicyMoves > 0)
      {
        childStartBlockIndex = (int)store.Children.AllocateEntriesStartBlock(NumPolicyMoves);

        Span<MCTSNodeStructChild> destChildren = Children;
        for (int i = 0; i < source.NumPolicyMoves; i++)
        {
          MCTSNodeStructChild sourceChild = source.ChildAtIndex(i);

          if (sourceChild.IsExpanded && sourceChild.ChildRef.Index.Index <= highestIndexToKeep.Index)
          {
            if (sourceChild.ChildRef.N > 0) clonedNumChildrenVisited++;
            clonedNumChildrenExpanded++;
            clonedSumPVisited += sourceChild.P;

            // Allocate a new node and link it into this child slot
            MCTSNodeStructIndex newChildIndex = store.Nodes.AllocateNext();
            destChildren[i].SetExpandedChildIndex(newChildIndex);

            ref MCTSNodeStruct newChild = ref store.Nodes.nodes[newChildIndex.Index];
            newChild.ParentIndex = Index;
            newChild.CloneSubtree(store, activeTranspositionVisitors, ref sourceChild.ChildRef, highestIndexToKeep);
            clonedN += newChild.N;
            clonedSumW -= newChild.W; // subtraction since the newChild.W is from perspective of other side
          }
          else
          {
            if (sourceChild.IsExpanded)
            {
              ref MCTSNodeStruct childNodeRef = ref sourceChild.ChildRef;
              destChildren[i].SetUnexpandedPolicyValues(childNodeRef.PriorMove, childNodeRef.P);
            }
            else
            {
              destChildren[i] = sourceChild;
            }

          }

        }

        N = clonedN;
        NumChildrenVisited = (byte)clonedNumChildrenVisited;
        NumChildrenExpanded = (byte)clonedNumChildrenExpanded;
        W = (FP16)clonedSumW;
      }
      #endregion

    }

  }
}