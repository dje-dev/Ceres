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
using System.Collections.Concurrent;
using System.Diagnostics;

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.Params;
using Ceres.MCTS.Search;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  public partial struct MCTSNodeStruct
  {
    static bool IsSuitable(in MCTSNodeStruct refNode) => 
                      !refNode.IsTranspositionLinked 
                   && !FP16.IsNaN(refNode.V)
                   && !refNode.HasRepetitions; // don't use since might have different repetition count from this node


    /// <summary>
    /// Determine how many nodes could be visited from this node 
    /// if it is used as a transposition root for some other node.
    /// 
    /// Values up to 3 may be returned, if sufficiently many nodes
    /// exist in the subtree and are suitable for use
    /// (e.g. not in flight, not transposition linked).
    /// 
    /// </summary>
    internal int NumUsableSubnodesForCloning(MCTSNodeStore store)
    {
      if (IsTranspositionLinked || Terminal == GameResult.NotInitialized)
      {
        return 0;
      }

      int count = 1;
      if (NumChildrenExpanded > 0)
      {
        ref readonly MCTSNodeStruct child = ref store[in this, 0];
        if (IsSuitable(in child))
        {
          count++;

#if NOT
          // Someday worry about non-equality of repetitions
          // with transposition children.
          if (nodeCloningTarget.Context.ParamsSearch.TestFlag)
          {
            Position posAfter = PosAfterMoveWithRepetitionsSet(nodeCloningTarget, child.PriorMove);

            MCTSNode childNode = nodeCloningTarget.Context.Tree.GetNode(child.Index); // *** inefficient??
            childNode.Annotate();
            Position childNodePos = childNode.InfoRef.Annotation.Pos;

            if (!posAfter.EqualAsRepetition(in childNodePos))
            {
              Console.WriteLine("skip different repetitions");
              Console.WriteLine(posAfter);
              Console.WriteLine(childNodePos);
            }
            else
            {
//              Console.WriteLine("ok to use");
            }
          }
#endif
          // Check if subchild available.
          if (child.NumChildrenExpanded > 0)
          {
            ref readonly MCTSNodeStruct subchild = ref store[in child, 0]; // TODO: make faster???
            if (IsSuitable(in subchild))
            {
              count++;
            }
          }

          // Check if sibing is available.
          // Note that sibling is only considered if the primary child was also available,
          // since when cloning we need to proceed strictly in order and the primary child comes first.
          if (NumChildrenExpanded > 1)
          {
            ref readonly MCTSNodeStruct sibling = ref store[in this, 1];
            if (IsSuitable(in sibling))
            {
              count++;
            }
          }

        }
      }

      return count;
    }

    /// <summary>
    /// Fixes up this node to be fully materialized by
    /// creating a subtree nodes (copied from transposition root)
    /// and breaking the association with the former transposition root
    /// by clearing TranspositionRootIndex and NumVisitsPendingTranspositionRootExtraction.
    /// </summary>
    /// <param name="tree"></param>
    internal void MaterializeSubtreeFromTranspositionRoot(MCTSTree tree)
    {      
      Debug.Assert(N >= 1 && N <= 3);
      Debug.Assert(IsTranspositionLinked);
      Debug.Assert(NumChildrenExpanded == 0);

      ParamsSearch paramsSearch = MCTSManager.ThreadSearchContext.ParamsSearch;
      if (paramsSearch.TranspositionRootBackupSubtreeFracs[0] 
       != paramsSearch.TranspositionCloneNodeSubtreeFracs[0])
      {
        throw new Exception("First element of TranspositionRootBackupSubtreeFracs and TranspositionRootCloneSubtreeFracs must be same");
      }

      MCTSNodeStructIndex startingTranspositionRootIndex = new MCTSNodeStructIndex(TranspositionRootIndex);

      // Copy children immediately at this node.
      CopyUnexpandedChildrenFromOtherNode(tree, startingTranspositionRootIndex);

      // Copy one or possibly two children from subtree to
      // make the copied subtree look what it would if we had not used transposition root reuse
      // (except that the accumulator values may be different if TranspositionCloneNodeSubtreeFracs are nonzero).
      if (N > 1)
      {
        float subtreeFrac = paramsSearch.TranspositionCloneNodeSubtreeFracs[1];
        ref readonly MCTSNodeStruct transpositionRootRef = ref tree.Store.Nodes.nodes[startingTranspositionRootIndex.Index];
        bool cloneSubchild = N == 3;
        TryCloneChild(tree, in transpositionRootRef, ref this, 0, subtreeFrac, cloneSubchild);
      }
    }


    static bool IsValidTranspositionLinkedSourceNode(MCTSNode testNode) => testNode.IsNotNull && testNode.StructRef.IsValidTranspositionLinkedSource;
    public readonly bool IsValidTranspositionLinkedSource => !IsTranspositionLinked
                                                          && !FP16.IsNaN(V)
                                                          && Terminal != Chess.GameResult.NotInitialized;


    /// <summary>
    /// Clones an extant node in tree as a new child in a specified other node.
    /// </summary>
    /// <param name="sourceParent"></param>
    /// <param name="targetParent"></param>
    /// <param name="childIndex"></param>
    /// <param name="cloneSubchildIfPossible"></param>
    /// <returns></returns>
    public static MCTSNodeStructIndex TryCloneChild(MCTSTree tree, 
                                                    in MCTSNodeStruct sourceParentRef, ref MCTSNodeStruct targetParentRef,
                                                    int childIndex, 
                                                    float subtreeFraction,
                                                    bool cloneSubchildIfPossible)
    {
      Debug.Assert(!float.IsNaN(subtreeFraction));
      Debug.Assert(sourceParentRef.NumChildrenExpanded >= childIndex + 1);

      ref readonly MCTSNodeStruct sourceChildRef = ref sourceParentRef.ChildAtIndexRef(childIndex);
      Debug.Assert(sourceChildRef.IsValidTranspositionLinkedSource);
      MCTSNodeStructIndex targetChildIndex = targetParentRef.CreateChild(tree.Store, childIndex);
      ref MCTSNodeStruct targetChildRef = ref tree.Store.Nodes.nodes[targetChildIndex.Index];
      targetParentRef.NumChildrenVisited = (byte)(childIndex + 1);

      // TODO: avoid ChildAtIndex to avoid dictionary lookup?
      targetChildRef.CopyUnexpandedChildrenFromOtherNode(tree, new MCTSNodeStructIndex(sourceChildRef.Index.Index));

      if (LeafEvaluatorTransposition.TRACK_VIRTUAL_VISITS)
      {
        MCTSEventSource.TestMetric1++;
      }

      Debug.Assert(!double.IsNaN(sourceParentRef.W));
      Debug.Assert(!double.IsNaN(sourceChildRef.W));
      Debug.Assert(sourceChildRef.Terminal != Chess.GameResult.NotInitialized);

      float SUBTREE_FRAC = subtreeFraction;
      float NODE_FRAC = 1.0f - SUBTREE_FRAC;

      targetChildRef.Terminal = sourceChildRef.Terminal;
      targetChildRef.N = 1;
      targetChildRef.W = sourceChildRef.V * NODE_FRAC            + sourceChildRef.Q    * SUBTREE_FRAC;
      targetChildRef.mSum = sourceChildRef.MPosition * NODE_FRAC + sourceChildRef.MAvg * SUBTREE_FRAC;
      targetChildRef.dSum = sourceChildRef.DrawP * NODE_FRAC     + sourceChildRef.DAvg * SUBTREE_FRAC;

      targetChildRef.MPosition = sourceChildRef.MPosition;
      targetChildRef.WinP = sourceChildRef.WinP;
      targetChildRef.LossP = sourceChildRef.LossP;

      targetChildRef.ZobristHash = sourceChildRef.ZobristHash;
      targetChildRef.VSecondary = sourceChildRef.VSecondary;
      targetChildRef.Uncertainty = sourceChildRef.Uncertainty;

      targetChildRef.PriorMove = sourceChildRef.PriorMove;
      targetChildRef.miscFields.IndexInParent = (byte)sourceChildRef.IndexInParent;

      //targetChildRef.HashCrosscheck = sourceChildRef.HashCrosscheck;

      // TODO: centralize this logic better (similar to what is in BackupApply).
      if (targetChildRef.Terminal == Chess.GameResult.Draw)
      {
        targetParentRef.DrawKnownToExistAmongChildren = true;
      }

      if (ParamsSelect.VIsForcedLoss(targetChildRef.V))
      {
        targetChildRef.SetProvenLossAndPropagateToParent(targetChildRef.LossP, targetChildRef.MPosition);
      }

      // Possibly move over a second sub-child in the clone.
      // The second descendent (if it exists and is usable)
      // must be either the child of the child, or its sibling.
      if (cloneSubchildIfPossible)
      {
        Debug.Assert(childIndex == 0);

        ref MCTSNodeStruct dummyRef = ref tree.Store.Nodes.nodes[0];
        
        ref readonly MCTSNodeStruct candidateSourceChildChild = ref dummyRef;
        bool candidateSourceChildChildValid = false;
        if (sourceChildRef.NumChildrenVisited > 0)
        {
          candidateSourceChildChild = ref sourceChildRef.ChildAtIndexRef(0);
          candidateSourceChildChildValid = candidateSourceChildChild.IsValidTranspositionLinkedSource;
        }

        ref readonly MCTSNodeStruct candidateSourceChildSibling =  ref dummyRef;
        bool candidateSourceChildSiblingValid = false;
        if (sourceParentRef.NumChildrenVisited > 1)
        {
          candidateSourceChildSibling = ref sourceParentRef.ChildAtIndexRef(1);
          candidateSourceChildSiblingValid = candidateSourceChildSibling.IsValidTranspositionLinkedSource;
        }

        // Make sure only one or the other is marked as valid.
        if (candidateSourceChildChildValid && candidateSourceChildSiblingValid)
        {
          // Both child/child and sibling have been created, use the one selected (created) first.
          if (candidateSourceChildChild.Index.Index < candidateSourceChildSibling.Index.Index)
          {
            candidateSourceChildSiblingValid = false;
          }
          else
          {
            candidateSourceChildChildValid = false;
          }
        }

        float subchildSubtreeFraction = MCTSManager.ThreadSearchContext.ParamsSearch.TranspositionCloneNodeSubtreeFracs[2];
        
        // Do the clone from the transpose child.
        MCTSNodeStructIndex clonedSubchildIndex = default;
        if (candidateSourceChildChildValid)
        {
          clonedSubchildIndex = TryCloneChild(tree, in sourceChildRef, ref targetChildRef, 0, subchildSubtreeFraction, false);
        }
        else if (candidateSourceChildSiblingValid)
        {
          clonedSubchildIndex = TryCloneChild(tree, in sourceParentRef, ref  targetParentRef, 1, subchildSubtreeFraction,  false);
        }

        // In the case of child of the child, the parent statistics
        // need to be updated to reflect this visit from the child.
        // (However not if a sibling was chosen since then the parent
        // is the transposed noded itself which has already had
        // accumulators (N, W, etc.) updated from the backup apply of
        // the earlier "virtual" visits.
        if (candidateSourceChildChildValid)
        {
          ref MCTSNodeStruct clonedSubchild = ref tree.Store.Nodes.nodes[clonedSubchildIndex.Index];

          Debug.Assert(targetChildRef.Index.Index == clonedSubchild.ParentIndex.Index);
          Debug.Assert(clonedSubchild.N == 1);
          Debug.Assert(targetChildRef.N == 1);

          // Update parent statistics to reflect the added subchild.
          // Note that we want to pick up any possible root blending,
          // therefore we take the subtree accumulators (already reflecting such blending)
          // rather than the node values (such as V) which are pure.
          targetChildRef.N++;

          targetChildRef.W += -1 * (clonedSubchild.V * NODE_FRAC + clonedSubchild.Q * SUBTREE_FRAC);
          targetChildRef.mSum += clonedSubchild.MPosition * NODE_FRAC + clonedSubchild.MAvg * SUBTREE_FRAC;
          targetChildRef.dSum += clonedSubchild.DrawP * NODE_FRAC + clonedSubchild.DAvg * SUBTREE_FRAC;
        }
      }

      return targetChildIndex;    
  }


    public void CloneSubtree(MCTSNodeStore store,
                             ConcurrentDictionary<int, MCTSNodeTranspositionVisitor> transpositionDictionary,
                             ref MCTSNodeStruct source, int numNodesToClone)
    {
      throw new NotImplementedException(); // see below
    }


    #region Helpers

    public static Position PosAfterMoveWithRepetitionsSet(MCTSNode node, EncodedMove move)
    {
      //      MCTSNodeStructChild child = node.ChildAtIndexRef(childIndex);
      //      Debug.Assert(!child.IsExpanded);

      MGPosition posMG = node.Annotation.PosMG;
      posMG.MakeMove(ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, in posMG));
      Position pos = posMG.ToPosition;

      if (node.IsRoot || node.Parent.IsRoot)
      {
        return pos; // TODO: what about prior positions in store?
      }

      //      ulong thisHash = pos.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98, true);

      MCTSNode pred = node.Parent;
      int repCount = 0;
      int plyUpCount = 2;
      int maxPlies = ParamsSearch.DrawByRepetitionLookbackPlies;
      while (plyUpCount <= maxPlies)
      {
        //        if (pred.StructRef.ZobristHash == thisHash)
        if (pred.InfoRef.Annotation.Pos.EqualAsRepetition(in pos))
        {
          //          Console.WriteLine("Found pred match " + pred.Depth + " " + repCount);
          repCount++;
        }
        if (pred.IsRoot || pred.Parent.IsRoot)
        {
          break;
        }
        pred = pred.Parent.Parent;
        plyUpCount += 2;
      };

      pos.MiscInfo.SetRepetitionCount(repCount);
      return pos;
    }

    #endregion

  }

}


#if NOT

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
      MCTSNodeIteratorInVisitOrder visitor = new MCTSNodeIteratorInVisitOrder(store, source.Index);
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
        if (source.NumVisitsPendingTranspositionRootExtraction > 1)
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
#endif
