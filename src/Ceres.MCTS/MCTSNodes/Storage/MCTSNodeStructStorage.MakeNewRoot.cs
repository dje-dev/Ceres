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
using System.Diagnostics;

using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  public partial class MCTSNodeStructStorage
  {
    // TODO: move this into MCTSNodeStoreClass??

    #region Structure modification

    /// <summary>
    /// Makes node within the tree the root child, reorganizing the nodes and child arrays.
    /// 
    /// Critically, we do this operation in situ to avoid having to transiently allocate 
    /// extremely large memory objects.
    /// 
    /// The operation consists of 3 stages:
    ///   - traverse the subtree breadth-first starting at the new root, 
    ///     building a bitmap of which nodes are children.
    ///     
    ///   - traverse the node array sequentially, processing each node that is a member of this bitmap by
    ///     moving it into the new position in the store (including modifying associated child and parent references)
    ///     Also we build a table of all the new children that need to be moved.
    ///     
    ///   - using the child table built above, shift all children down. Because nodes may have children written
    ///     out of order, we don't know for sure there is enough space available. Therefore we sort the table
    ///     based on their new position in the table and then shift them down, insuring we don't overwrite
    ///     children yet to be shifted.
    ///       
    /// Additionally we may have to recreate the transposition roots dictionary because
    /// the set of roots is smaller (the retined subtree only) and the node indices will change.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="newRootChild"></param>
    /// <param name="newPriorMoves"></param>
    /// <param name="transpositionRoots"></param>
    public static void MakeChildNewRoot(MCTSNodeStore store, float policySoftmax, 
                                        ref MCTSNodeStruct newRootChild,
                                        PositionWithHistory newPriorMoves,
                                        PositionEvalCache cacheNonRetainedNodes,
                                        TranspositionRootsDict transpositionRoots)
    {
#if DEBUG
      store.Validate();
#endif

      // Nothing to do if the requested node is already currently the root
      if (newRootChild.Index == store.RootNode.Index)
      {
        // Nothing changing in the tree, just flush the cache references
        store.ClearAllCacheIndices();
      }
      else
      {
        DoMakeChildNewRoot(store, policySoftmax, ref newRootChild, newPriorMoves, cacheNonRetainedNodes, transpositionRoots);
      }

#if DEBUG
      store.Validate();
#endif
    }

    static void DoMakeChildNewRoot(MCTSNodeStore store, float policySoftmax, ref MCTSNodeStruct newRootChild,
                                        PositionWithHistory newPriorMoves,
                                        PositionEvalCache cacheNonRetainedNodes,
                                        TranspositionRootsDict transpositionRoots)
    {
      ChildStartIndexToNodeIndex[] childrenToNodes;

      uint numNodesUsed;
      uint numChildrenUsed;
      BitArray includedNodes;

      int newRootChildIndex = newRootChild.Index.Index;
      int newIndexOfNewParent = -1;

      int nextAvailableNodeIndex = 1;

      // Traverse this subtree, building a bit array of visited nodes
      includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(store, ref newRootChild, out numNodesUsed);

      //using (new TimingBlock("Build position cache "))
      if (cacheNonRetainedNodes != null)
      {
        long estNumNodes = store.RootNode.N - numNodesUsed;
        cacheNonRetainedNodes.InitializeWithSize((int)estNumNodes);
        ExtractPositionCacheNonRetainedNodes(store, policySoftmax, includedNodes, in newRootChild, cacheNonRetainedNodes);
      }

      // We will constract a table indicating the starting index and length of
      // children associated with the nodes we are extracting
      childrenToNodes = GC.AllocateUninitializedArray<ChildStartIndexToNodeIndex>((int)numNodesUsed);

      void RewriteNodes()
      {
        // TODO: Consider that the above is possibly all we need to do in some case
        //       Suppose the subtree is very large relative to the whole
        //       This approach would be much faster, and orphan an only small part of the storage

        // Now scan all above nodes. 
        // If they don't belong, ignore. 
        // If they do belong, swap them down to the next available lower location
        // Note that this can't be parallelized, since we have to do it strictly in order of node index
        int numRewrittenNodesDone = 0;
        for (int i = 2; i < store.Nodes.nextFreeIndex; i++)
        {
          if (includedNodes.Get(i))
          {
            ref MCTSNodeStruct thisNode = ref store.Nodes.nodes[i];

            // Reset any cache entry
            thisNode.CacheIndex = 0;

            // Not possible to support transposition linked nodes,
            // since the root may be in a part of the tree that is not retained
            // and possibly already overwritten.
            // We expect them to have already been materialized by the time we reach this point.
            Debug.Assert(!thisNode.IsTranspositionLinked);
            Debug.Assert(thisNode.NumNodesTranspositionExtracted == 0);

            // Remember this location if this is the new parent
            if (i == newRootChildIndex) newIndexOfNewParent = nextAvailableNodeIndex;

            // Move the actual node
            MoveNodePosition(store, new MCTSNodeStructIndex(i), new MCTSNodeStructIndex(nextAvailableNodeIndex));

            // Reset all transposition information
            thisNode.NextTranspositionLinked = 0;

            childrenToNodes[numRewrittenNodesDone] = new ChildStartIndexToNodeIndex(thisNode.childStartBlockIndex, nextAvailableNodeIndex, thisNode.NumPolicyMoves);

            // Re-insert this into the transpositionRoots (with the updated node index)
            // TODO: this is expensive, try to optimize by
            //       possibly avoiding duplicates or parallelizing
            transpositionRoots?.TryAdd(thisNode.ZobristHash, nextAvailableNodeIndex);

            Debug.Assert(thisNode.NumNodesTranspositionExtracted == 0);

            numRewrittenNodesDone++;
            nextAvailableNodeIndex++;
          }
        }
      }

      // Rewrite nodes (and associated children)
      RewriteNodes();

      // Perform a sort so we can shift down the children in order of apperance
      // This is necessary because it guarantess we will always have sufficient room
      new Span<ChildStartIndexToNodeIndex>(childrenToNodes).Sort();
      RewriteChildren(store, childrenToNodes);

      // Finally swap this new root in the root position (slot 1)
      if (newIndexOfNewParent != 1)
      {
        SwapNodePositions(store, new MCTSNodeStructIndex(newIndexOfNewParent), new MCTSNodeStructIndex(1));
      }

      // Finally, make this as the root (no parent)
      store.Nodes.nodes[1].ParentIndex = default;
      store.Nodes.nodes[1].PriorMove = default;
      store.Nodes.nodes[1].CacheIndex = 0;

      // Zero out the new unused space between the new top of nodes and the prior top of nodes
      // since MCTSNodeStruct is assumed to be all zeros when allocated
      int numUnusedNodes = store.Nodes.nextFreeIndex - nextAvailableNodeIndex;
      store.Nodes.nodes.Clear(nextAvailableNodeIndex, numUnusedNodes);

      // Mark the new top of the nodes
      store.Nodes.nextFreeIndex = nextAvailableNodeIndex;

      // Update the prior moves
      store.Nodes.PriorMoves = new PositionWithHistory(newPriorMoves);
    }

    private static void ExtractPositionCacheNonRetainedNodes(MCTSNodeStore store, float policySoftmax, BitArray includedNodes, in MCTSNodeStruct newRoot, 
                                                             PositionEvalCache cacheNonRetainedNodes)
    {
      for (int nodeIndex = 1; nodeIndex < store.Nodes.NumTotalNodes; nodeIndex++)
      {
        ref MCTSNodeStruct nodeRef = ref store.Nodes.nodes[nodeIndex];

        bool nodeRetained = includedNodes[nodeIndex];
        int curGeneration = nodeRef.ReuseGenerationNum;
        if (!includedNodes[nodeIndex] || curGeneration > 0)
        {
          // This node is either being retained,
          // or was already part of an older (discarded) generation
          if (curGeneration < byte.MaxValue) nodeRef.ReuseGenerationNum++;
          continue;
        }
        {
          if (nodeRef.IsTranspositionLinked)
          {
            continue;
          }

          // TODO: someday filter out impossible positions given new root
          // (consider pieces on board, pawns pushed castling rights, etc.)
          //bool isEligible = true;
          //if (nodeRef.PieceCount > numPiecesNewRoot)
          //  isEligible = false;

          CompressedPolicyVector policy = default;
          MCTSNodeStructUtils.ExtractPolicyVector(policySoftmax, in nodeRef, ref policy);

          if (nodeRef.ZobristHash == 0)
          {
            throw new Exception("Internal error: node encountered without hash");
          }
          // TODO: could the cast to FP16 below lose useful precision? Perhaps save as float instead

          if (nodeRef.Terminal == Chess.GameResult.Unknown)
          {
            cacheNonRetainedNodes.Store(nodeRef.ZobristHash, nodeRef.Terminal, (FP16)nodeRef.W, nodeRef.LossP, nodeRef.MPosition, in policy);
          }
        }
      }

    }


    private static void RewriteChildren(MCTSNodeStore store, ChildStartIndexToNodeIndex[] childrenToNodes)
    {
      int nextAvailableChildBlockIndex = 1;

      // Loop thru all the new nodes to process (already sorted by child start index)
      for (int i = 0; i < childrenToNodes.Length; i++)
      {
        ChildStartIndexToNodeIndex childStartToNode = childrenToNodes[i];

        if (childStartToNode.PriorChildStartBlockIndex != 0)
        {
          // Move the actual children entries
          store.Children.CopyEntries(childStartToNode.PriorChildStartBlockIndex, nextAvailableChildBlockIndex, childStartToNode.NumPolicyMoves);

          // Modify the child start index in the node itself
          store.Nodes.nodes[childStartToNode.NewNodeIndex].childStartBlockIndex = nextAvailableChildBlockIndex;

          // Advance our index of next child
          nextAvailableChildBlockIndex += MCTSNodeStructChildStorage.NumBlocksReservedForNumChildren(childStartToNode.NumPolicyMoves);
        }
      }

      // Reset children to new length
      // Note that it is not assumed the unused portion of the array is zeros,
      // so no need to zero anything here
      store.Children.nextFreeBlockIndex = nextAvailableChildBlockIndex;
    }

    static void ModifyChildrensParentRef(MCTSNodeStore store, ref MCTSNodeStruct node, MCTSNodeStructIndex newParentIndex)
    {
      Span<MCTSNodeStructChild> children = node.ChildrenFromStore(store);
      int numChildrenExpanded = node.NumChildrenExpanded;
      for (int i = 0; i < numChildrenExpanded; i++)
      {
        children[i].ChildRefFromStore(store).ParentIndex = newParentIndex;
      }
    }

    public static void MoveNodePosition(MCTSNodeStore store, MCTSNodeStructIndex from, MCTSNodeStructIndex to)
    {
      if (from != to)
      {
        Span<MCTSNodeStruct> nodes = store.Nodes.Span;

        Debug.Assert(!nodes[from.Index].IsRoot);

        ref MCTSNodeStruct fromNodeRef = ref nodes[from.Index];

        // Swap references from parents to these children
        ref MCTSNodeStruct parent = ref nodes[fromNodeRef.ParentRef.Index.Index];
        parent.ModifyExpandedChildIndex(store, from, to);

        // Swap the parent references of any children in both
        ModifyChildrensParentRef(store, ref fromNodeRef, to);

        // Swap nodes themselves
        nodes[to.Index] = nodes[from.Index];
      }
    }


    public static void SwapNodePositions(MCTSNodeStore store, MCTSNodeStructIndex i1, MCTSNodeStructIndex i2)
    {
      //      if (nodes[i1.Index].ParentIndex != i2 && nodes[i2.Index].ParentIndex != i1)
      //         Console.WriteLine("Internal error: not supported");

      if (i1 != i2)
      {
        MemoryBufferOS<MCTSNodeStruct> nodesBuffer = store.Nodes.nodes;

        // Swap references from parents to these children
        ModifyParentsChildRef(store, i1, i2);
        ModifyParentsChildRef(store, i2, i1);

        // Swap the parent references of any children in both
        ModifyChildrensParentRef(store, ref nodesBuffer[i1.Index], i2);
        ModifyChildrensParentRef(store, ref nodesBuffer[i2.Index], i1);

        // Swap nodes themselves
        MCTSNodeStruct temp = nodesBuffer[i1.Index];
        nodesBuffer[i1.Index] = nodesBuffer[i2.Index];
        nodesBuffer[i2.Index] = temp;
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

    #endregion
  }

  #region Helper classes

  /// <summary>
  /// Helper class used to track association between children start indices
  /// and node indices during tree reconstruction.
  /// </summary>
  readonly struct ChildStartIndexToNodeIndex : IComparable<ChildStartIndexToNodeIndex>
  {
    public readonly int PriorChildStartBlockIndex;
    public readonly int NewNodeIndex;
    public readonly short NumPolicyMoves;

    internal ChildStartIndexToNodeIndex(int childStartIndex, int nodeIndex, short numPolicyMoves)
    {
      PriorChildStartBlockIndex = childStartIndex;
      NewNodeIndex = nodeIndex;
      NumPolicyMoves = numPolicyMoves;
    }

    public int CompareTo(ChildStartIndexToNodeIndex other)
    {
      if (other.PriorChildStartBlockIndex < PriorChildStartBlockIndex)
      {
        return 1;
      }

      return other.PriorChildStartBlockIndex > PriorChildStartBlockIndex ? -1 : 0;
    }

    public override string ToString()
    {
      return $"<Prior child entries starting at {PriorChildStartBlockIndex} for node at new index {NewNodeIndex} with policy move count {NumPolicyMoves}";
    }
  }

    #endregion
}
