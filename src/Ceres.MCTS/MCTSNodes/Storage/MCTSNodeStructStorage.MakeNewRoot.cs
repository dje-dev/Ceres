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
    // TODO: move this into MCTSNodeStoreClass??

    // TODO: someday remove this, always use fast mode
    internal const bool USE_FAST_TREE_REBUILD = true;

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
    /// <param name="tree"></param>
    /// <param name="newRootChild"></param>
    /// <param name="newPriorMoves"></param>
    /// <param name="transpositionRoots"></param>
    public static void MakeChildNewRoot(MCTSTree tree, float policySoftmax, 
                                        ref MCTSNodeStruct newRootChild,
                                        PositionWithHistory newPriorMoves,
                                        PositionEvalCache cacheNonRetainedNodes,
                                        TranspositionRootsDict transpositionRoots)
    {
#if DEBUG
      tree.Store.Validate();
#endif

      // Nothing to do if the requested node is already currently the root
      if (newRootChild.Index == tree.Store.RootNode.Index)
      {
        // Nothing changing in the tree, just flush the cache references
        tree.Store.ClearAllCacheIndices();
      }
      else
      {
        DoMakeChildNewRoot(tree, policySoftmax, ref newRootChild, newPriorMoves, cacheNonRetainedNodes, transpositionRoots);
      }

#if DEBUG
      tree.Store.Validate();
#endif
    }

    static void MaterializeNonRetainedNodes(MCTSTree tree, BitArray includedNodes, int newRootChildIndex)
    {
      int numPrematerialized = 0;
      MCTSNodeStore store = tree.Store;
      MemoryBufferOS<MCTSNodeStruct> rawNodes = store.Nodes.nodes;

      for (int i = 2; i < store.Nodes.nextFreeIndex; i++)
      {
        if (includedNodes.Get(i))
        {
          ref MCTSNodeStruct thisNode = ref store.Nodes.nodes[i];

          // Materialize any nodes which are transposition linked to a non-retained node
          if (thisNode.IsTranspositionLinked)
          {
            ref MCTSNodeStruct nodeRef = ref rawNodes[i];
            int transpositionRootIndex = nodeRef.TranspositionRootIndex;
            Debug.Assert(!rawNodes[transpositionRootIndex].IsTranspositionLinked);

            bool linkedNodeRetained = includedNodes.Get(transpositionRootIndex);

            // NOTE: Generally it is not expected that a node will link forward to a transposition root,
            //       so this will probably always be false, but for safety prematerialize if this happens.
            bool linkedNodeSequentiallyLater = transpositionRootIndex > i;

            if (!linkedNodeRetained || i == newRootChildIndex || linkedNodeSequentiallyLater)
            {
              numPrematerialized++;
              
              // We are not retaining the transposition root, so we must 
              // unlink the node from parent (copying over all child information).
              nodeRef.CopyUnexpandedChildrenFromOtherNode(tree, new MCTSNodeStructIndex(transpositionRootIndex), true);
            }
          }
        }

      }

#if DEBUG
      tree.Store.Validate();
#endif
    }


    static void DoMakeChildNewRoot(MCTSTree tree, float policySoftmax, ref MCTSNodeStruct newRootChild,
                                   PositionWithHistory newPriorMoves,
                                   PositionEvalCache cacheNonRetainedNodes,
                                   TranspositionRootsDict transpositionRoots)
    {

      MCTSNodeStore store = tree.Store;
      ChildStartIndexToNodeIndex[] childrenToNodes;

      uint numNodesUsed;
      int newRootChildIndex = newRootChild.Index.Index;
      int newIndexOfNewParent = -1;

      int nextAvailableNodeIndex = 1;

      // Traverse this subtree, building a bit array of visited nodes
      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(store, ref newRootChild, out numNodesUsed);

      if (USE_FAST_TREE_REBUILD)
      {
        MaterializeNonRetainedNodes(tree, includedNodes, newRootChildIndex);
      }

      // Possibly extract retained nodes into a cache.
      if (cacheNonRetainedNodes != null)
      {
        long estNumNodes = store.RootNode.N - numNodesUsed;
        cacheNonRetainedNodes.InitializeWithSize((int)estNumNodes);
        ExtractPositionCacheNonRetainedNodes(store, policySoftmax, includedNodes, in newRootChild, cacheNonRetainedNodes);
      }

      // We will constract a table indicating the starting index and length of
      // children associated with the nodes we are extracting
      childrenToNodes = GC.AllocateUninitializedArray<ChildStartIndexToNodeIndex>((int)numNodesUsed);


      int RewriteNodesBuildChildrenInfo()
      {
        Task presorter = null;
        int numChildrenFixups = 0;

        // TODO: Consider that the above is possibly all we need to do in some case
        //       Suppose the subtree is very large relative to the whole
        //       This approach would be much faster, and orphan an only small part of the storage
        MemoryBufferOS<MCTSNodeStruct> rawNodes = store.Nodes.nodes;
        Span<MCTSNodeStruct> nodesSpan = store.Nodes.Span;

        // Number of nodes after which the parallel presorter will be started
        int presortBegin = (int)(store.Nodes.nextFreeIndex * 0.70f);

        // Now scan all above nodes. 
        // If they don't belong, ignore. 
        // If they do belong, swap them down to the next available lower location
        // Note that this can't be parallelized, since we have to do it strictly in order of node index
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
            Debug.Assert(USE_FAST_TREE_REBUILD || !thisNode.IsTranspositionLinked);
            Debug.Assert(USE_FAST_TREE_REBUILD || thisNode.NumNodesTranspositionExtracted == 0);

            // Remember this location if this is the new parent
            if (i == newRootChildIndex)
            {
              newIndexOfNewParent = nextAvailableNodeIndex;
            }

            // TODO: could we omit putting in the nodes which are transposition linked here?
            //       This would reduce size of structure we need to sort and process.
            //       But will this throw off alignment with numRewrittenNodesDone ?
            if (!thisNode.IsTranspositionLinked)
            {
              childrenToNodes[numChildrenFixups++] = new ChildStartIndexToNodeIndex(thisNode.childStartBlockIndex, nextAvailableNodeIndex, thisNode.NumPolicyMoves);
            }

            if (thisNode.IsTranspositionLinked)
            {
              // This is a transposition linked node for which the 
              // transposition root node will remain in the retained tree.
              // Just point it back to the new index of the root node.
              // Note that this is safe because the trees are always built and rebuilt (for make new root)
              // retaining chronological order of visits, so the roots will always appear
              // sequentially before linked nodes that refer to them.
              Debug.Assert(USE_FAST_TREE_REBUILD);
              bool found = transpositionRoots.TryGetValue(thisNode.ZobristHash, out int transpositionRootIndex);
              if (!found)
              {
                throw new Exception("Internal error: expected transposition root not found.");
              }
              else
              {
                Debug.Assert(rawNodes[transpositionRootIndex].ZobristHash == thisNode.ZobristHash);
                thisNode.TranspositionRootIndex = transpositionRootIndex;
              }
            }
            else
            {
              // Re-insert this into the transpositionRoots (with the updated node index)
              // TODO: this is expensive, try to optimize by
              //       possibly avoiding duplicates or parallelizing
              transpositionRoots?.TryAdd(thisNode.ZobristHash, nextAvailableNodeIndex);
            }

            // Move the actual node
            SwapNodesPosition(store, nodesSpan, new MCTSNodeStructIndex(i), new MCTSNodeStructIndex(nextAvailableNodeIndex));

            nextAvailableNodeIndex++;

            const bool ENABLE_PRESORTER = false; // TODO: enable after more correctness testing
            if (ENABLE_PRESORTER && presorter == null && i > presortBegin)
            {
              // We are mostly done traversing the entires and have accumulate a large number of ChildStartIndexToNodeIndex.
              // Later we'll need to sort all of these accumulated so far (and subsequently).
              // But at this point we can start a paralell thread to sort the entries created so far,
              // which significantly speeds up the full sort of all items which will happen subsequently.
              presorter = new Task(() =>
              {
                var childrenToNodesSoFar = new Span<ChildStartIndexToNodeIndex>(childrenToNodes).Slice(0, numChildrenFixups);
                childrenToNodesSoFar.Sort();
              });
              presorter.Start();
            }
          }
        }

        presorter?.Wait();
        return numChildrenFixups;
      }


      // Rewrite nodes (and associated children)
      int numChildrenFixups = RewriteNodesBuildChildrenInfo();

      // Perform a sort so we can shift down the children in order of apperance
      // This is necessary because it guarantess we will always have sufficient room
      var usedChildStartIndexInfo = new Span<ChildStartIndexToNodeIndex>(childrenToNodes).Slice(0, numChildrenFixups);
      usedChildStartIndexInfo.Sort();
      RewriteChildren(store, usedChildStartIndexInfo);

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


    private static void RewriteChildren(MCTSNodeStore store, Span<ChildStartIndexToNodeIndex> childrenToNodes)
    {
      int nextAvailableChildBlockIndex = 1;

      // Loop thru all the new nodes to process (already sorted by child start index)
      for (int i = 0; i < childrenToNodes.Length; i++)
      {
        ChildStartIndexToNodeIndex childStartToNode = childrenToNodes[i];

        // Skip entries with negative index (which are transposition points)
        bool isTranspositionLinked = childStartToNode.PriorChildStartBlockIndex < 0;
        if (isTranspositionLinked)
        {
          // Nothing to do.
        }
        else if (childStartToNode.PriorChildStartBlockIndex > 0)
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
      // so no need to zero anything here.
      store.Children.nextFreeBlockIndex = nextAvailableChildBlockIndex;
    }


    static void ModifyChildrensParentRef(MCTSNodeStore store, Span<MCTSNodeStruct> rawNodes, 
                                         ref MCTSNodeStruct node, MCTSNodeStructIndex newParentIndex)
    {
      if (!node.IsTranspositionLinked)
      {
        Span<MCTSNodeStructChild> children = node.ChildrenFromStore(store);
        int numChildrenExpanded = node.NumChildrenExpanded;
        for (int i = 0; i < numChildrenExpanded; i++)
        {
          rawNodes[children[i].ChildIndex.Index].ParentIndex = newParentIndex;
        }
      }
    }


    /// <summary>
    /// Swaps the position of two nodes within the store,
    /// updating linked data structures as needed.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="nodes"></param>
    /// <param name="from"></param>
    /// <param name="to"></param>
    public static void SwapNodesPosition(MCTSNodeStore store, Span<MCTSNodeStruct> nodes, 
                                         MCTSNodeStructIndex from, MCTSNodeStructIndex to)
    {
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
