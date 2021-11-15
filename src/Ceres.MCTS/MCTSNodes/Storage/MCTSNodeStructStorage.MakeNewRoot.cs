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
using System.Security.Cryptography.Xml;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.MoveGen;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  public unsafe partial class MCTSNodeStructStorage
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
    /// <param name="tree"></param>
    /// <param name="newRootChild"></param>
    /// <param name="newPriorMoves"></param>
    /// <param name="transpositionRoots"></param>
    public static void MakeChildNewRoot(MCTSTree tree,
                                        ref MCTSNodeStruct newRootChild,
                                        PositionWithHistory newPriorMoves,
                                        PositionEvalCache cacheNonRetainedNodes,
                                        TranspositionRootsDict transpositionRoots,
                                        bool tryKeepTranspositionRootsMaxN)
    {
#if DEBUG
      tree.Store.Validate(tree.TranspositionRoots);
#endif

      float policySoftmax = tree.Root.Context.ParamsSelect.PolicySoftmax;

      // Nothing to do if the requested node is already currently the root
      if (newRootChild.Index == tree.Store.RootNode.Index)
      {
        // Nothing changing in the tree, nothing to do.
      }
      else
      {
        DoMakeChildNewRootRewrite(tree, policySoftmax, ref newRootChild, newPriorMoves, cacheNonRetainedNodes, transpositionRoots, tryKeepTranspositionRootsMaxN);

        // Resized committed memory to new size.
        tree.Store.ResizeToCurrent();
      }

#if DEBUG
      tree.Store.Validate(transpositionRoots, true);
#endif
    }

    /// <summary>
    /// Clones values of root node to a newly allocated node at end of store,
    /// updates all children of root to point to new node, 
    /// and returns index of newly created node.
    /// </summary>
    /// <param name="tree"></param>
    /// <returns></returns>
    static MCTSNodeStructIndex CloneRootAtEnd(MCTSTree tree)
    {
      MCTSNodeStore store = tree.Store;

      // Allocate new node
      MCTSNodeStructIndex childNodeIndex = store.Nodes.AllocateNext();
      ref MCTSNodeStruct childNodeRef = ref store.Nodes.nodes[childNodeIndex.Index];

      ModifyChildrensParentRef(store, store.Nodes.nodes.Span, ref store.Nodes.nodes[1], childNodeIndex);

      return childNodeIndex;
    }


    static void MakeNodeNotTranspositionRoot(MCTSTree tree, TranspositionRootsDict transpositionRoots, int indexOfNewRootBeforeRewrite, ref MCTSNodeStruct newRootChild)
    {
      if (!newRootChild.IsRoot && newRootChild.IsTranspositionRoot)
      {
        int nextFreeIndexBeforeMaterialization = tree.Store.Nodes.nextFreeIndex;
        Span<MCTSNodeStruct> nodes = tree.Store.Nodes.nodes.Span;

        for (int ix = 1; ix < nextFreeIndexBeforeMaterialization; ix++)
        {
          ref MCTSNodeStruct thisNode = ref nodes[ix];

          if (thisNode.IsTranspositionLinked
           && thisNode.TranspositionRootIndex == indexOfNewRootBeforeRewrite
           )
          {
            thisNode.MaterializeSubtreeFromTranspositionRoot(tree);
          }
        }

        // Remove from transposition dictionary which disallows
        // from being used subsequently as a transposition root.
        // First remove immediate parent.
        if (transpositionRoots != null)
        {
          bool removed = transpositionRoots.Remove(newRootChild.ZobristHash, newRootChild.Index.Index);
          Debug.Assert(removed);
        }
        newRootChild.IsTranspositionRoot = false;
      }
    }


    /// <summary>
    /// Makes an existing node in the tree the new root node
    /// by simply swapping that node into the root position
    /// and adjusting data structures which refer to these indices.
    /// 
    /// All nodes are retainined in both the tree and transposition table
    /// (though some nodes are marked as OldGeneration and no longer reachable from root).
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="newRootChild"></param>
    /// <param name="newPriorMoves"></param>
    /// <param name="cacheNonRetainedNodes"></param>
    /// <param name="transpositionRoots"></param>
    /// <param name="tryKeepTranspositionRootsMaxN"></param>
    public static void DoMakeChildNewRootSwapRoot(MCTSTree tree, ref MCTSNodeStruct newRootChild,
                                                  PositionWithHistory newPriorMoves,
                                                  PositionEvalCache cacheNonRetainedNodes,
                                                  TranspositionRootsDict transpositionRoots,
                                                  bool tryKeepTranspositionRootsMaxN,
                                                  bool keepCacheItems)
    {
      if (tryKeepTranspositionRootsMaxN)
      {
        throw new NotImplementedException("tryKeepTranspositionRootsMaxN");
      }
#if DEBUG
      tree.Store.Validate(tree.TranspositionRoots, false);
#endif

      DateTime start = DateTime.Now;

      MCTSNodeStore store = tree.Store;
      Span<MCTSNodeStruct> nodes = tree.Store.Nodes.Span;
      int indexOfNewRootBeforeRewrite = newRootChild.Index.Index;

      if (MCTSParamsFixed.NEW_ROOT_SWAP_RETAIN_NODE_CACHE)
      {
        // Remove any possible cached nodes associated with old or new root,
        // since they might be reused and the move may invalidate some of their fields.
        tree.NodeCache.Remove(new MCTSNodeStructIndex(1));
        tree.NodeCache.Remove(newRootChild.Index);
      }

      // Note that the new root is copied to the root position
      // but also left in place at its current position to maintain
      // integrity of the current tree nodes which reference it.
      // We also need to preserve the old root by cloning it to
      // a new node at end of store (and updating children) to preserve full tree integrity.
      MCTSNodeStructIndex rootAtEnd = CloneRootAtEnd(tree);

      // this new node will be marked as old generation below because it is unreachable
      ref MCTSNodeStruct endNode = ref nodes[rootAtEnd.Index];

      if (transpositionRoots != null && endNode.IsTranspositionRoot)
      {
        Debug.Assert(transpositionRoots.Remove(endNode.ZobristHash, 1));
        Debug.Assert(transpositionRoots.TryAdd(endNode.ZobristHash, rootAtEnd.Index, rootAtEnd.Index, nodes));
      }

      // Get array of included nodes, marking others as now belonging to a prior generation.
      uint numNodesUsed;
      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(store, ref newRootChild, true, out numNodesUsed, null, 0, !keepCacheItems);

      int nodesStartCount = tree.Store.Nodes.NumTotalNodes;
      int numNodes = store.Nodes.nextFreeIndex;

      // Swap new root node into place at index 1.
      CopyNodeIntoRootPosition(store, indexOfNewRootBeforeRewrite, newPriorMoves);
      nodes[1].IsTranspositionRoot = false;

      // Update BitArray to reflect this swap.
      includedNodes[indexOfNewRootBeforeRewrite] = false;
      nodes[indexOfNewRootBeforeRewrite].IsOldGeneration = true;
      store.Nodes.NumOldGeneration++;
      includedNodes[1] = true;

      //double elapsed = (DateTime.Now - start).TotalSeconds;
      //Console.WriteLine($"swap time: {elapsed}s  {nodesStartCount} --> {numNodesUsed} try delete: {numTryRemove} succeed: + {numSucceedRemove}");

#if DEBUG
      store.Validate(tree.TranspositionRoots, !keepCacheItems);
#endif
    }



    /// <summary>
    /// Materializes any nodes which are transposition linked 
    /// to nodes not being retained in tree
    /// and returns back the BitArray of nodes to be retrained in tree.
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="newRootChild"></param>
    /// <param name="numNodesUsed"></param>
    /// <returns></returns>
    static BitArray IncludedNodesAfterMaterializationOfNonRetainedLinkedNodes(MCTSTree tree, ref MCTSNodeStruct newRootChild, out uint numNodesUsed, bool clearCacheItems)
    {
      numNodesUsed = 0;
      int numNodesBeforeMaterialization = tree.Store.Nodes.NumTotalNodes;

      // Estimate the maximum number of nodes likely be added to the tree by the materialization process.
      // Typically this is just a few percent (or less).
      const float FRACTION_EXTRA_NODES = 0.07f;
      int numExtraPaddingNodesAtEnd = 1 + (int)(newRootChild.N * FRACTION_EXTRA_NODES);

      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(tree.Store, ref newRootChild, false, out numNodesUsed, numExtraPaddingNodesAtEnd, clearCacheItems);

      // In the rare situation where the actual number of extra nodes materialized
      // nodes exceeds our allowance of extra nodes in the BitArray,
      // we have no choice but to recompute and return a completely new BitArray.
      int numNodesAddedDuringMaterialization = MaterializeNodesWithNonRetainedTranspositionRoots(tree, includedNodes, newRootChild.Index.Index);
      if (numNodesAddedDuringMaterialization > numExtraPaddingNodesAtEnd)
      {
        return MCTSNodeStructUtils.BitArrayNodesInSubtree(tree.Store, ref newRootChild, false, out numNodesUsed, numExtraPaddingNodesAtEnd, false);
      }

      if (numNodesAddedDuringMaterialization > 0)
      {
        for (int i = (int)numNodesBeforeMaterialization; i < numNodesBeforeMaterialization + numNodesAddedDuringMaterialization; i++)
        {
          includedNodes[i] = true;
        }
        numNodesUsed += (uint)numNodesAddedDuringMaterialization;
      }

      return includedNodes;
    }


    /// <summary>
    /// Makes an existing node in the tree the new root node
    /// by rewriting (compacting) all remaining nodes.
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="policySoftmax"></param>
    /// <param name="newRootChild"></param>
    /// <param name="newPriorMoves"></param>
    /// <param name="cacheNonRetainedNodes"></param>
    /// <param name="transpositionRoots"></param>
    /// <param name="tryKeepTranspositionRootsMaxN"></param>
    static void DoMakeChildNewRootRewrite(MCTSTree tree, float policySoftmax, ref MCTSNodeStruct newRootChild,
                                          PositionWithHistory newPriorMoves,
                                          PositionEvalCache cacheNonRetainedNodes,
                                          TranspositionRootsDict transpositionRoots,
                                          bool tryKeepTranspositionRootsMaxN)
    {
      //DateTime start = DateTime.Now;
      int nodesStartCount = tree.Store.Nodes.NumTotalNodes;

      MCTSNodeStore store = tree.Store;
      ChildStartIndexToNodeIndex[] childrenToNodes;

      uint numNodesUsed;
      int numNodesBeforeMaterialization = store.Nodes.NumTotalNodes;
      int indexOfNewRootBeforeRewrite = newRootChild.Index.Index;
      int newIndexOfNewParent = -1;

      int nextAvailableNodeIndex = 1;

      // Traverse this subtree, building a bit array of visited nodes
      BitArray includedNodes = IncludedNodesAfterMaterializationOfNonRetainedLinkedNodes(tree, ref newRootChild, out numNodesUsed, true);
      

      // Possibly extract retained nodes into a cache.
      if (cacheNonRetainedNodes != null)
      {
        const bool SUPPORT_CACHE_CONCURRENCY = true; // extraction done in parallel
        float fracNodesToVisits = (float)store.RootNode.N / store.Nodes.NumTotalNodes;
        int estNumCacheNodes = Math.Max(1000, (int)(0.5f * (store.RootNode.N - newRootChild.N) * fracNodesToVisits));

        cacheNonRetainedNodes.InitializeWithSize(SUPPORT_CACHE_CONCURRENCY, (int)estNumCacheNodes);

        float softmax = tree.Root.Context.ParamsSelect.PolicySoftmax;       
        MCTSNodeStorePositionExtractorToCache.ExtractPositionCacheNodes(store, softmax, includedNodes, in newRootChild, 
                                                         MCTSNodeStorePositionExtractorToCache.ExtractMode.ExtractNonRetained, 
                                                         cacheNonRetainedNodes, tree.TranspositionRoots);
      }

      // Constract a table indicating the starting index and length of
      // children associated with the nodes being extracted.
      childrenToNodes = GC.AllocateUninitializedArray<ChildStartIndexToNodeIndex>((int)numNodesUsed);

      // Create an array which will map old node indices to new node indices.
      // This allows the transposition root indices to be replaced with
      // updated values as the table is sequentially traversed.
      // Note that this insures that the transposition linked nodes are always linked
      // back to exactly the same node during the rebuild process
      // (rather than to just any node in the same equivalence class).
      // The exactly linkages is necessary because the linkage to the tranposition root may
      // run more than 1 node deep (up to 3) and is relative to one specific node from which it was originally linked.
      int numNodesRebuilt = 0;
      int[] mapOldIndicesToNewIndices = GC.AllocateUninitializedArray<int>((int)store.Nodes.NumTotalNodes);

// TODO:
// Consider eliminating special treatment of new root node everywhere
// except in main loop below where the targetNewIndex is remapped to 1 instead of nextAvailableNodeIndex
//mapOldIndicesToNewIndices[indexOfNewRootBeforeRewrite] = 1;

      int RewriteNodesBuildChildrenInfo()
      {
        Task presorter = null;
        int numChildrenFixups = 0;

        Span<MCTSNodeStruct> nodesSpan = store.Nodes.Span;

        // Number of nodes after which the parallel presorter will be started
        int presortBegin = (int)(store.Nodes.nextFreeIndex * 0.70f);

        // Now scan all above nodes. 
        // If they don't belong, ignore. 
        // If they do belong, swap them down to the next available lower location
        // Note that this can't be parallelized, since we have to do it strictly in order of node index
        for (int i = 1; i < store.Nodes.nextFreeIndex; i++)
        {
          if (includedNodes.Get(i))
          {
            mapOldIndicesToNewIndices[i] = ++numNodesRebuilt;

            ref MCTSNodeStruct thisNode = ref store.Nodes.nodes[i];

            // Reset any cache entry.
            thisNode.CachedInfoPtr = null;

            // Remember this location if this is the new parent.
            if (i == indexOfNewRootBeforeRewrite)
            {
              newIndexOfNewParent = nextAvailableNodeIndex;
            }

            if (thisNode.IsTranspositionLinked)
            {
              // This is a transposition linked node for which the 
              // transposition root node will remain in the retained tree.
              // Just point it back to the new index of the root node.
              // Note that this is safe because the trees are always built and rebuilt (for make new root)
              // retaining chronological order of visits, so the roots will always appear
              // sequentially before linked nodes that refer to them.
              int newTranspositionRootIndex = mapOldIndicesToNewIndices[thisNode.TranspositionRootIndex];
              Debug.Assert(nodesSpan[newTranspositionRootIndex].ZobristHash == thisNode.ZobristHash);
              thisNode.TranspositionRootIndex = newTranspositionRootIndex;
            }
            else
            {
              if (thisNode.NumPolicyMoves > 0)
              {
                childrenToNodes[numChildrenFixups++] = new ChildStartIndexToNodeIndex(thisNode.childStartBlockIndex, nextAvailableNodeIndex, thisNode.NumPolicyMoves);
              }

              if (tryKeepTranspositionRootsMaxN)
              {
                throw new NotImplementedException();
                //transpositionRoots?.AddOrPossiblyUpdateUsingN(nodesSpan, thisNode.ZobristHash, nextAvailableNodeIndex, nodesSpan[i].N);
              }

              if (thisNode.IsTranspositionRoot 
              && nextAvailableNodeIndex != 1 // never allow root to be transposition linked
              )
              {
                // Re-insert this into the transpositionRoots (with the updated node index)
                // TODO: this is expensive, try to optimize by
                //       possibly avoiding duplicates or parallelizing
                // TODO: possibly avoid re-adding this if already present? This preserves same order.
                transpositionRoots?.TryAdd(thisNode.ZobristHash, nextAvailableNodeIndex, i, nodesSpan);
              }
            }

            // Move the actual node
            MoveNodeDown(store, nodesSpan, new MCTSNodeStructIndex(i), new MCTSNodeStructIndex(nextAvailableNodeIndex));

            nextAvailableNodeIndex++;

            const bool ENABLE_PRESORTER = true;
            if (ENABLE_PRESORTER && presorter == null && i > presortBegin)
            {
              // Capture the number of items to be fixed up by the task.
              int presorterNumChildrenFixups = numChildrenFixups;

              // We are mostly done traversing the entires and have accumulate a large number of ChildStartIndexToNodeIndex.
              // Later we'll need to sort all of these accumulated so far (and subsequently).
              // But at this point we can start a paralell thread to sort the entries created so far,
              // which significantly speeds up the full sort of all items which will happen subsequently.
              presorter = new Task(() =>
              {
                var childrenToNodesSoFar = new Span<ChildStartIndexToNodeIndex>(childrenToNodes).Slice(0, presorterNumChildrenFixups);
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

      SwapNodeIntoRootPosition(store, newIndexOfNewParent, newPriorMoves);

      // Zero out the new unused space between the new top of nodes and the prior top of nodes
      // since MCTSNodeStruct is assumed to be all zeros when allocated
      int numUnusedNodes = store.Nodes.nextFreeIndex - nextAvailableNodeIndex;
      store.Nodes.nodes.Clear(nextAvailableNodeIndex, numUnusedNodes);

      // Mark the new top of the nodes
      store.Nodes.nextFreeIndex = nextAvailableNodeIndex;
      store.Nodes.NumOldGeneration = 0;

      //double elapsed = (DateTime.Now - start).TotalSeconds;
      //Console.WriteLine($"rewrite time: {elapsed}s  {nodesStartCount} --> {numNodesUsed}");
    }

    private static void CopyNodeIntoRootPosition(MCTSNodeStore store, int newIndexOfNewParent, PositionWithHistory newPriorMoves)
    {
      Debug.Assert(newIndexOfNewParent != 1);

      store.Nodes.nodes[1] = store.Nodes.nodes[newIndexOfNewParent];

      // Finally, make this as the root (no parent)
      store.Nodes.nodes[1].ParentIndex = default;
      store.Nodes.nodes[1].PriorMove = default;
      
      ModifyChildrensParentRef(store, store.Nodes.nodes.Span, ref store.Nodes.nodes[1], new MCTSNodeStructIndex(1));

      // Update the prior moves
      store.Nodes.PriorMoves = new PositionWithHistory(newPriorMoves);
    }


    private static void SwapNodeIntoRootPosition(MCTSNodeStore store, int newIndexOfNewParent, PositionWithHistory newPriorMoves)
    {
      // Finally swap this new root in the root position (slot 1)
      if (newIndexOfNewParent != 1)
      {
        SwapNodePositions(store, new MCTSNodeStructIndex(newIndexOfNewParent), new MCTSNodeStructIndex(1));
      }

      // Finally, make this as the root (no parent)
      store.Nodes.nodes[1].ParentIndex = default;
      store.Nodes.nodes[1].PriorMove = default;
      store.Nodes.nodes[1].CachedInfoPtr = null;

      // Update the prior moves
      store.Nodes.PriorMoves = new PositionWithHistory(newPriorMoves);
    }




    /// <summary>
    /// Processes all the entries in the childrenToNodes span,
    /// moving child entries into new locations.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="childrenToNodes"></param>
    private static void RewriteChildren(MCTSNodeStore store, Span<ChildStartIndexToNodeIndex> childrenToNodes)
    {
      int nextAvailableChildBlockIndex = 1;

      // Loop thru all the new nodes to process (already sorted by child start index).
      for (int i = 0; i < childrenToNodes.Length; i++)
      {
        ChildStartIndexToNodeIndex childStartToNode = childrenToNodes[i];

        // Transposition-linked nodes are not intended to be put in this table.
        bool isTranspositionLinked = childStartToNode.PriorChildStartBlockIndex < 0;
        Debug.Assert(!isTranspositionLinked);
        Debug.Assert(childStartToNode.NumPolicyMoves > 0);

        // Move the actual children entries.
        store.Children.CopyEntries(childStartToNode.PriorChildStartBlockIndex, nextAvailableChildBlockIndex, childStartToNode.NumPolicyMoves);

        // Modify the child start index in the node itself.
        store.Nodes.nodes[childStartToNode.NewNodeIndex].childStartBlockIndex = nextAvailableChildBlockIndex;

        // Advance our index of next child.
        nextAvailableChildBlockIndex += MCTSNodeStructChildStorage.NumBlocksReservedForNumChildren(childStartToNode.NumPolicyMoves);
      }

      // Reset children to new length.
      // Note that it is not assumed the unused portion of the array is zeros,
      // so no need to zero anything here.
      store.Children.nextFreeBlockIndex = nextAvailableChildBlockIndex;
    }

#endregion


#region Helpers


    /// <summary>
    /// Sequentially traverses nodes and materializes any nodes which 
    /// have transposition roots which will not be included in the new tree.
    /// (additionally any nodes having root index pointing to new root are materialized).
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="includedNodes"></param>
    /// <param name="newRootChildIndex"></param>
    /// <returns>the number of nodes added during this operation</returns>
    static int MaterializeNodesWithNonRetainedTranspositionRoots(MCTSTree tree, BitArray includedNodes, int newRootChildIndex)
    {
      int numPrematerialized = 0;
      MCTSNodeStore store = tree.Store;
      MemoryBufferOS<MCTSNodeStruct> rawNodes = store.Nodes.nodes;

      // Record the starting number of nodes 
      // (before any nodes are possibly added by this method).
      int startingNumNodes = store.Nodes.NumTotalNodes;

      for (int i = 1; i < startingNumNodes; i++)
      {
        if (includedNodes.Get(i))
        {
          ref MCTSNodeStruct nodeRef = ref rawNodes[i];

          // Materialize any nodes which are transposition linked to a non-retained node
          if (nodeRef.IsTranspositionLinked)
          {
            int transpositionRootIndex = nodeRef.TranspositionRootIndex;
            Debug.Assert(!rawNodes[transpositionRootIndex].IsTranspositionLinked);

            // Determine if the transposition root will remain in the tree.
            bool linkedNodeRetained = includedNodes.Get(transpositionRootIndex);

            // NOTE: Generally it is not expected that a node will link forward to a transposition root,
            //       so this will probably always be false, but for safety prematerialize if this happens.
            bool linkedNodeSequentiallyLater = transpositionRootIndex > i;

            if (!linkedNodeRetained
              || linkedNodeSequentiallyLater
              || transpositionRootIndex == newRootChildIndex // materialize since the index will be changing
              )
            {
              numPrematerialized++;

              // We are not retaining the transposition root, so we must 
              // unlink the node from parent and copy over all
              // children which we have already visited (virtually).
              nodeRef.MaterializeSubtreeFromTranspositionRoot(tree);
            }
          }
        }
      }

      int numNodesAdded = tree.Store.Nodes.NumTotalNodes - startingNumNodes;
      return numNodesAdded;
    }


#if NOT
    public static PositionEvalCache ExtractCacheNodesInSubtree(MCTSTree tree, ref MCTSNodeStruct newRootChild)
    {
      float softmax = tree.Root.Context.ParamsSelect.PolicySoftmax;

      // Traverse this subtree, building a bit array of visited nodes
      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(tree.Store, ref newRootChild, false, out uint numNodesUsed, 0, false);

      long estNumNodes = tree.Store.RootNode.N - numNodesUsed;
      const bool SUPPORT_CACHE_CONCURRENCY = false; // No need for concurrency, is read-only during play.
      PositionEvalCache cache = new PositionEvalCache(SUPPORT_CACHE_CONCURRENCY, (int)estNumNodes);

      ExtractPositionCacheNodes(tree.Store, softmax, includedNodes, in newRootChild, ExtractMode.ExtractRetained, cache, tree.TranspositionRoots);

      return cache;
    }
#endif

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
