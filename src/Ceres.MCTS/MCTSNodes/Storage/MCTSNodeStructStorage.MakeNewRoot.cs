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
using Ceres.MCTS.Evaluators;
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
    public static void MakeChildNewRoot(MCTSTree tree,
                                        ref MCTSNodeStruct newRootChild,
                                        PositionWithHistory newPriorMoves,
                                        PositionEvalCache cacheNonRetainedNodes,
                                        TranspositionRootsDict transpositionRoots,
                                        bool tryKeepTranspositionRootsMaxN)
    {
#if DEBUG
      tree.Store.Validate();
#endif

      float policySoftmax = tree.Root.Context.ParamsSelect.PolicySoftmax;

      // Nothing to do if the requested node is already currently the root
      if (newRootChild.Index == tree.Store.RootNode.Index)
      {
        // Nothing changing in the tree, just flush the cache references
        tree.Store.ClearAllCacheIndices();
      }
      else
      {
        DoMakeChildNewRootRewrite(tree, policySoftmax, ref newRootChild, newPriorMoves, cacheNonRetainedNodes, transpositionRoots, tryKeepTranspositionRootsMaxN);
      }

#if DEBUG
      tree.Store.Validate(true);
#endif
    }


    public static void DoMakeChildNewRootSwapRoot(MCTSTree tree, ref MCTSNodeStruct newRootChild,
                                                  PositionWithHistory newPriorMoves,
                                                  PositionEvalCache cacheNonRetainedNodes,
                                                  TranspositionRootsDict transpositionRoots,
                                                  bool tryKeepTranspositionRootsMaxN)
    {
      MCTSNodeStore store = tree.Store;
      Span<MCTSNodeStruct> nodes = tree.Store.Nodes.Span;
      int newRootChildIndex = newRootChild.Index.Index;
#if DEBUG
      store.Validate(false);
#endif

      // Get array of included nodes, marking others as now belonging to a prior generation.
      uint numNodesUsed;
      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(store, ref newRootChild, true, out numNodesUsed);

      // Materialize any nodes which point to transposition roots that do not survive
      // TODO: potentially this is unnecessary; except for the new root node moving,
      //       everything else remains in place and we could keep these linkages.
      MaterializeNodesWithNonRetainedTranspositionRoots(tree, includedNodes, newRootChildIndex);

      // Swap new root node into place at index 1.
      SwapNodeIntoRootPosition(store, newRootChildIndex, newPriorMoves);

      // Update BitArray to reflect this swap.
      includedNodes[newRootChildIndex] = false;
      includedNodes[1] = true;

      // Traverse included nodes, and for each:
      //   - try to add to transposition dictionary
      //   - zero CacheIndex
      for (int i = 1; i < store.Nodes.nextFreeIndex; i++)
      {
        if (includedNodes.Get(i))
        {
          ref MCTSNodeStruct thisNode = ref nodes[i];

          Debug.Assert(!thisNode.IsOldGeneration);

          thisNode.CacheIndex = 0;

          // No need to reset pending visits, they can carry forward.
          // thisNode.NumVisitsPendingTranspositionRootExtraction = 0;

          if (thisNode.IsTranspositionLinked)
          {
            Debug.Assert(transpositionRoots.TryGetValue(thisNode.ZobristHash, out _));
          }
          else
          {
            // Re-insert this into the transpositionRoots
            // TODO: this is expensive, try to optimize by
            //       possibly avoiding duplicates or parallelizing
            if (tryKeepTranspositionRootsMaxN)
            {
              transpositionRoots?.AddOrPossiblyUpdate(nodes, thisNode.ZobristHash, i, nodes[i].N);
            }
            else
            {
              transpositionRoots?.TryAdd(thisNode.ZobristHash, i);
            }
          }
        }
      }


#if DEBUG
      store.Validate(true);
#endif
    }

    static void DoMakeChildNewRootRewrite(MCTSTree tree, float policySoftmax, ref MCTSNodeStruct newRootChild,
                                          PositionWithHistory newPriorMoves,
                                          PositionEvalCache cacheNonRetainedNodes,
                                          TranspositionRootsDict transpositionRoots,
                                          bool tryKeepTranspositionRootsMaxN)
    {
#if NOT
      Console.WriteLine("\r\nBEFORE REBUILD");
tree.Store.Dump(true);

DoMakeChildNewRootSwapRoot(tree, policySoftmax, ref newRootChild, newPriorMoves, cacheNonRetainedNodes, transpositionRoots);

Console.WriteLine("\r\nAFTER REBUILD");
tree.Store.Dump(true);
return;
#endif

      MCTSNodeStore store = tree.Store;
      ChildStartIndexToNodeIndex[] childrenToNodes;

      uint numNodesUsed;
      int newRootChildIndex = newRootChild.Index.Index;
      int newIndexOfNewParent = -1;

      int nextAvailableNodeIndex = 1;

      // Traverse this subtree, building a bit array of visited nodes
      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(store, ref newRootChild, false, out numNodesUsed);

      if (USE_FAST_TREE_REBUILD)
      {
        MaterializeNodesWithNonRetainedTranspositionRoots(tree, includedNodes, newRootChildIndex);
      }

      // Possibly extract retained nodes into a cache.
      if (cacheNonRetainedNodes != null)
      {
        long estNumNodes = store.RootNode.N - numNodesUsed;
        const bool SUPPORT_CACHE_CONCURRENCY = false; // No need for concurrency, is read-only during play.
        cacheNonRetainedNodes.InitializeWithSize(SUPPORT_CACHE_CONCURRENCY, (int)estNumNodes);
        float softmax = tree.Root.Context.ParamsSelect.PolicySoftmax;
        ExtractPositionCacheNodes(store, softmax, includedNodes, in newRootChild, ExtractMode.ExtractNonRetained, cacheNonRetainedNodes);
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
        for (int i = 1; i < store.Nodes.nextFreeIndex; i++)
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
            Debug.Assert(USE_FAST_TREE_REBUILD || thisNode.NumVisitsPendingTranspositionRootExtraction == 0);

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
                Console.Write("Warning: expected transposition root not found for " + thisNode);

                // Try to fix up node so this is recoverable and we don't crash (albeit at cost of a major distortion).
                thisNode.TranspositionRootIndex = 0;
                thisNode.NumVisitsPendingTranspositionRootExtraction = 0;
                thisNode.Terminal = Chess.GameResult.Draw;
              }
              else
              {
                Debug.Assert(rawNodes[transpositionRootIndex].ZobristHash == thisNode.ZobristHash);
                thisNode.TranspositionRootIndex = transpositionRootIndex;
              }
            }
            else
            {
              if (tryKeepTranspositionRootsMaxN)
              {
                transpositionRoots?.AddOrPossiblyUpdate(nodesSpan, thisNode.ZobristHash,
                                                        nextAvailableNodeIndex, nodesSpan[i].N);
              }
              else
              {
                // Re-insert this into the transpositionRoots (with the updated node index)
                // TODO: this is expensive, try to optimize by
                //       possibly avoiding duplicates or parallelizing
                transpositionRoots?.TryAdd(thisNode.ZobristHash, nextAvailableNodeIndex);
              }
            }

            // Move the actual node
            SwapNodesPosition(store, nodesSpan, new MCTSNodeStructIndex(i), new MCTSNodeStructIndex(nextAvailableNodeIndex));

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
      store.Nodes.nodes[1].CacheIndex = 0;

      // Update the prior moves
      store.Nodes.PriorMoves = new PositionWithHistory(newPriorMoves);
    }


    private enum ExtractMode { ExtractNonRetained, ExtractRetained };

    private static void ExtractPositionCacheNodes(MCTSNodeStore store, float policySoftmax,
                                                  BitArray includedNodes, in MCTSNodeStruct newRoot,
                                                  ExtractMode extractMode,
                                                  PositionEvalCache cacheNodes)
    {
      for (int nodeIndex = 1; nodeIndex < store.Nodes.NumTotalNodes; nodeIndex++)
      {
        ref MCTSNodeStruct nodeRef = ref store.Nodes.nodes[nodeIndex];


        if ((includedNodes[nodeIndex] != (extractMode == ExtractMode.ExtractRetained)) || nodeRef.IsOldGeneration)
        {
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
            cacheNodes.Store(nodeRef.ZobristHash, nodeRef.Terminal, nodeRef.WinP, nodeRef.LossP, nodeRef.MPosition, in policy);
          }
        }
      }
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
    static void MaterializeNodesWithNonRetainedTranspositionRoots(MCTSTree tree, BitArray includedNodes, int newRootChildIndex)
    {
      int numPrematerialized = 0;
      MCTSNodeStore store = tree.Store;
      MemoryBufferOS<MCTSNodeStruct> rawNodes = store.Nodes.nodes;

      for (int i = 1; i < store.Nodes.nextFreeIndex; i++)
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
              // unlink the node from parent (copying over all child information).
              nodeRef.CopyUnexpandedChildrenFromOtherNode(tree, new MCTSNodeStructIndex(transpositionRootIndex), true);
              nodeRef.NumVisitsPendingTranspositionRootExtraction = 0;
            }
          }
        }
      }
    }


    public static PositionEvalCache ExtractCacheNodesInSubtree(MCTSTree tree, ref MCTSNodeStruct newRootChild)
    {
      float softmax = tree.Root.Context.ParamsSelect.PolicySoftmax;

      // Traverse this subtree, building a bit array of visited nodes
      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(tree.Store, ref newRootChild, false, out uint numNodesUsed);

      long estNumNodes = tree.Store.RootNode.N - numNodesUsed;
      const bool SUPPORT_CACHE_CONCURRENCY = false; // No need for concurrency, is read-only during play.
      PositionEvalCache cache = new PositionEvalCache(SUPPORT_CACHE_CONCURRENCY, (int)estNumNodes);

      ExtractPositionCacheNodes(tree.Store, softmax, includedNodes, in newRootChild, ExtractMode.ExtractRetained, cache);

      return cache;
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
