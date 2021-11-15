#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#define SPAN

#region Using directives

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  [Serializable]
  public class MCTSNodeStructChildStorage
  {
    /// <summary>
    /// Only about 2.1 billion values are available for the 
    /// the child start indices stored in MCTSNodeStruct
    /// (since they are ints, with negative values reserved).
    /// 
    /// However with an average of about 32 children per node,
    /// (considering that some will be unexpanded transposition nodes)
    /// this would severely limit the maximum search tree size.
    /// 
    /// Therefore we allocate children in blocks and the values 
    /// we record are indexes of the block, not the index of the child.
    /// 
    /// For example:
    ///    1 per block --> 2.1 billion  / 32_per_node =    65,625,000 nodes
    ///   16 per block --> 33.6 billion / 32_per_node = 1,050,000,000 nodes
    ///   32 per block --> 67.2 billion / 32_per_node = 2,100,000,000 nodes
    ///   
    /// The value of 16 allows a large search tree and also has the benefit
    /// that the blocks fall on cache lines (64 bytes), while keeping small
    /// average number of chilren slots unused.
    /// </summary>
    public const int NUM_CHILDREN_PER_BLOCK = MCTSParamsFixed.ENABLE_MAX_SEARCH_TREE ? 32 : 16;

    /// <summary>
    /// Maximum number of nodes for which we have room to accomodate
    /// children, as constrained by the data structures.
    /// See comment above.
    /// </summary>
    public const int MAX_NODES = MCTSParamsFixed.ENABLE_MAX_SEARCH_TREE ? 2_100_000_000 : 1_050_000_000;

    /// <summary>
    /// The main store to which these children below.
    /// </summary>
    public readonly MCTSNodeStore ParentStore;

    /// <summary>
    /// Keep track of index of next available block.
    /// </summary>
    internal int nextFreeBlockIndex = 1; // never allocate index 0 (null node)

#if SPAN
    const string SharedMemChildrenName = MCTSParamsFixed.STORAGE_USE_EXISTING_SHARED_MEM
                                          ? "CeresSharedChildren"
                                          : null;

    /// <summary>
    /// Low-level operating system data structure holding children nodes.
    /// </summary>
    internal MemoryBufferOS<MCTSNodeStructChild> childIndices;

    /// <summary>
    /// Copy of reference to associated nodes.
    /// </summary>
    internal MemoryBufferOS<MCTSNodeStruct> nodes;

    public Span<MCTSNodeStructChild> Span => childIndices.Span;

    internal void CopyEntries(int sourceBlockIndex, int destinationBlockIndex, int numChildren)
      =>  childIndices.CopyEntries((long)sourceBlockIndex * (long)NUM_CHILDREN_PER_BLOCK, 
                                   (long)destinationBlockIndex * (long)NUM_CHILDREN_PER_BLOCK, 
                                   numChildren);

#else
    internal MCTSNodeStructChild[] childIndices;
    internal Span<MCTSNodeStructChild> Span => childIndices.AsSpan();
    internal void CopyEntries(int sourceBlockIndex, int destinationBlockIndex, int numChildren)
    {
      Array.Copy(childIndices, 
                 sourceBlockIndex  * NUM_CHILDREN_PER_BLOCK, 
                 childIndices, 
                 destinationBlockIndex * NUM_CHILDREN_PER_BLOCK, 
                 numChildren);
    }

#endif

    /// <summary>
    /// Maximum number of children which this child store is configured to hold.
    /// </summary>
    public readonly long MaxChildren;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentStore"></param>
    /// <param name="maxChildren"></param>
    public MCTSNodeStructChildStorage(MCTSNodeStore parentStore, long maxChildren)
    {
      ParentStore = parentStore;
      nodes = parentStore.Nodes.nodes;
#if SPAN
      MaxChildren = 1 + maxChildren;
      childIndices = new MemoryBufferOS<MCTSNodeStructChild>(MaxChildren,
                                                             MCTSParamsFixed.TryEnableLargePages,
                                                             SharedMemChildrenName, MCTSParamsFixed.STORAGE_USE_EXISTING_SHARED_MEM,
                                                             MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC);
#else
    throw new NotImplementedException("Currently only SPAN mode is supported because otherwise GC may relocate nodes violating assumption");
    childIndices = new MCTSNodeStructChild[1 + numChildren];
#endif
    }

    /// <summary>
    /// Pointer to raw underlying memory at beginning of child store.
    /// </summary>
    public unsafe void* RawMemory => childIndices.RawMemory;


    /// <summary>
    /// Releases storage associated with the store.
    /// </summary>
    public void Deallocate()
    {
#if SPAN
      childIndices.Dispose();
#endif
      childIndices = null;
    }

    /// <summary>
    /// Indexer that returns reference to node at specified child index.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    public ref MCTSNodeStruct this[int childIndex]
    {
      get => ref nodes[childIndex];
    }

    /// <summary>
    /// Number of allocated children blocks.
    /// 
    /// Note that this is not the actual number of children in use,
    /// but rather the number of children slots allocated including the initial null child block 
    /// and potential padding for each set of children.
    /// </summary>
    public long NumAllocatedChildren => (long)NumTotalBlocks * (long)NUM_CHILDREN_PER_BLOCK;

    internal long NumTotalBlocks => nextFreeBlockIndex; // includes reserved null entry at 0

    readonly object lockObj = new ();

    /// <summary>
    /// Insures a specified number of children are allocated.
    /// </summary>
    /// <param name="numChildren"></param>
    internal void InsureAllocated(int numChildren) => childIndices.InsureAllocated(numChildren);



    /// <summary>
    /// Resizes memory store to exactly fit current used space.
    /// </summary>
    public void ResizeToCurrent() => ResizeToNumChildren(nextFreeBlockIndex * MCTSNodeStructChildStorage.NUM_CHILDREN_PER_BLOCK);


    /// <summary>
    /// Resizes underlying memory block to commit only specified number of items.
    /// </summary>
    /// <param name="numChildren"></param>
    /// <exception cref="Exception"></exception>
    void ResizeToNumChildren(int numChildren)
    {
      if (numChildren < nextFreeBlockIndex)
      {
        throw new ArgumentException("Attempt to resize MCTSNodeStructChildStorage to size smaller than current number of used nodes.");
      }
      else if (numChildren > childIndices.NumItemsAllocated)
      {
        throw new ArgumentException("Attempt to resize MCTSNodeStructChildStorage to size larger than current.");
      }

      childIndices.ResizeToNumItems(numChildren);
    }


    /// <summary>
    /// Determins number of blocks needed to hold a specified number of children.
    /// </summary>
    /// <param name="numChildren"></param>
    /// <returns></returns>
    static internal int NumBlocksReservedForNumChildren(int numChildren)
    {
      bool fitsExactly = numChildren % NUM_CHILDREN_PER_BLOCK == 0;

      return (numChildren / NUM_CHILDREN_PER_BLOCK) + (fitsExactly ? 0 : 1);
    }



    /// <summary>
    /// Returns the index of block at which a specified number of new 
    /// children are guaranteed to be allocated.
    /// </summary>
    /// <param name="numPolicyMoves"></param>s
    /// <returns></returns>    
    public long AllocateEntriesStartBlock(int numPolicyMoves)
    {
      int numBlocksRequired = NumBlocksReservedForNumChildren(numPolicyMoves);

      // Take next available (lock-free)
      long newNextFreeBlockIndex = Interlocked.Add(ref nextFreeBlockIndex, numBlocksRequired);

      // Check for overflow (with padding for page effects)
      long newNumEntriesWithPadding = newNextFreeBlockIndex * NUM_CHILDREN_PER_BLOCK + (1024 * 2048 / MCTSNodeStruct.MCTSNodeStructSizeBytes);
      if (newNumEntriesWithPadding >= childIndices.Length)
      {
        throw new Exception($"MCTSNodeStructChildStorage overflow, max size {childIndices.Length}. "
                           + "The number of child pointers (moves per position) exceeded the expected maximum value considered plausible.");
      }

      // Make sure we have allocated to this length
      if (childIndices.NumItemsAllocated <= newNumEntriesWithPadding)
      {
        lock (lockObj)
        {
          childIndices.InsureAllocated(newNumEntriesWithPadding);
        }
      }

      return newNextFreeBlockIndex - numBlocksRequired;
    }


    /// <summary>
    /// Sets the raw data in the child node to specified values.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="childIndex"></param>
    /// <param name="p"></param>
    /// <param name="moveIndex"></param>
    public void SetNodeChildAsPAndMove(in MCTSNodeStruct node, int childIndex, FP16 p, EncodedMove moveIndex)
    {
      Span<MCTSNodeStructChild> overflows = SpanForNode(in node);
      overflows[childIndex].p = p;
      overflows[childIndex].lc0PositionMoveRawValue = (ushort)moveIndex.IndexNeuralNet;
    }


    /// <summary>
    /// Returns a span of MCTSNodeStructChild which covers thie children for 
    /// node with a specified index.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public Span<MCTSNodeStructChild> SpanForNode(MCTSNodeStructIndex nodeIndex)
    {
      ref readonly MCTSNodeStruct nodeRef = ref nodes[nodeIndex.Index];

      if (nodeRef.NumPolicyMoves == 0) return Span<MCTSNodeStructChild>.Empty;

      Debug.Assert(nodeRef.childStartBlockIndex > 0);
#if SPAN
      return childIndices.Slice(nodeRef.ChildStartIndex, nodeRef.NumPolicyMoves);
#else
      return new Span<MCTSNodeStructChild>(childIndices, node.ChildStartIndex, node.NumPolicyMoves);
#endif
    }


    /// <summary>
    /// Returns a span of MCTSNodeStructChild which covers thie children for a specified node.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public Span<MCTSNodeStructChild> SpanForNode(in MCTSNodeStruct node)
    {
      if (node.NumPolicyMoves == 0) return Span<MCTSNodeStructChild>.Empty;

      Debug.Assert(node.childStartBlockIndex > 0);
#if SPAN
      return childIndices.Slice(node.ChildStartIndex, node.NumPolicyMoves);
#else
      return new Span<MCTSNodeStructChild>(childIndices, node.ChildStartIndex, node.NumPolicyMoves);
#endif
    }


    /// <summary>
    /// Returns string summary of object.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<MCTSNodeStructChildStorage NumAllocatedChildren={NumAllocatedChildren} UsedNodes~{nextFreeBlockIndex*NUM_CHILDREN_PER_BLOCK}>";
    }

  }
}
