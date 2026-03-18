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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Manages raw storage of graph nodes in a single contiguous array of structures. 
/// 
/// This approach has mostly benefits over using .NET objects for each node, with advantages:
///   1) eliminates allocation/garbage collector overhead since no managed objects are created
///   2) possibility of cache-line aligning (64 bytes) nodes for greater memory access efficiency
///   3) enhances memory locality, with the possibility of dynamic node reordering to further enhance
///      (though this is probably not helpful if the nodes are already cache-aligned)
///   4) enables possibility of using operating system large pages for greater access efficiency
///   5) the entire graph could be trivially serialized/deserialized by writing contiguous block of memory,
///      which could be useful for saving state of game analysis for later use/sharing, or 
///      transporting subgraphs to distributed computers on the network,
///      or distributing graph across a memory hierarchy with different performance characteristics 
///      (e.g. DRAM vs. Intel Optane)
///   6) "pointers" from one node occupy less memory (4 bytes instead of 8),
///      saving perhaps 8 (2 * 4) bytes per node, on average. This is critical because about 2/3
///      of the total memory consumed is used by the child/parent linkages.
/// but disadvantages:
///   1) the code is somewhat more complex (using awkward ref structs or unsafe pointers) and possibly error prone
///   2) there is some overhead with using array indexing instead of direct memory pointers
///   3) changing the root of the graph and releasing unused nodes is no longer 
///      as trivial as just changing the root pointer (the graph somes must be repacked with full rewrite).
public partial class GNodeStore
{
  /// <summary>
  /// The set of prior moves which were speciifed as preceeding
  /// the root node of this store.
  /// TODO: consider moving this up a level to the GraphStore.
  /// </summary>
  public PositionWithHistory PositionHistory { get; internal set; }

  /// <summary>
  /// Maximum number of nodes which this store is configured to hold.
  /// </summary>
  public int MaxNodes { init; get; }

  /// <summary>
  /// If the network has action values that should be stored.
  /// </summary>
  public bool HasState { init; get; }


  /// <summary>
  /// All nodes stored in a single resizable memory array located at a fixed address.
  /// </summary>
  public MemoryBufferOS<GNodeStruct> nodes;


  [DebuggerBrowsable(DebuggerBrowsableState.Never)]
  public Span<GNodeStruct> Span => nodes.Span;

  internal const int FIRST_ALLOCATED_INDEX = 1;

  readonly Lock lockObj = new();


  /// <summary>
  /// Returns the numbrer of nodes allocated so far.
  /// </summary>
  public int NumUsedNodes => nextFreeIndex - FIRST_ALLOCATED_INDEX;

  /// <summary>
  /// Returns the number of nodes in use (all allocated nodes plus one unused root node at beginning).
  /// </summary>
  public int NumTotalNodes => nextFreeIndex; // includes reserved null entry at 0

  /// <summary>
  /// The index indicating the next free node slot.
  /// </summary>
  internal int nextFreeIndex = FIRST_ALLOCATED_INDEX; // Index 0 reserved, indicates null node

  /// <summary>
  /// Parent store to which this nodes store belongs.
  /// </summary>
  public readonly GraphStore ParentStore;


  public Half[][] AllStates; // TODO: make this more efficient


  /// <summary>
  /// Address of the first node in the store.
  /// This is guaranteed to be a fixed address, so we can use it to calculate the offset of any node.
  /// </summary>
  readonly long addressNodeZero;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="parentStore"></param>
  /// <param name="numNodes"></param>
  /// <param name="hasState"></param>
  /// <param name="priorMoves"></param>
  /// <param name="useIncrementalAlloc"></param>
  /// <param name="largePages"></param>
  /// <param name="useExistingSharedMem"></param>
  public unsafe GNodeStore(GraphStore parentStore, int numNodes, bool hasState,
                          PositionWithHistory priorMoves, bool useIncrementalAlloc,
                          bool largePages, bool useExistingSharedMem)
  {
    ParentStore = parentStore;
    MaxNodes = numNodes;
    HasState = hasState;

    parentStore.DebugLogInfo($"GNodeStore with {numNodes} nodes, large pages: {largePages}, shared memory: {useExistingSharedMem}");

    string memorySegmentName = useExistingSharedMem ? "CeresSharedNodes" : null;

    nodes = new MemoryBufferOS<GNodeStruct>(numNodes + BUFFER_NODES, largePages, memorySegmentName, useExistingSharedMem, useIncrementalAlloc);

    Reset(priorMoves, false);

    addressNodeZero = (long)(IntPtr)Unsafe.AsPointer(ref nodes[0]);
  }


  /// <summary>
  /// Underlying memory buffer.
  /// </summary>
  public MemoryBufferOS<GNodeStruct> MemoryBufferOSStore => nodes;


  /// <summary>
  /// Releases the node store.
  /// </summary>
  public void Deallocate() => nodes.Dispose();


  /// <summary>
  /// Allocates and returns the next available node index.
  /// </summary>
  /// <returns></returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public NodeIndex AllocateNext()
  {
    // Take next available (lock-free)
    int gotIndex = Interlocked.Increment(ref nextFreeIndex) - 1;

    // Check for overflow (with page buffer)
    if (nodes.NumItemsAllocated <= gotIndex + BUFFER_NODES)
    {
      lock (lockObj)
      {
        nodes.InsureAllocated(gotIndex + BUFFER_NODES);
      }
    }

    return new NodeIndex(gotIndex);
  }


  /// <summary>
  /// Resizes memory store to exactly fit current used space.
  /// </summary>
  public void ResizeToCurrent() => ResizeToNumNodes(NumTotalNodes);


  /// <summary>
  /// Resizes underlying memory block to commit only specified number of items.
  /// </summary>
  /// <param name="numNodes"></param>
  /// <exception cref="Exception"></exception>
  void ResizeToNumNodes(int numNodes)
  {
    if (numNodes < NumTotalNodes)
    {
      throw new ArgumentException("Attempt to resize MCTSNodeStructStorage to size smaller than current number of used nodes.");
    }
    else if (numNodes > nodes.NumItemsAllocated)
    {
      throw new ArgumentException("Attempt to resize MCTSNodeStructStorage to size larger than current.");
    }

    nodes.ResizeToNumItems(numNodes);
  }


  /// <summary>
  /// Overallocate sufficiently to make sure allocation reaches to end of a (possibly large) page
  /// </summary>
  static int BUFFER_NODES => (2048 * 1024) / GNodeStruct.MCGSNodeStructSizeBytes;


  /// <summary>
  /// Returns the index of a node given its address.
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  [DebuggerStepThrough]
  unsafe internal nuint IndexOfNodeAtAddress(GNodeStruct* nodePtr)
    => ((nuint)nodePtr - (nuint)addressNodeZero) / (nuint)sizeof(GNodeStruct);



  /// <summary>
  /// Returns the index of a node given a reference to it.
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  unsafe internal long NodeOffsetFromFirst(in GNodeStruct node)
  {
    // Back out the index based on the virtual address relative to 
    // the starting virtual address of the node store (given the fixed node size).
    nuint nodePtr = (nuint)Unsafe.AsPointer(ref Unsafe.AsRef<GNodeStruct>(in node));
    nuint basePtr = (nuint)Unsafe.AsPointer(ref nodes[0]);
    return (long)(nodePtr - basePtr);
  }


  /// <summary>
  /// Resets the state of the node store back to empty, 
  /// possibly also clearing the state of all nodes.
  /// </summary>
  /// <param name="priorMoves"></param>
  /// <param name="clearMemory"></param>
  public void Reset(PositionWithHistory priorMoves, bool clearMemory = true)
  {
    // If starting position not specified, assume starting position
    PositionHistory = priorMoves ?? new(Position.StartPosition);

    if (clearMemory)
    {
      // Clear underlying memory in store
      // Note that MCTSNodeStructChildStorage does not need to be cleared, 
      // since we always fully fill in any newly allocated child array fields
      nodes.Clear(0, nextFreeIndex);
    }

    if (HasState)
    {
      // TODO: make appropriate sized and resizable!
      AllStates = new Half[1_000_000][];
    }

    nextFreeIndex = 1;

    // Cause the root node to be allocated.
    AllocateNext();
  }


  /// <summary>
  /// Sets the prior moves and updates related state.
  /// </summary>
  /// <param name="priorMoves"></param>
  internal void SetPriorMoves(PositionWithHistory priorMoves) => PositionHistory = priorMoves;


  /// <summary>
  /// Returns a string summary of the object.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<MCTSNodeStructStorage MaxNodes={MaxNodes} UsedNodes={NumUsedNodes} Rooted at {PositionHistory}>";
  }  
}
