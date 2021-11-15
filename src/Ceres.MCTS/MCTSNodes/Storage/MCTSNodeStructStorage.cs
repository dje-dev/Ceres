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
using System.Threading;

using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCTS.MTCSNodes.Struct;


#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  /// <summary>
  /// Manages raw storage of tree nodes in a single contighous
  /// array of structures. 
  /// 
  /// This approach has mostly benefits over using .NET objects for each node, with advantages:
  ///   1) eliminates allocation/garbage collector overhead since no managed objects are created
  ///   2) possibility of cache-line aligning (64 bytes) nodes for greater memory access efficiency
  ///   3) enhances memory locality, with the possibility of dynamic node reordering to further enhance
  ///      (though this is probably not helpful if the nodes are already cache-aligned)
  ///   4) enables possibility of using operating system large pages for greater access efficiency
  ///   5) the entire tree can be trivially serialized/deserialized by writing contiguous block of memory,
  ///      which could be useful for saving state of game analysis for later use/sharing, or 
  ///      transporting subtrees to distributed computers on the network,
  ///      or distributing tree across a memory hierarchy with different performance characteristics 
  ///      (e.g. DRAM vs. Intel Optane)
  ///   6) "pointers" from one node occupy less memory (4 bytes instead of 8),
  ///      saving perhaps 8 (2 * 4) bytes per node, no average
  /// but disadvantages:
  ///   1) the code is somewhat more complex (using refs) and possibly error prone
  ///   2) there is some overhead with using array indexing instead of direct memory pointers
  ///   3) changing the root of the tree and releasing unused nodes is no longer 
  ///      as trivial as just changing the root pointer (the tree must be rewritten).
  public partial class MCTSNodeStructStorage
  {
    /// <summary>
    /// The set of prior moves which were speciifed as preceeding
    /// the root node of this store.
    /// </summary>
    public PositionWithHistory PriorMoves { get; internal set; }

    /// <summary>
    /// Maximum number of nodes which this store is configured to hold.
    /// </summary>
    public readonly int MaxNodes;

    /// <summary>
#if SPAN
    public const bool SPAN = true;

    public MemoryBufferOS<MCTSNodeStruct> nodes;

    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    public Span<MCTSNodeStruct> Span => nodes.Span;
#else
    public const bool SPAN = false;
    public MCTSNodeStruct[] nodes;
    public Span<MCTSNodeStruct> Span => nodes.AsSpan();
#endif

    internal const int FIRST_ALLOCATED_INDEX = 1;

    readonly object lockObj = new();

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
    /// Number of nodes which have been marked as belonging to old generation of tree,
    /// no longer used or reachable from current root.
    /// </summary>
    public int NumOldGeneration;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="numNodes"></param>
    /// <param name="priorMoves"></param>
    /// <param name="useIncrementalAlloc"></param>
    /// <param name="largePages"></param>
    /// <param name="useExistingSharedMem"></param>
    public MCTSNodeStructStorage(int numNodes, PositionWithHistory priorMoves, bool useIncrementalAlloc,
                                 bool largePages, bool useExistingSharedMem)
    {
      MaxNodes = numNodes;

#if SPAN
      string memorySegmentName = useExistingSharedMem ? "CeresSharedNodes" : null;

      nodes = new MemoryBufferOS<MCTSNodeStruct>(numNodes + BUFFER_NODES, largePages, memorySegmentName,
                                                 useExistingSharedMem,
                                                 useIncrementalAlloc);
#else
    throw new NotImplementedException("Currently only SPAN mode is supported because otherwise GC may relocate nodes violating assumption");
      const bool ALIGN_64 = false;// not working for some reason
      nodes = AllocateCache(numNodes, ALIGN_64);
#endif

      Reset(priorMoves, false);
    }


    /// <summary>
    /// Releases the node store.
    /// </summary>
    public void Deallocate()
    {
#if SPAN
      nodes.Dispose();
#endif
      nodes = null;
    }

    /// <summary>
    /// Allocates and returns the next available node index.
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MCTSNodeStructIndex AllocateNext()
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

      return new MCTSNodeStructIndex(gotIndex);
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
    /// Overallocate sufficiently to make sure allocation reaches to end of a (possibly huge) page
    /// </summary>
    static int BUFFER_NODES => (2048 * 1024) / MCTSNodeStruct.MCTSNodeStructSizeBytes;

    public void InsureAllocated(int numNodes) => nodes.InsureAllocated(numNodes);


    /// <summary>
    /// Returns the index of a node given a reference to it.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    unsafe internal long NodeOffsetFromFirst(in MCTSNodeStruct node)
    {
      // Back out the index based on the virtual address relative to 
      // the starting virtual address of the node store (given the fixed node size).
      long offset = (long)(IntPtr)Unsafe.AsPointer(ref Unsafe.AsRef<MCTSNodeStruct>(in node));
      long firstOffset = (long)(IntPtr)Unsafe.AsPointer(ref nodes[0]);
      return offset - firstOffset;
    }


    /// <summary>
    /// Returns if a specified node is a member of this node store.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public bool BelongsToStorage(ref MCTSNodeStruct node)
    {
      long offset = NodeOffsetFromFirst(in node);
      return (offset > 0 && offset < MaxNodes * MCTSNodeStruct.MCTSNodeStructSizeBytes);
    }


    /// <summary>
    /// Verifies a given node reference points to a node within this store,
    /// otherwise throws an Exception.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public unsafe bool VerifyBelongsToStorage(ref MCTSNodeStruct node)
    {
      if (!BelongsToStorage(ref node))
      {
        throw new Exception("Invalid node. Should be created within storage index " + node.ToString());
      }

      return true;
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
      if (priorMoves == null) priorMoves = new PositionWithHistory(MGPosition.FromPosition(Position.StartPosition));

      if (clearMemory)
      {
        // Clear underlying memory in store
        // Note that MCTSNodeStructChildStorage does not need to be cleared, 
        // since we always fully fill in any newly allocated child array fields
#if SPAN
        nodes.Clear(0, nextFreeIndex);
#else
      Array.Clear(nodes, 0, nextFreeIndex);
#endif
      }

      nextFreeIndex = 1;

      PriorMoves = priorMoves;

      // Allocate
      MCTSNodeStructIndex rootNodeIndex = AllocateNext();

      // Initialize fields
      nodes[rootNodeIndex.Index].Initialize(default, 0, (FP16)1.0, default);

      nodes[rootNodeIndex.Index].NumPieces = priorMoves.FinalPosition.PieceCount;
      nodes[rootNodeIndex.Index].NumRank2Pawns = priorMoves.FinalPosMG.NumPawnsRank2;
    }



    /// <summary>
    /// Returns a string summary of the object.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<MCTSNodeStructStorage MaxNodes={MaxNodes} UsedNodes={NumUsedNodes} Rooted at {PriorMoves}>";
    }


    /// <summary>
    /// Dumps the contents node store to the Console for diagnostic purposes.
    /// </summary>
    public void Dump()
    {
      Console.WriteLine("\r\nNODE STORAGE ");
      for (int i = 0; i < nextFreeIndex; i++)
      {
        string childStr = "";
        foreach (MCTSNodeStructChild child in nodes[i].Children)
        {
          if (child.IsExpanded)
          {
            childStr += $"[child={child.ChildIndex.Index} parent={child.ChildRef.ParentIndex.Index}] ";
          }
        }

        Console.WriteLine($" {i,9} {nodes[i]} children-> {childStr}");
      }
    }


  }
}
