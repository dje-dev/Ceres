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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Ceres.Base.OperatingSystem;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GEdges;

/// <summary>
/// Raw block which accommodates multiple GEdgeStruct.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
[InlineArray(GEdgeStore.NUM_EDGES_PER_BLOCK)]
public struct GEdgeStructBlocked
{
  private GEdgeStruct _element0;
}


[Serializable]
public class GEdgeStore
{
  /// <summary>
  /// Edge structs are typically are allocated in blocks of more than one.
  /// This has two benefits: 
  ///   - improves memory locality since related edges are stored adjacently, and
  ///   - fewer blocks will be needed to contain edges therefore
  ///     we can fit more edges into a 32 bit index
  ///     (e.g. 2bn blocks -> 6bn edges -> 0.5bn graph nodes (assuming 12 edges per node).
  /// 
  /// Modern microprocessors (Intel, AMD, ARM) use a granularity of 
  /// two adjacent 64-byte cache lines. Hence 4 edges per block (128/32) may be optimal,
  /// at the cost of greater memory consumption due to more unused edge slots.
  /// </summary>
  public const int NUM_EDGES_PER_BLOCK = 4;


  /// <summary>
  /// The main store to which these children below.
  /// </summary>
  public GraphStore ParentStore { init; get; }


  /// <summary>
  /// Keep track of index of next available block.
  /// </summary>
  internal int nextFreeBlockIndex = 1; // never allocate index 0 (null node)

  /// <summary>
  /// Low-level operating system data structure holding children nodes.
  /// </summary>
  internal MemoryBufferOS<GEdgeStructBlocked> edgeStoreMemoryBuffer;

  /// <summary>
  /// Copy of reference to associated nodes.
  /// </summary>
  internal MemoryBufferOS<GNodeStruct> nodes;

  // Added constant for extra items to meet the minimum extra buffer size requirement.
  private const int BUFFER_EXTRA_ITEMS = 16384; // Ensure at least 256 kbytes extra (256*1024 bytes)

  internal void CopyBlockedEntires(int sourceBlockIndex, int destinationBlockIndex, int numBlocks)
      => edgeStoreMemoryBuffer.CopyEntries(sourceBlockIndex, destinationBlockIndex,  numBlocks);
      
     

  /// <summary>
  /// Maximum number of children which this child store is configured to hold.
  /// </summary>
  public readonly long MaxChildren;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="parentStore"></param>
  /// <param name="maxChildren"></param>
  public GEdgeStore(GraphStore parentStore, long maxChildren, bool tryEnableLargePages)
  {
    ParentStore = parentStore;
    nodes = parentStore.NodesStore.nodes;
    MaxChildren = 1 + maxChildren;

    parentStore.DebugLogInfo($"GEdgeStore: Allocating {MaxChildren} edges, tryEnableLargePages={tryEnableLargePages}.");

    edgeStoreMemoryBuffer = new MemoryBufferOS<GEdgeStructBlocked>(
                                MaxChildren / NUM_EDGES_PER_BLOCK, 
                                tryEnableLargePages, null, false, GraphStoreConfig.STORAGE_USE_INCREMENTAL_ALLOC);
  }


  /// <summary>
  /// Underlying memory buffer.
  /// </summary>
  public MemoryBufferOS<GEdgeStructBlocked> MemoryBufferOSStore => edgeStoreMemoryBuffer;


  /// <summary>
  /// Pointer to raw underlying memory at beginning of child store.
  /// </summary>
  public unsafe void* RawMemory => edgeStoreMemoryBuffer.RawMemory;


  /// <summary>
  /// Releases storage associated with the store.
  /// </summary>
  public void Deallocate() => edgeStoreMemoryBuffer.Dispose();


  /// <summary>
  /// Number of allocated edge visit blocks.
  /// 
  /// Note that this is not the actual number of children in use,
  /// but rather the number of children slots allocated including the initial null child block 
  /// and potential padding for each set of children.
  /// </summary>
  public long NumAllocatedEdges => NumAllocatedBlocks * NUM_EDGES_PER_BLOCK;


  /// <summary>
  /// Returns the number of allocated blocks in the store.
  /// </summary>
  internal long NumAllocatedBlocks => nextFreeBlockIndex; // includes reserved null entry at 0


  /// <summary>
  /// Lock object to ensure thread safety when resizing the memory store.
  /// </summary>
  readonly Lock lockObj = new();


  /// <summary>
  /// Resizes memory store to exactly fit current used space.
  /// </summary>
  public void ResizeToCurrent() => ResizeToNumChildren((long)nextFreeBlockIndex * NUM_EDGES_PER_BLOCK);


  /// <summary>
  /// Resizes underlying memory block to commit only specified number of items.
  /// </summary>
  /// <param name="numEdges"></param>
  /// <exception cref="Exception"></exception>
  void ResizeToNumChildren(long numEdges)
  {
    if (numEdges < nextFreeBlockIndex)
    {
      throw new ArgumentException("Attempt to resize GEdgeStore to size smaller than current number of used nodes.");
    }
    else if (numEdges > edgeStoreMemoryBuffer.NumItemsAllocated)
    {
      throw new ArgumentException("Attempt to resize GEdgeStore to size larger than current.");
    }

    edgeStoreMemoryBuffer.ResizeToNumItems(numEdges);
  }


  /// <summary>
  /// Returns the index of newly allocated block.
  /// </summary>
  /// <returns></returns>    
  public long AllocatedNewBlock()
  {
    // Take next available (lock-free)
    long newNextFreeBlockIndex = Interlocked.Add(ref nextFreeBlockIndex, 1);

    // Check for overflow (with padding for page effects)
    long newNumEntries = newNextFreeBlockIndex * NUM_EDGES_PER_BLOCK + 1 + 2048;
    if (newNumEntries >= edgeStoreMemoryBuffer.Length)
    {
      throw new Exception($"GEdgeStore overflow, max size {edgeStoreMemoryBuffer.Length}. ");
    }

    // Thread-safe allocation check and grow
    lock (lockObj)
    {
      if (edgeStoreMemoryBuffer.NumItemsAllocated <= newNumEntries)
      {
        edgeStoreMemoryBuffer.InsureAllocated(newNumEntries);
      }
    }

    return newNextFreeBlockIndex - 1;
  }


  /// <summary>
  /// Returns reference to GEdgeStruct at a specified block index.
  /// </summary>
  /// <param name="blockIndex"></param>
  /// <returns></returns>
  public ref GEdgeStruct this[int blockIndex] => ref edgeStoreMemoryBuffer[blockIndex][0];


  /// <summary>
  /// Returns a span of GEdgeStruct starting at a specified block.
  /// </summary>
  /// <param name="blockIndex"></param>
  /// <returns></returns>
  public Span<GEdgeStruct> SpanAtBlockIndex(int blockIndex)
    => MemoryMarshal.CreateSpan(ref edgeStoreMemoryBuffer[blockIndex][0], NUM_EDGES_PER_BLOCK);



  /// <summary>
  /// Returns string summary of object.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<GEdgeStore NumAllocatedChildren={NumAllocatedEdges} UsedNodes~{nextFreeBlockIndex * NUM_EDGES_PER_BLOCK}>";
  }

}
