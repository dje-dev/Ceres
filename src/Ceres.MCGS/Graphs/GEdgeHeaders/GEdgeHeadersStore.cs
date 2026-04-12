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
using System.Threading;
using Ceres.Base.OperatingSystem;
using Ceres.MCGS.Graphs.GraphStores;


#endregion

namespace Ceres.MCGS.Graphs.GEdgeHeaders;

[Serializable]
public class GEdgeHeadersStore : MemoryBufferOSBlocked<GEdgeHeaderStruct>
{
  /// <summary>
  /// Only about 2.1 billion values are available for the 
  /// child start indices stored in MCGSNodeStruct (edgeHeaderBlockIndexOrDeferredNode)
  /// (since they are ints, with negative values reserved).
  /// 
  /// However with an average of about 32 children per node,
  /// (considering that some will be unexpanded transposition nodes)
  /// this would severely limit the maximum search graph size.
  /// 
  /// Therefore we allocate children in blocks and the values 
  /// we record are indexes of the block, not the index of the child.
  /// 
  /// For example:
  ///    1 per block --> 2.1 billion  / 32_per_node =    65,625,000 nodes
  ///   16 per block --> 33.6 billion / 32_per_node = 1,050,000,000 nodes
  ///   32 per block --> 67.2 billion / 32_per_node = 2,100,000,000 nodes
  ///   
  /// The value of 16 or 32 allows a large search graph and also has the benefit
  /// that the blocks fall on cache lines (64 bytes or 128 bytes), while keeping small
  /// average number of children slots unused.
  /// </summary>
  public const long NUM_EDGE_HEADERS_PER_BLOCK = 32;


  /// <summary>
  /// Maximum number of nodes for which we have room to accommodate
  /// children, as constrained by the data structures.
  /// See comment above.
  /// </summary>
  public const int MAX_NODES = GraphStoreConfig.ENABLE_MAX_SEARCH_GRAPH ? 2_100_000_000 : 1_100_000_000;

  /// <summary>
  /// Extra edge headers for oversizing.
  /// </summary>
  const int BUFFER_EXTRA_EDGE_HEADERS = 64 * 1024;


  /// <summary>
  /// Maximum number of edge headers (including extra padding).
  /// </summary>
  public readonly long MaxEdgeHeaders;


  public GEdgeHeadersStore(GraphStore parentStore, long maxEdgeHeaders, bool tryEnableLargePages)
      : base(maxEdgeHeaders + BUFFER_EXTRA_EDGE_HEADERS,
             (int)NUM_EDGE_HEADERS_PER_BLOCK,
             BUFFER_EXTRA_EDGE_HEADERS,
             tryEnableLargePages,
             null,
             false,
             GraphStoreConfig.STORAGE_USE_INCREMENTAL_ALLOC)
  {
    parentStore.DebugLogInfo($"GEdgeHeadersStore: Allocating {MaxEdgeHeaders} edge headers," 
                           + $"maxItems={maxEdgeHeaders + BUFFER_EXTRA_EDGE_HEADERS}"
                           + $"itemsPerBlock={NUM_EDGE_HEADERS_PER_BLOCK}"
                           + $"bufferExtraItems={BUFFER_EXTRA_EDGE_HEADERS}"
                           + $"tryEnableLargePages={tryEnableLargePages}"); 

    MaxEdgeHeaders = maxEdgeHeaders + BUFFER_EXTRA_EDGE_HEADERS;
  }


  /// <summary>
  /// Underlying memory buffer.
  /// </summary>
  public MemoryBufferOS<GEdgeHeaderStruct> MemoryBufferOSStore => Entries;


  /// <summary>
  /// Returns a span of GMoveInfoStruct covering the edge headers for a node,
  /// starting at the specified absolute index.
  /// This overload accepts a byte count.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
  public Span<GEdgeHeaderStruct> SpanAtIndex(long childStartIndex, byte numPolicyMoves)
    => SpanAtIndex(childStartIndex, (int)numPolicyMoves);
  

  /// <summary>
  /// Returns a span covering move infos for a node starting at the specified block.
  /// This overload accepts a byte count.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
  public Span<GEdgeHeaderStruct> SpanAtBlockIndex(long moveInfosStartBlockIndex, byte numPolicyMoves)
    => SpanAtBlockIndex(moveInfosStartBlockIndex, (int)numPolicyMoves);


  /// <summary>
  /// Resets the next free block index.
  /// </summary>
  internal new int NextFreeBlockIndex
  {
    get => nextFreeBlockIndex;
    set => nextFreeBlockIndex = value;
  }


  /// <summary>
  /// Returns a string representation of this store.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<GEdgeHeadersStore NumAllocatedItems={NumAllocatedItems} UsedNodes~{nextFreeBlockIndex * NUM_EDGE_HEADERS_PER_BLOCK}>";
  }
}
