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
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Search.Params;


#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Raw low-level structure used to hold core MCTS tree node data.
/// Note that size exactly 64 bytes to align with cache lines.
/// 
/// N.B. because the struct is marked as readonly, it is important to mark as many
///      properties/methods as readonly as possible to avoid compiler making defensive copies:
///        - improves efficiency
///        - insures correctness so that the index of a node can be determined by its address
///          (insure it remains in the MCTSNodeStore)
/// 
/// Notes on children layout:
///   - child arrays (expanded or unexpanded) are always sorted descending by P
///     (with ties believed impossible due to code run at policy initialization which enforces this)
///   - NumChildrenExpanded tracks how many moves have been expanded to be their own nodes 
///   - NumChildrenVisited field tracks how many children have actually been visited (N > 0)
///   - typically NumChildrenExpanded and NumChildrenVisited have the same value, since we generally 
///     the MCTS algorithm will choose new children to visit strictly in order by P
///   - however NumChildrenExpanded and NumChildrenVisited could rarely differ. For example,
///     if we are enforcing limits on NN batch size in leaf selection and a selected node  
///     ends up being dropped because it exceeded the limit and another child with a higher index
///     was also selected in the same batch, the dropped node will have been expanded but not visited.
///      
/// </summary>
public partial struct GNodeStruct
{
  #region Data

  /// <summary>
  /// Neural network evaluation of win - loss.
  /// </summary>
  public readonly float V => WinP - LossP;

  /// <summary>
  /// Returns if this node has been evaluated.
  /// </summary>
  public readonly bool IsEvaluated => !FP16.IsNaN(WinP);

  /// <summary>
  /// Draw probability
  /// </summary>
  public readonly float DrawP => ParamsSelect.VIsForcedResult(V) ? 0 : (1.0f - ((float)WinP + (float)LossP));


  public readonly float QUpdatesWtdAvg { get => throw new NotImplementedException(); set { } }

  public readonly float QUpdatesWtdVariance { get => throw new NotImplementedException(); set { } }

  //    public float VVariance => W == 0 ? 0 : (VSumSquares - (float)W) / N;


  internal readonly float VSumSquares { get => float.NaN; set { } }

  /// <summary>
  /// Variance of all V values backed up from subtree
  /// </summary>
  public readonly float VVariance => throw new NotImplementedException();// W == 0 ? 0 : (VSumSquares - (float)W) / N;

  #endregion


  /// <summary>
  /// Sets Q to be a NaN (suppressing typical Debug assertion which prevents this).
  /// </summary>
  internal void SetQNaN() => q = double.NaN;
  

  /// <summary>
  /// Returns reference to child node at specified index within children.
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public ref readonly GNodeStruct ChildAtIndexRef(int childIndex)
  {
    throw new NotImplementedException();

    //      Debug.Assert(childIndex < NumPolicyMoves);
    //      int childNodeIndex = MoveInfos[childIndex].ChildNodeIndex.Index;
    //      return ref Context.Store.Nodes.nodes[childNodeIndex];
  }


  /// <summary>
  /// Returns child information at specified index.
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public readonly GEdgeHeaderStruct ChildAtIndex(int childIndex)
  {
    throw new NotImplementedException();
    //      Debug.Assert(childIndex < NumPolicyMoves);
    //      return Context.Store.MoveInfos.childIndices[ChildInfo.ChildInfoStartIndex(Context.Store) + childIndex];
  }


  /// <summary>
  /// Returns index of specified node within the store.
  /// 
  /// TODO: someday we want to mark this as readonly (which is critical for readonly refs)
  ///       However seemingly no efficient way to do that (using Unsafe class). See CTNodeStorage.NodeOffsetFromFirst.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public unsafe readonly NodeIndex IndexInStore(GraphStore store)
  {
    // This relies on fact that the array of nodes resides at a fixed address
    // which will be assured if either:
    //   - the object is on the large object heap, and compaction is not enabled for this heap, or
    //   - we are using nonmanaged memory
    int thisIndex = (int)((long)store.NodesStore.NodeOffsetFromFirst(in this) / MCGSNodeStructSizeBytes);
    Debug.Assert(thisIndex <= store.NodesStore.NumUsedNodes);
    return new NodeIndex(thisIndex);
  }


  /// <summary>
  /// Returns index of node within node array.
  /// </summary>
  public readonly NodeIndex Index
  {
    get => throw new NotSupportedException("GNodeStruct.Index requires a store reference. Use IndexInStore(store) instead.");    
  }


  // Choice of divisor of 64 is very important for performance, for two reasons:
  //   - cache line is generally 64 bytes, thus this allows us to allocate in an array 
  //     such that each is aligned with a cache line and exactly one memory access per node, and
  //   - because power of 2, indexing into the array of nodes can be done using shifting instead of multiplication
  // By using a const here, compiler recognize opportunity to use shifting 
  internal const int MCGSNodeStructSizeBytes = 64;


  /// <summary>
  /// Perform various integrity checks on structure layout (debug mode only).
  /// </summary>
  [Conditional("DEBUG")]
  internal static void ValidateMCGSNodeStruct()
  {
    // Verify expected size
    long size = Marshal.SizeOf<GNodeStruct>();
    if (size != MCGSNodeStructSizeBytes)
    {
      throw new Exception("Internal error, wrong size MCTSNodeStruct " + size);
    }

    // We rely upon immovability of the array of nodes 
    //Debug.Assert(System.Runtime.GCSettings.LargeObjectHeapCompactionMode == GCLargeObjectHeapCompactionMode.Default);

    Debug.Assert(Marshal.SizeOf<NodeIndex>() == 4);
  }
}
