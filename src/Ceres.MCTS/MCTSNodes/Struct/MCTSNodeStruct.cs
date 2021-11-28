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

using Ceres.Base;
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Raw low-level structure used to hold core MCTS tree node data.
  /// Note that size exactly 64 bytes to align with cache lines.
  /// 
  /// N.B. because the struct is marked as readonly, it is important to mark as many
  ///      properties/methods as readonly as possible to avoid compiler making defensive copies:
  ///        - improves efficiency
  ///        - insures correctness so that the index of a node can be determined by by its address
  ///          (insure it remains in the MCTSNodeStore)
  /// 
  /// Notes on children layout:
  ///   - child arrays (expanded or unexpanded) are always sorted desecending by P
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
  public partial struct MCTSNodeStruct
  {
    #region Data

    /// <summary>
    /// Neural network evaluation of win - loss.
    /// </summary>
    public readonly FP16 V => WinP - LossP;

    /// <summary>
    /// Draw probability
    /// </summary>
    public readonly FP16 DrawP => ParamsSelect.VIsForcedResult(V) ? 0 : (FP16)(1.0f - (WinP + LossP));


    /// <summary>
    /// Average M (moves left estimate) for this subtree
    /// </summary>
    public readonly float MAvg => MSUM_ENABLED ? (mSum / N) : float.NaN;

    /// <summary>
    /// The starting index of entries for this node within the child info array
    /// Uninitialized (null) nodes will have an invalid value of zero
    /// Negative values are special indicators that the node is
    /// linked to a transposition root (having index the negative of that value)
    /// </summary>
    internal readonly long ChildStartIndex { get => (long)childStartBlockIndex * (long)MCTSNodeStructChildStorage.NUM_CHILDREN_PER_BLOCK; }

    /// <summary>
    /// Currently not used (and no room in structure for it).
    /// </summary>
    public int NextTranspositionLinked
    {
      get => throw new NotImplementedException();
      set { if (value != 0) throw new NotImplementedException(); }
    }
    //    public short TranspositionNumBorrowed;


    public readonly float QUpdatesWtdAvg { get => throw new NotImplementedException(); set { } }

    public readonly float QUpdatesWtdVariance { get => throw new NotImplementedException(); set { } }

    //    public float VVariance => W == 0 ? 0 : (VSumSquares - (float)W) / N;


    internal readonly float VSumSquares { get => float.NaN; set { } }

    /// <summary>
    /// Variance of all V values backed up from subtree
    /// </summary>
    public readonly float VVariance => W == 0 ? 0 : (VSumSquares - (float)W) / N;

    /// <summary>
    /// Average win probability of subtree
    /// </summary>
    public readonly float WAvg => 1.0f - DAvg - LAvg;

    /// <summary>
    /// Average draw probability of subtree
    /// </summary>
    public readonly float DAvg => dSum / N;

    /// <summary>
    /// Average loss probability of subbtree
    /// </summary>
    public readonly float LAvg => 0.5f * (1.0f - DAvg - (float)Q);

    public byte NumChildrenVisited
    {
      readonly get => numChildrenVisited <= 64 ? numChildrenVisited : (byte)0;
      set => numChildrenVisited = value;
    }


    #endregion

    public readonly bool IsRoot => ParentIndex.IsNull;

    public readonly bool IsNull => Index.IsNull;

    public readonly bool IsEvaluated => !float.IsNaN(V);

    public readonly bool IsInFlight => (NInFlight + NInFlight2) > 0;


    public const int MAX_NUM_VISITS_PENDING_TRANSPOSITION_ROOT_EXTRACTION = 256 - 64 - 1;

    /// <summary>
    /// The number of visits yet to be processed which will have their values taken from the 
    /// the transposition root (or zero if not transposition linked).
    /// This is encoded in the numChildrenVisited.
    /// </summary>
    public int NumVisitsPendingTranspositionRootExtraction
    {
      readonly get => numChildrenVisited <= 64 ? 0 : 256 - numChildrenVisited;
      set
      {
        Debug.Assert(value >= 0 && value <= MAX_NUM_VISITS_PENDING_TRANSPOSITION_ROOT_EXTRACTION);
        numChildrenVisited = (value == 0) ? (byte)0 : (byte)(256 - value);
      }
    }


    /// <summary>
    /// If the tree is truncated at this node and generating position
    /// values via the subtree linked to its tranposition root
    /// </summary>
    public readonly bool IsTranspositionLinked => childStartBlockIndex < 0;

    public int TranspositionRootIndex
    {
      // Note: we make use of ChildStartIndex as a place to store this value
      readonly get
      {
        if (childStartBlockIndex >= 0)
          return 0;
        else
          return -childStartBlockIndex;
      }
      set
      {
        Debug.Assert(value >= 0);
        //Debug.Assert(value != 1);
        childStartBlockIndex = -value;
      }
    }


    /// <summary>
    /// Sets fields to their inital values upon node creation
    /// Note that it is assumed the node is already in the default (zeroed) state.
    /// 
    /// NOTE: Try to keep changes synchronized with MCTSNodeStructFields.ResetExpandedState.
    /// </summary>
    /// <param name="parentIndex"></param>
    /// <param name="indexInParent"></param>
    /// <param name="p"></param>
    /// <param name="priorMove"></param>
    public void Initialize(MCTSNodeStructIndex parentIndex, int indexInParent,  FP16 p, EncodedMove priorMove)
    {
      ParentIndex = parentIndex;
      miscFields.IndexInParent = (byte)indexInParent;
      P = p;
      PriorMove = priorMove;

      WinP = FP16.NaN;
      LossP = FP16.NaN;
      MPosition = 0;
      SecondaryNN = false;
      IsOldGeneration = false;
      ZobristHash = 0;
      //HashCrosscheck = 0;

#if FEATURE_UNCERTAINTY
      Uncertainty = UNCERTAINTY_PRIOR;
#endif
      //Weight = 1.0f;
    }


    /// <summary>
    /// Returns index of child containing a specified move (or null if none).
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public readonly int? ChildIndexWithMove(MGMove move)
    {
      EncodedMove lzMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(move);
      Span<MCTSNodeStructChild> children = this.Children;
      for (int i = 0; i < NumPolicyMoves; i++)
      {
        if (children[i].Move == lzMove)
        {
          return i;
        }
      }

      return null;
    }

    /// <summary>
    /// If a node has been deleted then Parent is set to a special value
    /// </summary>
    const int DETACHED_NODE_NINFLIGHT_MARKER = short.MaxValue;


    /// <summary>
    /// Returns if the node has been removed from the currently active tree
    /// The raw data still exists in the store, but the root of the subtree 
    /// to which this node belongs is detched (unlinked to from rest of tree).
    /// 
    /// We represent this setting the NInFlight to a special market value.
    /// </summary>
    public bool Detached
    {
      readonly get
      {
        return NInFlight == DETACHED_NODE_NINFLIGHT_MARKER;
      }
      internal set
      {
        NInFlight = DETACHED_NODE_NINFLIGHT_MARKER;
      }
    }



    /// <summary>
    /// Average evaluation value of visited children
    /// (from perspective of side to move)
    /// </summary>
    public readonly double Q => W / N;


    /// <summary>
    /// Returns reference to parent of this node.
    /// </summary>
    public readonly ref MCTSNodeStruct ParentRef
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        Debug.Assert(!ParentIndex.IsNull);
        return ref MCTSNodeStoreContext.Nodes[ParentIndex.Index];
      }
    }

    /// <summary>
    /// Returns N of parent node.
    /// </summary>
    public readonly int ParentN => ParentRef.IsNull ? 0 : ParentRef.N;


    /// <summary>
    /// Returns reference to child node at specified index within children.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref readonly MCTSNodeStruct ChildAtIndexRef(int childIndex)
    {
      Debug.Assert(childIndex < NumPolicyMoves);
      return ref Children[childIndex].ChildRef;
    }


    /// <summary>
    /// Returns child information at specified index.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly MCTSNodeStructChild ChildAtIndex(int childIndex)
    {
      Debug.Assert(childIndex < NumPolicyMoves);
      return MCTSNodeStoreContext.Children[ChildStartIndex + childIndex];
    }


    /// <summary>
    /// Returns index of specified node within the store.
    /// 
    /// TODO: someday we want to mark this as readonly (which is critical for readonly refs)
    ///       However seemingly no efficient way to do that (using Unsafe class). See CTNodeStorage.NodeOffsetFromFirst.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe readonly MCTSNodeStructIndex IndexInStore(MCTSNodeStore store)
    {
      // This relies on fact that the array of nodes resides at a fixed address
      // which will be assured if either:
      //   - the object is on the large object heap, and compaction is not enabled for this heap, or
      //   - we are using nonmanaged memory
      int thisIndex = (int)((long)store.Nodes.NodeOffsetFromFirst(in this) / MCTSNodeStructSizeBytes);
      Debug.Assert(thisIndex <= store.Nodes.NumUsedNodes);
      return new MCTSNodeStructIndex(thisIndex);
    }


    /// <summary>
    /// Returns index of node within node array.
    /// 
    /// TODO: someday we want to mark this as readonly (which is critical for readonly refs)
    ///       However seemingly no efficient way to do that (using Unsafe class). See CTNodeStorage.NodeOffsetFromFirst.
    /// </summary>
    public unsafe readonly MCTSNodeStructIndex Index
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return IndexInStore(MCTSNodeStoreContext.Store);
      }
    }

    // Choice of divisor of 64 is very important for performance, for two reasons:
    //   - cache line is generally 64 bytes, thus this allows us to allocate in an array 
    //     such that each is aligned with a cache line and exactly one memory access per node, and
    //   - because power of 2, indexing into the array of nodes can be done using shifting instead of multiplication
    // By using a const here, compiler recognize opportunity to use shifting 
    internal const int MCTSNodeStructSizeBytes = 64;


    /// <summary>
    /// Perform various integrity checks on structure layout (debug mode only).
    /// </summary>
    [Conditional("DEBUG")]
    internal static void ValidateMCTSNodeStruct()
    {
      // Verify expected size
      long size = Marshal.SizeOf<MCTSNodeStruct>();
      if (size != MCTSNodeStructSizeBytes)
      {
        throw new Exception("Internal error, wrong size MCTSNodeStruct " + size);
      }

      // We rely upon immovability of the array of nodes 
      Debug.Assert(GCSettings.LargeObjectHeapCompactionMode == GCLargeObjectHeapCompactionMode.Default);

      Debug.Assert(Marshal.SizeOf<MCTSNodeStructIndex>() == 4);
    }



    public override string ToString()
    {
      string indexStr = $"#{Index.Index}";
      string oldStr = IsOldGeneration ? " OLD" : "";
      bool isWhite = (DepthInTree % 2 == 1) == (MCTSManager.ThreadSearchContext.Tree.Store.Nodes.PriorMoves.FinalPosition.SideToMove == SideType.White);
      return $"<Node [#{indexStr}] {oldStr} Depth{DepthInTree} {Terminal} {(isWhite?PriorMove:PriorMove.Flipped)} ({N},{NInFlight},{NInFlight2})  P={P * 100.0f:F3}% "
            + $"V={V:F3}" + (VSecondary == 0 ? "" : $"VSecondary={VSecondary:F3} ") + $" W={W:F3} "
            + $"MPos={MPosition:F3} MAvg={MAvg:F3} "
           + $"Parent={(ParentIndex.IsNull ? "none" : ParentIndex.Index.ToString())}"
           + (IsTranspositionLinked ? $" TRANSPOSITION LINKED, pending { NumVisitsPendingTranspositionRootExtraction}" : "")
           + $" Score=(?) > with {NumPolicyMoves} policy moves"
           + (SecondaryNN ? " [SECONDARY]" : "");
      //    + $" Score={score,6:F2} > with {NumPolicyMoves} policy moves"; // can't do this until/if we restore IndexWithinParentsChildren or do linear search to find
    }


    /// <summary>
    /// Returns span of children structures belonging to this node.
    /// </summary>
    /// <param name="store"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Span<MCTSNodeStructChild> ChildrenFromStore(MCTSNodeStore store)
    {
      return store.Children.SpanForNode(in this);
    }


    /// <summary>
    /// Returns span of children structures belonging to this node.
    /// </summary>
    public readonly Span<MCTSNodeStructChild> Children
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        if (NumPolicyMoves == 0)
        {
          return new Span<MCTSNodeStructChild>();
        }
        else
        {
          return MCTSNodeStoreContext.Store.Children.SpanForNode(in this);
        }
      }
    }


    /// <summary>
    /// Computes and returns the depth of node in the tree by ascending to root.
    /// </summary>
    public readonly byte DepthInTree
    {
      get
      {
        int count = 0;
        ref readonly MCTSNodeStruct node = ref this;
        while (!node.IsRoot)
        {
          // Don't allow infinite loop (this makes debugging difficult due to debugger displays hanging)
          if (count > 9999)
          {
            return 255;
          }

          count++;
          node = ref node.ParentRef;
        }

        return count > 255 ? (byte)255 : (byte)count;
      }
    }

  }

}
