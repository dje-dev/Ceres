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
using Ceres.Base.OperatingSystem;
using Ceres.Base.Environment;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.Positions;
using Ceres.MCTS.Params;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Iteration;
using System.Diagnostics;
using System.Runtime.CompilerServices;


#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  /// <summary>
  /// A store of MCTS tree nodes (and associated children information) 
  /// representing the state of an Monte Carlo tree search.
  /// </summary>
  public partial class MCTSNodeStore : IDisposable
  {
    /// <summary>
    /// Approximate average total bytes consumed by a node, child pointers, 
    /// and associated data structures (such as possible entry transposition roots table).
    /// </summary>
    const ulong APPROX_BYTES_PER_NODE = 16
                                      + MCTSNodeStruct.MCTSNodeStructSizeBytes
                                      + (4 * TYPICAL_AVG_CHILDREN_PER_NODE);

    /// <summary>
    /// Maximum number of nodes for which which the store could accomodate.
    /// This is bounded by physical RAM and also data structure limitations.
    /// </summary>
    public static int MAX_NODES
    {
      get
      {
        long maxByMemory = (long)HardwareManager.MemorySize / (long)APPROX_BYTES_PER_NODE;
        maxByMemory = Math.Min(int.MaxValue, maxByMemory);
        return Math.Min((int)maxByMemory, MCTSNodeStructChildStorage.MAX_NODES);
      }
    }

    /// <summary>
    /// The maximum number of nodes for which this store was configured.
    /// </summary>
    public readonly int MaxNodes;

    /// <summary>
    /// Underlying storage for tree nodes.
    /// </summary>
    public MCTSNodeStructStorage Nodes;

    /// <summary>
    /// Underlying storage for children information related to each node.
    /// </summary>
    public MCTSNodeStructChildStorage Children;

    /// <summary>
    /// Returns the index of the root node.
    /// In the current implementation this is always 1.
    /// </summary>
    public MCTSNodeStructIndex RootIndex => new MCTSNodeStructIndex(1);

    public const int MAX_AVG_CHILDREN_PER_NODE = 55; // conservative value unlikely to be exceeded
    public const int TYPICAL_AVG_CHILDREN_PER_NODE = 40; // typical value

    /// <summary>
    /// Returns reference to the root node.
    /// </summary>
    public ref MCTSNodeStruct RootNode => ref Nodes.nodes[RootIndex.Index];

    /// <summary>
    /// The fraction of the nodes in use, relative to the maximum configured size of the store.
    /// </summary>
    public float FractionInUse => (float)Nodes.NumTotalNodes / (float)Nodes.MaxNodes;


    /// <summary>
    /// Constructor to create a store of specified maximum size.
    /// </summary>
    /// <param name="maxNodes"></param>
    /// <param name="priorMoves"></param>
    public MCTSNodeStore(int maxNodes, PositionWithHistory priorMoves = null)
    {
      if (priorMoves == null) priorMoves = PositionWithHistory.StartPosition;

      MaxNodes = maxNodes;
      int allocNodes = maxNodes;

      Nodes = new MCTSNodeStructStorage(allocNodes, null,
                                        MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC,
                                        MCTSParamsFixed.TryEnableLargePages,
                                        MCTSParamsFixed.STORAGE_USE_EXISTING_SHARED_MEM);

      long reserveChildren = maxNodes * (long)MAX_AVG_CHILDREN_PER_NODE;
      Children = new MCTSNodeStructChildStorage(this, reserveChildren);

      // Save a copy of the prior moves
      Nodes.PriorMoves = new PositionWithHistory(priorMoves);

      CeresEnvironment.LogInfo("NodeStore", "Init", $"MCTSNodeStore created with max {maxNodes} nodes, max {reserveChildren} children");

      MCTSNodeStruct.ValidateMCTSNodeStruct();
    }

    #region IDisposable Support

    private bool disposedValue = false; // To detect redundant calls


    /// <summary>
    /// Disposes of the store, releasing all nodes and children.
    /// </summary>
    /// <param name="disposing"></param>
    protected virtual void Dispose(bool disposing)
    {
      if (!disposedValue)
      {
        if (disposing)
        {
          // TODO: dispose managed state (managed objects).
        }

        Nodes?.Deallocate();
        Nodes = null;
        Children?.Deallocate();
        Children = null;

        disposedValue = true;
      }
    }

    static bool haveWarned = false;



    /// <summary>
    /// Destructor to release store.
    /// </summary>
    ~MCTSNodeStore()
    {
      // TODO: cleanup
      if (!haveWarned)
      {
        //Console.WriteLine("Finalizer for MCTSNodeStore called. Would be better to explicitly dispose");
        haveWarned = true;
      }
      Dispose(false);
    }

    readonly object disposingLock = new object();


    /// <summary>
    /// Release the store.
    /// </summary>
    public void Dispose()
    {
      lock (disposingLock)
      {
        Dispose(true);
        GC.SuppressFinalize(this);
      }
    }

    /// <summary>
    /// Resizes underlying memory block to commit only up to currently used space.
    /// </summary>
    /// <param name="numNodes"></param>
    /// <exception cref="Exception"></exception>
    public void ResizeToCurrent()
    {
      Nodes.ResizeToCurrent();
      Children.ResizeToCurrent();
    }

    #endregion

    #region Reorganization

    /// <summary>
    /// Resets the CacheIndex field to all used nodes to zero.
    /// </summary>
    public unsafe void ClearAllCacheIndices()
    {
      for (int i = 1; i < Nodes.nextFreeIndex; i++)
      {
        ref MCTSNodeStruct nodeRef = ref Nodes.nodes[i];
        nodeRef.CachedInfoPtr = null;
      }
    }


    #endregion

    #region Validation
    
    /// <summary>
    /// Diagnostic method which traverses full tree and performs 
    /// a variety of integrity checks on internal consistency,
    /// throwing an Exception if any fails.
    /// </summary>
    /// <param name="transpositionRoots"></param>
    /// <param name="expectCacheIndexZero"></param>
    public unsafe void Validate(TranspositionRootsDict transpositionRoots, bool expectCacheIndexZero = false)
    {
      int numWarnings = 0;

      void Assert(bool condition, string err)
      {
        if (!condition)
        {
          throw new Exception($"MCTSNodeStore::Validate failed: {err} ");
        }
      }

      void AssertNode(bool condition, string err, int nodeIndex, in MCTSNodeStruct node, bool warnOnly = false)
      {
        if (!condition)
        {
          string errStr = $"MCTSNodeStore::Validate failed: {err} on node: #{nodeIndex} {node.Terminal} Parent={node.ParentIndex} N={node.N} V={node.V,5:F2} TL={node.IsTranspositionLinked} PendTR={node.NumVisitsPendingTranspositionRootExtraction} ";
          if (warnOnly)
          {
            if (numWarnings == 0)
            {
              Console.WriteLine(errStr);
              Console.WriteLine("NOTE: Suppressing subsequent warnings for validation of tree.");
            }
            numWarnings++;
          }
          else
          {
            throw new Exception(errStr);
          }
        }
      }

      if (transpositionRoots != null)
      {
        Span<MCTSNodeStruct> nodes = Nodes.Span;

        // Check roots
        foreach (var kvp in transpositionRoots.Dictionary)
        {
          AssertNode(nodes[kvp.Value].IsTranspositionRoot, "Entry in transposition roots dictionary is not marked as transposition root",
                     kvp.Value, in nodes[kvp.Value]);
        }
      }

      Assert(Nodes.nodes[0].N == 0, "Null node");
      Assert(Nodes.nodes[1].IsRoot, "IsRoot");

      BitArray includedNodes = MCTSNodeStructUtils.BitArrayNodesInSubtree(this, ref Nodes.nodes[1], false, out _, 0, false);

      // Validate all nodes
      for (int i = 1; i < Nodes.nextFreeIndex; i++)
      {
        Span<MCTSNodeStruct> nodes = Nodes.Span;
        ref MCTSNodeStruct nodeR = ref nodes[i];

#if NOT
        // Verify PriorMove is valid.
        // TODO: Currently this is slow, requires retracing to root 
        //       and generating moves all the way down.
        //       Improve by doing a depth first descent in a separate pass, 
        //       and/or making parallel.
        if (!nodeR.IsOldGeneration)
        {
          Chess.MoveGen.MGPosition thisPos = nodeR.CalcPosition(this);
        }
#endif

        if (nodeR.IsOldGeneration)
        {
          AssertNode(!includedNodes[i], "Old generation node in live tree", i, in nodes[i], true);
          return;
        }

        AssertNode(nodeR.NumPieces >= 2 && nodeR.NumPieces <= 32, $"NumPieces not in [2,32]", i, in nodeR, true);
        AssertNode(nodeR.NumRank2Pawns <= 16, $"NumRank2Pawns > 16", i, in nodeR, true);
        AssertNode(nodeR.ZobristHash != 0, $"ZobristHash zero", i, in nodeR, true);
        if (!MCTSParamsFixed.UNINITIALIZED_TREE_NODES_ALLOWED)
        {
          AssertNode(nodeR.Terminal != Chess.GameResult.NotInitialized, "Node not initialized", i, in nodeR);
        }
        AssertNode(!expectCacheIndexZero || nodeR.CachedInfoPtr == null, "CacheIndex zeroed", i, in nodeR);

        if (nodeR.TranspositionRootIndex == 0 && nodeR.N > 0 && !nodeR.Terminal.IsTerminal() && nodeR.NumPolicyMoves == 0)
        {
          AssertNode(false, "Non-transposition linked node without policy initialized", i, in nodeR);
        }

        AssertNode(!nodeR.IsInFlight, "Node in flight", i, in nodeR);

        // Verify parent has a child that points to this node
        if (!nodeR.IsRoot)
        {
          int indexInParentsChildList = nodeR.ParentRef.IndexOfExpandedChildForIndex(this, nodeR.Index);
          AssertNode(indexInParentsChildList >= 0, "Parent's child list contains node", i, in nodeR);
        }

        Assert(nodeR.ParentIndex.Index != 0 || nodeR.Index.Index == 1, "Non-old generation nodes at indices other than 1 have a parent");

        if (nodeR.NumPolicyMoves > 0)
        {
          AssertNode(nodeR.childStartBlockIndex != 0, "ChildStartIndex nonzero", i, in nodeR);
        }

        if (nodeR.NumVisitsPendingTranspositionRootExtraction > 0)
        {
          AssertNode(nodeR.TranspositionRootIndex != 0,
            $"TranspositionRootIndex zero when NumVisitsPendingTranspositionRootExtraction > 0 : {nodeR.NumVisitsPendingTranspositionRootExtraction} {nodeR.TranspositionRootIndex}", i, in nodeR);

          ref MCTSNodeStruct transpositionRoot = ref Nodes.nodes[nodeR.TranspositionRootIndex];
          int numNeededValues = nodeR.N + nodeR.NumVisitsPendingTranspositionRootExtraction;
          int numAvailableValues = transpositionRoot.NumUsableSubnodesForCloning(this);
          AssertNode(numNeededValues <= numAvailableValues, $"Num needed transposition copy nodes less than number available:  {numNeededValues} {numAvailableValues}", i, in nodeR);
        }

        if (nodeR.IsTranspositionLinked)
        {
          ref readonly MCTSNodeStruct transpositionRoot = ref nodes[nodeR.TranspositionRootIndex];
          AssertNode(!transpositionRoot.IsTranspositionLinked, "transposition root was itself transposition linked", i, in nodeR);
          AssertNode(nodeR.ZobristHash == transpositionRoot.ZobristHash, $"transposition link was not to same Zobrist hash with V values: {nodeR.V} vs. {transpositionRoot.V}", i, in nodeR, true);
          AssertNode(transpositionRoot.IsTranspositionRoot, "node marked as transposition linked links to node not marked as transposition root", nodeR.TranspositionRootIndex, in transpositionRoot, false);
          // Don't report error if proven win (V > 1) because this conversion may have happened later
          AssertNode(MathF.Abs(nodeR.V - transpositionRoot.V) < 0.03f || nodeR.V > 1.0f || transpositionRoot.V > 1.0f,
                     $"transposition root had different V {nodeR.V} {transpositionRoot.V}", i, in nodeR, true);
        }
        else
        {
          // Check children
          float lastP = float.MaxValue;
          foreach (MCTSNodeStructChild child in nodeR.Children)
          {
            if (child.p > lastP)
            {
              AssertNode(child.p <= lastP, $"Children were not in descending order by prior probability: {child.p} vs. {lastP}", i, in nodeR);
              lastP = child.p;
            }
          }

          // Verify all expanded children point back to ourself
          int numExpanded = 0;
          int numChildren = 0;
          bool haveSeenUnexpanded = false;
          int sumN = 1;

          foreach (MCTSNodeStructChild child in Children.SpanForNode(in nodeR))
          {
            sumN += child.N;
            numChildren++;
            if (child.IsExpanded)
            {
              // Any expanded nodes should appear before all unexpanded nodes
              AssertNode(!haveSeenUnexpanded, "expanded after unexpanded", i, in nodeR);
              AssertNode(child.ChildRef.ParentIndex == nodeR.Index, $"ParentRef is {child.ChildRef.ParentIndex}", i, in nodeR);
              AssertNode(child.N <= nodeR.N, "child N", i, in nodeR);

              numExpanded++;
            }
            else
            {
              haveSeenUnexpanded = true;
            }
          }

          if (!nodeR.Terminal.IsTerminal() && nodeR.NumVisitsPendingTranspositionRootExtraction == 0)
          {
            AssertNode(nodeR.N == sumN, $"N {nodeR.N} versus sum children {sumN}", i, in nodeR);
          }

          AssertNode(nodeR.NumPolicyMoves == numChildren, "NumPolicyMoves", i, in nodeR);

          // Verify the NumChildrenVisited is correct
          AssertNode(numExpanded == nodeR.NumChildrenExpanded, "NumChildrenVisited", i, in nodeR);
        }
      }

      if (numWarnings > 0)
      {
        Console.WriteLine($"Number of tree validation warnings: {numWarnings}");
      }
    }

    #endregion

    #region Accessors

    /// <summary>
    /// Returns reference to node at specified index within store.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public ref readonly MCTSNodeStruct this[MCTSNodeStructIndex index]
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return ref Nodes.nodes[index.Index];
      }
    }

    /// <summary>
    /// Returns reference to node at specified index within store.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public ref readonly MCTSNodeStruct this[int index]
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return ref Nodes.nodes[index];
      }
    }

    /// <summary>
    /// Returns reference to child node at specified index within children.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    public ref readonly MCTSNodeStruct this[in MCTSNodeStruct nodeRef, int childIndex]
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        Debug.Assert(childIndex < nodeRef.NumChildrenExpanded);
        return ref Children[nodeRef.ChildAtIndex(childIndex).ChildIndex.Index];
      }
    }

    #endregion

    #region Diagnostic output

    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return Nodes == null
          ? "<MCTSNodeStore DISPOSED>"
          : $"<MCTSNodeStore Nodes Occupied {Nodes.NumUsedNodes,-15:N0}"
           + $" Chldren Allocated {Children.NumAllocatedChildren,-15:N0}>";
    }


    /// <summary>
    /// Diagnostic method to dump contents of store to Console
    /// (optionally with full child detail).
    /// </summary>
    /// <param name="childDetail"></param>
    /// <param name="annotater">optionally a method called for each node which can further annotate</param>
    public void Dump(bool childDetail, Func<MCTSNodeStructIndex, string> annotater = null)
    {
      Console.WriteLine();
      Console.WriteLine();
      Console.WriteLine(ToString());
      Console.WriteLine("Prior moves " + Nodes.PriorMoves);
      Console.WriteLine();
      for (int i = 1; i <= Nodes.NumUsedNodes; i++)
      {
        ref MCTSNodeStruct node = ref Nodes.nodes[i];
        int depth = node.DepthInTree;
        bool isWhite = Nodes.PriorMoves.FinalPosition.SideToMove == SideType.White == (depth % 2 == 1);
        string sideChar = isWhite ? "w" : "b";
        EncodedMove moveCorrectPerspective = isWhite ? node.PriorMove : node.PriorMove.Flipped;

        string annotation = annotater?.Invoke(new MCTSNodeStructIndex(i));
        Console.WriteLine($"{i,7} Depth={depth} N={node.N} {sideChar}:{moveCorrectPerspective} V={node.V,6:F3} {node.Terminal} W={node.W,10:F3} Parent={node.ParentIndex.Index} " +
                          $"InFlights={node.NInFlight}/{node.NInFlight2}" +
                          $"ChildStartIndex={node.ChildStartIndex} NumPolicyMoves={node.NumPolicyMoves} Cached?={node.IsCached} " + annotation);
        if (node.IsOldGeneration)
        {
          Console.WriteLine("          OLD GENERATION");
        }
        else if (node.IsTranspositionLinked)
        {
          Console.WriteLine("          Transposition Linked to " + -node.ChildStartIndex);
        }
        else if (childDetail)
        {
          int maxExpandedIndex = node.NumChildrenExpanded - 1;

          int childIndex = 0;
          foreach (MCTSNodeStructChild child in node.Children)
          {
            Console.Write($"          [{node.ChildStartIndex + childIndex++,8}] ");
            if (child.IsExpanded)
            {
              if (MCTSNodeStoreContext.Store != null)
              {
                Console.WriteLine($"{child.ChildIndex} --> {child.ChildRefFromStore(this).ToString()}");
              }
              else
              {
                Console.WriteLine($"{child.ChildIndex}");
              }
            }
            else
            {
              Console.WriteLine($"{(isWhite ? child.Move.Flipped : child.Move)} {child.P} ");
            }

            if (childIndex > maxExpandedIndex + 2)
            {
              Console.WriteLine($"    (followed by {node.NumPolicyMoves - childIndex} additional unexpanded children)");
              break;
            }
          }
        }
        Console.WriteLine();
      }
    }

#endregion
  }
}


#if EXPERIMENTAL
    /// <summary>
    /// 
    /// </summary>
    /// <param name="movesMade"></param>
    /// <param name="thresholdFractionNodesRetained"></param>
    /// <returns></returns>
    public bool ResetRootAssumingMovesMade(IEnumerable<MGMove> movesMade, float thresholdFractionNodesRetained)
    {
      PositionWithHistory staringPriorMove = Nodes.PriorMoves;
      MGPosition position = Nodes.PriorMoves.FinalPosMG;

      ref MCTSNodeStruct priorRoot = ref RootNode;

      // Advance root node and update prior moves
      ref MCTSNodeStruct newRoot = ref priorRoot;
      foreach (MGMove moveMade in movesMade)
      {
        // Append the moves made to the prior moves
        Nodes.PriorMoves.AppendMove(moveMade);

        bool foundChild = false;

        // Find this new root node (after these moves)
        foreach (MCTSNodeStructChild child in newRoot.Children)
        {
          if (child.IsExpanded)
          {
            MGMove thisChildMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(child.Move, in position);
            if (thisChildMove == moveMade)
            {
// NO! can't modify structure since we may need to abandon without modifying
//              // Mark this node as deleted
              newRoot.Detached = true;

              // Advance new root to reflect this move
              newRoot = ref child.ChildRef;

              // Advance position
              position.MakeMove(thisChildMove);

              // Done looking for match
              foundChild = true;
              break;
            }
          }
        }

        if (!foundChild)
        {
          // Restore the tree the way we originally found it
          Nodes.PriorMoves = staringPriorMove;
          Console.WriteLine("No follow - not found");
          return false;
        }
      }

      // Only switch to this root if it meets the threshold size
      float fractionNodesCouldBeRetained = (float)newRoot.N / (float)priorRoot.N;
      if (fractionNodesCouldBeRetained < thresholdFractionNodesRetained)
      {
        Console.WriteLine("No follow - fraction too small " + fractionNodesCouldBeRetained);
        return false;
      }

      // The new root no longer has a parent
      newRoot.ParentIndex = MCTSNodeStructIndex.Null;

      // Reset root to this node
      RootIndex = newRoot.Index;

      // TODO: should we consider removing the node from the transposition root table?

      // Success
      return true;
    }

#endif
