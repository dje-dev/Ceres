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

using Ceres.Base.OperatingSystem;
using Ceres.Base.Environment;
using Ceres.Chess.Positions;
using Ceres.MCTS.Params;
using Ceres.MCTS.MTCSNodes.Struct;

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
    /// Approximate average total bytes consumed by a node, child pointers, and associated data structures.
    /// For very large trees with many transposition-linked nodes (not needing their own copy of child info)
    /// this number can be as small as 130 bytes (on top of which there will be some fixed overhead of circa 5GB).
    /// </summary>
    const ulong APPROX_BYTES_PER_NODE = 160;

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
    public MCTSNodeStructIndex RootIndex { get; internal set; }

    public const int AVG_CHILDREN_PER_NODE = 55;

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
      
      long reserveChildren = maxNodes * (long)AVG_CHILDREN_PER_NODE;
      Children = new MCTSNodeStructChildStorage(this, reserveChildren);
      
      // Save a copy of the prior moves
      Nodes.PriorMoves = new PositionWithHistory(priorMoves);

      CeresEnvironment.LogInfo("NodeStore", "Init", $"MCTSNodeStore created with max {maxNodes} nodes, max {reserveChildren} children");

      MCTSNodeStruct.ValidateMCTSNodeStruct();
      RootIndex = new MCTSNodeStructIndex(1);
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

    #endregion

    #region Reorganization

    /// <summary>
    /// Resets the CacheIndex field to all used nodes to zero.
    /// </summary>
    public void ClearAllCacheIndices()
    {
      for (int i = 1; i < Nodes.nextFreeIndex; i++)
      {
        Nodes.nodes[i].CacheIndex = 0;
      }
    }


    #endregion

    #region Validation

    /// <summary>
    /// Diagnostic method which traverses full tree and performs 
    /// a variety of integrity checks on internal consistency,
    /// throwing an Exception if any fails.
    /// </summary>
    public void Validate(bool expectCacheIndexZero = false)
    {
      void Assert(bool condition, string err)
      {
        if (!condition)
        {
          throw new Exception($"MCTSNodeStore::Validate failed: {err} ");
        }
      }

      Assert(Nodes.nodes[0].N == 0, "Null node");
      Assert(Nodes.nodes[1].IsRoot, "IsRoot");


      // Validate all nodes
      for (int i = 1; i < Nodes.nextFreeIndex; i++)
      {
        ref MCTSNodeStruct nodeR = ref Nodes.nodes[i];

        if (nodeR.IsOldGeneration)
        {
          continue;
        }

        Assert(!expectCacheIndexZero || nodeR.CacheIndex == 0, "CacheIndex zeroed");

        Assert(!nodeR.IsInFlight, "Node in flight");

        // Verify parent has a child that points to this node
        if (!nodeR.IsRoot)
        {
          int indexInParentsChildList = nodeR.ParentRef.IndexOfExpandedChildForIndex(this, nodeR.Index);
          Assert(indexInParentsChildList >= 0, "Parent's child list contains node");
        }

        Assert(nodeR.ParentIndex.Index != 0 || nodeR.Index.Index == 1, "Non-old generation nodes at indices other than 1 have a parent");

        if (nodeR.NumPolicyMoves > 0)
        {
          Assert(nodeR.childStartBlockIndex != 0, "ChildStartIndex nonzero");
        }

        if (nodeR.NumNodesTranspositionExtracted > 0)
        {
          Assert(nodeR.TranspositionRootIndex != 0, 
            $"TranspositionRootIndex zero when NumNodesTranspositionExtracted > 0 : {nodeR.NumNodesTranspositionExtracted} {nodeR.TranspositionRootIndex}");
        }

        if (nodeR.IsTranspositionLinked)
        {
          ref readonly MCTSNodeStruct transpositionRoot = ref Nodes.nodes[nodeR.TranspositionRootIndex];
          Assert(!transpositionRoot.IsTranspositionLinked, "transposition root was itself transposition linked");
          Assert(nodeR.ZobristHash == transpositionRoot.ZobristHash, "transposition link was not to same Zobrist hash");
        }
        else
        {
          // Verify all expanded children point back to ourself
          int numExpanded = 0;
          int numChildren = 0;
          bool haveSeenUnexpanded = false;
          foreach (MCTSNodeStructChild child in Children.SpanForNode(in nodeR))
          {
            numChildren++;
            if (child.IsExpanded)
            {
              // Any expanded nodes should appear before all unexpanded nodes
              Assert(!haveSeenUnexpanded, "expanded after unexpanded");
              Assert(child.ChildRef.ParentIndex == nodeR.Index, "ParentRef");
              Assert(child.N <= nodeR.N, "child N");

              numExpanded++;
            }
            else
            {
              haveSeenUnexpanded = true;
            }
          }

          Assert(nodeR.NumPolicyMoves == numChildren, "NumPolicyMoves");

          // Verify the NumChildrenVisited is correct
          Assert(numExpanded == nodeR.NumChildrenExpanded, "NumChildrenVisited");
        }
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
    public void Dump(bool childDetail, Func<MCTSNodeStructIndex,string> annotater = null)
    {
      Console.WriteLine();
      Console.WriteLine();
      Console.WriteLine(ToString());
      Console.WriteLine("Prior moves " + Nodes.PriorMoves);
      Console.WriteLine();
      for (int i = 1; i <= Nodes.NumUsedNodes; i++)
      {
        ref MCTSNodeStruct node = ref Nodes.nodes[i];

        string annotation = annotater?.Invoke(new MCTSNodeStructIndex(i));
        Console.WriteLine($"{i,7} {node.PriorMove} {node.V,6:F2} {node.Terminal} {node.W,9:F2} Parent={node.ParentIndex.Index} " +
                          $"InFlights={node.NInFlight}/{node.NInFlight2}" +
                          $"ChildStartIndex={node.ChildStartIndex} NumPolicyMoves={node.NumPolicyMoves} CacheIndex={node.CacheIndex} " + annotation);
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
              Console.WriteLine($"{child.Move} {child.P} ");
            }

            if (childIndex > maxExpandedIndex + 1)
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
