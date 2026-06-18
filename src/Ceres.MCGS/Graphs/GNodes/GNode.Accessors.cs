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

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

public readonly partial struct GNode : IComparable<GNode>, IEquatable<GNode>
{
  #region Fields containing accumulated values during search

    /// <summary>
    /// Number of visits to self plus children.
    /// </summary>
    public readonly int N => NodeRef.N;

    #endregion

    #region Fields directly from neural network output or search initialization

    /// <summary>
    /// Number of children which have been expanded 
    /// (a corresponding node created in the graph).
    /// </summary>
    public readonly byte NumEdgesExpanded => NodeRef.NumEdgesExpanded;

  /// <summary>
  /// Win probability.
  /// </summary>
  public readonly FP16 WinP => NodeRef.WinP;

  /// <summary>
  /// Loss probability.
  /// </summary>
  public readonly FP16 LossP => NodeRef.LossP;

  /// <summary>
  /// Draw probability.
  /// </summary>
  public readonly float DrawP => NodeRef.DrawP;
  
  /// <summary>
  /// Uncertainty associated with value score.
  /// </summary>
  public readonly FP16 UncertaintyValue => NodeRef.UncertaintyValue;

  /// <summary>
  /// Uncertainty associated with policy.
  /// </summary>
  public readonly FP16 UncertaintyPolicy => NodeRef.UncertaintyPolicy;

  /// <summary>
  /// Exponentially-weighted (~50-visit) volatility of the leaf values backed up through this node,
  /// measured as RMS deviation about the node's Q. Nonzero only when search was run with
  /// ParamsSearch.TrackLeafValueVolatility enabled.
  /// </summary>
  public readonly double LeafValueVolatility => NodeRef.LeafValueVolatility.RunningStdDev;

  /// <summary>
  /// Bias-corrected variant of <see cref="LeafValueVolatility"/> that removes the EWMA cold-start
  /// under-reporting by supplying this node's N as the effective sample count. Prefer this over the
  /// raw value when comparing volatility across nodes with differing visit counts (otherwise small-N
  /// nodes look spuriously settled). See <see cref="RunningStdDevShort.RunningStdDevDebiased"/>.
  /// </summary>
  public readonly double LeafValueVolatilityDebiased => NodeRef.LeafValueVolatility.RunningStdDevDebiased(N);

  /// <summary>
  /// Fortress probability metric: minimum P(NEVER) over all pawn squares.
  /// High values indicate a pawn unlikely to ever move, suggesting fortress-like structure.
  /// </summary>
  public readonly float FortressP => NodeRef.FortressP;

  /// <summary>
  /// Number of policy moves (children)
  /// Possibly this set of moves is incomplete due to either:
  ///   - implementation decision to "throw away" lowest probability moves to save storage, or
  ///   - error in policy evaluation which resulted in certain legal moves not being recognized
  /// </summary>
  public readonly byte NumPolicyMoves => NodeRef.NumPolicyMoves;

  /// <summary>
  /// Hash value for the position (standalone, i.e. not inclusive of any history).
  /// </summary>
  public readonly PosHash64 HashStandalone => NodeRef.HashStandalone;

  #endregion

  #region Miscellaneous fields

  /// <summary>
  /// Terminal status of the node.
  /// </summary>
  public readonly GameResult Terminal => NodeRef.Terminal;

  /// <summary>
  /// If a checkmate has been shown to exist among the children.
  /// </summary>
  public readonly bool CheckmateKnownToExistAmongChildren => NodeRef.CheckmateKnownToExistAmongChildren;

  /// <summary>
  /// If a draw has been shown to exist among the children.
  /// </summary>
  public readonly bool DrawKnownToExistAmongChildren => NodeRef.DrawKnownToExistAmongChildren;


  /// <summary>
  /// Helper to set ChildNodeHasDrawKnownToExist on all parent edges.
  /// </summary>
  private void SetDrawKnownToExistOnParentEdges(GNode node)
  {
    if (MCGSParamsFixed.ENABLE_DRAW_KNOWN_TO_EXIST)
    {
      foreach (GEdge parentEdge in node.ParentEdges)
      {
        parentEdge.ChildNodeHasDrawKnownToExist = true;
#if NOT
        // We don't bother to reset Q here. 
        // The adjustment to edge Q will be made dynamically can continually in GatherChildInfoViaChildren.
        if (parentEdge.Q < 0)
        {
          parentEdge.Q = 0;
          // TODO: set D (somehow?)
        }
#endif
      }
    }
  }

  public void SetDrawKnownToExistAtNode()
  {
    if (MCGSParamsFixed.ENABLE_DRAW_KNOWN_TO_EXIST)
    {
      NodeRef.DrawKnownToExistAmongChildren = true;
      NodeRef.SiblingsQFrac = 0;

      if (!NodeRef.IsSearchRoot)
      {
        SetDrawKnownToExistOnParentEdges(this);
      }
    }
  }

#if NOT
  public void SetDrawKnownToExistAtParents()
  {
    if (MCGSParamsFixed.ENABLE_DRAW_KNOWN_TO_EXIST)
    {
      foreach (GEdge childEdgeToThisNode in ParentEdges)
      {
        childEdgeToThisNode.ParentNode.NodeRef.DrawKnownToExistAmongChildren = true;
        if (!childEdgeToThisNode.ParentNode.IsGraphRoot)
        {
          SetDrawKnownToExistOnParentEdges(childEdgeToThisNode.ParentNode);
        }
      }
    }
  }
#endif

  /// <summary>
  /// If the node belonged to a prior seacrh graph but is 
  /// now unreachable due to a new root having been swapped into place.
  /// </summary>
  public readonly bool IsOldGeneration => NodeRef.miscFields.IsOldGeneration;

  /// <summary>
  /// If the position has one more repetitions in the history.
  /// </summary>
  public readonly bool HasRepetitions => NodeRef.miscFields.HasRepetitions;

  /// <summary>
  /// Fraction of this node's visits which terminated at a history-sensitive
  /// (repetition/50-move) terminal draw edge anywhere beneath it
  /// (stochastically-rounded single-byte running average; see GNodeStruct).
  /// </summary>
  public readonly double RepDrawFraction => NodeRef.RepDrawFraction;

  /// <summary>
  /// Number of pieces on board.
  /// </summary>
  public readonly byte NumPieces => NodeRef.miscFields.NumPieces;

  /// <summary>
  /// Number of pawns still on second rank.
  /// </summary>
  public readonly byte NumRank2Pawns => NodeRef.miscFields.NumRank2Pawns;

  /// <summary>
  /// If the node corresponds to a position with white to play.
  /// </summary>
  public readonly bool IsWhite => NodeRef.miscFields.IsWhite;


  public readonly bool UnusedBool1
  {
    get => NodeRef.miscFields.UnusedBool1;
    set => NodeRef.miscFields.UnusedBool1 = value;
  }


  /// <summary>
  /// Indicator if the first node in the virtual transposition subgraph
  /// has been identified as a draw by repetition.
  /// </summary>
  public readonly bool IsDirty => NodeRef.miscFields.IsDirty;

#endregion


  internal int BlockIndexIntoEdgeHeaderStore
  {
    get
    {
      // Do not expect to find deferred copy here
      // (should have called TryDoDeferredPolicyCopyIfNeeded already).
      Debug.Assert(!NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex);
      return NodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
    }

    set => NodeRef.edgeHeaderBlockIndexOrDeferredNode = new EdgeHeaderBlockIndexOrNodeIndex(value);    
  }

  public readonly bool BlockIndexIntoEdgeHeaderStoreIsDeferred => NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex;

  public readonly bool IsPendingPolicyCopy => NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex;


  /// <summary>
  /// If a deferred policy copy is pending, attempts to materialize it now
  /// (allocating edge headers and copying policy from the source node).
  /// Returns false if the source node's lock could not be acquired.
  ///
  /// The source lock is acquired NON-blockingly: the caller holds this node's lock
  /// (typically as a select-phase expansion parent), and a blocking acquire of a
  /// second node lock while holding one creates a deadlock surface (see the same
  /// pattern in TranspositionAutoExtension). On contention the copy simply remains
  /// deferred; callers abort/retry at their level.
  /// </summary>
  internal bool TryDoDeferredPolicyCopyIfNeeded()
  {
    if (!IsPendingPolicyCopy)
    {
      return true;
    }

    ref GNodeStruct nodeRef = ref NodeRef;

    // The stored value is a pointer to the source node; allocate and copy values now.
    GNode copyFrom = Graph[nodeRef.edgeHeaderBlockIndexOrDeferredNode.NodeIndex];

    Debug.Assert(this.IsLocked);
    if (!copyFrom.TryAcquireLock())
    {
      return false;
    }
    try
    {
      nodeRef.edgeHeaderBlockIndexOrDeferredNode.Clear();
      Graph.AllocateAndCopyPolicyValues(copyFrom, this);
    }
    finally
    {
      copyFrom.ReleaseLock();
    }

    return true;
  }


  #region Value related accessors

  /// <summary>
  /// Returns if this node has been evaluated.
  /// </summary>
  public readonly bool IsEvaluated => !FP16.IsNaN(WinP);


  /// <summary>
  /// Sum of all V from all N evals (self and child visits).
  /// Represented in high precision (double) to avoid "round to zero" update errors when W becomes large.
  /// </summary>
  public readonly double Q => NodeRef.Q;

  /// <summary>
  /// Average draw percentage.
  /// </summary>
  public readonly double D => NodeRef.D;

  /// <summary>
  /// Computes the D value (draw probability) from immediate children.
  /// Used for cold-path root D recomputation (e.g., UCI display).
  /// This eliminates possible staleness (but only down one level).
  /// </summary>
  /// <returns>The computed D value.</returns>
  public readonly double ComputeDFromChildren()
  {
    if (N <= 1)
    {
      return DrawP;
    }

    double dSum = DrawP; // self contribution (1 visit for initial eval)
    for (int i = 0; i < NumEdgesExpanded; i++)
    {
      GEdge edge = ChildEdgeAtIndex(i);
      if (edge.N > 0)
      {
        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
        {
          int nNonRep = edge.N - edge.NDrawByRepetition;
          dSum += edge.ChildNode.D * nNonRep + 1.0 * edge.NDrawByRepetition;
        }
        else if (edge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn)
        {
          dSum += 1.0 * edge.N;
        }
        // TerminalEdgeDecisive: D=0, no contribution
      }
    }
    return dSum / N;
  }

  /// <summary>
  /// Draw probability to use for display (UCI WDL, dumps, SVG).
  ///
  /// Returns the raw neural-network DrawP for an unsearched node (N &lt;= 1), otherwise the
  /// exact-from-children aggregate (ComputeDFromChildren), which is correct one level down
  /// regardless of any residual staleness in this node's stored D (e.g. from off-path
  /// multi-parent visits). Pair with the always-correct Q to derive consistent W/L:
  ///   W = (Q + 1 - D) / 2,  L = (1 - D - Q) / 2.
  /// </summary>
  public readonly double ComputeDForDisplay() => N <= 1 ? DrawP : ComputeDFromChildren();

  /// <summary>
  /// Average win percentage.
  /// </summary>
  public readonly float W => (float)((Q + 1 - D) / 2.0);


  /// <summary>
  /// Average loss percentage.  
  /// </summary>
  public readonly float L => (float)((1 - D - Q) / 2.0);


  /// <summary>
  /// Neural network evaluation of win - loss.
  /// </summary>
  public readonly float V => (float)WinP - (float)LossP;

  #endregion


  #region Lock Operations

  /// <summary>
  /// Acquires the lock for this node.
  /// Delegates to the underlying NodeRef.LockRef to ensure atomic operations on the actual lock byte.
  /// </summary>
  public void AcquireLock() => NodeRef.LockRef.Acquire();

  /// <summary>
  /// Attempts to acquire the lock for this node without blocking,
  /// returning false if the lock is currently held (by any thread, including this one).
  /// </summary>
  public bool TryAcquireLock() => NodeRef.LockRef.TryAcquire();

  /// <summary>
  /// Releases the lock for this node.
  /// Delegates to the underlying NodeRef.LockRef to ensure atomic operations on the actual lock byte.
  /// </summary>
  public void ReleaseLock() => NodeRef.LockRef.Release();

  /// <summary>
  /// Returns whether this node's lock is currently held by any thread.
  /// Delegates to the underlying NodeRef.LockRef to ensure proper volatile reads.
  /// </summary>
  public readonly bool IsLocked => NodeRef.LockRef.IsLocked;

  /// <summary>
  /// If it is known that this node's lock is held by the current thread
  /// (but it is not always possible to determine this).
  /// </summary>
  public readonly bool IsKnownLockedByThisThread => NodeRef.LockRef.IsKnownLockedByThisThread;


  public void SetLockIllegalValue() => NodeRef.LockRef.SetIllegalValue();

  #endregion
}
