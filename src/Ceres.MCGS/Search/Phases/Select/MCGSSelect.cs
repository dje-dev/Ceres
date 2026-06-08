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
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Strategies;
using Ceres.MCGS.Search.RPO;
using Ceres.MCGS.Utils;
using Microsoft.Extensions.ObjectPool;

#endregion

namespace Ceres.MCGS.Search.Phases;

/// <summary>
/// Performs the select phase of MCGS, recursively descending the graph 
/// until a node is encountered from which an evaluation can be extracted.
/// </summary>
public class MCGSSelect
{
  public readonly MCGSEngine Engine;

  public readonly NNEvaluatorDef EvaluatorDef;
  public readonly NNEvaluator Evaluator;

  // Static readonly array for the common single-visit case.
  [ThreadStatic]
  private static short[] singleVisitArray;

  // Struct to hold deferred subpath information (avoids boxing)
  private readonly struct DeferredSubPath : IComparable<DeferredSubPath>
  {
    public readonly MCGSPath SubPath;
    public readonly int NumVisits;
    public readonly bool LaunchParallel;

    public DeferredSubPath(MCGSPath subPath, int numVisits, bool launchParallel)
    {
      SubPath = subPath;
      NumVisits = numVisits;
      LaunchParallel = launchParallel;
    }

    public int CompareTo(DeferredSubPath other) => 0; // Not used for sorting
  }

  // Pooled object policy for ListBounded<DeferredSubPath>
  private sealed class DeferredSubPathListPolicy : PooledObjectPolicy<ListBounded<DeferredSubPath>>
  {
    public override ListBounded<DeferredSubPath> Create() => new ListBounded<DeferredSubPath>(64);

    public override bool Return(ListBounded<DeferredSubPath> obj)
    {
      obj.Clear(false);
      return true;
    }
  }

  // ThreadStatic pool for ListBounded<DeferredSubPath>
  [ThreadStatic]
  private static DefaultObjectPool<ListBounded<DeferredSubPath>> deferredSubPathsPool;

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="engine"></param>
  public MCGSSelect(MCGSEngine engine)
  {
    Engine = engine;
  }


  /// <summary>
  /// Populates specified MCGSPath, recursively descending the graph until
  /// a node is encountered from which an evaluation can be extracted 
  /// (either immediately or via a subsequent NN evaluation).
  /// (or subsequently created from an NN evaluation).
  /// 
  /// Performs recursive tree descent:
  ///   1. Initialize path from search root(or continue from parent's child).
  ///   2. Acquire parent lock and do deferred policy copy.
  ///   3. Determine children to consider: NumEdgesExpanded + numTargetVisits, clamped to NumPolicyMoves.
  ///   4. Fast path: If 1 visit and 0 expanded edges, directly assign visit to child 0 (bypass PUCT).
  ///   5. PUCT selection via PUCTSelector.ComputeTopChildScores() -> PUCTScoreCalcVector.ScoreCalcMulti() with SIMD-vectorized score computation.
  ///      Handles virtual loss, FPU, CPUCT with log scaling, temperature, futility pruning, checkmate, certainty propagation.
  ///   6. Q reset (graph mode): Immediately propagate recomputed child stats to parent Q.
  ///   7. Capacity abort check.
  ///   8. ProcessChildren via SelectVisitsEnumerator (multi-pass: unexpanded first, then visited children in two passes for parallelism). For each child with visits > 0, dispatches to one of the descent condition handlers below.
  ///   9. Release parent lock, then dispatch deferred recursive descents (possibly via worker pool).
  ///   
  /// Descent condition taxonomy for ExtendPathsRecursively -> ProcessChildren.
  ///
  /// EXPANDED CHILD (childIndex < NumEdgesExpanded):
  ///   A. Terminal edge (edge type is TerminalEdgeDrawn/Decisive)
  ///      -> Terminate TerminalEdge, accept all visits.
  ///   B. Draw by repetition in coalesce mode (position duplicate + PositionEquivalence mode)
  ///      -> Terminate DrawByRepetitionInCoalesceMode.
  ///   C. Child node is terminal (childNode.Terminal.IsTerminal())
  ///      -> Terminate Terminal, accept all visits.
  ///   D. Transposition with sufficient N (IsTranspositionSufficientN)
  ///      -> Terminate TranspositionLinkNodeSufficientN, accept all visits.
  ///   E. Visited child, insufficient N (childNode.N > 0, needs more visits)
  ///      -> PossiblyBranched, defer recursive descent.
  ///   F. Unvisited child, in-flight collision (childNode.N == 0, already in this batch)
  ///      -> CalcUnvisitedLeafMethod: Abort / PiggybackPendingNNEval / AlreadyNNEvaluated.
  ///
  /// UNEXPANDED CHILD (childIndex >= NumEdgesExpanded):
  ///   G. Terminal result (checkmate, stalemate, insufficient material, 50-move, tablebase, ply-1 TB)
  ///      -> Create terminal edge, terminate TerminalEdge.
  ///   H. Draw by repetition in coalesce mode (unexpanded variant)
  ///      -> Create node (possibly), terminate DrawByRepetitionInCoalesceMode.
  ///   I. Node creation collision (another thread already expanding same position)
  ///      -> Terminate PiggybackPendingNNEval, accept 1 visit.
  ///   J. New node created
  ///      -> Terminate PendingNeuralNetEval or TranspositionCopyValues (if twin exists).
  ///   K. Transposition link to existing node
  ///      K1. childNode.N == 0: CalcUnvisitedLeafMethod (Abort / Piggyback / AlreadyEvaluated).
  ///      K2. Sufficient N:     Terminate TranspositionLinkNodeSufficientN.
  ///      K3. Insufficient N:   Defer recursive descent.
  /// </summary>
  /// <param name="iterator"></param>
  /// <param name="path"></param>
  /// <param name="numAttemptedVisits"></param>
  public void ExtendPathsRecursively(MCGSIterator iterator, MCGSPath path, int numAttemptedVisits)
  {
    Debug.Assert(numAttemptedVisits > 0);

    MCGSPathsSet pathsSet = iterator.PathsSet;

    ParamsSearch paramsSearch = iterator.Engine.Manager.ParamsSearch;
    bool graphEnabled = paramsSearch.EnableGraph;
    bool enablePUCSuboptimalityThreshold = paramsSearch.VisitSuboptimalityRejectThreshold.HasValue;
    bool parallelEnabled = paramsSearch.Execution.SelectOperationParallelThresholdNumVisits < int.MaxValue;
    float cpuctMultiplier = iterator.CPUCTMultiplier;
    bool refreshSibling = MCGSParamsFixed.REFRESH_SIBLING_DURING_SELECT_PHASE && paramsSearch.EnablePseudoTranspositionBlending;

    GNode parentNode;
    if (path == null)
    {
      path = iterator.AllocatedPath(NumInitialSlotsFromNumVisits(0, numAttemptedVisits));
      path.RunningHash = Engine.SearchRootRunningHash;
      parentNode = path.Engine.SearchRootNode;

      if (Engine.NeedsPlySinceLastMove)
      {
        Engine.SearchRootPlySinceLastMove.AsSpan().CopyTo(path.PlySinceLastMove.SquarePlySince);
      }
    }
    else
    {
      // The parent of the next level was the child of the prior level.
      parentNode = path.LeafVisitRef.ParentChildEdge.ChildNode;
    }
    Debug.Assert(path.NumVisitsInPath == 0 || path.LeafVisitRef.ChildNode.N > 0); // TODO: Not true if prefetch?

    parentNode.AcquireLock();

    parentNode.DoDeferredPolicyCopyIfNeeded();

    const bool ALSO_COMPUTE_CHILD_SCORES = false;

    MGPosition parentPosMG = path.NumVisitsInPath == 0 ? path.Engine.SearchRootPosMG : path.LeafVisitRef.ChildPosition;
    Debug.Assert(parentPosMG != default);

    MCGSSelectBackupStrategyBase strategy = path.Strategy;

    // Reorder unvisited children by PUCT scores (blending policy and action head)
    // on second visit, before any child selection (fast path or full PUCT).
    if (parentNode.NumEdgesExpanded == 0 && parentNode.N >= 1)
    {
      strategy.PossiblyActionResortUnvisitedChildren(parentNode, Engine.Graph);
    }

    int numChildrenToConsider = strategy.NumChildrenToConsider(parentNode, numAttemptedVisits);
    numChildrenToConsider = Math.Min(parentNode.NumPolicyMoves, numChildrenToConsider);

    // Select children
    Span<short> childVisitCounts;
    Span<double> scores;
    NodeSelectAccumulator childStats;
    if (numAttemptedVisits == 1
     //     && parentNode.N == 1
     && parentNode.NumEdgesExpanded == 0
     && !ALSO_COMPUTE_CHILD_SCORES)
    {
      // Shortcut common case of first visit to any child sd always first child.
      // Reset the reused ThreadStatic array to represent a single visit to first child.
      singleVisitArray ??= new short[1];
      singleVisitArray[0] = 1;
      childVisitCounts = singleVisitArray;
      childStats = new NodeSelectAccumulator(1, parentNode.V, parentNode.DrawP, 1);
      scores = default;
    }
    else
    {
      float temperatureMultiplierBase = false && Engine.Manager.ParamsSearch.TestFlag2 ? (0.95f + 1 * 0.25f * parentNode.UncertaintyPolicy) : 1.0f;

      // Possible adjustement for path-dependent CPUCT scaling
      const float THRESHOLD_SUBOPTIMALITY_POSSIBLY_SCALE_CPUCT = 0.15f;
      if (Engine.Manager.ParamsSearch.EnablePathDependentCPUCTScaling
       && path.MaxQSubOptimality > THRESHOLD_SUBOPTIMALITY_POSSIBLY_SCALE_CPUCT)
      {
        // Experimental idea is to reduce exportation if we have
        // already seen highly explortory visits above since
        // we will have few samples of the branch and therefore
        // want to suppress subsequent exploration.
        // The attenuation is proportional to the max suboptimality seen.
        // NOTE: high values of multiplier (such as 0.2) definitely worse than no scaling
        const float MIN_CPUCT = 0.1f;
        const float CPUCT_SCALE_MULTIPLIER = 0.2f;
        cpuctMultiplier *= MathF.Max(MIN_CPUCT, 1.0f - CPUCT_SCALE_MULTIPLIER * path.MaxQSubOptimality);
      }

      if (false && parentNode.N > 10 && Engine.Manager.ParamsSearch.SelectExplorationForUncertaintyAtNode > 0)
      {
        cpuctMultiplier *= 1f + 0.2f * (float)(parentNode.NodeRef.LeafValueVolatility.RunningStdDev - 0.1);
      }

      //#if FEATURE_UNCERTAINTY_POLICY
      if (
        //-7 Elo
        false && Engine.Manager.ParamsSearch.TestFlag2)
      {
        Debug.Assert(!float.IsNaN(parentNode.UncertaintyPolicy));
        //Console.WriteLine("UCCP: " + parentNode.UncertaintyPolicy + "  " + parentNode.UncertaintyValue);
        // Research idea: increase CPUCT if high policy uncertainty.
        // Low uncertainty may just indicate network saw position many times
        // (not any reflection of optimal policy).
        // However high uncertanity indicates we should somewhat discount policy.
        // Tests did not show improvement.
        cpuctMultiplier *= (parentNode.UncertaintyPolicy > 0.3 || parentNode.UncertaintyValue > 0.5)
          ? StatUtils.Bounded(1f
          + 0.15f * parentNode.UncertaintyPolicy
          + 0.25f * parentNode.UncertaintyValue,
          0.5f, 1.5f)
          : 0.95f; // Scale to keep average close to 1.0        
      }
      //#endif

      if ((Engine.Manager.ParamsSearch.MoveOrderingPhase == ParamsSearch.MoveOrderingPhaseEnum.ChildSelection
       || Engine.Manager.ParamsSearch.MoveOrderingPhase == ParamsSearch.MoveOrderingPhaseEnum.NodeInitializationAndChildSelect)
       && !parentNode.IsSearchRoot)
      {
        const int MAX_LOOK_RIGHT = 5;
        int firstIndex = parentNode.NumEdgesExpanded; // only allow reordering if not yet expanded (guaranteening no side effects)
        int numLookRight = Math.Min(MAX_LOOK_RIGHT, numAttemptedVisits);
        parentNode.CheckMoveOrderRearrangeAtIndex(in path.LeafVisitRef.ChildPosition, firstIndex, firstIndex + MAX_LOOK_RIGHT, MCGSParamsFixed.MOVE_ORDERING_MIN_RATIO_POLICY);
      }

      // If EnableBackupPropagationToParentsOffDirectPath was true, back up phase may have marked some edges as stale.
      // We allow SelectChildren to refresh these edges (if running in graph mode when desynchronization is possible).
      // Ordinarily the edge Q should only be updated in conjunction with a backup to the parent node,
      // but we know that the parent pure Q will be reset using the (refreshed child) computed here.
      bool REFRESH_STALE_EDGES = graphEnabled; // edges only marked stale in graph mode
      childStats = strategy.SelectChildren(parentNode, path.IteratorID, path.NumVisitsInPath, numChildrenToConsider,
                                           numAttemptedVisits, ALSO_COMPUTE_CHILD_SCORES,
                                           cpuctMultiplier, 1,
                                           REFRESH_STALE_EDGES,
                                           Engine.Manager.RootMovesPruningStatus,
                                           out childVisitCounts, out scores);

      // CB-GPUCT graph-aware absorption: ScoreCalc may have refused to allocate
      // some (or all) of the requested visits when every child was already at or
      // over its pi_bar visit target. The unallocated visits are absorbed at the
      // parent here - increment N, keep Q stable (delta_W = parent.Q * deltaN so
      // the visit-weighted Q is unchanged; in V_bar mode the value gets recomputed
      // from current child stats and is also unchanged since children weren't touched).
      int numAbsorbedAtParent = numAttemptedVisits - childStats.NumVisitsAccepted;
      if (numAbsorbedAtParent > 0
          && Engine.Manager.ParamsSelect.CBGPUCTSelectActive
          && Engine.Manager.ParamsSelect.CBGPUCT_SelectCrossParentNEnabled)
      {
        strategy.BackupToNode(parentNode, numAbsorbedAtParent,
                              parentNode.Q * numAbsorbedAtParent,
                              parentNode.D * numAbsorbedAtParent);
        if (MCGSParamsFixed.DEBUG_CBGPUCT)
        {
          Console.WriteLine($"[CBGPUCT] absorb at parent node=#{parentNode.Index.Index} "
                          + $"absorbed={numAbsorbedAtParent} (of {numAttemptedVisits}) "
                          + $"newN={parentNode.N} Q={parentNode.Q:+0.0000;-0.0000}");
        }
        // Reduce numAttemptedVisits so downstream assertions and processing reflect
        // only the visits that actually went to children.
        numAttemptedVisits = childStats.NumVisitsAccepted;
      }

#if DEBUG // Temporarily using conditional compilation due to TensorPrimives versioning issue
      Debug.Assert(enablePUCSuboptimalityThreshold || TensorPrimitives.Sum(childVisitCounts) == numAttemptedVisits);
#endif

      if (enablePUCSuboptimalityThreshold)
      {
        numAttemptedVisits = ApplyPUCTSuboptimalityThreshold(path, numAttemptedVisits, childStats.NumVisitsAccepted, childVisitCounts);
      }
    }

    // TODO: can this be removed (along with the acquire elsewhere in this file)
    //parentNode.ReleaseLock(); 
    Debug.Assert(!double.IsNaN(childStats.SumW));

    if (graphEnabled) // node Q values can only become desynchronized if graph mode enabled
    {
      // Immediately store the recomputed summary stats back to parent node.
      // This has multiple advantages:
      //   - simple (no need to carry this information somewhere else)
      //   - avoids potential concurrency issues with (for example) overlapping iterator seeing different values
      //   - allows the more updated value to be immediately utilized (e.g. by the other overlapping iterator)
      //   - we can ask SelectChildren to refresh edges above, safe in the knowledge that these updates will 
      //     be applied to the parent node here (maintaining correctness of the pure Q).

      Debug.Assert(parentNode.N == childStats.SumN
                || (Engine.Manager.ParamsSelect.CBGPUCTSelectActive && Engine.Manager.ParamsSelect.CBGPUCT_SelectCrossParentNEnabled));

      if (MCGSParamsFixed.RESET_Q_DURING_SELECT_PHASE_FROM_ALL_CHILDREN)
      {
        if (Engine.Manager.ParamsSelect.CBGPUCTBackupActive)
        {
          // No-op: under CBGPUCT, parent.Q is recomputed by ComputeVBar during BackupToNode,
          // and edge.QChild for multi-parent transposition edges is now refreshed proactively
          // in Graph.GatherChildInfoViaChildren (gated on CBGPUCT_CrossParentNEnabled).
        }
        else
        {
          parentNode.ResetQUsingSumWChildrenAndSelf(childStats.SumW, refreshSibling);
        }
      }

      // TODO: update D?
      // TODO: (?) We do not (cannot) update DSum. It is NaN here because it is not computed during gather.
      //           The problem is it would be too expensive to gather because it would be necessary to go the GNode for the value.
      // parentNode.NodeRef.DSum = childStats.SumD;
    }

    if (CapacityAbortNeeded(iterator, path, pathsSet))
    {
      parentNode.ReleaseLock();
      return;
    }

#if DEBUG
    if (Engine.Manager.ParamsSearch.VisitSuboptimalityRejectThreshold is null)
    {
      Debug.Assert(childVisitCounts[..numChildrenToConsider].ToArray().Sum(x => x) == numAttemptedVisits);
    }
#endif

    // Create Sliced versions of our spans to only consider the selected children
    // This will likely allow JIT to elide span bounds checks in the loop
    childVisitCounts = childVisitCounts[..numChildrenToConsider];
    scores = ALSO_COMPUTE_CHILD_SCORES ? scores[..numChildrenToConsider] : default;

    // Loop thru child slots and process any with nonzero visits.
    // Note that we must process visits to any not yet expanded children first,
    // before any possible subtasks are launched on already expanded children.
    // This insures  updates to this node's edges/edge headers are complete before any concurrency.
    int numVisitsRemaining = numAttemptedVisits;
#if DEBUG // Temporarily using conditional compilation due to TensorPrimives versioning issue
    Debug.Assert(TensorPrimitives.Sum(childVisitCounts) == numVisitsRemaining);
#endif

    numVisitsRemaining = ProcessChildren(iterator, path, numAttemptedVisits, pathsSet,
                                         parentNode, in parentPosMG, childVisitCounts,
                                         numChildrenToConsider, numVisitsRemaining,
                                         cpuctMultiplier);

    MCGSParamsFixed.Assert(path.TerminationReason != MCGSPathTerminationReason.NotYetTerminated, "NotYetTerminated");
  }


  /// <summary>
  /// Builds a path that begins at the search root, forcibly descends to a specified inner node
  /// (startNode) along the most-visited ancestor line (at each step choosing the highest-N parent
  /// edge that is strictly closer to the search root), and then continues normal PUCT/CBGPUCT
  /// selection from startNode to a leaf.
  ///
  /// Used by MCGSManager.DoSearchInnerNodes to run "deep rollouts" originating at inner nodes.
  /// Because the resulting path spans searchRoot -> ... -> startNode -> ... -> leaf, the subsequent
  /// backup propagates the leaf evaluation all the way up to the search root, and the neural network
  /// history planes remain continuous (no special handling needed in the evaluator).
  ///
  /// startNode must be an already-evaluated, non-terminal, strict descendant of the search root
  /// (the caller is responsible for filtering); the exploration multiplier in force during the
  /// descent from startNode is iterator.CPUCTMultiplier.
  /// </summary>
  /// <param name="iterator"></param>
  /// <param name="startNode"></param>
  /// <param name="numAttemptedVisits"></param>
  internal void ExtendPathFromInnerNode(MCGSIterator iterator, GNode startNode, int numAttemptedVisits)
  {
    Debug.Assert(numAttemptedVisits > 0);
    Debug.Assert(!startNode.IsNull);
    Debug.Assert(!startNode.IsSearchRoot, "ExtendPathFromInnerNode requires a strict descendant of the search root");

    // Walk up from startNode to the search root, collecting node indices leaf-to-root. At each step
    // follow the highest-N incoming (parent) edge, but only among parents that are strictly closer
    // to the search root. Constraining to strictly-decreasing depth-from-search-root both guarantees
    // we terminate at the search root and rules out cycles from transposition / repetition
    // back-edges (a naive max-N walk could loop: e.g. A's busiest parent is B and B's is A). The
    // acyclic tree parent always qualifies (it sits at depth-1), so a valid choice always exists.
    const int MAX_DEPTH = 512;
    Span<int> nodeIndicesLeafToRoot = stackalloc int[MAX_DEPTH];
    int depth = 0;
    GNode walk = startNode;
    int walkDepth = DepthFromSearchRootOrNegative(startNode);
    if (walkDepth < 0)
    {
      throw new ArgumentException($"DoSearchInnerNodes start node #{startNode.Index.Index} is not a descendant of the search root.");
    }
    while (!walk.IsSearchRoot)
    {
      Debug.Assert(depth < MAX_DEPTH, "ExtendPathFromInnerNode: depth exceeded maximum");
      nodeIndicesLeafToRoot[depth++] = walk.Index.Index;

      // Pick the highest-N parent edge whose parent is strictly closer to the search root.
      GNode chosenParent = default;
      int chosenParentN = -1;
      int chosenParentDepth = walkDepth;
      foreach (GEdge parentEdge in walk.ParentEdges)
      {
        GNode candidate = parentEdge.ParentNode;
        int candidateDepth = DepthFromSearchRootOrNegative(candidate);
        if (candidateDepth >= 0 && candidateDepth < walkDepth && parentEdge.N > chosenParentN)
        {
          chosenParent = candidate;
          chosenParentN = parentEdge.N;
          chosenParentDepth = candidateDepth;
        }
      }

      if (chosenParentN < 0)
      {
        // No rootward parent found (should not happen, since the tree parent is always at depth-1);
        // fall back to the acyclic tree parent.
        chosenParent = Engine.Graph[walk.TreeParentNodeIndex];
        chosenParentDepth = walkDepth - 1;
      }

      walk = chosenParent;
      walkDepth = chosenParentDepth;
    }

    // Allocate the path and seed it with the search root context.
    MCGSPath path = iterator.AllocatedPath(NumInitialSlotsFromNumVisits(startNode.N, numAttemptedVisits));
    path.InnerSearchStartNode = startNode;
    path.InnerSearchStartDepth = depth;
    path.RunningHash = Engine.SearchRootRunningHash;
    if (Engine.NeedsPlySinceLastMove)
    {
      Engine.SearchRootPlySinceLastMove.AsSpan().CopyTo(path.PlySinceLastMove.SquarePlySince);
    }

    // Force descent from the search root down to startNode (root-to-leaf order), recording a
    // single-visit slot for each step so that backup later propagates upward to the search root.
    MGPosition parentPosMG = Engine.SearchRootPosMG;
    GNode parentNode = Engine.SearchRootNode;
    for (int i = depth - 1; i >= 0; i--)
    {
      GNode childNode = Engine.Graph[nodeIndicesLeafToRoot[i]];
      int childIndex = parentNode.IndexOfChildInChildEdges(childNode.Index);
      Debug.Assert(childIndex >= 0, "ExtendPathFromInnerNode: tree-parent child edge not found");
      GEdge childEdge = parentNode.ChildEdgeAtIndex(childIndex);
      AddForcedPrefixVisit(path, parentNode, childIndex, childEdge, ref parentPosMG);
      parentNode = childNode;
    }

    Debug.Assert(path.NumVisitsInPath == depth);
    Debug.Assert(path.LeafVisitRef.ParentChildEdge.ChildNode.Index == startNode.Index);

    // Continue normal selection (PUCT/CBGPUCT) from startNode.
    ExtendPathsRecursively(iterator, path, numAttemptedVisits);
  }


  /// <summary>
  /// Returns the depth of a node from the search root (number of tree-parent edges to reach it),
  /// or -1 if the node is not a descendant of the search root (its tree-parent chain reaches the
  /// graph root first). Walking via the acyclic TreeParentEdge guarantees this terminates; it is
  /// used to keep the inner-node prefix walk moving strictly toward the search root.
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  private int DepthFromSearchRootOrNegative(GNode node)
  {
    const int MAX_DEPTH = 512;
    int depth = 0;
    GNode n = node;
    while (!n.IsSearchRoot)
    {
      if (n.IsGraphRoot || depth >= MAX_DEPTH)
      {
        return -1;
      }
      n = Engine.Graph[n.TreeParentNodeIndex];
      depth++;
    }
    return depth;
  }


  /// <summary>
  /// Records a single forced (non-PUCT) visit along an already-expanded child edge while building
  /// the search-root -> startNode prefix in ExtendPathFromInnerNode. Mirrors the per-visit
  /// bookkeeping of the normal expanded-child descent (position, hashes, edge in-flight, running
  /// hash, ply) but always assigns exactly one visit to the given child.
  /// </summary>
  private void AddForcedPrefixVisit(MCGSPath path, GNode parentNode, int childIndex,
                                    GEdge childEdge, ref MGPosition parentPosMG)
  {
    Debug.Assert(childEdge.Type == GEdgeStruct.EdgeType.ChildEdge);

    MGMove moveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(childEdge.Move, in parentPosMG);
    MGPosition childPos = parentPosMG;
    childPos.MakeMove(moveMG);
    bool moveIrreversible = parentPosMG.IsIrreversibleMove(moveMG, in childPos);

    PosHash64 childHash64 = MGPositionHashing.Hash64(in childPos);
    PosHash96 childHash96 = MGPositionHashing.Hash96(in childPos);

    ref MCGSPathVisit visit = ref path.AddVisit(parentNode, childIndex, in childPos, 1, childHash64, moveIrreversible);
    visit.ParentChildEdge = childEdge;
    GNodeStruct.UpdateEdgeNInFlightForIterator(childEdge, path.IteratorID, 1);

    // Maintain the running multiset hash and ply-since-last-move exactly as PossiblyBranched does
    // for an ordinary descended visit (history resets on an irreversible move).
    if (moveIrreversible)
    {
      path.RunningHash = default;
    }
    path.RunningHash.Add(childHash96);

    if (Engine.NeedsPlySinceLastMove)
    {
      PlySinceLastMoveArray.ApplyMoveWithSwap(ref path.PlySinceLastMove, ref path.PlySinceLastMoveTemp, in moveMG);
    }

    parentPosMG = childPos;
  }

  private static bool CapacityAbortNeeded(MCGSIterator iterator, MCGSPath path, MCGSPathsSet pathsSet)
  {
    if (iterator.IsApproachingMaxPathCapacity) // for example, when a prefetch operation has reached the target
    {
      path.TerminationReason = MCGSPathTerminationReason.Abort;
      if (path.NumVisitsInPath > 0)
      {
        pathsSet.AddPath(path, 0);
      }
      return true;
    }

    return false;
  }


  internal static void CalcPseudotranspositionContribution(GNode parentNode,
                                                           int numPendingVisitsParentNode,
                                                           out double siblingAvgQ,
                                                           out float extraNFromTranspositionAlias)
  {
    siblingAvgQ = 0;
    extraNFromTranspositionAlias = 0;

    if (!NodeIndexSet.IsEligibleForPseudoTranspositionContribution(parentNode))
    {
      return;
    }

    // Get the position hash with move50 and repetition information
    PosHash64WithMove50AndReps hash64WithMove50AndReps
      = MGPositionHashing.Hash64WithMove50AndRepsAdded(parentNode.HashStandalone,
                                                       parentNode.HasRepetitions ? 1 : 0,
                                                       parentNode.NodeRef.Move50Category);

    // Use the new helper method to get transposition stats
    // This handles both direct node index and set of nodes
    (extraNFromTranspositionAlias, siblingAvgQ)
      = parentNode.Graph.GetTranspositionStats(parentNode, parentNode.N + numPendingVisitsParentNode, hash64WithMove50AndReps);

    // Limit sibling contribution to maintain MAX_FRACTION_SIBLING of total effective N
    const float SCALING_TERM = (MCGSParamsFixed.SIBLING_WT_MAX_FRACTION / (1 - MCGSParamsFixed.SIBLING_WT_MAX_FRACTION));
    extraNFromTranspositionAlias = Math.Min(extraNFromTranspositionAlias, SCALING_TERM * (parentNode.N + numPendingVisitsParentNode));
  }


  /// <summary>
  /// Determines the dynamic threshold of visits to a child node,
  /// based off of the ParamsSearchExecution but possibly adjusted if the
  /// graph is large (since paths become longer and parallelism is more beneficial).
  /// </summary>
  int ParallelThresholdToUse
  {
    get
    {
      int threshold = Engine.Manager.ParamsSearch.Execution.SelectOperationParallelThresholdNumVisits; ;// (bigGraph ? 15 : 20) : 9999;
      if (threshold < int.MaxValue)
      {
        if (Engine.SearchRootNode.N > 50_000_000)
        {
          threshold = (int)(threshold * 0.40f);
        }
        else if (Engine.SearchRootNode.N > 15_000_000)
        {
          threshold = (int)(threshold * 0.60f);
        }
        else if (Engine.SearchRootNode.N > 3_000_000)
        {
          threshold = (int)(threshold * 0.80f);
        }
      }
      return threshold;
    }
  }


  private int ProcessChildren(MCGSIterator iterator,
                              MCGSPath path,
                              int numAttemptedVisits,
                              MCGSPathsSet pathsSet,
                              GNode parentNode,
                              in MGPosition parentPosMG,
                              Span<short> childVisitCounts,
                              int numChildrenToConsider,
                              int numVisitsRemaining,
                              float cpuctMultiplier)
  {
    ParamsSearch paramsSearch = iterator.Engine.Manager.ParamsSearch;
    int minRepetitionCountForDraw = paramsSearch.TwofoldDrawEnabled ? 1 : 2;
    bool graphEnabled = paramsSearch.EnableGraph;
    float transpositionStopMinSupportRatio = paramsSearch.PathTranspositionMode == PathMode.PositionAndHistoryEquivalence
                                           ? paramsSearch.TranspositionStopMinSupportRatioPositionAndHistoryMode
                                           : paramsSearch.TranspositionStopMinSupportRatioPositionMode;
    bool parallelEnabled = paramsSearch.Execution.SelectOperationParallelThresholdNumVisits < int.MaxValue;
    int THRESHOLD_PARALLEL = ParallelThresholdToUse;

    bool multipass = numAttemptedVisits >= 3 * THRESHOLD_PARALLEL;

    if (parallelEnabled)
    {
      multipass = true;
    }

    // Loop over child visits and preform first-level processing (create associated MCGSPathVisit, etc.).
    // However do not initiate recursive descent yet, instead tarcking deferred subpaths to process later.
    // This allows the parent lock to be released before commencing recursive descent.
    ListBounded<DeferredSubPath> deferredSubPaths = null;

    // Enforce in-order edge expansion. The edge store requires unexpanded children to be expanded
    // in strictly ascending index order (creating edge k derives its storage block from edge k-1).
    // Compact the unexpanded children into contiguous slots beginning at NumEdgesExpanded, swapping both edge headers.
    int scanLimit = Math.Min(numChildrenToConsider, childVisitCounts.Length);
    int writeSlot = parentNode.NumEdgesExpanded;
    for (int k = parentNode.NumEdgesExpanded; k < scanLimit; k++)
    {
      if (childVisitCounts[k] > 0)
      {
        if (k != writeSlot)
        {
          parentNode.SwapChildEdgeHeaders(writeSlot, k);
          (childVisitCounts[writeSlot], childVisitCounts[k]) = (childVisitCounts[k], childVisitCounts[writeSlot]);
        }
        writeSlot++;
      }
    }


    SelectVisitsEnumerator childVisitScans = new(numChildrenToConsider, parentNode.NumEdgesExpanded, multipass);
    foreach ((SelectVisitsEnumerator.VisitsPhase phase, int childIndex) in childVisitScans)
    {
      if (numVisitsRemaining == 0)
      {
        break;
      }

      int numVisitsThisChild = childVisitCounts[childIndex];
      if (numVisitsThisChild == 0)
      {
        continue;
      }

      if (phase == SelectVisitsEnumerator.VisitsPhase.MultiPassVisitedLaunchParallel && numVisitsThisChild < THRESHOLD_PARALLEL)
      {
        continue;
      }

      if (phase == SelectVisitsEnumerator.VisitsPhase.MultiPassVisitedProcessNonParallel
            && numVisitsThisChild >= THRESHOLD_PARALLEL)
      {
        //        continue; // TODO: can we safely restore this?
      }

      bool isExpanded = childIndex < parentNode.NumEdgesExpanded;

      int numVisitsToAssign = numVisitsThisChild;

      bool canLaunchParallel = multipass
                            && (numVisitsRemaining - numVisitsThisChild) >= 0.5 * THRESHOLD_PARALLEL
                            && numVisitsThisChild >= 1.0 * THRESHOLD_PARALLEL;

      if (phase == SelectVisitsEnumerator.VisitsPhase.MultiPassVisitedLaunchParallel && !canLaunchParallel)
      {
        continue;
      }

      numVisitsRemaining -= numVisitsThisChild;

      // Extract position, move and edge information needed below.
      (EncodedMove move, GEdge childEdge, GNode childNode) = GetChildMoveAndEdge(parentNode, childIndex, isExpanded);
      ChildPositionInfo childPosInfo = ComputeChildPositionInfo(path, in parentPosMG, move, graphEnabled);

      if (isExpanded && childPosInfo.isDrawByRepetitionInCoalesceMode)
      {
        HandleDrawByRepetitionInCoalesceMode(path, pathsSet, childEdge, numVisitsRemaining,
                                             numVisitsThisChild, ref childPosInfo,
                                             childVisitCounts, childIndex);
      }
      else if (isExpanded)
      {
        ProcessExpandedChild(iterator, path, pathsSet, parentNode, childEdge, childNode,
                             numVisitsToAssign, childVisitCounts, childIndex,
                             ref childPosInfo, graphEnabled,
                             transpositionStopMinSupportRatio, numVisitsRemaining,
                             canLaunchParallel, ref deferredSubPaths);
      }
      else // not expanded
      {
        Debug.Assert(!parentNode.EdgeHeadersSpan[childIndex].IsExpanded);

        ProcessUnexpandedChild(iterator, path, pathsSet, parentNode, childIndex,
                               ref childPosInfo, childVisitCounts, numVisitsToAssign,
                               numVisitsRemaining, minRepetitionCountForDraw,
                               graphEnabled, transpositionStopMinSupportRatio, canLaunchParallel, ref deferredSubPaths);
      }
    }

    parentNode.ReleaseLock();

    // Dispatch deferred recursive descents.
    if (deferredSubPaths != null)
    {
      foreach (DeferredSubPath deferredSubPath in deferredSubPaths)
      {
        if (deferredSubPath.LaunchParallel)
        {
          Engine.GetWorkerPool(iterator.IteratorID).SubmitWorkItem(ExtendPathsHelper, new ExtendPathsWorkerInfo(this, iterator, deferredSubPath.SubPath, deferredSubPath.NumVisits));
        }
        else
        {
          ExtendPathsRecursively(iterator, deferredSubPath.SubPath, deferredSubPath.NumVisits);
        }
      }

      // Return to pool
      ReturnDeferredSubPathsList(deferredSubPaths);
    }

    return numVisitsRemaining;
  }


  /// <summary>
  /// Helper to extract move and edge information for a child.
  /// </summary>
  private static (EncodedMove move, GEdge childEdge, GNode childNode) GetChildMoveAndEdge(GNode parentNode, int childIndex, bool isExpanded)
  {
    EncodedMove move;
    GEdge childEdge = default;
    GNode childNode = default;

    if (isExpanded)
    {
      childEdge = parentNode.ChildEdgeAtIndex(childIndex);
      if (!childEdge.Type.IsTerminal())
      {
        childNode = childEdge.ChildNode;
      }
      move = childEdge.Move;
    }
    else
    {
      Debug.Assert(!parentNode.EdgeHeadersSpan[childIndex].IsUnintialized);
      move = parentNode.EdgeHeadersSpan[childIndex].Move;
    }

    return (move, childEdge, childNode);
  }


  /// <summary>
  /// Container for child position information to avoid repeated calculations.
  /// </summary>
  private ref struct ChildPositionInfo
  {
    public MGMove moveMG;
    public MGPosition childPos;
    public PosHash64 childPositionHash64;
    public PosHash96 childPositionHash96;
    public PosHash96MultisetFinalized childPositionAndSequenceHashFinalized;
    public bool wasIrreversibleMove;
    public bool positionDuplicate;
    public bool isDrawByRepetitionInCoalesceMode;
  }


  /// <summary>
  /// Computes all position-related information for a child node.
  /// </summary>
  private ChildPositionInfo ComputeChildPositionInfo(MCGSPath path, in MGPosition parentPosMG,
                                                      EncodedMove move, bool graphEnabled)
  {
    ChildPositionInfo info = new();

    MGMove moveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, in parentPosMG);
    info.moveMG = moveMG;
    info.childPos = parentPosMG;
    info.childPos.MakeMove(moveMG);
    info.wasIrreversibleMove = parentPosMG.IsIrreversibleMove(moveMG, in info.childPos);

    info.childPositionHash64 = MGPositionHashing.Hash64(in info.childPos);
    PosHash64WithMove50AndReps posHash64WithMove50AndReps = MGPositionHashing.Hash64WithMove50AndRepsAdded(
      info.childPositionHash64, info.childPos.RepetitionCount, info.childPos.Move50Category);
    info.childPositionHash96 = MGPositionHashing.Hash96(in info.childPos);

    if (graphEnabled && path.PathMode == PathMode.PositionAndHistoryEquivalence)
    {
      // Update running hash with this position (but resetting history if was irreversible).
      info.childPositionAndSequenceHashFinalized = info.wasIrreversibleMove
        ? new PosHash96MultisetFinalized(info.childPositionHash96.High, info.childPositionHash96.Low)
        : path.RunningHash.Finalized(info.childPositionHash96);
    }
    else
    {
      // Replace hash code with standalone hash so a all edges map to shared node.
      // Use extra available slot with another hash to reduce hash collision probability.
      int extraHash = HashCode.Combine(info.childPos.A, info.childPos.B, info.childPos.C, info.childPos.D);
      info.childPositionAndSequenceHashFinalized = new PosHash96MultisetFinalized((uint)extraHash, info.childPositionHash64.Hash);
    }

    // Update repetition count in position (including considering prehistory)
    info.positionDuplicate = path.HashFoundInHistoryOrPrehistory(info.childPositionHash64);
    info.isDrawByRepetitionInCoalesceMode = info.positionDuplicate && path.PathMode == PathMode.PositionEquivalence;
    info.childPos.RepetitionCount = (byte)(info.positionDuplicate ? 1 : 0);

    return info;
  }


  /// <summary>
  /// Processes an already-expanded child edge.
  /// Returns true if handled (caller should continue to next child).
  /// </summary>
  private void ProcessExpandedChild(MCGSIterator iterator, MCGSPath path, MCGSPathsSet pathsSet,
                                    GNode parentNode, GEdge childEdge, GNode childNode,
                                    int numVisitsThisChild, Span<short> childVisitCounts, int childIndex,
                                    ref ChildPositionInfo childPosInfo,
                                    bool graphEnabled, float transpositionStopMinSupportRatio,
                                    int numVisitsRemaining, bool canLaunchParallel,
                                    ref ListBounded<DeferredSubPath> deferredSubPaths)
  {
    // Create a new visit entry in the path for this child.
    ref MCGSPathVisit newVisit = ref AddVisitFromChildPosInfo(path, parentNode, childIndex,
                                                              ref childPosInfo, numVisitsThisChild);
    newVisit.ParentChildEdge = childEdge;
    GNodeStruct.UpdateEdgeNInFlightForIterator(childEdge, path.IteratorID, numVisitsThisChild);

    if (childEdge.Type.IsTerminal())
    {
      Debug.Assert(!childPosInfo.isDrawByRepetitionInCoalesceMode);

      // Revisit a terminal edge --> repeat terminal eval, end descent.
      TerminatePathAndAddToSet(path, pathsSet, MCGSPathTerminationReason.TerminalEdge,
                               numVisitsRemaining, numVisitsThisChild, childPosInfo.childPositionHash96,
                               childPosInfo.wasIrreversibleMove, childVisitCounts, childIndex, false,
                               childPosInfo.moveMG);
      return;
    }

    if (childPosInfo.positionDuplicate
     && !haveWarnedDuplicate
     && Engine.Manager.ParamsSearch.PathTranspositionMode != PathMode.PositionEquivalence)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Internal error: found duplicate but path continued nonterminal " + childEdge);
      haveWarnedDuplicate = true;
    }

    Debug.Assert(childEdge.Type == GEdgeStruct.EdgeType.ChildEdge);

    if (childNode.Terminal.IsTerminal())
    {
      // Revisit a terminal --> repeat terminal eval, end descent.
      TerminatePathAndAddToSet(path, pathsSet, MCGSPathTerminationReason.Terminal,
                              numVisitsRemaining, numVisitsThisChild, childPosInfo.childPositionHash96,
                              childPosInfo.wasIrreversibleMove, childVisitCounts, childIndex, false,
                              childPosInfo.moveMG);
    }
    else if (childPosInfo.isDrawByRepetitionInCoalesceMode)
    {
      TerminatePathAndAddToSet(path, pathsSet, MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode,
                              numVisitsRemaining, numVisitsThisChild, childPosInfo.childPositionHash96,
                              childPosInfo.wasIrreversibleMove, childVisitCounts, childIndex, true,
                              childPosInfo.moveMG);
    }
    else if (IsTranspositionSufficientN(graphEnabled, transpositionStopMinSupportRatio,
                                        childEdge.NInFlightForIterator(iterator.IteratorID), childEdge.N, childNode,
                                        iterator.DisableTranspositionSufficiencyStop))
    {
      // Already expanded, connects to a transposition node with already sufficient N, stop descent.
      //  Revisit transposition node with sufficient visits --> extract evaluation, end descent.
      TerminatePathAndAddToSet(path, pathsSet, MCGSPathTerminationReason.TranspositionLinkNodeSufficientN,
                              numVisitsRemaining, numVisitsThisChild, childPosInfo.childPositionHash96,
                              childPosInfo.wasIrreversibleMove, childVisitCounts, childIndex, false,
                              childPosInfo.moveMG);
    }
    else
    {
      if (childNode.N > 0)
      {
        // Revisit child already visited --> mark in flight, continue descent.
        MCGSPath subPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                                 childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                                 childPosInfo.moveMG);
        // Add to list to be continued.
        deferredSubPaths ??= GetDeferredSubPathsList();
        deferredSubPaths.Add(new DeferredSubPath(subPath, numVisitsThisChild, canLaunchParallel));
        childVisitCounts[childIndex] = 0;
      }
      else
      {
        // CASE 3b: Abort; revisited child already visited but only in flight from this batch (never yet evaluated).
        CalcUnvisitedLeafMethod(iterator, childEdge, numVisitsThisChild, out int numVisitsToAccept, out MCGSPathTerminationReason terminationReason);
        path.TerminationReason = terminationReason;
        MCGSPath newPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                                 childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                                 childPosInfo.moveMG);
        pathsSet.AddPath(newPath, numVisitsToAccept);
        childVisitCounts[childIndex] = 0;
      }
    }
  }

  /// <summary>
  /// Helper method to add a visit to the path using ChildPositionInfo.
  /// </summary>
  private static ref MCGSPathVisit AddVisitFromChildPosInfo(MCGSPath path, GNode parentNode, int indexInParent,
                                                            ref ChildPositionInfo childPosInfo, int numVisits)
  {
    return ref path.AddVisit(parentNode, indexInParent,
                             in childPosInfo.childPos, numVisits,
                             childPosInfo.childPositionHash64,
                             childPosInfo.wasIrreversibleMove);
  }


  /// <summary>
  /// Processes an unexpanded child (creating new node or terminal edge).
  /// Returns true if handled (caller should continue to next child).
  /// </summary>
  private bool ProcessUnexpandedChild(MCGSIterator iterator, MCGSPath path, MCGSPathsSet pathsSet,
                                      GNode parentNode, int childIndex,
                                      ref ChildPositionInfo childPosInfo,
                                      Span<short> childVisitCounts, int numVisitsThisChild,
                                      int numVisitsRemaining, int minRepetitionCountForDraw,
                                      bool graphEnabled, float transpositionStopMinSupportRatio,
                                      bool canLaunchParallel,
                                      ref ListBounded<DeferredSubPath> deferredSubPaths)
  {
    MGMoveList childMoves = MGMoveGen.GeneratedMoves(in childPosInfo.childPos);

    // Determine if game result can be immediately determined (various mate and draw conditions).
    const bool possiblyUseTablebase = true;
    (GameResult result, float v, float d, bool wasDrawByRepetition) resultInfo =
      path.CalcPathTerminationFromUnexpandedLeaf(minRepetitionCountForDraw, in childPosInfo.childPos, childMoves, possiblyUseTablebase);

    // In coalesce mode we must not create terminal draw edges for repetitions
    // because other visits via other paths may not be draws.
    if (resultInfo.result != GameResult.Unknown && !childPosInfo.isDrawByRepetitionInCoalesceMode)
    {
      return DoTerminalUnexpandedChild(path, pathsSet, parentNode, childIndex,
                                       ref childPosInfo, childMoves, childVisitCounts,
                                       numVisitsThisChild, numVisitsRemaining, resultInfo);
    }

    return DoNonTerminalUnexpandedChild(iterator, path, pathsSet, parentNode, childIndex,
                                        ref childPosInfo, ref childMoves,
                                        childVisitCounts, numVisitsThisChild,
                                        childPosInfo.isDrawByRepetitionInCoalesceMode, numVisitsRemaining,
                                        graphEnabled, transpositionStopMinSupportRatio,
                                        canLaunchParallel, ref deferredSubPaths);
  }


  /// <summary>
  /// Handles creation of terminal edge for unexpanded child.
  /// </summary>
  private bool DoTerminalUnexpandedChild(MCGSPath path, MCGSPathsSet pathsSet, GNode parentNode,
                                            int childIndex,
                                            ref ChildPositionInfo childPosInfo, MGMoveList childMoves,
                                            Span<short> childVisitCounts, int numVisitsThisChild,
                                            int numVisitsRemaining,
                                            (GameResult result, float v, float d, bool wasDrawByRepetition) resultInfo)
  {
    Debug.Assert(!float.IsNaN(resultInfo.v) && !float.IsNaN(resultInfo.d));

    // Create a new visit entry in the path for this child.
    ref MCGSPathVisit newVisit = ref AddVisitFromChildPosInfo(path, parentNode, childIndex,
                                                              ref childPosInfo, numVisitsThisChild);

    // CASE 4a: (with terminal edges enabled) known game result (checkmate, stalemate, draw by insufficient material, etc.)
    // Create terminal edge (no associated child node).

    bool propagateAsDraw = resultInfo.v == 0;
    GEdge newEdge = path.Graph.AddNewTerminalEdge(parentNode, childIndex, resultInfo.v, resultInfo.d,
                                                   numVisitsThisChild, propagateAsDraw);
    newVisit.ParentChildEdge = newEdge;
    GNodeStruct.UpdateEdgeNInFlightForIterator(newEdge, path.IteratorID, numVisitsThisChild);

    path.TerminationReason = MCGSPathTerminationReason.TerminalEdge;

    MCGSPath newPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                             childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                             childPosInfo.moveMG);
    newVisit.MovesList = childMoves;
    pathsSet.AddPath(newPath, numVisitsThisChild);
    childVisitCounts[childIndex] = 0;
    return true;
  }


  /// <summary>
  /// Handles creation of new node or link to existing transposition node.
  /// </summary>
  private bool DoNonTerminalUnexpandedChild(MCGSIterator iterator, MCGSPath path, MCGSPathsSet pathsSet,
                                            GNode parentNode, int childIndex,
                                            ref ChildPositionInfo childPosInfo, ref MGMoveList childMoves,
                                            Span<short> childVisitCounts, int numVisitsThisChild,
                                            bool isDrawByRepetitionInCoalesceMode,
                                            int numVisitsRemaining, bool graphEnabled,
                                            float transpositionStopMinSupportRatio,
                                            bool canLaunchParallel,
                                            ref ListBounded<DeferredSubPath> deferredSubPaths)
  {
    // CASE 4b: create a new node, or link to existing
    (GEdge childEdge, bool wasCollision) =
      path.Graph.AddEdgeToNewOrExistingNode(parentNode, childIndex, in childPosInfo.childPos,
                                            childPosInfo.childPositionHash64,
                                            childPosInfo.childPositionAndSequenceHashFinalized,
                                            childMoves,
                                            out bool wasCreated, out GNode standaloneTranspositionNode,
                                            true);

    // CB-GPUCT: when CBGPUCT_CrossParentNFraction > 0, seed edge.QChild from child.Q
    // for transposition links so that ComputeVBar / ScoreCalc do not read an invalid
    // -edge.Q=0 before the first BackupToEdge.  edge.N stays at 0 (its per-edge default);
    // any cross-parent contribution is folded in on the fly by ScoreCalc via the
    // blend in CBGPUCT_CrossParentNFraction.  Terminal edges have no destination node,
    // so we skip them.
    ParamsSelect ps = iterator.Engine.Manager.ParamsSelect;
    if (ps.CBGPUCTSelectActive && ps.CBGPUCT_SelectCrossParentNEnabled && !wasCollision
        && childEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
    {
      childEdge.QChild = childEdge.ChildNode.Q;
    }


    if (childPosInfo.isDrawByRepetitionInCoalesceMode)
    {
      if (wasCreated)
      {
        // A new node was created that the draw by repetition edge connects to.
        // However child node will remain unvisited (N=0, V and Q = NaN).
        childEdge.ChildNode.NodeRef.SetQNaN();
      }

      HandleDrawByRepetitionInCoalesceMode(path, pathsSet, childEdge, numVisitsRemaining,
                                           numVisitsThisChild, ref childPosInfo,
                                           childVisitCounts, childIndex);

      return true;
    }

    // Create a new visit entry in the path for this child.
    ref MCGSPathVisit newVisit = ref AddVisitFromChildPosInfo(path, parentNode, childIndex,
                                                              ref childPosInfo, numVisitsThisChild);

    if (wasCollision && !isDrawByRepetitionInCoalesceMode)
    {
      path.TerminationReason = MCGSPathTerminationReason.PiggybackPendingNNEval;
      newVisit.ParentChildEdge = childEdge;
      MCGSPath newPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                               childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                               childPosInfo.moveMG);
      pathsSet.AddPath(newPath, 1);
      GNodeStruct.UpdateEdgeNInFlightForIterator(childEdge, path.IteratorID, numVisitsThisChild);
      childVisitCounts[childIndex] = 0;
      return true;
    }

    Debug.Assert(!childPosInfo.positionDuplicate
               || childEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn
               || Engine.Manager.ParamsSearch.PathTranspositionMode == PathMode.PositionEquivalence);

    // Update the visit edge now that we have created a child.
    newVisit.ParentChildEdge = childEdge;
    GNode childNode = childEdge.ChildNode;

    if (wasCreated)
    {
      return DoNewlyCreatedNode(path, pathsSet, childEdge, childNode, standaloneTranspositionNode,
                                   ref newVisit.MovesList, ref childPosInfo, childMoves, childVisitCounts, childIndex,
                                   numVisitsThisChild, numVisitsRemaining, isDrawByRepetitionInCoalesceMode);
    }
    else
    {
      bool handled = DoTranspositionLink(iterator, path, pathsSet, childEdge, childNode,
                                         ref childPosInfo, childVisitCounts, childIndex,
                                         numVisitsThisChild, numVisitsRemaining, graphEnabled,
                                         transpositionStopMinSupportRatio);

      if (!handled)
      {
        // DoTranspositionLink signaled that this needs recursive descent.
        // Add to deferred subpaths so it's processed after parent lock is released.
        MCGSPath splitPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                                   childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                                   childPosInfo.moveMG);
        deferredSubPaths ??= GetDeferredSubPathsList();
        deferredSubPaths.Add(new DeferredSubPath(splitPath, numVisitsThisChild, canLaunchParallel));
        childVisitCounts[childIndex] = 0;
        return true;
      }

      return handled;
    }
  }


  /// <summary>
  /// Handles a newly created node (either requiring NN eval or copying from transposition).
  /// </summary>
  private bool DoNewlyCreatedNode(MCGSPath path, MCGSPathsSet pathsSet, GEdge childEdge,
                                     GNode childNode, GNode standaloneTranspositionNode,
                                     ref MGMoveList movesList, ref ChildPositionInfo childPosInfo, MGMoveList childMoves,
                                     Span<short> childVisitCounts, int childIndex,
                                     int numVisitsThisChild, int numVisitsRemaining,
                                     bool isDrawByRepetitionInCoalesceMode)
  {
    Debug.Assert(!(isDrawByRepetitionInCoalesceMode && childNode.Terminal.IsTerminal()));

    int numToAccept = (childNode.Terminal.IsTerminal() || isDrawByRepetitionInCoalesceMode) ? numVisitsThisChild : 1;

    if (!childPosInfo.positionDuplicate
     && !standaloneTranspositionNode.IsNull
     && standaloneTranspositionNode.IsEvaluated)
    {
      // The NN evaluation (value, policy, etc.) will later be copied from the standaloneTranspositionNode.
      // TODO: This was arbitrarily choosen as first node, in the sibling set,
      //       in theory we could pick any one which was evaluated
      //       or maybe take an average value/policy across them
      path.TerminationReason = MCGSPathTerminationReason.TranspositionCopyValues;
      // Potentially transposition node is from another graph (opponent graph reuse)
      // We must copy everything here (including policy) and not remember
      // reference to this foreign node.

      bool copyPolicyImmediate = standaloneTranspositionNode.Graph != Engine.Graph;
      Engine.Graph.CopyNodeValues(0, standaloneTranspositionNode, childEdge.ChildNode, copyPolicyImmediate);
    }
    else
    {
      // Visit to child node just created.
      // Terminate descent (will result in neural network evaluation unless terminal). Accept only one visit.
      path.TerminationReason = childNode.Terminal.IsTerminal()
        ? MCGSPathTerminationReason.Terminal
        : MCGSPathTerminationReason.PendingNeuralNetEval;
    }

    MCGSPath newPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                             childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                             childPosInfo.moveMG);
    pathsSet.AddPath(newPath, numToAccept);
    childVisitCounts[childIndex] = 0;
    movesList = childMoves;
    GNodeStruct.UpdateEdgeNInFlightForIterator(childEdge, path.IteratorID, numVisitsThisChild);
    return true;
  }


  /// <summary>
  /// Handles link to existing transposition node.
  /// </summary>
  private bool DoTranspositionLink(MCGSIterator iterator, MCGSPath path, MCGSPathsSet pathsSet,
                                   GEdge childEdge, GNode childNode, ref ChildPositionInfo childPosInfo,
                                   Span<short> childVisitCounts, int childIndex,
                                   int numVisitsThisChild, int numVisitsRemaining,
                                   bool graphEnabled, float transpositionStopMinSupportRatio)
  {
    GNodeStruct.UpdateEdgeNInFlightForIterator(childEdge, path.IteratorID, numVisitsThisChild);

    if (childPosInfo.positionDuplicate
     && !haveWarnedDuplicate
     && Engine.Manager.ParamsSearch.PathTranspositionMode != PathMode.PositionEquivalence)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Internal error: found duplicate but path continued nonterminal " + childEdge);
      haveWarnedDuplicate = true;
    }

    if (childEdge.ChildNode.N == 0)
    {
      CalcUnvisitedLeafMethod(iterator, childEdge, numVisitsThisChild, out int numVisitsToAccept, out MCGSPathTerminationReason terminationReason);
      path.TerminationReason = terminationReason;
      MCGSPath newVisitPiggy = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                                     childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                                     childPosInfo.moveMG);
      pathsSet.AddPath(newVisitPiggy, numVisitsToAccept);
    }
    else if (IsTranspositionSufficientN(graphEnabled, transpositionStopMinSupportRatio,
                                        childEdge.NInFlightForIterator(iterator.IteratorID), 
                                        childEdge.N, childNode, iterator.DisableTranspositionSufficiencyStop))
    {
      // Sufficient visit transposition node --> create edge, extract evaluation, end descent.
      // It must have nonzero visits and thus be able to satisfy our needs without going deeper
      path.TerminationReason = MCGSPathTerminationReason.TranspositionLinkNodeSufficientN;
      MCGSPath newPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                               childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                               childPosInfo.moveMG);
      pathsSet.AddPath(newPath, numVisitsThisChild);
    }
    else
    {
      // Was transposition node but not sufficient visits to stop.
      // Need to continue descent, but DEFER this so it can be processed after parent lock is released.
      // Return false to signal that this needs to be added to deferredSubPaths.
      return false;
    }

    childVisitCounts[childIndex] = 0;
    return true;
  }


  /// <summary>
  /// Helper to terminate a path and add it to the pathsSet with common cleanup.
  /// </summary>
  private void TerminatePathAndAddToSet(MCGSPath path, MCGSPathsSet pathsSet,
                                       MCGSPathTerminationReason terminationReason,
                                       int numVisitsRemaining, int numVisitsThisChild,
                                       PosHash96 childPositionHash96, bool wasIrreversibleMove,
                                       Span<short> childVisitCounts, int childIndex,
                                       bool isDrawByRepetitionInCoalesceMode,
                                       MGMove moveMG)
  {
    path.TerminationReason = terminationReason;
    MCGSPath newPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                             childPositionHash96, wasIrreversibleMove,
                                             moveMG);
    pathsSet.AddPath(newPath, numVisitsThisChild);
    childVisitCounts[childIndex] = 0;
  }


  /// <summary>
  /// Helper to handle draw by repetition in coalesce mode - terminates path and updates edge.
  /// </summary>
  private void HandleDrawByRepetitionInCoalesceMode(MCGSPath path, MCGSPathsSet pathsSet,
                                                    GEdge childEdge, int numVisitsRemaining,
                                                    int numVisitsThisChild, ref ChildPositionInfo childPosInfo,
                                                    Span<short> childVisitCounts,
                                                    int childIndex)
  {
    //ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Draw by repetition detected in coalesce mode on edge {childEdge}");
    ref MCGSPathVisit newVisit = ref AddVisitFromChildPosInfo(path, childEdge.ParentNode, childIndex,
                                                              ref childPosInfo, numVisitsThisChild);
    newVisit.ParentChildEdge = childEdge;

    path.TerminationReason = MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode;
    MCGSPath newPath = path.PossiblyBranched(numVisitsRemaining, numVisitsThisChild,
                                             childPosInfo.childPositionHash96, childPosInfo.wasIrreversibleMove,
                                             childPosInfo.moveMG);
    pathsSet.AddPath(newPath, numVisitsThisChild);

    childVisitCounts[childIndex] = 0;
    GNodeStruct.UpdateEdgeNInFlightForIterator(childEdge, path.IteratorID, numVisitsThisChild);
  }


  static void CalcUnvisitedLeafMethod(MCGSIterator iterator,
                                        GEdge childEdge,
                                        int numVisitsThisPath,
                                        out int numVisitsToAccept,
                                        out MCGSPathTerminationReason terminationReason)
  {
    int numInFlightAllIteratorsThisEdge = childEdge.NumInFlight0 + childEdge.NumInFlight1;
    bool weAreOnlyInFlightOnThisEdge = numInFlightAllIteratorsThisEdge == numVisitsThisPath;
    if (!weAreOnlyInFlightOnThisEdge)
    {
      // We (or the other iterator) are already visiting along this edge.
      // Since there is only one leaf available, don't allow this or subsequent to visit again.
      terminationReason = MCGSPathTerminationReason.Abort;
      numVisitsToAccept = 0;
    }
    else if (childEdge.ChildNode.IsEvaluated)
    {
      // Node found already evaluated, possibly due to:
      //   - a TranspsitionCopyValues from another node
      //   - like due to a prior prefetch operation.
      // We just use this evaluation without scheduling an NN evaluation.
      terminationReason = MCGSPathTerminationReason.AlreadyNNEvaluated;
      numVisitsToAccept = childEdge.ChildNode.Terminal.IsTerminal() ? numVisitsThisPath : 1;
    }
    else
    {
      // This node must be in the process of being evaluated along a different parent edge.
      // NOTE: this assumption holds because MCGS search never gratituous expands edges.
      (int numInFlightAllEdgesAllIterators, int numInFlightAllEdgesThisIterator) = NumInFlightAllEdgesToNode(iterator.IteratorID, childEdge.ChildNode);
      int numInFlightAllEdgesNotThisIterator = numInFlightAllEdgesAllIterators - numInFlightAllEdgesThisIterator;

      if (numInFlightAllEdgesNotThisIterator > 0)
      {
        // If the other iterator is the one that triggered the valuation
        // then it is not safe to piggyback becuase we cannot be sure
        // the evaluation will be complete by the time we do our backup.
        // Therefore abort.
        terminationReason = MCGSPathTerminationReason.Abort;
        numVisitsToAccept = 0;
      }
      else
      {
        // The evaluation will be performed in this batch and available for piggybacking.
        terminationReason = MCGSPathTerminationReason.PiggybackPendingNNEval;
        numVisitsToAccept = 1;
      }
    }
  }


  #region Helper methods


  /// <summary>
  /// Returns if a visit to a transposition node has N sufficienty large to be
  /// used as the immediate backup value (suppressing further descent).
  /// 
  /// Note that sufficiency is judged relative to a target which includes
  /// the initial edgeN plus the total number of visits in flight from this iterator
  /// (even within a single iterator we may have multiple paths that reconverge on this node).
  /// 
  /// This target may then be scaled by a multiplier to encourage deeper exploration
  /// of frequently visited nodes.
  /// </summary>
  /// <returns></returns>
  static bool IsTranspositionSufficientN(bool graphEnabled, float transpositionStopMinSupportRatio,
                                         int numVisitsTotalThisIterator, int edgeN, GNode childNode,
                                         bool disableTranspositionSufficiencyStop)
  {
    if (disableTranspositionSufficiencyStop)
    {
      // Inner-node deep rollouts bypass the sufficiency stop so descent continues through
      // well-visited transposition nodes to a true frontier leaf or terminal.
      return false;
    }

    if (graphEnabled)
    {
      // This could be a transposition node.
      // In graph mode we hope/expect the subtrees can often reflect more visits
      // than direclty requested from each individaul node (since serach effort is shared).
      // Based on some configurable scaling factor we decide if this subtree is
      // "already big enough" that we can stop descent here and just use the subtree value as is.
      int scaledTargetN = (int)MathF.Round(transpositionStopMinSupportRatio * (edgeN + numVisitsTotalThisIterator));

      if (MCGSParamsFixed.REDESCENT_MUTIPLIER_ADJUST && childNode.N < 30)
      {
        // Adjust scaling so that for smaller absolute N we require a higher redescent mutiplier,
        // reflecting that absolute uncertainty is higher for smaller N.
        const double REDESCENT_MULTIPLIER_ADJ_FACTOR = 2;
        double scale = 1.0f + (REDESCENT_MULTIPLIER_ADJ_FACTOR / Math.Sqrt(childNode.N));
        scaledTargetN = (int)(scaledTargetN * scale);
      }

      return childNode.N > scaledTargetN;
    }
    else
    {
      // This search may be using NonGraphModeEnableTranspositionCopy
      // which could have caused some exploratory visits to already have been processed
      // sent to the child even though they were not "on policy" and are not
      // reflected in the N for the edge. 
      // If so, always just reuse this value.
      return childNode.N > edgeN + numVisitsTotalThisIterator;
    }
  }


  /// <summary>
  /// Returns a hint of the suggested number of initial slots to allocate within MCGSPath
  /// for a given number of visits traveling along this path.
  /// 
  /// Better sizing reduces fragmentation within slots leading to potentially lower memory usage
  /// and even more important better memory locality and performance (the impact is surprisingly large).
  /// </summary>
  /// <param name="numVisits"></param>
  /// <param name="numVisitsAttempted"></param>
  /// <returns></returns>
  internal static int? NumInitialSlotsFromNumVisits(int nodeN, int numVisitsAttempted)
  {
    const int QUANTUM = ArraySegmentPool<MCGSPathVisit>.GROWTH_QUANTUM;

    // For low parent N, we know the deepest possible path extension has low depth.
    return Math.Min(nodeN + 1, QUANTUM);
  }



  internal static double QWhenNoChildren(GNode parentNode, ParamsSelect paramsSelect)
  {
    double parentSumPVisited = 0;
    foreach (GEdge edge in parentNode.ChildEdgesExpanded)
    {
      parentSumPVisited += edge.P;
    }

    return paramsSelect.CalcQWhenNoChildren(parentNode.IsSearchRoot, parentNode.Q, parentSumPVisited);
  }


  internal const bool RPO_CHOOSES_NEW_CHILDREN = false;

  public static int NumExploratory = 0;


  bool haveWarnedDuplicate = false; // TODO: remove after debugging complete?

  internal static readonly LiveStats statsReject = MCGSParamsFixed.LOG_LIVE_STATS ? new("PUCTRejects", 4f) : null;

  /// <summary>
  /// If the child edge is in the process of being evaluated by this iterator in this batch.
  /// </summary>
  /// <param name="iteratorID"></param>
  /// <param name="childEdge"></param>
  /// <returns></returns>
  private static (int numInFlightAllIterators, int numInFlightThisIterator) NumInFlightAllEdgesToNode(int iteratorID, GNode node)
  {
    int numInFlightAllIterators = 0;
    int numInFlightThisIterator = 0;
    using (new NodeLockBlock(node))
    {
      foreach (GEdge parentEdge in node.ParentEdges)
      {
        numInFlightThisIterator += parentEdge.NInFlightForIterator(iteratorID);
        numInFlightAllIterators += parentEdge.NInFlightForIterator(0)
                                 + parentEdge.NInFlightForIterator(1);
      }
    }

    return (numInFlightAllIterators, numInFlightThisIterator);
  }




  private static int ApplyPUCTSuboptimalityThreshold(MCGSPath path,
                                                    int numAttemptedVisits,
                                                    int numAcceptedVisits, Span<short> childVisitCounts)
  {
    {
#if DEBUG
      {
        int sumSelectedVisits = 0;
        foreach (short s in childVisitCounts)
        {
          sumSelectedVisits += s;
        }
        Debug.Assert(sumSelectedVisits == numAcceptedVisits);
      }
#endif

      statsReject?.Add(numAttemptedVisits, numAttemptedVisits - numAcceptedVisits);

      if (numAcceptedVisits < numAttemptedVisits)// && path.numSlotsUsed > 0)
      {
        // Backout the number of visits that were not selected of the path visits above.
        int delta = numAcceptedVisits - numAttemptedVisits;
        foreach (MCGSPathVisitMember pp in path.PathVisitsLeafToRoot)
        {
          ref MCGSPathVisit pathVisit = ref pp.PathVisitRef;
          Interlocked.Add(ref pathVisit.NumVisitsAttemptedPendingBackup, delta);
          Interlocked.Add(ref pathVisit.NumVisitsAttempted, delta);
          GNodeStruct.UpdateEdgeNInFlightForIterator(pathVisit.ParentChildEdge, path.IteratorID, delta);
        }
        numAttemptedVisits = numAcceptedVisits;
      }
    }

    return numAttemptedVisits;
  }

  internal static readonly LiveStats fiftyMoveCounterDraw = MCGSParamsFixed.LOG_LIVE_STATS ? new LiveStats("50 move rule, draw(99)", 1) : null;


  #region Helper methods for pooling

  /// <summary>
  /// Gets a ListBounded from the pool (or creates a new one if pool is empty).
  /// </summary>
  private static ListBounded<DeferredSubPath> GetDeferredSubPathsList()
  {
    deferredSubPathsPool ??= new DefaultObjectPool<ListBounded<DeferredSubPath>>(new DeferredSubPathListPolicy(), maximumRetained: 128);
    return deferredSubPathsPool.Get();
  }

  /// <summary>
  /// Returns a ListBounded to the pool for reuse.
  /// </summary>
  private static void ReturnDeferredSubPathsList(ListBounded<DeferredSubPath> list)
  {
    deferredSubPathsPool.Return(list);
  }

  #endregion

  /// <summary>
  /// State struct to avoid closure allocation when launching parallel recursive descent tasks.
  /// </summary>
  public readonly record struct ExtendPathsWorkerInfo(MCGSSelect Selector, MCGSIterator Iterator, MCGSPath SubPath, int NumVisits);

  /// <summary>
  /// Static helper to call ExtendPathsRecursively with state to avoid closure allocation.
  /// </summary>
  private static void ExtendPathsHelper(ExtendPathsWorkerInfo info)
  {
    info.Selector.ExtendPathsRecursively(info.Iterator, info.SubPath, info.NumVisits);
  }

  #endregion
}
