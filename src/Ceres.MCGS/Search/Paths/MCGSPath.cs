# region License notice
/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/
# endregion

# region Using directives

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Search.Phases.Evaluation;
using Ceres.MCGS.Search.Strategies;

# endregion

namespace Ceres.MCGS.Search.Paths;

/// <summary>
/// Represents a path through connected nodes in the MCGS graph (typically arising during search).
/// 
/// NOTE: (1) The method PossiblyBranched explicitly copies
///           over many of the fields from one path to another.
///           Therefore additions to the field list likely require remediations there.
///       (2) Similarly, the ResetFields method explicitly resets all fields
///           and needs to be updated for any changes to fields.
/// </summary>
public partial class MCGSPath : IEquatable<MCGSPath>, IComparable<MCGSPath>
{
  #region Fields

  /// <summary>
  /// The parent iterator which is processing this path.
  /// </summary>
  public readonly MCGSIterator Iterator;

  /// <summary>
  /// Mode to be used when processing transpositions.
  /// </summary>
  public readonly PathMode PathMode;

  /// <summary>
  /// Unique identifying ID (used for debugging)    
  /// </summary>
  public int PathID;

  /// <summary>
  /// A reference to a segment of an array containing MCGSPathVisit elements associated with this path.
  /// </summary>
  internal ArraySegmentRef<MCGSPathVisit> slots;

  /// <summary>
  /// The parent path from which this path branched off.
  /// </summary>
  internal MCGSPath parent;

  /// <summary>
  /// The index of the slow within the parent which was the 
  /// visit before the branch to this path occurred.
  /// </summary>
  internal int parentSlotIndexLastUsedThisSegment;

  /// <summary>
  /// The number of MCGSPathVisit slots that are uniquely captured by this path.
  /// </summary>
  internal int numSlotsUsed;

  /// <summary>
  /// Maximum Q suboptimality across all visits seen so far
  /// (comparing the current Q of the chosen child versus the parent Q).
  /// </summary>
  public float MaxQSubOptimality;

  /// <summary>
  /// The reason triggering this path to be terminated.
  /// </summary>
  public MCGSPathTerminationReason TerminationReason;

  /// <summary>
  /// Information about the termination (leaf) evaluation
  /// (used to transfer between select/evaluate phases and backup phase).
  /// </summary>
  public SelectTerminationInfo TerminationInfo;

  /// <summary>
  /// A running multiset hash over all positions seen so far
  /// (since last irreversible move).
  /// </summary>
  public PosHash96MultisetRunning RunningHash;

  /// <summary>
  /// Per-square ply-since-last-move values at the current path frontier (64 bytes).
  /// Maintained incrementally during selection via PossiblyBranched.
  /// Only valid when Engine.NeedsPlySinceLastMove is true.
  /// </summary>
  internal PlySinceLastMoveArray PlySinceLastMove;

  /// <summary>
  /// Ping-pong buffer used by PlySinceLastMoveUpdater.ApplyMoveWithSwap.
  /// </summary>
  internal PlySinceLastMoveArray PlySinceLastMoveTemp;

  /// <summary>
  /// Total number of nodes in the complete path 
  /// (including all antecedent segments before the final branch).
  /// </summary>
  public int NumVisitsInPath;

  // If not null, this is the index of the visit at which backup should resume (after a split or a backup resumption)
  public short? PendingResumptionNextSlotToUse;

  #endregion

  /// <summary>
  /// Resets all fields to default values.
  /// 
  /// TODO: consider alternate design 
  ///       (put all the mutable state in a single struct field 
  ///       and reset that struct to default).
  /// </summary>
  internal void Reinitialize()
  {
    PathID = default;
    slots = default;
    parent = default;
    parentSlotIndexLastUsedThisSegment = default;
    numSlotsUsed = default;
    MaxQSubOptimality = default;
    TerminationReason = default;
    TerminationInfo = default;
    RunningHash = default;
    NumVisitsInPath = default;
    PendingResumptionNextSlotToUse = default;
    // Note: PlySinceLastMove and PlySinceLastMoveTemp are NOT reset here.
    // They are value types that are re-populated when the path is initialized in ExtendPathsRecursively.
    // Their contents are overwritten when Engine.NeedsPlySinceLastMove is true.
  }

  #region Helper accessors

  /// <summary>
  /// The engine associated with this iterator.
  /// </summary>
  public MCGSEngine Engine => Iterator.Engine;

  /// <summary>
  /// The index of this iterator (used to distinguish between multiple overlapping iterators).
  /// Currently only indices 0 and possibly 1 are used.
  /// </summary>
  public int IteratorID => Iterator.IteratorID;

  public Graph Graph => Engine.Graph;

  /// <summary>
  /// The algorithm to be used during during the backup phase.
  /// </summary>
  public BackupMethodEnum BackupMode => Iterator.BackupMode;

  /// <summary>
  /// The strategy to be used during backup.
  /// </summary>
  public MCGSSelectBackupStrategyBase Strategy => Engine.Strategy;

  public bool IsRootInitializationPath => NumVisitsInPath == 1 && slots[0].IsRootInitializationPath;

  public ref MCGSPathVisit FirstVisitRef => ref slots[0];

  #endregion


  /// <summary>
  /// Constructor for a new (root) path.
  /// </summary>
  /// <param name="iterator"></param>
  public MCGSPath(MCGSIterator iterator)
  {
    Iterator = iterator;
    PathMode = Engine.Manager.ParamsSearch.PathTranspositionMode;
  }


  /// <summary>
  /// Resolve a global pathIndex into the specific segment and its local slot index.
  /// </summary>
  private (MCGSPath segment, int localIndex) ResolveSegment(int pathIndex)
  {
    Debug.Assert(pathIndex >= 0 && pathIndex < NumVisitsInPath,
                 $"pathIndex {pathIndex} out of [0..{NumVisitsInPath})");

    // 1) collect the chain of segments from root → ... → this
    List<MCGSPath> chain = new(NumVisitsInPath);
    for (MCGSPath p = this; p != null; p = p.parent)
    {
      chain.Add(p);
    }
    chain.Reverse();  // now chain[0] is root, chain[^1] is `this`

    // 2) walk them in order, summing how many visits each contributes
    int cursor = 0;
    for (int i = 0; i < chain.Count; i++)
    {
      MCGSPath seg = chain[i];
      int segCount;

      if (i < chain.Count - 1)
      {
        // For a non‑leaf segment, the *child* tells us how many slots it inherited:
        MCGSPath child = chain[i + 1];
        segCount = child.parentSlotIndexLastUsedThisSegment + 1;
      }
      else
      {
        // Leaf segment contributes all of its local slots
        segCount = seg.numSlotsUsed;
      }

      Debug.Assert(segCount >= 0 && segCount <= seg.numSlotsUsed, $"Segment at depth {i} contributes {segCount} slots but has {seg.numSlotsUsed}");

      // 3) does our pathIndex live in [cursor .. cursor+segCount)?
      if (pathIndex < cursor + segCount)
      {
        int localIndex = pathIndex - cursor;
        Debug.Assert(localIndex < seg.numSlotsUsed, $"Computed localIndex {localIndex} out of [0..{seg.numSlotsUsed})");
        return (seg, localIndex);
      }

      cursor += segCount;
    }

    throw new InvalidOperationException($"Internal error: could not resolve pathIndex {pathIndex}");
  }


  /// <summary>
  /// Returns reference to MCGSPathVisit at specified absolute index within the full path.
  /// </summary>
  public ref MCGSPathVisit this[int pathIndex]
  {
    get
    {
      (MCGSPath segment, int localIndex) = ResolveSegment(pathIndex);
      return ref segment.slots[localIndex];
    }
  }


  /// <summary>
  /// Returns a ref to the visit at the composite position.
  /// </summary>
  public ref MCGSPathVisit VisitRef(int pathIndex)
  {
    (MCGSPath segment, int localIndex) = ResolveSegment(pathIndex);
    return ref segment.slots[localIndex];
  }

  /// <summary>
  /// Returns the leaf (last) node in the path.
  /// </summary>
  public GNode LeafNode => slots[numSlotsUsed - 1].ParentChildEdge.ChildNode;


  /// <summary>
  /// Adds the root visit to a new path.
  /// </summary>
  /// <param name="rootPosition">The root position.</param>
  public void AddRoot(in MGPosition rootPosition)
  {
    Debug.Assert(NumVisitsInPath == 0);

    // Create a new visit for the root node.
    PosHash64 hash64 = MGPositionHashing.Hash64(in rootPosition);
    MCGSPathVisit rootVisit = new(this, default, -1, hash64)
    {
      ChildPosition = rootPosition
    };
    slots[numSlotsUsed++] = rootVisit;
    NumVisitsInPath = 1;
  }


  private void EnsureCapacity(int neededCapacity) => slots.EnsureSize(neededCapacity);


  /// <summary>
  /// Adds a new visit to this path segment.
  /// </summary>
  /// <param name="parentNode">The parent node in the graph.</param>
  /// <param name="childPosition">The position associated with the child node.</param>
  /// <param name="indexInParent">The index of the child edge in the parent.</param>
  /// <param name="numVisits">The number of visits (typically used in updating counts).</param>
  /// <returns>The created MCGSPathVisit.</returns>
  public ref MCGSPathVisit AddVisit(GNode parentNode,
                                    int indexInParent,
                                    in MGPosition childPosition,
                                    int numVisits,
                                    PosHash64 childHashStandalone64,
                                    bool moveIrreversible)
  {
    Debug.Assert(parentNode.IsEvaluated);

    EnsureCapacity(numSlotsUsed + 1);

    slots[numSlotsUsed].Init(this, parentNode, indexInParent,
                             childHashStandalone64,
                             numVisits,
                             moveIrreversible,
                             in childPosition);

    NumVisitsInPath++;
    numSlotsUsed++;

    if (numSlotsUsed > 1)
    {
      Debug.Assert(slots[numSlotsUsed - 1].ParentChildEdge.ParentNode
                != slots[numSlotsUsed - 2].ParentChildEdge.ParentNode);
    }

    return ref slots[numSlotsUsed - 1];
  }


  /// <summary>
  /// Returns an allocation‐free enumerable over the visits of this path,
  /// starting from the leaf (last visit) and iterating backwards to the root.
  /// </summary>
  internal MCGSPathLeafToRootEnumerable PathVisitsLeafToRoot => new(this);

  internal MCGSPathLeafToRootEnumerable PathVisitsLastBackedUpToRoot

    => new(this, PendingResumptionNextSlotToUse == null ? numSlotsUsed : PendingResumptionNextSlotToUse.Value + 1);


  /// <summary>
  /// Returns if visiting specified node as next path visit
  /// would create a cycle in the path (i.e. revisit a node already visited).
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  public bool VisitToNodeWouldCreateCycle(NodeIndex descendentNodeIndex, bool useIrreversibilityOptimization)
  {
    foreach (MCGSPathVisitMember pathMember in PathVisitsLeafToRoot)
    {
      ref readonly MCGSPathVisit visitRef = ref pathMember.PathVisitRef;
      ref readonly GEdge visitEdge = ref visitRef.ParentChildEdge;

      if (visitEdge.Type == GEdgeStruct.EdgeType.ChildEdge
       && !pathMember.IsPathLeaf
       && visitEdge.ChildNode.Index == descendentNodeIndex)
      {
        return true;
      }

      // But in the special case of the root visit we need to also check the parent (since it is not the child of any 
      if (pathMember.IsRoot && visitEdge.ParentNode.Index == descendentNodeIndex)
      {
        return true;
      }

      if (useIrreversibilityOptimization && visitRef.MoveIrreverisible)
      {
        // Short circuit, if irreversible then parent (nor antecedents) could map back to position.
        return false;
      }
    }

    return false;
  }


  /// <summary>
  /// Returns if a cycle exists in the path.
  /// </summary>
  public bool DebugCheckCycleExists
  {
    get
    {
      if (Graph.GraphEnabled)
      {
        HashSet<int> visitedIndices = new(NumVisitsInPath);
        foreach (MCGSPathVisitMember visitPair in PathVisitsLeafToRoot)
        {
          if (!visitedIndices.Add(visitPair.PathVisitRef.ParentChildEdge.ParentNode.Index.Index))
          {
            return true;
          }
        }
      }

      return false;
    }
  }


  /// <summary>
  /// Returns reference to the last visit in the path.
  /// </summary>
  public ref MCGSPathVisit LeafVisitRef => ref slots[numSlotsUsed - 1];


  /// <summary>
  /// Returns a PositionWithHistory representing the full sequence of positions
  /// from the search root (including prehistory) to the leaf position.
  /// </summary>
  public PositionWithHistory LeafPositionWithHistory
  {
    get
    {
      // Get prehistory positions from Graph.Store.HistoryHashes.
      ReadOnlySpan<MGPosition> prehistoryPositions = Graph.Store.HistoryHashes.PriorPositionsMG;

      // Get positions from graph root to search root.
      GraphRootToSearchRootNodeInfo[] graphToSearchRootPath = Engine.SearchRootPathFromGraphRoot;
      int graphToSearchRootCount = graphToSearchRootPath?.Length ?? 0;

      // Count path visits (excluding root initialization paths).
      int pathVisitCount = IsRootInitializationPath ? 0 : NumVisitsInPath;

      // Calculate total positions:
      // - Prehistory positions
      // - Graph-root-to-search-root positions
      // - Path visit positions (child positions from each visit)
      int totalPositions = prehistoryPositions.Length
                         + graphToSearchRootCount
                         + pathVisitCount;

      // Collect all positions in order.
      Position[] allPositions = new Position[totalPositions];
      int index = 0;

      // 1. Add prehistory positions.
      for (int i = 0; i < prehistoryPositions.Length; i++)
      {
        allPositions[index++] = prehistoryPositions[i].ToPosition;
      }

      // 2. Add positions from graph root to search root.
      if (graphToSearchRootPath != null)
      {
        for (int i = 0; i < graphToSearchRootPath.Length; i++)
        {
          allPositions[index++] = graphToSearchRootPath[i].ChildPosMG.ToPosition;
        }
      }

      // 3. Add path visit positions (from root to leaf).
      if (pathVisitCount > 0)
      {
        // Collect path visits from leaf to root, then reverse.
        Span<MGPosition> pathPositions = stackalloc MGPosition[pathVisitCount];
        int pathIndex = pathVisitCount - 1;

        foreach (MCGSPathVisitMember visitMember in PathVisitsLeafToRoot)
        {
          pathPositions[pathIndex--] = visitMember.PathVisitRef.ChildPosition;
        }

        // Copy path positions in forward order.
        for (int i = 0; i < pathVisitCount; i++)
        {
          allPositions[index++] = pathPositions[i].ToPosition;
        }
      }

      // Construct PositionWithHistory from the collected positions.
      // The first position may be missing en passant info when reconstructing moves.
      return new PositionWithHistory(allPositions.AsSpan(), firstPositionMayBeMissingEnPassant: true, recalcRepetitions: true);
    }
  }


  public ref MCGSPathVisit LastVisitDuringBackup =>
    ref PendingResumptionNextSlotToUse is null ? ref slots[numSlotsUsed - 1]
                                               : ref slots[PendingResumptionNextSlotToUse.Value];


  public bool HashFoundInHistoryOrPrehistory(PosHash64 matchHashValue)
  {
    bool haveSeenRepetition = false;

    // Process all visits in the composite path.
    foreach (MCGSPathVisitMember visitPair in PathVisitsLeafToRoot)
    {
      ref readonly MCGSPathVisit visitRef = ref visitPair.PathVisitRef;

      if (visitRef.ChildNodeHashStandalone64 == matchHashValue)
      {
        return true;
      }


      if (visitRef.MoveIrreverisible)
      {
        // The move that transitioned from parent to child was irreversible.
        // Therefore no prior position in the path (or prehistory)
        // could possibly be a repetition with any subsequent position.
        return false;
      }
    }

    // We checked all path visits above, but only the chlid in each.
    // Therefore search root has not yet been processed - so we do that here.
    if (Engine.SearchRootNode.HashStandalone == matchHashValue)
    {
      return true;
    }

    return HashFoundInGraphRootPathOrPrehistory(Engine.Graph, Engine.SearchRootPathFromGraphRoot,
                                                matchHashValue, ref haveSeenRepetition);
  }


  # region Split operation


  /// <summary>
  /// Accepts a base path and determines if we need to split into two paths
  /// 
  /// If no visits remaining, no need for split and simply returns the base path.
  /// Otherwise:
  ///   - a clone of the current path is created and returned
  ///   - the current path is adjusted so that the last recorded visit is deleted
  ///     and the visit count becomes the remaining visit count (unless keepLastVisit is true);
  /// </summary>
  /// <param name="numVisitsRemaining"></param>
  /// <param name="numVisitsAttemptedThisPath"></param>
  /// <param name="thisChildHashStandalone"></param>
  /// <param name="moveWasIrreversibleMove"></param>
  /// <returns></returns>
  public MCGSPath PossiblyBranched(int numVisitsRemaining,
                                   int numVisitsAttemptedThisPath,
                                   PosHash96 thisChildHashStandalone,
                                   bool moveWasIrreversibleMove,
                                   MGMove moveMG)
  {
    if (numVisitsRemaining == 0)
    {
      // Not branched! This is the final surviving path passing thru this node.
      Debug.Assert(LeafVisitRef.NumVisitsAttempted == numVisitsAttemptedThisPath);
      if (moveWasIrreversibleMove)
      {
        RunningHash = default;
      }

      RunningHash.Add(thisChildHashStandalone);

      if (Engine.NeedsPlySinceLastMove)
      {
        PlySinceLastMoveArray.ApplyMoveWithSwap(ref PlySinceLastMove, ref PlySinceLastMoveTemp, in moveMG);
      }

      return this;
    }

    /// Creates a new path cloned from an existing path.
    /// Instead of copying the history of visits, the new path records a reference to its parent 
    /// (the previous full path) and the index of the last visit in the parent at the moment of cloning.
    MCGSPath splittingPath = Iterator.AllocatedPath(MCGSSelect.NumInitialSlotsFromNumVisits(LeafVisitRef.ParentChildEdge.ParentNode.N, numVisitsAttemptedThisPath));

    // Adjust parent information for this new path.
    if (numSlotsUsed == 1)
    {
      // Not descended from any slot at this level.
      // Point parent back to parent path.
      splittingPath.parent = parent;
      splittingPath.parentSlotIndexLastUsedThisSegment = parentSlotIndexLastUsedThisSegment;
    }
    else
    {
      splittingPath.parent = this;
      splittingPath.parentSlotIndexLastUsedThisSegment = numSlotsUsed - 2;
    }


    splittingPath.slots[0] = slots[numSlotsUsed - 1];
    splittingPath.slots[0].ParentPath = splittingPath;
    splittingPath.slots[0].NumVisitsAttempted = (short)numVisitsAttemptedThisPath;

    // other fields such as ParentFirstVisitCacheInfoIndex also copied by above statement
    // already copied above      splittingPath.slots[0].ParentFirstVisitCacheInfoIndex = this.slots[numSlotsUsed - 1].ParentFirstVisitCacheInfoIndex;

    splittingPath.numSlotsUsed = 1;

    splittingPath.NumVisitsInPath = this.NumVisitsInPath;
    splittingPath.TerminationReason = this.TerminationReason;
    splittingPath.MaxQSubOptimality = this.MaxQSubOptimality;

    splittingPath.RunningHash = this.RunningHash;
    if (moveWasIrreversibleMove)
    {
      splittingPath.RunningHash = default;
    }

    splittingPath.RunningHash.Add(thisChildHashStandalone);

    if (Engine.NeedsPlySinceLastMove)
    {
      ((ReadOnlySpan<byte>)PlySinceLastMove.SquarePlySince).CopyTo(splittingPath.PlySinceLastMove.SquarePlySince);
      PlySinceLastMoveArray.ApplyMoveWithSwap(ref splittingPath.PlySinceLastMove, ref splittingPath.PlySinceLastMoveTemp, in moveMG);
      // this.PlySinceLastMove stays at pre-move state (correct for next child)
    }

    // Remove the last visit from the current path (unless keepLastVisit is true).
    TerminationReason = MCGSPathTerminationReason.NotYetTerminated;
    NumVisitsInPath--;
    numSlotsUsed--;

    return splittingPath;
  }

  #endregion


  /// <summary>
  /// Writes a textual dump of this path.
  /// Iterates over the full composite sequence of visits.
  /// </summary>
  public void DumpLocalVisits()
  {
    Console.WriteLine();
    Console.WriteLine($"LOCAL VISITS for MCGS PATH {this}");
    for (int i = 0; i < numSlotsUsed; i++)
    {
      MCGSPathVisit visit = slots[i];
      Console.WriteLine($"#{i,3} {visit}");
    }
  }

  /// <summary>
  /// Writes a textual dump of this path.
  /// Iterates over the full composite sequence of visits.
  /// </summary>
  public void DumpAllVisits()
  {
    Console.WriteLine($"ALL VISITS for MCGS PATH {this}");
    bool haveSeenAccepted = false;
    for (int i = 0; i < NumVisitsInPath; i++)
    {
      MCGSPathVisit visit = this[i];
      if (this[i].NumVisitsAccepted is not null)
      {
        if (!haveSeenAccepted && !this[i].ParentChildEdge.ParentNode.IsSearchRoot)
        {
          // Mark demarcation between already and not yet backed up visits.
          Console.WriteLine("..........");
        }
        haveSeenAccepted = true;
      }

      Console.WriteLine($"#{i,3} {visit}");
    }
    Console.WriteLine();
  }


  /// <summary>
  /// Verify that all internal fields remain in a consistent state.
  /// </summary>
  [Conditional("DEBUG")]
  internal void DebugValidateState(bool pathIsFinalized)
  {

    Debug.Assert(!pathIsFinalized || TerminationReason != MCGSPathTerminationReason.NotYetTerminated);

    if (pathIsFinalized)
    {
      // Select phase will determine the number of visits accepted
      // at the last path visit (only). Only later (during backup) is this assigned with antecedent visits.
      Debug.Assert(LeafVisitRef.NumVisitsAccepted is not null);

      // Any leaf visit originating from a new neural network evaluation should have at most one visit accepted.  
      Debug.Assert(!(TerminationReason == MCGSPathTerminationReason.PendingNeuralNetEval
                    || TerminationReason == MCGSPathTerminationReason.AlreadyNNEvaluated
                    || TerminationReason == MCGSPathTerminationReason.PiggybackPendingNNEval)
                   || LeafVisitRef.NumVisitsAccepted <= 1);
    }

    // Basic slot invariants
    Debug.Assert(slots != null, "slots array must not be null");
    Debug.Assert(numSlotsUsed >= 0 && numSlotsUsed <= slots.NumItemsAllocated, $"numSlotsUsed ({numSlotsUsed}) must be in [0, slots.Length({slots.NumItemsAllocated})]");

    Debug.Assert(numSlotsUsed >= 0
    //      && numSlotsUsed == slots.Length   // logical length must match
    && numSlotsUsed <= slots.NumItemsAllocated // and stay within capacity
    , $"numSlotsUsed={numSlotsUsed}, Length={slots.NumItemsAllocated}, Cap={slots.NumItemsAllocated}");

    // Each local slot must have been initialized properly
    for (int i = 0; i < numSlotsUsed; i++)
    {
      MCGSPathVisit sv = slots[i];
      Debug.Assert(!sv.Equals(default(MCGSPathVisit)), $"slots[{i}] is still default(MCGSPathVisit)");
      Debug.Assert(sv.ParentPath == this, $"slots[{i}].ParentPath must point back to this");
    }

    // Ensure parent linkage makes sense
    if (parent != null)
    {
      Debug.Assert(parentSlotIndexLastUsedThisSegment < parent.numSlotsUsed,
                   "parentSlotIndexLastUsedThisSegment must reference a valid slot in parent");
    }

    // Begin composite‑visit count check.
    int expectedVisits = numSlotsUsed;

    // First include the slots inherited from this path's parent.
    if (parent != null)
    {
      Debug.Assert(parentSlotIndexLastUsedThisSegment >= -1 && parentSlotIndexLastUsedThisSegment < parent.numSlotsUsed,
                   $"this.parentSlotIndexLastUsedThisSegment out of range: {parentSlotIndexLastUsedThisSegment}");
      expectedVisits += (parentSlotIndexLastUsedThisSegment + 1);
    }

    // Then include all further ancestral inheritance.
    MCGSPath ancestor = parent;
    while (ancestor?.parent != null)
    {
      int inherited = ancestor.parentSlotIndexLastUsedThisSegment;
      Debug.Assert(inherited >= -1 && inherited < ancestor.parent.numSlotsUsed,
                   $"ancestor.parentSlotIndexLastUsedThisSegment out of range on {ancestor}: {inherited}");
      expectedVisits += (inherited + 1);
      ancestor = ancestor.parent;
    }

    Debug.Assert(NumVisitsInPath == expectedVisits, $"NumVisitsInPath ({NumVisitsInPath}) disagrees with computed total ({expectedVisits})");

    // Check the composite parent→child linkage and other properties across the full path.
    if (NumVisitsInPath > 1)
    {
      // This check is expensive if we use the indexer this[i].
      // Instead, we can iterate through the segments of the path.
      // This avoids repeated calls to ResolveSegment().

      // 1) collect the chain of segments from root to this
      List<MCGSPath> chain = [];
      for (MCGSPath p = this; p != null; p = p.parent)
      {
        chain.Add(p);
      }
      chain.Reverse();  // now chain[0] is root, chain[^1] is this

      // Get first visit to start comparison.
      ref MCGSPathVisit prevVisit = ref chain[0].slots[0];
      int visitCount = 1;

      for (int segIdx = 0; segIdx < chain.Count; segIdx++)
      {
        MCGSPath seg = chain[segIdx];
        int segCount;

        if (segIdx < chain.Count - 1)
        {
          MCGSPath child = chain[segIdx + 1];
          segCount = child.parentSlotIndexLastUsedThisSegment + 1;
        }
        else
        {
          segCount = seg.numSlotsUsed;
        }

        int firstSegmentIndex = (segIdx == 0 ? 1 : 0);
        for (int j = firstSegmentIndex; j < segCount; j++)
        {
          ref MCGSPathVisit currentVisit = ref seg.slots[j];

          // Validate that the chain is consistent.
          GNode prevChild = prevVisit.ParentChildEdge.ChildNode;
          GNode thisParent = currentVisit.ParentChildEdge.ParentNode;
          Debug.Assert(prevChild == thisParent, $"Path linkage broken between visits {visitCount - 1} and {visitCount}: {prevChild} != {thisParent}");
          Debug.Assert(prevVisit.ParentChildEdge.ParentNode != currentVisit.ParentChildEdge.ParentNode);
          Debug.Assert(prevVisit.ParentChildEdge.Type == GEdgeStruct.EdgeType.ChildEdge);

          // All non-leaf nodes in the path must be evaluated.
          Debug.Assert(prevVisit.ChildNode.IsEvaluated);

          prevVisit = ref currentVisit;
          visitCount++;
        }
      }

      Debug.Assert(visitCount == NumVisitsInPath);
    }

    // If this is the root‐init path, ensure the single visit is marked root
    if (IsRootInitializationPath)
    {
      Debug.Assert(NumVisitsInPath == 1, "Root‐init path must have exactly one visit");
      Debug.Assert(slots[0].IsRootInitializationPath, "Root slot must have IndexOfChildInParent == -1");
    }

    // Verify Last8PositionsIndices is of expected length.
    int expectedNumPriorPositions = 0;
    if (numSlotsUsed > 0)
    {
      ref readonly MCGSPathVisit lastPathVisit = ref slots[numSlotsUsed - 1];
      expectedNumPriorPositions = lastPathVisit.ParentChildEdge.Type.IsTerminal() ? NumVisitsInPath - 1 : NumVisitsInPath;
      expectedNumPriorPositions = Math.Min(expectedNumPriorPositions, EncodedPositionBatchFlat.NUM_HISTORY_POSITIONS);
    }

    if (NumVisitsInPath > 0 && LeafVisitRef.ParentChildEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn)
    {
      Debug.Assert(TerminationReason == MCGSPathTerminationReason.TerminalEdge
                || TerminationReason == MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode);
    }
  }


  /// <summary>
  /// Returns a string representation of the path sequence.
  /// </summary>
  public string PathSequenceString
  {
    get
    {
      // Build a string of the indices of nodes being visited.
      StringBuilder nodes = new();
      nodes.Append("[" + $"{Engine.SearchRootNode.Index.Index},"); // Root node (1) always appears first
      for (int i = 0; i < NumVisitsInPath; i++)
      {
        MCGSPathVisit thisVisit = this[i];
        bool isChildEdge = thisVisit.ParentChildEdge.Type == GEdgeStruct.EdgeType.ChildEdge;

        string sepChar = (isChildEdge && thisVisit.ParentPath.slots[0].ChildNode == thisVisit.ChildNode) ? "|" : ",";
        nodes.Append((i > 0 ? sepChar : "") + (isChildEdge ? this[i].ChildNode.Index.Index : ""));
      }

      string termStr = slots[numSlotsUsed - 1].ParentChildEdge.Type switch
      {
        GEdgeStruct.EdgeType.TerminalEdgeDrawn => "D",
        GEdgeStruct.EdgeType.TerminalEdgeDecisive => slots[numSlotsUsed - 1].ParentChildEdge.Q < 0 ? "L" : "W",
        _ => ""
      };

      nodes.Append(termStr + "]");
      return nodes.ToString();
    }
  }


  public override string ToString()
  {
    if (PendingResumptionNextSlotToUse is not null)
    {
      ref readonly MCGSPathVisit lastProcessedPathVisit = ref slots[PendingResumptionNextSlotToUse.Value];
      //        ref readonly InFlightInfo inFlightInfo = ref lastProcessedPathVisit.ParentInFlightInfo;
    }

    string lastPathVisitStr;

    if (PendingResumptionNextSlotToUse == null)
    {
      lastPathVisitStr = $"final: {slots[numSlotsUsed - 1].ParentChildEdge}>";
    }
    else
    {
      lastPathVisitStr = $"resume: {slots[PendingResumptionNextSlotToUse.Value].ParentChildEdge}>";
    }

    return NumVisitsInPath == 0 ? $"<MCGSPath #{PathID} (empty)>"
                               : $"<MCGSPath #{PathID} {TerminationReason} {PathSequenceString} len={NumVisitsInPath} "
                               + $"{lastPathVisitStr}  {TerminationReason} >";
  }


  public void DumpLast8Positions()
  {
    throw new NotImplementedException();
#if NOT
    InFlightInfoIndex[] indices = Last8PositionsIndices.ToArray();
    Position posLast = default;
    foreach (InFlightInfoIndex index in indices)
    {
      ref readonly InFlightInfo referenceInfoItemRef = ref Iterator.PathVisitInfoCache.ItemRefAt(Last8PositionsIndices.LastItem.Value);

      MGMove move = default;
      if (posLast != default)
      {
        move = MCGSEvaluatorNeuralNet.MoveBetweenPositions(in posLast, referenceInfoItemRef.Position.ToPosition);
      }

      Console.WriteLine($"Position {index} : {referenceInfoItemRef.Position.ToPosition.FEN} {move}");

      posLast = referenceInfoItemRef.Position.ToPosition;
    }
#endif
  }


  public bool Equals(MCGSPath other)
  {
    return other != null && ReferenceEquals(this, other);
  }


  public int CompareTo(MCGSPath other) => this.PathID.CompareTo(other.PathID);



  public enum PathRepetitionResult { None, InPreHistory, InPath };

  /// <summary>
  /// Returns if the path contains any repetitions.
  /// </summary>
  /// <returns></returns>
  public PathRepetitionResult RepetitionStatus(bool debugDumpDetail = false)
  {
    // TODO: update this. It refers to:
    //         Graph.Store.PriorPositionsMG[i]
    //         Graph.Store.PriorPositionsMGHashesStandalone[i]
    //       etc. It is not aware of the existence of the nodes between the graph root and the search root.
    throw new NotImplementedException();
#if NOT
    // Build a HashSet of all seen positions (standalone hash).
    HashSet<PositionHash96> visitedHashes = new();

    // Process hashes from prehistory moves.
    for (int i = 0; i < Graph.Store.PriorPositionsMGHashesStandalone.Length; i++)
    {
      PositionHash96 positionHashePre = Graph.Store.PriorPositionsMGHashesStandalone[i];
      if (debugDumpDetail)
      {
        string repStr = visitedHashes.Contains(positionHashePre) ? "*" : " ";
        Console.WriteLine(repStr + " pre " + positionHashePre + " " + Graph.Store.PriorPositionsMG[i].ToPosition.FEN);
      }

      if (visitedHashes.Contains(positionHashePre))
      {
        return PathRepetitionResult.InPreHistory;
      }
      visitedHashes.Add(positionHashePre);
    }

    // Process hashes from the path visits.
    for (int i = 0; i < NumVisitsInPath; i++)
    {
      MCGSPathVisit visit = this[i];
      if (debugDumpDetail)
      {
        string repStr = visitedHashes.Contains(visit.ChildNodeHashStandalone) ? "*" : " ";
        throw new NotImplementedException();
        //Console.WriteLine(repStr + " path " + hash + " " + visit.ChildInFlightInfo.Position.ToPosition.FEN + " " + visit);
      }

      if (visitedHashes.Contains(visit.ChildNodeHashStandalone))
      {
        return PathRepetitionResult.InPath;
      }
      visitedHashes.Add(visit.ChildNodeHashStandalone);
    }


    return PathRepetitionResult.None;
#endif
  }

  /// <summary>
  /// Determines the termination state of a path based 
  /// on the position and move list of an unexpanded leaf node.
  /// </summary>
  internal (GameResult result, float v, float d, bool drawByRepetition)
    CalcPathTerminationFromUnexpandedLeaf(int minRepetitionCountForDraw,
                                         in MGPosition childPos,
                                         MGMoveList childMoves,
                                         bool possiblyUseTablebase)
  {
    if (childMoves.NumMovesUsed == 0)
    {
      if (MGMoveGen.IsInCheck(in childPos, childPos.BlackToMove))// TODO: is there a flag set already by the GenerateMoves above?
      {
        return (GameResult.Checkmate, -1, 0, false);
      }
      else
      {
        // No moves available, game is a draw by stalemate
        return (GameResult.Draw, 0, 1, false);
      }
    }
    else if (childPos.CheckDrawBasedOnMaterial == Position.PositionDrawStatus.DrawByInsufficientMaterial)
    {
      return (GameResult.Draw, 0, 1, false);
    }
    else if (childPos.Rule50Count >= 100)
    {
      return (GameResult.Draw, 0, 1, false);
    }
    else if (childPos.RepetitionCount >= minRepetitionCountForDraw)
    {
      return (GameResult.Draw, 0, 1, true);
    }
    else if (possiblyUseTablebase && childPos.PieceCount <= Engine.Manager.evaluatorTB?.MaxCardinality)
    {
      SelectTerminationInfo terminationInfo = new();
      bool foundInTB = Engine.Manager.evaluatorTB.Lookup(this, childPos.ToPosition, ref terminationInfo);
      if (foundInTB)
      {
        return (terminationInfo.GameResult, terminationInfo.V, terminationInfo.DrawP, false);
      }
    }
    else if (possiblyUseTablebase && TryGetTablebasePly1Termination(in childPos, childMoves, out float tbV, out float tbD))
    {
      // Ply-1 tablebase extension: position has one more piece than tablebase covers,
      // but a capture move exists that leads to a tablebase-proven win.
      return (GameResult.Checkmate, tbV, tbD, false);
    }

    return (GameResult.Unknown, float.NaN, float.NaN, false);
  }


  /// <summary>
  /// Attempts to evaluate a position that is 1 ply away from tablebase coverage.
  /// Succeeds only when:
  ///   - the position has exactly one more piece than tablebase max cardinality, and
  ///   - there exists at least one capture move leading to a tablebase-proven loss for the opponent.
  /// </summary>
  /// <param name="childPos">The position to evaluate.</param>
  /// <param name="childMoves">Legal moves from the position.</param>
  /// <param name="v">Output: the value (1.0 for win).</param>
  /// <param name="d">Output: the draw probability (0.0 for decisive).</param>
  /// <returns>True if a winning capture into tablebase was found.</returns>
  private bool TryGetTablebasePly1Termination(in MGPosition childPos, MGMoveList childMoves, out float v, out float d)
  {
    v = float.NaN;
    d = float.NaN;

    EvaluatorSyzygy evaluatorTB = Engine.Manager.evaluatorTB;
    if (evaluatorTB == null)
    {
      return false;
    }

    // Only applies when position has exactly one more piece than tablebase covers.
    if (childPos.PieceCount != evaluatorTB.MaxCardinality + 1)
    {
      return false;
    }

    // Iterate over capture moves looking for one that leads to a tablebase loss for opponent.
    for (int i = 0; i < childMoves.NumMovesUsed; i++)
    {
      MGMove move = childMoves.MovesArray[i];
      if (!move.Capture)
      {
        continue;
      }

      // Apply the capture move.
      MGPosition capturePos = childPos;
      capturePos.MakeMove(move);

      // After capture, piece count should now be within tablebase range.
      if (capturePos.PieceCount > evaluatorTB.MaxCardinality)
      {
        continue;
      }

      // Probe the tablebase for the resulting position.
      SelectTerminationInfo tbInfo = new();
      bool found = evaluatorTB.Lookup(this, capturePos.ToPosition, ref tbInfo);

      if (found && tbInfo.V < 0)
      {
        // Opponent loses after this capture, meaning we have a winning move.
        // Return win from our perspective.
        v = 1.0f;
        d = 0.0f;
        return true;
      }
    }

    return false;
  }



  public bool PathSameAs(MCGSPath otherPath)
  {
    if (otherPath.NumVisitsInPath != this.NumVisitsInPath)
    {
      return false;
    }

    for (int i = 0; i < otherPath.NumVisitsInPath; i++)
    {
      // Compare by parent node and child index.
      if (otherPath[i].ParentChildEdge.ParentNode != this[i].ParentChildEdge.ParentNode ||
          otherPath[i].IndexOfChildInParent != this[i].IndexOfChildInParent)
      {
        return false;
      }
    }
    return true;
  }


  /// <summary>
  /// Searches for a matching hash in the graph-root-to-search-root path and prehistory hashes.
  /// </summary>
  /// <param name="matchHashValue">The hash value to search for.</param>
  /// <param name="nodesGraphToSearchRoot">Array of nodes from graph root to search root.</param>
  /// <param name="prehistoryHashes">Array of prehistory position hashes.</param>
  /// <param name="haveSeenRepetition">Tracks whether a repetition has already been seen (for 3-fold detection).</param>
  /// <returns>True if the hash is found in the graph root path or prehistory.</returns>
  public static bool HashFoundInGraphRootPathOrPrehistory(Graph graph,
                                                          ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot,
                                                          PosHash64 matchHashValue,
                                                          ref bool haveSeenRepetition)
  {
    ReadOnlySpan<PosHash64> prehistoryHashes = graph.Store.HistoryHashes.PriorPositionsHashes64;

    // Process all nodes (if any) from the search root node up to the graph root node.
    // Do not process the node at index 0 (search root node) since it is already tested by caller.
    int arrayLength = nodesGraphToSearchRoot.Length;
    for (int i = arrayLength - 1; i >= 1; i--)
    {
      ref readonly GraphRootToSearchRootNodeInfo nodeFromGraphRootToSearchRoot = ref nodesGraphToSearchRoot[i];
      Debug.Assert(nodeFromGraphRootToSearchRoot.ChildNode.HashStandalone == nodeFromGraphRootToSearchRoot.ChildHashStandalone64);

      if (nodeFromGraphRootToSearchRoot.ChildHashStandalone64 == matchHashValue)
      {
        if (!MCGSParamsFixed.FIX_DRP_NEEDS_3_BEFORE_ROOT || haveSeenRepetition)
        {
          return true;
        }
        haveSeenRepetition = true;
      }

      if (nodeFromGraphRootToSearchRoot.MoveToChildIrreversible)
      {
        return false;
      }
    }

    // Process all prehistory visits (this will also include the root node move).
    // TODO: For efficiency add tracking and checking of irreversibility also in the prehistory.
    int numPriorPositions = prehistoryHashes.Length;
    for (int i = 0; i < numPriorPositions; i++)
    {
      if (prehistoryHashes[i] == matchHashValue)
      {
        if (!MCGSParamsFixed.FIX_DRP_NEEDS_3_BEFORE_ROOT || haveSeenRepetition)
        {
          return true;
        }
        haveSeenRepetition = true;
      }
    }

    return false;
  }
}
