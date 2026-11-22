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
using System.Collections.Generic;

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// One triggered graft: the anchor node at which a probe fired and the witness move line
/// to be grafted below it (already truncated to GraftMaxPlies). For a refutation trigger
/// the line is DominantMove followed by the probe's refutation PV ("your move fails
/// because..."); for a discovery trigger it is the probe's own PV.
/// </summary>
internal sealed class ProbeGraftPlan
{
  /// <summary>Graph index of the anchor node.</summary>
  public int AnchorNodeIndex;

  /// <summary>Witness moves to graft, starting with the move at the anchor.</summary>
  public MGMove[] Moves;

  /// <summary>True for a refutation trigger, false for a discovery trigger.</summary>
  public bool IsRefutation;

  /// <summary>Probe margin in centipawns (diagnostics only; refutation triggers).</summary>
  public int MarginCp;

  /// <summary>Prober iteration depth of the triggering result (diagnostics only).</summary>
  public int ProbeDepth;
}


/// <summary>
/// Executes triggered grafts against the live graph using only existing machinery:
/// each active graft is advanced one ply per pump via MCGSIterator.RunProbeSpecs with
/// commitInsteadOfDrop = true, so every grafted visit flows through the stock
/// select/evaluate/backup pipeline (forced spine from the search root, forced child at the
/// cursor, stock selection below, aggregated NN batch, normal backup). All grafts of a pump
/// advance in lockstep so their specs aggregate into as few NN batches as possible.
///
/// A witness move whose child slot is not yet expanded can only be forced when it is the
/// next in-order expansion hole (the edge store expands strictly in ascending index order),
/// so the scheduler backfills any intermediate holes with single-visit specs in the same
/// call - exactly the expansions stock search would perform next at that node, charged to
/// the graft NN budget.
///
/// On completion the optional prior nudge is applied along the applied path (bounded floor
/// raise with proportional renormalization of sibling priors, per-node cumulative mass cap),
/// and optionally the path's edges are marked IsStale for a StaleDrain pass to propagate.
///
/// All methods run on the pump thread only, at the post-batch quiescent point.
/// </summary>
internal sealed class ProbeGraftScheduler
{
  /// <summary>Pumps a graft may stall (e.g. on a contended deferred policy copy) before being aborted.</summary>
  const int MAX_STALLS = 8;

  /// <summary>
  /// Consecutive pumps a graft may miss its forced-crossing path record before being abandoned.
  /// Under graph reuse (i.e. in game play) these misses dominated aborts entirely, so retrying
  /// converts most of them into completed grafts; see AbortReason.NoPathRecord.
  /// </summary>
  const int MAX_RECORD_MISSES = 8;

  /// <summary>One graft being advanced ply-by-ply.</summary>
  sealed class ActiveGraft
  {
    public ProbeGraftPlan Plan;

    /// <summary>Graph index of the node the next witness move applies at.</summary>
    public int CursorNodeIndex;

    /// <summary>Index into Plan.Moves of the next move to graft.</summary>
    public int NextMoveIndex;

    /// <summary>Plies successfully applied so far.</summary>
    public int PliesApplied;

    /// <summary>Consecutive pumps this graft could not make progress.</summary>
    public int StallCount;

    /// <summary>
    /// Consecutive pumps on which the forced crossing produced no path record. Kept separate from
    /// StallCount, which TryResolveCursorNode resets on every successful cursor resolution.
    /// </summary>
    public int RecordMissCount;

    /// <summary>Anchor node Q immediately before the first grafted ply (impact baseline).</summary>
    public double AnchorQBefore;

    /// <summary>(parent node index, child index) of every applied ply, anchor first.</summary>
    public readonly List<(int NodeIndex, int ChildIndex)> AppliedPath = new();

    /// <summary>Child slot resolved for the pending ply of the current pump (-1 when none).</summary>
    public int PendingChildIndex = -1;

    /// <summary>Raw EncodedMove resolved at PendingChildIndex (re-verified after the call,
    /// guarding against unexpanded-header rearrangement by concurrent specs of the pump).</summary>
    public ushort PendingMoveRaw;
  }

  readonly ProbeGraftCoordinator Coordinator;
  readonly ParamsProbeGraft Params;
  readonly ProbeGraftStats Stats;

  readonly Queue<ProbeGraftPlan> pending = new();
  readonly List<ActiveGraft> active = new();

  /// <summary>Per-node cumulative prior mass moved by nudges (the PriorNudgeMaxTotalMass ledger).</summary>
  readonly Dictionary<int, float> nudgeMassUsed = new();

  /// <summary>Anchor impact baselines of completed grafts (consumed at end of search).</summary>
  internal readonly List<(int AnchorNodeIndex, double AnchorQBefore)> CompletedGraftImpacts = new();


  internal ProbeGraftScheduler(ProbeGraftCoordinator coordinator, ParamsProbeGraft parms, ProbeGraftStats stats)
  {
    Coordinator = coordinator;
    Params = parms;
    Stats = stats;
  }


  /// <summary>If any grafts are active or awaiting promotion.</summary>
  internal bool HasWork => active.Count > 0 || pending.Count > 0;


  /// <summary>Queues a triggered graft plan for execution.</summary>
  /// <param name="plan"></param>
  internal void Enqueue(ProbeGraftPlan plan) => pending.Enqueue(plan);


  /// <summary>
  /// Advances every active graft by (at most) one ply, promoting pending plans into free
  /// slots first and enforcing the MaxGraftEvalFraction NN budget (pause, not discard).
  /// Returns whether any graft ply was actually applied (i.e. committed visits ran).
  /// </summary>
  /// <param name="iterator"></param>
  /// <returns></returns>
  internal bool AdvanceOnePly(MCGSIterator iterator)
  {
    Graph graph = iterator.Engine.Graph;

    while (active.Count < Params.MaxActiveGrafts && pending.Count > 0)
    {
      PromoteIfViable(pending.Dequeue(), graph);
    }

    if (active.Count == 0)
    {
      return false;
    }

    // NN budget: pause advancement (grafts stay active) while over the eval fraction cap.
    if (Stats.NumGraftNNEvals > (long)(Params.MaxGraftEvalFraction * Coordinator.Manager.NumEvalsThisSearch))
    {
      Stats.NumGraftsPausedBudget++;
      return false;
    }

    // Phase 1: resolve each graft's next move to a child slot and build its spec group.
    // Groups are packed greedily into RunProbeSpecs calls respecting the MaxBatchSize
    // contract (sum of requested visits per call).
    int maxBatchSize = Coordinator.Manager.ParamsSearch.Execution.MaxBatchSize;
    List<ProbeSpec> callSpecs = new();
    List<ActiveGraft> callGrafts = new();
    int callVisits = 0;
    bool anyPlyApplied = false;

    // Iterate over a copy: completion/abort removes from the live list.
    ActiveGraft[] graftsThisPump = active.ToArray();
    foreach (ActiveGraft graft in graftsThisPump)
    {
      graft.PendingChildIndex = -1;

      GNode cursor = TryResolveCursorNode(graph, graft);
      if (cursor.IsNull)
      {
        continue; // already handled (aborted/completed) inside TryResolveCursorNode
      }

      if (!TryBuildSpecGroup(cursor, graft, maxBatchSize, out List<ProbeSpec> group, out int groupVisits))
      {
        continue; // handled (aborted or stalled)
      }

      if (callVisits + groupVisits > maxBatchSize && callSpecs.Count > 0)
      {
        anyPlyApplied |= ExecuteCall(iterator, graph, callSpecs, callGrafts);
        callSpecs = new List<ProbeSpec>();
        callGrafts = new List<ActiveGraft>();
        callVisits = 0;
      }

      callSpecs.AddRange(group);
      callGrafts.Add(graft);
      callVisits += groupVisits;
    }

    if (callSpecs.Count > 0)
    {
      anyPlyApplied |= ExecuteCall(iterator, graph, callSpecs, callGrafts);
    }

    return anyPlyApplied;
  }


  /// <summary>
  /// Promotes a pending plan to active if its anchor is still a viable graft start,
  /// recording the anchor Q impact baseline (before the first grafted ply).
  /// </summary>
  void PromoteIfViable(ProbeGraftPlan plan, Graph graph)
  {
    Stats.NumGraftsStarted++;

    GNode anchor = ProbeGraftCoordinator.TryResolveNode(graph, plan.AnchorNodeIndex);
    if (anchor.IsNull || anchor.Terminal.IsTerminal() || !anchor.IsEvaluated || anchor.NumPolicyMoves == 0)
    {
      RecordAbort(AbortReason.AnchorNotViable);
      return;
    }

    active.Add(new ActiveGraft()
    {
      Plan = plan,
      CursorNodeIndex = plan.AnchorNodeIndex,
      AnchorQBefore = anchor.Q,
    });
  }


  /// <summary>
  /// Resolves and validates the graft's cursor node, aborting or terminally completing the
  /// graft when it cannot continue. Returns a null GNode when the caller should skip it.
  /// </summary>
  GNode TryResolveCursorNode(Graph graph, ActiveGraft graft)
  {
    GNode cursor = ProbeGraftCoordinator.TryResolveNode(graph, graft.CursorNodeIndex);
    if (cursor.IsNull || !cursor.IsEvaluated)
    {
      Abort(graft, AbortReason.CursorGone);
      return default;
    }

    if (cursor.Terminal.IsTerminal())
    {
      // Previous ply ran into a terminal node: the refutation reached a mate/draw whose
      // value stock backup already propagated - a successful ending.
      CompleteTerminal(graft);
      return default;
    }

    // A cursor created by transposition value-copy may still carry a deferred policy copy
    // (in which case NumPolicyMoves reads 0 until materialization - so this must precede
    // the no-policy-moves abort below); materialize it now (pump thread, quiescent graph)
    // so witness moves can be resolved. The source-node lock is acquired non-blockingly;
    // on the (rare) contention the graft simply stalls until a later pump.
    if (cursor.IsPendingPolicyCopy)
    {
      cursor.AcquireLock();
      bool materialized;
      try
      {
        materialized = cursor.TryDoDeferredPolicyCopyIfNeeded();
      }
      finally
      {
        cursor.ReleaseLock();
      }

      if (!materialized)
      {
        if (++graft.StallCount > MAX_STALLS)
        {
          Abort(graft, AbortReason.PolicyCopyStalled);
        }
        return default;
      }
    }

    if (cursor.NumPolicyMoves == 0)
    {
      Abort(graft, AbortReason.NoPolicyMoves);
      return default;
    }

    graft.StallCount = 0;
    return cursor;
  }


  /// <summary>
  /// Resolves the graft's next witness move to a child slot at the cursor and builds the
  /// spec group for this ply: single-visit backfill specs for any expansion holes below the
  /// target slot (strictly ascending, per the edge store's in-order expansion contract)
  /// followed by the target spec with GraftVisitsPerPly visits.
  /// </summary>
  bool TryBuildSpecGroup(GNode cursor, ActiveGraft graft, int maxBatchSize,
                         out List<ProbeSpec> group, out int groupVisits)
  {
    group = null;
    groupVisits = 0;

    MGPosition cursorPos = cursor.CalcPosition();
    MGMove nextMove = graft.Plan.Moves[graft.NextMoveIndex];
    int childIndex = ResolveMoveToChildIndex(cursor, in cursorPos, nextMove);
    if (childIndex < 0)
    {
      Abort(graft, AbortReason.MoveNotFound);
      return false;
    }

    graft.PendingMoveRaw = (childIndex < cursor.NumEdgesExpanded
                              ? cursor.ChildEdgeAtIndex(childIndex).Move
                              : cursor.EdgeHeadersSpan[childIndex].Move).RawValue;

    int numExpanded = cursor.NumEdgesExpanded;
    int numBackfill = Math.Max(0, childIndex - numExpanded);
    groupVisits = numBackfill + Params.GraftVisitsPerPly;
    if (groupVisits > maxBatchSize)
    {
      // Cannot fit this ply in a single call (implausible: <= 63 backfills); be defensive.
      Abort(graft, AbortReason.PlyTooLarge);
      return false;
    }

    group = new List<ProbeSpec>(numBackfill + 1);
    for (int slot = numExpanded; slot < childIndex; slot++)
    {
      group.Add(new ProbeSpec(cursor, slot, 1));
    }
    group.Add(new ProbeSpec(cursor, childIndex, Params.GraftVisitsPerPly));

    graft.PendingChildIndex = childIndex;
    return true;
  }


  /// <summary>
  /// Runs one aggregated RunProbeSpecs call in commit mode and advances each participating
  /// graft according to the resulting graph state (and path records). Returns whether any
  /// graft ply was applied.
  /// </summary>
  bool ExecuteCall(MCGSIterator iterator, Graph graph, List<ProbeSpec> specs, List<ActiveGraft> grafts)
  {
    int evalsBefore = Coordinator.Manager.NumEvalsThisSearch;
    List<ProbePathRecord> records = iterator.RunProbeSpecs(specs, null, commitInsteadOfDrop: true);
    Stats.NumGraftNNEvals += Coordinator.Manager.NumEvalsThisSearch - evalsBefore;

    bool anyApplied = false;
    foreach (ActiveGraft graft in grafts)
    {
      anyApplied |= AdvanceGraftAfterCall(graph, graft, records);
    }

    return anyApplied;
  }


  /// <summary>
  /// Inspects the post-call graph state (and this call's path records) at the graft's
  /// pending (cursor, child) crossing and advances, terminally completes, or aborts it.
  /// </summary>
  bool AdvanceGraftAfterCall(Graph graph, ActiveGraft graft, List<ProbePathRecord> records)
  {
    int childIndex = graft.PendingChildIndex;
    graft.PendingChildIndex = -1;
    if (childIndex < 0)
    {
      return false;
    }

    GNode cursor = ProbeGraftCoordinator.TryResolveNode(graph, graft.CursorNodeIndex);
    if (cursor.IsNull || childIndex >= cursor.NumEdgesExpanded)
    {
      // The forced expansion did not happen (path evaporated/aborted on the spine).
      Abort(graft, AbortReason.SlotNotExpanded);
      return false;
    }

    GEdge edge = cursor.ChildEdgeAtIndex(childIndex);
    if (edge.Move.RawValue != graft.PendingMoveRaw)
    {
      // The slot no longer carries the resolved witness move (e.g. an unexpanded-header
      // rearrangement performed by another spec's stock descent in this same pump);
      // the forced visits went through a different move - do not advance along it.
      Abort(graft, AbortReason.WitnessMoved);
      return false;
    }

    if (edge.Type.IsTerminal())
    {
      // The witness move itself is terminal (mate or terminal draw edge): success.
      RecordAppliedPly(graft, childIndex);
      CompleteTerminal(graft);
      return true;
    }

    if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeIndex.IsNull)
    {
      Abort(graft, AbortReason.NotChildEdge);
      return false;
    }

    // Classify using this call's record for the forced crossing when available.
    ProbePathRecord record = FindRecord(records, graft.CursorNodeIndex, childIndex);
    if (record == null)
    {
      // No record means this call's forced visit was dropped during select (typically a
      // transposition collision within the probe batch), which is transient. The spine itself is
      // intact -- cursor, witness slot and child edge were all re-validated immediately above -- so
      // the ply is retried on a later pump rather than discarding the graft and the probe behind
      // it. Only a graft that keeps missing is abandoned.
      if (++graft.RecordMissCount > MAX_RECORD_MISSES)
      {
        Abort(graft, AbortReason.NoPathRecord);
      }
      else
      {
        Stats.NumGraftPlyRetriesNoRecord++;
      }
      return false;
    }

    graft.RecordMissCount = 0;
    if (record.TerminationReason == MCGSPathTerminationReason.Abort)
    {
      Abort(graft, AbortReason.PathRecordAborted);
      return false;
    }

    if (record.TerminationReason == MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode
     && record.Hops.Length - 1 == record.SpecStartHopIndex)
    {
      // The forced crossing itself was consumed as a repetition draw (already backed up
      // as a draw by the stock pipeline): a successful terminal ending.
      RecordAppliedPly(graft, childIndex);
      CompleteTerminal(graft);
      return true;
    }

    GNode child = edge.ChildNode;
    if (child.IsNull || !child.IsEvaluated || child.N == 0)
    {
      Abort(graft, AbortReason.ChildUnvisited);
      return false;
    }

    RecordAppliedPly(graft, childIndex);
    graft.CursorNodeIndex = child.Index.Index;
    graft.NextMoveIndex++;

    if (graft.NextMoveIndex >= graft.Plan.Moves.Length || graft.PliesApplied >= Params.GraftMaxPlies)
    {
      CompleteNormal(graft, graph);
    }

    return true;
  }


  void RecordAppliedPly(ActiveGraft graft, int childIndex)
  {
    graft.AppliedPath.Add((graft.CursorNodeIndex, childIndex));
    graft.PliesApplied++;
    Stats.NumGraftPliesApplied++;
  }


  /// <summary>
  /// Returns a path record of this call for the given forced (parent, child) crossing,
  /// preferring one that was not aborted; null when no path reached the crossing.
  /// </summary>
  static ProbePathRecord FindRecord(List<ProbePathRecord> records, int parentNodeIndex, int childIndex)
  {
    ProbePathRecord found = null;
    foreach (ProbePathRecord record in records)
    {
      if (record.SpecParentNodeIndex.Index == parentNodeIndex && record.SpecChildIndex == childIndex)
      {
        found = record;
        if (record.TerminationReason != MCGSPathTerminationReason.Abort)
        {
          return record;
        }
      }
    }
    return found;
  }


  void CompleteNormal(ActiveGraft graft, Graph graph)
  {
    Stats.NumGraftsCompleted++;
    FinishApplied(graft, graph, "done");
  }


  void CompleteTerminal(ActiveGraft graft)
  {
    Stats.NumGraftsEndedTerminal++;
    FinishApplied(graft, Coordinator.Manager.Engine.Graph, "terminal");
  }


  /// <summary>
  /// Common completion path for successfully-ended grafts: prior nudges, optional stale
  /// marking, and impact-baseline retention.
  /// </summary>
  void FinishApplied(ActiveGraft graft, Graph graph, string outcome)
  {
    if (Params.AllowPriorNudges && Params.PriorNudgeFloor > 0)
    {
      ApplyPriorNudges(graft, graph);
    }

    if (Params.MarkGraftPathStale)
    {
      MarkPathStale(graft, graph);
    }

    CompletedGraftImpacts.Add((graft.Plan.AnchorNodeIndex, graft.AnchorQBefore));
    active.Remove(graft);
    Coordinator.LogEvent($"graft {outcome} anchor=#{graft.Plan.AnchorNodeIndex} "
                       + $"{(graft.Plan.IsRefutation ? "ref" : "disc")} marginCp={graft.Plan.MarginCp} "
                       + $"probeDepth={graft.Plan.ProbeDepth} plies={graft.PliesApplied}");
  }


  /// <summary>
  /// Why a graft was abandoned. Every value except MoveNotFound rolls up into the
  /// NumGraftsAbortedPathDropped total, but each is counted separately so the dominant
  /// failure mode is visible without a full event dump.
  /// </summary>
  internal enum AbortReason
  {
    /// <summary>Witness move is not among the cursor's policy moves.</summary>
    MoveNotFound,

    /// <summary>Cursor node no longer resolvable or not yet evaluated.</summary>
    CursorGone,

    /// <summary>Deferred policy copy could not be materialized within MAX_STALLS pumps.</summary>
    PolicyCopyStalled,

    /// <summary>Cursor has no policy moves.</summary>
    NoPolicyMoves,

    /// <summary>Ply would not fit in one RunProbeSpecs call.</summary>
    PlyTooLarge,

    /// <summary>Forced expansion did not materialize the target slot.</summary>
    SlotNotExpanded,

    /// <summary>Target slot no longer carries the resolved witness move.</summary>
    WitnessMoved,

    /// <summary>Edge is not a normal child edge after the forced visit.</summary>
    NotChildEdge,

    /// <summary>RunProbeSpecs returned no path record at all for the forced crossing.</summary>
    NoPathRecord,

    /// <summary>A path record exists for the forced crossing but its descent aborted.</summary>
    PathRecordAborted,

    /// <summary>Child exists but was not evaluated / received no visit.</summary>
    ChildUnvisited,

    /// <summary>Anchor was not a viable graft start when promoted.</summary>
    AnchorNotViable,
  }


  void Abort(ActiveGraft graft, AbortReason reason)
  {
    RecordAbort(reason);

    active.Remove(graft);
    Coordinator.LogEvent($"graft abort anchor=#{graft.Plan.AnchorNodeIndex} "
                       + $"{reason} plies={graft.PliesApplied}");
  }


  /// <summary>
  /// Increments the aggregate and per-reason abort counters.
  /// </summary>
  /// <param name="reason"></param>
  void RecordAbort(AbortReason reason)
  {
    if (reason == AbortReason.MoveNotFound)
    {
      Stats.NumGraftsAbortedMoveNotFound++;
    }
    else
    {
      Stats.NumGraftsAbortedPathDropped++;
    }

    Stats.NumGraftsAbortedByReason[(int)reason]++;
  }


  /// <summary>
  /// Raises the prior of each grafted move to at least PriorNudgeFloor, renormalizing the
  /// node's other priors down proportionally, honoring the per-node cumulative mass cap.
  /// Only expanded edges are ever nudged (the graft itself expanded them), so the
  /// descending-P ordering invariant of the unexpanded header tail is preserved (its
  /// members are only ever scaled by a common factor).
  /// </summary>
  void ApplyPriorNudges(ActiveGraft graft, Graph graph)
  {
    foreach ((int nodeIndex, int childIndex) in graft.AppliedPath)
    {
      GNode node = ProbeGraftCoordinator.TryResolveNode(graph, nodeIndex);
      if (node.IsNull || node.IsPendingPolicyCopy || childIndex >= node.NumEdgesExpanded)
      {
        continue;
      }

      GEdge edge = node.ChildEdgeAtIndex(childIndex);
      float p = edge.P;
      if (float.IsNaN(p) || p >= Params.PriorNudgeFloor)
      {
        continue;
      }

      nudgeMassUsed.TryGetValue(nodeIndex, out float used);
      float delta = Params.PriorNudgeFloor - p;
      float available = Params.PriorNudgeMaxTotalMass - used;
      if (available <= 0)
      {
        Stats.NumPriorNudgesCappedByMass++;
        continue;
      }
      if (delta > available)
      {
        delta = available;
        Stats.NumPriorNudgesCappedByMass++;
      }

      float newP = p + delta;
      float scale = (1.0f - newP) / (1.0f - p); // p <= 0.2 by validation, so denominator >= 0.8

      edge.SetPolicyPrior((FP16)newP);

      int numExpanded = node.NumEdgesExpanded;
      for (int slot = 0; slot < numExpanded; slot++)
      {
        if (slot != childIndex)
        {
          GEdge sibling = node.ChildEdgeAtIndex(slot);
          sibling.SetPolicyPrior((FP16)((float)sibling.P * scale));
        }
      }

      Span<Graphs.GEdgeHeaders.GEdgeHeaderStruct> headers = node.EdgeHeadersSpan;
      for (int slot = numExpanded; slot < node.NumPolicyMoves; slot++)
      {
        headers[slot].SetUnexpandedP((FP16)((float)headers[slot].P * scale));
      }

      nudgeMassUsed[nodeIndex] = used + delta;
      Stats.NumPriorNudgesApplied++;
    }
  }


  /// <summary>
  /// Marks the edges along the applied graft path IsStale so a configured
  /// PostBackupQMode == StaleDrain pass propagates the graft's evidence upward promptly.
  /// Reuses the existing flag/mechanism only (no new propagation code).
  /// </summary>
  void MarkPathStale(ActiveGraft graft, Graph graph)
  {
    foreach ((int nodeIndex, int childIndex) in graft.AppliedPath)
    {
      GNode node = ProbeGraftCoordinator.TryResolveNode(graph, nodeIndex);
      if (!node.IsNull && childIndex < node.NumEdgesExpanded)
      {
        GEdge edge = node.ChildEdgeAtIndex(childIndex);
        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
        {
          edge.IsStale = true;
        }
      }
    }
  }


  /// <summary>
  /// Resolves a witness move to the index of the matching child within the node's policy
  /// move list (expanded edges and unexpanded headers), or -1 when absent. Comparison is
  /// position-aware (each stored EncodedMove is converted to an MGMove at the node's
  /// position) to be robust to encoding-convention differences.
  /// </summary>
  internal static int ResolveMoveToChildIndex(GNode node, in MGPosition nodePos, MGMove move)
  {
    int numPolicy = node.NumPolicyMoves;
    int numExpanded = node.NumEdgesExpanded;
    for (int i = 0; i < numPolicy; i++)
    {
      EncodedMove encoded = i < numExpanded ? node.ChildEdgeAtIndex(i).Move
                                            : node.EdgeHeadersSpan[i].Move;
      MGMove candidate = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encoded, in nodePos);
      if (candidate == move)
      {
        return i;
      }
    }

    return -1;
  }
}
