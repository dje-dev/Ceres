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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

using Ceres.Chess.Probing;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.LC0.Positions;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Phases;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Per-search coordinator of the PICKET M1 refutation-grafting feature: owns the resolved
/// probe source session, the result inbox, the target ledger (ProbeGraftTargeter), the graft
/// scheduler (ProbeGraftScheduler), and the per-search statistics (ProbeGraftStats).
///
/// Created by MCGSManager's constructor when ParamsProbeGraft.AnyConsumerEnabled (an
/// unknown ProbeSourceID fails fast there); ended by MCGSManager.DoSearchInner at end of
/// search; disposed with the manager as a safety net. An MCGSManager is strictly
/// one-search, so this object is too.
///
/// Threading and invariants:
///  - Probe completion callbacks arrive on arbitrary threads and ONLY enqueue into the
///    inbox (plus an interlocked pending count); they never touch the graph, params, or
///    any other stats field.
///  - Two entry points from MCGSIterator.RunOnce, both serialized by pumpLock (the coordinator
///    state is single-writer at any instant even though the calls may come from either
///    iterator's thread under DualOverlappedIterators):
///    (1) DrainInBackupGate, INSIDE the backup gate (PhaseCoordinator.EnterBackup excludes the
///        other iterator's select and backup; only its NN evaluate can be in flight, and
///        QRecomputeHelper.RecomputeNodeQ ignores edges with N == 0): the stamp drain -- results
///        interpreted, stamp bytes written, ancestors recomputed (ProbeStamps.RecomputeUpward),
///        at most ProbeStampMaxAppliesPerBatch stamps per batch, the rest carried in the inbox.
///        It issues no visits and never re-enters a coordinator gate, so it is legal under
///        overlapped iterators (the "Route 4" design, SWARM doc IX.2 item 7d).
///    (2) PumpAtQuiescentPoint, AFTER the gate on the iterator's own thread: incremental stamp
///        targeting over that iterator's own (still intact) PathsSet -- graph reads only, plus
///        one byte (ProbedN) per submitted node -- and, for iterator 0 only, the legacy consumers
///        (periodic sweep, graft advance, root adapt) that issue directed visits through
///        RunProbeSpecs and therefore still require DualOverlappedIterators off (Validate).
///    Lock order is gate -> pumpLock only (the gate-side drain uses TryEnter and defers to the
///    next batch when the other iterator is targeting), so no cycle is possible.
///  - The RunProbeSpecs contract (verified in MCGSIterator.Probe.cs): quiescent graph
///    required (it self-checks root in-flight and throws), single-iterator harnesses only
///    (DualOverlappedIterators is validated off for those consumers), commitInsteadOfDrop = true
///    for grafts, per-call spec visit totals chunked to Execution.MaxBatchSize.
///  - End of search: EndSession is called before the coordinator is dropped; the
///    IProbeSource contract guarantees no callbacks after it returns, so there is
///    no use-after-dispose of the inbox.
///  - Feature exceptions never propagate into search: the pump boundary catch logs the
///    first exception once and disables the feature for the remainder of the search
///    (search integrity beats feature integrity).
/// </summary>
public sealed class ProbeGraftCoordinator : IDisposable
{
  /// <summary>|delta Q| threshold used by impact stats and Shadow verdicts
  /// (matches BottomUpQRecalculator.LARGE_DELTA_THRESHOLD).</summary>
  const double LARGE_DELTA_THRESHOLD = 0.03;

  /// <summary>Maximum retained per-event log lines (VerboseEventLogging).</summary>
  const int MAX_EVENT_LOG_LINES = 512;

  /// <summary>Process-wide monotonically increasing probe request ID.</summary>
  static long nextRequestID;

  /// <summary>One recorded would-have-grafted event (Shadow mode) awaiting its verdict.</summary>
  sealed class ShadowEvent
  {
    /// <summary>Graph index of the node whose probe triggered (the would-be graft anchor).</summary>
    public int AnchorNodeIndex;
    /// <summary>True for the refutation trigger, false for the discovery trigger.</summary>
    public bool IsRefutation;
    /// <summary>Dominant child (slot and edge Q) at trigger time, for scoring the verdict later.</summary>
    public int DominantChildIndexAtTrigger;
    public double DominantEdgeQAtTrigger;
    /// <summary>Anchor node's Q at trigger time.</summary>
    public double AnchorQAtTrigger;
    /// <summary>First move of the line the probe preferred (what would have been grafted).</summary>
    public MGMove ProbeBestMove;
  }

  /// <summary>The owning (one-search) manager.</summary>
  public readonly MCGSManager Manager;

  /// <summary>Per-search statistics (exposed as MCGSManager.ProbeGraftStats).</summary>
  public readonly ProbeGraftStats Stats = new();

  readonly ParamsProbeGraft Params;
  readonly IProbeSource source;

  /// <summary>This search's session on the (possibly process-wide shared) probe source.</summary>
  IProbeSession session;
  readonly ProbeGraftTargeter targeter;
  readonly ProbeGraftScheduler scheduler;
  readonly CancellationTokenSource cts = new();

  /// <summary>Result inbox: the completion callback only enqueues here.</summary>
  readonly ConcurrentQueue<ProbeResult> inbox = new();

  /// <summary>Interlocked count of enqueued-but-undrained results (cheap idle check).</summary>
  int numPendingResults;

  /// <summary>
  /// Serializes the two entry points (gate-side stamp drain, post-gate targeting), which may run
  /// on different iterator threads under DualOverlappedIterators. See the class remarks.
  /// </summary>
  readonly object pumpLock = new();

  /// <summary>Reusable scratch for ProbeStamps.RecomputeUpward (avoids two allocations per stamp).</summary>
  readonly PriorityQueue<GNode, int> recomputeQueue = new();
  readonly HashSet<int> recomputeSeen = new();

  /// <summary>Paces target sweeps (restarted after each sweep).</summary>
  readonly Stopwatch sweepTimer = Stopwatch.StartNew();

  readonly List<ShadowEvent> shadowEvents = new();
  readonly List<string> eventLog = new();

  bool disabledByError;
  bool endSearchDone;
  bool disposed;


  /// <summary>True when this coordinator created (and must dispose) its probe source itself.</summary>
  bool ownsSource;

  /// <summary>Parsed ProbeStampCpWeights of THIS coordinator's params (null = all 1).</summary>
  readonly (int MinCp, float Weight)[] cpWeightTable;


  /// <summary>
  /// Constructor: validates parameters, uses the engine-resolved probe source when present
  /// (else resolves it from the registry and owns the instance; an unknown ID throws - fail
  /// fast at search start, never silently), and begins the probe session.
  /// </summary>
  /// <param name="manager"></param>
  public ProbeGraftCoordinator(MCGSManager manager)
  {
    Manager = manager;
    Params = manager.ParamsSearch.ProbeGraft;

    Debug.Assert(Params != null && Params.AnyConsumerEnabled);
    Params.Validate(manager.ParamsSearch);

    source = manager.ResolvedProbeSource;
    if (source == null)
    {
      // Fallback for consumers constructing MCGSManager directly (no engine wrapper): the factory
      // returns a fresh instance per call, so this coordinator owns it and disposes it at
      // ShutdownSession (otherwise a farm would leak per search).
      if (!ProbeSourceRegistry.TryCreate(Params.ProbeSourceID, out source))
      {
        throw new Exception($"ParamsProbeGraft.ProbeSourceID '{Params.ProbeSourceID}' "
                          + "is not registered with ProbeSourceRegistry.");
      }
      ownsSource = true;
    }

    cpWeightTable = ProbeStamps.ParseCpWeights(Params.ProbeStampCpWeights);
    targeter = new ProbeGraftTargeter(Params);
    scheduler = new ProbeGraftScheduler(this, Params, Stats);

    session = source.BeginSession(OnProbeCompleted);
  }


  /// <summary>
  /// Completion sink handed to the probe source (arbitrary thread): enqueue-only.
  /// </summary>
  /// <param name="result"></param>
  void OnProbeCompleted(ProbeResult result)
  {
    inbox.Enqueue(result);
    Interlocked.Increment(ref numPendingResults);
  }


  /// <summary>
  /// Post-gate entry point, invoked at the end of each MCGSIterator.RunOnce batch on that
  /// iterator's own thread (backup complete, coordinator gates exited, this iterator's PathsSet
  /// intact). Cheap when idle. Body: (1) incremental stamp targeting over the batch's paths (any
  /// iterator); then, iterator 0 only (these need the single-iterator quiescent graph):
  /// (2) the legacy drain when stamps are not the consumer (stamp results drain inside the gate,
  /// see DrainInBackupGate); (3) advance active grafts one ply; (4) the periodic target sweep;
  /// (5) root adapt. Steps 3-5 issue directed visits and reset the PathsSet, which is why the
  /// targeting runs first.
  /// </summary>
  /// <param name="iterator"></param>
  public void PumpAtQuiescentPoint(MCGSIterator iterator)
  {
    if (disabledByError || endSearchDone)
    {
      return;
    }

    bool stampMode = Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active;
    bool incremental = stampMode && Params.ProbeStampIncrementalTargeting;
    bool legacyPump = iterator.IteratorID == 0;
    bool anyResults = !stampMode && legacyPump && Volatile.Read(ref numPendingResults) != 0;
    // The periodic sweep serves the graft/root-veto consumers and the sweep-based stamp form; an
    // advocate-only configuration submits its own root probes and must not pay for sweep probes
    // nobody consumes.
    bool sweepConsumers = Params.Mode != ParamsProbeGraft.ModeType.Disabled
                       || Params.RootAdaptMode == ParamsProbeGraft.RootAdaptModeType.Active
                       || stampMode;
    bool sweepDue = !incremental && legacyPump && sweepConsumers
                 && sweepTimer.Elapsed.TotalSeconds >= Params.TargetSweepIntervalSeconds;
    bool incrementalDue = incremental && !ProbeBudgetExhausted;
    bool graftDue = legacyPump && Params.Mode == ParamsProbeGraft.ModeType.Active && scheduler.HasWork;
    bool rootAdaptDue = legacyPump && Params.RootAdaptMode == ParamsProbeGraft.RootAdaptModeType.Active && !rootAdaptDone;
    bool advocateDue = legacyPump && Params.AdvocateMode == ParamsProbeGraft.AdvocateModeType.Active;
    if (!anyResults && !graftDue && !sweepDue && !incrementalDue && !rootAdaptDue && !advocateDue)
    {
      return;
    }

    long t0 = Stopwatch.GetTimestamp();
    lock (pumpLock)
    {
      try
      {
        // Once the search has decided to stop, stop consuming budget (results still drain
        // so the stats and Shadow ledgers stay complete).
        bool searchStopping = Manager.StopStatus != MCGSManager.SearchStopStatus.Continue;

        if (!searchStopping && incrementalDue)
        {
          long tSweep = Stopwatch.GetTimestamp();
          RunIncrementalAndSubmit(iterator);
          Stats.TimeTargetSweepMs += MsSince(tSweep);
        }

        if (anyResults)
        {
          long tDrain = Stopwatch.GetTimestamp();
          DrainInbox(iterator.Engine, int.MaxValue);
          Stats.TimeDrainMs += MsSince(tDrain);
        }

        if (!searchStopping && graftDue)
        {
          long tGraft = Stopwatch.GetTimestamp();
          ushort rootTopBefore = TopNRootMoveRaw(iterator.Engine);
          bool applied = scheduler.AdvanceOnePly(iterator);
          if (applied && TopNRootMoveRaw(iterator.Engine) != rootTopBefore)
          {
            Stats.NumRootBestMoveChangesAfterGraft++;
          }
          Stats.TimeGraftAdvanceMs += MsSince(tGraft);
        }

        if (!searchStopping && sweepDue)
        {
          long tSweep = Stopwatch.GetTimestamp();
          RunSweepAndSubmit(iterator.Engine);
          sweepTimer.Restart();
          Stats.TimeTargetSweepMs += MsSince(tSweep);
        }

        if (rootAdaptDue)
        {
          if (searchStopping)
          {
            FinishRootAdapt(iterator.Engine, abandoned: true);
          }
          else
          {
            PumpRootAdapt(iterator);
          }
        }

        if (advocateDue)
        {
          PumpAdvocate(iterator, searchStopping);
        }
      }
      catch (Exception exc)
      {
        DisableAfterError(exc);
      }
      finally
      {
        Stats.TimePumpTotalMs += MsSince(t0);
      }
    }
  }


  /// <summary>
  /// In-gate entry point (Route 4), invoked from MCGSIterator.RunOnce after the batch's backup
  /// and before PostBackupRecomputeQIfEnabled, while the iterator holds the select/backup
  /// exclusion lock: drains completed stamp probes and applies them (stamp bytes + upward
  /// recompute), at most ProbeStampMaxAppliesPerBatch stamps per batch; the remainder waits in
  /// the inbox for the next batch of either iterator. No visits are issued and no coordinator
  /// gate is entered, so this is legal under DualOverlappedIterators (the other iterator can only
  /// be in its NN evaluate phase, whose in-flight visits are invisible to the recompute).
  /// Deferred to the next batch when the other iterator is inside the post-gate pump, so the
  /// gate hold is never extended by waiting on pumpLock.
  /// </summary>
  /// <param name="iterator"></param>
  public void DrainInBackupGate(MCGSIterator iterator)
  {
    bool stampWork = Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active
                  && Volatile.Read(ref numPendingResults) != 0;
    bool attentionWork = advAttentionDirty;   // advocate: publish the pending attention snapshot
    if (disabledByError || endSearchDone || (!stampWork && !attentionWork))
    {
      return;
    }

    if (!Monitor.TryEnter(pumpLock))
    {
      Stats.NumGateDrainsSkippedBusy++;   // only ever incremented by the (single) gate holder
      return;
    }

    long t0 = Stopwatch.GetTimestamp();
    try
    {
      if (advAttentionDirty)
      {
        iterator.Engine.Graph.SetAttentionEntries(advPendingSnapshot);
        advAttentionDirty = false;
      }
      if (stampWork)
      {
        Stats.NumGateDrains++;
        int budget = Params.ProbeStampMaxAppliesPerBatch <= 0 ? int.MaxValue : Params.ProbeStampMaxAppliesPerBatch;
        DrainInbox(iterator.Engine, budget);
      }
    }
    catch (Exception exc)
    {
      DisableAfterError(exc);   // never let an exception escape while the gate is held
    }
    finally
    {
      double ms = MsSince(t0);
      Stats.TimeDrainMs += ms;
      Stats.TimePumpTotalMs += ms;
      if (ms > Stats.MaxGateDrainMs)
      {
        Stats.MaxGateDrainMs = ms;
      }
      Monitor.Exit(pumpLock);
    }
  }


  /// <summary>
  /// Finalizes the feature for this search: cancels/ends the probe session (no callbacks
  /// occur after EndSession returns), discards any late results, computes impact deltas
  /// and Shadow verdicts against the final graph, and prints the one-line summary when
  /// enabled. Idempotent; invoked from MCGSManager.DoSearchInner at end of search.
  /// </summary>
  public void EndSearch()
  {
    if (!endSearchDone && Params.RootAdaptMode == ParamsProbeGraft.RootAdaptModeType.Active)
    {
      try
      {
        if (rootAdaptActive)
        {
          FinishRootAdapt(Manager.Engine, abandoned: true);
        }
        if (rootAdaptFlipped && rootAdaptMoveRaw == TopNRootMoveRaw(Manager.Engine))
        {
          Stats.NumRootAdaptFlipsHeld++;
        }
      }
      catch (Exception exc)
      {
        DisableAfterError(exc);
      }
    }

    if (!endSearchDone && Params.AdvocateMode == ParamsProbeGraft.AdvocateModeType.Active)
    {
      try
      {
        if (advCampActive)
        {
          FinishAdvocateCampaign(Manager.Engine, abandoned: true);
        }
        advPendingEngagement = null;
        ReleaseAdvocateSolo();
        advRootCts?.Cancel();
        advFavCts?.Cancel();
        if (AdvocateContinuousActive && advRootMaxDepth > 0)
        {
          // The open-ended slots' final callbacks arrive after this session ends (and are dropped), so
          // the depth each slot reached this move is recorded here from the coordinator's own state.
          Stats.NumAdvocateSlotRuns++;
          Stats.AdvocateRootDepthSum += advRootMaxDepth;
          Stats.AdvocateFavDepthSum += advFavMaxDepth;
        }
        // Engagement conversion: did the final (top-N) root move become the Dissenter's move, and did it
        // leave the favourite the Dissenter argued against? Per search, latest engagement.
        if (advEngagedThisSearch && Manager.Engine != null)
        {
          Stats.NumAdvocateEngagedMoves++;
          ushort finalRaw = TopNRootMoveRaw(Manager.Engine);
          if (finalRaw == advEngagedDissenterRaw)
          {
            Stats.NumAdvocateConversions++;
          }
          if (finalRaw != advEngagedFavRaw)
          {
            Stats.NumAdvocateFavAbandoned++;
          }
        }
        // The attention snapshot must not outlive this coordinator: node indices survive only
        // until the next graph rewrite, and the next search's advocate re-establishes quickly.
        Manager.Engine?.Graph?.SetAttentionEntries(null);
      }
      catch (Exception exc)
      {
        DisableAfterError(exc);
      }
    }

    if (endSearchDone)
    {
      return;
    }
    endSearchDone = true;

    ShutdownSession();

    // Like the pump, end-of-search work must never propagate an exception into search.
    try
    {
      while (inbox.TryDequeue(out _))
      {
        Interlocked.Decrement(ref numPendingResults);
        Stats.NumResultsDiscardedAtEnd++;
      }

      Stats.NumDistinctNodesProbed = targeter.NumDistinctNodesProbed;

      if (Manager.Engine != null)
      {
        ComputeImpactStats(Manager.Engine.Graph);
        ComputeShadowVerdicts(Manager.Engine.Graph);
      }

      // ParamsSearch.DumpHybridSearchStats is the research switch and takes precedence over the
      // terse per-search line; Off leaves the legacy EnableStatsSummary behavior untouched.
      ParamsSearch.HybridStatsDumpType dumpMode = Manager.ParamsSearch.DumpHybridSearchStats;
      if (dumpMode == ParamsSearch.HybridStatsDumpType.Full)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan,
          Stats.SearchReport(Params.Mode, Manager.NumEvalsThisSearch, source?.Description));
      }
      else if (dumpMode == ParamsSearch.HybridStatsDumpType.Compact || Params.EnableStatsSummary)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan,
                                      Stats.SummaryLine(Params.Mode, Manager.NumEvalsThisSearch, ConsumerLabel));
      }

      // Optionally fold this search into the process-wide aggregate used by multi-search
      // harnesses (measurement campaigns, matches, tournament managers).
      if (ProbeGraftStats.EnableGlobalAccumulation)
      {
        ProbeGraftStats.AccumulateGlobal(Stats, Params.Mode, Manager.NumEvalsThisSearch);
      }
    }
    catch (Exception exc)
    {
      DisableAfterError(exc);
    }
  }


  /// <summary>
  /// Safety net: ensures the probe session is ended even if the search never completed
  /// normally. The source instance itself may be shared (registry factory) and is never
  /// disposed here.
  /// </summary>
  public void Dispose()
  {
    if (disposed)
    {
      return;
    }
    disposed = true;

    if (!endSearchDone)
    {
      endSearchDone = true;
      ShutdownSession();
    }

    cts.Dispose();
  }


  /// <summary>Retained per-event log lines (populated when VerboseEventLogging).</summary>
  public IReadOnlyList<string> EventLog => eventLog;


  #region Result drain and trigger interpretation

  /// <summary>
  /// Verdict of the most recent completed root probe in this search: what would play at
  /// the root, and by how much it preferred that over the move the search was favoring when the
  /// probe was issued. Consumed at move commit (Shadow: recorded only).
  /// </summary>
  public readonly struct RootVerdict
  {
    /// <summary>False for the default value (no root probe has completed this search).</summary>
    public readonly bool HasValue;
    /// <summary>The probe's preferred root move.</summary>
    public readonly MGMove ProbeBestMove;
    /// <summary>The move the search favored when the probe was issued.</summary>
    public readonly MGMove SearchDominantMove;
    /// <summary>The probe's own margin (cp) of its move over the search's favorite.</summary>
    public readonly int MarginCp;
    /// <summary>Depth the probe search reached.</summary>
    public readonly int ProbeDepth;

    public RootVerdict(MGMove probeBestMove, MGMove searchDominantMove, int marginCp, int probeDepth)
    {
      HasValue = true;
      ProbeBestMove = probeBestMove;
      SearchDominantMove = searchDominantMove;
      MarginCp = marginCp;
      ProbeDepth = probeDepth;
    }
  }


  /// <summary>Most recent root probe verdict of this search (see RootVerdict).</summary>
  public RootVerdict LastRootVerdict { get; private set; }


  /// <summary>
  /// Records the root probe verdict. A verdict only counts as a veto candidate when the probe both
  /// searched deeply enough to be trusted and preferred a different move by a clear margin; the
  /// comparison against the move actually played happens at commit, since the search continues
  /// after this probe returns and may change its mind on its own.
  /// </summary>
  /// <param name="snapshot"></param>
  /// <param name="result"></param>
  void RecordRootVerdict(ProbeGraftTargetSnapshot snapshot, ProbeResult result)
  {
    Stats.NumRootProbeVerdicts++;

    if (result.Depth < Params.RootVetoMinProbeDepth || result.DominantMoveScoreCp == int.MinValue)
    {
      return;
    }

    int marginCp = result.BestScoreCp - result.DominantMoveScoreCp;
    LastRootVerdict = new RootVerdict(result.BestMove, snapshot.DominantMove, marginCp, result.Depth);

    if (result.BestMove != snapshot.DominantMove && marginCp >= Params.RootVetoMinMarginCp)
    {
      Stats.NumRootVetoCandidates++;
    }
  }


  /// <summary>
  /// Drains and interprets completed probe results. At most <paramref name="maxStampApplies"/>
  /// results that actually STAMP (bytes written + ancestors recomputed) are consumed per call;
  /// agreements, clears and rejections are cheap and not counted. Results left in the inbox are
  /// consumed by a later drain (of either iterator).
  /// </summary>
  void DrainInbox(MCGSEngine engine, int maxStampApplies)
  {
    int applied = 0;
    while (applied < maxStampApplies && inbox.TryDequeue(out ProbeResult result))
    {
      Interlocked.Decrement(ref numPendingResults);
      long stampsBefore = Stats.NumStamps;
      InterpretResult(engine, result);
      if (Stats.NumStamps != stampsBefore)
      {
        applied++;
      }
    }
    if (applied >= maxStampApplies && Volatile.Read(ref numPendingResults) != 0)
    {
      Stats.NumGateDrainsBudgetBound++;
    }
  }


  /// <summary>
  /// Applies the trigger rules to one completed probe result: staleness check, then
  /// refutation trigger (primary), discovery trigger (secondary), else agreement.
  /// Ambiguous snapshots are discarded and counted - a discarded probe costs nothing.
  /// </summary>
  void InterpretResult(MCGSEngine engine, ProbeResult result)
  {
    // Advocate Dissenter probes carry their own token; the verdict is only STORED here (this can
    // run inside the backup gate) and applied at the next advocate pump, which has the iterator
    // needed for traces/campaigns.
    if (result.RequesterToken is AdvocateToken advToken)
    {
      if (advToken.Continuous)
      {
        InterpretContinuousResult(engine, advToken, result);
        return;
      }
      if (result.IsInterim)
      {
        return;   // not requested for budgeted probes
      }
      switch (advToken.Slot)
      {
        case AdvocateSlot.Root: advRootInFlight = false; break;
        case AdvocateSlot.Fav: advFavInFlight = false; break;
        default: advProbeInFlight = false; break;
      }
      if (!result.Completed)
      {
        Stats.NumProbesIncomplete++;
        return;
      }
      Stats.NumProbesCompleted++;
      Stats.ProbeLatencySumMs += result.ElapsedMilliseconds;
      Stats.ProbeNodesSum += result.Nodes;
      Stats.ProbeDepthSum += result.Depth;
      switch (advToken.Slot)
      {
        case AdvocateSlot.Root:
          advRootLatest = result;
          advRootLatestSeq = advToken.Seq;
          Stats.NumAdvocateRootProbes++;
          break;
        case AdvocateSlot.Fav:
          if (!result.BestMove.IsNull && result.BestMove != advToken.FavMove && !advDualUnsupported)
          {
            // The source ran an unrestricted search: RestrictToMove is not honoured. Fall back to
            // the single-probe form for the rest of this search (and log once).
            advDualUnsupported = true;
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
              "[ProbeGraft] probe source ignores ProbeRequest.RestrictToMove; advocate dual probes disabled (single-probe form).");
            break;
          }
          advFavLatest = result;
          advFavLatestSeq = advToken.Seq;
          advFavLatestRaw = advToken.FavRaw;
          advFavLatestMove = advToken.FavMove;
          Stats.NumAdvocateFavProbes++;
          break;
        default:
          advPendingVerdict = result;
          advPendingToken = advToken;
          break;
      }
      return;
    }

    ProbeGraftTargetSnapshot snapshot = (ProbeGraftTargetSnapshot)result.RequesterToken;

    if (!result.Completed)
    {
      Stats.NumProbesIncomplete++;
      return;
    }

    Stats.NumProbesCompleted++;
    Stats.ProbeLatencySumMs += result.ElapsedMilliseconds;
    if (result.ElapsedMilliseconds > Stats.ProbeLatencyMaxMs)
    {
      Stats.ProbeLatencyMaxMs = result.ElapsedMilliseconds;
    }
    Stats.ProbeNodesSum += result.Nodes;
    Stats.ProbeDepthSum += result.Depth;

    // Staleness: within a search node indices are stable (graph rewrites happen between
    // searches), but the node must still be eligible and, for refutation triggering, the
    // dominant child must still be the snapshot's.
    GNode node = TryResolveNode(engine.Graph, snapshot.NodeIndex);
    bool eligible = !node.IsNull
                 && node.IsEvaluated
                 && node.N > 0
                 && !node.Terminal.IsTerminal()
                 && node.NumPolicyMoves > 0
                 && (!node.IsSearchRoot || Params.TargetIncludeSearchRoot);
    if (!eligible)
    {
      Stats.NumResultsStale++;
      return;
    }

    if (Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active)
    {
      HandleStampResult(engine, node, snapshot, result);
      return;
    }

    // The dominant child must still be the snapshot's dominant child; when the visit
    // distribution has shifted since the probe was issued, the result answers a question
    // the graph is no longer asking - discard it (a discarded probe costs nothing).
    if (snapshot.DominantChildIndex >= 0
     && DominantChildIndex(node) != snapshot.DominantChildIndex)
    {
      Stats.NumResultsStale++;
      return;
    }

    if (Params.RootVetoMode != ParamsProbeGraft.RootVetoModeType.Disabled && node.IsSearchRoot)
    {
      RecordRootVerdict(snapshot, result);
    }

    bool refutation = result.DominantMoveScoreCp != int.MinValue
                   && snapshot.DominantChildIndex >= 0
                   && result.BestMove != snapshot.DominantMove
                   && (result.BestScoreCp - result.DominantMoveScoreCp) >= Params.TriggerMarginCp
                   && result.Depth >= Params.TriggerMinProbeDepth
                   && snapshot.DominantFractionGateHeld;

    if (refutation)
    {
      Stats.NumTriggersRefutation++;
      Stats.RecordTriggerDepth(snapshot.Depth);
      MGMove[] moves = BuildRefutationLine(snapshot.DominantMove, result.RefutationPV);
      HandleTrigger(node, snapshot, result, moves, isRefutation: true,
                    marginCp: result.BestScoreCp - result.DominantMoveScoreCp);
      return;
    }

    // Discovery trigger: the prober likes a move the policy effectively ignores.
    if (Params.PriorNudgeFloor > 0
     && result.Depth >= Params.TriggerMinProbeDepth
     && result.PV.Count > 0
     && result.BestMove != snapshot.DominantMove)
    {
      float bestMovePrior = PriorOfMoveAtNode(node, result.BestMove);
      if (!float.IsNaN(bestMovePrior) && bestMovePrior < Params.PriorNudgeFloor)
      {
        Stats.NumTriggersDiscovery++;
        Stats.RecordTriggerDepth(snapshot.Depth);
        MGMove[] moves = TruncateLine(result.PV);
        HandleTrigger(node, snapshot, result, moves, isRefutation: false, marginCp: 0);
        return;
      }
    }

    // Agreement - the expected common case; its cheapness is the point.
    Stats.NumProbesAgreed++;
  }


  /// <summary>
  /// AB stamps (ProbeStampMode): a first probe flags |A - Q| >= theta and submits a confirmation probe
  /// (or stamps directly when no confirmation is configured); a confirmation probe that keeps the
  /// sign and reaches the fraction threshold stamps the node (ProbeStamps.Stamp) and recomputes its
  /// ancestors up to the search root (ProbeStamps.RecomputeUpward). Runs on the pump thread at the
  /// quiescent point; never issues visits.
  /// </summary>
  void HandleStampResult(MCGSEngine engine, GNode node, ProbeGraftTargetSnapshot snapshot, ProbeResult result)
  {
    double a = ProbeCpToLogisticQ(result.BestScoreCp);
    double q = node.Q;
    if (double.IsNaN(q))
    {
      Stats.NumResultsStale++;
      return;
    }
    double d = a - q;
    float theta = Params.ProbeStampTheta;

    if (!snapshot.IsConfirm)
    {
      if (Math.Abs(d) < theta)
      {
        // Agreement (or a re-probe finding the disagreement gone): clear any earlier stamp.
        Stats.NumProbesAgreed++;
        if (node.HasProbeStamp)
        {
          ProbeStamps.Clear(node);
          Stats.NumStampsCleared++;
        }
        return;
      }

      Stats.NumStampFlags++;
      if (Params.ProbeStampConfirmMultiplier > 0)
      {
        ProbeGraftTargetSnapshot confirm = new()
        {
          NodeIndex = snapshot.NodeIndex,
          N = snapshot.N,
          Q = snapshot.Q,
          Depth = snapshot.Depth,
          DominantChildIndex = -1,
          MovesFromRoot = snapshot.MovesFromRoot,
          Priority = snapshot.Priority,
          IsConfirm = true,
          FirstD = d,
          ProbeNodes = snapshot.ProbeNodes * Params.ProbeStampConfirmMultiplier,
        };
        if (SubmitStampProbe(confirm))
        {
          Stats.NumStampConfirmsSubmitted++;
        }
        else
        {
          Stats.NumProbesRejectedBySource++;
        }
        return;
      }
      ApplyStamp(engine, node, snapshot, a, result.BestScoreCp);
      return;
    }

    // Confirmation result.
    bool ok = Math.Sign(d) == Math.Sign(snapshot.FirstD) && Math.Abs(d) >= Params.ProbeStampConfirmFraction * theta;
    if (!ok)
    {
      Stats.NumStampConfirmsRejected++;
      if (node.HasProbeStamp)
      {
        ProbeStamps.Clear(node);
        Stats.NumStampsCleared++;
      }
      return;
    }
    ApplyStamp(engine, node, snapshot, a, result.BestScoreCp);
  }


  void ApplyStamp(MCGSEngine engine, GNode node, ProbeGraftTargetSnapshot snapshot, double a, int probeCp)
  {
    if (node.N > Params.ProbeStampMaxN * 4)
    {
      Stats.NumStampsStaleN++;   // the node outgrew the probe while it was in flight
      return;
    }

    // Precision-indexed weight by the probe's |cp| (ProbeStampCpWeights): scales n0; 0 suppresses.
    int n0 = ProbeStamps.WeightedN0(snapshot.N, ProbeStamps.CpWeight(probeCp, cpWeightTable));
    if (n0 <= 0)
    {
      Stats.NumStampsWeightedOut++;
      return;
    }
    ProbeStamps.Stamp(node, a, n0);
    Stats.NumStamps++;

    long t0 = Stopwatch.GetTimestamp();
    ushort rootTopBefore = TopNRootMoveRaw(engine);
    ProbeStamps.RecomputeResult rr = ProbeStamps.RecomputeUpward(node, engine.SearchRootNode, Manager.ParamsSelect,
                                                                 recomputeQueue, recomputeSeen);
    Stats.NumStampAncestorsMoved += rr.AncestorsMoved;
    if (TopNRootMoveRaw(engine) != rootTopBefore)
    {
      Stats.NumStampRootTopChanges++;
    }
    Stats.TimeStampRecomputeMs += MsSince(t0);

    if (Params.VerboseEventLogging)
    {
      LogEvent($"stamp node=#{snapshot.NodeIndex} depth={snapshot.Depth} N={node.N} n0={snapshot.N} Q={node.Q:+0.000;-0.000} A={a:+0.000;-0.000} "
             + $"recomputed={rr.Recomputed} moved={rr.AncestorsMoved} rootQ {rr.RootQBefore:+0.000;-0.000}->{rr.RootQAfter:+0.000;-0.000}");
    }
  }


  /// <summary>Submits one stamp-mode probe for the snapshot (position rebuilt from the walk's move path).</summary>
  bool SubmitStampProbe(ProbeGraftTargetSnapshot snapshot)
  {
    PositionWithHistory position = new PositionWithHistory(SearchRootPositionWithHistory());
    foreach (MGMove move in snapshot.MovesFromRoot)
    {
      position.AppendMove(move);
    }
    ProbeRequest request = new()
    {
      RequestID = Interlocked.Increment(ref nextRequestID),
      RequesterToken = snapshot,
      Position = position,
      DominantMove = default,          // value only: no verification search of the favourite
      NodeBudget = snapshot.ProbeNodes,
      CancellationToken = cts.Token,
    };
    return session.SubmitProbe(request);
  }


  /// <summary>Probe cp (side to move) -> Ceres logistic Q; mate-range scores pin to +/-1.</summary>
  static double ProbeCpToLogisticQ(int cp)
  {
    if (cp >= 20_000) return 1.0;
    if (cp <= -20_000) return -1.0;
    return Math.Clamp(EncodedEvalLogistic.CentipawnToLogistic(cp), -1.0, 1.0);
  }


  /// <summary>
  /// Routes a triggered event: Active mode queues a graft plan; Shadow mode records the
  /// would-have-grafted event for its end-of-search verdict.
  /// </summary>
  void HandleTrigger(GNode node, ProbeGraftTargetSnapshot snapshot, ProbeResult result,
                     MGMove[] moves, bool isRefutation, int marginCp)
  {
    if (Params.VerboseEventLogging)
    {
      LogEvent($"trigger {(isRefutation ? "ref" : "disc")} node=#{snapshot.NodeIndex} N={snapshot.N} "
             + $"dom={snapshot.DominantMove} best={result.BestMove} marginCp={marginCp} depth={result.Depth}");
    }

    if (moves.Length == 0)
    {
      return;
    }

    if (Params.Mode == ParamsProbeGraft.ModeType.Active)
    {
      scheduler.Enqueue(new ProbeGraftPlan()
      {
        AnchorNodeIndex = snapshot.NodeIndex,
        Moves = moves,
        IsRefutation = isRefutation,
        MarginCp = marginCp,
        ProbeDepth = result.Depth,
      });
    }
    else
    {
      double dominantEdgeQ = snapshot.DominantChildIndex >= 0
                          && snapshot.DominantChildIndex < node.NumEdgesExpanded
                           ? node.ChildEdgeAtIndex(snapshot.DominantChildIndex).Q
                           : double.NaN;
      shadowEvents.Add(new ShadowEvent()
      {
        AnchorNodeIndex = snapshot.NodeIndex,
        IsRefutation = isRefutation,
        DominantChildIndexAtTrigger = snapshot.DominantChildIndex,
        DominantEdgeQAtTrigger = dominantEdgeQ,
        AnchorQAtTrigger = node.Q,
        ProbeBestMove = result.BestMove,
      });
    }
  }


  MGMove[] BuildRefutationLine(MGMove dominantMove, IReadOnlyList<MGMove> refutationPV)
  {
    int numReply = Math.Min(refutationPV.Count, Params.GraftMaxPlies - 1);
    MGMove[] moves = new MGMove[1 + numReply];
    moves[0] = dominantMove;
    for (int i = 0; i < numReply; i++)
    {
      moves[1 + i] = refutationPV[i];
    }
    return moves;
  }


  MGMove[] TruncateLine(IReadOnlyList<MGMove> pv)
  {
    int num = Math.Min(pv.Count, Params.GraftMaxPlies);
    MGMove[] moves = new MGMove[num];
    for (int i = 0; i < num; i++)
    {
      moves[i] = pv[i];
    }
    return moves;
  }

  #endregion

  #region Target sweep and probe submission

  PositionWithHistory searchRootPositionWithHistory;


  /// <summary>
  /// Returns the position history of the SEARCH root (cached; fixed for the duration of a search).
  ///
  /// Manager.StartPosAndPriorMoves is the history of the GRAPH root, which is the same node only
  /// when the graph is not reused. Under graph reuse (as in game play) the search root sits below
  /// the graph root, so the moves along Engine.SearchRootPathFromGraphRoot must be appended before
  /// the sweep's root-relative move path (otherwise those moves are illegal in the resulting
  /// position and probe submission throws).
  /// </summary>
  /// <returns></returns>
  PositionWithHistory SearchRootPositionWithHistory()
  {
    if (searchRootPositionWithHistory == null)
    {
      PositionWithHistory position = new PositionWithHistory(Manager.StartPosAndPriorMoves);
      GraphRootToSearchRootNodeInfo[] pathFromGraphRoot = Manager.Engine.SearchRootPathFromGraphRoot;
      if (pathFromGraphRoot != null)
      {
        foreach (GraphRootToSearchRootNodeInfo nodeInfo in pathFromGraphRoot)
        {
          position.AppendMove(nodeInfo.MoveToChild);
        }
      }
      searchRootPositionWithHistory = position;
    }

    return searchRootPositionWithHistory;
  }


  /// <summary>
  /// Runs one target sweep and submits probe requests for the selected snapshots, up to
  /// the in-flight cap. Request positions are the search-root history plus the walk's move
  /// path (preserving repetition/50-move context for the prober).
  /// </summary>
  void RunSweepAndSubmit(MCGSEngine engine)
  {
    Stats.NumTargetSweeps++;
    bool stampModeBudget = Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active;
    if (stampModeBudget && ProbeBudgetExhausted)
    {
      return;   // per-search probe budget exhausted: no more sweeps this search
    }

    List<ProbeGraftTargetSnapshot> selected = targeter.Sweep(engine);
    Stats.NumTargetsSelected += selected.Count;
    SubmitSnapshots(selected);
  }


  /// <summary>
  /// Probes still allowed this search under ProbeStampMaxProbesPerSearch (0 = unlimited: the
  /// prober is then paced only by MaxInFlightProbes and the source's saturation backoff, so a
  /// long move keeps the pool busy to the end instead of going quiet once a fixed count is spent).
  /// </summary>
  long RemainingProbeBudget => Params.ProbeStampMaxProbesPerSearch <= 0
                               ? long.MaxValue
                               : Params.ProbeStampMaxProbesPerSearch - Stats.NumProbesSubmitted;

  bool ProbeBudgetExhausted => RemainingProbeBudget <= 0;

  /// <summary>Incremental targeting runs only when at least this many probe slots are free.</summary>
  const int MIN_INCREMENTAL_CAPACITY = 4;


  /// <summary>
  /// Incremental stamp-mode targeting (ProbeStampIncrementalTargeting): collects the
  /// threshold-crossing nodes on the batch's backed-up paths and submits them, subject to
  /// the same in-flight cap and per-search budget as the sweep.
  /// </summary>
  void RunIncrementalAndSubmit(MCGSIterator iterator)
  {
    if (ProbeBudgetExhausted)
    {
      return;
    }
    // Collect only what can be submitted now: when the prober is saturated (in-flight cap reached)
    // the pass is skipped, and otherwise bounded to 2x the free slots -- the session-5 match
    // showed a 10:1 candidates:submissions ratio (1.45 s/search) without this bound.
    if (sourceSaturatedBackoff > 0)
    {
      sourceSaturatedBackoff--;
      return;   // the shared probe source rejected a submission recently: let it drain
    }
    int capacity = Params.MaxInFlightProbes - session.NumInFlight;
    capacity = (int)Math.Min(capacity, RemainingProbeBudget);
    if (capacity < MIN_INCREMENTAL_CAPACITY)
    {
      return;   // nothing (worth) submitting; the nodes will be seen again when revisited
    }

    Stats.NumTargetSweeps++;
    List<ProbeGraftTargetSnapshot> selected = targeter.CollectFromPaths(iterator, maxCandidates: 2 * capacity);
    Stats.NumTargetsSelected += selected.Count;
    SubmitSnapshots(selected);
  }


  /// <summary>
  /// Submits probe requests for the selected snapshots (highest priority first), up to the
  /// in-flight cap and, in stamp mode, the per-search budget. Request positions are the
  /// search-root history plus the snapshot's move path (preserving repetition/50-move context).
  /// </summary>
  void SubmitSnapshots(List<ProbeGraftTargetSnapshot> selected)
  {
    bool stampModeBudget = Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active;
    int capacity = Params.MaxInFlightProbes - session.NumInFlight;
    if (stampModeBudget)
    {
      capacity = (int)Math.Min(capacity, RemainingProbeBudget);
    }
    foreach (ProbeGraftTargetSnapshot snapshot in selected)
    {
      if (capacity <= 0)
      {
        break;
      }

      PositionWithHistory position = new PositionWithHistory(SearchRootPositionWithHistory());
      foreach (MGMove move in snapshot.MovesFromRoot)
      {
        position.AppendMove(move);
      }

      bool stampMode = Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active;
      if (stampMode)
      {
        snapshot.ProbeNodes = (int)Math.Clamp((long)Params.ProbeStampBudgetRatio * snapshot.N, Params.ProbeNodeBudget, Params.ProbeStampBudgetMax);
      }
      ProbeRequest request = new()
      {
        RequestID = Interlocked.Increment(ref nextRequestID),
        RequesterToken = snapshot,
        Position = position,
        DominantMove = stampMode ? default : snapshot.DominantMove,
        NodeBudget = stampMode ? snapshot.ProbeNodes : Params.ProbeNodeBudget,
        CancellationToken = cts.Token,
      };

      if (session.SubmitProbe(request))
      {
        Stats.NumProbesSubmitted++;
        Stats.RecordProbeDepth(snapshot.Depth);
        targeter.RecordProbeSubmitted(snapshot);
        capacity--;
      }
      else
      {
        // The source's queue is shared by every concurrent search (the per-session in-flight cap
        // cannot see it): treat a rejection as saturation, stop this pass and back off.
        Stats.NumProbesRejectedBySource++;
        sourceSaturatedBackoff = SOURCE_SATURATED_BACKOFF_PUMPS;
        break;
      }
    }
  }


  /// <summary>Pumps left to skip incremental targeting after the probe source rejected a submission.</summary>
  int sourceSaturatedBackoff;
  const int SOURCE_SATURATED_BACKOFF_PUMPS = 8;

  #endregion

  #region End-of-search impact and Shadow verdicts

  void ComputeImpactStats(Graph graph)
  {
    foreach ((int anchorIndex, double qBefore) in scheduler.CompletedGraftImpacts)
    {
      GNode anchor = TryResolveNode(graph, anchorIndex);
      if (anchor.IsNull)
      {
        continue;
      }

      double delta = Math.Abs(anchor.Q - qBefore);
      Stats.SumAbsAnchorQDeltaAtEnd += delta;
      if (delta > Stats.MaxAbsAnchorQDeltaAtEnd)
      {
        Stats.MaxAbsAnchorQDeltaAtEnd = delta;
      }
      if (delta > LARGE_DELTA_THRESHOLD)
      {
        Stats.NumAnchorsQMovedOverThreshold++;
      }

      LogEvent($"impact anchor=#{anchorIndex} qBefore={qBefore:F3} qEnd={anchor.Q:F3} absDQ={delta:F3}");
    }
  }


  /// <summary>
  /// Computes the per-event Shadow verdicts against the final graph: an event is
  /// CONFIRMED when plain search resolved it on its own - for a refutation event, the
  /// anchor's visit-dominant child changed away from the refuted move or the refuted
  /// edge's (child-perspective) Q rose by more than the threshold (the move was revealed
  /// worse for the anchor); for a discovery event, the probe's move became visit-dominant
  /// or the anchor Q rose by more than the threshold. Everything else is UNRESOLVED -
  /// the cases where Active mode would differ from baseline.
  /// </summary>
  void ComputeShadowVerdicts(Graph graph)
  {
    Stats.ShadowNumEvents = shadowEvents.Count;

    foreach (ShadowEvent shadowEvent in shadowEvents)
    {
      bool confirmed = false;
      GNode node = TryResolveNode(graph, shadowEvent.AnchorNodeIndex);
      if (!node.IsNull && node.NumEdgesExpanded > 0)
      {
        int dominantNow = DominantChildIndex(node);

        if (shadowEvent.IsRefutation)
        {
          bool dominantChanged = dominantNow != shadowEvent.DominantChildIndexAtTrigger;
          bool refutedEdgeQRose = !double.IsNaN(shadowEvent.DominantEdgeQAtTrigger)
                               && shadowEvent.DominantChildIndexAtTrigger < node.NumEdgesExpanded
                               && (node.ChildEdgeAtIndex(shadowEvent.DominantChildIndexAtTrigger).Q
                                   - shadowEvent.DominantEdgeQAtTrigger) > LARGE_DELTA_THRESHOLD;
          confirmed = dominantChanged || refutedEdgeQRose;
        }
        else
        {
          bool becameDominant = false;
          if (dominantNow >= 0)
          {
            MGPosition nodePos = node.CalcPosition();
            becameDominant = node.ChildEdgeAtIndex(dominantNow).MoveMGFromPos(in nodePos)
                          == shadowEvent.ProbeBestMove;
          }
          bool anchorQRose = (node.Q - shadowEvent.AnchorQAtTrigger) > LARGE_DELTA_THRESHOLD;
          confirmed = becameDominant || anchorQRose;
        }
      }

      if (confirmed)
      {
        Stats.ShadowNumConfirmedBySearch++;
      }
      else
      {
        Stats.ShadowNumUnresolved++;
      }
    }
  }

  #endregion

  #region Helpers

  /// <summary>
  /// Resolves a node index against the graph, returning a null GNode when out of range.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="nodeIndex"></param>
  internal static GNode TryResolveNode(Graph graph, int nodeIndex)
  {
    int numUsed = graph.Store.NodesStore.NumUsedNodes;
    if (nodeIndex < GNodeStore.FIRST_ALLOCATED_INDEX
     || nodeIndex >= GNodeStore.FIRST_ALLOCATED_INDEX + numUsed)
    {
      return default;
    }

    return graph[nodeIndex];
  }


  /// <summary>
  /// Index of the node's most-visited expanded child edge (-1 when no child has visits).
  /// </summary>
  /// <param name="node"></param>
  static int DominantChildIndex(GNode node)
  {
    int dominantIndex = -1;
    int dominantN = 0;
    int numExpanded = node.NumEdgesExpanded;
    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = node.ChildEdgeAtIndex(slot);
      if (edge.N > dominantN)
      {
        dominantN = edge.N;
        dominantIndex = slot;
      }
    }
    return dominantIndex;
  }


  /// <summary>
  /// Policy prior of the given move at the node (expanded edge or unexpanded header),
  /// NaN when the move is absent from the node's policy move list.
  /// </summary>
  static float PriorOfMoveAtNode(GNode node, MGMove move)
  {
    if (node.IsPendingPolicyCopy)
    {
      return float.NaN;
    }

    MGPosition nodePos = node.CalcPosition();
    int numPolicy = node.NumPolicyMoves;
    int numExpanded = node.NumEdgesExpanded;
    for (int i = 0; i < numPolicy; i++)
    {
      if (i < numExpanded)
      {
        GEdge edge = node.ChildEdgeAtIndex(i);
        if (edge.MoveMGFromPos(in nodePos) == move)
        {
          return edge.P;
        }
      }
      else
      {
        Graphs.GEdgeHeaders.GEdgeHeaderStruct header = node.EdgeHeadersSpan[i];
        if (Chess.MoveGen.Converters.ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(header.Move, in nodePos) == move)
        {
          return header.P;
        }
      }
    }

    return float.NaN;
  }


  /// <summary>
  /// Raw encoded move of the search root's most-visited expanded edge
  /// (cheap best-move snapshot for the root-flip impact counter).
  /// </summary>
  #region Root adaptive directed visits (LOCUS rootadapt)

  // One campaign per search: once the budget fraction has elapsed and the latest root probe
  // prefers a move other than the current top-N move, force visits into that move one chunk per
  // pump until it is best-by-Q at the root (flip) or the cap is reached (roll back).
  bool rootAdaptDone;
  bool rootAdaptActive;
  bool rootAdaptFlipped;
  int rootAdaptSlot = -1;
  ushort rootAdaptMoveRaw = ushort.MaxValue;
  int rootAdaptVisitsApplied;
  int rootAdaptCap;


  /// <summary>Fraction of this search's budget consumed so far (1 when unknown).</summary>
  double SearchProgressFraction()
  {
    SearchLimit limit = Manager.SearchLimit;
    if (limit == null || limit.Value <= 0)
    {
      return 1;
    }
    if (limit.Type == SearchLimitType.NodesPerTree)
    {
      double remaining = limit.Value - Manager.RootNWhenSearchStarted;
      return remaining <= 0 ? 1 : Manager.NumNodesVisitedThisSearch / remaining;
    }
    if (limit.IsNodesLimit)
    {
      return Manager.NumNodesVisitedThisSearch / (double)limit.Value;
    }
    return (DateTime.Now - Manager.StartTimeThisSearch).TotalSeconds / limit.Value;
  }


  void PumpRootAdapt(MCGSIterator iterator)
  {
    MCGSEngine engine = iterator.Engine;
    GNode root = engine.SearchRootNode;

    if (!rootAdaptActive)
    {
      if (!LastRootVerdict.HasValue || SearchProgressFraction() < Params.RootAdaptTriggerFraction)
      {
        return;
      }

      MGPosition rootPos = engine.SearchRootPosMG;
      int slot = ProbeGraftScheduler.ResolveMoveToChildIndex(root, in rootPos, LastRootVerdict.ProbeBestMove);
      if (slot < 0)
      {
        rootAdaptDone = true;
        return;
      }
      ushort probeRaw = (slot < root.NumEdgesExpanded ? root.ChildEdgeAtIndex(slot).Move
                                                        : root.EdgeHeadersSpan[slot].Move).RawValue;
      if (probeRaw == TopNRootMoveRaw(engine))
      {
        rootAdaptDone = true;   // the search already prefers the probe's move
        return;
      }

      SearchLimit limit = Manager.SearchLimit;
      int cap = Params.RootAdaptMaxVisits;
      if (limit != null && limit.IsNodesLimit && limit.Value > 0)
      {
        cap = Math.Min(cap, Math.Max(Params.RootAdaptVisitsPerPump, (int)(Params.RootAdaptMaxVisitsFraction * limit.Value)));
      }

      rootAdaptActive = true;
      rootAdaptSlot = slot;
      rootAdaptMoveRaw = probeRaw;
      rootAdaptVisitsApplied = 0;
      rootAdaptCap = cap;
      Stats.NumRootAdaptTriggers++;
    }

    int chunk = Math.Min(Params.RootAdaptVisitsPerPump, rootAdaptCap - rootAdaptVisitsApplied);
    DirectedVisits.Outcome step = chunk > 0
      ? DirectedVisits.ForceChildVisits(iterator, root, rootAdaptSlot, chunk)
      : new DirectedVisits.Outcome(0, 0, 0, "CapReached");
    rootAdaptVisitsApplied += step.VisitsApplied;
    Stats.NumRootAdaptVisits += step.VisitsApplied;

    bool decided = false;
    if (rootAdaptSlot < root.NumEdgesExpanded)
    {
      GEdge edge = root.ChildEdgeAtIndex(rootAdaptSlot);
      if (edge.Move.RawValue != rootAdaptMoveRaw)
      {
        FinishRootAdapt(engine, abandoned: true);   // slot reordered: abandon (rollback)
        return;
      }
      if (edge.Type != GEdgeStruct.EdgeType.Uninitialized
          && -edge.Q >= DirectedVisits.BestSiblingValue(root, rootAdaptSlot) + Params.RootAdaptMarginQ)
      {
        rootAdaptFlipped = true;
        decided = true;
      }
    }

    if (decided)
    {
      Stats.NumRootAdaptFlips++;
      rootAdaptActive = false;
      rootAdaptDone = true;
      if (Params.VerboseEventLogging)
      {
        Console.WriteLine($"[ROOTADAPT] flip after {rootAdaptVisitsApplied} visits (cap {rootAdaptCap})");
      }
      return;
    }

    if (step.Abort != null || step.VisitsApplied == 0 || rootAdaptVisitsApplied >= rootAdaptCap)
    {
      FinishRootAdapt(engine, abandoned: true);
    }
  }


  /// <summary>Ends an unflipped campaign: rolls its visits back (if configured) and marks it done.</summary>
  void FinishRootAdapt(MCGSEngine engine, bool abandoned)
  {
    if (rootAdaptActive && abandoned && !rootAdaptFlipped && Params.RootAdaptRollback && rootAdaptVisitsApplied > 0)
    {
      GNode root = engine.SearchRootNode;
      if (rootAdaptSlot >= 0 && rootAdaptSlot < root.NumEdgesExpanded
          && root.ChildEdgeAtIndex(rootAdaptSlot).Move.RawValue == rootAdaptMoveRaw)
      {
        DirectedVisits.ForgetEdgeVisits(root, rootAdaptSlot, rootAdaptVisitsApplied);
      }
      Stats.NumRootAdaptRollbacks++;
      if (Params.VerboseEventLogging)
      {
        Console.WriteLine($"[ROOTADAPT] no flip after {rootAdaptVisitsApplied} visits (cap {rootAdaptCap}); rolled back");
      }
    }
    rootAdaptActive = false;
    rootAdaptDone = true;
  }

  #endregion


  #region Advocate (SWARM Part XI: in-search Dissenter + frontier attention + root campaigns)

  /// <summary>Which Dissenter search a probe belongs to: the single combined probe, or (dual-probe
  /// form) the persistent unrestricted root search / the search restricted to the favourite.</summary>
  enum AdvocateSlot { Single, Root, Fav }

  /// <summary>Correlation token of a Dissenter probe: sequence + the favourite it questioned + slot.</summary>
  sealed record AdvocateToken(int Seq, ushort FavRaw, MGMove FavMove, AdvocateSlot Slot = AdvocateSlot.Single, bool Continuous = false);

  /// <summary>One live attention target (hysteresis bookkeeping for retirement).</summary>
  sealed class AdvocateTargetState
  {
    public char Side;                 // 'A' = along the prober's PV, 'R' = along the refutation of the favourite
    public int TrackedIndex = -1;     // node the hysteresis watches (first chain child)
    public double QAtFlag;
    public int NAtFlag;
    public bool Retired;
    public readonly List<(int Parent, int Child)> EntryKeys = new();
  }

  // Dissenter state.
  int advSeq;
  bool advProbeInFlight;
  long advLastVerdictTicks;
  ushort advLastVerdictFavRaw = ushort.MaxValue;
  ProbeResult advPendingVerdict;
  AdvocateToken advPendingToken;

  // Engagement-conversion bookkeeping (per search): the pair argued by the LAST engagement, scored at
  // EndSearch against the final top-N root move (see ProbeGraftStats.NumAdvocateEngagedMoves).
  bool advEngagedThisSearch;
  ushort advEngagedDissenterRaw = ushort.MaxValue;
  ushort advEngagedFavRaw = ushort.MaxValue;

  /// <summary>An engaged verdict whose actions (frontier walks/traces, root entry, campaign) are pending
  /// the transient solo window (applied immediately when no window is needed). RefutationOnly = the
  /// dual-probe favourite alarm: only the refutation-side frontier attention, no root entry, no campaign.</summary>
  sealed record AdvocateEngagement(AdvocateToken Token, ProbeResult Result, int Margin, float W, bool RefutationOnly = false);
  AdvocateEngagement advPendingEngagement;

  // Dual-probe state (Params.AdvocateDualProbes; SWARM doc XI.10). Slot Root: the persistent
  // unrestricted root search, resubmitted back-to-back (warm TT => continuous deepening), whose
  // verdict stays valid across favourite changes. Slot Fav: the search restricted to the current
  // favourite, resubmitted back-to-back and retargeted when the favourite changes (a stale in-flight
  // favourite probe finishes and is discarded: cancellation is per session). Verdicts are assembled
  // from the LATEST result of each slot; a (root, fav) pair is evaluated once.
  bool advRootInFlight;
  bool advFavInFlight;
  ushort advFavInFlightRaw = ushort.MaxValue;
  ProbeResult advRootLatest;
  int advRootLatestSeq = -1;
  ProbeResult advFavLatest;
  int advFavLatestSeq = -1;
  ushort advFavLatestRaw = ushort.MaxValue;
  MGMove advFavLatestMove;
  int advEvalRootSeq = -1;
  int advEvalFavSeq = -1;
  /// <summary>Set when the probe source ignored RestrictToMove (fell back to the single-probe form).</summary>
  bool advDualUnsupported;

  bool AdvocateDualActive => Params.AdvocateDualProbes && !advDualUnsupported;

  // Continuous dual probes (Params.AdvocateContinuousProbes; SWARM doc XI.11): open-ended searches
  // with per-iteration interim reports. The root slot runs for the whole move (advRootDone when it
  // ended by itself: mate / max depth); the favourite slot is cancelled (its own token) and restarted
  // whenever the favourite changes. Score-by-depth histories give matched-depth verdicts.
  CancellationTokenSource advRootCts;
  CancellationTokenSource advFavCts;
  bool advRootDone;
  ushort advFavTargetRaw = ushort.MaxValue;
  bool advFavCancelRequested;
  readonly Dictionary<int, (int ScoreCp, ushort BestRaw)> advRootHist = new();
  readonly Dictionary<int, int> advFavHist = new();
  int advRootMaxDepth;
  int advFavMaxDepth;
  long advRootStartTicks;
  long advFavStartTicks;
  int advLastEngageDepth = int.MinValue;
  ushort advLastEngageFavRaw = ushort.MaxValue;
  int advLastAlarmDepth = int.MinValue;
  ushort advLastAlarmFavRaw = ushort.MaxValue;
  (int Root, int Fav) advEvalDepthPair = (-1, -1);

  bool AdvocateContinuousActive => AdvocateDualActive && Params.AdvocateContinuousProbes;

  /// <summary>The pump iterator (directed visits are issued from iterator 0 only).</summary>
  const int PUMP_ITERATOR_ID = 0;

  // Solo-window accounting only (correctness never relies on it: SoloGranted is re-checked before
  // every directed batch): whether a granted window is currently being timed, and since when.
  bool advSoloTiming;
  long advSoloGrantTicks;

  /// <summary>True when directed visits (traces/campaigns) need the PhaseCoordinator solo window.</summary>
  bool AdvocateWantsSolo => Manager.ParamsSearch.Execution.DualOverlappedIterators
                         && Params.AdvocateDirectedVisitsEnabled;

  /// <summary>
  /// Non-blocking readiness check before any advocate directed visit: requests the solo window
  /// (if needed) and reports whether it is granted NOW (peer parked at its batch boundary or no
  /// peer loop active). A refusal defers the work to a later pump (two-pump handshake); iterator 0
  /// keeps running its own batches meanwhile, so nothing ever waits under pumpLock.
  /// </summary>
  bool AdvocateSoloReady(MCGSIterator iterator)
  {
    if (!AdvocateWantsSolo)
    {
      return true;
    }
    PhaseCoordinator coordinator = iterator.Engine.Coordinator;
    coordinator.RequestSolo(iterator.IteratorID);
    if (coordinator.SoloGranted(iterator.IteratorID))
    {
      if (!advSoloTiming)
      {
        advSoloTiming = true;
        advSoloGrantTicks = Stopwatch.GetTimestamp();
        Stats.NumAdvocateSoloWindows++;
      }
      return true;
    }
    if (advSoloTiming)
    {
      // Lost the window (the owner loop exited and was relaunched): close the timed interval.
      Stats.AdvocateSoloHeldMs += MsSince(advSoloGrantTicks);
      advSoloTiming = false;
    }
    Stats.NumAdvocateSoloDeferred++;
    return false;
  }

  /// <summary>Releases the solo window (if this pump owns it) and closes the timing interval.</summary>
  void ReleaseAdvocateSolo()
  {
    if (advSoloTiming)
    {
      Stats.AdvocateSoloHeldMs += MsSince(advSoloGrantTicks);
      advSoloTiming = false;
    }
    Manager.Engine?.Coordinator.ReleaseSolo(PUMP_ITERATOR_ID);
  }

  // Attention state (entries written only under pumpLock; snapshot published by the gate drain).
  readonly List<ProbeAttentionEntry> advEntries = new();
  readonly List<AdvocateTargetState> advTargets = new();
  int advTraceVisitsUsed;
  volatile ProbeAttentionEntry[] advPendingSnapshot;
  volatile bool advAttentionDirty;

  // Campaign state (mirrors rootadapt; caps shared with the RootAdapt* parameters).
  int advCampaignsThisMove;
  bool advCampActive;
  bool advCampFlipped;
  int advCampSlot = -1;
  ushort advCampMoveRaw = ushort.MaxValue;
  int advCampVisits;
  int advCampCap;


  /// <summary>
  /// The advocate pump (iterator 0, under pumpLock): continues an active campaign, applies a
  /// stored Dissenter verdict (attribution + attention + campaign trigger), and keeps the
  /// Dissenter probing on its cadence (immediately when the root favourite changed, else every
  /// AdvocateMinIntervalSeconds). One probe with DominantMove = the favourite returns the
  /// prober's best move/score/PV, the favourite's score and its refutation line together.
  /// </summary>
  void PumpAdvocate(MCGSIterator iterator, bool searchStopping)
  {
    MCGSEngine engine = iterator.Engine;

    if (advCampActive)
    {
      if (searchStopping)
      {
        FinishAdvocateCampaign(engine, abandoned: true);
      }
      else
      {
        StepAdvocateCampaign(iterator);
      }
    }
    if (searchStopping)
    {
      advPendingEngagement = null;
      ReleaseAdvocateSolo();
      return;
    }

    if (advPendingVerdict != null)
    {
      ProbeResult verdict = advPendingVerdict;
      AdvocateToken token = advPendingToken;
      advPendingVerdict = null;
      advPendingToken = null;
      ApplyAdvocateVerdict(iterator, token, verdict);   // gate + bookkeeping; may queue an engagement
    }

    if (AdvocateContinuousActive)
    {
      EvaluateContinuousVerdict(iterator);   // matched-depth verdict from the score-by-depth histories
    }
    else if (AdvocateDualActive)
    {
      EvaluateDualVerdict(iterator);   // assembles a verdict from the latest root + favourite results
    }

    if (advPendingEngagement != null && AdvocateSoloReady(iterator))
    {
      AdvocateEngagement engagement = advPendingEngagement;
      advPendingEngagement = null;
      ActOnAdvocateEngagement(iterator, engagement);
    }

    if (!advCampActive && advPendingEngagement == null)
    {
      ReleaseAdvocateSolo();   // no directed work outstanding: let the peer iterator resume
    }

    if (AdvocateContinuousActive)
    {
      // Open-ended slots: the root search runs for the whole move; the favourite search is cancelled
      // and restarted when the favourite changes (its final callback frees the slot).
      ushort favRawNow = TopNRootMoveRaw(engine);
      if (favRawNow != ushort.MaxValue)
      {
        if (!advRootInFlight && !advRootDone)
        {
          SubmitContinuousProbe(engine, favRawNow, AdvocateSlot.Root);
        }
        if (advFavInFlight && advFavTargetRaw != favRawNow && !advFavCancelRequested)
        {
          advFavCts?.Cancel();
          advFavCancelRequested = true;
        }
        if (!advFavInFlight)
        {
          SubmitContinuousProbe(engine, favRawNow, AdvocateSlot.Fav);
        }
      }
    }
    else if (AdvocateDualActive)
    {
      // Both slots run back-to-back ("always running"); the favourite slot follows the favourite.
      ushort favRawNow = TopNRootMoveRaw(engine);
      if (favRawNow != ushort.MaxValue)
      {
        if (!advRootInFlight)
        {
          SubmitAdvocateProbe(engine, favRawNow, AdvocateSlot.Root);
        }
        if (!advFavInFlight)
        {
          SubmitAdvocateProbe(engine, favRawNow, AdvocateSlot.Fav);
        }
      }
    }
    else if (!advProbeInFlight && !advCampActive && advPendingEngagement == null)
    {
      ushort favRaw = TopNRootMoveRaw(engine);
      bool favChanged = favRaw != advLastVerdictFavRaw;
      bool intervalDue = advLastVerdictTicks == 0
                      || MsSince(advLastVerdictTicks) >= Params.AdvocateMinIntervalSeconds * 1000.0;
      if (favRaw != ushort.MaxValue && (favChanged || intervalDue))
      {
        SubmitAdvocateProbe(engine, favRaw);
      }
    }
  }


  /// <summary>Submits one Dissenter probe of the search root: Single = DominantMove = current favourite
  /// (one probe answers both questions); Root = unrestricted; Fav = restricted to the favourite.</summary>
  void SubmitAdvocateProbe(MCGSEngine engine, ushort favRaw, AdvocateSlot slot = AdvocateSlot.Single)
  {
    GNode root = engine.SearchRootNode;
    MGPosition rootPos = engine.SearchRootPosMG;
    MGMove favMove = default;
    int bestN = -1;
    foreach (GEdge edge in root.ChildEdgesExpanded)
    {
      if (edge.N > bestN)
      {
        bestN = edge.N;
        favMove = edge.MoveMGFromPos(in rootPos);
      }
    }
    if (favMove.IsNull)
    {
      return;
    }

    ProbeRequest request = new()
    {
      RequestID = Interlocked.Increment(ref nextRequestID),
      RequesterToken = new AdvocateToken(++advSeq, favRaw, favMove, slot),
      Position = new PositionWithHistory(SearchRootPositionWithHistory()),
      DominantMove = slot == AdvocateSlot.Single ? favMove : default,
      RestrictToMove = slot == AdvocateSlot.Fav ? favMove : default,
      NodeBudget = Params.AdvocateProbeNodes,
      CancellationToken = cts.Token,
    };
    if (session.SubmitProbe(request))
    {
      switch (slot)
      {
        case AdvocateSlot.Root: advRootInFlight = true; break;
        case AdvocateSlot.Fav: advFavInFlight = true; advFavInFlightRaw = favRaw; break;
        default: advProbeInFlight = true; break;
      }
      Stats.NumAdvocateProbes++;
      Stats.NumProbesSubmitted++;
    }
  }


  /// <summary>Submits an open-ended, interim-reporting search for one continuous slot (own cancellation token).</summary>
  void SubmitContinuousProbe(MCGSEngine engine, ushort favRaw, AdvocateSlot slot)
  {
    GNode root = engine.SearchRootNode;
    MGPosition rootPos = engine.SearchRootPosMG;
    MGMove favMove = default;
    int bestN = -1;
    foreach (GEdge edge in root.ChildEdgesExpanded)
    {
      if (edge.N > bestN)
      {
        bestN = edge.N;
        favMove = edge.MoveMGFromPos(in rootPos);
      }
    }
    if (favMove.IsNull)
    {
      return;
    }

    CancellationTokenSource slotCts = new();
    ProbeRequest request = new()
    {
      RequestID = Interlocked.Increment(ref nextRequestID),
      RequesterToken = new AdvocateToken(++advSeq, favRaw, favMove, slot, Continuous: true),
      Position = new PositionWithHistory(SearchRootPositionWithHistory()),
      RestrictToMove = slot == AdvocateSlot.Fav ? favMove : default,
      NodeBudget = int.MaxValue,
      ReportInterim = true,
      CancellationToken = slotCts.Token,
    };
    if (!session.SubmitProbe(request))
    {
      slotCts.Dispose();
      return;
    }
    if (slot == AdvocateSlot.Root)
    {
      advRootCts?.Dispose();
      advRootCts = slotCts;
      advRootInFlight = true;
      advRootStartTicks = Stopwatch.GetTimestamp();
    }
    else
    {
      advFavCts?.Dispose();
      advFavCts = slotCts;
      advFavInFlight = true;
      advFavCancelRequested = false;
      advFavTargetRaw = favRaw;
      advFavStartTicks = Stopwatch.GetTimestamp();
      advFavHist.Clear();
      advFavMaxDepth = 0;
      advFavLatest = null;
      advFavLatestRaw = ushort.MaxValue;
    }
    Stats.NumAdvocateProbes++;
    Stats.NumProbesSubmitted++;
  }


  /// <summary>Interim / final callbacks of the continuous slots (stored only; evaluated at the pump).</summary>
  void InterpretContinuousResult(MCGSEngine engine, AdvocateToken token, ProbeResult result)
  {
    if (result.IsInterim)
    {
      if (result.Completed)
      {
        ApplyContinuousReport(engine, token, result);
      }
      return;
    }

    // Final callback: the slot is free. A natural end (Completed) is also the slot's last report.
    if (result.Completed)
    {
      ApplyContinuousReport(engine, token, result);
    }
    if (token.Slot == AdvocateSlot.Root)
    {
      advRootInFlight = false;
      if (result.Completed)
      {
        advRootDone = true;
      }
      Stats.NumProbesCompleted++;
      Stats.ProbeDepthSum += advRootMaxDepth;
      Stats.ProbeLatencySumMs += MsSince(advRootStartTicks);
    }
    else if (token.Slot == AdvocateSlot.Fav)
    {
      advFavInFlight = false;
      advFavCancelRequested = false;
      if (token.FavRaw == advFavTargetRaw)
      {
        Stats.NumProbesCompleted++;
        Stats.ProbeDepthSum += advFavMaxDepth;
        Stats.ProbeLatencySumMs += MsSince(advFavStartTicks);
      }
    }
  }


  /// <summary>Records one completed iteration of a continuous slot into its score-by-depth history.</summary>
  void ApplyContinuousReport(MCGSEngine engine, AdvocateToken token, ProbeResult result)
  {
    if (result.BestMove.IsNull || result.Depth <= 0)
    {
      return;
    }
    if (token.Slot == AdvocateSlot.Root)
    {
      advRootLatest = result;
      advRootLatestSeq = token.Seq;
      advRootHist[result.Depth] = (result.BestScoreCp, RootMoveRaw(engine, result.BestMove));
      if (result.Depth > advRootMaxDepth)
      {
        advRootMaxDepth = result.Depth;
      }
      Stats.NumAdvocateRootProbes++;
    }
    else if (token.Slot == AdvocateSlot.Fav)
    {
      if (token.FavRaw != advFavTargetRaw)
      {
        return;   // a cancelled (re-targeted) favourite search still reporting
      }
      if (result.BestMove != token.FavMove)
      {
        if (!advDualUnsupported)
        {
          advDualUnsupported = true;
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
            "[ProbeGraft] probe source ignores ProbeRequest.RestrictToMove; advocate dual probes disabled (single-probe form).");
        }
        return;
      }
      advFavLatest = result;
      advFavLatestSeq = token.Seq;
      advFavLatestRaw = token.FavRaw;
      advFavLatestMove = token.FavMove;
      advFavHist[result.Depth] = result.BestScoreCp;
      if (result.Depth > advFavMaxDepth)
      {
        advFavMaxDepth = result.Depth;
      }
      Stats.NumAdvocateFavProbes++;
    }
  }


  /// <summary>
  /// Continuous-form verdict at MATCHED depth d = min(root depth, favourite depth) >= AdvocateMinVerdictDepth,
  /// evaluated once per (root depth, favourite depth) pair. Disagreement: the root search's best move at
  /// depth d and d-1 (and its latest report) is the same non-favourite move and root.score - fav.score
  /// >= AdvocateMarginCp at both depths -> engagement (repeated for the same favourite only after
  /// AdvocateReengageDepthStep more plies). Agreement: a fail-low of the favourite search by
  /// >= AdvocateMarginCp over its last two iterations -> refutation-only alarm.
  /// </summary>
  void EvaluateContinuousVerdict(MCGSIterator iterator)
  {
    MCGSEngine engine = iterator.Engine;
    if (advPendingEngagement != null || advRootLatest == null || advFavLatest == null)
    {
      return;
    }
    ushort favRaw = TopNRootMoveRaw(engine);
    if (advFavLatestRaw != favRaw || advFavTargetRaw != favRaw)
    {
      return;   // the favourite moved on; the favourite slot is re-targeting
    }
    int d = Math.Min(advRootMaxDepth, advFavMaxDepth);
    if (d < Params.AdvocateMinVerdictDepth)
    {
      return;
    }
    if (advEvalDepthPair == (advRootMaxDepth, advFavMaxDepth))
    {
      return;   // nothing new since the last evaluation
    }
    advEvalDepthPair = (advRootMaxDepth, advFavMaxDepth);
    advLastVerdictTicks = Stopwatch.GetTimestamp();
    advLastVerdictFavRaw = favRaw;

    if (!advRootHist.TryGetValue(d, out (int ScoreCp, ushort BestRaw) r) || !advRootHist.TryGetValue(d - 1, out (int ScoreCp, ushort BestRaw) r1)
     || !advFavHist.TryGetValue(d, out int f) || !advFavHist.TryGetValue(d - 1, out int f1))
    {
      return;
    }

    RetireFadedAdvocateTargets(engine);

    MGMove favMove = advFavLatestMove;
    AdvocateToken token = new(advSeq, favRaw, favMove);

    if (r.BestRaw != favRaw)
    {
      // Disagreement at matched depth: the alternative must be stable (same move at d-1 and in the
      // latest report) and the margin must hold at both depths.
      int margin = r.ScoreCp - f;
      int margin1 = r1.ScoreCp - f1;
      bool stable = r1.BestRaw == r.BestRaw && RootMoveRaw(engine, advRootLatest.BestMove) == r.BestRaw;
      if (!stable || margin < Params.AdvocateMarginCp || margin1 < Params.AdvocateMarginCp)
      {
        if (margin < Params.AdvocateMarginCp)
        {
          Stats.NumAdvocateBelowGate++;
        }
        PublishAdvocateAttention();
        return;
      }
      if (favRaw == advLastEngageFavRaw && d < advLastEngageDepth + Params.AdvocateReengageDepthStep)
      {
        return;   // already argued this favourite at (nearly) this depth
      }
      advLastEngageDepth = d;
      advLastEngageFavRaw = favRaw;

      Stats.NumAdvocateEngagements++;
      advEngagedThisSearch = true;
      advEngagedFavRaw = favRaw;
      advEngagedDissenterRaw = r.BestRaw;
      if (Params.VerboseEventLogging)
      {
        LogEvent($"advocate ENGAGED (continuous d{d}) best={advRootLatest.BestMove} cp={r.ScoreCp} "
               + $"vs fav={favMove} cp={f} margin={margin} (d{d - 1}: {margin1})");
      }
      ProbeResult verdict = advRootLatest with { DominantMoveScoreCp = f, RefutationPV = advFavLatest.RefutationPV };
      advPendingEngagement = new AdvocateEngagement(token, verdict, margin, Math.Min(1f, margin / (float)Params.AdvocateWFullCp));
      return;
    }

    // Agreement at matched depth: watch the favourite search's own trajectory for a fail-low.
    int fd = advFavMaxDepth;
    if (fd >= Params.AdvocateMinVerdictDepth + 2
     && advFavHist.TryGetValue(fd, out int fNow) && advFavHist.TryGetValue(fd - 1, out int fPrev) && advFavHist.TryGetValue(fd - 2, out int fPrev2))
    {
      int drop = fPrev2 - fNow;
      bool persisting = fPrev <= fPrev2 - Params.AdvocateMarginCp / 2;
      if (drop >= Params.AdvocateMarginCp && persisting
       && !(favRaw == advLastAlarmFavRaw && fd < advLastAlarmDepth + Params.AdvocateReengageDepthStep))
      {
        advLastAlarmDepth = fd;
        advLastAlarmFavRaw = favRaw;
        Stats.NumAdvocateRefutationAlarms++;
        if (Params.VerboseEventLogging)
        {
          LogEvent($"advocate FAIL-LOW ALARM fav={favMove} d{fd - 2}:{fPrev2} d{fd - 1}:{fPrev} d{fd}:{fNow}");
        }
        ProbeResult verdict = advRootLatest with { DominantMoveScoreCp = fNow, RefutationPV = advFavLatest.RefutationPV };
        advPendingEngagement = new AdvocateEngagement(token, verdict, drop, Math.Min(1f, drop / (float)Params.AdvocateWFullCp), RefutationOnly: true);
        return;
      }
    }
    Stats.NumAdvocateAgreements++;   // per evaluated depth pair (NumProbesAgreed is left to per-probe forms)
  }


  /// <summary>
  /// Dual-probe verdict: pairs the latest root-search result with the latest favourite-search result
  /// for the CURRENT favourite (a favourite result for another move is stale; the slot is already
  /// re-targeting). Each (root, fav) pair is evaluated once. Engagement when the root search prefers
  /// another move by >= AdvocateMarginCp over the favourite's full-budget score; a refutation-only
  /// alarm when the root search agrees with the favourite but the favourite search scores it
  /// >= AdvocateMarginCp lower than the root search believed (the deeper look found trouble).
  /// </summary>
  void EvaluateDualVerdict(MCGSIterator iterator)
  {
    MCGSEngine engine = iterator.Engine;
    if (advPendingEngagement != null || advRootLatest == null || advFavLatest == null)
    {
      return;
    }
    ushort favRaw = TopNRootMoveRaw(engine);
    if (advFavLatestRaw != favRaw)
    {
      if (advEvalFavSeq != advFavLatestSeq)
      {
        Stats.NumResultsStale++;   // answered for a favourite that has since changed
        advEvalFavSeq = advFavLatestSeq;
      }
      return;
    }
    if (advEvalRootSeq == advRootLatestSeq && advEvalFavSeq == advFavLatestSeq)
    {
      return;   // this pair was already evaluated
    }
    advEvalRootSeq = advRootLatestSeq;
    advEvalFavSeq = advFavLatestSeq;
    advLastVerdictTicks = Stopwatch.GetTimestamp();
    advLastVerdictFavRaw = favRaw;

    ProbeResult rootRes = advRootLatest;
    ProbeResult favRes = advFavLatest;
    if (rootRes.BestMove.IsNull || favRes.BestMove.IsNull)
    {
      return;
    }

    RetireFadedAdvocateTargets(engine);

    MGMove favMove = advFavLatestMove;
    AdvocateToken token = new(advSeq, favRaw, favMove);
    int margin = rootRes.BestScoreCp - favRes.BestScoreCp;
    ProbeResult verdict = rootRes with { DominantMoveScoreCp = favRes.BestScoreCp, RefutationPV = favRes.RefutationPV };

    if (rootRes.BestMove == favMove)
    {
      if (margin >= Params.AdvocateMarginCp)
      {
        // The favourite, examined at full budget, is worse than the root search's own view of it.
        Stats.NumAdvocateRefutationAlarms++;
        if (Params.VerboseEventLogging)
        {
          LogEvent($"advocate REFUTATION ALARM fav={favMove} root cp={rootRes.BestScoreCp} d{rootRes.Depth} "
                 + $"vs fav-search cp={favRes.BestScoreCp} d{favRes.Depth}");
        }
        advPendingEngagement = new AdvocateEngagement(token, verdict, margin,
                                                      Math.Min(1f, margin / (float)Params.AdvocateWFullCp), RefutationOnly: true);
        return;
      }
      Stats.NumAdvocateAgreements++;
      Stats.NumProbesAgreed++;
      return;
    }

    if (margin < Params.AdvocateMarginCp)
    {
      Stats.NumAdvocateBelowGate++;
      PublishAdvocateAttention();
      return;
    }

    Stats.NumAdvocateEngagements++;
    advEngagedThisSearch = true;
    advEngagedFavRaw = favRaw;
    advEngagedDissenterRaw = RootMoveRaw(engine, rootRes.BestMove);
    if (Params.VerboseEventLogging)
    {
      LogEvent($"advocate ENGAGED (dual) best={rootRes.BestMove} cp={rootRes.BestScoreCp} d{rootRes.Depth} "
             + $"vs fav={favMove} cp={favRes.BestScoreCp} d{favRes.Depth} margin={margin}");
    }
    advPendingEngagement = new AdvocateEngagement(token, verdict, margin, Math.Min(1f, margin / (float)Params.AdvocateWFullCp));
  }


  /// <summary>
  /// Applies one Dissenter verdict: agreement/gate bookkeeping, hysteresis retirement, frontier
  /// walks along both PVs (interior), the root-edge attention entry, snapshot publication and
  /// the campaign trigger. Attention is never a value write; campaigns keep the rollback.
  /// </summary>
  void ApplyAdvocateVerdict(MCGSIterator iterator, AdvocateToken token, ProbeResult result)
  {
    MCGSEngine engine = iterator.Engine;
    advLastVerdictTicks = Stopwatch.GetTimestamp();
    advLastVerdictFavRaw = token.FavRaw;

    // The favourite changed while the probe ran: the verdict answers a stale question.
    if (TopNRootMoveRaw(engine) != token.FavRaw)
    {
      Stats.NumResultsStale++;
      advLastVerdictFavRaw = ushort.MaxValue;   // re-probe on the next pump
      return;
    }
    if (result.BestMove.IsNull)
    {
      return;
    }

    RetireFadedAdvocateTargets(engine);

    if (result.BestMove == token.FavMove)
    {
      Stats.NumAdvocateAgreements++;
      Stats.NumProbesAgreed++;
      return;
    }

    int margin = result.DominantMoveScoreCp == int.MinValue
               ? int.MinValue
               : result.BestScoreCp - result.DominantMoveScoreCp;
    if (margin == int.MinValue || margin < Params.AdvocateMarginCp)
    {
      if (margin != int.MinValue)
      {
        Stats.NumAdvocateBelowGate++;
      }
      PublishAdvocateAttention();
      return;
    }

    Stats.NumAdvocateEngagements++;
    advEngagedThisSearch = true;
    advEngagedFavRaw = token.FavRaw;
    advEngagedDissenterRaw = RootMoveRaw(engine, result.BestMove);
    if (Params.VerboseEventLogging)
    {
      LogEvent($"advocate ENGAGED best={result.BestMove} cp={result.BestScoreCp} d{result.Depth} "
             + $"vs fav={token.FavMove} cp={result.DominantMoveScoreCp} margin={margin}");
    }

    // The actions (frontier walks + traces, root entry, campaign) may issue directed visits, which
    // under overlapped iterators need the solo window: queue them; PumpAdvocate acts when ready.
    advPendingEngagement = new AdvocateEngagement(token, result, margin, Math.Min(1f, margin / (float)Params.AdvocateWFullCp));
  }


  /// <summary>
  /// Acts on an engaged verdict (frontier walks along both PVs with traces, the root-edge attention
  /// entry, snapshot publication and the campaign trigger). Runs in the same pump as the verdict when
  /// no solo window is needed, else at the first pump at which the window is granted.
  /// </summary>
  void ActOnAdvocateEngagement(MCGSIterator iterator, AdvocateEngagement engagement)
  {
    MCGSEngine engine = iterator.Engine;
    AdvocateToken token = engagement.Token;
    ProbeResult result = engagement.Result;
    float w = engagement.W;

    // The favourite moved on while we waited for the window: the argument no longer applies.
    if (TopNRootMoveRaw(engine) != token.FavRaw)
    {
      return;
    }

    GNode root = engine.SearchRootNode;
    MGPosition rootPos = engine.SearchRootPosMG;

    if (Params.AdvocateInterior)
    {
      double aQ = ProbeCpToLogisticQ(result.BestScoreCp);
      double mQ = ProbeCpToLogisticQ(result.DominantMoveScoreCp);
      List<MGMove> pvR = new(result.RefutationPV.Count + 1) { token.FavMove };
      pvR.AddRange(result.RefutationPV);
      if (!engagement.RefutationOnly)
      {
        TryAddAdvocateTarget(iterator, root, result.PV, aQ, 'A', w);
      }
      TryAddAdvocateTarget(iterator, root, pvR, mQ, 'R', w);
    }

    if (engagement.RefutationOnly)
    {
      PublishAdvocateAttention();   // refutation alarm: attention on the favourite's refutation only
      return;
    }

    if (Params.AdvocateRootAttention)
    {
      int rslot = ProbeGraftScheduler.ResolveMoveToChildIndex(root, in rootPos, result.BestMove);
      if (rslot >= 0 && rslot < root.NumEdgesExpanded)
      {
        GEdge re = root.ChildEdgeAtIndex(rslot);
        if (re.Type == GEdgeStruct.EdgeType.ChildEdge)
        {
          GNode rchild = re.ChildNode;
          if (!rchild.IsNull && rchild.N > 0)
          {
            AddOrReplaceAdvocateEntry(root.Index.Index, rchild.Index.Index, w,
                                      Params.AdvocateAttentionKappa * Math.Max(1, rchild.N));
          }
        }
      }
    }

    PublishAdvocateAttention();

    double progress = SearchProgressFraction();
    if (!advCampActive
     && advCampaignsThisMove < Params.AdvocateMaxCampaignsPerMove
     && progress >= Params.AdvocateCampaignMinFraction
     && progress <= Params.AdvocateCampaignMaxFraction)   // a late start gets cut by the clock: don't bet
    {
      StartAdvocateCampaign(iterator, result.BestMove);
    }
  }


  /// <summary>
  /// Walks a probe PV through the live graph from the root to the first material break and turns
  /// it into an attention target (trace beyond the frontier + fading entries on the chain). The
  /// claim at ply k is the root-perspective probe score sign-adjusted per mover (the PV is
  /// negamax-consistent along itself). Side 'A': two-sided value break or an unexplored PV move
  /// at an established node; side 'R': one-sided break where the graph is ROSIER about the
  /// favourite's consequences than the refutation claims.
  /// </summary>
  void TryAddAdvocateTarget(MCGSIterator iterator, GNode root, IReadOnlyList<MGMove> pv,
                            double claimRootQ, char side, float w)
  {
    int live = 0;
    foreach (AdvocateTargetState t in advTargets)
    {
      if (!t.Retired && t.Side == side)
      {
        live++;
      }
    }
    if (live >= Params.AdvocateTargetsPerSide || pv == null || pv.Count == 0)
    {
      return;
    }

    float thetaEff = Params.AdvocateTheta > 0 ? Params.AdvocateTheta : Params.ProbeStampTheta;
    GNode cur = root;
    GNode anchor = default;
    int breakPly = -1;
    int numPlies = Math.Min(pv.Count, Params.AdvocateMaxPlies);
    for (int ply = 0; ply < numPlies; ply++)
    {
      if (cur.IsNull || !cur.IsEvaluated || cur.Terminal.IsTerminal())
      {
        return;
      }
      MGPosition cpos = cur.CalcPosition();
      int slot = ProbeGraftScheduler.ResolveMoveToChildIndex(cur, in cpos, pv[ply]);
      GNode child = default;
      bool unexplored = slot < 0 || slot >= cur.NumEdgesExpanded;
      if (!unexplored)
      {
        GEdge e = cur.ChildEdgeAtIndex(slot);
        if (e.Type != GEdgeStruct.EdgeType.ChildEdge)
        {
          unexplored = true;
        }
        else
        {
          child = e.ChildNode;
          if (child.IsNull || !child.IsEvaluated || child.N == 0)
          {
            unexplored = true;
          }
        }
      }
      if (unexplored)
      {
        if (cur.N < Params.AdvocateNMin)
        {
          return;   // the walk left the established graph in noise: nothing to argue about
        }
        anchor = cur;
        breakPly = ply;
        break;
      }
      int childPly = ply + 1;
      double childRootQ = (childPly % 2 == 0) ? child.Q : -child.Q;
      if (child.N >= Params.AdvocateNMin)
      {
        bool broke = side == 'A' ? Math.Abs(childRootQ - claimRootQ) >= thetaEff
                                 : childRootQ - claimRootQ >= thetaEff;
        if (broke)
        {
          anchor = cur;
          breakPly = ply;
          break;
        }
      }
      cur = child;
    }
    if (breakPly < 0)
    {
      return;
    }

    List<MGMove> remaining = new(pv.Count - breakPly);
    for (int i = breakPly; i < pv.Count; i++)
    {
      remaining.Add(pv[i]);
    }

    // Materialize the line beyond the frontier so the entries have nodes to point at.
    if (Params.AdvocateTracePlies > 0)
    {
      int traceBudget = AdvocateTraceBudget(iterator.Engine) - advTraceVisitsUsed;
      if (traceBudget > 0)
      {
        DirectedVisits.Outcome line = DirectedVisits.ApplyLine(iterator, anchor, remaining, 1,
                                                               Math.Min(Params.AdvocateTracePlies, remaining.Count));
        advTraceVisitsUsed += line.VisitsApplied;
        Stats.NumAdvocateTraceVisits += line.VisitsApplied;
      }
    }

    // Entry chain (parent, child) pairs along the remaining PV.
    List<(GNode Parent, GNode Child)> chain = new();
    cur = anchor;
    for (int i = 0; i < remaining.Count && chain.Count < Params.AdvocateAttnPlies + 1; i++)
    {
      if (cur.IsNull || !cur.IsEvaluated || cur.Terminal.IsTerminal())
      {
        break;
      }
      MGPosition cpos = cur.CalcPosition();
      int slot = ProbeGraftScheduler.ResolveMoveToChildIndex(cur, in cpos, remaining[i]);
      if (slot < 0 || slot >= cur.NumEdgesExpanded)
      {
        break;
      }
      GEdge e = cur.ChildEdgeAtIndex(slot);
      if (e.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        break;
      }
      GNode child = e.ChildNode;
      if (child.IsNull || child.N == 0)
      {
        break;
      }
      chain.Add((cur, child));
      cur = child;
    }
    if (chain.Count == 0)
    {
      return;
    }

    int trackedIndex = chain[0].Child.Index.Index;
    AdvocateTargetState prior = null;
    foreach (AdvocateTargetState t in advTargets)
    {
      if (t.TrackedIndex == trackedIndex)
      {
        prior = t;
        break;
      }
    }
    if (prior != null)
    {
      if (prior.Retired)
      {
        return;   // never re-arm a retired target
      }
      RemoveAdvocateEntries(prior);
      advTargets.Remove(prior);
    }

    AdvocateTargetState state = new()
    {
      Side = side,
      TrackedIndex = trackedIndex,
      QAtFlag = chain[0].Child.Q,
      NAtFlag = chain[0].Child.N,
    };
    foreach ((GNode p, GNode c) in chain)
    {
      float k = Params.AdvocateAttentionKappa * Math.Max(1, c.N);
      AddOrReplaceAdvocateEntry(p.Index.Index, c.Index.Index, w, k);
      state.EntryKeys.Add((p.Index.Index, c.Index.Index));
    }
    advTargets.Add(state);
  }


  /// <summary>Trace-visit budget for this search (fraction of the node limit, or of root N under time limits).</summary>
  int AdvocateTraceBudget(MCGSEngine engine)
  {
    SearchLimit limit = Manager.SearchLimit;
    double basis = limit != null && limit.IsNodesLimit && limit.Value > 0
                 ? limit.Value
                 : engine.SearchRootNode.N;
    return (int)(Params.AdvocateTraceBudgetFraction * basis);
  }


  /// <summary>Retires targets whose attention faded without moving their node (never re-armed).</summary>
  void RetireFadedAdvocateTargets(MCGSEngine engine)
  {
    foreach (AdvocateTargetState t in advTargets)
    {
      if (t.Retired || t.TrackedIndex < 0)
      {
        continue;
      }
      GNode node = TryResolveNode(engine.Graph, t.TrackedIndex);
      if (node.IsNull)
      {
        continue;
      }
      if (node.N >= (1 + Params.AdvocateAttentionKappa) * Math.Max(1, t.NAtFlag)
       && Math.Abs(node.Q - t.QAtFlag) < 0.02)
      {
        t.Retired = true;
        RemoveAdvocateEntries(t);
        Stats.NumAdvocateRetired++;
      }
    }
  }


  void AddOrReplaceAdvocateEntry(int parent, int child, float w, float k)
  {
    for (int i = 0; i < advEntries.Count; i++)
    {
      if (advEntries[i].ParentNodeIndex == parent && advEntries[i].ChildNodeIndex == child)
      {
        advEntries[i] = new ProbeAttentionEntry(parent, child, w, k);
        return;
      }
    }
    advEntries.Add(new ProbeAttentionEntry(parent, child, w, k));
  }


  void RemoveAdvocateEntries(AdvocateTargetState t)
  {
    foreach ((int p, int c) in t.EntryKeys)
    {
      advEntries.RemoveAll(e => e.ParentNodeIndex == p && e.ChildNodeIndex == c);
    }
    t.EntryKeys.Clear();
  }


  /// <summary>Builds the pending snapshot (newest 64 kept); the gate drain swaps it onto the graph.</summary>
  void PublishAdvocateAttention()
  {
    ProbeAttentionEntry[] snap = advEntries.Count == 0
                               ? null
                               : advEntries.Count <= 64
                                 ? advEntries.ToArray()
                                 : advEntries.GetRange(advEntries.Count - 64, 64).ToArray();
    advPendingSnapshot = snap;
    advAttentionDirty = true;
    Stats.NumAdvocateEntries = snap?.Length ?? 0;
  }


  /// <summary>Starts an adaptive root campaign into the prober's move (caps per RootAdapt* params).</summary>
  void StartAdvocateCampaign(MCGSIterator iterator, MGMove probeMove)
  {
    MCGSEngine engine = iterator.Engine;
    GNode root = engine.SearchRootNode;
    MGPosition rootPos = engine.SearchRootPosMG;
    int slot = ProbeGraftScheduler.ResolveMoveToChildIndex(root, in rootPos, probeMove);
    if (slot < 0)
    {
      return;
    }
    ushort probeRaw = RootMoveRawAtSlot(root, slot);
    if (probeRaw == TopNRootMoveRaw(engine))
    {
      return;   // the search already prefers the prober's move
    }

    SearchLimit limit = Manager.SearchLimit;
    int cap = Params.RootAdaptMaxVisits;
    if (limit != null && limit.IsNodesLimit && limit.Value > 0)
    {
      cap = Math.Min(cap, Math.Max(Params.RootAdaptVisitsPerPump, (int)(Params.RootAdaptMaxVisitsFraction * limit.Value)));
    }
    else
    {
      // Time limit: no node budget to take a fraction of; scale by the graph actually built so a
      // small increment move cannot spend half its visits on one campaign.
      cap = Math.Min(cap, Math.Max(Params.RootAdaptVisitsPerPump, (int)(Params.RootAdaptMaxVisitsFraction * root.N)));
    }

    advCampActive = true;
    advCampFlipped = false;
    advCampSlot = slot;
    advCampMoveRaw = probeRaw;
    advCampVisits = 0;
    advCampCap = cap;
    advCampaignsThisMove++;
    Stats.NumAdvocateCampaigns++;
  }


  /// <summary>One chunk of the active campaign (flip test, cap, abandon on slot reorder).</summary>
  void StepAdvocateCampaign(MCGSIterator iterator)
  {
    if (!AdvocateSoloReady(iterator))
    {
      return;   // the peer has not parked yet: this chunk runs at a later pump
    }

    MCGSEngine engine = iterator.Engine;
    GNode root = engine.SearchRootNode;

    int chunk = Math.Min(Params.RootAdaptVisitsPerPump, advCampCap - advCampVisits);
    DirectedVisits.Outcome step = chunk > 0
      ? DirectedVisits.ForceChildVisits(iterator, root, advCampSlot, chunk)
      : new DirectedVisits.Outcome(0, 0, 0, "CapReached");
    advCampVisits += step.VisitsApplied;
    Stats.NumAdvocateCampaignVisits += step.VisitsApplied;

    if (advCampSlot < root.NumEdgesExpanded)
    {
      GEdge edge = root.ChildEdgeAtIndex(advCampSlot);
      if (edge.Move.RawValue != advCampMoveRaw)
      {
        FinishAdvocateCampaign(engine, abandoned: true);   // slot reordered: abandon (rollback)
        return;
      }
      if (edge.Type != GEdgeStruct.EdgeType.Uninitialized
          && -edge.Q >= DirectedVisits.BestSiblingValue(root, advCampSlot) + Params.RootAdaptMarginQ)
      {
        advCampFlipped = true;
        advCampActive = false;
        Stats.NumAdvocateCampaignFlips++;
        if (Params.VerboseEventLogging)
        {
          LogEvent($"advocate campaign flip after {advCampVisits} visits (cap {advCampCap})");
        }
        return;
      }
    }

    if (step.Abort != null || step.VisitsApplied == 0 || advCampVisits >= advCampCap)
    {
      FinishAdvocateCampaign(engine, abandoned: true);
    }
  }


  /// <summary>Ends an unflipped campaign: rolls its visits back (if configured).</summary>
  void FinishAdvocateCampaign(MCGSEngine engine, bool abandoned)
  {
    if (advCampActive && abandoned && !advCampFlipped && Params.RootAdaptRollback && advCampVisits > 0)
    {
      GNode root = engine.SearchRootNode;
      if (advCampSlot >= 0 && advCampSlot < root.NumEdgesExpanded
          && root.ChildEdgeAtIndex(advCampSlot).Move.RawValue == advCampMoveRaw)
      {
        DirectedVisits.ForgetEdgeVisits(root, advCampSlot, advCampVisits);
      }
      Stats.NumAdvocateCampaignRollbacks++;
    }
    advCampActive = false;
  }

  #endregion


  /// <summary>Raw move value of the root's policy slot (expanded edge or header).</summary>
  static ushort RootMoveRawAtSlot(GNode root, int slot)
    => (slot < root.NumEdgesExpanded ? root.ChildEdgeAtIndex(slot).Move : root.EdgeHeadersSpan[slot].Move).RawValue;


  /// <summary>Raw move value of a root move (ushort.MaxValue when the move is not a root policy move).</summary>
  static ushort RootMoveRaw(MCGSEngine engine, MGMove move)
  {
    GNode root = engine.SearchRootNode;
    MGPosition rootPos = engine.SearchRootPosMG;
    int slot = ProbeGraftScheduler.ResolveMoveToChildIndex(root, in rootPos, move);
    return slot < 0 || slot >= root.NumPolicyMoves ? ushort.MaxValue : RootMoveRawAtSlot(root, slot);
  }


  static ushort TopNRootMoveRaw(MCGSEngine engine)
  {
    int bestN = -1;
    ushort raw = ushort.MaxValue;
    foreach (GEdge edge in engine.SearchRootNode.ChildEdgesExpanded)
    {
      if (edge.N > bestN)
      {
        bestN = edge.N;
        raw = edge.Move.RawValue;
      }
    }
    return raw;
  }


  /// <summary>
  /// Milliseconds elapsed since a Stopwatch.GetTimestamp reading.
  /// </summary>
  /// <param name="startTicks"></param>
  static double MsSince(long startTicks)
    => (Stopwatch.GetTimestamp() - startTicks) * 1000.0 / Stopwatch.Frequency;


  /// <summary>
  /// Records one event line (verbose console echo plus the bounded dump-block log).
  /// </summary>
  /// <param name="message"></param>
  internal void LogEvent(string message)
  {
    if (!Params.VerboseEventLogging)
    {
      return;
    }

    if (eventLog.Count < MAX_EVENT_LOG_LINES)
    {
      eventLog.Add(message);
    }
    ConsoleUtils.WriteLineColored(ConsoleColor.DarkCyan, "[ProbeGraft] " + message);
  }


  /// <summary>Suffix naming the non-graft consumers that are active (for log lines: "mode=Disabled+stamps").</summary>
  string ConsumerLabel => (Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active ? "+stamps" : "")
                        + (Params.RootAdaptMode == ParamsProbeGraft.RootAdaptModeType.Active ? "+rootadapt" : "")
                        + (Params.AdvocateMode == ParamsProbeGraft.AdvocateModeType.Active
                             ? (Params.AdvocateContinuousProbes ? "+advocate2c" : Params.AdvocateDualProbes ? "+advocate2" : "+advocate") : "");


  void DisableAfterError(Exception exc)
  {
    disabledByError = true;
    try
    {
      ReleaseAdvocateSolo();   // never leave the peer iterator parked behind a dead feature
    }
    catch
    {
    }
    Stats.DisabledByError = true;
    Stats.DisabledByErrorMessage = exc.Message;
    ConsoleUtils.WriteLineColored(ConsoleColor.Red,
      "[ProbeGraft] disabled for remainder of search after exception: " + exc.Message);
  }


  void ShutdownSession()
  {
    try
    {
      cts.Cancel();
      session?.EndSession();
    }
    catch (Exception exc)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "[ProbeGraft] EndSession failed: " + exc.Message);
    }

    if (ownsSource)
    {
      try
      {
        (source as IDisposable)?.Dispose();
      }
      catch (Exception exc)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "[ProbeGraft] source dispose failed: " + exc.Message);
      }
    }
  }

  #endregion
}
