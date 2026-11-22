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
using System.IO;
using System.Text;

using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Per-search statistics of the PICKET M1 refutation-grafting feature (usage counters,
/// timing, probe-side latency, and impact estimates), owned by the ProbeGraftCoordinator and
/// exposed as MCGSManager.ProbeGraftStats for external diagnostics.
///
/// Plain class with plain fields: all mutation occurs on the pump thread (the search
/// thread, at the post-batch quiescent point) - the probe completion callbacks never touch
/// this object (they only enqueue results into the coordinator's inbox).
///
/// Impact figures are correlational, not causal (Shadow mode plus A/B testing is the
/// causal instrument); they are tuning signals only.
/// </summary>
public sealed class ProbeGraftStats
{
  #region Usage counters

  /// <summary>Number of target-selection sweeps performed.</summary>
  public long NumTargetSweeps;

  /// <summary>Number of target nodes selected across all sweeps (before submission capping).</summary>
  public long NumTargetsSelected;

  /// <summary>Number of probe requests accepted by the source.</summary>
  public long NumProbesSubmitted;

  /// <summary>Number of probe requests the source declined to accept (queue full/unavailable).</summary>
  public long NumProbesRejectedBySource;

  /// <summary>Number of results received with Completed == true.</summary>
  public long NumProbesCompleted;

  /// <summary>Number of results received with Completed == false (cancelled/errored/degenerate).</summary>
  public long NumProbesIncomplete;

  /// <summary>Completed results discarded because the target snapshot no longer matched the graph.</summary>
  public long NumResultsStale;

  /// <summary>Results still in the inbox when the search ended (drained and discarded).</summary>
  public long NumResultsDiscardedAtEnd;

  /// <summary>Completed results where the probe agreed with the node (no trigger; the expected common case).</summary>
  public long NumProbesAgreed;

  /// <summary>Refutation triggers (probe refuted the visit-dominant move by at least the margin).</summary>
  public long NumTriggersRefutation;

  /// <summary>Discovery triggers (probe best move has prior below the nudge floor).</summary>
  public long NumTriggersDiscovery;

  /// <summary>Grafts promoted to actively advancing (Active mode only).</summary>
  public long NumGraftsStarted;

  /// <summary>Grafts that applied their full move list (or reached GraftMaxPlies).</summary>
  public long NumGraftsCompleted;

  /// <summary>Grafts that ended by running into a terminal (mate/draw) - a successful completion.</summary>
  public long NumGraftsEndedTerminal;

  /// <summary>Grafts aborted because a witness move was absent from the node's policy move list.</summary>
  public long NumGraftsAbortedMoveNotFound;

  /// <summary>Grafts aborted because their forced path evaporated (dropped/aborted visits).</summary>
  public long NumGraftsAbortedPathDropped;

  /// <summary>Graft plies retried because the forced crossing produced no path record.</summary>
  public long NumGraftPlyRetriesNoRecord;

  /// <summary>Completed root probes whose verdict was recorded (proposal M5).</summary>
  public long NumRootProbeVerdicts;

  /// <summary>Root verdicts that met the veto depth and margin bar (a veto WOULD be considered).</summary>
  public long NumRootVetoCandidates;

  /// <summary>Root adaptive directed visits (RootAdaptMode): campaigns started / ended in a flip / rolled back / visits forced.</summary>
  public long NumRootAdaptTriggers;
  public long NumRootAdaptFlips;
  public long NumRootAdaptRollbacks;
  public long NumRootAdaptVisits;
  /// <summary>Campaigns whose flipped move was still the move chosen (top-N) at end of search.</summary>
  public long NumRootAdaptFlipsHeld;

  /// <summary>Advocate (AdvocateMode): Dissenter probes / agreements / engagements (margin met),
  /// attention entries placed, trace visits, campaigns started / flipped / rolled back, targets retired.</summary>
  public long NumAdvocateProbes;
  public long NumAdvocateAgreements;
  public long NumAdvocateBelowGate;
  public long NumAdvocateEngagements;
  public long NumAdvocateEntries;
  public long NumAdvocateTraceVisits;
  public long NumAdvocateCampaigns;
  public long NumAdvocateCampaignFlips;
  public long NumAdvocateCampaignRollbacks;
  public long NumAdvocateCampaignVisits;
  public long NumAdvocateRetired;
  /// <summary>Advocate engagement conversion, counted per SEARCH (distinct moves), not per verdict: searches
  /// with at least one engagement; of those, searches whose final root move (top-N; the root move rule may
  /// still pick on Q, the same caveat as NumRootAdaptFlipsHeld) was the Dissenter's move of the LAST
  /// engagement, and searches whose final move left the favourite that engagement argued against.</summary>
  public long NumAdvocateEngagedMoves;
  public long NumAdvocateConversions;
  public long NumAdvocateFavAbandoned;
  /// <summary>Advocate transient solo window (directed visits under overlapped iterators): windows granted,
  /// pumps whose directed work was deferred while waiting for the peer to park, and total ms the window was
  /// held (the search runs single-iterator while held -- the direct cost figure).</summary>
  public long NumAdvocateSoloWindows;
  public long NumAdvocateSoloDeferred;
  public double AdvocateSoloHeldMs;
  /// <summary>Dual-probe advocate: root-slot probes, favourite-slot probes, refutation-only alarms
  /// (favourite scored AdvocateMarginCp below the root search's view of it).</summary>
  public long NumAdvocateRootProbes;
  public long NumAdvocateFavProbes;
  public long NumAdvocateRefutationAlarms;
  /// <summary>Continuous dual-probe advocate: root-slot runs (one per move) and the depth each slot
  /// reached when its run ended (root / favourite sums, for averages). In the continuous form
  /// NumAdvocateRootProbes / NumAdvocateFavProbes count completed ITERATIONS.</summary>
  public long NumAdvocateSlotRuns;
  public long AdvocateRootDepthSum;
  public long AdvocateFavDepthSum;

  /// <summary>AB stamps (ProbeStampMode): flags, confirms submitted / rejected, stamps applied / cleared, ancestors moved by recomputes, root top-Q changes.</summary>
  public long NumStampFlags;
  public long NumStampConfirmsSubmitted;
  public long NumStampConfirmsRejected;
  public long NumStamps;
  public long NumStampsCleared;
  public long NumStampsStaleN;
  public long NumStampsWeightedOut;
  public long NumStampAncestorsMoved;
  public long NumStampRootTopChanges;
  public double TimeStampRecomputeMs;

  /// <summary>
  /// In-gate stamp drains (Route 4, ProbeGraftCoordinator.DrainInBackupGate): drains performed;
  /// deferred because the other iterator held the pump lock; stopped by the per-batch apply
  /// budget with results still pending; and the longest single in-gate drain.
  /// </summary>
  public long NumGateDrains;
  public long NumGateDrainsSkippedBusy;
  public long NumGateDrainsBudgetBound;
  public double MaxGateDrainMs;

  /// <summary>
  /// Aborts broken down by ProbeGraftScheduler.AbortReason (indexed by the enum value).
  /// The PathDropped total above is a roll-up of every reason except MoveNotFound; this array
  /// is what identifies WHICH failure dominates, which the single total cannot.
  /// </summary>
  public long[] NumGraftsAbortedByReason = new long[NUM_ABORT_REASONS];

  /// <summary>Number of distinct AbortReason values (kept in sync with the enum).</summary>
  public const int NUM_ABORT_REASONS = 12;

  /// <summary>Pumps on which graft advancement was paused by the MaxGraftEvalFraction budget.</summary>
  public long NumGraftsPausedBudget;

  /// <summary>Total grafted plies applied (committed forced visits through a witness move).</summary>
  public long NumGraftPliesApplied;

  /// <summary>Total NN evaluations consumed by graft advancement (budget numerator).</summary>
  public long NumGraftNNEvals;

  /// <summary>Prior nudges applied (one per nudged ply).</summary>
  public long NumPriorNudgesApplied;

  /// <summary>Prior nudges skipped or clamped by the per-node cumulative mass cap.</summary>
  public long NumPriorNudgesCappedByMass;

  /// <summary>If the feature disabled itself for the remainder of the search after an exception.</summary>
  public bool DisabledByError;

  /// <summary>Message of the first exception that disabled the feature (null if none).</summary>
  public string DisabledByErrorMessage;

  #endregion

  #region Timing (milliseconds, accumulated on the pump thread via Stopwatch.GetTimestamp deltas)

  /// <summary>Total time spent inside non-idle pumps (the search-thread cost of the feature).</summary>
  public double TimePumpTotalMs;

  /// <summary>Portion of pump time spent draining/interpreting probe results.</summary>
  public double TimeDrainMs;

  /// <summary>Portion of pump time spent advancing grafts (including the RunProbeSpecs calls).</summary>
  public double TimeGraftAdvanceMs;

  /// <summary>Portion of pump time spent in target sweeps and probe submission.</summary>
  public double TimeTargetSweepMs;

  /// <summary>Sum of probe wall latencies (from result metadata; for averages).</summary>
  public double ProbeLatencySumMs;

  /// <summary>Maximum probe wall latency observed.</summary>
  public double ProbeLatencyMaxMs;

  /// <summary>Sum of prober node counts over completed probes.</summary>
  public long ProbeNodesSum;

  /// <summary>Sum of prober iteration depths over completed probes.</summary>
  public long ProbeDepthSum;

  /// <summary>Sum of selective depth over completed probes (0 when a source does not report it).</summary>
  public long ProbeSelDepthSum;

  /// <summary>Completed probes that stopped on the node budget rather than converging.</summary>
  public long NumProbesHitNodeBudget;

  /// <summary>Searches in which at least one graft was started (for cross-search rates).</summary>
  public long NumSearchesWithGraft;

  /// <summary>Searches in which at least one trigger fired.</summary>
  public long NumSearchesWithTrigger;

  /// <summary>Searches accumulated into this object (global accumulation only).</summary>
  public long NumSearchesAccumulated;

  #endregion

  #region Impact (correlational)

  /// <summary>Sum over completed grafts of |anchor Q at end of search - anchor Q before first grafted ply|.</summary>
  public double SumAbsAnchorQDeltaAtEnd;

  /// <summary>Maximum |anchor Q delta| over completed grafts.</summary>
  public double MaxAbsAnchorQDeltaAtEnd;

  /// <summary>Completed grafts whose |anchor Q delta| exceeded 0.03 (BottomUpQRecalculator.LARGE_DELTA_THRESHOLD).</summary>
  public long NumAnchorsQMovedOverThreshold;

  /// <summary>Times the root's top-N move changed across a pump that applied graft plies.</summary>
  public long NumRootBestMoveChangesAfterGraft;

  #endregion

  #region Shadow verdicts

  /// <summary>Shadow mode: number of would-have-grafted events recorded.</summary>
  public long ShadowNumEvents;

  /// <summary>Shadow mode: events plain search resolved on its own by end of search.</summary>
  public long ShadowNumConfirmedBySearch;

  /// <summary>Shadow mode: events left unresolved by plain search (the feature's potential value).</summary>
  public long ShadowNumUnresolved;

  #endregion



  #region Target depth distribution (diagnostic)

  /// <summary>Number of depth bins tracked below the search root (deeper targets fold into the last bin).</summary>
  public const int MAX_DEPTH_BINS = 16;

  /// <summary>Probes submitted, binned by the target node's depth below the search root.</summary>
  public long[] ProbesSubmittedAtDepth = new long[MAX_DEPTH_BINS];

  /// <summary>Triggers (refutation plus discovery), binned by the target node's depth below the search root.</summary>
  public long[] TriggersAtDepth = new long[MAX_DEPTH_BINS];

  /// <summary>Number of distinct graph nodes ever probed during the search (probes per node = submitted / this).</summary>
  public long NumDistinctNodesProbed;

  /// <summary>
  /// Renders the depth histograms as "dN:probes/triggers" over the occupied bins.
  /// </summary>
  /// <returns></returns>
  /// <summary>
  /// One line naming the abort causes in descending order, so the dominant failure mode is
  /// visible without enabling the full event log. Only non-zero reasons are listed.
  /// </summary>
  /// <summary>
  /// Multi-line hybrid statistics block for one search (ParamsSearch.DumpHybridSearchStats = Full).
  ///
  /// Every count is paired with the denominator that makes it interpretable: probes as a fraction
  /// of those submitted, triggers as a fraction of probes that actually completed, graft cost as a
  /// fraction of the search's own NN evaluations. Raw counts alone have repeatedly proved
  /// misleading here -- a large trigger count means nothing if few probes complete, and graft
  /// activity means nothing if it never reaches the root.
  /// </summary>
  /// <param name="mode"></param>
  /// <param name="numSearchEvals"></param>
  /// <param name="probeSourceDescription"></param>
  /// <param name="searchSeconds"></param>
  public string SearchReport(ParamsProbeGraft.ModeType mode, long numSearchEvals,
                             string probeSourceDescription = null, double searchSeconds = 0)
  {
    long trig = NumTriggersRefutation + NumTriggersDiscovery;
    long aborts = NumGraftsAbortedMoveNotFound + NumGraftsAbortedPathDropped;
    string Pct(double num, double den) => den <= 0 ? "n/a" : $"{100.0 * num / den:F1}%";

    StringBuilder sb = new();
    sb.AppendLine("=== HYBRID STATISTICS (PICKET) ===");
    sb.AppendLine($"  mode          : {mode}"
                + (probeSourceDescription == null ? "" : $"   source: {probeSourceDescription}"));
    sb.AppendLine($"  search        : {numSearchEvals:N0} NN evals"
                + (searchSeconds > 0 ? $"   {searchSeconds:F2}s" : ""));
    sb.AppendLine($"  probes        : submitted {NumProbesSubmitted:N0}   completed {NumProbesCompleted:N0} "
                + $"({Pct(NumProbesCompleted, NumProbesSubmitted)})   stale {NumResultsStale:N0} "
                + $"({Pct(NumResultsStale, NumProbesCompleted)})   incomplete {NumProbesIncomplete:N0}");
    if (NumProbesCompleted > 0)
    {
      sb.AppendLine($"                  depth avg {(double)ProbeDepthSum / NumProbesCompleted:F1}"
                  + (ProbeSelDepthSum > 0 ? $"  seldepth avg {(double)ProbeSelDepthSum / NumProbesCompleted:F1}" : "")
                  + $"   latency avg {ProbeLatencySumMs / NumProbesCompleted:F0}ms max {ProbeLatencyMaxMs:F0}ms"
                  + (NumProbesHitNodeBudget > 0
                       ? $"   hit budget {Pct(NumProbesHitNodeBudget, NumProbesCompleted)}" : ""));
    }
    sb.AppendLine($"  triggers      : refutation {NumTriggersRefutation:N0}  discovery {NumTriggersDiscovery:N0}"
                + $"   = {Pct(trig, NumProbesCompleted)} of completed probes");
    sb.AppendLine($"  grafts        : started {NumGraftsStarted:N0}  completed {NumGraftsCompleted:N0} "
                + $"({Pct(NumGraftsCompleted, NumGraftsStarted)})  aborted {aborts:N0} "
                + $"({Pct(aborts, NumGraftsStarted)})  terminal {NumGraftsEndedTerminal:N0}");
    sb.AppendLine($"                  forced plies {NumGraftPliesApplied:N0}   graft NN evals {NumGraftNNEvals:N0} "
                + $"({Pct(NumGraftNNEvals, numSearchEvals)} of search)   prior nudges {NumPriorNudgesApplied:N0}");
    sb.AppendLine($"  impact        : anchors |dQ|>0.03: {NumAnchorsQMovedOverThreshold:N0}   "
                + $"max |dQ| {MaxAbsAnchorQDeltaAtEnd:F3}   ROOT MOVE CHANGES {NumRootBestMoveChangesAfterGraft:N0}");
    if (NumRootProbeVerdicts > 0)
    {
      sb.AppendLine($"  root verdicts : {NumRootProbeVerdicts:N0}   veto candidates {NumRootVetoCandidates:N0} "
                  + $"({Pct(NumRootVetoCandidates, NumRootProbeVerdicts)})");
    }
    if (ShadowNumEvents > 0)
    {
      sb.AppendLine($"  shadow        : events {ShadowNumEvents:N0}  confirmed {ShadowNumConfirmedBySearch:N0}  "
                  + $"unresolved {ShadowNumUnresolved:N0} ({Pct(ShadowNumUnresolved, ShadowNumEvents)})");
    }
    sb.AppendLine($"  cost          : pump {TimePumpTotalMs:F0}ms (sweep {TimeTargetSweepMs:F0} / graft {TimeGraftAdvanceMs:F0} "
                + $"/ drain {TimeDrainMs:F0})"
                + (searchSeconds > 0 ? $"   = {Pct(TimePumpTotalMs / 1000.0, searchSeconds)} of search" : ""));
    if (aborts > 0)
    {
      sb.AppendLine("  " + AbortBreakdownLine());
    }
    return sb.ToString().TrimEnd();
  }


  /// <summary>
  /// Cross-search aggregate report: the per-search RATES that decide whether the feature is worth
  /// carrying. Requires EnableGlobalAccumulation; read via GlobalAggregateReport().
  /// </summary>
  /// <param name="numSearches"></param>
  public string AggregateReport(long numSearches)
  {
    long trig = NumTriggersRefutation + NumTriggersDiscovery;
    string Pct(double num, double den) => den <= 0 ? "n/a" : $"{100.0 * num / den:F2}%";
    StringBuilder sb = new();
    sb.AppendLine($"=== HYBRID AGGREGATE over {numSearches:N0} searches ===");
    sb.AppendLine($"  probes per search            : {(double)NumProbesSubmitted / Math.Max(1, numSearches):F2} submitted, "
                + $"{(double)NumProbesCompleted / Math.Max(1, numSearches):F2} completed");
    sb.AppendLine($"  grafts started per search    : {(double)NumGraftsStarted / Math.Max(1, numSearches):F2}");
    sb.AppendLine($"  searches with >=1 trigger    : {Pct(NumSearchesWithTrigger, numSearches)}");
    sb.AppendLine($"  searches with >=1 graft      : {Pct(NumSearchesWithGraft, numSearches)}");
    sb.AppendLine($"  forced plies per search      : {(double)NumGraftPliesApplied / Math.Max(1, numSearches):F1}");
    sb.AppendLine($"  anchors moved |dQ|>0.03      : {NumAnchorsQMovedOverThreshold:N0} = "
                + $"{Pct(NumAnchorsQMovedOverThreshold, NumGraftsCompleted)} of completed grafts, "
                + $"{Pct(NumAnchorsQMovedOverThreshold, numSearches)} of searches");
    sb.AppendLine($"  largest anchor Q move        : {MaxAbsAnchorQDeltaAtEnd:F3}");
    sb.AppendLine($"  root move changes            : {NumRootBestMoveChangesAfterGraft:N0} = "
                + $"{Pct(NumRootBestMoveChangesAfterGraft, numSearches)} of searches");
    sb.AppendLine($"  graft abort rate             : {Pct(NumGraftsAbortedMoveNotFound + NumGraftsAbortedPathDropped, NumGraftsStarted)}");
    sb.Append($"  pump cost per search         : {TimePumpTotalMs / Math.Max(1, numSearches):F1}ms");
    return sb.ToString();
  }


  /// <summary>One line breaking graft aborts down by AbortReason (identifies the dominant failure mode).</summary>
  public string AbortBreakdownLine()
  {
    long total = NumGraftsAbortedMoveNotFound + NumGraftsAbortedPathDropped;
    if (total == 0)
    {
      return "[ProbeGraft-aborts] none";
    }

    List<(string Name, long Count)> rows = new();
    string[] names = Enum.GetNames(typeof(Search.ProbeGraft.ProbeGraftScheduler.AbortReason));
    for (int i = 0; i < NUM_ABORT_REASONS && i < names.Length; i++)
    {
      if (NumGraftsAbortedByReason[i] > 0)
      {
        rows.Add((names[i], NumGraftsAbortedByReason[i]));
      }
    }
    rows.Sort((a, b) => b.Count.CompareTo(a.Count));

    StringBuilder sb = new();
    sb.Append($"[ProbeGraft-aborts] total={total} retriesNoRecord={NumGraftPlyRetriesNoRecord} ");
    foreach ((string name, long count) in rows)
    {
      sb.Append($"{name}={count} ({100.0 * count / total:F1}%) ");
    }
    return sb.ToString().TrimEnd();
  }


  /// <summary>One line with the probes/triggers histogram by node depth below the search root.</summary>
  public string DepthHistogramLine()
  {
    StringBuilder builder = new();
    builder.Append("[ProbeGraft-depth] distinctNodes=" + NumDistinctNodesProbed + " probes/triggers by depth:");
    for (int depth = 0; depth < MAX_DEPTH_BINS; depth++)
    {
      if (ProbesSubmittedAtDepth[depth] > 0 || TriggersAtDepth[depth] > 0)
      {
        builder.Append($" d{depth}:{ProbesSubmittedAtDepth[depth]}/{TriggersAtDepth[depth]}");
      }
    }
    return builder.ToString();
  }


  /// <summary>
  /// Records one submitted probe against its target's depth below the search root.
  /// </summary>
  /// <param name="depth"></param>
  public void RecordProbeDepth(int depth) => ProbesSubmittedAtDepth[Math.Clamp(depth, 0, MAX_DEPTH_BINS - 1)]++;


  /// <summary>
  /// Records one trigger against its target's depth below the search root.
  /// </summary>
  /// <param name="depth"></param>
  public void RecordTriggerDepth(int depth) => TriggersAtDepth[Math.Clamp(depth, 0, MAX_DEPTH_BINS - 1)]++;

  #endregion


  #region Aggregation across searches

  /// <summary>
  /// Adds all counters of another instance into this one (sums, except the max-valued
  /// fields which are maxed). Used to aggregate per-search statistics over a set of
  /// searches (a measurement campaign, a game, a match).
  /// </summary>
  /// <param name="other"></param>
  public void AccumulateFrom(ProbeGraftStats other)
  {
    if (other == null)
    {
      return;
    }

    NumTargetSweeps += other.NumTargetSweeps;
    NumTargetsSelected += other.NumTargetsSelected;
    NumProbesSubmitted += other.NumProbesSubmitted;
    NumProbesRejectedBySource += other.NumProbesRejectedBySource;
    NumProbesCompleted += other.NumProbesCompleted;
    NumProbesIncomplete += other.NumProbesIncomplete;
    NumResultsStale += other.NumResultsStale;
    NumResultsDiscardedAtEnd += other.NumResultsDiscardedAtEnd;
    NumProbesAgreed += other.NumProbesAgreed;
    NumTriggersRefutation += other.NumTriggersRefutation;
    NumTriggersDiscovery += other.NumTriggersDiscovery;
    NumGraftsStarted += other.NumGraftsStarted;
    NumGraftsCompleted += other.NumGraftsCompleted;
    NumGraftsEndedTerminal += other.NumGraftsEndedTerminal;
    NumGraftsAbortedMoveNotFound += other.NumGraftsAbortedMoveNotFound;
    NumGraftsAbortedPathDropped += other.NumGraftsAbortedPathDropped;
    NumGraftPlyRetriesNoRecord += other.NumGraftPlyRetriesNoRecord;
    NumRootProbeVerdicts += other.NumRootProbeVerdicts;
    NumRootVetoCandidates += other.NumRootVetoCandidates;
    NumRootAdaptTriggers += other.NumRootAdaptTriggers;
    NumRootAdaptFlips += other.NumRootAdaptFlips;
    NumRootAdaptRollbacks += other.NumRootAdaptRollbacks;
    NumRootAdaptVisits += other.NumRootAdaptVisits;
    NumRootAdaptFlipsHeld += other.NumRootAdaptFlipsHeld;
    NumAdvocateProbes += other.NumAdvocateProbes;
    NumAdvocateAgreements += other.NumAdvocateAgreements;
    NumAdvocateBelowGate += other.NumAdvocateBelowGate;
    NumAdvocateEngagements += other.NumAdvocateEngagements;
    NumAdvocateEntries += other.NumAdvocateEntries;
    NumAdvocateTraceVisits += other.NumAdvocateTraceVisits;
    NumAdvocateCampaigns += other.NumAdvocateCampaigns;
    NumAdvocateCampaignFlips += other.NumAdvocateCampaignFlips;
    NumAdvocateCampaignRollbacks += other.NumAdvocateCampaignRollbacks;
    NumAdvocateCampaignVisits += other.NumAdvocateCampaignVisits;
    NumAdvocateRetired += other.NumAdvocateRetired;
    NumAdvocateEngagedMoves += other.NumAdvocateEngagedMoves;
    NumAdvocateConversions += other.NumAdvocateConversions;
    NumAdvocateFavAbandoned += other.NumAdvocateFavAbandoned;
    NumAdvocateSoloWindows += other.NumAdvocateSoloWindows;
    NumAdvocateSoloDeferred += other.NumAdvocateSoloDeferred;
    AdvocateSoloHeldMs += other.AdvocateSoloHeldMs;
    NumAdvocateRootProbes += other.NumAdvocateRootProbes;
    NumAdvocateFavProbes += other.NumAdvocateFavProbes;
    NumAdvocateRefutationAlarms += other.NumAdvocateRefutationAlarms;
    NumAdvocateSlotRuns += other.NumAdvocateSlotRuns;
    AdvocateRootDepthSum += other.AdvocateRootDepthSum;
    AdvocateFavDepthSum += other.AdvocateFavDepthSum;
    NumStampFlags += other.NumStampFlags;
    NumStampConfirmsSubmitted += other.NumStampConfirmsSubmitted;
    NumStampConfirmsRejected += other.NumStampConfirmsRejected;
    NumStamps += other.NumStamps;
    NumStampsCleared += other.NumStampsCleared;
    NumStampsStaleN += other.NumStampsStaleN;
    NumStampsWeightedOut += other.NumStampsWeightedOut;
    NumStampAncestorsMoved += other.NumStampAncestorsMoved;
    NumStampRootTopChanges += other.NumStampRootTopChanges;
    TimeStampRecomputeMs += other.TimeStampRecomputeMs;
    NumGateDrains += other.NumGateDrains;
    NumGateDrainsSkippedBusy += other.NumGateDrainsSkippedBusy;
    NumGateDrainsBudgetBound += other.NumGateDrainsBudgetBound;
    MaxGateDrainMs = Math.Max(MaxGateDrainMs, other.MaxGateDrainMs);
    for (int i = 0; i < NUM_ABORT_REASONS; i++)
    {
      NumGraftsAbortedByReason[i] += other.NumGraftsAbortedByReason[i];
    }
    NumGraftsPausedBudget += other.NumGraftsPausedBudget;
    NumGraftPliesApplied += other.NumGraftPliesApplied;
    NumGraftNNEvals += other.NumGraftNNEvals;
    NumPriorNudgesApplied += other.NumPriorNudgesApplied;
    NumPriorNudgesCappedByMass += other.NumPriorNudgesCappedByMass;

    TimePumpTotalMs += other.TimePumpTotalMs;
    TimeDrainMs += other.TimeDrainMs;
    TimeGraftAdvanceMs += other.TimeGraftAdvanceMs;
    TimeTargetSweepMs += other.TimeTargetSweepMs;
    ProbeLatencySumMs += other.ProbeLatencySumMs;
    ProbeLatencyMaxMs = Math.Max(ProbeLatencyMaxMs, other.ProbeLatencyMaxMs);
    ProbeNodesSum += other.ProbeNodesSum;
    ProbeDepthSum += other.ProbeDepthSum;
    ProbeSelDepthSum += other.ProbeSelDepthSum;
    NumProbesHitNodeBudget += other.NumProbesHitNodeBudget;
    // Cross-search rates need per-search indicators, which only the accumulator can form.
    NumSearchesWithGraft += other.NumGraftsStarted > 0 ? 1 : 0;
    NumSearchesWithTrigger += (other.NumTriggersRefutation + other.NumTriggersDiscovery) > 0 ? 1 : 0;
    NumSearchesAccumulated++;

    SumAbsAnchorQDeltaAtEnd += other.SumAbsAnchorQDeltaAtEnd;
    MaxAbsAnchorQDeltaAtEnd = Math.Max(MaxAbsAnchorQDeltaAtEnd, other.MaxAbsAnchorQDeltaAtEnd);
    NumAnchorsQMovedOverThreshold += other.NumAnchorsQMovedOverThreshold;
    NumRootBestMoveChangesAfterGraft += other.NumRootBestMoveChangesAfterGraft;

    NumDistinctNodesProbed += other.NumDistinctNodesProbed;
    for (int depth = 0; depth < MAX_DEPTH_BINS; depth++)
    {
      ProbesSubmittedAtDepth[depth] += other.ProbesSubmittedAtDepth[depth];
      TriggersAtDepth[depth] += other.TriggersAtDepth[depth];
    }

    ShadowNumEvents += other.ShadowNumEvents;
    ShadowNumConfirmedBySearch += other.ShadowNumConfirmedBySearch;
    ShadowNumUnresolved += other.ShadowNumUnresolved;

    if (other.DisabledByError && !DisabledByError)
    {
      DisabledByError = true;
      DisabledByErrorMessage = other.DisabledByErrorMessage;
    }
  }

  #endregion

  #region Process-wide accumulation (multi-search harnesses)

  /// <summary>
  /// If per-search statistics should be accumulated into a process-wide aggregate as each
  /// search ends. Intended for harnesses that run many searches (games, matches, measurement
  /// campaigns, tournament managers) and cannot conveniently collect per-search objects.
  /// Off by default; the aggregate is inert unless explicitly enabled.
  /// </summary>
  public static bool EnableGlobalAccumulation = false;

  static readonly object globalLockObj = new();
  static readonly ProbeGraftStats globalStats = new();
  static int globalNumSearches;
  static long globalNumEvals;
  static ParamsProbeGraft.ModeType globalMode = ParamsProbeGraft.ModeType.Disabled;

  /// <summary>
  /// Clears the process-wide aggregate (call before each measurement arm).
  /// </summary>
  public static void ResetGlobal()
  {
    lock (globalLockObj)
    {
      globalStats.ClearAll();
      globalNumSearches = 0;
      globalNumEvals = 0;
      globalMode = ParamsProbeGraft.ModeType.Disabled;
    }
  }


  /// <summary>
  /// Adds the statistics of one completed search into the process-wide aggregate.
  /// </summary>
  /// <param name="stats"></param>
  /// <param name="mode"></param>
  /// <param name="numEvalsThisSearch"></param>
  public static void AccumulateGlobal(ProbeGraftStats stats, ParamsProbeGraft.ModeType mode, int numEvalsThisSearch)
  {
    lock (globalLockObj)
    {
      globalStats.AccumulateFrom(stats);
      globalNumSearches++;
      globalNumEvals += numEvalsThisSearch;
      globalMode = mode;
    }
  }


  /// <summary>
  /// Returns a snapshot copy of the process-wide aggregate, the number of searches
  /// contributing to it, and the total NN evaluations across those searches.
  /// </summary>
  /// <returns></returns>
  public static (ProbeGraftStats Stats, int NumSearches, long NumEvals) GlobalSnapshot()
  {
    lock (globalLockObj)
    {
      ProbeGraftStats copy = new();
      copy.AccumulateFrom(globalStats);
      return (copy, globalNumSearches, globalNumEvals);
    }
  }


  /// <summary>
  /// Renders a one-line summary of the process-wide aggregate (same layout as the
  /// per-search summary, prefixed with the number of contributing searches).
  /// </summary>
  /// <returns></returns>
  /// <summary>
  /// Cross-search aggregate report for everything accumulated since the last ResetGlobal.
  /// </summary>
  public static string GlobalAggregateReport()
  {
    (ProbeGraftStats stats, int numSearches, long _) = GlobalSnapshot();
    return stats.AggregateReport(numSearches);
  }


  /// <summary>Renders the process-wide aggregate built by AccumulateGlobal (multi-search harnesses).</summary>
  public static string GlobalSummaryLine()
  {
    (ProbeGraftStats stats, int numSearches, long numEvals) = GlobalSnapshot();
    int evalsForLine = (int)Math.Min(numEvals, int.MaxValue);
    return $"[ProbeGraft-TOTAL over {numSearches} searches] " + stats.SummaryLine(globalMode, evalsForLine);
  }


  /// <summary>
  /// Resets every field to its default value.
  /// </summary>
  void ClearAll()
  {
    NumTargetSweeps = 0;
    NumTargetsSelected = 0;
    NumProbesSubmitted = 0;
    NumProbesRejectedBySource = 0;
    NumProbesCompleted = 0;
    NumProbesIncomplete = 0;
    NumResultsStale = 0;
    NumResultsDiscardedAtEnd = 0;
    NumProbesAgreed = 0;
    NumTriggersRefutation = 0;
    NumTriggersDiscovery = 0;
    NumGraftsStarted = 0;
    NumGraftsCompleted = 0;
    NumGraftsEndedTerminal = 0;
    NumGraftsAbortedMoveNotFound = 0;
    NumGraftsAbortedPathDropped = 0;
    NumGraftPlyRetriesNoRecord = 0;
    NumRootProbeVerdicts = 0;
    NumRootVetoCandidates = 0;
    NumRootAdaptTriggers = 0;
    NumRootAdaptFlips = 0;
    NumRootAdaptRollbacks = 0;
    NumRootAdaptVisits = 0;
    NumRootAdaptFlipsHeld = 0;
    NumAdvocateProbes = 0; NumAdvocateAgreements = 0; NumAdvocateBelowGate = 0; NumAdvocateEngagements = 0; NumAdvocateEntries = 0;
    NumAdvocateTraceVisits = 0; NumAdvocateCampaigns = 0; NumAdvocateCampaignFlips = 0; NumAdvocateCampaignRollbacks = 0;
    NumAdvocateCampaignVisits = 0; NumAdvocateRetired = 0;
    NumAdvocateEngagedMoves = 0; NumAdvocateConversions = 0; NumAdvocateFavAbandoned = 0;
    NumAdvocateSoloWindows = 0; NumAdvocateSoloDeferred = 0; AdvocateSoloHeldMs = 0;
    NumAdvocateRootProbes = 0; NumAdvocateFavProbes = 0; NumAdvocateRefutationAlarms = 0;
    NumAdvocateSlotRuns = 0; AdvocateRootDepthSum = 0; AdvocateFavDepthSum = 0;
    NumStampFlags = 0; NumStampConfirmsSubmitted = 0; NumStampConfirmsRejected = 0; NumStamps = 0; NumStampsCleared = 0;
    NumStampsStaleN = 0; NumStampsWeightedOut = 0; NumStampAncestorsMoved = 0; NumStampRootTopChanges = 0; TimeStampRecomputeMs = 0;
    NumGateDrains = 0; NumGateDrainsSkippedBusy = 0; NumGateDrainsBudgetBound = 0; MaxGateDrainMs = 0;
    Array.Clear(NumGraftsAbortedByReason);
    NumGraftsPausedBudget = 0;
    NumGraftPliesApplied = 0;
    NumGraftNNEvals = 0;
    NumPriorNudgesApplied = 0;
    NumPriorNudgesCappedByMass = 0;
    DisabledByError = false;
    DisabledByErrorMessage = null;
    TimePumpTotalMs = 0;
    TimeDrainMs = 0;
    TimeGraftAdvanceMs = 0;
    TimeTargetSweepMs = 0;
    ProbeLatencySumMs = 0;
    ProbeLatencyMaxMs = 0;
    ProbeNodesSum = 0;
    ProbeDepthSum = 0;
    SumAbsAnchorQDeltaAtEnd = 0;
    MaxAbsAnchorQDeltaAtEnd = 0;
    NumAnchorsQMovedOverThreshold = 0;
    NumRootBestMoveChangesAfterGraft = 0;
    ShadowNumEvents = 0;
    ShadowNumConfirmedBySearch = 0;
    ShadowNumUnresolved = 0;
    NumDistinctNodesProbed = 0;
    Array.Clear(ProbesSubmittedAtDepth);
    Array.Clear(TriggersAtDepth);
  }

  #endregion


  /// <summary>
  /// Renders the one-line end-of-search summary (see ProbeGraftCoordinator.EndSearch).
  /// </summary>
  /// <param name="mode"></param>
  /// <param name="numEvalsThisSearch"></param>
  /// <param name="consumerLabel">suffix naming the non-graft consumers active (e.g. "+stamps")</param>
  /// <returns></returns>
  public string SummaryLine(ParamsProbeGraft.ModeType mode, int numEvalsThisSearch, string consumerLabel = null)
  {
    double graftEvalPct = numEvalsThisSearch > 0 ? 100.0 * NumGraftNNEvals / numEvalsThisSearch : 0;
    double probeLatAvg = NumProbesCompleted > 0 ? ProbeLatencySumMs / NumProbesCompleted : 0;

    string line = $"[ProbeGraft] mode={mode}{consumerLabel} probes: sub={NumProbesSubmitted} rejected={NumProbesRejectedBySource} done={NumProbesCompleted} "
                + $"agree={NumProbesAgreed} stale={NumResultsStale} | triggers: ref={NumTriggersRefutation} disc={NumTriggersDiscovery} "
                + $"grafts: start={NumGraftsStarted} done={NumGraftsCompleted} term={NumGraftsEndedTerminal} "
                + $"abort={NumGraftsAbortedMoveNotFound + NumGraftsAbortedPathDropped} "
                + $"(mnf={NumGraftsAbortedMoveNotFound} pd={NumGraftsAbortedPathDropped} pause={NumGraftsPausedBudget}) "
                + $"plies={NumGraftPliesApplied} "
                + $"evals={NumGraftNNEvals} ({graftEvalPct:F1}% of {numEvalsThisSearch / 1000.0:F1}k) nudges={NumPriorNudgesApplied} "
                + $"| pump={TimePumpTotalMs:F1}ms (drain={TimeDrainMs:F1} graft={TimeGraftAdvanceMs:F1} sweep={TimeTargetSweepMs:F1} n={NumTargetSweeps} sel={NumTargetsSelected} newNodes={NumDistinctNodesProbed}) "
                + $"probeLatAvg={probeLatAvg:F0}ms | impact: dQmax={MaxAbsAnchorQDeltaAtEnd:F3} "
                + $"movedGT.03={NumAnchorsQMovedOverThreshold} rootFlips={NumRootBestMoveChangesAfterGraft}";

    if (NumRootAdaptTriggers > 0 || NumRootProbeVerdicts > 0)
    {
      line += $" | rootadapt: verdicts={NumRootProbeVerdicts} trig={NumRootAdaptTriggers} flips={NumRootAdaptFlips} "
            + $"held={NumRootAdaptFlipsHeld} rollback={NumRootAdaptRollbacks} visits={NumRootAdaptVisits}";
    }

    if (NumAdvocateProbes > 0)
    {
      line += $" | advocate: probes={NumAdvocateProbes} agree={NumAdvocateAgreements} belowGate={NumAdvocateBelowGate} "
            + $"engaged={NumAdvocateEngagements} entries={NumAdvocateEntries} trace={NumAdvocateTraceVisits} retired={NumAdvocateRetired} "
            + $"campaigns={NumAdvocateCampaigns} flips={NumAdvocateCampaignFlips} rollback={NumAdvocateCampaignRollbacks} visits={NumAdvocateCampaignVisits} "
            + $"engagedMoves={NumAdvocateEngagedMoves} conv={NumAdvocateConversions} favLeft={NumAdvocateFavAbandoned}";
      if (NumAdvocateSoloWindows > 0 || NumAdvocateSoloDeferred > 0)
      {
        line += $" solo={NumAdvocateSoloWindows} deferred={NumAdvocateSoloDeferred} held={AdvocateSoloHeldMs:F0}ms";
      }
      if (NumAdvocateRootProbes > 0 || NumAdvocateFavProbes > 0)
      {
        line += $" dual: root={NumAdvocateRootProbes} fav={NumAdvocateFavProbes} refAlarms={NumAdvocateRefutationAlarms}";
      }
      if (NumAdvocateSlotRuns > 0)
      {
        line += $" cont: runs={NumAdvocateSlotRuns} depthAvg root={(double)AdvocateRootDepthSum / NumAdvocateSlotRuns:F1} fav={(double)AdvocateFavDepthSum / NumAdvocateSlotRuns:F1}";
      }
    }

    if (NumStampFlags > 0 || NumStamps > 0)
    {
      line += $" | stamps: flags={NumStampFlags} confirms={NumStampConfirmsSubmitted} rej={NumStampConfirmsRejected} stamped={NumStamps} "
            + $"cleared={NumStampsCleared} staleN={NumStampsStaleN} wOut={NumStampsWeightedOut} ancMoved={NumStampAncestorsMoved} rootTop={NumStampRootTopChanges} recompute={TimeStampRecomputeMs:F1}ms"
            + $" gate={NumGateDrains}/{NumGateDrainsSkippedBusy}/{NumGateDrainsBudgetBound}/{MaxGateDrainMs:F1}ms";
    }

    if (mode == ParamsProbeGraft.ModeType.Shadow)
    {
      line += $" | shadow: ev={ShadowNumEvents} conf={ShadowNumConfirmedBySearch} unres={ShadowNumUnresolved}";
    }

    if (DisabledByError)
    {
      line += $" | DISABLED-BY-ERROR: {DisabledByErrorMessage}";
    }

    return line;
  }


  /// <summary>
  /// Writes the full field dump (one "label : value" per line) for the diagnostics dump block.
  /// </summary>
  /// <param name="writer"></param>
  public void Dump(TextWriter writer)
  {
    writer.WriteLine("NumTargetSweeps               : " + NumTargetSweeps);
    writer.WriteLine("NumTargetsSelected            : " + NumTargetsSelected);
    writer.WriteLine("NumProbesSubmitted            : " + NumProbesSubmitted);
    writer.WriteLine("NumProbesRejectedBySource     : " + NumProbesRejectedBySource);
    writer.WriteLine("NumProbesCompleted            : " + NumProbesCompleted);
    writer.WriteLine("NumProbesIncomplete           : " + NumProbesIncomplete);
    writer.WriteLine("NumResultsStale               : " + NumResultsStale);
    writer.WriteLine("NumResultsDiscardedAtEnd      : " + NumResultsDiscardedAtEnd);
    writer.WriteLine("NumProbesAgreed               : " + NumProbesAgreed);
    writer.WriteLine("NumTriggersRefutation         : " + NumTriggersRefutation);
    writer.WriteLine("NumTriggersDiscovery          : " + NumTriggersDiscovery);
    writer.WriteLine("NumGraftsStarted              : " + NumGraftsStarted);
    writer.WriteLine("NumGraftsCompleted            : " + NumGraftsCompleted);
    writer.WriteLine("NumGraftsEndedTerminal        : " + NumGraftsEndedTerminal);
    writer.WriteLine("NumGraftsAbortedMoveNotFound  : " + NumGraftsAbortedMoveNotFound);
    writer.WriteLine("NumGraftsAbortedPathDropped   : " + NumGraftsAbortedPathDropped);
    writer.WriteLine("NumGraftsPausedBudget         : " + NumGraftsPausedBudget);
    writer.WriteLine("NumGraftPliesApplied          : " + NumGraftPliesApplied);
    writer.WriteLine("NumGraftNNEvals               : " + NumGraftNNEvals);
    writer.WriteLine("NumPriorNudgesApplied         : " + NumPriorNudgesApplied);
    writer.WriteLine("NumPriorNudgesCappedByMass    : " + NumPriorNudgesCappedByMass);
    writer.WriteLine("DisabledByError               : " + DisabledByError
                     + (DisabledByError ? " (" + DisabledByErrorMessage + ")" : ""));
    writer.WriteLine("TimePumpTotalMs               : " + TimePumpTotalMs.ToString("F2"));
    writer.WriteLine("TimeDrainMs                   : " + TimeDrainMs.ToString("F2"));
    writer.WriteLine("TimeGraftAdvanceMs            : " + TimeGraftAdvanceMs.ToString("F2"));
    writer.WriteLine("TimeTargetSweepMs             : " + TimeTargetSweepMs.ToString("F2"));
    writer.WriteLine("ProbeLatencySumMs             : " + ProbeLatencySumMs.ToString("F1"));
    writer.WriteLine("ProbeLatencyMaxMs             : " + ProbeLatencyMaxMs.ToString("F1"));
    writer.WriteLine("ProbeNodesSum                 : " + ProbeNodesSum);
    writer.WriteLine("ProbeDepthSum                 : " + ProbeDepthSum);
    writer.WriteLine("SumAbsAnchorQDeltaAtEnd       : " + SumAbsAnchorQDeltaAtEnd.ToString("F4"));
    writer.WriteLine("MaxAbsAnchorQDeltaAtEnd       : " + MaxAbsAnchorQDeltaAtEnd.ToString("F4"));
    writer.WriteLine("NumAnchorsQMovedOverThreshold : " + NumAnchorsQMovedOverThreshold);
    writer.WriteLine("NumRootBestMoveChangesAfterGraft : " + NumRootBestMoveChangesAfterGraft);
    writer.WriteLine("NumAdvocateEngagements        : " + NumAdvocateEngagements);
    writer.WriteLine("NumAdvocateEngagedMoves       : " + NumAdvocateEngagedMoves);
    writer.WriteLine("NumAdvocateConversions        : " + NumAdvocateConversions);
    writer.WriteLine("NumAdvocateFavAbandoned       : " + NumAdvocateFavAbandoned);
    writer.WriteLine("NumAdvocateSoloWindows        : " + NumAdvocateSoloWindows);
    writer.WriteLine("NumAdvocateSoloDeferred       : " + NumAdvocateSoloDeferred);
    writer.WriteLine("AdvocateSoloHeldMs            : " + AdvocateSoloHeldMs.ToString("F1"));
    writer.WriteLine("NumAdvocateRootProbes         : " + NumAdvocateRootProbes);
    writer.WriteLine("NumAdvocateFavProbes          : " + NumAdvocateFavProbes);
    writer.WriteLine("NumAdvocateRefutationAlarms   : " + NumAdvocateRefutationAlarms);
    writer.WriteLine("NumAdvocateSlotRuns           : " + NumAdvocateSlotRuns);
    writer.WriteLine("AdvocateRootDepthSum          : " + AdvocateRootDepthSum);
    writer.WriteLine("AdvocateFavDepthSum           : " + AdvocateFavDepthSum);
    writer.WriteLine("ShadowNumEvents               : " + ShadowNumEvents);
    writer.WriteLine("ShadowNumConfirmedBySearch    : " + ShadowNumConfirmedBySearch);
    writer.WriteLine("ShadowNumUnresolved           : " + ShadowNumUnresolved);
    writer.WriteLine(DepthHistogramLine());
  }
}
