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

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Parameters for the opt-in "refutation grafting" hybrid feature (PICKET M1).
///
/// Important interior nodes of the live search graph are asynchronously verified by an
/// external exact-search probe source (see ProbeSourceRegistry and the
/// Ceres.Chess.Probing.IProbeSource contract). When a probe refutes the move
/// currently dominating a node's visit distribution (or surfaces a strong move the policy
/// prior ignores), the probe's witness line is grafted into the graph as forced, committed,
/// NN-evaluated visits executed through the stock select/evaluate/backup pipeline, plus an
/// optional bounded prior nudge. No probe-derived value is ever written into the graph: the NN
/// evaluations of the grafted line are the sole evidence and ordinary backup draws the
/// conclusion. Probe scores are consulted only as trigger thresholds.
///
/// Disabled by default (a true no-op). Shadow mode runs targeting, probes and trigger
/// decisions and logs end-of-search verdicts without ever mutating the graph.
/// </summary>
[Serializable]
public record ParamsProbeGraft
{
  /// <summary>
  /// Constructor.
  /// </summary>
  public ParamsProbeGraft()
  {
  }

  /// <summary>
  /// Operating mode for the feature.
  /// </summary>
  public enum ModeType
  {
    /// <summary>Disabled: no targeting, no probes, no grafts (default; a true no-op).</summary>
    Disabled,

    /// <summary>
    /// Shadow: targeting and probes run and all decisions are computed and logged
    /// (including end-of-search verdicts), but no graft visits and no prior nudges
    /// are applied - the graph is never mutated by the feature.
    ///
    /// Verdict interpretation: a shadow event CONFIRMED by search is one plain search
    /// resolved anyway (a graft would only have accelerated it); UNRESOLVED events are
    /// the feature's potential value - they are where Active mode differs from baseline.
    /// </summary>
    Shadow,

    /// <summary>Active: grafts and prior nudges are applied.</summary>
    Active
  };

  /// <summary>Operating mode (Disabled / Shadow / Active).</summary>
  public ModeType Mode = ModeType.Disabled;

  /// <summary>
  /// True when any probe consumer is enabled, i.e. the coordinator (probe source session, inbox,
  /// targeter) must be constructed: the graft consumer (Mode != Disabled: Shadow observes, Active
  /// grafts), the probe stamps (ProbeStampMode Active) or root adapt. The consumers compose --
  /// stamps do NOT need Mode != Disabled; the deployment configuration for stamps is
  /// Mode = Disabled, ProbeStampMode = Active.
  /// </summary>
  public bool AnyConsumerEnabled => Mode != ModeType.Disabled
                                 || ProbeStampMode == ProbeStampModeType.Active
                                 || RootAdaptMode == RootAdaptModeType.Active
                                 || AdvocateMode == AdvocateModeType.Active;

  /// <summary>
  /// True when the advocate issues directed visits (PV traces or root campaigns). Under
  /// DualOverlappedIterators these run inside the PhaseCoordinator transient solo window (the peer
  /// iterator parks at a batch boundary for the duration of the campaign/trace; SWARM doc XI.8.1),
  /// with serial select scoped to the forced descents themselves. An attention-only advocate (no
  /// traces, no campaigns) writes only the graph's attention snapshot and never requests a window.
  /// </summary>
  public bool AdvocateDirectedVisitsEnabled => AdvocateMode == AdvocateModeType.Active
                                            && (AdvocateTracePlies > 0 || AdvocateMaxCampaignsPerMove > 0);

  /// <summary>
  /// True when this configuration may run with Execution.DualOverlappedIterators: every active
  /// consumer is gate-safe -- the probe stamps (incremental targeting, applied inside the backup
  /// gate) and/or the advocate (attention snapshot swaps in the gate; its directed visits, if any,
  /// run in the PhaseCoordinator transient solo window). The graft consumer (Mode Active), root
  /// adapt and the periodic full-graph sweep still need a single-iterator quiescent graph at the
  /// post-batch pump.
  /// </summary>
  public bool OverlappedIteratorsAllowed => Mode != ModeType.Active
                                         && RootAdaptMode == RootAdaptModeType.Disabled
                                         && (ProbeStampMode != ProbeStampModeType.Active || ProbeStampIncrementalTargeting)
                                         && (ProbeStampMode == ProbeStampModeType.Active
                                          || AdvocateMode == AdvocateModeType.Active);

  #region Probe stamps (SWARM Part VIII: external probe values consumed through the backup)

  /// <summary>Probe stamps: Disabled, or Active (nodes probed and stamped in-search; see ProbeStampMode).</summary>
  public enum ProbeStampModeType { Disabled, Active };

  /// <summary>
  /// When Active, the coordinator's targeting sweep probes every node whose N lies in
  /// [TargetNCutAbs, ProbeStampMaxN] (re-probed when N grows by ReprobeNGrowthFactor), flags
  /// |A - Q| >= ProbeStampTheta (A = probe cp mapped to Ceres' logistic Q, node mover's view),
  /// confirms with one probe of ProbeStampConfirmMultiplier x the budget, and STAMPS the node
  /// (ProbeStamps.Stamp + RecomputeUpward) at the pump. The stamp's weight in the node's parents
  /// is ParamsSelect.TPS_ProbeStampKappa (which must be > 0 for the stamps to have any effect).
  /// Replaces the refutation/discovery graft triggers for the probes it issues.
  /// </summary>
  public ProbeStampModeType ProbeStampMode = ProbeStampModeType.Disabled;

  /// <summary>Flag threshold |A - Q| in logistic Q units.</summary>
  public float ProbeStampTheta = 0.25f;

  /// <summary>Confirmation probe budget as a multiple of the first probe (0 = stamp on the first probe).</summary>
  public int ProbeStampConfirmMultiplier = 3;

  /// <summary>Confirmation must keep the sign and reach this fraction of ProbeStampTheta.</summary>
  public float ProbeStampConfirmFraction = 0.7f;

  /// <summary>Nodes with N above this are not probed for stamping (the ratio principle).</summary>
  public int ProbeStampMaxN = 2048;

  /// <summary>Probe budget = clamp(ProbeStampBudgetRatio x N, ProbeNodeBudget, ProbeStampBudgetMax).</summary>
  public int ProbeStampBudgetRatio = 10;

  /// <summary>Cap of the ratio-scaled probe budget.</summary>
  public int ProbeStampBudgetMax = 100_000;

  /// <summary>
  /// Cap on probes submitted per search in stamp mode (first probes and confirmations together);
  /// 0 = unlimited (default). A fixed cap bounds the prober CPU per move but makes the pool go
  /// quiet once it is spent (at ~1,000-1,500 probes per 50k nodes a cap of 1,500 bound at
  /// ~50-100k nodes); unlimited, the prober is paced by MaxInFlightProbes and the source's
  /// saturation backoff so long moves keep it busy to the end.
  /// </summary>
  public int ProbeStampMaxProbesPerSearch = 0;

  /// <summary>
  /// Maximum stamps applied (bytes written + ancestors recomputed) per batch by the in-gate drain
  /// (ProbeGraftCoordinator.DrainInBackupGate); the rest wait in the inbox for the next batch.
  /// Bounds the extension of the select/backup exclusion window per batch (a stamp recompute is
  /// ~0.1-1 ms). 0 = unlimited.
  /// </summary>
  public int ProbeStampMaxAppliesPerBatch = 8;

  /// <summary>
  /// When true (default), stamp-mode targeting is incremental: at every pump the nodes on the
  /// batch's backed-up paths are checked for crossing the probe threshold (N in
  /// [TargetNCutAbs, ProbeStampMaxN], never probed or grown by ReprobeNGrowthFactor since the last
  /// probe, per the node's persistent ProbedN record), instead of a periodic full-graph sweep every
  /// TargetSweepIntervalSeconds. Removes the sweep cost and, under graph reuse, the re-probing of
  /// nodes already probed in earlier moves. When false the periodic sweep is used.
  /// </summary>
  public bool ProbeStampIncrementalTargeting = true;

  /// <summary>
  /// Precision-indexed stamp weights by the probe's |cp| (null = all 1): "minCp:weight,..." buckets,
  /// e.g. "0:1.5,1:0.5,50:0,100:0.7,150:1,200:1.5". The stamp's n0 is scaled by the weight
  /// (K = kappa x n0 x w); weight 0 suppresses the stamp. See ProbeStamps.CpWeight.
  /// </summary>
  public string ProbeStampCpWeights = null;

  #endregion

  #region Advocate (SWARM Part XI: continuous Dissenter + frontier attention + root campaigns)

  /// <summary>Advocate: Disabled, or Active (in-search Dissenter probes of the root; see AdvocateMode).</summary>
  public enum AdvocateModeType { Disabled, Active };

  /// <summary>
  /// When Active, the coordinator runs the in-search ADVOCATE (Part XI): a recurring "Dissenter"
  /// probe of the search root (one probe with DominantMove = the current max-N root move, so the
  /// result carries the prober's best move/score/PV AND the favourite's score and refutation line).
  /// When the prober disagrees by AdvocateMarginCp or more it feeds back ATTENTION, never values:
  /// fading select bonuses (Graph.AttentionEntries, consumed via ParamsSelect.AttentionBonusEpsilon,
  /// which must be > 0 for any effect) on the root edge (AdvocateRootAttention) and on the frontier
  /// found by walking both PVs through the live graph (AdvocateInterior), short PV traces beyond
  /// the frontier, and the proven adaptive root campaign with rollback (caps shared with the
  /// RootAdapt* parameters). Composes with the stamps; the deployment configuration is
  /// Mode = Disabled, AdvocateMode = Active.
  /// </summary>
  public AdvocateModeType AdvocateMode = AdvocateModeType.Disabled;

  /// <summary>
  /// Dual-probe Dissenter (SWARM doc XI.10): TWO always-running prober searches instead of one
  /// probe per verdict -- a persistent unrestricted ROOT search ("what would a tactician play?",
  /// resubmitted back-to-back on the warm transposition table, its verdict valid across favourite
  /// changes) and a FAVOURITE search restricted (searchmoves) to the move MCGS currently favours,
  /// retargeted whenever the favourite changes. A verdict is assembled from the latest of each:
  /// engagement when root.best != favourite and root.score - fav.score >= AdvocateMarginCp; and a
  /// refutation-only alarm (frontier attention on the favourite's refutation, no campaign) when the
  /// root search agrees with the favourite but the full-budget favourite search scores it
  /// AdvocateMarginCp lower. Needs a probe source honouring ProbeRequest.RestrictToMove (the
  /// Ceres.AlphaBeta farm does; otherwise falls back to the single-probe form) and >= 2 workers.
  /// </summary>
  public bool AdvocateDualProbes = false;

  /// <summary>
  /// Continuous form of the dual-probe Dissenter (XI.11): the root and favourite searches are
  /// OPEN-ENDED (unbounded budget, ended by cancellation at move end -- the favourite search is
  /// cancelled and restarted the moment the favourite changes) and report after every completed
  /// iteration, so verdict depth grows with the move clock and nothing in flight is wasted. Verdicts
  /// are decided at MATCHED depth from each search's score-by-depth history: engagement when the root
  /// search prefers another move by >= AdvocateMarginCp at depth d and d-1 (d = min of the two
  /// depths, >= AdvocateMinVerdictDepth); a refutation-only alarm when the favourite search fails low
  /// by >= AdvocateMarginCp over its last two iterations. Requires AdvocateDualProbes and a source
  /// honouring ProbeRequest.ReportInterim (the Ceres.AlphaBeta farm).
  /// </summary>
  public bool AdvocateContinuousProbes = false;

  /// <summary>Continuous form: minimum matched depth before any verdict is drawn.</summary>
  public int AdvocateMinVerdictDepth = 10;

  /// <summary>Continuous form: a verdict on the same favourite is repeated only after the matched
  /// depth has advanced by this many plies.</summary>
  public int AdvocateReengageDepthStep = 2;

  /// <summary>Prober node budget of each Dissenter probe (the X.9 dose-response favours deep probes).</summary>
  public int AdvocateProbeNodes = 200_000;

  /// <summary>Engagement gate: prober margin (cp) of its move over the search favourite.</summary>
  public int AdvocateMarginCp = 100;

  /// <summary>Attention fade mass K = kappa x child N at entry time (bonus = eps x W x K/(N+K)).</summary>
  public float AdvocateAttentionKappa = 2.0f;

  /// <summary>Full weight (W = 1) margin: entry weight W = min(1, margin / this).</summary>
  public int AdvocateWFullCp = 400;

  /// <summary>Place an attention entry on the root -> prober-move edge while engaged
  /// (the step-change knob on the disagreement suite, XI.5.2).</summary>
  public bool AdvocateRootAttention = true;

  /// <summary>Walk both PVs to the first material break and place interior attention there.</summary>
  public bool AdvocateInterior = true;

  /// <summary>Plies of PV materialized beyond a frontier, 1 visit per ply (0 = no traces;
  /// traces are directed visits and forbid overlapped iterators).</summary>
  public int AdvocateTracePlies = 6;

  /// <summary>Attention entries on the frontier edge plus this many successor PV edges.</summary>
  public int AdvocateAttnPlies = 2;

  /// <summary>Frontier targets held concurrently per side (prober's PV / refutation of the favourite).</summary>
  public int AdvocateTargetsPerSide = 2;

  /// <summary>Minimum node N for a frontier value break (below is noise).</summary>
  public int AdvocateNMin = 32;

  /// <summary>Frontier value-break threshold |claim - Q| in logistic Q units (0 = use ProbeStampTheta).</summary>
  public float AdvocateTheta = 0.25f;

  /// <summary>Max plies walked along either PV during attribution.</summary>
  public int AdvocateMaxPlies = 12;

  /// <summary>Trace-visit budget per search as a fraction of the node limit (or of root N under time limits).</summary>
  public float AdvocateTraceBudgetFraction = 0.03f;

  /// <summary>Dissenter cadence: minimum seconds between probes while the verdict is unchanged
  /// (a favourite change re-probes immediately).</summary>
  public float AdvocateMinIntervalSeconds = 0.5f;

  /// <summary>Root campaigns per move (0 = none; campaigns are directed visits, run in the
  /// transient solo window under overlapped iterators). Caps/chunking/rollback shared with RootAdaptVisitsPerPump,
  /// RootAdaptMaxVisits, RootAdaptMaxVisitsFraction, RootAdaptMarginQ, RootAdaptRollback.</summary>
  public int AdvocateMaxCampaignsPerMove = 4;

  /// <summary>Fraction of the search budget that must have elapsed before a campaign may start
  /// (early campaigns fight an unformed graph; suites fired from the second stage boundary).</summary>
  public float AdvocateCampaignMinFraction = 0.10f;

  /// <summary>Latest progress fraction at which a campaign may START (a flip needs ~8k visits to
  /// land; a campaign begun with less budget than that gets cut by the clock mid-flight and its
  /// visits are wasted -- the losing case's seconds are not refundable under a time limit).</summary>
  public float AdvocateCampaignMaxFraction = 0.85f;

  #endregion

  /// <summary>
  /// ID of the probe source registered with ProbeSourceRegistry. Null/empty with
  /// Mode != Disabled is a validation error (there is nothing to probe with).
  /// </summary>
  public string ProbeSourceID = null;

  #region Targeting (region walk; see ProbeGraftTargeter)

  /// <summary>Seconds between target-selection sweeps (also min spacing of probe issuance).</summary>
  public float TargetSweepIntervalSeconds = 0.5f;

  /// <summary>Max plies below the search root considered by the target sweep.</summary>
  public int TargetMaxDepth = 12;

  /// <summary>Node visit-count floor as a fraction of root N (mirrors CutFraction in RootMinimaxBlend).</summary>
  public float TargetCutFraction = 0.002f;

  /// <summary>Absolute node visit-count floor for targeting.</summary>
  public int TargetNCutAbs = 500;

  /// <summary>Max nodes selected per sweep (top by priority score).</summary>
  public int TargetMaxNodesPerSweep = 64;

  /// <summary>Max in-flight (issued, uncompleted) probe requests.</summary>
  public int MaxInFlightProbes = 256;

  /// <summary>
  /// Per-node minimum re-probe spacing, as a multiple of growth in the node's N
  /// (a node is re-probed only after its N has grown by this factor since its last probe).
  /// </summary>
  public float ReprobeNGrowthFactor = 2.0f;

  /// <summary>Node budget passed to the probe source per request.</summary>
  public int ProbeNodeBudget = 200_000;

  /// <summary>
  /// If the search root itself is eligible as a probe target. The root's probe asks the one
  /// question that maps directly onto the move actually played ("is the move we are about to
  /// play refuted?"), so it carries by far the most decision leverage; every other target can
  /// only matter to the extent its value propagates up to a root child. Off by default because
  /// acting on a root refutation is closer to a root veto than to ordinary grafting.
  /// </summary>
  public bool TargetIncludeSearchRoot = false;

  /// <summary>
  /// Max depth below the search root at which the shallow (faster) re-probe cadence applies.
  /// 0 restricts it to the search root itself, so with TargetIncludeSearchRoot false the shallow
  /// cadence is inert and re-probe spacing is uniformly ReprobeNGrowthFactor.
  /// </summary>
  public int ReprobeShallowMaxDepth = 0;

  /// <summary>
  /// Re-probe spacing (as a multiple of growth in the node's N) for nodes at or above
  /// ReprobeShallowMaxDepth. Because ReprobeNGrowthFactor is geometric, a node is probed only
  /// O(log N) times across a search, which saturates coverage of the few high-leverage nodes near
  /// the root long before probe capacity is exhausted. A smaller factor here re-probes those nodes
  /// as the search evolves. Ignored when it exceeds ReprobeNGrowthFactor.
  /// </summary>
  public float ReprobeNGrowthFactorShallow = 1.10f;

  /// <summary>
  /// How candidate targets are ranked within a sweep.
  /// </summary>
  public enum TargetPriorityModeType
  {
    /// <summary>By node visit count (highest N first), zero when the dominant-visit gate fails.</summary>
    VisitCount,

    /// <summary>
    /// By estimated proximity to changing the root's move choice: for a candidate X under root
    /// child C, the value change required at X to reorder the root is approximately
    /// (gap between C and the best alternative root child) * N(C) / N(X). Candidates are ranked by
    /// ascending required change, which folds leverage and root urgency into one score and demotes
    /// deep, low-influence nodes without a hard depth cap.
    /// </summary>
    RootFlipDistance
  }

  /// <summary>Ranking rule used to order candidates within a target sweep.</summary>
  public TargetPriorityModeType TargetPriorityMode = TargetPriorityModeType.VisitCount;

  /// <summary>
  /// RootFlipDistance mode only: candidates whose estimated required value change exceeds this
  /// (in Q units, so the full range is 2.0) are not probed at all, on the grounds that no plausible
  /// refutation there could change the move played. Values >= 2 disable the gate.
  /// </summary>
  public float TargetMaxRequiredDelta = 2.0f;

  #endregion

  #region Graft triggering (probe result interpretation)

  /// <summary>
  /// Minimum centipawn margin by which the probe must refute the dominant move at the node
  /// (fail-low of the dominant move / fail-high of an alternative, per the probe's own scores;
  /// used ONLY as a trigger threshold, never converted into a graph value).
  /// </summary>
  public int TriggerMarginCp = 200;

  /// <summary>Minimum probe search depth (iterations) for a result to be trusted as a trigger.</summary>
  public int TriggerMinProbeDepth = 10;

  /// <summary>
  /// Fraction of the node's visits the dominant child must hold for "refuted dominant move"
  /// triggering to apply (below this the node has no meaningful dominant line to refute).
  /// </summary>
  public float DominantChildMinVisitFraction = 0.45f;

  #endregion

  #region Graft application

  /// <summary>Maximum plies of the witness PV grafted (spine length cap).</summary>
  public int GraftMaxPlies = 16;

  /// <summary>Committed visits sent through each grafted ply (the ply's forced ProbeSpec NumVisits).</summary>
  public int GraftVisitsPerPly = 1;

  /// <summary>Maximum concurrently advancing grafts.</summary>
  public int MaxActiveGrafts = 32;

  /// <summary>
  /// Hard cap on total NN evaluations consumed by grafting per search, as a fraction of
  /// evaluations performed so far (grafting pauses when the cap is reached). Guards GPU budget.
  /// </summary>
  public float MaxGraftEvalFraction = 0.03f;

  /// <summary>
  /// If nonzero, the prior of the probe's refutation move at each grafted ply is raised to at
  /// least this value (renormalizing the node's other priors down proportionally), capped by
  /// PriorNudgeMaxTotalMass per node over the search. 0 disables nudging entirely.
  /// Also serves as the low-prior threshold of the secondary "discovery" trigger (see
  /// ProbeGraftCoordinator): a probe best move whose prior is below this value is one the
  /// policy effectively ignores.
  /// </summary>
  public float PriorNudgeFloor = 0.03f;

  /// <summary>
  /// If prior nudges are allowed to take effect. DEFAULT FALSE: nudging is inert unless explicitly
  /// enabled.
  ///
  /// Nudging is the only part of this feature that mutates persistent state. ApplyPriorNudges writes
  /// through GEdge.SetPolicyPrior directly into the graph, nothing restores the original policy, and
  /// game play reuses the graph across moves -- so a nudge outlives the search that applied it. The
  /// per-node PriorNudgeMaxTotalMass ledger lives on the scheduler, which is rebuilt each search, so
  /// the cap is per-move rather than per-game and repeated nudges at one node compound (each one
  /// rescales every sibling, so unnudged siblings decay geometrically).
  ///
  /// It is a separate switch from PriorNudgeFloor rather than a zero value of it because the floor
  /// doubles as the discovery trigger's low-prior threshold: zeroing the floor would silently
  /// disable discovery triggering as well, which is the large majority of all triggers.
  /// </summary>
  public bool AllowPriorNudges = false;

  /// <summary>Cap on cumulative prior mass moved by nudges at any single node.</summary>
  public float PriorNudgeMaxTotalMass = 0.15f;

  #region Root verification and veto (proposal M5)

  /// <summary>
  /// Root verification mode (proposal M5). The root probe asks the one question that maps directly
  /// onto the move played -- "is the move we are about to play refuted?" -- and, unlike grafting,
  /// acting on the answer does not have to fight averaging inertia: it is one decision at move
  /// commit rather than an attempt to move a large-N ancestor's Q with a handful of visits.
  /// </summary>
  public enum RootVetoModeType
  {
    /// <summary>No root verification.</summary>
    Disabled,

    /// <summary>
    /// Record what a veto WOULD have done without changing the move played. This is the kill-gate:
    /// it yields a fire rate and, when adjudicated against a deeper search, a precision.
    /// </summary>
    Shadow,

    /// <summary>
    /// Act on the verdict at move commit. NOT IMPLEMENTED -- rejected by Validate until the Shadow
    /// measurement justifies building it.
    /// </summary>
    Active
  }

  /// <summary>Root verification mode. See RootVetoModeType.</summary>
  public RootVetoModeType RootVetoMode = RootVetoModeType.Disabled;

  /// <summary>
  /// Minimum margin (centipawns) by which the probe must prefer its move over the move the search
  /// is about to play before a veto is considered. Deliberately higher than TriggerMarginCp: a veto
  /// overrides the decision outright, so it should fire only on a clear demonstrated difference.
  /// </summary>
  public int RootVetoMinMarginCp = 200;

  /// <summary>Minimum probe depth for a root verdict to be considered trustworthy.</summary>
  public int RootVetoMinProbeDepth = 10;

  #endregion


  #region Root adaptive directed visits (LOCUS rootadapt)

  /// <summary>Mode for the root adaptive directed-visit rule. See RootAdaptModeType.</summary>
  public enum RootAdaptModeType
  {
    /// <summary>Disabled (default).</summary>
    Disabled,

    /// <summary>
    /// Active: once the search has used RootAdaptTriggerFraction of its budget, if the latest root
    /// probe prefers a move other than the search's current top-N move, directed visits are forced
    /// into that move (one chunk per pump, stock PUCT below it, MCGS's own net evaluating) until
    /// its value exceeds the best reliable sibling's by RootAdaptMarginQ -- at which point ordinary
    /// selection takes over -- or the visit cap is reached, in which case the forced visits are
    /// rolled back in N only (the line keeps what it learned, the edge regains its exploration
    /// credit). No probe value is ever written into the graph; the move changes only if the net,
    /// led to the line, rates it above the incumbent. Requires RootVetoMode != Disabled (the root
    /// verdict is the trigger) and single-threaded select (SelectOperationParallelThresholdNumVisits
    /// = int.MaxValue).
    /// </summary>
    Active
  }

  /// <summary>Root adaptive directed-visit mode.</summary>
  public RootAdaptModeType RootAdaptMode = RootAdaptModeType.Disabled;

  /// <summary>Fraction of the search budget (nodes or time) that must have elapsed before the rule may fire.</summary>
  public float RootAdaptTriggerFraction = 0.6f;

  /// <summary>Cap on directed visits per search as a fraction of the node budget (when the limit is in nodes).</summary>
  public float RootAdaptMaxVisitsFraction = 0.2f;

  /// <summary>Absolute cap on directed visits per search.</summary>
  public int RootAdaptMaxVisits = 40_000;

  /// <summary>Visits forced per pump (each pump is one quiescent point between batches).</summary>
  public int RootAdaptVisitsPerPump = 256;

  /// <summary>Margin (Q units) by which the forced move's value must exceed the best reliable sibling's to count as a flip.</summary>
  public float RootAdaptMarginQ = 0.0f;

  /// <summary>If true, directed visits that did not produce a flip are rolled back in N (recommended).</summary>
  public bool RootAdaptRollback = true;

  #endregion

  /// <summary>
  /// If true, edges along a completed graft path are marked stale so that a configured
  /// PostBackupQMode == StaleDrain pass propagates the graft's evidence upward promptly.
  /// No effect (and validated as an error) unless PostBackupQMode == StaleDrain.
  /// </summary>
  public bool MarkGraftPathStale = false;

  #endregion

  #region Diagnostics

  /// <summary>If a one-line per-search summary of graft statistics is printed at end of search.</summary>
  public bool EnableStatsSummary = true;

  /// <summary>If per-event (probe/graft) detail lines are logged (verbose; Shadow-mode tuning aid).</summary>
  public bool VerboseEventLogging = false;

  #endregion

  /// <summary>
  /// Validates settings for self-consistency (including cross-checks against the parent
  /// ParamsSearch). Called from ParamsSearch.Validate(). All checks apply only when some
  /// consumer is enabled (AnyConsumerEnabled).
  /// </summary>
  /// <param name="parent"></param>
  public void Validate(ParamsSearch parent)
  {
    if (!AnyConsumerEnabled)
    {
      return;
    }

    if (RootVetoMode == RootVetoModeType.Active)
    {
      throw new Exception("ParamsProbeGraft.RootVetoMode Active is not implemented yet "
                        + "(build it only if the Shadow measurement justifies it).");
    }

    if (ProbeStampMode == ProbeStampModeType.Active)
    {
      if (ProbeStampTheta <= 0 || ProbeStampConfirmMultiplier < 0 || ProbeStampConfirmFraction < 0 || ProbeStampMaxN < 1
       || ProbeStampBudgetRatio < 1 || ProbeStampBudgetMax < ProbeNodeBudget)
      {
        throw new Exception("ParamsProbeGraft.ProbeStamp* settings out of range.");
      }
      if (ProbeStampMaxProbesPerSearch < 0 || ProbeStampMaxAppliesPerBatch < 0)
      {
        throw new Exception("ParamsProbeGraft.ProbeStampMaxProbesPerSearch / ProbeStampMaxAppliesPerBatch must be >= 0 (0 = unlimited).");
      }
      ProbeGraft.ProbeStamps.ParseCpWeights(ProbeStampCpWeights);   // validates the syntax and range
    }

    if (RootVetoMode != RootVetoModeType.Disabled && !TargetIncludeSearchRoot)
    {
      throw new Exception("ParamsProbeGraft.RootVetoMode requires TargetIncludeSearchRoot "
                        + "(the root is never probed otherwise, so no verdict can be formed).");
    }

    if (RootAdaptMode == RootAdaptModeType.Active)
    {
      if (RootVetoMode == RootVetoModeType.Disabled)
      {
        throw new Exception("ParamsProbeGraft.RootAdaptMode requires RootVetoMode != Disabled (the root verdict is its trigger).");
      }
      if (parent.Execution.SelectOperationParallelThresholdNumVisits != int.MaxValue)
      {
        throw new Exception("ParamsProbeGraft.RootAdaptMode requires Execution.SelectOperationParallelThresholdNumVisits == int.MaxValue "
                          + "(forced visits through one child are not supported by the parallel select-descent path).");
      }
      if (RootAdaptTriggerFraction < 0 || RootAdaptTriggerFraction >= 1 || RootAdaptMaxVisitsFraction <= 0 || RootAdaptMaxVisits <= 0 || RootAdaptVisitsPerPump <= 0)
      {
        throw new Exception("ParamsProbeGraft.RootAdapt* settings out of range.");
      }
    }

    if (AdvocateMode == AdvocateModeType.Active)
    {
      if (AdvocateProbeNodes < 1_000)
      {
        throw new Exception("ParamsProbeGraft.AdvocateProbeNodes must be at least 1000.");
      }
      if (AdvocateMarginCp < 0 || AdvocateWFullCp < 1)
      {
        throw new Exception("ParamsProbeGraft.AdvocateMarginCp/AdvocateWFullCp out of range.");
      }
      if (AdvocateAttentionKappa <= 0 || float.IsNaN(AdvocateAttentionKappa))
      {
        throw new Exception("ParamsProbeGraft.AdvocateAttentionKappa must be > 0.");
      }
      if (AdvocateTracePlies < 0 || AdvocateAttnPlies < 0 || AdvocateTargetsPerSide < 1
       || AdvocateNMin < 1 || AdvocateMaxPlies < 1 || AdvocateMaxCampaignsPerMove < 0
       || AdvocateMinVerdictDepth < 2 || AdvocateReengageDepthStep < 1)
      {
        throw new Exception("ParamsProbeGraft.Advocate* count settings out of range.");
      }
      if (AdvocateContinuousProbes && !AdvocateDualProbes)
      {
        throw new Exception("ParamsProbeGraft.AdvocateContinuousProbes requires AdvocateDualProbes.");
      }
      if (AdvocateTheta < 0 || float.IsNaN(AdvocateTheta)
       || AdvocateTraceBudgetFraction < 0 || AdvocateTraceBudgetFraction > 1
       || AdvocateCampaignMinFraction < 0 || AdvocateCampaignMinFraction >= 1
       || AdvocateCampaignMaxFraction <= AdvocateCampaignMinFraction || AdvocateCampaignMaxFraction > 1
       || AdvocateMinIntervalSeconds < 0)
      {
        throw new Exception("ParamsProbeGraft.Advocate* fraction/threshold settings out of range.");
      }
      // Directed visits under overlapped iterators run in the transient solo window, which parks the
      // peer at its batch boundary; the per-N-batches sync barrier would rendezvous with a parked
      // peer and spin until the search stops.
      if (AdvocateDirectedVisitsEnabled && parent.Execution.DualOverlappedIterators && parent.Execution.SyncEveryNBatches > 0)
      {
        throw new Exception("ParamsProbeGraft.AdvocateMode with traces or campaigns under Execution.DualOverlappedIterators "
                          + "requires Execution.SyncEveryNBatches == 0 (the solo window parks the peer iterator).");
      }
    }

    if (string.IsNullOrEmpty(ProbeSourceID))
    {
      throw new Exception("ParamsProbeGraft.ProbeSourceID must be set when any probe consumer is enabled (Mode, ProbeStampMode, RootAdaptMode or AdvocateMode).");
    }

    // Overlapped iterators are supported only for gate-safe consumers (Route 4 stamps, and the
    // advocate, whose directed visits run in the transient solo window). The legacy consumers
    // that issue directed visits through RunProbeSpecs outside a window (graft Mode Active,
    // RootAdapt) and the periodic full-graph sweep need the single-iterator quiescent graph at
    // the post-batch pump.
    if (parent.Execution.DualOverlappedIterators && !OverlappedIteratorsAllowed)
    {
      throw new Exception("ParamsProbeGraft: Execution.DualOverlappedIterators is only supported when every active "
                        + "consumer is gate-safe: probe stamps with ProbeStampIncrementalTargeting and/or the "
                        + "advocate (AdvocateMode Active; traces/campaigns run in the solo window), "
                        + "with Mode != Active and RootAdaptMode Disabled.");
    }

    if (MarkGraftPathStale && parent.PostBackupQMode != ParamsSearch.PostBackupQModeType.StaleDrain)
    {
      throw new Exception("ParamsProbeGraft.MarkGraftPathStale requires PostBackupQMode == StaleDrain.");
    }

    if (GraftMaxPlies < 1)
    {
      throw new Exception("ParamsProbeGraft.GraftMaxPlies must be at least 1.");
    }

    if (GraftVisitsPerPly < 1)
    {
      throw new Exception("ParamsProbeGraft.GraftVisitsPerPly must be at least 1.");
    }

    if (TriggerMarginCp < 0)
    {
      throw new Exception("ParamsProbeGraft.TriggerMarginCp must be non-negative.");
    }

    if (PriorNudgeFloor < 0 || PriorNudgeFloor > 0.2f)
    {
      throw new Exception("ParamsProbeGraft.PriorNudgeFloor must be in [0, 0.2].");
    }

    if (PriorNudgeFloor > 0 && (PriorNudgeMaxTotalMass <= 0 || PriorNudgeMaxTotalMass > 0.5f))
    {
      throw new Exception("ParamsProbeGraft.PriorNudgeMaxTotalMass must be in (0, 0.5] when nudging enabled.");
    }

    if (MaxGraftEvalFraction <= 0 || MaxGraftEvalFraction > 0.25f)
    {
      throw new Exception("ParamsProbeGraft.MaxGraftEvalFraction must be in (0, 0.25].");
    }

    if (TargetCutFraction < 0 || TargetCutFraction > 0.5f)
    {
      throw new Exception("ParamsProbeGraft.TargetCutFraction must be in [0, 0.5] (0 = the cut is TargetNCutAbs alone, independent of root N).");
    }

    if (TargetNCutAbs < 1)
    {
      throw new Exception("ParamsProbeGraft.TargetNCutAbs must be at least 1.");
    }

    if (TargetSweepIntervalSeconds <= 0)
    {
      throw new Exception("ParamsProbeGraft.TargetSweepIntervalSeconds must be positive.");
    }

    if (TargetMaxDepth < 1)
    {
      throw new Exception("ParamsProbeGraft.TargetMaxDepth must be at least 1.");
    }

    if (TargetMaxNodesPerSweep < 1)
    {
      throw new Exception("ParamsProbeGraft.TargetMaxNodesPerSweep must be at least 1.");
    }

    if (MaxInFlightProbes < 1)
    {
      throw new Exception("ParamsProbeGraft.MaxInFlightProbes must be at least 1.");
    }

    if (ReprobeNGrowthFactor < 1)
    {
      throw new Exception("ParamsProbeGraft.ReprobeNGrowthFactor must be at least 1.");
    }

    if (ReprobeNGrowthFactorShallow < 1)
    {
      throw new Exception("ParamsProbeGraft.ReprobeNGrowthFactorShallow must be at least 1.");
    }

    if (ReprobeShallowMaxDepth < 0)
    {
      throw new Exception("ParamsProbeGraft.ReprobeShallowMaxDepth cannot be negative.");
    }

    if (TargetMaxRequiredDelta <= 0)
    {
      throw new Exception("ParamsProbeGraft.TargetMaxRequiredDelta must be positive.");
    }

    if (ProbeNodeBudget < 1)
    {
      throw new Exception("ParamsProbeGraft.ProbeNodeBudget must be at least 1.");
    }

    if (MaxActiveGrafts < 1)
    {
      throw new Exception("ParamsProbeGraft.MaxActiveGrafts must be at least 1.");
    }
  }
}
