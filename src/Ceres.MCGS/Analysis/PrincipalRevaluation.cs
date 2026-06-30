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
using System.Linq;

using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Analysis;


/// <summary>
/// Interpretation assigned to a frontier position based on how its deep-rollout
/// evidence relates to what the search believed about it.
/// </summary>
public enum FrontierClass : byte
{
  /// <summary>Rollout evidence agrees with the search's value (within tolerance).</summary>
  Confirming,

  /// <summary>Rollouts indicate the position is worse for the root player than the search believed.</summary>
  SearchOptimistic,

  /// <summary>Rollouts indicate the position is better for the root player than the search believed.</summary>
  SearchPessimistic,

  /// <summary>Search saw a decisive value but deep rollouts consistently end near a draw.</summary>
  DeepDraw,

  /// <summary>Rollout outcomes are widely dispersed; the position needs more search, not reinterpretation.</summary>
  Volatile,

  /// <summary>Terminal or unexpandable node whose graph value is exact (no rollouts needed).</summary>
  ExactLeaf
}


/// <summary>
/// Per-ladder-stage rollout statistics for one frontier node
/// (all Q values from the frontier node's side-to-move perspective).
/// </summary>
public sealed class FrontierStageStats
{
  /// <summary>Exploration multiplier of the stage.</summary>
  public float Epsilon;

  /// <summary>Number of rollouts applied in this stage.</summary>
  public int NumRollouts;

  /// <summary>Rollout-frequency-weighted mean leaf Q (NaN if no rollouts).</summary>
  public double Mean;

  /// <summary>Number of distinct maximal lines explored.</summary>
  public int NumDistinct;

  /// <summary>Standard deviation of leaf Q over distinct maximal lines.</summary>
  public double StdDistinct;

  /// <summary>Median depth below the node over distinct maximal lines.</summary>
  public double MedianDepth;

  /// <summary>Maximum depth descended below the node across all rollouts of the stage.</summary>
  public int MaxDepth;

  /// <summary>Terminal outcome counts (node's perspective).</summary>
  public int TerminalWin, TerminalDraw, TerminalLoss;

  /// <summary>Leaf Q of each distinct maximal line.</summary>
  public IReadOnlyList<double> DistinctLeafQ;
}


/// <summary>
/// Revaluation evidence and conclusion for a single frontier (or exact-leaf) position.
/// Values VHat/Q0/Q1 are from the node's own side-to-move perspective.
/// </summary>
public sealed class FrontierEval
{
  public NodeIndex Node;
  public int N;

  /// <summary>Node Q snapshotted before the rollout ladder ran.</summary>
  public double Q0;

  /// <summary>Node Q after the rollout ladder ran (the high-support anchor).</summary>
  public double Q1;

  /// <summary>Blended revalued estimate used by the re-backup.</summary>
  public double VHat;

  /// <summary>Uncertainty of VHat.</summary>
  public double Sigma;

  /// <summary>Ladder-combined deep probe mean (NaN if no usable stage).</summary>
  public double VDeep;

  /// <summary>Uncertainty of the deep probe alone (NaN if no usable stage).</summary>
  public double SigmaDeep = double.NaN;

  /// <summary>
  /// Raw rollout-count-weighted mean leaf Q over all stages (node's perspective; NaN if no
  /// rollouts). Pure rollout evidence with no anchor blend and no stage weighting.
  /// </summary>
  public double RolloutMeanRaw = double.NaN;

  /// <summary>Pooled per-line dispersion across stages.</summary>
  public double SPooled;

  /// <summary>Fraction of distinct rollout leaves that are draw-like (|q| small or terminal draw).</summary>
  public double FracDraw;

  public FrontierClass Class;

  /// <summary>True when the greedy stage followed a single line to substantial depth.</summary>
  public bool ForcedDeeperRead;

  /// <summary>True when the greedy stage reached a terminal.</summary>
  public bool TerminalResolved;

  /// <summary>True when low- and high-exploration stage means differ sharply (refutation sensitivity).</summary>
  public bool TacticallyUnstable;

  /// <summary>True for terminal/unexpandable nodes whose value is exact.</summary>
  public bool IsExactLeaf;

  /// <summary>True if at least one ladder stage produced usable rollout statistics.</summary>
  public bool HasRolloutStats;

  /// <summary>
  /// Signed influence of this node's value on the root's averaged Q
  /// (sign encodes perspective parity; magnitude is the visit-mass weight).
  /// </summary>
  public double InfluenceS;

  /// <summary>Per-stage rollout statistics (empty for exact leaves / capped-out nodes).</summary>
  public List<FrontierStageStats> Stages = new();
}


/// <summary>
/// Revalued estimates for one root move (all Q values from the root player's perspective).
/// </summary>
public sealed class RootMoveReval
{
  /// <summary>Index of the corresponding edge within the root's child edges.</summary>
  public int ChildSlot;

  public MGMove Move;

  /// <summary>Visit count of the root edge (after the rollout ladder ran).</summary>
  public int EdgeN;

  /// <summary>
  /// Visit count of the root edge before the rollout ladder ran. Used (together with QOrig)
  /// for any "would this move have qualified under the standard chooser" test, which must be
  /// judged on the search's own evidence, not on state inflated by the revaluation's rollouts.
  /// </summary>
  public int EdgeNOriginal;

  /// <summary>Move Q before rollouts (snapshot of -edge.Q).</summary>
  public double QOrig;

  /// <summary>Revalued move Q under the visit-weighted-average operator.</summary>
  public double QAvg;

  /// <summary>Revalued move Q under the negamax operator.</summary>
  public double QNegamax;

  /// <summary>Revalued move Q under the soft-minimax (power mean) operator. The headline value.</summary>
  public double QSoft;

  public double SigmaAvg, SigmaNegamax, SigmaSoft;

  /// <summary>If the child subtree was actually revalued (otherwise Q values just echo the live edge Q).</summary>
  public bool InRegion;

  /// <summary>Influence-weighted fraction of this move's frontier mass classified Volatile.</summary>
  public double VolatileMass;

  /// <summary>Influence-weighted fraction of this move's frontier mass classified DeepDraw.</summary>
  public double DrawMass;

  /// <summary>Number of frontier/exact-leaf positions inside this move's revalued subtree.</summary>
  public int NumFrontier;

  /// <summary>Total distinct rollout lines explored below this move's frontier positions (all ladder stages).</summary>
  public int DistinctPaths;

  /// <summary>Total rollout visits applied below this move's frontier positions.</summary>
  public int RolloutVisits;

  /// <summary>Distinct-line-weighted average of the per-frontier-node median rollout depths (NaN if none).</summary>
  public double AvgRolloutDepth;

  /// <summary>Deepest rollout descent below any of this move's frontier positions.</summary>
  public int MaxRolloutDepth;

  /// <summary>
  /// Pure rollout-evidence Q: rollout-count-weighted mean of the stage means over this move's
  /// frontier positions, converted to the root player's perspective (NaN if no rollouts).
  /// Uncontaminated by the anchor blend - contrast with QAvg/QSoft.
  /// </summary>
  public double DRQ = double.NaN;

  /// <summary>
  /// Coverage-extrapolated move Q (root perspective): the influence-weighted mean of the
  /// blended frontier estimates over this move's subtree, renormalized over the subtree's
  /// covered mass - i.e. the move's frontier correction projected over its uncovered mass
  /// as well. The aggressive estimate (NaN if the subtree has no covered frontier).
  /// </summary>
  public double QExtrapolated = double.NaN;

  public double SigmaExtrapolated = double.NaN;

  /// <summary>Fraction of this move's value mass explained by the frontier cut (0 to 1).</summary>
  public double FrontierCoverage;
}


/// <summary>
/// Complete result of a principal-position rollout revaluation pass.
/// All root-level Q values are from the root player's (side to move at root) perspective.
/// </summary>
public sealed class PrincipalRevaluationResult
{
  /// <summary>Root Q snapshotted before the rollout ladder.</summary>
  public double RootQOriginal;

  /// <summary>Root Q as stored in the graph after the rollout ladder.</summary>
  public double RootQPostRollout;

  /// <summary>Re-backup root values under the three operators.</summary>
  public double RootQAvg, RootQNegamax, RootQSoft;

  public double RootSigmaAvg, RootSigmaNegamax, RootSigmaSoft;

  /// <summary>
  /// Re-backup root value under the average operator with frontier values left at their
  /// post-rollout graph Q (no blending) - the baseline for the linearity invariant.
  /// </summary>
  public double RootQAvgBaseline;

  /// <summary>First-order root correction: sum over frontier of influence times (VHat - Q1).</summary>
  public double FirstOrderDeltaQ;

  /// <summary>Total influence mass of the cut (1 = the cut fully explains root Q).</summary>
  public double Coverage;

  /// <summary>
  /// Coverage-renormalized root estimate from raw rollout evidence alone: the influence-weighted
  /// mean of the frontier rollout means (root perspective). Answers "what does deep play say the
  /// root is worth, if the uncovered mass behaves like the covered mass" - deliberately NOT
  /// diluted by below-cut values the way the re-backup operators are. NaN if no rollout stats.
  /// </summary>
  public double RootDRQ = double.NaN;

  public double RootDRQSigma = double.NaN;

  /// <summary>
  /// Coverage-renormalized root estimate from the blended frontier values (VHat): the
  /// influence-weighted mean over the cut (root perspective). Sits between RootDRQ (pure
  /// rollouts) and RootQAvg (full dilution by below-cut mass).
  /// </summary>
  public double RootQFrontierExtrapolated = double.NaN;

  public double RootQFrontierExtrapolatedSigma = double.NaN;

  /// <summary>RootDRQ - RootQOriginal (NaN if RootDRQ unavailable).</summary>
  public double RootQRawDelta = double.NaN;

  /// <summary>|RootQRawDelta| in units of RootDRQSigma.</summary>
  public double RootQRawZ = double.NaN;

  /// <summary>Verdict on whether the search's top-level Q is contradicted by the rollout evidence.</summary>
  public RootQAssessment Assessment = RootQAssessment.Unknown;

  /// <summary>Dominant frontier classification by influence mass (null = mixed).</summary>
  public FrontierClass? RootClass;

  /// <summary>Influence-mass fraction per classification (normalized over the cut).</summary>
  public Dictionary<FrontierClass, double> ClassMass = new();

  /// <summary>Per root move revalued values, one entry per visited root edge.</summary>
  public List<RootMoveReval> RootMoves = new();

  /// <summary>Frontier evidence keyed by node index (for diagnostics/dump integration).</summary>
  public Dictionary<int, FrontierEval> FrontierByNodeIndex = new();

  public int NumFrontier, NumExactLeaves, NumRolloutVisits;

  /// <summary>
  /// Per-ladder-stage execution summary: rollout visits actually applied and how many of the
  /// rollable nodes were stopped early by dry-up detection / by reaching a terminal
  /// (explains why fewer rollouts ran than were requested).
  /// </summary>
  public List<(float Epsilon, int Nodes, int Visits, int StoppedDry, int StoppedTerminal)> StageSummaries = new();

  /// <summary>Total distinct rollout lines explored over all frontier positions and ladder stages.</summary>
  public int TotalDistinctPaths;

  /// <summary>Distinct-line-weighted average of the per-frontier-node median rollout depths (NaN if none).</summary>
  public double AvgRolloutDepth;

  /// <summary>Deepest rollout descent below any frontier position.</summary>
  public int MaxRolloutDepth;

  public double ElapsedSecs;

  /// <summary>True if the phase aborted early (deadline/stop) and results may be partial.</summary>
  public bool Aborted;


  /// <summary>
  /// Returns the revalued soft-minimax Q (root perspective) for the root move at the
  /// specified child slot, if the move's subtree was actually revalued.
  /// </summary>
  public bool TryGetRevaluedQ(int childSlot, out float q, out float sigma)
  {
    foreach (RootMoveReval rm in RootMoves)
    {
      if (rm.ChildSlot == childSlot && rm.InRegion)
      {
        q = (float)rm.QSoft;
        sigma = (float)rm.SigmaSoft;
        return true;
      }
    }

    q = sigma = float.NaN;
    return false;
  }
}


/// <summary>
/// Verdict on whether the search's top-level Q is contradicted by the raw rollout evidence.
/// </summary>
public enum RootQAssessment : byte
{
  /// <summary>No rollout evidence available.</summary>
  Unknown,

  /// <summary>Rollout evidence agrees with the search's root Q.</summary>
  Consistent,

  /// <summary>Rollout evidence departs noticeably from the search's root Q.</summary>
  Drifting,

  /// <summary>Rollout evidence strongly contradicts the search's root Q (top-level Q is probably wrong).</summary>
  Suspect
}


/// <summary>
/// Outcome of evaluating whether the revalued per-move Q estimates would change the move choice.
/// </summary>
public enum RevaluationSwitchOutcome : byte
{
  /// <summary>The candidate won the blended-Q comparison among the close-in-Q top-level moves.</summary>
  SwitchByBlendedQ,

  /// <summary>The baseline move remains best under the blended-Q comparison.</summary>
  KeepBlendedQBest,

  /// <summary>Fewer than two top-level moves lie within the close-Q window (comparison not meaningful).</summary>
  KeepNoCloseAlternatives,

  /// <summary>A non-baseline move won the comparison but lacks sufficient rollout lines.</summary>
  KeepBlendWinnerUnsupported,

  /// <summary>The baseline decision was not an ordinary search result.</summary>
  KeepNotApplicable,

  /// <summary>The winner would lack the visit support the standard chooser requires (N-floor).</summary>
  KeepInsufficientN
}


/// <summary>
/// Result of the purely informational blended-Q move comparison against a baseline best-move
/// decision: the outcome, the moves involved, and the supporting numbers. ANALYSIS ONLY -
/// reported by the revalue-root command and the auto-analysis; move choice is never affected.
/// </summary>
public sealed class RevaluationSwitchDecision
{
  public RevaluationSwitchOutcome Outcome;

  /// <summary>The baseline (search-chosen) move.</summary>
  public MGMove BaselineMove;

  /// <summary>The best alternative considered (default if none).</summary>
  public MGMove CandidateMove;

  /// <summary>Root child slot of the candidate (-1 if none).</summary>
  public int CandidateSlot = -1;

  /// <summary>Blended (Q/DRQ) effective values of baseline/candidate (NaN unless evaluated).</summary>
  public double BaselineQBlend = double.NaN, CandidateQBlend = double.NaN;

  /// <summary>Distinct rollout lines of the candidate.</summary>
  public int CandidateDistinctPaths;

  /// <summary>Frontier coverage of the revaluation.</summary>
  public double Coverage;

  public bool WouldSwitch => Outcome is RevaluationSwitchOutcome.SwitchByBlendedQ;

  /// <summary>One-line human-readable account of the decision.</summary>
  public string Description => Outcome switch
  {
    RevaluationSwitchOutcome.SwitchByBlendedQ =>
      $"blended Q ({PrincipalRevaluation.MOVE_BLEND_WEIGHT_Q:F2}*Q + {PrincipalRevaluation.MOVE_BLEND_WEIGHT_DRQ:F2}*DRQ):"
      + $" {BaselineQBlend:F3} -> {CandidateQBlend:F3} (candidate #DR {CandidateDistinctPaths})",
    RevaluationSwitchOutcome.KeepBlendedQBest =>
      $"baseline best under blended Q ({BaselineQBlend:F3} vs best alt {CandidateQBlend:F3})",
    RevaluationSwitchOutcome.KeepNoCloseAlternatives =>
      $"no second move within {PrincipalRevaluation.MOVE_BLEND_CLOSE_Q_WINDOW:F2} of best Q",
    RevaluationSwitchOutcome.KeepBlendWinnerUnsupported =>
      $"raw-Q winner {CandidateMove} lacks rollout support (#DR {CandidateDistinctPaths} <= {PrincipalRevaluation.MOVE_BLEND_MIN_DISTINCT_PATHS})",
    RevaluationSwitchOutcome.KeepNotApplicable =>
      "baseline decision not an ordinary search result",
    RevaluationSwitchOutcome.KeepInsufficientN =>
      $"candidate {CandidateMove} would win but lacks the visit support required to qualify (N-floor)",
    _ => Outcome.ToString()
  };
}


/// <summary>
/// Post-search analysis pass that re-estimates the root (and per-root-move) value of a completed
/// MCGS search by combining deep-rollout evidence gathered at the frontier of the well-visited
/// subtree with a re-propagation of those revalued frontier estimates to the root under a family
/// of backup operators (visit-weighted average, negamax, soft-minimax power mean).
///
/// The only graph-mutating step is the rollout ladder itself (ordinary extra search visits that
/// back up normally); the re-backup computes exclusively into side buffers.
///
/// Note: rollout visits added by the ladder persist in the graph (and into any reused graph for
/// the following move). This is accepted by design.
/// </summary>
public static partial class PrincipalRevaluation
{
  #region Analysis tunable constants

  // NOTE: this is an ANALYSIS/DISPLAY feature only - move choice during play is never affected.
  // The MOVE_BLEND_* constants govern the purely informational "would the rollout evidence
  // prefer a different move" report (CalcBlendedQSwitchDecision) printed by revalue-root and
  // the ALWAYS_DUMP_SEARCH_INFO auto-analysis:
  //   1. Only when MORE THAN ONE top-level move lies within MOVE_BLEND_CLOSE_Q_WINDOW of the
  //      best move Q is the comparison meaningful at all.
  //   2. Each close move with more than MOVE_BLEND_MIN_DISTINCT_PATHS distinct rollout lines
  //      competes using blended Q = MOVE_BLEND_WEIGHT_Q * (move Q) + MOVE_BLEND_WEIGHT_DRQ * DRQ;
  //      close moves without sufficient rollout lines compete with their plain move Q.
  //   3. The move with the best (blended) Q wins; a hypothetical switch away from the baseline
  //      additionally requires the winner to have sufficient rollout lines and to satisfy the
  //      standard chooser's visit-support floor (judged on pre-rollout snapshot values).

  public const double MOVE_BLEND_CLOSE_Q_WINDOW = 0.03;
  public const int MOVE_BLEND_MIN_DISTINCT_PATHS = 20;   // strictly more than this required
  public const double MOVE_BLEND_WEIGHT_Q = 0.70;
  public const double MOVE_BLEND_WEIGHT_DRQ = 0.30;

  /// <summary>
  /// Minimum blended-Q advantage of a non-baseline winner required to report a switch
  /// (0 = pure argmax). Noise damper: blended values of close moves often differ by
  /// only a few thousandths, where a preference is essentially a coin flip.
  /// </summary>
  public const double MOVE_BLEND_SWITCH_MARGIN = 0.0;

  /// <summary>
  /// Display filter for the revalue-root per-move table: moves with Q more than this
  /// below the best move Q are omitted (the chosen move is always shown).
  /// </summary>
  public const double DUMP_MAX_Q_GAP_FROM_BEST = 0.10;

  /// <summary>
  /// Default dry-up rounds for analysis runs (the revalue-root command and the
  /// ALWAYS_DUMP_SEARCH_INFO auto-analysis).
  /// </summary>
  public const int ANALYSIS_DRY_UP_ROUNDS = 12;

  /// <summary>
  /// Default exploration multipliers for the rollout ladder (descending; the final greedy
  /// stage - the minimax sample with the largest estimator weight - runs on the
  /// best-informed graph).
  /// </summary>
  public static readonly float[] DEFAULT_EPSILON_LADDER = [0.15f, 0.04f, 0f];

  /// <summary>Frontier cut threshold as a fraction of root N.</summary>
  public const double CUT_FRACTION = 0.01;

  /// <summary>Maximum number of frontier positions from which rollouts are launched (top-K by N).</summary>
  public const int MAX_FRONTIER_POSITIONS = 64;

  /// <summary>Exponent of the visit-weighted power mean used by the soft-minimax backup operator.</summary>
  public const double SOFTMAX_P = 8;

  #endregion

  #region Constants

  /// <summary>Absolute floor on the frontier cut visit threshold.</summary>
  const int MIN_VISITS_ABS = 32;

  /// <summary>Uncertainty assigned to below-cut / cycle children consumed at their edge Q.</summary>
  const double SIGMA_BELOWCUT = 0.15;

  /// <summary>Variance floor added to the deep-probe estimate.</summary>
  const double SIGMA_DEEP_FLOOR = 0.03;

  /// <summary>Irreducible model uncertainty added to every blended frontier estimate.</summary>
  const double SIGMA_MODEL = 0.02;

  /// <summary>Base distrust of a frontier node's averaged Q (encodes dilution-bias prior).</summary>
  const double SIGMA_Q_BASE = 0.08;

  /// <summary>Scale on the NN value-head self-reported uncertainty in the anchor sigma.</summary>
  const double SIGMA_Q_UV_SCALE = 1.0;

  /// <summary>Default value-head uncertainty when the network does not provide one.</summary>
  const double UV_DEFAULT = 0.10;

  /// <summary>Minimum rollouts for a ladder stage to contribute to the estimator.</summary>
  const int MIN_STAGE_ROLLOUTS = 4;

  /// <summary>Minimum greedy-stage depth for the forced-line estimator override.</summary>
  const int FORCED_MIN_DEPTH = 8;

  /// <summary>|q| below which a rollout leaf counts as draw-like.</summary>
  const double DRAW_DELTA = 0.05;

  /// <summary>Anchor Q below which a greedy draw outcome triggers the draw-escape floor.</summary>
  const double DRAW_ESCAPE_Q = 0.10;

  /// <summary>Value floor applied by the draw-escape rule.</summary>
  const double DRAW_ESCAPE_MARGIN = 0.05;

  /// <summary>Sigma cap applied by the draw-escape rule.</summary>
  const double DRAW_ESCAPE_SIGMA_CAP = 0.10;

  /// <summary>|VHat - Q0| within which a frontier node is Confirming.</summary>
  const double CONFIRM_DELTA = 0.05;

  /// <summary>Pooled dispersion above which a frontier node is Volatile.</summary>
  const double VOLATILE_STD = 0.25;

  /// <summary>DeepDraw thresholds.</summary>
  const double DRAW_ABS_Q = 0.08;
  const double DRAW_STD = 0.10;
  const double DRAW_FRAC = 0.70;
  const double DRAW_MIN_PRIOR = 0.25;

  /// <summary>Gap between lowest- and highest-exploration stage means flagging tactical instability.</summary>
  const double TACTICAL_EPS_GAP = 0.15;

  /// <summary>Negamax/soft selection-noise model: children within this band of the max count toward m.</summary>
  const double SEL_BAND = 0.05;

  /// <summary>Negamax/soft selection-noise scale (sigma = scale * sqrt(ln(1+m))).</summary>
  const double SIGMA_SELECT = 0.03;

  /// <summary>Shift applied before the power mean so all child values are positive.</summary>
  const double SOFTMAX_SHIFT = 1.01;

  /// <summary>Backstop on nodes processed by the re-backup walk.</summary>
  const int MAX_REBACKUP_NODES = 1_000_000;

  /// <summary>Minimum class influence-mass fraction required to label the root.</summary>
  const double ROOT_CLASS_FRAC = 0.5;

  /// <summary>Root Q assessment thresholds: |RootDRQ - rootQ| and its z-score.</summary>
  const double ASSESS_DELTA_SUSPECT = 0.15;
  const double ASSESS_Z_SUSPECT = 2.5;
  const double ASSESS_DELTA_DRIFT = 0.08;
  const double ASSESS_Z_DRIFT = 1.5;

  /// <summary>Minimum-visit floor for the negamax child candidate set.</summary>
  const int NEGAMAX_MIN_N_ABS = 32;
  const double NEGAMAX_MIN_N_FRAC = 0.05;

  /// <summary>Canonical estimator weights by ladder stage rank (ascending epsilon).</summary>
  static readonly double[] LAMBDA_DEFAULT = [0.5, 0.3, 0.2];
  static readonly double[] LAMBDA_FORCED = [0.7, 0.2, 0.1];

  #endregion


  /// <summary>
  /// Runs the full revaluation pass against the manager's completed search.
  /// </summary>
  /// <param name="manager">Manager whose search has completed (graph quiescent).</param>
  /// <param name="roundsPerStage">Rollout rounds per frontier position per ladder stage.</param>
  /// <param name="epsilonLadder">Exploration multipliers for the rollout ladder (null = DEFAULT_EPSILON_LADDER).</param>
  /// <param name="deadline">Optional wall-clock deadline; remaining ladder stages are skipped once passed.</param>
  /// <param name="dryUpRounds">Dry-up rounds for the rollouts (0 disables dry-up detection).</param>
  public static PrincipalRevaluationResult Run(MCGSManager manager, int roundsPerStage,
                                               float[] epsilonLadder = null,
                                               DateTime? deadline = null,
                                               int dryUpRounds = ANALYSIS_DRY_UP_ROUNDS)
  {
    ArgumentNullException.ThrowIfNull(manager);

    DateTime startTime = DateTime.Now;
    GNode root = manager.Engine.SearchRootNode;
    PrincipalRevaluationResult result = new();

    int nCut = Math.Max(MIN_VISITS_ABS, (int)(CUT_FRACTION * root.N));

    // Phase 1: collect the above-cut region and classify its nodes.
    Region region = CollectRegion(root, nCut);
    result.NumExactLeaves = region.ExactLeaves.Count;

    // Phase 2: snapshot Q of the whole region (and root edge Qs) before any rollout backs up.
    foreach (KeyValuePair<int, NodeKind> kv in region.Kind)
    {
      region.SnapshotQ[kv.Key] = manager.Engine.Graph[new NodeIndex(kv.Key)].Q;
    }
    result.RootQOriginal = root.Q;
    double[] rootEdgeQ0 = new double[root.NumEdgesExpanded];
    int[] rootEdgeN0 = new int[root.NumEdgesExpanded];
    for (int slot = 0; slot < root.NumEdgesExpanded; slot++)
    {
      GEdge rootEdge = root.ChildEdgeAtIndex(slot);
      rootEdgeQ0[slot] = rootEdge.Q;
      rootEdgeN0[slot] = rootEdge.N;
    }

    // Phase 3: rollout ladder over the rollable frontier (top-K by N).
    List<GNode> rollable = region.Frontier
                                 .OrderByDescending(n => n.N)
                                 .Take(MAX_FRONTIER_POSITIONS)
                                 .ToList();
    result.NumFrontier = region.Frontier.Count;

    Dictionary<int, List<FrontierStageStats>> stagesByNode = new();
    if (rollable.Count > 0 && roundsPerStage > 0)
    {
      NodeIndex[] startNodes = rollable.Select(n => n.Index).ToArray();

      // Descending epsilon order: wide probes refine child values first so the final greedy
      // stage (which receives the largest estimator weight) runs on the best-informed graph.
      float[] ladder = (float[])(epsilonLadder ?? DEFAULT_EPSILON_LADDER).Clone();
      Array.Sort(ladder);
      Array.Reverse(ladder);

      MCGSManager.MCGSProgressCallback savedCallback = manager.ProgressCallback;
      try
      {
        foreach (float epsilon in ladder)
        {
          if ((deadline.HasValue && DateTime.Now >= deadline.Value) || manager.ExternalStopRequested)
          {
            result.Aborted = true;
            break;
          }

          bool isGreedyStage = epsilon == 0;
          DeepRolloutSet drSet = DeepRolloutSet.Run(manager, startNodes, roundsPerStage, epsilon,
                                                    stopNodeVisitsIfTerminalReached: isGreedyStage,
                                                    deadline: deadline,
                                                    dryUpRounds: dryUpRounds);
          int stageVisits = 0, stageDry = 0, stageTerminal = 0;
          foreach (DeepRolloutNodeStats s in drSet.Results)
          {
            stageVisits += s.NumVisits;
            if (s.StoppedByDryUp)
            {
              stageDry++;
            }
            if (s.StoppedByTerminal)
            {
              stageTerminal++;
            }
            if (!stagesByNode.TryGetValue(s.Node.Index, out List<FrontierStageStats> stages))
            {
              stagesByNode[s.Node.Index] = stages = new List<FrontierStageStats>();
            }

            stages.Add(new FrontierStageStats
            {
              Epsilon = epsilon,
              NumRollouts = s.NumVisits,
              Mean = s.AvgLeafQAllPaths,
              NumDistinct = s.NumDistinctPaths,
              StdDistinct = s.StdDevLeafQ,
              MedianDepth = s.MedianDepthBelowNode,
              MaxDepth = s.MaxDepthBelowNode,
              TerminalWin = s.NumTerminalWin,
              TerminalDraw = s.NumTerminalDraw,
              TerminalLoss = s.NumTerminalLoss,
              DistinctLeafQ = s.MaximalPathLeafQ
            });
            result.NumRolloutVisits += s.NumVisits;
          }

          result.StageSummaries.Add((epsilon, drSet.Results.Count, stageVisits, stageDry, stageTerminal));
        }
      }
      finally
      {
        manager.ProgressCallback = savedCallback;
      }
    }

    // Phase 4: per-frontier estimates (post-rollout Q is the anchor).
    bool trackVolatility = manager.ParamsSearch.TrackLeafValueVolatility;
    foreach (GNode f in region.Frontier)
    {
      stagesByNode.TryGetValue(f.Index.Index, out List<FrontierStageStats> stages);
      FrontierEval eval = BuildFrontierEval(f, region.SnapshotQ[f.Index.Index], stages, trackVolatility);
      result.FrontierByNodeIndex[f.Index.Index] = eval;
    }
    foreach (GNode x in region.ExactLeaves)
    {
      double q = x.Q;
      result.FrontierByNodeIndex[x.Index.Index] = new FrontierEval
      {
        Node = x.Index,
        N = x.N,
        Q0 = region.SnapshotQ[x.Index.Index],
        Q1 = q,
        VHat = double.IsNaN(q) ? 0 : q,
        Sigma = 0,
        VDeep = double.NaN,
        Class = FrontierClass.ExactLeaf,
        IsExactLeaf = true
      };
    }

    // Phase 5: re-backup with blended frontier values, and a baseline pass (frontier at its
    // post-rollout graph Q) for the linearity diagnostic.
    Rebackup main = RunRebackup(root, region, result.FrontierByNodeIndex, SOFTMAX_P, useVHat: true);
    Rebackup baseline = RunRebackup(root, region, result.FrontierByNodeIndex, SOFTMAX_P, useVHat: false);

    result.RootQPostRollout = root.Q;
    OpVal rootVal = main.Memo[root.Index.Index];
    result.RootQAvg = rootVal.A;
    result.RootQNegamax = rootVal.B;
    result.RootQSoft = rootVal.C;
    result.RootSigmaAvg = rootVal.SA;
    result.RootSigmaNegamax = rootVal.SB;
    result.RootSigmaSoft = rootVal.SC;
    result.RootQAvgBaseline = baseline.Memo[root.Index.Index].A;

    // Phase 6: influence pass over the (acyclic) accepted-edge skeleton, first-order
    // correction and classification aggregation.
    Dictionary<int, double> influence = PropagateInfluence(main, root.Index.Index);

    double coverage = 0;
    double firstOrder = 0;
    int totalDistinct = 0, maxRolloutDepth = 0;
    double depthSum = 0, depthWeight = 0;
    double sumSVHat = 0, varExtrap = 0;
    double rawCoverage = 0, sumSRaw = 0, varRaw = 0;
    Dictionary<FrontierClass, double> classMass = new();
    foreach (KeyValuePair<int, FrontierEval> kv in result.FrontierByNodeIndex)
    {
      FrontierEval eval = kv.Value;
      influence.TryGetValue(kv.Key, out double s);
      eval.InfluenceS = s;
      double w = Math.Abs(s);
      coverage += w;
      if (!eval.IsExactLeaf)
      {
        firstOrder += s * (eval.VHat - eval.Q1);

        // Optimistic/pessimistic need the root-perspective sign, which the influence chain encodes.
        FinalizeDirectionalClass(eval, Math.Sign(s));
      }

      classMass.TryGetValue(eval.Class, out double m);
      classMass[eval.Class] = m + w;

      // Coverage-renormalized root estimators: blended frontier values, and raw rollout means.
      sumSVHat += s * eval.VHat;
      varExtrap += s * s * eval.Sigma * eval.Sigma;
      if (!double.IsNaN(eval.RolloutMeanRaw))
      {
        rawCoverage += w;
        sumSRaw += s * eval.RolloutMeanRaw;
        double sd = double.IsNaN(eval.SigmaDeep) ? SIGMA_BELOWCUT : eval.SigmaDeep;
        varRaw += s * s * sd * sd;
      }

      foreach (FrontierStageStats stage in eval.Stages)
      {
        totalDistinct += stage.NumDistinct;
        if (stage.MaxDepth > maxRolloutDepth)
        {
          maxRolloutDepth = stage.MaxDepth;
        }
        if (stage.NumDistinct > 0 && !double.IsNaN(stage.MedianDepth))
        {
          depthSum += stage.MedianDepth * stage.NumDistinct;
          depthWeight += stage.NumDistinct;
        }
      }
    }

    result.TotalDistinctPaths = totalDistinct;
    result.AvgRolloutDepth = depthWeight > 0 ? depthSum / depthWeight : double.NaN;
    result.MaxRolloutDepth = maxRolloutDepth;

    if (coverage > 0)
    {
      result.RootQFrontierExtrapolated = sumSVHat / coverage;
      result.RootQFrontierExtrapolatedSigma = Math.Sqrt(varExtrap) / coverage;
    }
    if (rawCoverage > 0)
    {
      result.RootDRQ = sumSRaw / rawCoverage;
      result.RootDRQSigma = Math.Sqrt(varRaw) / rawCoverage;

      // Assessment of the search's top-level Q against the raw rollout evidence.
      result.RootQRawDelta = result.RootDRQ - result.RootQOriginal;
      result.RootQRawZ = Math.Abs(result.RootQRawDelta) / Math.Max(0.02, result.RootDRQSigma);
      double absDelta = Math.Abs(result.RootQRawDelta);
      result.Assessment = absDelta >= ASSESS_DELTA_SUSPECT && result.RootQRawZ >= ASSESS_Z_SUSPECT
                            ? RootQAssessment.Suspect
                            : absDelta >= ASSESS_DELTA_DRIFT && result.RootQRawZ >= ASSESS_Z_DRIFT
                                ? RootQAssessment.Drifting
                                : RootQAssessment.Consistent;
    }

    result.Coverage = coverage;
    result.FirstOrderDeltaQ = firstOrder;
    if (coverage > 0)
    {
      foreach (FrontierClass c in classMass.Keys.ToList())
      {
        classMass[c] /= coverage;
      }
    }
    result.ClassMass = classMass;
    KeyValuePair<FrontierClass, double> top = classMass.OrderByDescending(kv => kv.Value).FirstOrDefault();
    result.RootClass = top.Value > ROOT_CLASS_FRAC ? top.Key : null;

    // Phase 7: per-root-move outputs (capped to the snapshot length in case
    // the rollouts expanded additional root edges).
    MGPosition rootPos = root.CalcPosition();
    for (int slot = 0; slot < Math.Min((int)root.NumEdgesExpanded, rootEdgeQ0.Length); slot++)
    {
      GEdge e = root.ChildEdgeAtIndex(slot);
      if (e.N == 0)
      {
        continue;
      }

      RootMoveReval rm = new()
      {
        ChildSlot = slot,
        Move = e.MoveMGFromPos(in rootPos),
        EdgeN = e.N,
        EdgeNOriginal = rootEdgeN0[slot],
        QOrig = -rootEdgeQ0[slot]
      };

      double dilution = e.N == 0 ? 1 : (double)(e.N - e.NDrawByRepetition) / e.N;
      int childIdx = e.Type == GEdgeStruct.EdgeType.ChildEdge && !e.ChildNodeIndex.IsNull
                       ? e.ChildNodeIndex.Index : -1;
      if (childIdx >= 0 && main.Memo.TryGetValue(childIdx, out OpVal cv))
      {
        rm.InRegion = true;
        rm.QAvg = -cv.A * dilution;
        rm.QNegamax = -cv.B * dilution;
        rm.QSoft = -cv.C * dilution;
        rm.SigmaAvg = cv.SA * dilution;
        rm.SigmaNegamax = cv.SB * dilution;
        rm.SigmaSoft = cv.SC * dilution;

        // Frontier-mass composition and rollout statistics of this move's subtree
        // (separate influence seed at the child; transposed frontier positions are
        // counted toward every root move that reaches them).
        Dictionary<int, double> moveInfluence = PropagateInfluence(main, childIdx);
        double total = 0, volatileMass = 0, drawMass = 0;
        int numFrontier = 0;
        double moveDepthSum = 0, moveDepthWeight = 0;
        double drqSum = 0, drqWeight = 0;
        double xtrSum = 0, xtrVar = 0;
        foreach (KeyValuePair<int, FrontierEval> kv in result.FrontierByNodeIndex)
        {
          if (moveInfluence.TryGetValue(kv.Key, out double s) && s != 0)
          {
            double w = Math.Abs(s);
            total += w;
            numFrontier++;

            // Move-level extrapolation accumulators (child's perspective; converted below).
            xtrSum += s * kv.Value.VHat;
            xtrVar += s * s * kv.Value.Sigma * kv.Value.Sigma;
            if (kv.Value.Class == FrontierClass.Volatile)
            {
              volatileMass += w;
            }
            else if (kv.Value.Class == FrontierClass.DeepDraw)
            {
              drawMass += w;
            }

            // Sign of the (root-seeded) influence encodes the node's perspective parity
            // relative to the root, needed to aggregate stage means across mixed depths.
            int paritySign = Math.Sign(kv.Value.InfluenceS);

            foreach (FrontierStageStats stage in kv.Value.Stages)
            {
              rm.DistinctPaths += stage.NumDistinct;
              rm.RolloutVisits += stage.NumRollouts;
              if (stage.MaxDepth > rm.MaxRolloutDepth)
              {
                rm.MaxRolloutDepth = stage.MaxDepth;
              }
              if (stage.NumDistinct > 0 && !double.IsNaN(stage.MedianDepth))
              {
                moveDepthSum += stage.MedianDepth * stage.NumDistinct;
                moveDepthWeight += stage.NumDistinct;
              }
              if (paritySign != 0 && stage.NumRollouts > 0 && !double.IsNaN(stage.Mean))
              {
                drqSum += paritySign * stage.Mean * stage.NumRollouts;
                drqWeight += stage.NumRollouts;
              }
            }
          }
        }
        rm.NumFrontier = numFrontier;
        rm.VolatileMass = total > 0 ? volatileMass / total : 0;
        rm.DrawMass = total > 0 ? drawMass / total : 0;
        rm.AvgRolloutDepth = moveDepthWeight > 0 ? moveDepthSum / moveDepthWeight : double.NaN;
        rm.DRQ = drqWeight > 0 ? drqSum / drqWeight : double.NaN;
        rm.FrontierCoverage = total;
        if (total > 0)
        {
          // The accumulated sum is the cut's contribution to the child's value (child
          // perspective); renormalize over covered mass, then negate and dilute for the
          // root-perspective move value, exactly as the operator outputs are converted.
          rm.QExtrapolated = -(xtrSum / total) * dilution;
          rm.SigmaExtrapolated = Math.Sqrt(xtrVar) / total * dilution;
        }
      }
      else
      {
        rm.QAvg = rm.QNegamax = rm.QSoft = -e.Q;
        rm.SigmaAvg = rm.SigmaNegamax = rm.SigmaSoft = SIGMA_BELOWCUT;
      }

      result.RootMoves.Add(rm);
    }

    result.ElapsedSecs = (DateTime.Now - startTime).TotalSeconds;
    UpdateStatsAndMaybeDump(result);
    return result;
  }


  #region Blended-Q comparison (analysis only)

  /// <summary>
  /// Evaluates the purely informational blended-Q comparison against a baseline best-move
  /// decision: among the top-level moves within MOVE_BLEND_CLOSE_Q_WINDOW of the best move Q
  /// (at least two required), each move with more than MOVE_BLEND_MIN_DISTINCT_PATHS distinct
  /// rollout lines competes using
  ///   blended Q = MOVE_BLEND_WEIGHT_Q * (move Q) + MOVE_BLEND_WEIGHT_DRQ * DRQ
  /// (moves without sufficient lines compete with their plain move Q). A hypothetical switch
  /// away from the baseline is reported only if the winner has sufficient rollout lines and
  /// satisfies the standard chooser's visit-support floor, judged entirely on PRE-ROLLOUT
  /// snapshot values (QOrig/EdgeNOriginal) so the revaluation's own rollouts cannot qualify
  /// their own candidate.
  ///
  /// ANALYSIS ONLY: the result is reported by revalue-root and the auto-analysis;
  /// move choice during play is never affected.
  /// </summary>
  /// <param name="manager"></param>
  /// <param name="node">The search root node the baseline decision was made for.</param>
  /// <param name="baseline">The baseline best move decision.</param>
  /// <param name="reval">The revaluation result to judge against (provides QOrig/DRQ per move).</param>
  public static RevaluationSwitchDecision CalcBlendedQSwitchDecision(MCGSManager manager, GNode node,
                                                                     BestMoveInfoMCGS baseline,
                                                                     PrincipalRevaluationResult reval)
  {
    RevaluationSwitchDecision decision = new()
    {
      BaselineMove = baseline.BestMove,
      Coverage = reval.Coverage
    };

    if (baseline.Reason != BestMoveInfoMCGS.BestMoveReason.SearchResult
     || baseline.BestMoveEdge == default
     || baseline.BestMoveEdge.ChildNodeIndex.IsNull)
    {
      decision.Outcome = RevaluationSwitchOutcome.KeepNotApplicable;
      return decision;
    }

    int baselineSlot = node.IndexOfChildInChildEdges(baseline.BestMoveEdge.ChildNodeIndex);

    // The close set: allowed top-level moves within the window of the best move Q.
    double bestQ = double.NegativeInfinity;
    foreach (RootMoveReval rm in reval.RootMoves)
    {
      if (manager.TerminationManager.MoveAtIndexAllowed(rm.ChildSlot) && rm.QOrig > bestQ)
      {
        bestQ = rm.QOrig;
      }
    }

    List<RootMoveReval> closeMoves = new();
    foreach (RootMoveReval rm in reval.RootMoves)
    {
      if (manager.TerminationManager.MoveAtIndexAllowed(rm.ChildSlot)
       && rm.QOrig >= bestQ - MOVE_BLEND_CLOSE_Q_WINDOW)
      {
        closeMoves.Add(rm);
      }
    }

    if (closeMoves.Count < 2)
    {
      decision.Outcome = RevaluationSwitchOutcome.KeepNoCloseAlternatives;
      return decision;
    }

    static double EffectiveQ(RootMoveReval rm)
      => rm.DistinctPaths > MOVE_BLEND_MIN_DISTINCT_PATHS && !double.IsNaN(rm.DRQ)
           ? MOVE_BLEND_WEIGHT_Q * rm.QOrig + MOVE_BLEND_WEIGHT_DRQ * rm.DRQ
           : rm.QOrig;

    RootMoveReval winner = null, runnerUp = null;
    foreach (RootMoveReval rm in closeMoves)
    {
      if (winner == null || EffectiveQ(rm) > EffectiveQ(winner))
      {
        runnerUp = winner;
        winner = rm;
      }
      else if (runnerUp == null || EffectiveQ(rm) > EffectiveQ(runnerUp))
      {
        runnerUp = rm;
      }
    }

    // Baseline's effective value (the baseline may itself lie outside the close set;
    // fall back to looking it up among all root moves).
    RootMoveReval baselineRM = null;
    foreach (RootMoveReval rm in reval.RootMoves)
    {
      if (rm.ChildSlot == baselineSlot)
      {
        baselineRM = rm;
        break;
      }
    }
    decision.BaselineQBlend = baselineRM != null ? EffectiveQ(baselineRM) : baseline.QOfBest;

    if (winner.ChildSlot == baselineSlot)
    {
      decision.CandidateMove = runnerUp.Move;
      decision.CandidateSlot = runnerUp.ChildSlot;
      decision.CandidateQBlend = EffectiveQ(runnerUp);
      decision.CandidateDistinctPaths = runnerUp.DistinctPaths;
      decision.Outcome = RevaluationSwitchOutcome.KeepBlendedQBest;
      return decision;
    }

    decision.CandidateMove = winner.Move;
    decision.CandidateSlot = winner.ChildSlot;
    decision.CandidateQBlend = EffectiveQ(winner);
    decision.CandidateDistinctPaths = winner.DistinctPaths;

    // A non-baseline winner must carry actual rollout evidence.
    if (winner.DistinctPaths <= MOVE_BLEND_MIN_DISTINCT_PATHS)
    {
      decision.Outcome = RevaluationSwitchOutcome.KeepBlendWinnerUnsupported;
      return decision;
    }

    if (decision.CandidateQBlend - decision.BaselineQBlend < MOVE_BLEND_SWITCH_MARGIN)
    {
      decision.Outcome = RevaluationSwitchOutcome.KeepBlendedQBest;
      return decision;
    }

    // Respect the same visit-support qualification the standard move choice applies to a
    // top-Q candidate (MinFractionNToUseQ relative to the globally most-visited root child),
    // judged on pre-rollout snapshots.
    RootMoveReval bestN0Move = null;
    foreach (RootMoveReval rm in reval.RootMoves)
    {
      if (bestN0Move == null || rm.EdgeNOriginal > bestN0Move.EdgeNOriginal)
      {
        bestN0Move = rm;
      }
    }

    float qDiffFromBestN = MathF.Abs((float)(winner.QOrig - bestN0Move.QOrig));
    float minFracN = ManagerChooseBestMoveMCGS.MinFractionNToUseQ(
      ManagerChooseBestMoveMCGS.EffectiveBestMoveMode(manager.ParamsSearch, node.N), qDiffFromBestN);
    if (winner.EdgeNOriginal <= (int)(bestN0Move.EdgeNOriginal * minFracN))
    {
      decision.Outcome = RevaluationSwitchOutcome.KeepInsufficientN;
      return decision;
    }

    decision.Outcome = RevaluationSwitchOutcome.SwitchByBlendedQ;
    return decision;
  }

  #endregion

  #region Statistics

  /// <summary>Total revaluation passes run (process lifetime).</summary>
  public static long NumRuns;

  static readonly object statsLock = new();
  static double statSumAbsFirstOrderDelta;
  static double statSumElapsedSecs;
  static long statNumAborted;
  static long lastStatsDumpTicks;

  /// <summary>Interval between periodic statistics lines written to the console.</summary>
  const long STATS_DUMP_INTERVAL_MS = 180_000;

  /// <summary>
  /// Accumulates window statistics for the completed pass and periodically (about every
  /// 3 minutes, only when the feature is actually being exercised) writes a one-line
  /// summary to the console.
  /// </summary>
  static void UpdateStatsAndMaybeDump(PrincipalRevaluationResult r)
  {
    lock (statsLock)
    {
      NumRuns++;
      statSumAbsFirstOrderDelta += Math.Abs(r.FirstOrderDeltaQ);
      statSumElapsedSecs += r.ElapsedSecs;
      if (r.Aborted)
      {
        statNumAborted++;
      }

      long now = System.Environment.TickCount64;
      if (lastStatsDumpTicks == 0)
      {
        lastStatsDumpTicks = now;
      }
      else if (now - lastStatsDumpTicks >= STATS_DUMP_INTERVAL_MS)
      {
        lastStatsDumpTicks = now;
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
          $"[REVAL] runs={NumRuns}"
          + $" avg|dQ1|={statSumAbsFirstOrderDelta / NumRuns:F3}"
          + $" avgSec={statSumElapsedSecs / NumRuns:F2} aborts={statNumAborted}");
      }
    }
  }

  #endregion


  #region Region collection

  enum NodeKind : byte { Interior, Frontier, ExactLeaf }

  sealed class Region
  {
    public readonly Dictionary<int, NodeKind> Kind = new();
    public readonly Dictionary<int, double> SnapshotQ = new();
    public readonly List<GNode> Frontier = new();
    public readonly List<GNode> ExactLeaves = new();
  }


  /// <summary>
  /// Collects the above-cut region: all nodes reachable from the root through children whose
  /// node N is at least nCut, partitioned into interior nodes (some such child exists),
  /// frontier nodes (none does, but the node is rollable) and exact leaves (terminal /
  /// unexpanded / unevaluated). Deduplication is by node index (correct in both transposition
  /// modes); the visited set also guarantees termination in the presence of graph cycles.
  /// </summary>
  static Region CollectRegion(GNode root, int nCut)
  {
    Region region = new();
    Stack<GNode> stack = new();
    stack.Push(root);
    region.Kind[root.Index.Index] = NodeKind.Interior; // provisional; refined below

    while (stack.Count > 0)
    {
      GNode node = stack.Pop();
      int idx = node.Index.Index;

      bool isExactLeaf = node.Terminal.IsTerminal() || !node.IsEvaluated || node.NumEdgesExpanded == 0;
      if (isExactLeaf && !node.IsSearchRoot)
      {
        region.Kind[idx] = NodeKind.ExactLeaf;
        region.ExactLeaves.Add(node);
        continue;
      }

      bool anyAboveCut = false;
      for (int slot = 0; slot < node.NumEdgesExpanded; slot++)
      {
        GEdge e = node.ChildEdgeAtIndex(slot);
        if (e.Type != GEdgeStruct.EdgeType.ChildEdge || e.ChildNodeIndex.IsNull)
        {
          continue;
        }

        GNode child = e.ChildNode;
        if (child.N >= nCut)
        {
          anyAboveCut = true;
          int childIdx = child.Index.Index;
          if (!region.Kind.ContainsKey(childIdx))
          {
            region.Kind[childIdx] = NodeKind.Interior; // provisional
            stack.Push(child);
          }
        }
      }

      if (!anyAboveCut && !node.IsSearchRoot)
      {
        region.Kind[idx] = NodeKind.Frontier;
        region.Frontier.Add(node);
      }
      else
      {
        region.Kind[idx] = NodeKind.Interior;
      }
    }

    return region;
  }

  #endregion


  #region Frontier estimator

  /// <summary>
  /// Builds the blended estimate for one frontier node from its ladder stage statistics
  /// and post-rollout anchor Q, including classification (directional classes are
  /// finalized later once the influence sign is known).
  /// </summary>
  static FrontierEval BuildFrontierEval(GNode node, double q0, List<FrontierStageStats> stages,
                                        bool trackVolatility)
  {
    double q1 = node.Q;

    FrontierEval eval = new()
    {
      Node = node.Index,
      N = node.N,
      Q0 = q0,
      Q1 = q1,
      Stages = stages ?? new List<FrontierStageStats>()
    };

    // Anchor uncertainty: base distrust of frontier averages plus NN self-reported uncertainty.
    float uv = (float)node.UncertaintyValue;
    double u = FP16.IsNaN(node.UncertaintyValue) ? UV_DEFAULT : uv;
    double sigmaQ = SIGMA_Q_BASE + SIGMA_Q_UV_SCALE * u;
    if (trackVolatility)
    {
      sigmaQ = Math.Max(sigmaQ, node.LeafValueVolatilityDebiased);
    }

    // Raw rollout-count-weighted mean over all stages (pure evidence, no stage weighting).
    double rawSum = 0;
    int rawCount = 0;
    foreach (FrontierStageStats s in eval.Stages)
    {
      if (s.NumRollouts > 0 && !double.IsNaN(s.Mean))
      {
        rawSum += s.Mean * s.NumRollouts;
        rawCount += s.NumRollouts;
      }
    }
    eval.RolloutMeanRaw = rawCount > 0 ? rawSum / rawCount : double.NaN;

    // Usable stages, ordered by ascending epsilon (greedy first) for weight assignment.
    List<FrontierStageStats> used = eval.Stages
      .Where(s => s.NumRollouts >= MIN_STAGE_ROLLOUTS && !double.IsNaN(s.Mean))
      .OrderBy(s => s.Epsilon)
      .ToList();

    if (used.Count == 0)
    {
      eval.VHat = q1;
      eval.Sigma = sigmaQ;
      eval.VDeep = double.NaN;
      eval.Class = FrontierClass.Confirming;
      return eval;
    }

    eval.HasRolloutStats = true;

    FrontierStageStats greedy = used[0];
    eval.ForcedDeeperRead = greedy.Epsilon == 0 && greedy.NumDistinct == 1
                         && greedy.MedianDepth >= FORCED_MIN_DEPTH;
    eval.TerminalResolved = greedy.Epsilon == 0
                         && (greedy.TerminalWin + greedy.TerminalDraw + greedy.TerminalLoss) > 0;

    double[] lambdaPattern = eval.ForcedDeeperRead ? LAMBDA_FORCED : LAMBDA_DEFAULT;
    double[] lambda = new double[used.Count];
    double lambdaSum = 0;
    for (int i = 0; i < used.Count; i++)
    {
      // Extend the canonical pattern by halving if the ladder has extra stages.
      lambda[i] = i < lambdaPattern.Length ? lambdaPattern[i] : lambdaPattern[^1] * Math.Pow(0.5, i - lambdaPattern.Length + 1);
      lambdaSum += lambda[i];
    }

    double vDeep = 0, sWithin2 = 0;
    int kTot = 0;
    int pooledDrawish = 0, pooledTotal = 0;
    for (int i = 0; i < used.Count; i++)
    {
      double w = lambda[i] / lambdaSum;
      vDeep += w * used[i].Mean;
      double sd = double.IsNaN(used[i].StdDistinct) ? 0 : used[i].StdDistinct;
      sWithin2 += w * sd * sd;
      kTot += used[i].NumDistinct;

      foreach (double q in used[i].DistinctLeafQ)
      {
        pooledTotal++;
        if (Math.Abs(q) < DRAW_DELTA)
        {
          pooledDrawish++;
        }
      }
    }

    double sBetween = 0;
    if (used.Count > 1)
    {
      double meanOfMeans = used.Average(s => s.Mean);
      sBetween = Math.Sqrt(used.Sum(s => (s.Mean - meanOfMeans) * (s.Mean - meanOfMeans)) / used.Count);
    }

    double sigmaDeep2 = sWithin2 / Math.Max(1, kTot) + sBetween * sBetween + SIGMA_DEEP_FLOOR * SIGMA_DEEP_FLOOR;
    eval.SigmaDeep = Math.Sqrt(sigmaDeep2);
    double sigmaQ2 = sigmaQ * sigmaQ;

    double beta = sigmaQ2 / (sigmaQ2 + sigmaDeep2);
    double vHat = (1 - beta) * q1 + beta * vDeep;
    double sigma = Math.Sqrt(sigmaQ2 * sigmaDeep2 / (sigmaQ2 + sigmaDeep2)) + SIGMA_MODEL;

    // Draw-escape floor: if this position looked bad yet greedy play ended in a draw, the
    // worse side itself steered into the draw, bounding its true value near zero from below.
    if (q1 < -DRAW_ESCAPE_Q && greedy.Epsilon == 0 && greedy.TerminalDraw > 0)
    {
      vHat = Math.Max(vHat, -DRAW_ESCAPE_MARGIN);
      sigma = Math.Min(sigma, DRAW_ESCAPE_SIGMA_CAP);
    }

    eval.VDeep = vDeep;
    eval.VHat = Math.Clamp(vHat, -1, 1);
    eval.Sigma = sigma;
    eval.SPooled = Math.Sqrt(sWithin2);
    eval.FracDraw = pooledTotal > 0 ? (double)pooledDrawish / pooledTotal : 0;

    FrontierStageStats highEps = used[^1];
    eval.TacticallyUnstable = used.Count > 1
                           && Math.Abs(greedy.Mean - highEps.Mean) > TACTICAL_EPS_GAP;

    // Non-directional classification (directional refinement happens once influence sign is known).
    if (Math.Abs(vDeep) <= DRAW_ABS_Q && eval.SPooled <= DRAW_STD
     && eval.FracDraw >= DRAW_FRAC && Math.Abs(q0) >= DRAW_MIN_PRIOR)
    {
      eval.Class = FrontierClass.DeepDraw;
    }
    else if (eval.SPooled > VOLATILE_STD)
    {
      eval.Class = FrontierClass.Volatile;
    }
    else
    {
      eval.Class = FrontierClass.Confirming; // possibly overridden by FinalizeDirectionalClass
    }

    return eval;
  }


  /// <summary>
  /// Upgrades a Confirming classification to SearchOptimistic/SearchPessimistic when the
  /// revalued estimate departs from the pre-rollout belief by more than the tolerance,
  /// using the influence sign to express the delta in the root player's perspective.
  /// </summary>
  static void FinalizeDirectionalClass(FrontierEval eval, int influenceSign)
  {
    if (eval.Class != FrontierClass.Confirming || !eval.HasRolloutStats)
    {
      return;
    }

    double delta = eval.VHat - eval.Q0;
    if (Math.Abs(delta) <= CONFIRM_DELTA)
    {
      return;
    }

    double rootSignedDelta = influenceSign != 0 ? influenceSign * delta : delta;
    eval.Class = rootSignedDelta < 0 ? FrontierClass.SearchOptimistic : FrontierClass.SearchPessimistic;
  }

  #endregion


  #region Re-backup

  /// <summary>Per-node values under the three operators (A=avg, B=negamax, C=soft) and their sigmas.</summary>
  struct OpVal
  {
    public double A, B, C;
    public double SA, SB, SC;
  }

  sealed class Rebackup
  {
    public readonly Dictionary<int, OpVal> Memo = new();
    public readonly HashSet<int> OnStack = new();
    public readonly List<(int Parent, int Child, int EdgeN, int EdgeND)> Skeleton = new();
    public readonly Dictionary<int, double> NLoc = new();
    public readonly List<int> CompletionOrder = new();
    public double SoftmaxP;
    public int BudgetUsed;
  }


  /// <summary>
  /// Re-propagates frontier values to the root over the above-cut region under all three
  /// operators simultaneously, into side buffers (the graph is never written).
  /// When useVHat is false, frontier nodes contribute their post-rollout graph Q instead of
  /// the blended estimate (the baseline pass for the linearity diagnostic).
  /// </summary>
  static Rebackup RunRebackup(GNode root, Region region, Dictionary<int, FrontierEval> evals,
                              double softmaxP, bool useVHat)
  {
    Rebackup rb = new() { SoftmaxP = softmaxP };
    Reval(root, region, evals, rb, useVHat);
    return rb;
  }


  static OpVal Reval(GNode node, Region region, Dictionary<int, FrontierEval> evals,
                     Rebackup rb, bool useVHat)
  {
    int idx = node.Index.Index;
    if (rb.Memo.TryGetValue(idx, out OpVal cached))
    {
      return cached;
    }

    NodeKind kind = region.Kind[idx];

    if (kind != NodeKind.Interior)
    {
      FrontierEval eval = evals[idx];
      double v = useVHat || eval.IsExactLeaf ? eval.VHat : eval.Q1;
      if (double.IsNaN(v))
      {
        v = 0;
      }
      double sigma = useVHat ? eval.Sigma : 0;
      OpVal leafVal = new() { A = v, B = v, C = v, SA = sigma, SB = sigma, SC = sigma };
      rb.Memo[idx] = leafVal;
      rb.CompletionOrder.Add(idx);
      return leafVal;
    }

    if (++rb.BudgetUsed > MAX_REBACKUP_NODES)
    {
      double q = node.Q;
      OpVal capped = new() { A = q, B = q, C = q, SA = SIGMA_BELOWCUT, SB = SIGMA_BELOWCUT, SC = SIGMA_BELOWCUT };
      rb.Memo[idx] = capped;
      rb.CompletionOrder.Add(idx);
      return capped;
    }

    rb.OnStack.Add(idx);

    int numExpanded = node.NumEdgesExpanded;
    List<(double QA, double QB, double QC, double SA, double SB, double SC, double W)> children = new(numExpanded);

    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge e = node.ChildEdgeAtIndex(slot);
      if (e.N == 0)
      {
        continue;
      }

      double w = e.N;
      double qa, qb, qc, sa, sb, sc;

      if (e.Type != GEdgeStruct.EdgeType.ChildEdge || e.ChildNodeIndex.IsNull)
      {
        // Terminal edge: carries its own exact Q (child perspective), no recursion possible.
        qa = qb = qc = -e.Q;
        sa = sb = sc = 0;
      }
      else
      {
        int childIdx = e.ChildNodeIndex.Index;
        bool inRegion = region.Kind.ContainsKey(childIdx);

        if (inRegion && !rb.OnStack.Contains(childIdx))
        {
          OpVal cv = Reval(e.ChildNode, region, evals, rb, useVHat);
          double dilution = (double)(e.N - e.NDrawByRepetition) / e.N;
          qa = -cv.A * dilution;
          qb = -cv.B * dilution;
          qc = -cv.C * dilution;
          sa = cv.SA * dilution;
          sb = cv.SB * dilution;
          sc = cv.SC * dilution;
          rb.Skeleton.Add((idx, childIdx, e.N, e.NDrawByRepetition));
        }
        else
        {
          // Below-cut child, or a cycle back to an ancestor: consume at the edge's own
          // (dilution-adjusted) cached Q, exactly as the engine's gather does.
          qa = qb = qc = -e.Q;
          sa = sb = sc = SIGMA_BELOWCUT;
        }
      }

      if (double.IsNaN(qa))
      {
        continue; // defensively skip edges with undefined values
      }

      children.Add((qa, qb, qc, sa, sb, sc, w));
    }

    rb.OnStack.Remove(idx);

    OpVal opVal = ComputeOperators(node, children, rb.SoftmaxP);
    rb.NLoc[idx] = 1 + children.Sum(c => c.W);
    rb.Memo[idx] = opVal;
    rb.CompletionOrder.Add(idx);
    return opVal;
  }


  /// <summary>
  /// Computes the three backup operators and their uncertainty estimates from the
  /// gathered child contributions (already negated to this node's perspective).
  /// </summary>
  static OpVal ComputeOperators(GNode node,
                                List<(double QA, double QB, double QC, double SA, double SB, double SC, double W)> children,
                                double softmaxP)
  {
    if (children.Count == 0)
    {
      double q = node.Q;
      return new OpVal { A = q, B = q, C = q, SA = 0, SB = 0, SC = 0 };
    }

    double v = node.V;
    if (float.IsNaN(node.V))
    {
      v = 0;
    }

    double sumW = children.Sum(c => c.W);
    double nLoc = 1 + sumW;

    // (a) Visit-weighted average (the engine-consistent operator).
    double rA = (v + children.Sum(c => c.QA * c.W)) / nLoc;
    double varA = children.Sum(c => Math.Pow(c.W / nLoc, 2) * c.SA * c.SA);

    // Selection-noise term shared by the max-type operators.
    double maxQB = children.Max(c => c.QB);
    int numNearMax = children.Count(c => c.QB >= maxQB - SEL_BAND);
    double sigmaSel = SIGMA_SELECT * Math.Sqrt(Math.Log(1 + numNearMax));

    // (b) Negamax over visit-supported children.
    double maxW = children.Max(c => c.W);
    double thetaN = Math.Max(NEGAMAX_MIN_N_ABS, NEGAMAX_MIN_N_FRAC * maxW);
    double rB = double.NegativeInfinity;
    double sigmaArgmax = 0;
    for (int i = 0; i < children.Count; i++)
    {
      if (children[i].W >= thetaN && children[i].QB > rB)
      {
        rB = children[i].QB;
        sigmaArgmax = children[i].SB;
      }
    }
    double varB;
    if (double.IsNegativeInfinity(rB))
    {
      rB = rA;
      varB = varA;
    }
    else
    {
      varB = sigmaArgmax * sigmaArgmax + sigmaSel * sigmaSel;
    }

    // (c) Soft-minimax: visit-weighted power mean over shifted child values.
    double p = softmaxP;
    double sumWX = 0;
    double[] xPows = new double[children.Count];
    for (int i = 0; i < children.Count; i++)
    {
      double x = Math.Max(1e-6, children[i].QC + SOFTMAX_SHIFT);
      xPows[i] = children[i].W * Math.Pow(x, p);
      sumWX += xPows[i];
    }
    double rC = Math.Pow(sumWX / sumW, 1.0 / p) - SOFTMAX_SHIFT;
    double varC = sigmaSel * sigmaSel;
    for (int i = 0; i < children.Count; i++)
    {
      double omega = xPows[i] / sumWX;
      varC += omega * omega * children[i].SC * children[i].SC;
    }

    return new OpVal
    {
      A = rA,
      B = rB,
      C = rC,
      SA = Math.Sqrt(varA),
      SB = Math.Sqrt(varB),
      SC = Math.Sqrt(varC)
    };
  }


  #endregion


  #region Influence

  /// <summary>
  /// Propagates signed influence weights from the seed node down the accepted-edge skeleton
  /// (acyclic by construction). The returned weight of node n is the partial derivative of the
  /// seed's averaged value with respect to n's own-perspective value: each edge step multiplies
  /// by -(edgeN - edgeND) / NLoc(parent), so the sign encodes perspective parity and the
  /// magnitude the visit-mass fraction.
  /// </summary>
  static Dictionary<int, double> PropagateInfluence(Rebackup rb, int seedIdx)
  {
    Dictionary<int, double> influence = new() { [seedIdx] = 1.0 };

    // Group skeleton edges by parent for the forward sweep.
    Dictionary<int, List<(int Child, int EdgeN, int EdgeND)>> byParent = new();
    foreach ((int parent, int child, int eN, int eND) in rb.Skeleton)
    {
      if (!byParent.TryGetValue(parent, out List<(int, int, int)> list))
      {
        byParent[parent] = list = new List<(int, int, int)>();
      }
      list.Add((child, eN, eND));
    }

    // Reverse completion order is a topological order (parents before children) of the skeleton.
    for (int i = rb.CompletionOrder.Count - 1; i >= 0; i--)
    {
      int idx = rb.CompletionOrder[i];
      if (!influence.TryGetValue(idx, out double s) || s == 0
       || !byParent.TryGetValue(idx, out List<(int Child, int EdgeN, int EdgeND)> outEdges)
       || !rb.NLoc.TryGetValue(idx, out double nLoc))
      {
        continue;
      }

      foreach ((int child, int eN, int eND) in outEdges)
      {
        double factor = -((double)(eN - eND)) / nLoc;
        influence.TryGetValue(child, out double sc);
        influence[child] = sc + s * factor;
      }
    }

    return influence;
  }

  #endregion
}
