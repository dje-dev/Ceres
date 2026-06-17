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
using Ceres.Base.Math;
using Ceres.Chess.UserSettings;
using Ceres.MCGS.Search.RPO;

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Parameters related to leaf selection (CPUCT algorithm).
/// </summary>
[Serializable]
public record ParamsSelect
{
  /// <summary>
  ///  If the improved tuned ZZTune parameters should be used.
  /// </summary>
  public const bool USE_ZZTUNE = true;


  /// <summary>
  /// Shared default for the "base" constant in the Lc0-style log growth term
  /// log((N + base + 1) / base).  Used by the standard PUCT exploration formula
  /// (CPUCTBase / CPUCTBaseAtRoot) and by the CB-GPUCT coefficient log term
  /// (CBGPUCT_SelectLambdaCLogBase).
  /// </summary>
  public const float DEFAULT_LOG_GROWTH_BASE = USE_ZZTUNE ? 38739.0f : 45669.0f;

  /// <summary>
  /// Shared default for the multiplicative "factor" on the Lc0-style log growth term.
  /// Used by standard PUCT (CPUCTFactor / CPUCTFactorAtRoot) and by the CB-GPUCT
  /// coefficient log term (CBGPUCT_SelectLambdaCLogFactor).  Making CB-GPUCT use the
  /// same constants makes its exploration scaling parallel to standard PUCT's, so the
  /// two boosts compose (or cancel) in known ways.
  /// </summary>
  public const float DEFAULT_LOG_GROWTH_FACTOR = USE_ZZTUNE ? 3.894f : 3.973f;

  public enum FPUType
  {
    /// <summary>
    /// Use a fixed constant value.
    /// </summary>
    Absolute,

    /// <summary>
    /// Subtract a penalty from the parent's Q-value.
    /// </summary>
    Reduction,

    /// <summary>
    /// Use the parent's Q-value directly (no adjustment).
    /// </summary>
    Same,

    /// <summary>
    /// Use per-move value from the neural network action head.
    /// Falls back to Reduction when action data is unavailable.
    /// </summary>
    ActionHead,

    /// <summary>
    /// Generalized policy-imputed FPU that routes through RegularizedPolicyOptimum.Solve.
    ///	Uses policy-based imputation to estimate Q values.
    /// Per-child imputed Q via Boltzmann calibration over the policy, with an anchor (such as parent V).
    /// The choice of KL divergence is controlled by ParamsSelect.RPOFPURegularization
    /// (default ForwardKLSoftmax, identical to PolicyImputed; flip to ReverseKL for
    /// FPU values that are mathematically consistent with CB-GPUCT selection).
    /// </summary>
    PolicyImputedRPO,
  };


  /// <summary>
  /// Minimum policy value for moves (as a fraction) which forces policy probabilities to be at least this level.
  /// </summary>
  public const float MinPolicyProbability = 0.004f;

  /// <summary>
  /// Temperature (lambda) of the policy->Q imputation in RegularizedPolicyOptimum.Solve
  /// (fills Q for unvisited children = FPU, and the CB-GPUCT backup shrinkage prior).
  /// Default forward-KL form:  q(a) = anchorQ + tau*(log mu_a - E_mu[log mu]).  Higher tau
  /// spreads imputed Q further by policy (top-policy moves look better, tail moves worse),
  /// sharpening the search toward the policy rather than broadening it; tau -> 0 gives a
  /// flat FPU equal to anchorQ (parent Q).  Used when FPUMode is PolicyImputed or PolicyImputedRPO.
  /// </summary>
  public float PolicyImputationTau = 0.15f; // values less than 0.15 test as significantly inferior


  /// <summary>
  /// Choice of KL divergence used by the CB-GPUCT selection rule when computing pi_bar
  /// via RegularizedPolicyOptimum.Solve.  Default ReverseKL matches Grill et al. and
  /// the historical CBGPUCTScoreCalc behavior.
  /// </summary>
  public RPORegularization RPOSelectRegularization = RPORegularization.ReverseKL;

  /// <summary>
  /// Choice of KL divergence used by the CB-GPUCT V_bar backup when computing the
  /// regularized state value via RegularizedPolicyOptimum.Solve.  Default ReverseKL
  /// matches the historical CBGPUCTScoreCalc.ComputeVBar behavior.
  /// </summary>
  public RPORegularization RPOBackupRegularization = RPORegularization.ReverseKL;

  /// <summary>
  /// Choice of KL divergence used by the per-child FPU imputation (FPUMode = PolicyImputed
  /// or PolicyImputedRPO).  Default ForwardKLSoftmax preserves the historical Boltzmann
  /// behavior exactly.  Setting this to ReverseKL makes the FPU mathematically consistent
  /// with the CB-GPUCT selection rule (at the cost of needing to retune PolicyImputationTau).
  /// </summary>
  public RPORegularization RPOFPURegularization = RPORegularization.ForwardKLSoftmax;

  /// <summary>
  /// Amount by which unvisited child value scores are shifted in RPO mode
  /// from the value computed by RPO imputation.
  /// </summary>
  public float RPOFPUValue = -0.15f;

  /// <summary>
  /// Apportionment algorithm used to translate pi_bar into a concrete visit allocation
  /// in CB-GPUCT.  Default IterativeLargestDeficit matches the historical CBGPUCT inner
  /// loop exactly; HamiltonClosedForm is behaviorally equivalent up to tie-breaking and
  /// is faster for large batches.
  /// </summary>
  public RPOAllocationAlgorithm RPOSelectAllocator = RPOAllocationAlgorithm.IterativeLargestDeficit;

  /// <summary>
  /// Choice of base anchor VALUE used to calibrate per-child imputed Q for unvisited
  /// children.  The selected scalar is passed as the MatchValue/MatchChild anchor.Value
  /// to RegularizedPolicyOptimum.Solve, fixing the absolute level of the imputed q
  /// vector (relative spread among children comes from lambda * log mu).
  ///
  /// IMPORTANT: This enum controls only the anchor VALUE, not the anchor MODE.  The
  /// FPU call site additionally picks MatchChild(0) vs MatchValue mode based on
  /// whether child 0 is visited (a legacy convention - see PUCTSelector docs for
  /// the "legacy quirk" note).  Mode selection is orthogonal and unchanged here.
  ///
  /// Options:
  ///   ParentV                  - node.NodeRef.V (raw NN value head; unaffected by search).
  ///   ParentQ                  - node.Q (search-improved Q; under CBGPUCT this is the
  ///                              previous V_bar, selfV-blended and possibly stale).
  ///                              This reproduces the legacy FPU anchor value exactly.
  ///   FirstChildElseParentQ    - observed Q of child 0 (parent perspective) if visited,
  ///                              else node.Q.  Useful when child 0 is reliably the
  ///                              highest-P move and is visited first; this is what the
  ///                              MatchChild mode "should" do semantically (whereas
  ///                              legacy keeps the value at node.Q regardless).
  ///   BestChildElseParentQ     - max over visited children of -edge.Q (parent perspective),
  ///                              else node.Q.  Anchors on the strongest observation;
  ///                              corresponds to the "earlier dead code computing
  ///                              bestIndex" referenced in PUCTSelector docs.
  ///   BlendBestChildParentQ    - 0.5 * (best visited child Q + node.Q), else node.Q.
  ///                              Compromise between BestChild and ParentQ.
  /// </summary>
  public enum CBGPUCTQAnchorType
  {
    ParentV,
    ParentQ,
    FirstChildElseParentQ,
    BestChildElseParentQ,
    BlendBestChildParentQ,
  }

  /// <summary>
  /// Anchor VALUE used to calibrate per-child imputed Q in the FPU imputation
  /// (ApplyRPOImputedFPU during selection).  Default ParentQ exactly reproduces
  /// legacy behavior: anchor.Value = node.Q in both legacy branches (MatchChild
  /// when child 0 visited, MatchValue otherwise - the mode switch is unrelated to
  /// the value and remains driven by child 0's visit status regardless of this enum).
  /// </summary>
  public CBGPUCTQAnchorType CBGPUCT_QAnchorTypeFPU = CBGPUCTQAnchorType.ParentQ;


  /// <summary>
  /// When action head enabled, if the univisited children should be reordered upon first visit
  /// using not just policy but instead the PUCT scores inclusive of 
  /// the effect of both policy and value (i.e. using the action head values as fill-in).
  /// </summary>
  public bool ActionResortUnvisitedChildren = true;

  public float UCTRootNumeratorExponent = 0.5f;
  public float UCTNonRootNumeratorExponent = 0.5f;

  /// <summary>
  /// Exponent to be used in the denominator (for the root node).
  /// 1.0f for classical UCT, alternately for example 0.5 according to the method of Shah, Xie, Xu.
  /// </summary>
  public float UCTRootDenominatorExponent = 1.0f;

  /// <summary>
  /// Exponent to be used in the denominator (for non-root nodes).
  /// 1.0f for classical UCT, alternately for example 0.5 according to the method of Shah, Xie, Xu.
  /// </summary>
  public float UCTNonRootDenominatorExponent = 1.0f;

  public float RootCPUCTExtraMultiplierDivisor = 10_000f;
  public float RootCPUCTExtraMultiplierExponent = 0; // zero to disable, typical value 0.43f

  [CeresOption(Name = "cpuct", Desc = "Scaling used in node selection to encourage exploration (traditional UCT)", Default = "2.15")]
  public float CPUCT = USE_ZZTUNE ? 1.745f : 2.897f;

  [CeresOption(Name = "cpuct-base", Desc = "Constant (base) used in node selection to weight exploration", Default = "18368")]
  public float CPUCTBase = DEFAULT_LOG_GROWTH_BASE;

  [CeresOption(Name = "cpuct-factor", Desc = "Constant (factor) used in node selection to weight exploration", Default = "2.82")]
  public float CPUCTFactor = DEFAULT_LOG_GROWTH_FACTOR;


  [CeresOption(Name = "cpuct-at-root", Desc = "Scaling used in node selection (at root) to encourage exploration (traditional UCT)", Default = "3")]
  public float CPUCTAtRoot = USE_ZZTUNE ? 1.745f : 2.897f;


  [CeresOption(Name = "cpuct-base-at-root", Desc = "Constant (base) used in node selection (at root) to weight exploration", Default = "18368")]
  public float CPUCTBaseAtRoot = DEFAULT_LOG_GROWTH_BASE;


  [CeresOption(Name = "cpuct-factor-at-root", Desc = "Constant (factor) used in node selection (at root) to weight exploration", Default = "2.82")]
  public float CPUCTFactorAtRoot = DEFAULT_LOG_GROWTH_FACTOR;

  [CeresOption(Name = "policy-decay-factor", Desc = "Linear scaling factor used in node selection to shrink policy toward uniform as N grows. Zero to disable, typical value 5.0", Default = "0")]
  public float PolicyDecayFactor = 0;

  [CeresOption(Name = "policy-decay-exponent", Desc = "Exponent used in scaling factor (applied to N) used in node selection to shrink policy toward uniform as N grows. Zero to disable, typical value 0.38", Default = "0")]
  public float PolicyDecayExponent = 0.38f;

  /// <summary>
  /// Multiplicative reduction of CPUCT at low parent-N, decaying smoothly to zero as N grows.
  /// Effective CPUCT is scaled by (1 - CPUCTEarlyReductionAmount * exp(-parentN / CPUCTEarlyReductionNHalf)).
  /// Motivates more focused early search (closer to the policy prior) than vanilla PUCT.
  /// Zero to disable; typical exploratory value 0.4.
  /// </summary>
  [CeresOption(Name = "cpuct-early-reduction-amount", Desc = "Multiplicative reduction of CPUCT at low parent-N (0 to disable, typical 0.4)", Default = "0")]
  public float CPUCTEarlyReductionAmount = 0;

  /// <summary>
  /// Characteristic parent-N (in visits) over which CPUCTEarlyReductionAmount decays by 1/e.
  /// Smaller values make the reduction vanish sooner; larger values keep the reduction active longer.
  /// </summary>
  [CeresOption(Name = "cpuct-early-reduction-n-half", Desc = "Characteristic parent-N for exponential decay of CPUCT early reduction (typical 32)", Default = "32")]
  public float CPUCTEarlyReductionNHalf = 32f;

  /// <summary>
  /// Amount of relative virtual loss to apply in leaf selection to discourage collisions.
  /// Values closer to zero yield less distortion in choosing leafes (thus higher quality play)
  /// but slow down search speed because of excess collisions. 
  /// Values of -0.10 or -0.15 seem to work well; -0.03 clearly worse, -0.20 mixed/unclear.
  /// </summary>
  // Smaller values yield higher fidelity leaf selection 
  // (but possibly slightly slower due to increased collisions, especially at smaller node counts)      
  [CeresOption(Name = "vloss-relative", Desc = "Virtual loss (relative) to be applied when collisions encountered", Default = "-0.15")]
  public float VirtualLossDefaultRelative = -0.12f;

  /// <summary>
  /// Virtual loss to be used if VLossRelative is false.
  /// </summary>
  [CeresOption(Name = "vloss-absolute", Desc = "Virtual loss (absolute) to be applied when collisions encountered", Default = "-1.0")]
  public float VirtualLossDefaultAbsolute = -1.0f;

  public bool UseDynamicVLoss = false;


  #region CB-GPUCT (visit-target regularized-policy graph MCTS) parameters

  /// <summary>
  /// Selects which CB-GPUCT components are active
  /// (potentially regularized select/backup according to
  /// "Monte-Carlo tree search as regularized policy optimization" (Grill et al 2022).
  /// </summary>
  public enum CBGPUCTModeType
  {
    /// <summary>
    /// CB-GPUCT off; standard PUCT selection and standard Q-pure backup.
    /// </summary>
    None,

    /// <summary>
    /// Visit-target pi_bar selection (replaces PUCT scoring); standard backup.
    /// </summary>
    SelectOnly,

    /// <summary>
    /// Standard PUCT selection; V_bar regularized backup (overwrites node.Q).
    /// </summary>
    BackupOnly,

    /// <summary>
    /// Both: pi_bar selection AND V_bar regularized backup.
    /// </summary>
    SelectAndBackup
  }

  /// <summary>
  /// CB-GPUCT mode (see CBGPUCTModeType). Default None disables CB-GPUCT entirely.
  /// </summary>
  public CBGPUCTModeType CBGPUCT_Mode = CBGPUCTModeType.None;

  /// <summary>
  /// Convenience predicate: visit-target pi_bar selection is active.
  /// </summary>
  public bool CBGPUCTSelectActive => CBGPUCT_Mode == CBGPUCTModeType.SelectOnly
                                  || CBGPUCT_Mode == CBGPUCTModeType.SelectAndBackup;

  /// <summary>
  /// Convenience predicate: visit-target pi_bar selection is active for a parent node
  /// with the specified visit count <paramref name="n"/>.
  ///
  /// This is true only when CB-GPUCT select is enabled (see CBGPUCTSelectActive) AND the
  /// parent N has not exceeded CBGPUCT_PUCTAboveN.  Once the parent N grows strictly larger
  /// than CBGPUCT_PUCTAboveN the selection falls back to standard PUCT for that node.
  /// </summary>
  /// <param name="n">Parent node visit count N.</param>
  /// <returns>True if CB-GPUCT visit-target selection should run for this parent; false to fall back to PUCT.</returns>
  public bool CBGPUCTSelectActiveAtN(int n) => CBGPUCTSelectActive && n <= CBGPUCT_PUCTAboveN;


  /// <summary>
  /// Convenience predicate: V_bar regularized backup is active.
  /// </summary>
  public bool CBGPUCTBackupActive => CBGPUCT_Mode == CBGPUCTModeType.BackupOnly
                                  || CBGPUCT_Mode == CBGPUCTModeType.SelectAndBackup;

  #region Support-shrinkage (the sole CB-GPUCT robustness mechanism, select + backup)

  /// <summary>
  /// MASTER TOGGLE + prior strength for the unified support-shrinkage backup.
  ///
  /// When &gt; 0, ComputeVBar uses a single posterior-mean Q estimate per child
  ///   q_hat(a) = (N_a^p * q_obs(a) + K * m_a) / (N_a^p + K)     for visited children
  ///   q_hat(a) = m_a                                            for unvisited (the N_a = 0 limit)
  /// and uses it CONSISTENTLY for BOTH the pi_bar weights AND the V_bar dot product:
  ///   pi_bar = ReverseKL(mu, q_hat, lambda_N),
  ///   V_bar  = (sum_a pi_bar(a) * q_hat(a) * (N - 1) + selfV) / N.
  /// Here K = this knob (prior pseudo-visit count), p = CBGPUCT_BackupSupportShrinkageDecayExponent,
  /// N_a = child.N (statistical support; cross-parent in graph mode), and m_a is the
  /// empirical-Bayes prior: the support-weighted consensus of the children themselves,
  /// policy-shaped via the existing forward-KL imputation -
  ///   q_bar = sum_a N_a q_a / sum_a N_a,   m_a = q_bar + tau*(log mu_a - E_mu[log mu]),   tau = PolicyImputationTau,
  /// falling back to node.V when no child is visited.  N_a here (the consensus weight)
  /// defaults to child.N but is selectable via CBGPUCT_ConsensusWeight (e.g. edge.N
  /// for transposition-robustness); the shrinkage PRECISION below always uses child.N.
  ///
  /// Why one knob suffices: the bias being controlled (reverse-KL pi_bar over-weights
  /// high-q children, so V_bar is a soft-max of noisy child Q's -> winner's-curse /
  /// max-operator overestimation that compounds toward the root) is driven by the NOISE
  /// in each q_obs(a), which scales as 1/sqrt(N_a).  Shrinking each estimate toward the
  /// prior in proportion to its own support (the conjugate / James-Stein posterior mean)
  /// denoises it by exactly that amount, and using the same denoised value in the dot
  /// product bounds its contribution at the source.  A single-visit outlier (e.g. 12,553
  /// visits elsewhere, one new child returning a lucky +0.8) is shrunk hard toward the
  /// consensus so it "moves the needle a little, not a lot"; as consistent evidence
  /// accrues N_a grows, shrinkage vanishes, and the move is progressively believed
  /// (q_hat -> q_obs as N_a -> infinity).
  ///
  /// This is the SOLE backup robustness mechanism (the legacy Q-shrinkage, pi_bar caps,
  /// mean-preserve control variate, observed-only, anchor-type/anchorK blend, fixed-point
  /// iteration, and per-child lambda have all been removed in favor of it).  Still active
  /// alongside it: the lambda_N schedule (CBGPUCT_BackupLambda*), RPOBackupRegularization,
  /// and PolicyImputationTau.
  ///
  /// 0 disables shrinkage (plain Grill V_bar: q_hat = q_obs for visited children, the
  /// policy-implied prior for unvisited).  Typical setting 3-10; behavior is insensitive
  /// across that range.  Default 3.
  /// </summary>
  public float CBGPUCT_BackupSupportShrinkageK = 3.0f;

  /// <summary>
  /// Decay exponent p in the support-shrinkage posterior mean
  ///   q_hat(a) = (N_a^p * q_obs(a) + K * m_a) / (N_a^p + K).
  /// p = 1 is the principled conjugate / Bayesian rate (shrink weight K/(N_a + K),
  /// asymptotic 1/N decay).  p &gt; 1 makes the shrinkage vanish faster past N_a = 1
  /// while leaving the N_a = 1 value unchanged (since 1^p = 1) - motivated by NN-eval-
  /// based search where per-visit information can exceed the conjugate-Normal assumption.
  /// Typical range [1.0, 2.0]; default 1.0.  Only used when
  /// CBGPUCT_BackupSupportShrinkageK &gt; 0.
  /// </summary>
  public float CBGPUCT_BackupSupportShrinkageDecayExponent = 1.0f;

  /// <summary>
  /// Master toggle + strength (beta) of the BACKUP BREADTH BONUS: an additive max-entropy
  /// style credit that makes a node with multiple good moves back up a slightly higher
  /// value than an otherwise-equal node resting on a SINGLE good move.
  ///
  /// Why needed: the plain regularized value V_bar = sum_a pi_bar(a) q_hat(a) is an
  /// expected-q (a weighted average bounded by max_a q_hat).  Near-best alternatives drag
  /// that average DOWN, so V_bar actually mildly penalizes breadth; and as lambda_N -&gt; 0
  /// at high N, pi_bar collapses onto the single argmax child and V_bar -&gt; max_a q_hat,
  /// erasing any multi-move distinction entirely.  This term restores (and makes explicit)
  /// the dropped max-entropy bonus  -lambda*KL(pi_bar||mu)  from the soft-value identity
  ///   lambda*log sum_a mu_a exp(q_a/lambda) = E_pi[q] - lambda*KL(pi_bar||mu),
  /// the part the energy-only V_bar discards.
  ///
  /// HIGH-N PERSISTENCE: breadth is measured on a value-softmax at a fixed temperature
  /// CBGPUCT_BackupBreadthTemperature (NOT lambda_N), so unlike H(pi_bar) - which -&gt; 0 as
  /// lambda_N collapses - this signal survives at large N, exactly where the plain
  /// mechanism's benefit was observed to fade.
  ///
  /// ADVERSARIAL ASYMMETRY (free, no parity knob): the bonus is added in the node's OWN
  /// (side-to-move) perspective.  Through the existing negamax edge negation (-edge.Q) an
  /// ancestor reads it with alternating sign, so it rewards the mover's own optionality and
  /// penalizes the opponent's reply breadth (prophylaxis / "pose the opponent problems").
  ///
  /// SAFETY: gated by (1 - |V_bar|) so it fades to 0 at decided/terminal/won-lost nodes
  /// (proven mates already short-circuit before ComputeVBar), and hard-capped at
  /// CBGPUCT_BackupBreadthBonusMax so it can never override a genuinely winning narrow move
  /// (forced tactics / only-moves).  It is a tie-breaker among near-equal candidates, not a
  /// dominant term.
  ///
  /// 0 disables (default; no behavior change).  Suggested starting point when enabling:
  /// beta ~ 0.03-0.06 with the cap ~ 0.02-0.03.
  /// Related literature: maximum-entropy MCTS (MENTS, Xiao et al. 2019), soft Q / SAC
  /// (Haarnoja et al. 2018), MCTS-as-regularized-policy-optimization (Grill et al. 2020),
  /// convex/Tsallis regularization (Dam et al. 2021); "empowerment differential" for the
  /// adversarial framing.
  /// </summary>
  public float CBGPUCT_BackupBreadthBonusBeta = 0.0f; // 0.03 is reasonable value to enable

  /// <summary>
  /// Fixed softmax temperature tau_b used to measure breadth for the backup breadth bonus
  ///   w(a) proportional to mu(a) * exp((q_hat(a) - max_a q_hat) / tau_b),
  /// whose normalized entropy H(w)/log(#contributing) in [0,1] is the breadth fraction.
  /// Deliberately decoupled from lambda_N: this is what keeps the bonus alive at high N
  /// (a lambda_N-based measure would vanish as pi_bar sharpens).  SMALL tau_b -&gt; only
  /// children within ~tau_b of the best count as "good" (strict); LARGE tau_b -&gt; more
  /// children count (lenient).  Comparable in scale to the spread of the top children's Q
  /// (single digits of a centipawn-equivalent up to ~0.3). 
  /// Only used when CBGPUCT_BackupBreadthBonusBeta &gt; 0; tau_b &lt;= 0 disables the bonus.
  /// </summary>
  public float CBGPUCT_BackupBreadthTemperature = 0.02f;

  /// <summary>
  /// Hard cap on the magnitude of the backup breadth bonus (in Q units, [-1,1] value
  /// scale).  The bonus beta*breadthFrac*(1-|V_bar|) is clamped to [0, this] before being
  /// added to V_bar, guaranteeing it can never flip a decision dominated by a real value
  /// gap (e.g. a winning single move stays winning).  Default 0.03 (~a few centipawns).
  /// Only used when CBGPUCT_BackupBreadthBonusBeta &gt; 0.
  /// </summary>
  public float CBGPUCT_BackupBreadthBonusMax = 0.03f;

  /// <summary>
  /// Parent-N upper bound for CB-GPUCT visit-target SELECTION.  
  /// When the parent node's visit count N is strictly greater than this value, 
  /// CB-GPUCT selection is bypassed for that node and standard PUCT selection runs instead.
  /// Default int.MaxValue (never bypass on the high side).  
  /// Affects selection only; the V_bar backup (if active) is unchanged.
  /// </summary>
  public int CBGPUCT_PUCTAboveN = 50;

  /// <summary>
  /// Prior strength K for the SELECT support-shrinkage in ScoreCalc - the select-phase
  /// analogue of CBGPUCT_BackupSupportShrinkageK.  Each visited child's per-edge q (-W/N)
  /// is shrunk toward the policy-shaped consensus prior by edge.N (the per-edge q's
  /// statistical support); unvisited children keep their FPU value untouched (the FPU
  /// exploration signal is not altered).  The shrunk q then feeds the pi_bar visit-target Solve.
  /// 0 disables (plain RPO select).  
  /// Default 0 or 1 (tests suggest needs less shrinkaage than Backup uses by default).
  /// </summary>
  public float CBGPUCT_SelectSupportShrinkageK = 0;

  /// <summary>
  /// Decay exponent p for the SELECT support-shrinkage (see
  /// CBGPUCT_BackupSupportShrinkageDecayExponent for semantics; select uses edge.N as the
  /// per-child support).  Default 1.0.  Only used when CBGPUCT_SelectSupportShrinkageK &gt; 0.
  /// </summary>
  public float CBGPUCT_SelectSupportShrinkageDecayExponent = 1.0f;

  /// <summary>
  /// Weight basis for the children when forming the empirical-Bayes consensus anchor
  ///   q_bar = sum_a w_a q_a / sum_a w_a
  /// that the shrinkage prior m_a is built around.  Shared by both the select and backup
  /// support-shrinkage. Only the consensus weight is affected by this knob - the per-child
  /// shrinkage precision N_a^p / (N_a^p + K) uses child.N in backup and edge.N in select
  /// (the genuine statistical support of each phase's q - reliability, not preference,
  /// which transpositions sharpen rather than bias).
  ///
  /// The choice matters only in graph mode with transpositions, where child.N is
  /// inflated by transposition density - an exogenous graph property, not a sign that
  /// search preferred the move from this parent.  A heavily-transposed child still
  /// drives V_bar directly through pi_bar (it is barely shrunk and wins the weight on
  /// its own merits), so the consensus's only remaining job is to prime the prior for
  /// UNDER-EXPLORED local moves - for which the locally-revealed preference (edge.N) is
  /// the more relevant weight.
  /// WHY RAW child.N IS biased: under the hierarchical model q_a | theta_a ~ N(theta_a,
  /// s^2/N_a), theta_a ~ N(mu0, tau^2), the posterior-mean estimate of the anchor mu0 is
  /// q_bar = sum_a w_a q_a / sum_a w_a with w_a = 1/(tau^2 + s^2/N_a) proportional to
  /// N_a/(N_a + Kc), Kc = s^2/tau^2.  Weighting by raw N_a (the ChildN mode) is the
  /// Kc -> infinity limit, i.e. it assumes tau^2 -> 0: that every sibling move has the
  /// same true value.  For chess that is false (moves differ a lot -> tau^2 large -> Kc
  /// SMALL), so the correct weights sature: once a child is out of the high-noise
  /// region, extra N_a (e.g. from incidental transposition density) should not keep
  /// increasing its pull on the anchor.  The *Saturating modes implement that.
  ///   ChildN : weight by child.N (textbook precision pooling; the tau^2 = 0 limit).  A
  ///            BAD transposition hub - or even a benign low-policy one - can drag the
  ///            target and over-shrink a good but moderately-supported local move.
  ///            Matches the originally-validated behavior.
  ///   EdgeN  : weight by per-edge N (this parent's revealed preference; transposition-
  ///            free; reduces to ChildN exactly when there are no transpositions).
  ///   Policy : weight by the prior policy mu (i.e. anchor at E_mu[q]); fully N-free but
  ///            trusts the network prior, which is fragile when the policy is miscalibrated.
  ///   ChildNSaturating : weight by child.N / (child.N + Kc), Kc = CBGPUCT_ConsensusReliabilityK.
  ///            The principled finite-tau^2 estimator: low-N children are still suppressed 
  ///            proportional to their reliability, any sufficiently-explored child counts ~once, 
  ///            so a single huge-N transposition can no longer dominate the anchor.
  ///   PolicyChildNSaturating : weight by mu * child.N / (child.N + Kc).  As above, plus
  ///            an importance term so the anchor is set by the moves the policy actually
  ///            prioritizes - a reliable but LOW-policy drawish transposition then
  ///            contributes little even though its gate is ~1.  Most robust for backup.
  /// Default ChildN (no behavior change vs. the initial implementation; identical to
  /// EdgeN whenever there are no transpositions).  Only used when
  /// CBGPUCT_BackupSupportShrinkageK &gt; 0.  NOTE: the saturating gate is applied in the
  /// BACKUP consensus only; the SELECT consensus (whose q support is edge.N, already
  /// transposition-free) maps the *Saturating modes to their non-saturating equivalents
  /// (ChildNSaturating -> edge.N, PolicyChildNSaturating -> policy).
  /// </summary>
  public enum CBGPUCTConsensusWeightType { ChildN, EdgeN, Policy, ChildNSaturating, PolicyChildNSaturating }
  public CBGPUCTConsensusWeightType CBGPUCT_ConsensusWeight = CBGPUCTConsensusWeightType.PolicyChildNSaturating;

  /// <summary>
  /// Saturation half-point Kc for the *Saturating consensus weight modes: w_a =
  /// child.N / (child.N + Kc) (optionally times mu).  Interpretable as Kc = s^2/tau^2 =
  /// (per-visit Q noise variance) / (between-move value variance); a child reaches half
  /// its asymptotic weight at child.N = Kc.  SMALL Kc (moves differ a lot, the chess
  /// regime) flattens the weights quickly toward equal/importance weighting; LARGE Kc
  /// recovers raw precision pooling (the ChildN mode in the limit).  Single digits are
  /// expected for chess; defaults to the shrinkage K (3).  No effect unless
  /// CBGPUCT_ConsensusWeight is one of the *Saturating modes.  Kc &lt;= 0 makes the gate
  /// degenerate to 1 (uniform for ChildNSaturating, pure policy for PolicyChildNSaturating).
  /// </summary>
  public float CBGPUCT_ConsensusReliabilityK = 3.0f;

  #endregion

  /// <summary>
  /// Schedule used to compute lambda_N (the regularization strength) from sum N_a.
  /// LambdaExp is ignored when schedule is UCT or Log.
  /// Phase-specific instances (CBGPUCT_SelectLambda* and CBGPUCT_BackupLambda*) below.
  /// </summary>
  public enum CBGPUCTLambdaScheduleType
  {
    /// <summary>
    /// LambdaC * pow(sumN, LambdaExp) / (denomBase + sumN).  Grill/AlphaZero schedule:
    /// bump-shaped, peaks near sumN == denomBase, decays as 1/sqrt(N) asymptotically.
    /// Returns 0 at sumN == 0.  Selection default.
    /// </summary>
    AlphaZero,

    /// <summary>
    /// LambdaC * sqrt(log(sumN + e) / (denomBase + sumN)) UCT-style log term.
    /// Slower asymptotic decay than AlphaZero (sqrt(log N / N) vs 1/sqrt(N)),
    /// so regularization stays meaningful at higher sumN.
    /// </summary>
    UCT,

    /// <summary>
    /// LambdaC / pow(sumN, LambdaExp).  Plain inverse-power decay without a denominator
    /// constant or bump.  Monotonically decreasing in sumN (modulo log-growth on LambdaC).
    /// Returns 0 at sumN == 0 (avoids division by zero; no regularization to apply yet).
    /// Backup default.
    /// </summary>
    Pow,

    /// <summary>
    /// LambdaC / log(sumN + e).  Even slower decay than Pow (1/log N vs 1/N^exp), so
    /// regularization stays strong well into large N.  LambdaExp is unused.
    /// </summary>
    Log
  }

  /// <summary>
  /// Discrete choices for the constant offset in the lambda_N schedule denominator.
  /// Selection and backup take this independently (see CBGPUCT_SelectLambdaDenominatorBase
  /// and CBGPUCT_BackupLambdaDenominatorBase).  The numeric ladder {1, 3, 5, 10, 30} covers
  /// the practical range from "near-monotonic decay from N=1" to "Grill |A| for chess".
  /// NumMovesWithPolicyOver5Pct adapts per-node: counts children whose policy P exceeds
  /// 5%, giving an effective branching factor that ignores low-policy tail moves.
  /// </summary>
  public enum CBGPUCTLambdaDenominatorBaseType
  {
    One,
    Three,
    Five,
    Ten,
    Thirty,
    NumMovesWithPolicyOver5Pct
  }

  /// <summary>
  /// Constant offset in the SELECTION-phase lambda_N denominator:
  ///   Pow: lambda_N = c(sumN) * pow(sumN, LambdaExp) / (denomBase + sumN)
  ///   UCT: lambda_N = c(sumN) * sqrt(log(sumN + e) / (denomBase + sumN))
  /// In Grill et al. this would be |A| (the action-set cardinality).  Larger values produce
  /// stronger regularization at low visit counts (lambda peaks near denomBase), reflecting
  /// the algebraic accumulation of the per-child exploration floor across all actions.
  /// Default Ten approximates a "soft effective branching factor" without literally tying
  /// to per-node |A|.
  /// </summary>
  public CBGPUCTLambdaDenominatorBaseType CBGPUCT_SelectLambdaDenominatorBase = CBGPUCTLambdaDenominatorBaseType.Thirty;

  /// <summary>
  /// Constant offset in the BACKUP-phase lambda_N denominator.  Backup is not allocating
  /// visits and has no exploration-floor analog, so the algebraic |A| reasoning that
  /// motivates a larger denominator in selection does not apply.  Default One yields a
  /// monotonically-decreasing lambda past the brief initial rise: regularization is
  /// strongest at low total visits and decays smoothly as Q evidence accumulates - a
  /// plain Bayesian "trust the prior less as data piles up" curve.
  /// </summary>
  public CBGPUCTLambdaDenominatorBaseType CBGPUCT_BackupLambdaDenominatorBase = CBGPUCTLambdaDenominatorBaseType.One;

  /// <summary>
  /// Lambda schedule for the SELECTION phase (visit-target deficit pi_bar).
  /// </summary>
  public CBGPUCTLambdaScheduleType CBGPUCT_SelectLambdaSchedule = CBGPUCTLambdaScheduleType.AlphaZero;

  /// <summary>
  /// Multiplicative scale on lambda_N for the SELECTION phase.
  /// Larger values keep pi_bar closer to the prior P (more exploration);
  /// smaller values let pi_bar concentrate on high-Q actions.
  /// </summary>
  public float CBGPUCT_SelectLambdaC = 1f;

  /// <summary>
  /// Exponent on (sum N_a) in the Pow lambda_N schedule for the SELECTION phase.
  /// Value of 0.5 replicate Grill et al.
  /// </summary>
  public float CBGPUCT_SelectLambdaExp = 0.5f;

  /// <summary>
  /// Lc0-style log-growth coefficient for the SELECTION-phase lambda_N.  The effective
  /// coefficient that scales the base schedule grows with sumN as
  ///   c(sumN) = CBGPUCT_SelectLambdaC + CBGPUCT_SelectLambdaCLogFactor
  ///                                  * log((sumN + CBGPUCT_SelectLambdaCLogBase + 1) / CBGPUCT_SelectLambdaCLogBase)
  /// exactly paralleling Ceres-PUCT's CPUCT log growth.  The base lambda_N becomes
  ///   lambda_N(sumN) = c(sumN) * pow(sumN, LambdaExp) / (LambdaDenominatorBase + sumN)
  /// so at small sumN c(sumN) is approximately LambdaC (unchanged) and at large sumN it grows
  /// roughly as log(sumN), counteracting the 1/sqrt(N) decay of the base schedule and
  /// preserving exploration intensity at high visit counts.
  ///
  /// Defaults point to the shared DEFAULT_LOG_GROWTH_BASE / DEFAULT_LOG_GROWTH_FACTOR
  /// constants that are also used by standard CPUCT.  This means the CB-GPUCT lambda
  /// coefficient and the CPUCT exploration constant scale with sumN in parallel - you can
  /// reason about their compounded effect using the same arithmetic.
  ///
  /// Set CBGPUCT_SelectLambdaCLogFactor = 0 to disable the log growth (recovers the
  /// constant LambdaC schedule).  Only applies to selection - the BACKUP-phase lambda_N
  /// has no log term because backup computes V_bar (a value estimate), not an exploration
  /// policy.
  /// </summary>
  public float CBGPUCT_SelectLambdaCLogBase = DEFAULT_LOG_GROWTH_BASE;
  public float CBGPUCT_SelectLambdaCLogFactor = 0;// DEFAULT_LOG_GROWTH_FACTOR;

  /// <summary>
  /// Lambda schedule for the BACKUP phase (V_bar regularized value computation).
  /// </summary>
  public CBGPUCTLambdaScheduleType CBGPUCT_BackupLambdaSchedule = CBGPUCTLambdaScheduleType.AlphaZero;

  /// <summary>
  /// Multiplicative scale on lambda_N for the BACKUP phase.
  /// </summary>
  public float CBGPUCT_BackupLambdaC = 0.75f; // use 1.0 for AlphaZero schedule

  /// <summary>
  /// Exponent on (sum N_a) in the Pow lambda_N schedule for the BACKUP phase.
  /// Smaller values cause more rapid decay of lambda_N and thus weaker regularization of Q higher visit counts.
  /// </summary>
  public float CBGPUCT_BackupLambdaExp = 0.45f; // use 0.5 for AlphaZero schedule


  /// <summary>
  /// Fraction of the cross-parent (child node) visit count to fold into the
  /// per-parent visit-target deficit:
  ///   effective_N(a) = per_edge_N(a) + fraction * (child(a).N - per_edge_N(a))
  ///                  + nInFlight(a)
  ///
  /// 0.0 (default): per-edge only - each parent independently aims for its own
  ///                visit target, ignoring transposition visits via other parents.
  ///                Matches the legacy non-graph-aware behavior.
  /// 1.0          : fully count cross-parent visits - redirect exploration away from
  ///                children already well-visited via other parents.  Matches the
  ///                legacy graph-aware behavior.
  /// (0.0, 1.0)   : blend.  Treats each cross-parent visit as partially counting
  ///                toward this parent's visit target.
  ///
  /// The Q values used in V_bar are unaffected (they always reflect cross-parent
  /// backups), so re-routing visits in this way does not distort the parent's Q.
  /// When fraction &gt; 0 some visits may not be placeable (all children over-quota);
  /// MCGSSelect absorbs those visits at the parent.
  /// </summary>
  public float CBGPUCT_SelectCrossParentNFraction = 0.3f;

  /// <summary>
  /// Convenience predicate: true iff any cross-parent N contribution is folded into
  /// the visit-target deficit (i.e. CBGPUCT_CrossParentNFraction is strictly positive).
  /// </summary>
  internal bool CBGPUCT_SelectCrossParentNEnabled => CBGPUCT_SelectCrossParentNFraction > 0.0f;

  /// <summary>
  /// Optional value-uncertainty (leaf value volatility) scaling of the "actual child N"
  /// that the CB-GPUCT visit-target deficit compares against the RPO-justified target
  /// (pi_bar * (sumN + 1)).  When nonzero, each child's actual N is multiplied by
  ///   1 - CBGPUCT_SelectValueUncertaintyScalingFactor * (child.LeafValueVolatilityDebiased - 0.20)
  /// (applied only to children whose child.N exceeds 5, where the debiased volatility
  /// estimate is meaningful; the factor is floored at 0).  0.20 is the assumed average
  /// leaf-value volatility, so a child less (more) volatile than average is treated as if
  /// MORE (FEWER) visits had been performed to it - shrinking (growing) its deficit and
  /// thereby steering exploration toward the less-settled, more volatile children.
  ///
  /// Requires ParamsSearch.TrackLeafValueVolatility = true (enforced in ValidateAgainst).
  /// 0 disables.
  /// </summary>
  public float CBGPUCT_SelectValueUncertaintyScalingFactor = 0;

  #endregion

  /// <summary>
  /// For values > 0, the power mean of the children Q is used in PUCT instead of just Q.
  /// The coefficient used for a given child with N visits is N^POWER_MEAN_N_EXPONENT.
  /// If used, typical values are about 0.12.
  /// Similar to the Power-UCT algorithm described in : "Generalized Mean Estimation in Monte-Carlo Tree Search" by Dam/Klink/D'Eramo/Pters/Pajarinen (2020)
  /// </summary>
  [CeresOption(Name = "power-mean-n-exponent", Desc = "Exponent applied against N for node to determine coefficient used in power mean calculation of Q for child selection", Default = "0")]
  public float PowerMeanNExponent = 0f;

  [CeresOption(Name = "fpu-type", Desc = "Type of first play urgency (non-root nodes)", Default = "PolicyImputedRPO")]
  public FPUType FPUMode = FPUType.PolicyImputedRPO;

  [CeresOption(Name = "fpu-type-at-root", Desc = "Type of first play urgency (root nodes)", Default = "PolicyImputedRPO")]
  public FPUType FPUModeAtRoot = FPUType.PolicyImputedRPO;

  [CeresOption(Name = "fpu-value", Desc = "FPU constant used at root node", Default = "0.44")]
  public float FPUValue = USE_ZZTUNE ? 0.33f : 0.98416f;

  [CeresOption(Name = "fpu-value-at-root", Desc = "FPU constant used at root node", Default = "1")]
  public float FPUValueAtRoot = 1.0f;

  [CeresOption(Name = "policy-softmax", Desc = "Controls degree of flatness of policy via specified level of exponentation", Default = "1.61")]
  public float PolicySoftmax = USE_ZZTUNE ? 1.359f : 1.4f;

  /// <summary>
  /// Flag used for debugging/testing purposes that can be set to true to enable test code.
  /// </summary>
  public bool TestFlag = false;


  /// <summary>
  /// Constructor (uses default values for the class unless overridden in settings file).
  /// </summary>
  public ParamsSelect()
  {
    static void MaybeSet(float? value, ref float target) { if (value.HasValue) target = value.Value; }

    MaybeSet(CeresUserSettingsManager.Settings.CPUCT, ref CPUCT);
    MaybeSet(CeresUserSettingsManager.Settings.CPUCTBase, ref CPUCTBase);
    MaybeSet(CeresUserSettingsManager.Settings.CPUCTFactor, ref CPUCTFactor);
    MaybeSet(CeresUserSettingsManager.Settings.CPUCTAtRoot, ref CPUCTAtRoot);
    MaybeSet(CeresUserSettingsManager.Settings.CPUCTBaseAtRoot, ref CPUCTBaseAtRoot);
    MaybeSet(CeresUserSettingsManager.Settings.CPUCTFactorAtRoot, ref CPUCTFactorAtRoot);
    MaybeSet(CeresUserSettingsManager.Settings.PolicyTemperature, ref PolicySoftmax);
    MaybeSet(CeresUserSettingsManager.Settings.FPU, ref FPUValue);
    MaybeSet(CeresUserSettingsManager.Settings.FPUAtRoot, ref FPUValueAtRoot);
  }



  #region Helper methods

  // Using a virtual relative loss (a negative offset versus the Q of the parent node)
  // is found to be greatly superior to traditional fixed values such as -1
  internal const bool VLossRelative = true;

  public double VLossContribToW(double nInFlight, double parentQ)
    => VLossRelative ? nInFlight * (parentQ + VirtualLossDefaultRelative)
                     : nInFlight * VirtualLossDefaultAbsolute;


  #region Proven win/lost handling

  //TODO: encapsulate all the below in a separate class

  // Actual checkmates (as opposed to tablebase wins)
  // are assigned larger values to distinguish them.
  private const float V_PROVEN_WIN_LOSS_IMMEDIATE_CHECKMATE = 1.0f + 2.0f * V_PROVEN_DELTA;
  private const float V_PROVEN_WIN_LOSS_IMMEDIATE_NON_CHECKMATE = 1.0f + V_PROVEN_DELTA;

  // Distance that a proven win/loss is away from 1.0 or -1.0
  // In this way these definitive evaluations are given slightly more weight
  // than the approximations coming out of our value functions (which might also be very close to -1.0 or 1.0)
  private const float V_PROVEN_DELTA = 0.01f;

  private const float MAX_PLY_AWAY = 255;

  private const float UNITS_PER_PLY = V_PROVEN_DELTA / MAX_PLY_AWAY;

  static float VPenaltyForPlyToMate(int plyDistanceToMate)
  {
    // Prefer mates fewer moves away
    if (plyDistanceToMate == -1) return 0;
    return UNITS_PER_PLY * Math.Min(plyDistanceToMate, MAX_PLY_AWAY);
  }


  public static float WinPForProvenWin(int plyDistanceToMate, bool isImmediateCheckmate)
  {
    float baseValue = isImmediateCheckmate ? V_PROVEN_WIN_LOSS_IMMEDIATE_CHECKMATE : V_PROVEN_WIN_LOSS_IMMEDIATE_NON_CHECKMATE;
    return baseValue - VPenaltyForPlyToMate(plyDistanceToMate);
  }

  public static float LossPForProvenLoss(int plyDistanceToMate, bool isImmediateCheckmate)
  {
    return WinPForProvenWin(plyDistanceToMate, isImmediateCheckmate);
  }

  public static bool VIsForcedResult(float v) => MathF.Abs(v) > 1.0f;

  public static bool VIsForcedWin(float v) => v > 1.00f;
  public static bool VIsForcedLoss(float v) => v < -1.00f;

  public static bool VIsCheckmate(float v) => v < -V_PROVEN_WIN_LOSS_IMMEDIATE_NON_CHECKMATE;

  #endregion

  // TODO: these static UCT helpers would more cleanly be in their own helper class
  public static double UCTParentMultiplier(double parentN, double uctNumeratorPower)
    => uctNumeratorPower == 0.5 ? Math.Sqrt(parentN) : Math.Pow(parentN, uctNumeratorPower);


  /// <summary>
  /// Calculates CPUCT coefficient for a node with specified characteristics.
  /// </summary>
  /// <param name="parentIsRoot"></param>
  /// <param name="dualIteratorMode"></param>
  /// <param name="iteratorID"></param>
  /// <param name="parentN"></param>
  /// <returns></returns>
  public double CalcCPUCT(bool parentIsRoot, float parentN)
  {
    double CPUCT_EXTRA;
    if (parentIsRoot)
    {
      CPUCT_EXTRA = CPUCTFactorAtRoot == 0 ? 0 : CPUCTFactorAtRoot * MathUtils.FastLog((parentN + CPUCTBaseAtRoot + 1.0f) / CPUCTBaseAtRoot);
    }
    else
    {
      CPUCT_EXTRA = CPUCTFactor == 0 ? 0 : CPUCTFactor * MathUtils.FastLog((parentN + CPUCTBase + 1.0f) / CPUCTBase);
    }

    double cpuct = CPUCT + CPUCT_EXTRA;

    if (CPUCTEarlyReductionAmount > 0 && CPUCTEarlyReductionNHalf > 0)
    {
      double suppression = 1.0 - CPUCTEarlyReductionAmount * Math.Exp(-parentN / CPUCTEarlyReductionNHalf);
      cpuct *= suppression;
    }

    return cpuct;
  }


  internal double CalcFPUValue(bool isRoot) => isRoot && FPUModeAtRoot != FPUType.Same ? FPUValueAtRoot : FPUValue;
  internal FPUType GetFPUMode(bool isRoot) => isRoot && FPUModeAtRoot != FPUType.Same ? FPUModeAtRoot : FPUMode;


  /// <summary>
  /// Calculates the Q value to use for unvisited children (First Play Urgency).
  /// This centralizes all FPU mode logic in one place.
  /// </summary>
  /// <param name="isRoot">Whether the parent node is the search root.</param>
  /// <param name="parentQ">The Q value of the parent node.</param>
  /// <param name="parentSumPVisited">Sum of policy values for visited children.</param>
  /// <returns>The Q value to assign to unvisited children.</returns>
  /// <remarks>
  /// For ACPI mode, this returns a fallback value (parentQ). The actual per-child
  /// imputed values are computed separately and override this when available.
  /// </remarks>
  internal double CalcQWhenNoChildren(bool isRoot, double parentQ, double parentSumPVisited)
  {
    double fpuValue = -CalcFPUValue(isRoot);
    return GetFPUMode(isRoot) switch
    {
      FPUType.Absolute => fpuValue,
      FPUType.Reduction => parentQ + fpuValue * Math.Sqrt(parentSumPVisited),
      FPUType.Same => parentQ,
      FPUType.ActionHead => parentQ + fpuValue * Math.Sqrt(parentSumPVisited), // Scalar fallback; per-child values from action head override this in PUCTSelector
      FPUType.PolicyImputedRPO => parentQ + fpuValue * Math.Sqrt(parentSumPVisited), // Scalar fallback; per-child RPO-imputed values override this in PUCTSelector
      _ => throw new NotImplementedException($"Unknown FPUType: {GetFPUMode(isRoot)}")
    };
  }


  /// <summary>
  /// Validates all settigs for self-consistency.
  /// </summary>
  internal void Validate()
  {
    if (FPUMode != FPUType.ActionHead)
    {
      ActionResortUnvisitedChildren = false;
    }

    if (CBGPUCT_Mode != CBGPUCTModeType.None)
    {
      if (FPUMode != FPUType.PolicyImputedRPO
       || FPUModeAtRoot != FPUType.PolicyImputedRPO
       || RPOFPURegularization != RPORegularization.ForwardKLSoftmax)
      {
        throw new Exception($"CBGPUCT_Mode={CBGPUCT_Mode} requires "
                          + $"FPUMode=PolicyImputedRPO, FPUModeAtRoot=PolicyImputedRPO, and "
                          + $"RPOFPURegularization=ForwardKLSoftmax "
                          + $"(have FPUMode={FPUMode}, FPUModeAtRoot={FPUModeAtRoot}, "
                          + $"RPOFPURegularization={RPOFPURegularization}).");
      }
    }
  }


  /// <summary>
  /// Cross-validates this ParamsSelect against the ParamsSearch with which it will be used.
  /// Performed after the per-class Validate() calls.
  /// </summary>
  /// <param name="paramsSearch"></param>
  internal void ValidateAgainst(ParamsSearch paramsSearch)
  {
    if (CBGPUCT_SelectValueUncertaintyScalingFactor != 0 && !paramsSearch.TrackLeafValueVolatility)
    {
      throw new Exception("TrackLeafValueVolatility must be set to true when "
                        + "CBGPUCT_SelectValueUncertaintyScalingFactor is nonzero.");
    }

    if (CBGPUCT_Mode != CBGPUCTModeType.None)
    {
      if (!paramsSearch.EnableGraph)
      {
        throw new Exception($"CBGPUCT_Mode={CBGPUCT_Mode} requires ParamsSearch.EnableGraph=true.");
      }
      if (paramsSearch.PathTranspositionMode != PathMode.PositionEquivalence)
      {
        throw new Exception($"CBGPUCT_Mode={CBGPUCT_Mode} requires PathTranspositionMode=PositionEquivalence.");
      }

      if (MCGSParamsFixed.DEBUG_CBGPUCT)
      {
        Console.WriteLine($"[CBGPUCT] active mode={CBGPUCT_Mode} "
                        + $"select(C={CBGPUCT_SelectLambdaC:F3} sched={CBGPUCT_SelectLambdaSchedule} exp={CBGPUCT_SelectLambdaExp:F3} "
                        + $"cLogBase={CBGPUCT_SelectLambdaCLogBase:F0} cLogFactor={CBGPUCT_SelectLambdaCLogFactor:F3}) "
                        + $"backup(C={CBGPUCT_BackupLambdaC:F3} sched={CBGPUCT_BackupLambdaSchedule} exp={CBGPUCT_BackupLambdaExp:F3}) "
                        + $"selDenom={CBGPUCT_SelectLambdaDenominatorBase} bkpDenom={CBGPUCT_BackupLambdaDenominatorBase} "
                        + $"CrossParentNFraction={CBGPUCT_SelectCrossParentNFraction:F3} "
                        + $"(graph={paramsSearch.EnableGraph}, mode={paramsSearch.PathTranspositionMode})");
      }
    }
  }

  #endregion
}
