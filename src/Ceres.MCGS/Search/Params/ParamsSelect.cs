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
  public const float MinPolicyProbability = 0.005f;

  /// <summary>
  /// Regularization coefficient on Boltzmann calibration of policy-based imputation values.
  /// Higher values make the imputed Q values more closely track the policy (thus more exploration).
  /// Maps to lambda in RegularizedPolicyOptimum.Solve when FPUMode is PolicyImputed or PolicyImputedRPO.
  /// </summary>
  public float PolicyImputationTau = 0.10f;


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
  /// Apportionment algorithm used to translate pi_bar into a concrete visit allocation
  /// in CB-GPUCT.  Default IterativeLargestDeficit matches the historical CBGPUCT inner
  /// loop exactly; HamiltonClosedForm is behaviorally equivalent up to tie-breaking and
  /// is faster for large batches.
  /// </summary>
  public RPOAllocationAlgorithm RPOSelectAllocator = RPOAllocationAlgorithm.IterativeLargestDeficit;

  /// <summary>
  /// Number of Sinkhorn-style fixed-point refinement iterations applied after the
  /// initial Solve in CB-GPUCT selection.  After the first Solve produces pi_bar,
  /// the imputed q values for unvisited children are refreshed via
  ///   q(a) = v_parent + lambda * (1 - mu(a) / pi_bar(a))
  /// and the solve is repeated.  At the (un-clamped) fixed point the regularized
  /// child value v* equals v_parent, enforcing the soft-Bellman expectation that
  /// the parent value is consistent with the regularized aggregate of its children.
  /// This is the reverse-KL analogue of the MatchValue forward-KL anchor.
  ///
  /// Default 2 enables a small amount of refinement at low cost; set to 0 to disable
  /// the iteration entirely (behavior identical to legacy, zero overhead).  Typical
  /// productive range: 2 or 3 iterations.  Higher counts risk pushing unvisited q
  /// values toward the -1 clamp when the visited children's Q values are systematically
  /// higher than v_parent (which can over-suppress exploration).  Cost scales linearly
  /// with the iteration count.  Only takes effect when RPOSelectRegularization = ReverseKL.
  /// </summary>
  public int CBGPUCT_SelectFixedPointIterations = 2;

  /// <summary>
  /// Convergence threshold for the fixed-point refinement: the loop exits early once
  /// the largest single-child change in imputed q falls below this value.
  /// </summary>
  public double RPOSelectFixedPointTol = 1e-4;

  /// <summary>
  /// Number of Sinkhorn-style fixed-point refinement iterations applied after the
  /// initial Solve in CB-GPUCT BACKUP (V_bar computation).  After the first Solve
  /// produces pi_bar and qFill, the imputed q values for unvisited children are
  /// refreshed via
  ///   q(a) = vRef + lambda * (1 - mu(a) / pi_bar(a))
  /// where vRef is the current V_bar-pre-blend (= sum_i pi_bar[i] * q_used[i] with
  /// q_used = qRaw for visited slots and qFill for unvisited).  The Solve is then
  /// repeated.  At convergence E_y[q] equals V_bar, making V_bar the self-consistent
  /// regularized value of the state.
  ///
  /// Semantic difference from the SELECT-side fixed-point: select holds v_parent
  /// constant (the existing parent Q is observed); backup lets vRef float (it IS
  /// V_bar, the value being computed).  This removes the temporal coupling where
  /// today's V_bar would otherwise be anchored to yesterday's V_bar via the imputation.
  ///
  /// Default 0 disables this (no extra solves; behavior identical to legacy backup).
  /// Typical setting: 2 or 3 iterations.  Cost scales linearly with the iteration
  /// count.  Only takes effect when RPOBackupRegularization = ReverseKL (ForwardKL
  /// MatchValue enforces the equivalent constraint in closed form already).
  /// </summary>
  public int CBGPUCT_BackupFixedPointIterations = 0;

  /// <summary>
  /// Convergence threshold for the BACKUP fixed-point refinement.  Loop exits early
  /// once the largest single-child change in imputed q falls below this value.
  /// </summary>
  public double RPOBackupFixedPointTol = 1e-4;

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
  /// Anchor VALUE used to calibrate per-child imputed Q in the BACKUP imputation
  /// (ComputePolicyImpliedQ inside ComputeVBar).  Default ParentQ matches the
  /// legacy backup base-anchor (node.Q); the result is then optionally further
  /// blended with the visit-weighted observed child Q via CBGPUCT_BackupImputationAnchorK.
  /// </summary>
  public CBGPUCTQAnchorType CBGPUCT_QAnchorTypeBackup = CBGPUCTQAnchorType.ParentQ;

  /// <summary>
  /// Pseudo-visit count for blending nodeQ with the visit-weighted observed-child Q
  /// when forming the anchor for backup imputation (used by ComputePolicyImpliedQ
  /// inside ComputeVBar).  Effective formula:
  ///   anchor = (1 - w) * nodeQ + w * visitWeightedObservedQ
  ///   w     = sumN / (sumN + K)
  ///
  /// Motivation: nodeQ under CBGPUCT backup is the previous V_bar, which is selfV-
  /// blended and includes previous imputed contributions.  When few children are
  /// visited the selfV term dominates nodeQ; if selfV is over-optimistic relative
  /// to the children search has actually observed, imputed q's get anchored to a
  /// value that disagrees with the evidence and pi_bar mass flows to phantom-good
  /// unvisited children.  Blending in the empirical visit-weighted child Q lets
  /// search evidence override the stale anchor as N grows.
  ///
  /// Numerical example from a real backup with nodeQ=+0.353, observedAvg=-0.410
  /// (three visited children with q in [-0.706, -0.124]), sumN=3:
  ///     K=infinity (legacy) : anchor=+0.353, q(unvis mu=0.158)=+0.306  (vs evidence)
  ///     K=10                : anchor=+0.177, q=+0.130
  ///     K=3 (default)       : anchor=-0.028, q=-0.075   (matches empirical neighborhood)
  ///     K=1                 : anchor=-0.219, q=-0.266
  ///     K=0                 : anchor=-0.410, q=-0.457   (ignores prior entirely)
  /// At K=3 prior and data carry equal weight at sumN=3 and the data weight reaches
  /// 80% by sumN=15. Set to float.MaxValue to recover the legacy anchor-at-nodeQ behavior.
  /// </summary>
  public float CBGPUCT_BackupImputationAnchorK = 3.0f;

  /// <summary>
  /// When true, the V_bar dot product sums pi_bar * qRaw over VISITED children only
  /// (renormalizing pi_bar over the visited subset).  When false (default), it sums
  /// pi_bar * q over ALL children, with q imputed for unvisited slots via
  /// ComputePolicyImpliedQ.
  ///
  /// Rationale: Grill's V*(s) = sum_a pi*(s,a) Q(s,a) is over the full action set and
  /// assumes Q is known for all actions.  We don't have a Q-network, so unvisited Q
  /// must be imputed from the policy itself.  Including the policy-derived imputed q
  /// in V_bar effectively folds the policy's value-correlation hypothesis back into
  /// the value estimate - useful when the policy is reliable, harmful when it is not
  /// (since the policy already shapes pi_bar through the KL term, double-counting).
  ///
  /// "Observed only" mode is more epistemically honest: V_bar uses only first-hand
  /// search evidence (observed q + selfV at the parent), while pi_bar still benefits
  /// from the full policy-aware regularization (its visited entries' RELATIVE
  /// weighting reflects the regularization solve over the full action set; we just
  /// don't multiply unvisited slots' imputed q into the sum).
  ///
  /// INTERACTION WITH FIXED-POINT (CBGPUCT_BackupFixedPointIterations):
  /// When ObservedOnly is true, the fixed-point iteration's reference value vRef is
  /// also computed over visited only (renormalized pi_bar * qRaw).  This means the
  /// iteration converges to a different fixed point than the full-coverage variant -
  /// not "V_bar = E_y[q] over all actions" but "V_bar = E_y_renorm_visited[qRaw]".
  /// Imputed q's for unvisited slots are still updated by the Sinkhorn formula (so
  /// pi_bar still shifts among visited vs. unvisited), but the IMPUTED q's no longer
  /// contribute to the value the iteration is matching.  The effect: pi_bar pulls
  /// further toward visited children (since the value target ignores unvisited
  /// contributions) and the iteration converges faster (typically 1-2 iters).
  /// Recommendation: leave fixed-point off (the default 0) when first experimenting
  /// with ObservedOnly, then enable it once you've confirmed ObservedOnly is helping.
  /// </summary>
  public bool CBGPUCT_BackupVBarObservedOnly = false;

  /// <summary>
  /// Exponent for PER-CHILD lambda scaling in the BACKUP phase reverse-KL solve.
  /// Replaces the scalar lambda_N with a per-action vector
  ///   lambda_a = lambda_N * (max(N_a, epsilon) / N_max)^p
  /// where N_max is the max edge.N across the considered children and epsilon is
  /// a small floor so unvisited / N=0 slots don't multiply lambda by zero.
  ///
  /// Mathematical effect:
  /// Per-action lambda enters the reverse-KL closed form as
  ///   y(a) = lambda_a * mu(a) / (alpha - q(a))
  /// i.e. it is effectively a per-action prior re-weighting: lambda_a * mu(a) is
  /// the action's effective prior contribution.  A LOW lambda_a means LESS pi_bar
  /// mass on that action - so to suppress low-N children's pi_bar share, we set
  /// LOW lambda for LOW-N children.  This is the OPPOSITE direction of naive
  /// "regularization strength" intuition - in reverse-KL, higher lambda = more
  /// mass, not "more regularization toward mu" (per-action).  Scalar lambda is
  /// different: there, raising it pulls EVERYTHING toward mu uniformly because
  /// the sum-to-one normalization cancels.
  ///
  /// Practical effect: targets the "pi_bar oligarchy" pathology where a single
  /// low-N child with a happen-to-be-good observed q hijacks pi_bar at high
  /// totalN (when scalar lambda_N is small and the bisection approaches argmax-q).
  /// Per-child lambda gives that child a tiny coefficient, so its (alpha - q)
  /// denominator can no longer pull a lot of mass even when it is small.
  ///
  /// Settings:
  ///   0 (default)  : ratio^0 = 1 -> uniform lambda = scalar behavior (backward compat).
  ///   0.5 (sqrt)   : N=1 vs N_max=200 -> lambda_a is 7.1% of lambda_N.
  ///   1.0 (linear) : N=1 vs N_max=200 -> lambda_a is 0.5% of lambda_N.
  ///   2.0 (quadr)  : N=1 vs N_max=200 -> lambda_a is 0.0025% of lambda_N.
  ///
  /// Caveats:
  /// - Reverse-KL only (forward-KL closed form does not generalize cleanly).
  /// - Composes multiplicatively with Q-shrinkage if you enable both, but each
  ///   alone should be sufficient.  Start with one or the other.
  /// - Unvisited slots (edge.N == 0, including extended-coverage unexpanded
  ///   slots when MIN_P_FOR_Q_IF_UNVISITED admits them) get lambda_a near zero
  ///   for p > 0, so they receive essentially no pi_bar mass.  This effectively
  ///   nullifies extended coverage's pi_bar effect while keeping its V_bar
  ///   contribution; treat them as independent knobs.
  /// </summary>
  public float CBGPUCT_BackupLambdaPerChildExponent = 0.0f;



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
  /// Convenience predicate: V_bar regularized backup is active.
  /// </summary>
  public bool CBGPUCTBackupActive => CBGPUCT_Mode == CBGPUCTModeType.BackupOnly
                                  || CBGPUCT_Mode == CBGPUCTModeType.SelectAndBackup;

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
  public CBGPUCTLambdaDenominatorBaseType CBGPUCT_SelectLambdaDenominatorBase = CBGPUCTLambdaDenominatorBaseType.Ten;

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
  /// Bayesian shrinkage of each child's Q estimate toward a per-child policy-implied
  /// target prior to pi_bar computation.  Parameterized as the BLEND FRACTION OF THE
  /// PRIOR AT N=1:
  ///   FractionAtN1 = K / (K + 1)   (equivalently, K = FractionAtN1 / (1 - FractionAtN1))
  /// The schedule uses a power-law decay controlled by DecayExponent (see field below):
  ///   q_shrunk(a) = q(a) * (N^p/(N^p+K)) + q_target(a) * (K/(N^p+K))
  /// At p = 1 this reduces to the standard Bayesian conjugate form (asymptotic 1/N decay);
  /// at p > 1 the schedule decays faster past N=1 while leaving the N=1 value unchanged
  /// (since 1^p = 1 for any p).
  ///
  /// Decay table at FractionAtN1 = 0.13 (K ~ 0.15) under different DecayExponents:
  ///         N=1    N=2    N=5    N=10
  ///   p=1   13%    7.0%   2.9%   1.5%      (current Bayesian)
  ///   p=1.5 13%    5.0%   1.3%   0.47%     (default - moderately faster)
  ///   p=2   13%    3.6%   0.6%   0.15%     (aggressive)
  ///   p=3   13%    1.8%   0.12%  0.015%    (very aggressive)
  ///
  /// The target is the per-child policy-implied Q from ComputePolicyImpliedQ - low-mu
  /// children are shrunk toward a low target, high-mu children toward a high target,
  /// not a single scalar parentQ.
  ///
  /// The shrinkage is skipped for N == 0 (unvisited children retain their imputed Q
  /// without modification, so their pi_bar coefficient stays full and the UCB1-style
  /// "every legal child eventually gets visited" guarantee is preserved).
  ///
  /// Selection-phase setting.  Default 0 disables shrinkage entirely.
  /// </summary>
  public float CBGPUCT_SelectQShrinkageFractionAtN1 = 0.3f;

  /// <summary>
  /// Decay exponent for the SELECT-phase Q-shrinkage schedule.  See
  /// CBGPUCT_SelectQShrinkageFractionAtN1 for the formula and decay table.
  /// p = 1 gives standard Bayesian 1/N decay (theoretically principled).
  /// p > 1 makes the shrinkage vanish faster past N=1 (heuristic, motivated by
  /// "observations are more informative per visit than conjugate-Normal assumes",
  /// reasonable for NN-eval-based search where per-visit variance is low).
  /// Typical range: [1.0, 2.0].  Default 1.5.
  /// </summary>
  public float CBGPUCT_SelectQShrinkageDecayExponent = 1.5f;

  /// <summary>
  /// Same Bayesian Q-shrinkage as the select-phase pair but applied during the BACKUP
  /// phase (in CBGPUCTScoreCalc.ComputeVBar), only affecting pi_bar; the final V_bar
  /// dot product always uses raw (unshrunk) child Q to follow Grill's definition
  /// V_bar(s) = sum_a pi_bar(s,a) * q(s,a).  Same FractionAtN1 -> K reparameterization,
  /// same K/(N^p+K) schedule, same decay table as the select-phase docstring.
  ///
  /// Default 0 disables.
  /// </summary>
  public float CBGPUCT_BackupQShrinkageFractionAtN1 = 0.3f;

  /// <summary>
  /// Decay exponent for the BACKUP-phase Q-shrinkage schedule.  Independently tunable
  /// from the select version; see CBGPUCT_QSelectShrinkageDecayExponent for semantics
  /// and the decay table.  Default 1.5.
  /// </summary>
  public float CBGPUCT_BackupQShrinkageDecayExponent = 1.5f;

  /// <summary>
  /// BOUNDED RELATIVE pi_bar INFLUENCE CAP in the BACKUP phase.  After Solve
  /// produces pi_bar (and after any absolute pi_bar shrinkage via
  /// CBGPUCT_BackupPiBarShrinkagePseudoVisits), cap each visited child's pi_bar
  /// at a bounded-relative ceiling and redistribute the freed mass to the
  /// UNCAPPED visited siblings:
  ///   alpha_i = min(1, (N_i / N_max)^p)
  ///   cap_i   = alpha_i * pi_bar[i] + (1 - alpha_i) * mu_norm[i]
  ///   pi_bar[i] := min(pi_bar[i], cap_i)
  ///   freedMass = sum over capped slots of (pi_bar_pre - cap)
  ///   uncapped slots get a uniform multiplicative boost so visited mass is
  ///   exactly preserved.
  /// N_i is the CHILD NODE'S visit count (edge.ChildNode.N - the cumulative
  /// statistical support for the q estimate across all parents in graph mode,
  /// not the per-edge count); N_max is the max across considered children; for
  /// terminal edges edge.N is used in place of child.N.  mu_norm is mu
  /// normalized to sum to 1 over visited slots only.  Unvisited slots are
  /// neither capped nor scaled - Solve's imputation already gave them their
  /// appropriate pi_bar.
  ///
  /// PRINCIPLE: bounded RELATIVE INFLUENCE (not minimum-MSE estimation).  The
  /// cap-and-redistribute form is the truer reading of the principle than a
  /// symmetric pull toward mu would be:
  ///   - Slots whose pi_bar is at or BELOW their cap are untouched.  In
  ///     particular, the leader (alpha=1, cap=piBar) is always untouched -
  ///     and slots whose pi_bar is small for legitimate low-Q reasons are
  ///     never spuriously boosted toward muNorm.
  ///   - Slots whose pi_bar EXCEEDS their cap (the "single low-N rollout
  ///     dominates V_bar" failure case) get pulled down to cap exactly, and
  ///     the freed mass flows to better-supported siblings.  A capped slot
  ///     stays at cap (the influence bound is preserved strictly, not
  ///     partially undone by a global rescale).
  ///
  /// WHY NOT SYMMETRIC SHRINKAGE: a symmetric mixture pi_bar := alpha*piBar +
  /// (1-alpha)*muNorm has two failure modes.  First, for slots whose Q
  /// legitimately produced a pi_bar below muNorm, it boosts them upward -
  /// rewarding low-Q slots is the opposite of what we want.  Second, since
  /// muNorm sums to 1 over visited but piBar over visited sums to
  /// visitedSumPre, any rescale that preserves total visited mass also scales
  /// the leader (whose alpha=1 was supposed to make it invariant); the leader
  /// loses mass proportional to its visited-share excess (piBar - muNorm),
  /// which compounds into a systematic V_bar pessimism bias.  The one-sided
  /// cap avoids both pathologies by definition.
  ///
  /// alpha mechanics with p = 0.5 (suggested productive starting setting):
  ///   N_i = N_max          : alpha = 1       (cap = piBar; never capped)
  ///   N_i = N_max / 4      : alpha = 0.5     (cap = midpoint of piBar/muNorm)
  ///   N_i = 1, N_max = 200 : alpha = 0.071   (cap close to muNorm)
  /// At p = 1 (linear) the cap is tighter; at p = 0.25 it is looser.  Set 0
  /// to disable (no cap applied; current pi_bar passes through unchanged).
  ///
  /// Composes with the existing absolute pi_bar shrinkage (applied first when
  /// both are active).  Set CBGPUCT_BackupPiBarShrinkagePseudoVisits = 0 to
  /// disable absolute and leave only the bounded-relative cap, or vice versa.
  /// </summary>
  public float CBGPUCT_BackupPiBarShrinkageBoundedRelativeExponent = 0.0f;


  /// <summary>
  /// Bayesian-style shrinkage strength applied to pi_bar itself (per child) toward
  /// the normalized prior mu, weighted by per-child visit count.  For child a with
  /// per-edge N(a) visits:
  ///   pi_bar_shrunk(a) = pi_bar(a) * (N(a) / (N(a) + K)) + mu_norm(a) * (K / (N(a) + K))
  /// then renormalized so sum_a pi_bar_shrunk(a) = 1.
  ///
  /// Motivation (distinct from Q-shrinkage): a single high-Q rollout to a low-policy
  /// child can drive pi_bar to a value near 1.0 for that child, even though the
  /// estimate has almost no statistical support.  Q-shrinkage attempts to fix this by
  /// pulling Q toward parentQ, but is ineffective when the outlier's Q happens to be
  /// near parentQ.  Pi_bar shrinkage attacks the visit-distribution concentration
  /// directly: low-N children's pi_bar is pulled toward their prior, regardless of
  /// where their Q sits.  At N >> K the shrinkage decays to near-identity, so the
  /// RPO solution is preserved for well-visited children.
  ///
  /// Default 0 disables.  Typical setting: K in [2, 5] gives rapidly-decaying shrinkage
  /// (50% pull at N=K, less than 10% pull at N=10K).
  /// </summary>
  public float CBGPUCT_SelectPiBarShrinkagePseudoVisits = 0.0f;

  /// <summary>
  /// Same pi_bar shrinkage as the select-phase pair but applied during the BACKUP
  /// (V_bar) computation in CBGPUCTScoreCalc.ComputeVBar.  Highly recommended when
  /// using CB-GPUCT backup, because V_bar is otherwise vulnerable to domination by
  /// single-visit high-Q outliers (V_bar approx 1.0 * q_outlier even when other children
  /// have far more evidence).
  ///
  /// Default 0 disables.  Typical setting: K in [2, 5].
  /// </summary>
  public float CBGPUCT_BackupPiBarShrinkagePseudoVisits = 0.0f;


  /// <summary>
  /// Lambda schedule for the SELECTION phase (visit-target deficit pi_bar).
  /// </summary>
  public CBGPUCTLambdaScheduleType CBGPUCT_SelectLambdaSchedule = CBGPUCTLambdaScheduleType.AlphaZero;

  /// <summary>
  /// Multiplicative scale on lambda_N for the SELECTION phase.
  /// Larger values keep pi_bar closer to the prior P (more exploration);
  /// smaller values let pi_bar concentrate on high-Q actions.
  /// </summary>
  public float CBGPUCT_SelectLambdaC = 1.75f;

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
  public float CBGPUCT_SelectLambdaCLogFactor = DEFAULT_LOG_GROWTH_FACTOR;

  /// <summary>
  /// Lambda schedule for the BACKUP phase (V_bar regularized value computation).
  /// </summary>
  public CBGPUCTLambdaScheduleType CBGPUCT_BackupLambdaSchedule = CBGPUCTLambdaScheduleType.Pow;

  /// <summary>
  /// Multiplicative scale on lambda_N for the BACKUP phase.
  /// </summary>
  public float CBGPUCT_BackupLambdaC = 1f;

  /// <summary>
  /// Exponent on (sum N_a) in the Pow lambda_N schedule for the BACKUP phase.
  /// Smaller values cause more rapid decay of lambda_N and thus weaker regularization of Q higher visit counts.
  /// </summary>
  public float CBGPUCT_BackupLambdaExp = 0.5f;


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
  public float CBGPUCT_SelectCrossParentNFraction = 0.5f;

  /// <summary>
  /// Convenience predicate: true iff any cross-parent N contribution is folded into
  /// the visit-target deficit (i.e. CBGPUCT_CrossParentNFraction is strictly positive).
  /// </summary>
  internal bool CBGPUCT_SelectCrossParentNEnabled => CBGPUCT_SelectCrossParentNFraction > 0.0f;

  #endregion

  /// <summary>
  /// For values > 0, the power mean of the children Q is used in PUCT instead of just Q.
  /// The coefficient used for a given child with N visits is N^POWER_MEAN_N_EXPONENT.
  /// If used, typical values are about 0.12.
  /// Similar to the Power-UCT algorithm described in : "Generalized Mean Estimation in Monte-Carlo Tree Search" by Dam/Klink/D'Eramo/Pters/Pajarinen (2020)
  /// </summary>
  [CeresOption(Name = "power-mean-n-exponent", Desc = "Exponent applied against N for node to determine coefficient used in power mean calculation of Q for child selection", Default = "0")]
  public float PowerMeanNExponent = 0f;

  [CeresOption(Name = "fpu-type", Desc = "Type of first play urgency (non-root nodes)", Default = "Reduction")]
  public FPUType FPUMode = FPUType.Reduction;

  [CeresOption(Name = "fpu-type", Desc = "Type of first play urgency (root nodes)", Default = "Same")]
  public FPUType FPUModeAtRoot = FPUType.Same;

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
  }


  /// <summary>
  /// Cross-validates this ParamsSelect against the ParamsSearch with which it will be used.
  /// Performed after the per-class Validate() calls.
  /// </summary>
  /// <param name="paramsSearch"></param>
  internal void ValidateAgainst(ParamsSearch paramsSearch)
  {
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
