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
using System.Text.Json.Serialization;
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
  /// log((N + base + 1) / base), used by the standard PUCT exploration formula
  /// (CPUCTBase / CPUCTBaseAtRoot).
  /// </summary>
  public const float DEFAULT_LOG_GROWTH_BASE = USE_ZZTUNE ? 38739.0f : 45669.0f;

  /// <summary>
  /// Shared default for the multiplicative "factor" on the Lc0-style log growth term,
  /// used by standard PUCT (CPUCTFactor / CPUCTFactorAtRoot).
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
    /// (default ForwardKLSoftmax, identical to PolicyImputed).
    /// </summary>
    PolicyImputedRPO,
  };


  /// <summary>
  /// Minimum policy value for moves (as a fraction) which forces policy probabilities to be at least this level.
  /// </summary>
  public const float MinPolicyProbability = 0.004f;

  /// <summary>
  /// Temperature (lambda) of the policy->Q imputation in RegularizedPolicyOptimum.Solve
  /// (fills Q for unvisited children = FPU, and the TPS backup shrinkage targets).
  /// Default forward-KL form:  q(a) = anchorQ + tau*(log mu_a - E_mu[log mu]).  Higher tau
  /// spreads imputed Q further by policy (top-policy moves look better, tail moves worse),
  /// sharpening the search toward the policy rather than broadening it; tau -> 0 gives a
  /// flat FPU equal to anchorQ (parent Q).  Used when FPUMode is PolicyImputed or PolicyImputedRPO.
  /// </summary>
  public float PolicyImputationTau = 0.15f; // values less than 0.15 test as significantly inferior


  /// <summary>
  /// Choice of KL divergence used by the per-child FPU imputation (FPUMode = PolicyImputed
  /// or PolicyImputedRPO).  Default ForwardKLSoftmax preserves the historical Boltzmann
  /// behavior exactly (and is required by the TPS backup's shrinkage targets).
  /// </summary>
  public RPORegularization RPOFPURegularization = RPORegularization.ForwardKLSoftmax;

  /// <summary>
  /// Amount by which unvisited (unexpanded) child value scores are shifted in RPO mode
  /// from the value computed by RPO imputation.
  /// </summary>
  public float RPOFPUValue = 0.0f;

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
  ///   ParentQ                  - node.Q (search-improved Q; under the TPS backup this is
  ///                              the previous tempered-posterior V, selfV-blended and
  ///                              possibly stale).
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
  public enum FPUQAnchorType
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
  /// (Renamed from CBGPUCT_QAnchorTypeFPU in the 2026-07 TPS consolidation; it is
  /// genuinely FPU-owned.)
  /// </summary>
  public FPUQAnchorType FPU_QAnchorType = FPUQAnchorType.ParentQ;


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


  #region TPS (Tempered Posterior Search) parameters

  /// <summary>
  /// Selects which TPS components are active.  See TPS_PROPOSAL.md and
  /// TPS_CAMPAIGN_RESULTS.md (harness repo, IntegrationTests) for the design and the
  /// tuning/ablation evidence.
  /// </summary>
  public enum TPSModeType
  {
    /// <summary>
    /// TPS off; standard Ceres PUCT selection and standard Q-pure backup.
    /// </summary>
    None,

    /// <summary>
    /// Standard PUCT selection; TPS tempered-posterior backup
    /// (TPSScoreCalc.ComputeVBar overwrites node.Q).
    /// Requires FPUMode=PolicyImputedRPO + RPOFPURegularization=ForwardKLSoftmax
    /// (the imputation machinery supplies the shrinkage targets), graph mode, and
    /// ParamsSearch.TrackLeafValueVolatility (the measured-noise oracle; proven
    /// load-bearing by the 2026-07-13 frozen-s ablation) - all enforced in
    /// Validate/ValidateAgainst.
    /// </summary>
    BackupOnly,

    /// <summary>
    /// PLACEHOLDER (throws NotImplementedException in Validate): TPS selection.
    /// A new SELECT algorithm is forthcoming; the exponential-tilt and reverse-KL
    /// noise-lambda select designs tested in the 2026-07 campaign were removed.
    /// </summary>
    SelectOnly,

    /// <summary>
    /// PLACEHOLDER (throws NotImplementedException in Validate): TPS selection + backup.
    /// </summary>
    SelectAndBackup,
  }

  /// <summary>
  /// TPS mode (see TPSModeType). Default None = standard Ceres PUCT throughout.
  /// </summary>
  public TPSModeType TPS_Mode = TPSModeType.None;

  /// <summary>
  /// Convenience predicate: TPS tempered-posterior backup is active.
  /// </summary>
  [JsonIgnore]
  public bool TPSBackupActive => TPS_Mode == TPSModeType.BackupOnly
                              || TPS_Mode == TPSModeType.SelectAndBackup;

  /// <summary>
  /// Gate for the backup call sites: the TPS recompute-from-children backup replaces
  /// the standard visit-weighted Q backup.  (Alias of TPSBackupActive, kept as the
  /// stable name the backup sites reference.)
  /// </summary>
  [JsonIgnore]
  internal bool RegularizedBackupActive => TPSBackupActive;

  /// <summary>
  /// TPS backup temperature coefficient k_b: tau_backup = k_b * sigma_bar * sqrt(2 ln k),
  /// with k the number of visited children and sigma_bar the node's measured value-noise
  /// scale (support-weighted median of the children's standard errors s_i / sqrt(n_i + 1),
  /// s_i = measured leaf-value volatility).  The sqrt(2 ln k) factor is the analytic
  /// winner's-curse scale.  Dimensionless.  Default 0.25 from the 2026-07-13 tuning
  /// campaign (interior optimum; the analytic k_b~1 starting point was ~4x too warm).
  /// </summary>
  public float TPS_BackupTemperatureK = 0.25f;

  /// <summary>
  /// TPS robust-Q prior-confidence scale sigma0: each child's search Q is shrunk toward
  /// its policy-imputed FPU value with weight w = sigma0^2 / (sigma0^2 + sigma_hat^2)
  /// (inverse-variance shrinkage).  How much estimated standard error we tolerate before
  /// distrusting search Q.  Equivalent to James-Stein with K_eff = (s/sigma0)^2 per unit
  /// support.  Default 0.10 from the 2026-07-13 campaign (interior optimum; smaller
  /// values over-shrink toward the policy consensus, larger under-guard the max).
  /// </summary>
  public float TPS_ShrinkageSigma0 = 0.10f;

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

    if (TPS_Mode == TPSModeType.SelectOnly || TPS_Mode == TPSModeType.SelectAndBackup)
    {
      throw new NotImplementedException(
        $"TPS_Mode={TPS_Mode}: TPS select is not yet implemented "
      + $"(a new SELECT algorithm is forthcoming); use TPS_Mode=BackupOnly or None.");
    }

    if (TPS_Mode != TPSModeType.None)
    {
      if (FPUMode != FPUType.PolicyImputedRPO
       || FPUModeAtRoot != FPUType.PolicyImputedRPO
       || RPOFPURegularization != RPORegularization.ForwardKLSoftmax)
      {
        throw new Exception($"TPS_Mode={TPS_Mode} requires "
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
    if (TPSBackupActive && !paramsSearch.TrackLeafValueVolatility)
    {
      throw new Exception($"TrackLeafValueVolatility must be set to true when a TPS mode is active "
                        + $"(TPS_Mode={TPS_Mode}); the leaf-value volatility tracker is the TPS noise oracle.");
    }

    if (TPS_Mode != TPSModeType.None)
    {
      if (!paramsSearch.EnableGraph)
      {
        throw new Exception($"TPS_Mode={TPS_Mode} requires ParamsSearch.EnableGraph=true.");
      }
      if (paramsSearch.PathTranspositionMode != PathMode.PositionEquivalence)
      {
        throw new Exception($"TPS_Mode={TPS_Mode} requires PathTranspositionMode=PositionEquivalence.");
      }

      if (MCGSParamsFixed.DEBUG_TPS)
      {
        Console.WriteLine($"[TPS] active mode={TPS_Mode} "
                        + $"backup(k_b={TPS_BackupTemperatureK:F3} sigma0={TPS_ShrinkageSigma0:F3}) "
                        + $"(graph={paramsSearch.EnableGraph}, mode={paramsSearch.PathTranspositionMode})");
      }
    }
  }

  #endregion
}
