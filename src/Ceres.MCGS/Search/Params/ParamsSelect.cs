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
    ///	Uses policy-based imputation to estimate Q values.
    /// Per-child imputed Q via Boltzmann calibration over the policy:
    /// Q_i = anchor + tau * (log pi_i - log pi_anchor). 
    /// Anchored on the top-policy child's observed Q when it has visits, 
    /// else on parent V (so E_pi[Q] == parent V).
    /// </summary>
    PolicyImputed,
  };


  /// <summary>
  /// Minimum policy value for moves (as a fraction) which forces policy probabilities to be at least this level.
  /// </summary>
  public const float MinPolicyProbability = 0.005f;

  /// <summary>
  /// Regularization coefficient on Boltzmann calibration of policy-based imputation values. 
  /// Higher values make the imputed Q values more closely track the policy (thus more exploration).
  /// </summary>
  public float PolicyImputationTau = 0.10f;
  

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
  public float CPUCTBase = USE_ZZTUNE ? 38739 : 45669;

  [CeresOption(Name = "cpuct-factor", Desc = "Constant (factor) used in node selection to weight exploration", Default = "2.82")]
  public float CPUCTFactor = USE_ZZTUNE ? 3.894f : 3.973f;


  [CeresOption(Name = "cpuct-at-root", Desc = "Scaling used in node selection (at root) to encourage exploration (traditional UCT)", Default = "3")]
  public float CPUCTAtRoot = USE_ZZTUNE ? 1.745f : 2.897f;


  [CeresOption(Name = "cpuct-base-at-root", Desc = "Constant (base) used in node selection (at root) to weight exploration", Default = "18368")]
  public float CPUCTBaseAtRoot = USE_ZZTUNE ? 38739 : 45669;


  [CeresOption(Name = "cpuct-factor-at-root", Desc = "Constant (factor) used in node selection (at root) to weight exploration", Default = "2.82")]
  public float CPUCTFactorAtRoot = USE_ZZTUNE ? 3.894f : 3.973f;

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
  /// Schedule used to compute lambda_N (the regularization strength) from sum N_a:
  /// |A| = actual number of children (numChildren) in both schedules.
  /// LambdaExp is ignored when schedule is UCT.
  /// Phase-specific instances (CBGPUCT_SelectLambda* and CBGPUCT_BackupLambda*) below.
  /// </summary>
  public enum CBGPUCTLambdaScheduleType
  {
    /// <summary>
    ///  LambdaC * pow(sumN, LambdaExp) / (|A| + sumN). Standard Grill schedule.
    /// </summary>
    Pow,

    /// <summary>
    /// LambdaC * sqrt(log(sumN + e) / (|A| + sumN)) UCT-style log term. 
    /// Slower asymptotic decay than Pow (sqrt(log N / N) vs 1/sqrt(N)), 
    /// so regularization stays meaningful at higher sumN.
    /// </summary>
    UCT
  }

  /// <summary>
  /// The value used for |A| (number of actions) in the denominator of the lambda_N schedules. 
  /// According to Grill this would be average number of actions (e.g. 30)
  /// causing enhanced regularization at low visit counts, but empirically small values are found better.
  /// </summary>
  public const int CGPUCT_BaseNumVisits = 1;

  /// <summary>
  /// Number of "pseudovisits" used to shrink Q toward parent Q when computing Q for backup.
  /// </summary>
  public double CGPUCT_NumUncertaintyPseudovisits = 0;


  /// <summary>
  /// Lambda schedule for the SELECTION phase (visit-target deficit pi_bar).
  /// </summary>
  public CBGPUCTLambdaScheduleType CBGPUCT_SelectLambdaSchedule = CBGPUCTLambdaScheduleType.Pow;

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
  /// Lambda schedule for the BACKUP phase (V_bar regularized value computation).
  /// </summary>
  public CBGPUCTLambdaScheduleType CBGPUCT_BackupLambdaSchedule = CBGPUCTLambdaScheduleType.Pow;

  /// <summary>
  /// Multiplicative scale on lambda_N for the BACKUP phase.
  /// </summary>
  public float CBGPUCT_BackupLambdaC = 0.75f;

  /// <summary>
  /// Exponent on (sum N_a) in the Pow lambda_N schedule for the BACKUP phase.
  /// Smaller values cause more rapid decay of lambda_N and thus weaker regularization of Q higher visit counts.
  /// </summary>
  public float CBGPUCT_BackupLambdaExp = 0.4f;

  /// <summary>
  /// If true, the visit-target deficit uses the destination node's total
  /// (graph-aggregated) visit count rather than the per-edge visit count:
  ///   deficit(a) = pi_bar(a) * (sum_a edge.N + 1) - child(a).N
  /// In transposition cases this redirects exploration away from children
  /// already well-visited via other parents - sending visits where the
  /// confidence about the child's value is lowest. The Q values used in
  /// V_bar are unaffected (they always reflect cross-parent backups), so
  /// re-routing visits in this way does not distort the parent's Q.
  ///
  /// When false, deficit uses per-edge N - each parent independently aims
  /// for its own visit-count target without considering other parents'
  /// contributions to the same child.
  /// </summary>
  public bool CBGPUCT_GraphAwareDeficit = true;

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
      FPUType.PolicyImputed => parentQ + fpuValue * Math.Sqrt(parentSumPVisited), // Scalar fallback; per-child RPO-imputed values override this in PUCTSelector
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
                        + $"select(C={CBGPUCT_SelectLambdaC:F3} sched={CBGPUCT_SelectLambdaSchedule} exp={CBGPUCT_SelectLambdaExp:F3}) "
                        + $"backup(C={CBGPUCT_BackupLambdaC:F3} sched={CBGPUCT_BackupLambdaSchedule} exp={CBGPUCT_BackupLambdaExp:F3}) "
                        + $"GraphAwareDeficit={CBGPUCT_GraphAwareDeficit} "
                        + $"(graph={paramsSearch.EnableGraph}, mode={paramsSearch.PathTranspositionMode})");
      }
    }
  }

  #endregion
}
