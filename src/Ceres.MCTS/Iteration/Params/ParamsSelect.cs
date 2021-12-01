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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Ceres.Base.Math;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.MCTS.Params
{
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

    public enum FPUType { Absolute, Reduction, Same };

    /// <summary>
    /// Minimum policy value for moves (as a fraction) which forces policy probabilities to be at least this level.
    /// </summary>
    public const float MinPolicyProbability = 0.005f;


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
    public float CPUCT = USE_ZZTUNE ? 1.745f : 2.15f;

    [CeresOption(Name = "cpuct-base", Desc = "Constant (base) used in node selection to weight exploration", Default = "18368")]
    public float CPUCTBase = USE_ZZTUNE ? 38739 : 18368;

    [CeresOption(Name = "cpuct-factor", Desc = "Constant (factor) used in node selection to weight exploration", Default = "2.82")]
    public float CPUCTFactor = USE_ZZTUNE ? 3.894f : 2.82f;


    [CeresOption(Name = "cpuct-at-root", Desc = "Scaling used in node selection (at root) to encourage exploration (traditional UCT)", Default = "3")]
    public float CPUCTAtRoot = USE_ZZTUNE ? 1.745f : 2.15f;


    [CeresOption(Name = "cpuct-base-at-root", Desc = "Constant (base) used in node selection (at root) to weight exploration", Default = "18368")]
    public float CPUCTBaseAtRoot = USE_ZZTUNE ? 38739 : 18368;


    [CeresOption(Name = "cpuct-factor-at-root", Desc = "Constant (factor) used in node selection (at root) to weight exploration", Default = "2.82")]
    public float CPUCTFactorAtRoot = USE_ZZTUNE ? 3.894f : 2.82f;

    [CeresOption(Name = "policy-decay-factor", Desc = "Linear scaling factor used in node selection to shrink policy toward uniform as N grows. Zero to disable, typical value 5.0", Default = "0")]
    public float PolicyDecayFactor = 0;

    [CeresOption(Name = "policy-decay-exponent", Desc = "Exponent used in scaling factor (applied to N) used in node selection to shrink policy toward uniform as N grows. Zero to disable, typical value 0.38", Default = "0")]
    public float PolicyDecayExponent = 0.38f;

    /// <summary>
    /// Amount of relative virtual loss to apply in leaf selection to discourage collisions.
    /// Values closer to zero yield less distortion in choosing leafes (thus higher quality play)
    /// but slow down search speed because of excess collisions. Values of -0.10 or -0.15 work well.
    /// </summary>
    // Smaller values yield higher fidelity leaf selection 
    // (but possibly slightly slower due to increased collisions, especially at smaller node counts)      
    [CeresOption(Name = "vloss-relative", Desc = "Virtual loss (relative) to be applied when collisions encountered", Default = "-0.15")]
    public float VirtualLossDefaultRelative = -0.10f;

    /// <summary>
    /// Virtual loss to be used if VLossRelative is false.
    /// </summary>
    [CeresOption(Name = "vloss-absolute", Desc = "Virtual loss (absolute) to be applied when collisions encountered", Default = "-1.0")]
    public float VirtualLossDefaultAbsolute = -1.0f;

    public bool UseDynamicVLoss = false;


    /// <summary>
    /// For values > 0, the power mean of the children Q is used in PUCT instead of just Q.
    /// The coefficient used for a given child with N visits is N^POWER_MEAN_N_EXPONENT.
    /// If used, typical values are about 0.12.
    /// Similar to the Power-UCT algorithm described in : "Generalized Mean Estimation in Monte-Carlo Tree Search" by Dam/Klink/D'Eramo/Pters/Pajarinen (2020)
    /// </summary>
    [CeresOption(Name = "power-mean-n-exponent", Desc = "Exponent applied against N for node to determine coefficient used in power mean calculation of Q for child selection", Default = "0")]
    public float PowerMeanNExponent = 0f;

    [CeresOption(Name = "first-move-thompson-sampling-factor", Desc = "Scaling factor applied to Thompson sampling applied in search child selection at root node (typical 0.15 or 0.00 for disabled)", Default = "0")]
    public float FirstMoveThompsonSamplingFactor = 0.0f;

    [CeresOption(Name = "fpu-type", Desc = "Type of first play urgency (non-root nodes)", Default = "Reduction")]
    public FPUType FPUMode = FPUType.Reduction;

    [CeresOption(Name = "fpu-type", Desc = "Type of first play urgency (root nodes)", Default = "Same")]
    public FPUType FPUModeAtRoot = FPUType.Same;

    [CeresOption(Name = "fpu-value", Desc = "FPU constant used at root node", Default = "0.44")]
    public float FPUValue = USE_ZZTUNE ? 0.33f : 0.44f;

    [CeresOption(Name = "fpu-value-at-root", Desc = "FPU constant used at root node", Default = "1")]
    public float FPUValueAtRoot = 1.0f;

    [CeresOption(Name = "policy-softmax", Desc = "Controls degree of flatness of policy via specified level of exponentation", Default = "1.61")]
    public float PolicySoftmax = USE_ZZTUNE ? 1.359f : 1.61f;


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

    /// <summary>
    /// If using dual selectors, we perturb CPUCT by this fraction (one is perturbed up, one down)
    /// Among other beneifts, this reduces collisions in building batches because we get diversity from the two selectors.
    /// Tests were not definitive regarding the optimal value. Therefore we use a conservative value of circa 0.15 to 0.25
    /// which stays fairly close to standard UCT but still gets the job done in terms of producing diversity in batch collection.
    /// </summary>
    public const float CPUCTDualSelectorDiffFraction = 0.20f;

    const float DUAL_SELECTOR_0_CPUCT_MULTIPLIER = 1.0f + (CPUCTDualSelectorDiffFraction * 0.5f);
    const float DUAL_SELECTOR_1_CPUCT_MULTIPLIER = 1.0f - (CPUCTDualSelectorDiffFraction * 0.5f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    float CPUCTForSelector(bool dualSelectorMode, int selectorID, float baseCPUCT)
    {
      if (dualSelectorMode)
        return baseCPUCT * (selectorID == 0 ? DUAL_SELECTOR_0_CPUCT_MULTIPLIER : DUAL_SELECTOR_1_CPUCT_MULTIPLIER);
      else
        return baseCPUCT;
    }

    // Using a virtual relative loss (a negative offset versus the Q of the parent node)
    // is found to be greatly superior to traditional fixed values such as -1
    internal const bool VLossRelative = true;

    public float VLossContribToW(float nInFlight, float parentQ)
      => VLossRelative ? nInFlight * (parentQ + VirtualLossDefaultRelative)
                        : nInFlight * VirtualLossDefaultAbsolute;


    #region Proven win/lost handling

    //TODO: encapsulate all the below in a separate class

    // Actual checkmates (as opposed to tablebase wins)
    // are assigned larger values to distinguish them.
    private const float V_PROVEN_WIN_LOSS_IMMEDIATE_CHECKMATE = 1.0f + (2.0f * V_PROVEN_DELTA);
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
    public static float UCTParentMultiplier(float parentN, float uctNumeratorPower)
    {
      if (uctNumeratorPower == 0.5f)
        return MathF.Sqrt(parentN);
      else
      {
        return MathF.Pow(parentN, uctNumeratorPower);
      }
    }

    /// <summary>
    /// Calculators CPUCT coefficient for a node with specified characteristics.
    /// </summary>
    /// <param name="parentIsRoot"></param>
    /// <param name="dualSelectorMode"></param>
    /// <param name="selectorID"></param>
    /// <param name="parentN"></param>
    /// <returns></returns>
    public float CalcCPUCT(bool parentIsRoot, bool dualSelectorMode, int selectorID, float parentN)
    {
      if (parentIsRoot)
      {
        float CPUCT_EXTRA = (CPUCTFactorAtRoot == 0) ? 0 : CPUCTFactorAtRoot * FastLog.Ln((parentN + CPUCTBaseAtRoot + 1.0f) / CPUCTBaseAtRoot);
        float thisCPUCT = CPUCTAtRoot + CPUCT_EXTRA;
        float cpuctValue = CPUCTForSelector(dualSelectorMode, selectorID, thisCPUCT);
        return cpuctValue;
      }
      else
      {
        float CPUCT_EXTRA = (CPUCTFactor == 0) ? 0 : CPUCTFactor * FastLog.Ln((parentN + CPUCTBase + 1.0f) / CPUCTBase);
        float thisCPUCT = CPUCT + CPUCT_EXTRA;
        float cpuctValue = CPUCTForSelector(dualSelectorMode, selectorID, thisCPUCT);
        return cpuctValue;
      }
    }

    internal float CalcFPUValue(bool isRoot) => (isRoot && FPUModeAtRoot != FPUType.Same) ? FPUValueAtRoot : FPUValue;
    internal FPUType GetFPUMode(bool isRoot) => (isRoot && FPUModeAtRoot != FPUType.Same) ? FPUModeAtRoot : FPUMode;

    #endregion

  }

}

