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

#endregion

namespace Ceres.MCTS.Params
{
  /// <summary>
  /// Parameters related to leaf selection (CPUCT algorithm).
  /// </summary>
  [Serializable]
  public class ParamsSelect
  {
    public enum FPUType { Absolute, Reduction, Same };

    /// <summary>
    /// Minimum policy value for moves (as a fraction) which forces policy probabilities to be at least this level.
    /// </summary>
    public const float MinPolicyProbability = 0.005f;


    public bool RandomizeQ = false;
    public const float RandomizeScale = 0.010f * 10f;

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
    public float CPUCT = 2.15f;

    [CeresOption(Name = "cpuct-base", Desc = "Constant (base) used in node selection to weight exploration", Default = "18368")]
    public float CPUCTBase = 18368;

    [CeresOption(Name = "cpuct-factor", Desc = "Constant (factor) used in node selection to weight exploration", Default = "2.82")]
    public float CPUCTFactor = 2.82f;


    [CeresOption(Name = "cpuct-at-root", Desc = "Scaling used in node selection (at root) to encourage exploration (traditional UCT)", Default = "2.15")]
    public float CPUCTAtRoot = 2.15f;

    [CeresOption(Name = "cpuct-base-at-root", Desc = "Constant (base) used in node selection (at root) to weight exploration", Default = "18368")]
    public float CPUCTBaseAtRoot = 18368;

    /// <summary>
    /// 
    /// NOTE: Attempts to use 1.5x or 2.0x multipliers seemed modestly promising
    ///       at moderate node counts (and extremely beneficial with 10x128 nets).
    ///       But with larger networks and longer searches (e.g. 500k+/node) clearly worse.
    /// </summary>
    [CeresOption(Name = "cpuct-factor-at-root", Desc = "Constant (factor) used in node selection (at root) to weight exploration", Default = "2.82")]
    public float CPUCTFactorAtRoot = 2.82f;


    [CeresOption(Name = "cpuct2", Desc = "Scaling constant used in node selection to weight deep exploration", Default = "0")]
    public float CPUCT2 = 0; // Tried value of around 1.0, possibly also reducing CPUCT to 1/2 usual level. Possible small benefit

    // NOTE: value of 1.0 or 1.5 seemed possibly better in suites, but underperformed 3 190 5 in a T40 match @500,000 nodes per move
    [CeresOption(Name = "cpolicy-fade", Desc = "Scaling constant used in node selection to weight exploration (policy prior fade). Zero to disable, typical value 1.0", Default = "0")]
    public float CPolicyFade = 0;

    /// <summary>
    /// Amount of relative virtual loss to apply in leaf selection to discourage collisions.
    /// Values closer to zero yield less distortion in choosing leafes (thus higher quality play)
    /// but slow down search speed because of excess collisions. Values of -0.10 or -0.15 work well.
    /// </summary>
    // Smaller values yield higher fidelity leaf selection 
    // (but possibly slightly slower due to increased collisions, especially at smaller node counts)      
    [CeresOption(Name = "vloss-relative", Desc = "Virtual loss (relative) to be applied when collisions encountered", Default = "-0.15")]
    public float VirtualLossDefaultRelative = ParamsSearch.USE_CERES_ADJUSTMENTS ? -0.10f : -0.15f;

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

    [CeresOption(Name = "fpu-value", Desc = "FPU constant used at root node", Default = "-0.44")]
    public float FPUValue = -0.44f;

    [CeresOption(Name = "fpu-value-at-root", Desc = "FPU constant used at root node", Default = "-1")]
    public float FPUValueAtRoot = -1.0f;

    [CeresOption(Name = "policy-softmax", Desc = "Controls degree of flatness of policy via specified level of exponentation", Default = "1.61")]
    public float PolicySoftmax = 1.61f;

 
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

    private const float V_PROVEN_WIN_LOSS_IMMEDIATE = 1.0f + V_PROVEN_DELTA;

    // Distance that a proven win/loss is away from 1.0 or -1.0
    // In this way these definitive evaluations are given slightly more weight
    // than the approximations coming out of our value functions (which might also be very close to -1.0 or 1.0)
    private const float V_PROVEN_DELTA = 0.1f;

    private const float MAX_PLY_AWAY = 200;

    private const float UNITS_PER_PLY = V_PROVEN_DELTA / MAX_PLY_AWAY;

    static float VPenaltyForPlyToMate(int plyDistanceToMate = -1)
    {
      // Prefer mates fewer moves away
      if (plyDistanceToMate == -1) return 0;
      return UNITS_PER_PLY * Math.Min(plyDistanceToMate, MAX_PLY_AWAY);
    }

    public static float PlyToMateFromVWithProvenWinLoss(float v)
    {
      Debug.Assert(MathF.Abs(v) > 1.0f);
      float distance = MathF.Abs(v - 1) - V_PROVEN_DELTA;
      float UNITS_PER_PLY = V_PROVEN_DELTA / MAX_PLY_AWAY;
      return MathF.Abs(distance) / UNITS_PER_PLY;
    }

    public static float WinPForProvenWin(int plyDistanceToMate = -1) => V_PROVEN_WIN_LOSS_IMMEDIATE - VPenaltyForPlyToMate(plyDistanceToMate);
    public static float LossPForProvenLoss(int plyDistanceToMate = -1) => V_PROVEN_WIN_LOSS_IMMEDIATE - VPenaltyForPlyToMate(plyDistanceToMate);

    public static bool VIsForcedResult(float v) => MathF.Abs(v) > 1.0f;

    public static bool VIsForcedWin(float v) => v > 1.00f;
    public static bool VIsForcedLoss(float v) => v < -1.00f;


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


  }

}

