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

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres
{

  /// <summary>
  /// Subclass of NNEvaluatorOptions specialized for Ceres nets.
  /// </summary>
  [Serializable]
  public record NNEvaluatorOptionsCeres : NNEvaluatorOptions
  {
    public enum PlySinceLastMoveModeEnum
    {
      /// <summary>
      /// Raw value set to zero (e.g. if net not trained with this feature).
      /// </summary>
      Zero,

      /// <summary>
      /// Uses the 8 history planes to attempt to recontruct certain of 
      /// the plys since last move information, else using the default value (e.g. used for starting position).
      /// </summary>
      HistoryPlanesApproximation,

      /// <summary>
      /// Uses the exact values passed in to the evaluator,
      /// computed by the engine or from a PositionWithHistory.
      /// </summary>
      PlySinceLastMovesInputArray
    };


    // Values tuned for Ceres nets.
    public const float DEFAULT_FRACTION_VALUE2 = 0.4f;
    public const float DEFAULT_VALUE1_TEMPERATURE = 0.55f;
    public const float DEFAULT_VALUE2_TEMPERATURE = 1.5f;

    /// <summary>
    /// Default value for QNegativeBlunders/QPositiveBlunders.
    /// A value slightly above zero is generally optimal
    /// (since the training rarely saw values of exactly zero).
    /// </summary>
    public const float DEFAULT_Q_BLUNDER = 0.03f;

    #region Overrides

    /// <summary>
    /// Fraction of the value head 2 that is used to blend into the primary value.
    /// </summary>
    public override float FractionValueHead2 { get; init; } = DEFAULT_FRACTION_VALUE2;

    /// <summary>
    /// Temperature for the value head 2. 
    /// </summary>
    public override float ValueHead1Temperature { get; init; } = DEFAULT_VALUE1_TEMPERATURE;

    /// <summary>
    /// Temperature for the value head 2.
    /// </summary>
    public override float ValueHead2Temperature { get; init; } = DEFAULT_VALUE2_TEMPERATURE;

    #endregion


    /// <summary>
    /// If the prior state information should be used.
    /// </summary>
    public bool UsePriorState { get; init; } = false;

    /// <summary>
    /// If the action head should be used.
    /// </summary>
    public bool UseAction { get; init; } = false;

    /// <summary>
    /// Assumed magnitude (Q units) of adverse blunders that will follow in the game.
    /// </summary>
    public float QNegativeBlunders { get; set; } = DEFAULT_Q_BLUNDER;

    /// <summary>
    /// Assumed magnitude (Q units) of favorable blunders that will follow in the game.
    /// </summary>
    public float QPositiveBlunders { get; set; } = DEFAULT_Q_BLUNDER;

    /// <summary>
    /// If BF16 precision should be used for TensorRT execution.
    /// When true, disables FP16 and FP32 upcasting and uses BF16 instead.
    /// Best suited for datacenter GPUs.
    /// </summary>
    public bool UseBF16 { get; init; } = false;

    /// <summary>
    /// If engine refitting support should be enabled for TensorRT.
    /// When true, the engine can have its weights updated at runtime without rebuilding.
    /// Uses BuilderFlag::kREFIT_IDENTICAL which requires identical weight shapes.
    /// </summary>
    public bool Refittable { get; init; } = false;

    /// <summary>
    /// FP32 norm upcasting scope for TensorRT FP16 mode.
    /// -1 = use mode default, 0 = off, 1 = all norms,
    /// 2 = Q/K/V per-head only, 3 = smolgen only, 4 = Q/K/V + smolgen.
    /// </summary>
    public int Fp32AllNorms { get; init; } = -1;

    /// <summary>
    /// Mode for determining "plys since last move" value to feed into the neural network.
    /// </summary>
    public PlySinceLastMoveModeEnum PlySinceLastMoveMode { get; init; } = PlySinceLastMoveModeEnum.Zero;



    /// <summary>
    /// Default constructor.
    /// </summary>
    public NNEvaluatorOptionsCeres()
    {
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="qNegativeBlunders"></param>
    /// <param name="qPositiveBlunders"></param>
    /// <param name="fractionUndeblunderedValueHead"></param>
    /// <param name="monitorActivations"></param>
    /// <param name="valueHead1Temperature"></param>
    /// <param name="valueHead2Temperature"></param>
    /// <param name="valueHeadAveragePowerMeanOrder"></param>
    /// <param name="policyTemperatureBase"></param>
    /// <param name="policyTemperatureUncertaintyScalingFactor"></param>
    /// <param name="useAction"></param>
    /// <param name="usePriorState"></param>
    /// <exception cref="ArgumentException"></exception>
    public NNEvaluatorOptionsCeres(float qNegativeBlunders = DEFAULT_Q_BLUNDER,
                                   float qPositiveBlunders = DEFAULT_Q_BLUNDER,
                                   float fractionUndeblunderedValueHead = 0,
                                   bool monitorActivations = false,
                                   float valueHead1Temperature = 1,
                                   float valueHead2Temperature = 1,
                                   float valueHeadAveragePowerMeanOrder = 1,
                                   float policyTemperatureBase = 1,
                                   float policyTemperatureUncertaintyScalingFactor = 0,
                                   bool useAction = false,
                                   bool usePriorState = false,
                                   float value1UncertaintyTemperatureScalingFactor = 0,
                                   float value2UncertaintyTemperatureScalingFactor = 0)
    {
      if (valueHead1Temperature <= 0 || valueHead2Temperature <= 0)
      {
        throw new ArgumentException("Temperature must be strictly positive.");
      }

      QNegativeBlunders = qNegativeBlunders;
      QPositiveBlunders = qPositiveBlunders;
      FractionValueHead2 = fractionUndeblunderedValueHead;
      MonitorActivations = monitorActivations;
      ValueHead1Temperature = valueHead1Temperature;
      ValueHead2Temperature = valueHead2Temperature;
      PolicyTemperature = policyTemperatureBase;
      PolicyUncertaintyTemperatureScalingFactor = policyTemperatureUncertaintyScalingFactor;
      UseAction = useAction;
      UsePriorState = usePriorState;
      Value1UncertaintyTemperatureScalingFactor = value1UncertaintyTemperatureScalingFactor;
      Value2UncertaintyTemperatureScalingFactor = value2UncertaintyTemperatureScalingFactor;
    }

    /// <summary>
    /// Short string representation of the options.
    /// </summary>
    public string ShortStr => (UseAction ? "A" : "") + (UsePriorState ? "S" : "")
                           + $"NB: {QNegativeBlunders,4:F2}  PB: {QPositiveBlunders,4:F2} V2: {FractionValueHead2,4:F2} "
                           + $"T: {PolicyTemperature,4:F2}  TS: {PolicyUncertaintyTemperatureScalingFactor,4:F2} ";



    /// <summary>
    /// Applies options from a dictionary to the given options object (or creates a new one if null).
    /// </summary>
    /// <param name="optionsDict"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public override NNEvaluatorOptions OptionsWithOptionsDictApplied(Dictionary<string, string> optionsDict)
    {
      // Mostly just rely upon base class to parse options.
      NNEvaluatorOptions baseOptions = base.OptionsWithOptionsDictApplied(optionsDict);

      // But add on one more option.
      bool board4Mode = optionsDict != null && optionsDict.Keys.Contains("4BOARD");

      // Also up/down blunder default
      float blunderDown = CheckOptionSpecifiedElseDefaultFloat(optionsDict, "BLUN_NEG", DEFAULT_Q_BLUNDER);
      float blunderUp = CheckOptionSpecifiedElseDefaultFloat(optionsDict, "BLUN_POS", DEFAULT_Q_BLUNDER);

      float valueUncertaintyTempScalingFactor1 = CheckOptionSpecifiedElseDefaultFloat(optionsDict, "V1_UNC_SCALE", 0f);
      float valueUncertaintyTempScalingFactor2 = CheckOptionSpecifiedElseDefaultFloat(optionsDict, "V2_UNC_SCALE", 0f);

      bool useBF16 = CheckOptionSpecifiedElseDefaultBoolean(optionsDict, "BF16", false);
      bool refittable = CheckOptionSpecifiedElseDefaultBoolean(optionsDict, "REFITTABLE", false);
      int fp32AllNorms = CheckOptionSpecifiedElseDefaultInt(optionsDict, "FP32ALLNORMS", -1, -1, 4);

      PlySinceLastMoveModeEnum plySinceMode = baseOptions is NNEvaluatorOptionsCeres ceresOptions
                                                           ? ceresOptions.PlySinceLastMoveMode
                                                           : default;

      // Parse LASTPLY option if specified.
      if (optionsDict != null && optionsDict.TryGetValue("LASTPLY", out string lastPlyValue))
      {
        plySinceMode = lastPlyValue.ToUpperInvariant() switch
        {
          "ZERO" => PlySinceLastMoveModeEnum.Zero,
          "POS_HISTORY" => PlySinceLastMoveModeEnum.HistoryPlanesApproximation,
          "SEARCH_HISTORY" => PlySinceLastMoveModeEnum.PlySinceLastMovesInputArray,
          _ => throw new ArgumentException($"Invalid LASTPLY value '{lastPlyValue}'. Must be one of: ZERO, POS_HISTORY, SEARCH_HISTORY")
        };
      }

      // Return composite options.
      // TODO: This is brittle, if we add more options to the base class, we need to
      //       remember to add them here too.
      NNEvaluatorOptionsCeres optionsCeres = this with
      {
        UseAction = board4Mode,
        UsePriorState = board4Mode,

        QNegativeBlunders = blunderDown,
        QPositiveBlunders = blunderUp,
        FractionValueHead2 = baseOptions.FractionValueHead2,
        ValueHead1Temperature = baseOptions.ValueHead1Temperature,
        ValueHead2Temperature = baseOptions.ValueHead2Temperature,
        Value1UncertaintyTemperatureScalingFactor = valueUncertaintyTempScalingFactor1,
        Value2UncertaintyTemperatureScalingFactor = valueUncertaintyTempScalingFactor2,
        PolicyTemperature = baseOptions.PolicyTemperature,
        FractionPolicyHead2 = baseOptions.FractionPolicyHead2,
        Policy1Temperature = baseOptions.Policy1Temperature,
        Policy2Temperature = baseOptions.Policy2Temperature,
        Policy2BlendLogits = baseOptions.Policy2BlendLogits,
        PVExtensionDepth = baseOptions.PVExtensionDepth,
        UseMiddlegameSlowdown = baseOptions.UseMiddlegameSlowdown,
        EnableCUDAGraphs = baseOptions.EnableCUDAGraphs,
        OptimizationLevel = baseOptions.OptimizationLevel,
        PolicyUncertaintyTemperatureScalingFactor = baseOptions.PolicyUncertaintyTemperatureScalingFactor,
        PlySinceLastMoveMode = plySinceMode,
        UseBF16 = useBF16,
        Refittable = refittable,
        Fp32AllNorms = fp32AllNorms,
      };

      return optionsCeres;
    }

  }
}
