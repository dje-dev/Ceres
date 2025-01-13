﻿#region License notice

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
  public record NNEvaluatorOptionsCeres : NNEvaluatorOptions
  {
    // Values tuned for Ceres nets.
    public const float DEFAULT_FRACTION_VALUE2 = 0.4f;
    public const float DEFAULT_VALUE1_TEMPERATURE = 0.8f;
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


    #region Options helpers
    private static float CheckOptionSpecifiedElseDefault(Dictionary<string, string> options,
                                                         string optionKey,
                                                         float defaultValue)
    {
      float returnValue = defaultValue;
      string optionString = (options != null && options.Keys.Contains(optionKey))
                             ? options[optionKey]
                             : null;
      if (optionString != null)
      {
        if (!float.TryParse(optionString, out returnValue))
        {
          throw new Exception($"Invalid value for {optionKey}, expected number but got: {optionString}");
        }
      }

      return returnValue;
    }


    /// <summary>
    /// Applies options from a dictionary to the given options object (or creates a new one if null).
    /// </summary>
    /// <param name="options"></param>
    /// <param name="optionsDict"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public override NNEvaluatorOptions OptionsWithOptionsDictApplied(Dictionary<string, string> optionsDict)
    {
   
      float value1Temperature = CheckOptionSpecifiedElseDefault(optionsDict, "V1TEMP", ValueHead1Temperature);
      float value2Temperature = CheckOptionSpecifiedElseDefault(optionsDict, "V2TEMP", ValueHead2Temperature);
      float value2Weight = CheckOptionSpecifiedElseDefault(optionsDict, "V2FRAC", FractionValueHead2);
      Console.WriteLine(value2Weight + " " + value1Temperature + " " + value2Temperature);
      //          float policyUncertaintyScaling = CheckOptionSpecifiedElseDefault(netDef, options, "POLUNC_SCALE", DEFAULT_POLUNC_SCALE);        
      //          if (options != null && options.Keys.Contains("POLUNC"))

      bool board4Mode = optionsDict != null && optionsDict.Keys.Contains("4BOARD");

      NNEvaluatorOptionsCeres optionsCeres = this with
      {
        UseAction = board4Mode,
        UsePriorState = board4Mode,

        FractionValueHead2 = value2Weight, //options != null && options.Keys.Contains("USEV2") ? 0.5f : 0f
        ValueHead1Temperature = value1Temperature,
        ValueHead2Temperature = value2Temperature,
        //            PolicyUncertaintyTemperatureScalingFactor = policyUncertaintyScaling,
      };

      return optionsCeres;
    }

    #endregion

  }

}
