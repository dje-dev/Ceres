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

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Options relating to an NNEvaluator, including output head postprocessing.
  /// </summary>
  public record NNEvaluatorOptions
  {
    public const float DEFAULT_FRACTION_VALUE2    = 0f;
    public const float DEFAULT_VALUE1_TEMPERATURE = 1f;
    public const float DEFAULT_VALUE2_TEMPERATURE = 1f;
    public const float DEFAULT_POLICY_TEMPERATURE = 1f;

    #region Value Head Options

    /// <summary>
    /// Constructor.
    /// </summary>
    public NNEvaluatorOptions()
    {
    }

    /// <summary>
    /// Fraction of the value head 2 that is used to blend into the primary value.
    /// </summary>
    public virtual float FractionValueHead2 { get; init; } = DEFAULT_FRACTION_VALUE2;

    /// <summary>
    /// Temperature for the value head 2. 
    /// </summary>
    public virtual float ValueHead1Temperature { get; init; } = DEFAULT_VALUE1_TEMPERATURE;

    /// <summary>
    /// Temperature for the value head 2.
    /// </summary>
    public virtual float ValueHead2Temperature { get; init; } = DEFAULT_VALUE2_TEMPERATURE;

    /// <summary>
    /// Optional scaling factor that determines the amount by which 
    /// the value 1 temperature is scaled based on position-specific value uncertainty.
    /// </summary>
    public virtual float Value1UncertaintyTemperatureScalingFactor { get; init; } = 0;

    /// <summary>
    /// Optional scaling factor that determines the amount by which 
    /// the value 2 temperature is scaled based on position-specific value uncertainty.
    /// </summary>
    public virtual float Value2UncertaintyTemperatureScalingFactor { get; init; } = 0;

    /// <summary>
    /// Number of additional PV children by which each evaluated position is extended.
    /// </summary>
    public int PVExtensionDepth { get; init;  } = 0;

    #endregion

    #region Applying supplemental options

    /// <summary>
    /// Applies options from a dictionary to the given options object (or creates a new one if null).
    /// </summary>
    /// <param name="optionsDict"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public virtual NNEvaluatorOptions OptionsWithOptionsDictApplied(Dictionary<string, string> optionsDict)
    {
      float pvExtensionDepth = CheckOptionSpecifiedElseDefault(optionsDict, "DEPTH", PVExtensionDepth);

      float value1Temperature = CheckOptionSpecifiedElseDefault(optionsDict, "V1TEMP", ValueHead1Temperature);
      float value2Temperature = CheckOptionSpecifiedElseDefault(optionsDict, "V2TEMP", ValueHead2Temperature);
      float value2Weight = CheckOptionSpecifiedElseDefault(optionsDict, "V2FRAC", FractionValueHead2);
      float policyUncertaintyScaling = CheckOptionSpecifiedElseDefault(optionsDict, "POLUNC_SCALE", PolicyUncertaintyTemperatureScalingFactor);
      float policyTemperature = CheckOptionSpecifiedElseDefault(optionsDict, "POLTEMP", PolicyTemperature);

//      float blunderNegative = CheckOptionSpecifiedElseDefault(optionsDict, "BLUN_NEG", QDev);
//      float blunderPositive = CheckOptionSpecifiedElseDefault(optionsDict, "BLUN_POS", QDev);

      if (value2Weight != 0 || value1Temperature != 1 || value2Temperature != 1 || policyTemperature != 1)
      {
        Console.WriteLine("OVERRIDDEN V2FRAC/V1TEMP/V2TEMP/POLTEMP/BLUN_NEG/BLUN_POS: " + value2Weight + " " + value1Temperature + " "
                                                                      + value2Temperature + " " + policyTemperature);
//                                                                      blunderNegative + " " + blunderPositive);
      }

      NNEvaluatorOptions options = this with
      {
        FractionValueHead2 = value2Weight,
        ValueHead1Temperature = value1Temperature,
        ValueHead2Temperature = value2Temperature,
        PVExtensionDepth = (int)pvExtensionDepth,
        PolicyUncertaintyTemperatureScalingFactor = policyUncertaintyScaling,
        PolicyTemperature = CheckOptionSpecifiedElseDefault(optionsDict, "POLTEMP", policyTemperature),
      };

      return options;
    }

    #endregion

    #region Policy Head Options

    /// <summary>
    /// Base policy temperature to apply.
    /// </summary>
    public virtual float PolicyTemperature { get; init; } = 1.0f;

    /// <summary>
    /// Optional scaling factor that determines the amount by which 
    /// the policy temperature is scaled based on position-specific policy uncertainty.
    /// </summary>
    public virtual float PolicyUncertaintyTemperatureScalingFactor { get; init; } = 0;

    #endregion

    #region Head Overrides

    /// <summary>
    /// Optional list of head overrides.
    /// </summary>
    public NNEvaluatorHeadOverride[] HeadOverrides;


    #endregion

    #region Miscellaneous Options

    /// <summary>
    /// If true, monitor activations of the neural network. 
    /// </summary>
    public bool MonitorActivations { get; init; } = false;

    #endregion


    #region Options helpers

    /// <summary>
    /// Returns value in dictinoary with specified key (parsed as a float) specified default if not found.
    /// </summary>
    /// <param name="options"></param>
    /// <param name="optionKey"></param>
    /// <param name="defaultValue"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    protected static float CheckOptionSpecifiedElseDefault(Dictionary<string, string> options,
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

    #endregion
  }
}
