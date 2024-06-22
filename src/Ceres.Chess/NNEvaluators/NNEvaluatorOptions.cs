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

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Options relating to an NNEvaluator, including output head postprocessing.
  /// </summary>
  public record NNEvaluatorOptions
  {
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
    public float FractionValueHead2 { get; init; } = 0;

    /// <summary>
    /// Temperature for the value head 2. 
    /// </summary>
    public float ValueHead1Temperature { get; init; } = 1;

    /// <summary>
    /// Temperature for the value head 2.
    /// </summary>
    public float ValueHead2Temperature { get; init; } = 1;

    /// <summary>
    /// Optional scaling factor that determines the amount by which 
    /// the value 1 temperature is scaled based on position-specific value uncertainty.
    /// </summary>
    public float Value1UncertaintyTemperatureScalingFactor { get; init; } = 0;

    /// <summary>
    /// Optional scaling factor that determines the amount by which 
    /// the value 2 temperature is scaled based on position-specific value uncertainty.
    /// </summary>
    public float Value2UncertaintyTemperatureScalingFactor { get; init; } = 0;

    #endregion


    #region Policy Head Options

    /// <summary>
    /// Base policy temperature to apply.
    /// </summary>
    public float PolicyTemperature { get; init; } = 1.0f;

    /// <summary>
    /// Optional scaling factor that determines the amount by which 
    /// the policy temperature is scaled based on position-specific policy uncertainty.
    /// </summary>
    public float PolicyUncertaintyTemperatureScalingFactor { get; init; } = 0;

    #endregion


    #region Miscellaneous Options

    /// <summary>
    /// If true, monitor activations of the neural network. 
    /// </summary>
    public bool MonitorActivations { get; init; } = false;

    #endregion
  }
}
