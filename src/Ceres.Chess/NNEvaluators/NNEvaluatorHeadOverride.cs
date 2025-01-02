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

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Definition of an override for a head in a neural network evaluator.
  /// Allows for on-the-fly custom evaluation of a head output.
  /// </summary>
  public readonly record struct NNEvaluatorHeadOverride
  {
    public enum HeadTypeEnum { None, Value1, Value2, Policy, Action, State, UncertaintyV, UncertaintyP };

    /// <summary>
    /// Identification string of the override.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Type of the head to override.
    /// </summary>
    public readonly HeadTypeEnum HeadType;

    /// <summary>
    /// Name of the input layer to be fed into the HeadOverrideEvaluator.
    /// Example: "/headSharedLinear/Gemm"
    /// </summary>
    public readonly string InputLayerName;

    /// <summary>
    /// Name of the output from the input layer to be fed into the HeadOverrideEvaluator.
    /// Example: "/headSharedLinear/Gemm_output_0"
    /// </summary>
    public readonly string InputLayerOutputName;

    /// <summary>
    /// Evaluator function that returns override values for the head for a given input.
    /// </summary>
    public readonly Func<Half[], Half[]> HeadOverrideEvaluator;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="headType"></param>
    /// <param name="inputLayerName"></param>
    /// <param name="headOverrideEvaluator"></param>
    public NNEvaluatorHeadOverride(string id, 
                                   HeadTypeEnum headType, 
                                   string inputLayerName, 
                                   string inputLayerOutputName,
                                   Func<Half[], Half[]> headOverrideEvaluator)
    {
      ID = id;
      HeadType = headType;
      InputLayerName = inputLayerName;
      InputLayerOutputName = inputLayerOutputName;
      HeadOverrideEvaluator = headOverrideEvaluator;
    }

  }
}
