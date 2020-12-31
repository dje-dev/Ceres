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

using Chess.Ceres.NNEvaluators;
using System;

#endregion

namespace Ceres.Chess.NNEvaluators.Defs
{
  /// <summary>
  /// Defines a network to be used in NN evaluation.
  /// </summary>
  [Serializable]
  public record NNEvaluatorNetDef
  {
    /// <summary>
    /// Type of network.
    /// </summary>
    public NNEvaluatorType Type { get; init; }

    /// <summary>
    /// Identifier string for network.
    /// </summary>
    public string NetworkID { get; init; }

    /// <summary>
    /// Numerical precision to use in evaluating network.
    /// </summary>
    public NNEvaluatorPrecision Precision { get; init; }


    public NNEvaluatorNetDef(string networkID, NNEvaluatorType type, 
                             NNEvaluatorPrecision precision = NNEvaluatorPrecision.FP16)
    {
      Type = type;
      NetworkID = networkID;
      Precision = precision;
    }


    public override string ToString()
    {
      return $"<NNEvaluatorDef {Type} {NetworkID} {Precision}>";
    }
  }
}
