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
using Ceres.Base.DataTypes;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// Activations of certain inner layers.
  /// </summary>
  public class NNEvaluatorResultActivations
  {
    /// <summary>
    /// Activations of neurons in first fully connected layer of value head.
    /// </summary>
    public readonly FP16[] ValueFC1Activations;


    /// <summary>
    /// Activations of neurons in second fully connected layer of value head.
    /// </summary>
    public readonly FP16[] ValueFC2Activations;


    /// <summary>
    /// Constructor (from float).
    /// </summary>
    /// <param name="valueFC1Activations"></param>
    /// <param name="valueFC2Activations"></param>
    public NNEvaluatorResultActivations(int index, float[,] valueFC1Activations, float[,] valueFC2Activations)
    {
      if (valueFC1Activations != null)
      {
        ValueFC1Activations = new FP16[valueFC1Activations.GetLength(1)];
        for (int i=0;i< ValueFC1Activations.Length;i++)
        {
          ValueFC1Activations[i] = (FP16)valueFC1Activations[index, i];
        }
      }

      if (valueFC2Activations != null)
      {
        ValueFC2Activations = new FP16[valueFC2Activations.GetLength(1)];
        for (int i = 0; i < ValueFC2Activations.Length; i++)
        {
          ValueFC2Activations[i] = (FP16)valueFC2Activations[index, i];
        }
      }
    }


    /// <summary>
    /// Constructor (from FP16).
    /// </summary>
    /// <param name="valueFC1Activations"></param>
    /// <param name="valueFC2Activations"></param>
    public NNEvaluatorResultActivations(int index, FP16[,] valueFC1Activations, FP16[,] valueFC2Activations)
    {
      if (valueFC1Activations != null)
      {
        ValueFC1Activations = new FP16[valueFC1Activations.GetLength(1)];
        for (int i = 0; i < ValueFC1Activations.Length; i++)
        {
          ValueFC1Activations[i] = valueFC1Activations[index, i];
        }
      }

      if (valueFC2Activations != null)
      {
        ValueFC2Activations = new FP16[valueFC1Activations.GetLength(1)];
        for (int i = 0; i < ValueFC2Activations.Length; i++)
        {
          ValueFC2Activations[i] = valueFC2Activations[index, i];
        }
      }
    }

  }
}
