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
using System.Text;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  [Serializable]
  public record NNEvaluatorPerformanceStats
  {
    /// <summary>
    /// ID of neural network
    /// </summary>
    public string NNNetworkID { init; get; }

    /// <summary>
    /// Type of NN evaluator
    /// </summary>
    public Type EvaluatorType { init; get; }

    /// <summary>
    /// Index of device (GPU) used
    /// </summary>
    public int GPUID { init; get; }

    /// <summary>
    /// Estimated number of nodes per second with batch size of 1
    /// </summary>
    public float SingletonNPS { init; get; }

    /// <summary>
    /// Estimated number of nodes per second with large batch size
    /// </summary>
    public float BigBatchNPS { init; get; }

    /// <summary>
    /// Optional sequence of batch size at which nodes per second 
    /// reaches a peak before a step down due hardware characteristic)
    /// </summary>
    public int[] Breaks { init; get; }


    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      StringBuilder breaksStr = new StringBuilder("[");
      if (Breaks != null)
      {
        for (int i = 0; i < Breaks.Length; i++)
        {
          if (i > 0) breaksStr.Append(",");
          breaksStr.Append(Breaks[i]);
        }
      }
      breaksStr.Append("]");

      return $"<NNEvalNetPerformanceStats on GPU {GPUID} SingletonNPS={(int)SingletonNPS} BatchNPS={(int)BigBatchNPS} Breaks={breaksStr}>";
    }
  }
}
