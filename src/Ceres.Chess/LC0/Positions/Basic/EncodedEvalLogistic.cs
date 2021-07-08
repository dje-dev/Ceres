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
using Ceres.Base.Math;

#endregion

namespace Ceres.Chess.LC0.Positions
{
  /// <summary>
  /// Represents the scalar evaluation of a position by a neural network
  /// (as a logistic indicating probability of "winning minus losing"
  /// and also conversion back and for from centipawns scale 
  /// typically reported by chess programs.
  /// </summary>
  public readonly struct EncodedEvalLogistic
  {
    /// <summary>
    /// The actual raw logistic value from the neural network.
    /// </summary>
    public readonly float LogisticValue;

    public static EncodedEvalLogistic FromLogistic(float logisticValue) => new EncodedEvalLogistic(logisticValue);
    public static EncodedEvalLogistic FromCentipawn(float centipawnValue) => new EncodedEvalLogistic(CentipawnToLogistic(centipawnValue));

    public float CentipawnValue => LogisticToCentipawn(LogisticValue);

    /// <summary>
    /// Based on LC0 definition "centipawn."
    /// </summary>
    /// <param name="wl"></param>
    /// <returns></returns>
    public static float WinLossToCentipawn(float wl)
    {
      if (wl <= -1)
      {
        return -9999;
      }
      else if (wl >= 1)
      {
        return 9999;
      }
      else
      {
        // This function is not well behaved outside [-1, 1].
        return StatUtils.Bounded(CENTIPAWN_MULT * MathF.Tan(CENTIPAWN_TAN_MULT * wl), -9999, 9999);
      }
    }

    /// <summary>
    /// Outer multiplier on in conversion formula.
    /// </summary>
    const float  CENTIPAWN_MULT = 90;

    /// <summary>
    /// Multiplier on value passed to Tan function.
    /// </summary>
    const float CENTIPAWN_TAN_MULT = 1.5637541897f;

    /// <summary>
    /// Converts logistic value to centipawn.
    /// </summary>
    /// <param name="logistic"></param>
    /// <returns></returns>
    public static float LogisticToCentipawn(float logistic)
    {
      // Make sure the logistic is within a valid range [-1, 1]
      float bounded = MathHelpers.Bounded(logistic, -1, 1);
      float centipawn =  (float)Math.Round(CENTIPAWN_MULT * Math.Tan(CENTIPAWN_TAN_MULT * bounded), 2);

      // Avoid showing extreme centipawn values outside a reasonable range
      const float MAX_CENTIPAWN = 9_999;
      return MathHelpers.Bounded(centipawn, -MAX_CENTIPAWN, MAX_CENTIPAWN);
    }

    /// <summary>
    /// Inversion of logistic to centipawn function.
    /// </summary>
    /// <param name="centipawnValue"></param>
    /// <returns></returns>
    static float CentipawnToLogisticUnbounded(float centipawnValue) => (float)MathF.Atanh((1.0f / CENTIPAWN_TAN_MULT) * MathF.Atan((float)(centipawnValue / CENTIPAWN_MULT)));


    /// <summary>
    /// Converts centipawn to equivalent logistic value.
    /// </summary>
    /// <param name="centipawnValue"></param>
    /// <returns></returns>
    public static float CentipawnToLogistic(float centipawnValue)
    {
      float raw = CentipawnToLogisticUnbounded(centipawnValue);

      // Set bound (values convered from another program's centipawn evaluation could sometimes fall outside [-1.0, 1.0]
      if (raw < -1)
      {
        return -1;
      }
      else if (raw > 1)
      {
        return 1;
      }
      else
      {
        return raw;
      }
    }

    #region Static helpers

    public static FP16[] ToLogisticsArray(EncodedEvalLogistic[] evals)
    {
      FP16[] ret = new FP16[evals.Length];
      for (int i = 0; i < ret.Length; i++)
        ret[i] = (FP16)evals[i].LogisticValue;
      return ret;
    }

    public static float[] ToCentiapawnsArray(EncodedEvalLogistic[] evals)
    {
      float[] ret = new float[evals.Length];
      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = (float)evals[i].CentipawnValue;
      }

      return ret;
    }

    public static EncodedEvalLogistic[] FromLogisticArray(FP16[] logistics)
    {
      EncodedEvalLogistic[] ret = new EncodedEvalLogistic[logistics.Length];
      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = FromLogistic(logistics[i]);
      }

      return ret;
    }

    #endregion

    #region Internals
    public EncodedEvalLogistic(float val) => LogisticValue = val;

#endregion

  }
}
