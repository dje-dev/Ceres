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

namespace Ceres.Chess
{
  /// <summary>
  /// Utility class for calculating the scaling factor for WDL values to achieve a desired V value.
  /// </summary>
  public static class WDLScalingCalculator
  {
    /// <summary>
    /// Returns the WDL values that will result in the given V value when rescaled.
    /// </summary>
    /// <param name="w"></param>
    /// <param name="l"></param>
    /// <param name="targetV"></param>
    /// <returns></returns>
    public static (float w, float d, float l) RescaledWDLToDesiredV(float w, float l, float targetV)
    {
      // Handle exact win/draw/loss as special cases.
      if (targetV > 0.998f)
      {
        return (1, 0, 0);
      }
      else if (targetV < -0.998f)
      {
        return (0, 0, 1);
      }
      else if (targetV == 0)
      {
        return (0, 1, 0); // TODO: consider if there is a better way to distribute some weight to win/loss?
      }

      float rescaleFactor = CalcWLScalingFactor(w, l, targetV);

      return (w * rescaleFactor, 1 - (w * rescaleFactor + l / rescaleFactor), l / rescaleFactor);
    }


    static float CalcWLScalingFactor(float initialW, float initialL, float targetV)
    {
      // Coefficients for the quadratic equation: a*S^2 + b*S + c = 0
      float a = initialW;
      float b = -targetV;
      float c = -initialL;

      // Calculate the discriminant.
      float discriminant = b * b - 4 * a * c;

      if (discriminant < 0)
      {
        throw new Exception("No real solutions exist for the given parameters.");
      }

      // Calculate both roots
      float sqrtDiscriminant = MathF.Sqrt(discriminant);
      float S1 = (-b + sqrtDiscriminant) / (2 * a);
      float S2 = (-b - sqrtDiscriminant) / (2 * a);

      // Choose the positive, meaningful root
      float S = S1 > 0 ? S1 : S2;

      if (S <= 0)
      {
        throw new Exception("No positive scaling factor found.");
      }

      return S;
    }

  }

}
