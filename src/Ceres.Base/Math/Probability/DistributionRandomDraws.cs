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
using static System.Math;

#endregion

namespace Ceres.Base.Math.Probability
{
  /// <summary>
  /// Set of static helper methods relating to common distributions (such as Gaussian)
  /// </summary>
  public static class DistributionRandomDraws
  {
    [ThreadStatic]
    private static System.Random rand;

    static double GetRand()
    {
      if (rand == null) rand = new();
      return rand.NextDouble();
    }

    /// <summary>
    /// Returns a draw from a uniform distribution with specified parameters.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public static double UniformRandomDraw(double min = 0, double max = 1) => (min + (max - min) * GetRand());

    /// <summary>
    /// Returns a draw from a Gaussian distribution with specified parameters.
    /// </summary>
    /// <param name="mu"></param>
    /// <param name="stdDev"></param>
    /// <returns></returns>
    public static double GaussianRandomDraw(double mu = 0, double stdDev = 1)
    {
      return stdDev
           * Sqrt(-2 * Log(UniformRandomDraw()))
           * Cos(2 * PI * UniformRandomDraw()) + mu;
    }


    /// <summary>
    /// Returns a draw from a gamma disitribution with specified parameters.
    /// Uses algorithm of Marsaglia and Tsang.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="scale"></param>
    /// <returns></returns>
    public static double GammaRandomDraw(double shape, double scale)
    {
      if (shape < 1)
      {
        // Resolve recursively
        return (GammaRandomDraw(shape + 1, scale) * Pow(UniformRandomDraw(), 1 / shape));
      }
      else
      {
        double d = shape - 1.0 / 3.0;
        double c = 1 / Sqrt(9 * d);
        double z, v, p;

        do
        {
          z = GaussianRandomDraw();
          v = Pow(1 + c * z, 3);
          p = 0.5 * Pow(z, 2) + d - d * v + d * Log(v);
        } while ((z < -1 / c) || (Log(UniformRandomDraw()) > p));

        return (d * v) / scale;
      }
    }
  }
}
