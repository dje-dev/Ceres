#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;
using System.Collections.Generic;

#endregion

namespace Ceres.Base.Math
{
  /// <summary>
  /// Computes the best fitting (by sum of squares errors)
  /// quadratic curve (parabola) to a set of data points.
  /// 
  /// The optimal fit coefficients are recalculated 
  /// upon each call to FitY, each time reflecting the 
  /// current set of samples added.
  /// 
  /// Originally based on code from https://www.codeproject.com/Articles/63170/Least-Squares-Regression-for-Quadratic-Curve-Fitti.
  /// </summary>
  public class QuadraticRegression
  {
    /// <summary>
    /// Set of samples to be fit.
    /// </summary>
    public readonly List<(double, double)> Samples = new();

    /// <summary>
    /// Adds specified sample.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    public void AddSample(double x, double y) => Samples.Add((x, y));


    /// <summary>
    /// Returns the fit value (estimate) given a specified X.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public double FitY(double x)
    {
      return QuadraticTerm() * System.Math.Pow(x, 2) + LinearTerm() * x + Intercept();
    }


    /// <summary>
    /// Returns the coefficent in the quadratic term.
    /// </summary>
    /// <returns></returns>
    public double QuadraticTerm()
    {
      double s40 = SX4();
      double s30 = SXCubed();
      double s20 = SXSquared();
      double s10 = SX();
      double s00 = Samples.Count;


      double s21 = SXSquaredY();
      double s11 = SXY();
      double s01 = SY();

      return (s21 * (s20 * s00 - s10 * s10) -
              s11 * (s30 * s00 - s10 * s20) +
              s01 * (s30 * s10 - s20 * s20))
              /
              (s40 * (s20 * s00 - s10 * s10) -
               s30 * (s30 * s00 - s10 * s20) +
               s20 * (s30 * s10 - s20 * s20));
    }


    /// <summary>
    /// Returns the coefficent in the linear term.
    /// </summary>
    /// <returns></returns>
    public double LinearTerm()
    {
      double s40 = SX4();
      double s30 = SXCubed();
      double s20 = SXSquared();
      double s10 = SX();
      double s00 = Samples.Count;

      double s21 = SXSquaredY();
      double s11 = SXY();
      double s01 = SY();

      return (s40 * (s11 * s00 - s01 * s10) -
              s30 * (s21 * s00 - s01 * s20) +
              s20 * (s21 * s10 - s11 * s20))
              /
              (s40 * (s20 * s00 - s10 * s10) -
               s30 * (s30 * s00 - s10 * s20) +
               s20 * (s30 * s10 - s20 * s20));
    }


    /// <summary>
    /// Returns the intercept term.
    /// </summary>
    /// <returns></returns>
    public double Intercept()
    {
      double s40 = SX4();
      double s30 = SXCubed();
      double s20 = SXSquared();
      double s10 = SX();
      double s00 = Samples.Count;

      double s21 = SXSquaredY();
      double s11 = SXY();
      double s01 = SY();

      return (s40 * (s20 * s01 - s10 * s11) -
              s30 * (s30 * s01 - s10 * s21) +
              s20 * (s30 * s11 - s20 * s21))
              /
              (s40 * (s20 * s00 - s10 * s10) -
               s30 * (s30 * s00 - s10 * s20) +
               s20 * (s30 * s10 - s20 * s20));
    }


    /// <summary>
    /// Returns the R-squared of the fit.
    /// </summary>
    /// <returns></returns>
    public double RSquared() => 1 - SumSquaredErrors() / SumSquaresTot();


    #region Internal helpers

    double Accumulate(Func<(double,double), double> func)
    {
      double acc = 0;
      foreach ((double, double) item in Samples)
      {
        acc += func(item);
      }
      return acc;
    }

    double SX() => Accumulate(item => item.Item1);
    double SY() => Accumulate(item => item.Item2);

    double SXSquared() => Accumulate(item => System.Math.Pow(item.Item1, 2));
    double SXCubed() => Accumulate(item => System.Math.Pow(item.Item1, 3));
    double SX4() => Accumulate(item => System.Math.Pow(item.Item1, 4));

    double SXY() => Accumulate(item => item.Item1 * item.Item2);
    double SXSquaredY() => Accumulate(item => System.Math.Pow(item.Item1, 2) * item.Item2);

    double SumSquaresTot() => Accumulate(item => System.Math.Pow(item.Item2 - YAvg(), 2));

    double SumSquaredErrors() => Accumulate(item => System.Math.Pow(item.Item2 - FitY(item.Item1), 2));

    double YAvg()
    {
      double y_tot = 0;
      foreach (var item in Samples)
      {
        y_tot += item.Item2;
      }
      return y_tot / Samples.Count;
    }

    #endregion
  }
}
