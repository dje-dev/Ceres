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
  public static class StatsEnumerable
  {
    /// <summary>
    /// Returns Pearson correlation coeffient between two IEnumerables of double.
    /// </summary>
    /// <param name="xs"></param>
    /// <param name="ys"></param>
    /// <returns></returns>
    public static double Correlation(IEnumerable<double> xs, IEnumerable<double> ys)
    {
      double sx = 0.0;
      double sy = 0.0;
      double sxx = 0.0;
      double syy = 0.0;
      double sxy = 0.0;

      int n = 0;

      IEnumerator<double> enumeratorX = xs.GetEnumerator();
      IEnumerator<double> enumeratorY = ys.GetEnumerator();

      foreach (double x in xs)
      {
        if (!enumeratorY.MoveNext()) throw new Exception("IEnumerable not of same size");
        double y = enumeratorY.Current;
        n++;

        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
      }

      // Covariation.
      double cov = sxy / n - sx * sy / n / n;

      // Standard error of x.
      double sigmax = System.Math.Sqrt(sxx / n - sx * sx / n / n);

      // Standard error of y.
      double sigmay = System.Math.Sqrt(syy / n - sy * sy / n / n);

      // Correlation is just a normalized covariation.
      return cov / sigmax / sigmay;
    }


    /// <summary>
    /// Returns Pearson correlation coeffient between two IEnumerables of floats..
    /// </summary>
    /// <param name="xs"></param>
    /// <param name="ys"></param>
    /// <returns></returns>
    public static float Correlation(IEnumerable<float> xs, IEnumerable<float> ys)
    {
      //TODO: check here that arrays are not null, of the same length etc

      double sx = 0.0;
      double sy = 0.0;
      double sxx = 0.0;
      double syy = 0.0;
      double sxy = 0.0;

      int n = 0;

      IEnumerator<float> enumeratorX = xs.GetEnumerator();
      IEnumerator<float> enumeratorY = ys.GetEnumerator();

      foreach (float x in xs)
      {
        if (!enumeratorY.MoveNext()) throw new Exception("IEnumerable not of same size");
        float y = enumeratorY.Current;
        n++;

        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
      }

      // Covariation.
      double cov = sxy / n - sx * sy / n / n;

      // Standard error of x.
      double sigmax = System.Math.Sqrt(sxx / n - sx * sx / n / n);

      // Standard error of y.
      double sigmay = System.Math.Sqrt(syy / n - sy * sy / n / n);

      // Correlation is just a normalized covariation.
      return (float)(cov / sigmax / sigmay);
    }

  }

}
