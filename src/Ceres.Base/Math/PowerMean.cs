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

namespace Ceres.Base.Math
{
  /// <summary>
  /// Online calculator for power mean over a series of weighted values
  /// </summary>
  public struct PowerMeanCalculator
  {
    /// <summary>
    /// The power coefficient.
    /// </summary>
    public readonly double P;

    /// <summary>
    /// The total number of values that will possibly be added.
    /// </summary>
    public readonly int N;

    int numAdded;
    double accumulator;

    /// <summary>
    /// Constructs a power mean calculator with a specified power coefficient and total count.
    /// </summary>
    /// <param name="n">the total number of values that will be added</param>
    /// <param name="p">the power coefficient to be used for the mean</param>
    public PowerMeanCalculator(int n, double p)
    {
      N = n;
      P = p;
      accumulator = 0;
      numAdded = 0;
    }

    /// <summary>
    /// Adds a specified value with a specified count (weight).
    /// </summary>
    /// <param name="value"></param>
    /// <param name="n"></param>
    public void AddValue(double value, int n)
    {
      double frac = n / (float)N;
      accumulator += frac * Pow(value, P);
      numAdded+=n;
    }


    /// <summary>
    /// Returns the power mean over all the added data.
    /// </summary>
    public double PowerMean
    {
      get
      {
        if (numAdded != N) throw new Exception($"Incorrect number of items added: {numAdded} vs. expected {N}");
        return Pow(accumulator, 1.0 / P);
      }
    }
  }
}
