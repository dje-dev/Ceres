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

namespace Ceres.Base.Algorithms
{
  /// <summary>
  /// Static utility class for the bisection minimization algorithm.
  /// </summary>
  public static class Bisection
  {
    /// <summary>
    /// Finds the minimum of a function using the bisection method,
    /// for a given evaluator and error tolerance.
    /// </summary>
    /// <param name="function">function to be minimized (one dimensional y=f(x))</param>
    /// <param name="lowerBound">minimum value in range to be evaluated</param>
    /// <param name="upperBound">maximum value in range to be evaluated</param>
    /// <param name="tolerance">epsilon tolerance at which iterations is terminated</param>
    /// <param name="verbose">if verbose debugging information should be output to console</param>
    /// <returns></returns>
    public static (float OptimalX, float OptimalY)
      FindMinimum(Func<float, float> function,
                  float lowerBound,
                  float upperBound,
                  float tolerance = 0.001f,
                  bool verbose = false)
    {
      if (lowerBound >= upperBound)
      {
        throw new ArgumentException("Lower bound must be less than upper bound");
      } 

      float bestValue = (lowerBound + upperBound) / 2;
      float minFunctionValue = float.MaxValue;

      while ((upperBound - lowerBound) > tolerance)
      {
        float diff = (upperBound - lowerBound) / 3;
        float mid1 = lowerBound + diff;
        float mid2 = upperBound - diff;

        float functionValue1 = function(mid1);
        float functionValue2 = function(mid2);

        if (functionValue1 < functionValue2)
        {
          upperBound = mid2;
        }
        else
        {
          lowerBound = mid1;
        }

        float currentBestFunctionValue = System.Math.Min(functionValue1, functionValue2);
        if (currentBestFunctionValue < minFunctionValue)
        {
          minFunctionValue = currentBestFunctionValue;
          bestValue = (functionValue1 < functionValue2) ? mid1 : mid2;
        }
      }

      if (verbose)
      {
        Console.WriteLine($"Best Value: {bestValue}, Min Function Value: {minFunctionValue}");
      }
      return (bestValue, minFunctionValue);
    }
  }
}
