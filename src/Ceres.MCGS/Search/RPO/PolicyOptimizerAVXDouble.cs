#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Ceres.MCGS.Search.RPO;

public static class PolicyOptimizerAVXDouble
{
  public static Span<float> Optimize(float epsilonTol, Span<float> Q, float lambdaN, Span<float> P)
  {
    if (Q.Length == 0)
    {
      throw new ArgumentException("Vector Q must contain at least one element.", nameof(Q));
    }
    if (Q.Length != P.Length)
    {
      throw new ArgumentException("Q and P must have identical lengths.");
    }
    if (lambdaN < 0.0f)
    {
      throw new ArgumentOutOfRangeException(nameof(lambdaN), "lambdaN must be non-negative.");
    }

    int actionCount = Q.Length;

    // Degenerate case: lambda close to zero
    if (lambdaN <= float.Epsilon)
    {
      int best = 0;
      Single bestQ = Q[0];
      for (int i = 1; i < actionCount; ++i)
      {
        if (Q[i] > bestQ)
        {
          best = i;
          bestQ = Q[i];
        }
      }
      Single[] greedy = new Single[actionCount];
      greedy[best] = 1.0f;
      return greedy;
    }

    double maxQ = double.NegativeInfinity;
    for (int i = 0; i < actionCount; ++i)
    {
      maxQ = Math.Max(maxQ, Q[i]);
    }

    double lower = maxQ + 1e-6;                    // f(lower)   > 0
    double upper = lower * 2.0;                    // expand until f(upper) < 0

    while (ComputeF(upper, Q, P, lambdaN) > 0.0)
    {
      upper *= 2.0;
    }

    // Bisection
    double mid = 0.0;
    double eps = Math.Max(1e-12, epsilonTol);
    while (upper - lower > eps)
    {
      mid = 0.5 * (lower + upper);
      if (ComputeF(mid, Q, P, lambdaN) > 0.0)
      {
        lower = mid;
      }
      else
      {
        upper = mid;
      }
    }
    double mu = 0.5 * (lower + upper);

    if (mu < 0)
    {
      // Rarely (for numerical reasons) the mu be computed as (slightly) negative.
      // This would cause problems in the next step.
      mu = 0.01f;
    }

    float[] piArr = new float[actionCount];

    Span<double> tmp = stackalloc double[actionCount];
    double sumCheck = 0.0;

    for (int i = 0; i < actionCount; ++i)
    {
      double denom = mu - Q[i];
      Debug.Assert(denom > 0.0, "Denominator must be positive");
      double value = lambdaN * P[i] / denom;
      tmp[i] = value;
      sumCheck += value;
    }
    NormaliseAndCopy(tmp, piArr, sumCheck);


    return piArr;
  }


  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double ComputeF(double mu, Span<float> Q, Span<float> P, double lambdaN)
  {
    double sum = 0.0;
    for (int i = 0; i < Q.Length; ++i)
    {
      sum += lambdaN * P[i] / (mu - Q[i]);
    }
    return sum - 1.0;
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static void NormaliseAndCopy
  (
      Span<double> source,
      Single[] destination,
      double sum
  )
  {
    double inv = 1.0 / sum;
    for (int i = 0; i < destination.Length; ++i)
    {
      destination[i] = (Single)(source[i] * inv);
    }
  }
}
