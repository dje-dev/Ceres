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
using System.Runtime.CompilerServices;

namespace Ceres.MCGS.Search.RPO;

/// <summary>
/// Optimiser with reverse-KL regularisation
/// ( component-wise scale factors <paramref name="scalingFactors"/> ).
/// </summary>
public static class PolicyOptimizerReverse
{
  public static Span<float> Optimize(float epsilonTol, Span<float> Q, float lambdaN, Span<float> P, Span<float> scalingFactors)
  {
    int n = Q.Length;

    //  basic validation
    if (n == 0)
    {
      throw new ArgumentException("Q must contain at least one element.", nameof(Q));
    }
    if (P.Length != n)
    {
      throw new ArgumentException("Q and P must have identical lengths.");
    }
    if (!scalingFactors.IsEmpty && scalingFactors.Length != n)
    {
      throw new ArgumentException("scalingFactors must be empty or the same length as Q.");
    }
    if (lambdaN < 0.0f)
    {
      throw new ArgumentOutOfRangeException(nameof(lambdaN), "lambdaN must be non-negative.");
    }

    //------------------------------------------------------------------
    //  greedy shortcut when lambda approaches zero 0  (KL carries no weight)
    //------------------------------------------------------------------
    if (lambdaN <= float.Epsilon)
    {
      int best = 0;
      float bestQ = Q[0];
      for (int i = 1; i < n; ++i)
      {
        if (Q[i] > bestQ)
        {
          best = i;
          bestQ = Q[i];
        }
      }
      float[] greedy = new float[n];
      greedy[best] = 1.0f;
      return greedy;
    }

    //------------------------------------------------------------------
    //  promote to double for stability during exponentiation
    //------------------------------------------------------------------
    Span<double> Qd = stackalloc double[n];
    Span<double> Pd = stackalloc double[n];
    Span<double> Sd = stackalloc double[n];

    for (int i = 0; i < n; ++i)
    {
      Qd[i] = Q[i];
      Pd[i] = P[i];
      Sd[i] = scalingFactors.IsEmpty ? 1.0 : scalingFactors[i];
      if (Sd[i] <= 0.0)
      {
        throw new ArgumentOutOfRangeException(nameof(scalingFactors), "All scale factors must be positive.");
      }
    }

    //------------------------------------------------------------------
    //  scalar root-finding
    //------------------------------------------------------------------
    double muLo = 0.0;
    double gLo = ComputeG(muLo, Qd, Pd, Sd, lambdaN);

    // ensure we bracket 1 : g increases strictly with mu
    double step = 1.0;
    double muHi, gHi;

    if (gLo < 1.0)  // need larger mu
    {
      muHi = muLo + step;
      gHi = ComputeG(muHi, Qd, Pd, Sd, lambdaN);
      while (gHi < 1.0)
      {
        muLo = muHi;
        gLo = gHi;
        step *= 2.0;
        muHi = muLo + step;
        gHi = ComputeG(muHi, Qd, Pd, Sd, lambdaN);
      }
    }
    else  // gLo > 1 --> need smaller mu
    {
      muHi = muLo;
      gHi = gLo;
      muLo = muHi - step;
      gLo = ComputeG(muLo, Qd, Pd, Sd, lambdaN);
      while (gLo > 1.0)
      {
        muHi = muLo;
        gHi = gLo;
        step *= 2.0;
        muLo = muHi - step;
        gLo = ComputeG(muLo, Qd, Pd, Sd, lambdaN);
      }
    }

    // bisection
    double tol = Math.Max(1e-9, epsilonTol);
    int iterationCount = 0;
    while (muHi - muLo > tol)
    {
      double muMid = 0.5 * (muLo + muHi);
      double gMid = ComputeG(muMid, Qd, Pd, Sd, lambdaN);

      if (gMid < 1.0)
      {
        muLo = muMid;
      }
      else
      {
        muHi = muMid;
      }
      iterationCount++;
    }

    double mu = 0.5 * (muLo + muHi);

    //  compute pi
    float[] pi = new float[n];
    double gMu = ComputeG(mu, Qd, Pd, Sd, lambdaN);   // normalising constant

    for (int i = 0; i < n; ++i)
    {
      double exponent = (Qd[i] + mu) / (lambdaN * Sd[i]) - 1.0;
      double num = Pd[i] * Math.Exp(exponent);
      pi[i] = (float)(num / gMu);
    }

    return pi;
  }



  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double ComputeG
  (
      double mu,
      Span<double> Q,
      Span<double> P,
      Span<double> S,
      double lambdaN
  )
  {
    double sum = 0.0;
    for (int i = 0; i < Q.Length; ++i)
    {
      double exponent = (Q[i] + mu) / (lambdaN * S[i]) - 1.0;
      sum += P[i] * Math.Exp(exponent);
    }
    return sum;
  }
}
