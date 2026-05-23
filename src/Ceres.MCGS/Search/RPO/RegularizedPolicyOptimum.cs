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
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.MCGS.Search.RPO;

/// <summary>
/// Unified primitive for the regularized policy optimization (RPO) framework of
/// Grill et al. "Monte-Carlo Tree Search as Regularized Policy Optimization" (2020).
///
/// Both reverse-KL and forward-KL/softmax variants optimize
///     y*(a; mu, q, lambda) = argmax_y { sum_a y(a) q(a)  -  lambda * D(mu, y) }
/// with the divergence choice D selected by RPORegularization.
///
/// One call to Solve produces three coupled outputs from the inputs (mu, q, lambda):
///   - y*       : the improved policy (visit-target distribution)
///   - q_fill   : the q vector with any NaN entries imputed by the inverse closed form
///   - v*       : the regularized state value  v* = sum_a y*(a) q_fill(a)
///
/// Closed forms:
///   ReverseKL          y(a) = lambda * mu(a) / (alpha - q(a)),   alpha by bisection
///   ForwardKLSoftmax   y(a) proportional to mu(a) * exp(q(a) / lambda)
///   ForwardKL inverse  q(a) = lambda * log(mu(a)) + C(s),   C(s) set by anchor
///
/// No outside Ceres.MCGS dependencies.  Stack-allocates scratch when the action set
/// is small; rents from ArrayPool otherwise.  TensorPrimitives are used where they
/// vectorize naturally; bisection of alpha is scalar.
/// </summary>
public static class RegularizedPolicyOptimum
{
  /// <summary>Threshold below which scratch buffers are stack-allocated.</summary>
  private const int STACKALLOC_MAX = 64;


  /// <summary>
  /// Solves the regularized policy improvement problem.
  /// </summary>
  /// <param name="mu">Prior policy probabilities (length n).  Need not sum to 1.</param>
  /// <param name="q">Per-action values (length n).  NaN entries are imputed.</param>
  /// <param name="lambda">Regularization strength.  Must be greater than or equal to 0 for reverse KL, greater than 0 for forward KL.</param>
  /// <param name="anchor">Determines the free intercept C(s) for forward-KL imputation.  Must be None for reverse KL.</param>
  /// <param name="regularization">ReverseKL (Grill) or ForwardKLSoftmax (Boltzmann).</param>
  /// <param name="yOut">Output buffer for y* (length greater than or equal to n).  May be empty if y* is not needed.</param>
  /// <param name="qFillOut">Output buffer for q_fill (length greater than or equal to n).  May be empty if not needed.</param>
  /// <param name="vStarOut">Output: v* = sum_a y*(a) q_fill(a).</param>
  /// <param name="options">Tuning knobs.  If the BisectionIterations field is 0, RPOOptions.Default is used.</param>
  /// <param name="nanFallbackQ">Fallback value used for NaN entries in q under reverse KL.  If itself NaN, the mean of the finite q's is used.</param>
  public static void Solve(ReadOnlySpan<double> mu,
                           ReadOnlySpan<double> q,
                           double lambda,
                           RPOAnchor anchor,
                           RPORegularization regularization,
                           Span<double> yOut,
                           Span<double> qFillOut,
                           out double vStarOut,
                           RPOOptions options = default,
                           double nanFallbackQ = double.NaN)
  {
    if (mu.Length != q.Length)
    {
      throw new ArgumentException("mu and q must have equal length.");
    }
    if (!yOut.IsEmpty && yOut.Length < mu.Length)
    {
      throw new ArgumentException("yOut is shorter than mu.");
    }
    if (!qFillOut.IsEmpty && qFillOut.Length < mu.Length)
    {
      throw new ArgumentException("qFillOut is shorter than mu.");
    }

    RPOOptions opts = options.BisectionIterations <= 0 ? RPOOptions.Default : options;

    int n = mu.Length;
    if (n == 0)
    {
      vStarOut = 0.0;
      return;
    }

    switch (regularization)
    {
      case RPORegularization.ReverseKL:
        if (anchor.Mode != RPOAnchorMode.None)
        {
          throw new ArgumentException("Reverse KL solve requires anchor mode = None (the closed form has no intercept freedom).");
        }
        if (!(lambda >= 0.0))
        {
          throw new ArgumentOutOfRangeException(nameof(lambda), "Reverse KL requires lambda >= 0.");
        }
        SolveReverseKL(mu, q, lambda, yOut, qFillOut, out vStarOut, opts, nanFallbackQ);
        return;

      case RPORegularization.ForwardKLSoftmax:
        if (!(lambda > 0.0))
        {
          throw new ArgumentOutOfRangeException(nameof(lambda), "Forward KL requires lambda > 0.");
        }
        SolveForwardKL(mu, q, lambda, anchor, yOut, qFillOut, out vStarOut, opts);
        return;

      default:
        throw new ArgumentOutOfRangeException(nameof(regularization));
    }
  }


  // ----------------------------------------------------------------------------
  // Reverse KL  (Grill et al.):  y(a) = lambda * mu(a) / (alpha - q(a))
  // ----------------------------------------------------------------------------

  private static void SolveReverseKL(ReadOnlySpan<double> mu,
                                     ReadOnlySpan<double> q,
                                     double lambda,
                                     Span<double> yOut,
                                     Span<double> qFillOut,
                                     out double vStarOut,
                                     RPOOptions opts,
                                     double nanFallbackQ)
  {
    int n = mu.Length;

    // Scratch buffers: muNorm, qFill, coeff, y (always needed even if outputs are empty).
    double[] rentedMuNorm = null;
    double[] rentedQFill = null;
    double[] rentedCoeff = null;
    double[] rentedY = null;

    Span<double> muNorm = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedMuNorm = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);
    Span<double> qFill  = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedQFill  = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);
    Span<double> coeff  = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedCoeff  = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);
    Span<double> yLocal = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedY      = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);

    try
    {
      NormalizeMu(mu, muNorm, opts.MinPriorProbability);

      // Pre-pass: determine fallback for NaN q entries.
      double fallback = ResolveFallback(q, nanFallbackQ, opts.ClampQToUnitInterval);

      double maxQEff = double.NegativeInfinity;
      bool anyPositiveCoeff = false;
      for (int i = 0; i < n; i++)
      {
        double qi = q[i];
        if (!IsFinite(qi))
        {
          qi = fallback;
        }
        if (opts.ClampQToUnitInterval)
        {
          qi = Clamp(qi, -1.0, 1.0);
        }
        qFill[i] = qi;

        double c = lambda * muNorm[i];
        if (c < 0.0 || !IsFinite(c))
        {
          c = 0.0;
        }
        coeff[i] = c;

        if (c > 0.0)
        {
          anyPositiveCoeff = true;
          if (qi > maxQEff)
          {
            maxQEff = qi;
          }
        }
      }

      // Degenerate cases: greedy on q if lambda is 0 or all coefficients are 0.
      if (!anyPositiveCoeff || lambda <= 1e-12)
      {
        GreedyOnQ(qFill, yLocal);
      }
      else if (!TrySolveAlphaBisection(coeff, qFill, maxQEff, opts.BisectionIterations,
                                       opts.BisectionResidualTol, out double alpha))
      {
        // Bisection failed: fall back to normalized prior (closest to greedy-prior choice).
        WriteNormalizedPrior(muNorm, yLocal);
      }
      else
      {
        double sumY = 0.0;
        for (int i = 0; i < n; i++)
        {
          double denom = alpha - qFill[i];
          double yi = denom > 0.0 ? coeff[i] / denom : 0.0;
          if (!IsFinite(yi) || yi < 0.0)
          {
            yi = 0.0;
          }
          yLocal[i] = yi;
          sumY += yi;
        }
        if (!(sumY > 0.0) || !IsFinite(sumY))
        {
          WriteNormalizedPrior(muNorm, yLocal);
        }
        else
        {
          // Two-pass renormalization to reduce drift (matches legacy ComputePosterior).
          double inv = 1.0 / sumY;
          double sumRenorm = 0.0;
          for (int i = 0; i < n; i++)
          {
            double v = yLocal[i] * inv;
            if (!IsFinite(v) || v < 0.0)
            {
              v = 0.0;
            }
            yLocal[i] = v;
            sumRenorm += v;
          }
          if (sumRenorm > 0.0 && IsFinite(sumRenorm))
          {
            double invRenorm = 1.0 / sumRenorm;
            for (int i = 0; i < n; i++)
            {
              yLocal[i] *= invRenorm;
            }
          }
        }
      }

      // Copy outputs and compute vStar.
      vStarOut = 0.0;
      for (int i = 0; i < n; i++)
      {
        vStarOut += yLocal[i] * qFill[i];
      }
      if (!yOut.IsEmpty)
      {
        yLocal.CopyTo(yOut);
      }
      if (!qFillOut.IsEmpty)
      {
        qFill.CopyTo(qFillOut);
      }
    }
    finally
    {
      if (rentedMuNorm != null) ArrayPool<double>.Shared.Return(rentedMuNorm);
      if (rentedQFill  != null) ArrayPool<double>.Shared.Return(rentedQFill);
      if (rentedCoeff  != null) ArrayPool<double>.Shared.Return(rentedCoeff);
      if (rentedY      != null) ArrayPool<double>.Shared.Return(rentedY);
    }
  }


  // ----------------------------------------------------------------------------
  // Forward KL (Boltzmann softmax):  y(a) proportional to mu(a) * exp(q(a) / lambda)
  // ----------------------------------------------------------------------------

  private static void SolveForwardKL(ReadOnlySpan<double> mu,
                                     ReadOnlySpan<double> q,
                                     double lambda,
                                     RPOAnchor anchor,
                                     Span<double> yOut,
                                     Span<double> qFillOut,
                                     out double vStarOut,
                                     RPOOptions opts)
  {
    int n = mu.Length;

    double[] rentedMuNorm = null;
    double[] rentedLogMu  = null;
    double[] rentedQFill  = null;
    double[] rentedY      = null;

    Span<double> muNorm = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedMuNorm = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);
    Span<double> logMu  = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedLogMu  = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);
    Span<double> qFill  = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedQFill  = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);
    Span<double> yLocal = n <= STACKALLOC_MAX ? stackalloc double[n] : (rentedY      = ArrayPool<double>.Shared.Rent(n)).AsSpan(0, n);

    try
    {
      // Normalize and floor mu (also enforce a hard floor so log is well-defined).
      const double LOG_FLOOR = 1e-10;
      double minPrior = opts.MinPriorProbability;
      double effectiveFloor = minPrior > LOG_FLOOR ? minPrior : LOG_FLOOR;
      NormalizeMu(mu, muNorm, effectiveFloor);

      // Vectorized log of mu.
      TensorPrimitives.Log(muNorm, logMu);

      // Determine C(s) from the anchor.
      double cIntercept = ResolveForwardKLIntercept(muNorm, logMu, anchor, lambda);

      // Build q_fill: preserve finite q, impute NaN via  q_i = lambda * log(mu_i) + C(s).
      for (int i = 0; i < n; i++)
      {
        double qi = q[i];
        if (!IsFinite(qi))
        {
          qi = lambda * logMu[i] + cIntercept;
        }
        if (opts.ClampQToUnitInterval)
        {
          qi = Clamp(qi, -1.0, 1.0);
        }
        qFill[i] = qi;
      }

      // Compute y_i proportional to mu_i * exp(q_i / lambda) with numerically-stable shift.
      // exp((q - qMax) / lambda) stays in [0, 1], avoiding overflow when lambda is small.
      double maxQ = double.NegativeInfinity;
      for (int i = 0; i < n; i++)
      {
        if (qFill[i] > maxQ)
        {
          maxQ = qFill[i];
        }
      }
      double invLambda = 1.0 / lambda;
      double sumW = 0.0;
      for (int i = 0; i < n; i++)
      {
        double w = muNorm[i] * Math.Exp((qFill[i] - maxQ) * invLambda);
        if (!IsFinite(w) || w < 0.0)
        {
          w = 0.0;
        }
        yLocal[i] = w;
        sumW += w;
      }
      if (sumW > 0.0 && IsFinite(sumW))
      {
        double inv = 1.0 / sumW;
        for (int i = 0; i < n; i++)
        {
          yLocal[i] *= inv;
        }
      }
      else
      {
        WriteNormalizedPrior(muNorm, yLocal);
      }

      vStarOut = 0.0;
      for (int i = 0; i < n; i++)
      {
        vStarOut += yLocal[i] * qFill[i];
      }

      if (!yOut.IsEmpty)
      {
        yLocal.CopyTo(yOut);
      }
      if (!qFillOut.IsEmpty)
      {
        qFill.CopyTo(qFillOut);
      }
    }
    finally
    {
      if (rentedMuNorm != null) ArrayPool<double>.Shared.Return(rentedMuNorm);
      if (rentedLogMu  != null) ArrayPool<double>.Shared.Return(rentedLogMu);
      if (rentedQFill  != null) ArrayPool<double>.Shared.Return(rentedQFill);
      if (rentedY      != null) ArrayPool<double>.Shared.Return(rentedY);
    }
  }


  /// <summary>
  /// Resolves C(s) for the forward-KL inverse identity  q_i = lambda * log(mu_i) + C(s).
  ///   - MatchValue : E_mu[q] = anchor.Value  =>  C = anchor.Value - lambda * E_mu[log mu] = anchor.Value + lambda * H(mu)
  ///   - MatchChild : q_anchorIdx = anchor.Value  =>  C = anchor.Value - lambda * log(mu_anchorIdx)
  ///   - None       : not supported for forward KL with NaN q (caller should pre-fill).
  ///                  Returns 0; if all q are finite, the value is unused.
  /// </summary>
  private static double ResolveForwardKLIntercept(ReadOnlySpan<double> muNorm,
                                                  ReadOnlySpan<double> logMu,
                                                  RPOAnchor anchor,
                                                  double lambda)
  {
    switch (anchor.Mode)
    {
      case RPOAnchorMode.MatchValue:
      {
        double entropy = 0.0;
        for (int i = 0; i < muNorm.Length; i++)
        {
          entropy -= muNorm[i] * logMu[i];
        }
        return anchor.Value + lambda * entropy;
      }

      case RPOAnchorMode.MatchChild:
      {
        if ((uint)anchor.Index >= (uint)muNorm.Length)
        {
          throw new ArgumentOutOfRangeException(nameof(anchor), "MatchChild anchor index is out of range.");
        }
        return anchor.Value - lambda * logMu[anchor.Index];
      }

      case RPOAnchorMode.None:
        // Caller has not specified an intercept.  Imputed slots (if any) will collapse
        // to lambda * log(mu_i), which is fine if no q is NaN.
        return 0.0;

      default:
        throw new ArgumentOutOfRangeException(nameof(anchor));
    }
  }


  // ----------------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------------

  /// <summary>
  /// Copies mu into muNorm, replacing NaN/negative entries with 0 and applying
  /// an optional floor, then normalizes to sum exactly 1.  If the total mass is
  /// 0, falls back to a uniform distribution.
  /// </summary>
  private static void NormalizeMu(ReadOnlySpan<double> mu, Span<double> muNorm, double minProb)
  {
    int n = mu.Length;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
      double v = mu[i];
      if (!IsFinite(v) || v < 0.0)
      {
        v = 0.0;
      }
      if (minProb > 0.0 && v < minProb)
      {
        v = minProb;
      }
      muNorm[i] = v;
      sum += v;
    }
    if (sum > 0.0 && IsFinite(sum))
    {
      double inv = 1.0 / sum;
      for (int i = 0; i < n; i++)
      {
        muNorm[i] *= inv;
      }
    }
    else
    {
      double uni = 1.0 / n;
      for (int i = 0; i < n; i++)
      {
        muNorm[i] = uni;
      }
    }
  }


  /// <summary>
  /// Selects the fallback value for NaN q entries.  If the caller supplied a finite
  /// nanFallbackQ, uses that (optionally clamped).  Otherwise uses the mean of finite
  /// q's; if no finite q exists, returns 0.
  /// </summary>
  private static double ResolveFallback(ReadOnlySpan<double> q, double nanFallbackQ, bool clampToUnit)
  {
    if (IsFinite(nanFallbackQ))
    {
      return clampToUnit ? Clamp(nanFallbackQ, -1.0, 1.0) : nanFallbackQ;
    }
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < q.Length; i++)
    {
      if (IsFinite(q[i]))
      {
        sum += q[i];
        count++;
      }
    }
    if (count == 0)
    {
      return 0.0;
    }
    double mean = sum / count;
    return clampToUnit ? Clamp(mean, -1.0, 1.0) : mean;
  }


  /// <summary>
  /// Solves  sum_i coeff_i / (alpha - qEff_i) = 1  for alpha &gt; maxQEff via bisection.
  /// Returns true on success.  Halts early once the residual |sum - 1| is below
  /// residualTol or once the iteration cap is reached.
  /// </summary>
  private static bool TrySolveAlphaBisection(ReadOnlySpan<double> coeff,
                                             ReadOnlySpan<double> qEff,
                                             double maxQEff,
                                             int iterations,
                                             double residualTol,
                                             out double alpha)
  {
    const double EPS = 1e-12;
    double aLo = maxQEff + EPS;
    double aHi = maxQEff + 1.0;
    if (!(aHi > aLo))
    {
      aHi = aLo + 1.0;
    }

    // Expand aHi until sum at aHi is below 1.
    double sumAtHi = SumCoeffOverAlphaMinusQ(coeff, qEff, aHi);
    int expand = 0;
    while (sumAtHi > 1.0 && expand < 64 && IsFinite(aHi))
    {
      double span = aHi - maxQEff;
      if (!(span > 0.0) || !IsFinite(span))
      {
        span = 1.0;
      }
      aHi = maxQEff + (span * 2.0);
      sumAtHi = SumCoeffOverAlphaMinusQ(coeff, qEff, aHi);
      expand++;
    }
    if (!(sumAtHi < 1.0) || !IsFinite(sumAtHi) || !IsFinite(aHi))
    {
      alpha = 0.0;
      return false;
    }

    // Bisect.
    for (int it = 0; it < iterations; it++)
    {
      double mid = 0.5 * (aLo + aHi);
      double s = SumCoeffOverAlphaMinusQ(coeff, qEff, mid);
      if (!IsFinite(s))
      {
        aLo = mid;
        continue;
      }
      if (Math.Abs(s - 1.0) < residualTol)
      {
        alpha = mid;
        return alpha > maxQEff && IsFinite(alpha);
      }
      if (s > 1.0)
      {
        aLo = mid;
      }
      else
      {
        aHi = mid;
      }
    }

    alpha = aHi;
    return alpha > maxQEff && IsFinite(alpha);
  }


  /// <summary>
  /// Scalar evaluation of  sum_i coeff_i / (alpha - qEff_i).  Assumes all coeff &gt;= 0
  /// and alpha &gt; max(qEff) so denominators are positive.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double SumCoeffOverAlphaMinusQ(ReadOnlySpan<double> coeff, ReadOnlySpan<double> qEff, double alpha)
  {
    double sum = 0.0;
    int n = coeff.Length;
    for (int i = 0; i < n; i++)
    {
      sum += coeff[i] / (alpha - qEff[i]);
    }
    return sum;
  }


  private static void GreedyOnQ(ReadOnlySpan<double> qEff, Span<double> yOut)
  {
    int n = qEff.Length;
    int best = 0;
    double bestQ = qEff[0];
    for (int i = 1; i < n; i++)
    {
      if (qEff[i] > bestQ)
      {
        bestQ = qEff[i];
        best = i;
      }
    }
    for (int i = 0; i < n; i++)
    {
      yOut[i] = 0.0;
    }
    yOut[best] = 1.0;
  }


  private static void WriteNormalizedPrior(ReadOnlySpan<double> muNorm, Span<double> yOut)
  {
    // muNorm is already normalized to sum 1 (or uniform if degenerate).
    muNorm.CopyTo(yOut);
  }


  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static bool IsFinite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double Clamp(double x, double lo, double hi)
  {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }
}
