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
/// No outside Ceres.MCGS dependencies.  Scratch lives in per-thread reusable buffers
/// (the action set is bounded by MAX_ACTIONS, so nothing is ever heap-rented per call);
/// TensorPrimitives are used where they vectorize naturally; bisection of alpha is scalar.
/// </summary>
public static class RegularizedPolicyOptimum
{
  /// <summary>
  /// Maximum number of actions (children) the solver ever sees in a single call.
  /// Equal to the engine's MAX_CHILDREN; the per-thread scratch buffers below are
  /// sized to this so the live action set always fits without renting.
  /// </summary>
  private const int MAX_ACTIONS = 64;

  // Per-thread scratch reused across Solve calls in place of per-call stackalloc/ArrayPool.
  // Each buffer is lazily allocated once per thread to MAX_ACTIONS and then sliced to the
  // live action count n on every call.  [ThreadStatic] gives each search thread its own set
  // (no cross-thread sharing), mirroring CBGPUCTScoreCalc / PUCTScoreCalcVector.  The reverse-
  // and forward-KL paths never run concurrently on one thread, so the three buffers they share
  // (muNorm, qFill, y) are safe to reuse between them.  Correctness note: every element in
  // [0, n) is fully written before it is read on each call, so the loss of stackalloc's implicit
  // zero-initialization is immaterial; slicing to exactly n also preserves the length validations
  // and the vStar dot-product semantics.
  [ThreadStatic] private static double[] bufferMuNorm;
  [ThreadStatic] private static double[] bufferQFill;
  [ThreadStatic] private static double[] bufferCoeff;
  [ThreadStatic] private static double[] bufferY;
  [ThreadStatic] private static double[] bufferNewtD;
  [ThreadStatic] private static double[] bufferNewtRatio;
  [ThreadStatic] private static double[] bufferLogMu;


  /// <summary>
  /// Solves the regularized policy improvement problem.
  /// </summary>
  /// <param name="mu">Prior policy probabilities (length n).  Need not sum to 1.</param>
  /// <param name="q">Per-action values (length n).  NaN entries are imputed.</param>
  /// <param name="lambda">Regularization strength (scalar).  Must be greater than or equal to 0 for reverse KL, greater than 0 for forward KL.</param>
  /// <param name="anchor">Determines the free intercept C(s) for forward-KL imputation.  Must be None for reverse KL.</param>
  /// <param name="regularization">ReverseKL (Grill) or ForwardKLSoftmax (Boltzmann).</param>
  /// <param name="yOut">Output buffer for y* (length greater than or equal to n).  May be empty if y* is not needed.</param>
  /// <param name="qFillOut">Output buffer for q_fill (length greater than or equal to n).  May be empty if not needed.</param>
  /// <param name="vStarOut">Output: v* = sum_a y*(a) q_fill(a).</param>
  /// <param name="options">Tuning knobs.  If the BisectionIterations field is 0, RPOOptions.Default is used.</param>
  /// <param name="nanFallbackQ">Fallback value used for NaN entries in q under reverse KL.  If itself NaN, the mean of the finite q's is used.</param>
  /// <param name="lambdaPerChild">Optional per-action lambda vector (length n).  When non-empty, replaces the scalar lambda in the per-action coefficient: coeff[i] = lambdaPerChild[i] * mu[i].  When empty, scalar lambda is used.  REVERSE-KL ONLY (forward-KL closed form does not generalize naturally to per-child lambda; an empty span is enforced there).</param>
  public static void Solve(ReadOnlySpan<double> mu,
                           ReadOnlySpan<double> q,
                           double lambda,
                           RPOAnchor anchor,
                           RPORegularization regularization,
                           Span<double> yOut,
                           Span<double> qFillOut,
                           out double vStarOut,
                           RPOOptions options = default,
                           double nanFallbackQ = double.NaN,
                           ReadOnlySpan<double> lambdaPerChild = default)
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
    if (!lambdaPerChild.IsEmpty && lambdaPerChild.Length < mu.Length)
    {
      throw new ArgumentException("lambdaPerChild is shorter than mu.");
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
        if (lambdaPerChild.IsEmpty && !(lambda >= 0.0))
        {
          throw new ArgumentOutOfRangeException(nameof(lambda), "Reverse KL requires lambda >= 0.");
        }
        SolveReverseKL(mu, q, lambda, lambdaPerChild, yOut, qFillOut, out vStarOut, opts, nanFallbackQ);
        return;

      case RPORegularization.ForwardKLSoftmax:
        if (!lambdaPerChild.IsEmpty)
        {
          throw new ArgumentException("ForwardKLSoftmax does not support lambdaPerChild (must be empty).");
        }
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
                                     ReadOnlySpan<double> lambdaPerChild,
                                     Span<double> yOut,
                                     Span<double> qFillOut,
                                     out double vStarOut,
                                     RPOOptions opts,
                                     double nanFallbackQ)
  {
    int n = mu.Length;
    bool perChild = !lambdaPerChild.IsEmpty;

    // Per-thread scratch (see field declarations): always needed even if outputs are empty.
    Span<double> muNorm = (bufferMuNorm ??= new double[MAX_ACTIONS]).AsSpan(0, n);
    Span<double> qFill = (bufferQFill ??= new double[MAX_ACTIONS]).AsSpan(0, n);
    Span<double> coeff = (bufferCoeff ??= new double[MAX_ACTIONS]).AsSpan(0, n);
    Span<double> yLocal = (bufferY ??= new double[MAX_ACTIONS]).AsSpan(0, n);

    // Scratch for the vectorized Newton evaluator (TrySolveAlphaNewton / EvalSumAndDeriv).
    Span<double> newtD = (bufferNewtD ??= new double[MAX_ACTIONS]).AsSpan(0, n);
    Span<double> newtRatio = (bufferNewtRatio ??= new double[MAX_ACTIONS]).AsSpan(0, n);

    NormalizeMu(mu, muNorm, opts.MinPriorProbability);

    // Pre-pass: determine fallback for NaN q entries.
    double fallback = ResolveFallback(q, nanFallbackQ, opts.ClampQ);

    double maxQEff = double.NegativeInfinity;
    bool anyPositiveCoeff = false;
    double maxLambdaEff = 0.0;
    for (int i = 0; i < n; i++)
    {
      double qi = q[i];
      if (!IsFinite(qi))
      {
        qi = fallback;
      }
      if (opts.ClampQ)
      { 
        qi = Clamp(qi, -1.2, 1.2);
      }
      qFill[i] = qi;

      // Per-action lambda when supplied; otherwise scalar lambda for all.
      double lambdaI = perChild ? lambdaPerChild[i] : lambda;
      if (lambdaI < 0.0 || !IsFinite(lambdaI))
      {
        lambdaI = 0.0;
      }
      if (lambdaI > maxLambdaEff) maxLambdaEff = lambdaI;

      double c = lambdaI * muNorm[i];
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

    // Degenerate cases: greedy on q if no positive coefficient or every lambda is tiny.
    // In per-child mode, the gate is on max(lambdaPerChild); otherwise the scalar lambda.
    double lambdaForDegen = perChild ? maxLambdaEff : lambda;
    if (!anyPositiveCoeff || lambdaForDegen <= 1e-12)
    {
      GreedyOnQ(qFill, yLocal);
    }
    else if (!TrySolveAlphaNewton(coeff, qFill, maxQEff, opts.BisectionIterations,
                                  opts.BisectionResidualTol, newtD, newtRatio, out double alpha))
    {
      // Root-find failed to bracket: fall back to normalized prior (closest to greedy-prior choice).
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
          TensorPrimitives.Multiply(yLocal, invRenorm, yLocal);
        }
      }
    }

    // Copy outputs and compute vStar = sum_i y_i * qFill_i (vectorized dot product).
    vStarOut = TensorPrimitives.Dot(yLocal, qFill);
    if (!yOut.IsEmpty)
    {
      yLocal.CopyTo(yOut);
    }
    if (!qFillOut.IsEmpty)
    {
      qFill.CopyTo(qFillOut);
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

    // Per-thread scratch (see field declarations).
    Span<double> muNorm = (bufferMuNorm ??= new double[MAX_ACTIONS]).AsSpan(0, n);
    Span<double> logMu = (bufferLogMu ??= new double[MAX_ACTIONS]).AsSpan(0, n);
    Span<double> qFill = (bufferQFill ??= new double[MAX_ACTIONS]).AsSpan(0, n);
    Span<double> yLocal = (bufferY ??= new double[MAX_ACTIONS]).AsSpan(0, n);

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
      if (opts.ClampQ)
      {
        qi = Clamp(qi, -1.0, 1.0);
      }
      qFill[i] = qi;
    }

    // Compute y_i proportional to mu_i * exp(q_i / lambda) with numerically-stable shift.
    // exp((q - qMax) / lambda) stays in [0, 1], avoiding overflow when lambda is small.
    double maxQ = TensorPrimitives.Max(qFill);
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
      TensorPrimitives.Multiply(yLocal, inv, yLocal);
    }
    else
    {
      WriteNormalizedPrior(muNorm, yLocal);
    }

    vStarOut = TensorPrimitives.Dot(yLocal, qFill);

    if (!yOut.IsEmpty)
    {
      yLocal.CopyTo(yOut);
    }
    if (!qFillOut.IsEmpty)
    {
      qFill.CopyTo(qFillOut);
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
          // entropy = -E_mu[log mu] = -sum_i muNorm_i * logMu_i  (vectorized dot product).
          double entropy = -TensorPrimitives.Dot(muNorm, logMu);
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
      TensorPrimitives.Multiply(muNorm, inv, muNorm);
    }
    else
    {
      double uni = 1.0 / n;
      muNorm.Fill(uni);
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
  /// Solves  S(alpha) = sum_i coeff_i / (alpha - qEff_i) = 1  for alpha &gt; maxQEff using a
  /// safeguarded Newton iteration, exiting as soon as the residual |S(alpha) - 1| falls below
  /// residualTol (or the iteration cap maxIterations is reached).  Returns true on success.
  ///
  /// On (maxQEff, +inf) the objective S is smooth, strictly decreasing, and convex, so Newton
  /// converges quadratically - typically in a handful of steps versus the ~20 a bisection to the
  /// same tolerance needs.  This is the Numerical-Recipes "rtsafe" hybrid working on the strictly
  /// increasing g(a) = 1 - S(a): a Newton step is taken only when it stays inside the maintained
  /// bracket [xl, xh] AND would at least halve the previous step; otherwise a bisection step is
  /// taken.  That second guard is what makes convergence provably no slower than pure bisection
  /// (a plain "Newton-in-bracket-else-bisect" loop can accept tiny Newton steps and end up worse
  /// than bisection at a fixed iteration budget).
  ///
  /// d and ratio are caller-provided scratch buffers (length n) consumed by the vectorized
  /// evaluator EvalSumAndDeriv.
  /// </summary>
  private static bool TrySolveAlphaNewton(ReadOnlySpan<double> coeff,
                                          ReadOnlySpan<double> qEff,
                                          double maxQEff,
                                          int maxIterations,
                                          double residualTol,
                                          Span<double> d,
                                          Span<double> ratio,
                                          out double alpha)
  {
    const double EPS = 1e-12;
    const double STEP_TOL = 1e-13;     // alpha-resolution at which further refinement is pointless

    // Bracket [xl, xh] for the strictly-increasing g(a) = 1 - S(a):
    //   xl just above the largest active qEff, where S -> +inf  => g(xl) < 0
    //   xh grown (doubling the span above maxQEff) until S(xh) < 1 => g(xh) > 0
    double xl = maxQEff + EPS;
    double xh = maxQEff + 1.0;
    if (!(xh > xl))
    {
      xh = xl + 1.0;
    }

    EvalSumAndDeriv(coeff, qEff, xh, d, ratio, out double sHi, out _);
    int expand = 0;
    while (sHi > 1.0 && expand < 64 && IsFinite(xh))
    {
      double span = xh - maxQEff;
      if (!(span > 0.0) || !IsFinite(span))
      {
        span = 1.0;
      }
      xh = maxQEff + (span * 2.0);
      EvalSumAndDeriv(coeff, qEff, xh, d, ratio, out sHi, out _);
      expand++;
    }
    if (!(sHi < 1.0) || !IsFinite(sHi) || !IsFinite(xh))
    {
      alpha = 0.0;
      return false;
    }

    // rtsafe on g(a) = 1 - S(a), g'(a) = -S'(a) > 0, seeded at the bracket midpoint.
    // Eval-first ordering (evaluate rts, fold it into the bracket, then step) means K iterations
    // perform K bracket halvings in the worst case - exactly matching pure bisection, so the
    // hybrid is never slower - while Newton acceleration kicks in wherever the function is tame.
    double rts = 0.5 * (xl + xh);
    double dx = xh - xl;
    double dxOld = dx;

    for (int it = 0; it < maxIterations; it++)
    {
      EvalSumAndDeriv(coeff, qEff, rts, d, ratio, out double s, out double sPrime);
      double g = 1.0 - s;
      double gp = -sPrime;

      // Fold this evaluation into the bracket (g increasing: g < 0 -> raise xl, else lower xh),
      // then exit if the residual is within tolerance.
      if (IsFinite(g))
      {
        if (g < 0.0)
        {
          xl = rts;
        }
        else
        {
          xh = rts;
        }

        if (Math.Abs(g) < residualTol)
        {
          alpha = rts;
          return rts > maxQEff && IsFinite(rts);
        }
      }

      // Choose the next iterate: a Newton step only when it stays inside [xl, xh] AND would at
      // least halve the prior step; otherwise a bisection step (which always makes progress).
      bool useBisection =
           !IsFinite(g) || !IsFinite(gp)
        || (((rts - xh) * gp - g) * ((rts - xl) * gp - g) > 0.0)
        || (Math.Abs(2.0 * g) > Math.Abs(dxOld * gp));

      dxOld = dx;
      double prev = rts;
      bool noProgress;
      if (useBisection)
      {
        dx = 0.5 * (xh - xl);
        rts = xl + dx;
        noProgress = (rts == xl);     // bracket collapsed to a point
      }
      else
      {
        dx = g / gp;
        rts = prev - dx;
        noProgress = (rts == prev);   // Newton step underflowed (no change in alpha)
      }

      if (noProgress || Math.Abs(dx) < STEP_TOL)
      {
        alpha = rts;
        return rts > maxQEff && IsFinite(rts);
      }
    }

    // Iteration cap reached: rts is still bracketed and finite, so return it as the estimate.
    alpha = rts;
    return rts > maxQEff && IsFinite(rts);
  }


  /// <summary>
  /// Evaluates, at a given alpha, both the constraint LHS and its derivative:
  ///   S(alpha)  =  sum_i coeff_i / (alpha - qEff_i)
  ///   S'(alpha) = -sum_i coeff_i / (alpha - qEff_i)^2
  /// Vectorized with TensorPrimitives.  Assumes coeff_i &gt;= 0 and alpha &gt; max(qEff_i : coeff_i &gt; 0)
  /// so that every active denominator is positive (entries with coeff_i == 0 contribute 0).
  ///
  /// Works in terms of d_i = qEff_i - alpha (= -(alpha - qEff_i)) so the scalar-minus-vector can
  /// use the available Subtract(span, scalar) overload; the sign is folded into the two reductions.
  /// Because (alpha - qEff_i)^2 == d_i^2, the derivative needs no extra sign handling.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static void EvalSumAndDeriv(ReadOnlySpan<double> coeff, ReadOnlySpan<double> qEff, double alpha,
                                      Span<double> d, Span<double> ratio,
                                      out double s, out double sPrime)
  {
    // d_i = qEff_i - alpha
    TensorPrimitives.Subtract(qEff, alpha, d);

    // ratio_i = coeff_i / d_i ;  S = sum_i coeff_i/(alpha - qEff_i) = -sum_i ratio_i
    TensorPrimitives.Divide(coeff, d, ratio);
    s = -TensorPrimitives.Sum(ratio);

    // ratio_i <- ratio_i / d_i = coeff_i / d_i^2 = coeff_i/(alpha - qEff_i)^2 ;  S' = -sum_i ratio_i
    TensorPrimitives.Divide(ratio, d, ratio);
    sPrime = -TensorPrimitives.Sum(ratio);
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
