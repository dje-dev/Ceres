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
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Search.RPO;

public static class RPOTestsNEW
{
  public static double CalcRPOQ(GNode node, double qWhenNoChildren, int numChildrenToConsider, double rpoLambdaP)
  {
    const double BASE_U = 2.5;
    const double LAMBDA_U = 0.3f; // 0.5 was worse

    double lambdaPScaled = rpoLambdaP / Math.Sqrt(node.N);
    int numExpanded = node.NumEdgesExpanded;

    Span<(double Q, double P, double U)> actions = stackalloc (double Q, double P, double U)[numExpanded];

    for (int i = 0; i < numExpanded; i++)
    {
      bool drawExists = node.DrawKnownToExistAmongChildren;
      if (i < numExpanded)
      {
        GEdge edge = node.ChildEdgeAtIndex(i);
        actions[i].P = edge.P;

        int n = edge.N;
        if (n == 0)
        {
          actions[i].Q = node.Q; // fill-in for the case where this edge is still in flight and empty
          actions[i].U = BASE_U;
        }
        else
        {
          actions[i].Q = -edge.Q;
          actions[i].U = BASE_U / Math.Sqrt(edge.N);
        }
      }
      else
      {
        actions[i].P = node.EdgeHeadersSpan[i].P;
        actions[i].U = BASE_U; // Unclear what to use here
        actions[i].Q = qWhenNoChildren;
      }
    }

    Span<double> posteriorP = stackalloc double[numExpanded];
    Span<double> posteriorQ = stackalloc double[numExpanded];

    var options = new ReverseKlPosteriorPolicy.Options();
    ReverseKlPosteriorPolicy.ComputePosterior(actions,
                                              lambdaPScaled,
                                              LAMBDA_U,

                                              0.15, 1,
                                              posteriorP, posteriorQ,
                                              options);

    // compute dot product of posteriorP and posteriorQ 
    throw new NotImplementedException("Needs remediation, next line disabled out for TCEC compile failure");
    double dotProd = 0;// TensorPrimitives.Dot(posteriorP, posteriorQ);

    // Compute final q (also including impact of self)
    double q = (dotProd * (node.N - 1) + node.V) / node.N; 

    if (q < 0 && node.DrawKnownToExistAmongChildren)
    {
      return 0;
    }
    else
    {
      return q;
    }
  }
}

#if NOT
        if (weighted)
        {

          // It's not clear what to shrink toward;
          // e.g. for low-policy moves we expect possibly much worse than current Q.
          float qToShrinkTo = (float)node.Q - 0.15f; // bias downward slightly

          // Apply weighting to blend in parent Q value based on visit count
          // (shrink parent contribution based on sqrt(child_n)).
          float fractionParent = 0.2f * (1.0f / MathF.Sqrt(vN[i]));
          qN[i] = (fractionParent * qToShrinkTo) + (1.0f - fractionParent) * qN[i];
        }
      }
      else
      {
        qN[i] = qWhenNoChildren; // + MCGSSelect.RPO_NEW_CHILDREN_FPU_ADJUSTMENT;
        vN[i] = 0;
#endif


/// <summary>
/// Computes a "posterior" move distribution pi from (Q, priorP) using the reverse-KL
/// regularized policy improvement objective described by Grill et al. (RPO view of MCTS).
///
/// Core closed form (reverse KL):
///   pi_i = (lambda * priorP_i) / (alpha - qEff_i)
/// where alpha is chosen such that sum_i pi_i = 1 and alpha > max_i qEff_i.
///
/// This implementation recomputes pi each call (e.g., each node-visit), writes outputs into caller buffers,
/// and imputes missing Q/U values when Q and U are NaN.
/// </summary>
public static class ReverseKlPosteriorPolicy
{
  /// <summary>
  /// How uncertainty U (standard deviation of Q) affects the computation.
  /// </summary>
  public enum UncertaintyMode
  {
    /// <summary>No uncertainty handling; qEff = Q; coeff = lambdaP * priorP.</summary>
    None = 0,

    /// <summary>
    /// Robust / risk-averse: qEff = Q - lambdaQ * U.
    /// Justification: worst-case (lower-confidence) value under bounded/normal uncertainty.
    /// </summary>
    LowerConfidenceBound = 1,

    /// <summary>
    /// Optimistic / exploration-biased: qEff = Q + lambdaQ * U.
    /// Justification: upper-confidence heuristic to favor high-uncertainty actions.
    /// </summary>
    UpperConfidenceBound = 2,

    /// <summary>
    /// Keep qEff = Q, but increase the prior-regularization weight for uncertain actions:
    /// coeff_i = lambdaP * priorP_i * (1 + lambdaQ * U_i).
    /// Justification: when value is uncertain, rely more on the prior (stronger regularization).
    /// </summary>
    IncreasePriorWhenUncertain = 3,

    /// <summary>
    /// Keep qEff = Q, but decrease the prior-regularization weight for uncertain actions:
    /// coeff_i = lambdaP * priorP_i / (1 + lambdaQ * U_i).
    /// Justification: when value is uncertain, allow posterior to move away from the prior.
    /// </summary>
    DecreasePriorWhenUncertain = 4,

    /// <summary>
    /// Risk-averse variance penalty: qEff = Q - lambdaQ * U^2.
    /// Justification: quadratic penalty proportional to variance (U^2).
    /// </summary>
    VariancePenalty = 5
  }

  public class Options
  {
    public Options(UncertaintyMode uncertaintyMode = UncertaintyMode.LowerConfidenceBound,
                   double minPriorProbability = 0.0,
                    bool clampQToUnitInterval = true,
                    bool clampEffectiveQToUnitInterval = false,
                    int bisectionIterations = 20)
    {
      UncertaintyMode = uncertaintyMode;
      MinPriorProbability = minPriorProbability;
      ClampQToUnitInterval = clampQToUnitInterval;
      ClampEffectiveQToUnitInterval = clampEffectiveQToUnitInterval;
      BisectionIterations = bisectionIterations;
    }

    public UncertaintyMode UncertaintyMode { get; }

    /// <summary>
    /// Optional epsilon-smoothing floor applied to priorP before renormalization.
    /// Note: reverse-KL gives zero posterior mass to actions with exactly zero priorP;
    /// smoothing avoids "hard zeros" if desired.
    /// </summary>
    public double MinPriorProbability { get; }

    /// <summary>Clamp imputed/raw Q to [-1, +1].</summary>
    public bool ClampQToUnitInterval { get; }

    /// <summary>Clamp qEff to [-1, +1] after applying uncertainty transforms.</summary>
    public bool ClampEffectiveQToUnitInterval { get; }

    /// <summary>Number of bisection iterations used to solve for alpha.</summary>
    public int BisectionIterations { get; }
  }

  /// <summary>
  /// Compute posterior probabilities into <paramref name="posteriorPi"/> and an imputed Q vector into <paramref name="qImputed"/>.
  ///
  /// Inputs:
  ///   actions[i] = (Q, PriorP, U)
  ///     Q in [-1,+1] when provided; U is stddev of Q; PriorP should sum to 1.
  ///   lambdaP: strength of prior regularization (must be >= 0; if <= ~0 becomes greedy on qEff)
  ///   lambdaQ: strength of uncertainty effect (interpretation depends on Options.UncertaintyMode)
  ///   rootQ/rootU: fallback values used when an action has both Q and U as NaN (or individually missing).
  ///
  /// Outputs:
  ///   posteriorPi[i] = posterior probability for action i (sums to 1)
  ///   qImputed[i]    = original Q if finite else imputed Q
  ///
  /// Notes:
  ///   - This method performs no heap allocations for typical chess branching factors (uses stackalloc).
  ///   - Caller may treat qImputed as ReadOnlySpan<double> after the call.
  /// </summary>
  public static void ComputePosterior(ReadOnlySpan<(double Q, double PriorP, double U)> actions,
                                                    double lambdaP,
                                                    double lambdaQ,
                                                    double rootQ,
                                                    double rootU,
                                                    Span<double> posteriorPi,
                                                    Span<double> qImputed,
                                                    Options options)
  {
    int n = actions.Length;
    if (posteriorPi.Length < n)
    {
      throw new ArgumentException("posteriorPi must have length >= actions.Length", nameof(posteriorPi));
    }

    if (qImputed.Length < n)
    {
      throw new ArgumentException("qImputed must have length >= actions.Length", nameof(qImputed));
    }

    if (n == 0)
    {
      return;
    }

    if (options.BisectionIterations <= 0)
    {
      throw new ArgumentOutOfRangeException(nameof(options), "BisectionIterations must be positive.");
    }

    double fallbackQ = DetermineFallbackQ(actions, rootQ, options.ClampQToUnitInterval);
    double fallbackU = DetermineFallbackU(actions, rootU);

    // Sanitize and (optionally) floor priors, then normalize.
    double minPrior = options.MinPriorProbability;
    if (minPrior < 0.0)
    {
      minPrior = 0.0;
    }

    // Stack buffers:
    // coeff[i] = effective numerator coefficient (includes lambdaP, prior normalization, and any uncertainty weight factor)
    // qEff[i]  = effective Q (possibly adjusted for uncertainty)
    Span<double> coeff = n <= 512 ? stackalloc double[n] : new double[n];
    Span<double> qEff = n <= 512 ? stackalloc double[n] : new double[n];

    double priorSum = 0.0;
    for (int i = 0; i < n; i++)
    {
      double p = actions[i].PriorP;
      if (!IsFinite(p) || p < 0.0)
      {
        p = 0.0;
      }

      if (minPrior > 0.0 && p < minPrior)
      {
        p = minPrior;
      }

      priorSum += p;
    }

    bool useUniformPrior = priorSum <= 0.0 || !IsFinite(priorSum);
    double invPriorSum = useUniformPrior ? 0.0 : (1.0 / priorSum);

    // Build qImputed, qEff, coeff; also find max qEff over coeff>0.
    double maxQEff = double.NegativeInfinity;
    bool anyPositiveCoeff = false;

    for (int i = 0; i < n; i++)
    {
      double q = actions[i].Q;
      double u = actions[i].U;

      bool qFinite = IsFinite(q);
      bool uFinite = IsFinite(u);

      if (!qFinite)
      {
        q = fallbackQ;
      }

      if (!uFinite)
      {
        u = fallbackU;
      }

      if (u < 0.0)
      {
        u = -u;
      }

      if (options.ClampQToUnitInterval)
      {
        q = Clamp(q, -1.0, 1.0);
      }

      qImputed[i] = q;

      double pRaw = actions[i].PriorP;
      if (!IsFinite(pRaw) || pRaw < 0.0)
      {
        pRaw = 0.0;
      }

      if (minPrior > 0.0 && pRaw < minPrior)
      {
        pRaw = minPrior;
      }

      double pNorm = useUniformPrior ? (1.0 / (double)n) : (pRaw * invPriorSum);

      double qEffective = q;
      double weightFactor = 1.0;

      ApplyUncertaintyMode(options.UncertaintyMode, q, u, lambdaQ, out qEffective, out weightFactor);

      if (options.ClampEffectiveQToUnitInterval)
      {
        qEffective = Clamp(qEffective, -1.0, 1.0);
      }

      qEff[i] = qEffective;

      double c = lambdaP * pNorm * weightFactor;
      if (c < 0.0 || !IsFinite(c))
      {
        c = 0.0;
      }

      coeff[i] = c;

      if (c > 0.0)
      {
        anyPositiveCoeff = true;
        if (qEffective > maxQEff)
        {
          maxQEff = qEffective;
        }
      }
    }

    // Degenerate cases.
    if (!anyPositiveCoeff)
    {
      // If lambdaP is 0 (or priors are all 0), fall back to greedy on qEff.
      GreedyOnQ(qEff, posteriorPi.Slice(0, n));
      return;
    }

    if (lambdaP <= 1e-12)
    {
      GreedyOnQ(qEff, posteriorPi.Slice(0, n));
      return;
    }

    // Solve for alpha: sum_i coeff_i / (alpha - qEff_i) = 1, alpha > maxQEff.
    double alpha;
    bool alphaOk = TrySolveAlphaBisection(coeff, qEff, maxQEff, options.BisectionIterations, out alpha);

    if (!alphaOk)
    {
      // Fallback: use normalized prior (with smoothing/uniform if needed).
      WriteNormalizedPrior(actions, minPrior, posteriorPi.Slice(0, n));
      return;
    }

    // Compute posterior pi.
    double sumPi = 0.0;
    for (int i = 0; i < n; i++)
    {
      double denom = alpha - qEff[i];
      if (denom <= 0.0 || !IsFinite(denom))
      {
        posteriorPi[i] = 0.0;
        continue;
      }

      double pi = coeff[i] / denom;
      if (pi < 0.0 || !IsFinite(pi))
      {
        pi = 0.0;
      }

      posteriorPi[i] = pi;
      sumPi += pi;
    }

    if (!(sumPi > 0.0) || !IsFinite(sumPi))
    {
      WriteNormalizedPrior(actions, minPrior, posteriorPi.Slice(0, n));
      return;
    }

    double invSumPi = 1.0 / sumPi;
    double renormSum = 0.0;

    for (int i = 0; i < n; i++)
    {
      double v = posteriorPi[i] * invSumPi;
      if (v < 0.0 || !IsFinite(v))
      {
        v = 0.0;
      }

      posteriorPi[i] = v;
      renormSum += v;
    }

    // Final tiny renormalization to reduce drift.
    if (renormSum > 0.0 && IsFinite(renormSum))
    {
      double invRenorm = 1.0 / renormSum;
      for (int i = 0; i < n; i++)
      {
        posteriorPi[i] = posteriorPi[i] * invRenorm;
      }
    }
  }

  private static void ApplyUncertaintyMode(UncertaintyMode mode,
                                           double q,
                                           double u,
                                           double lambdaQ,
                                           out double qEffective,
                                           out double weightFactor)
  {
    qEffective = q;
    weightFactor = 1.0;

    if (!IsFinite(u))
    {
      u = 0.0;
    }

    if (!IsFinite(lambdaQ))
    {
      lambdaQ = 0.0;
    }

    if (lambdaQ < 0.0)
    {
      // Negative lambdaQ is allowed but unusual; treat as magnitude and keep sign via mode choice.
      lambdaQ = -lambdaQ;
    }

    if (mode == UncertaintyMode.None)
    {
      return;
    }

    if (mode == UncertaintyMode.LowerConfidenceBound)
    {
      qEffective = q - (lambdaQ * u);
      return;
    }

    if (mode == UncertaintyMode.UpperConfidenceBound)
    {
      qEffective = q + (lambdaQ * u);
      return;
    }

    if (mode == UncertaintyMode.VariancePenalty)
    {
      qEffective = q - (lambdaQ * u * u);
      return;
    }

    if (mode == UncertaintyMode.IncreasePriorWhenUncertain)
    {
      // coeff_i = lambdaP * priorP_i * (1 + lambdaQ * U_i)
      weightFactor = 1.0 + (lambdaQ * u);
      if (weightFactor < 1e-12 || !IsFinite(weightFactor))
      {
        weightFactor = 1e-12;
      }
      return;
    }

    if (mode == UncertaintyMode.DecreasePriorWhenUncertain)
    {
      // coeff_i = lambdaP * priorP_i / (1 + lambdaQ * U_i)
      double denom = 1.0 + (lambdaQ * u);
      if (denom < 1e-12 || !IsFinite(denom))
      {
        denom = 1e-12;
      }
      weightFactor = 1.0 / denom;
      return;
    }
  }

  private static bool TrySolveAlphaBisection(ReadOnlySpan<double> coeff,
                                             ReadOnlySpan<double> qEff,
                                             double maxQEff,
                                             int iterations,
                                             out double alpha)
  {
    // Need alpha > maxQEff. Start slightly above.
    double eps = 1e-12;
    double low = maxQEff + eps;

    // Find a high such that sum < 1 at high (function is monotone decreasing in alpha).
    double high = maxQEff + 1.0;
    if (!(high > low))
    {
      high = low + 1.0;
    }

    double sumAtHigh = SumCoeffOverAlphaMinusQ(coeff, qEff, high);
    int expand = 0;

    while (sumAtHigh > 1.0 && expand < 64 && IsFinite(high))
    {
      double span = high - maxQEff;
      if (!(span > 0.0) || !IsFinite(span))
      {
        span = 1.0;
      }

      high = maxQEff + (span * 2.0);
      sumAtHigh = SumCoeffOverAlphaMinusQ(coeff, qEff, high);
      expand++;
    }

    if (!(sumAtHigh < 1.0) || !IsFinite(sumAtHigh) || !IsFinite(high))
    {
      alpha = 0.0;
      return false;
    }

    // Bisection on equation sum(alpha) = 1.
    double aLo = low;
    double aHi = high;

    for (int it = 0; it < iterations; it++)
    {
      double mid = 0.5 * (aLo + aHi);
      double s = SumCoeffOverAlphaMinusQ(coeff, qEff, mid);

      if (!IsFinite(s))
      {
        // If numerical trouble, push alpha upward.
        aLo = mid;
        continue;
      }

      if (s > 1.0)
      {
        // alpha too small -> increase alpha
        aLo = mid;
      }
      else
      {
        // alpha too large -> decrease alpha
        aHi = mid;
      }
    }

    alpha = aHi;
    return alpha > maxQEff && IsFinite(alpha);
  }


  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double SumCoeffOverAlphaMinusQ(ReadOnlySpan<double> coeff, ReadOnlySpan<double> qEff, double alpha)
  {
    int n = coeff.Length;

    // Use vectorized path for x64 with AVX support and sufficient length.
    if (Avx.IsSupported && n >= Vector256<double>.Count)
    {
      return SumCoeffOverAlphaMinusQVectorized(coeff, qEff, alpha);
    }

    return SumCoeffOverAlphaMinusQScalar(coeff, qEff, alpha);
  }



  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double SumCoeffOverAlphaMinusQScalar(ReadOnlySpan<double> coeff, ReadOnlySpan<double> qEff, double alpha)
  {
    // Assumes: all coeff[i] > 0, all values finite.
    double sum = 0.0;
    int n = coeff.Length;

    for (int i = 0; i < n; i++)
    {
      sum += coeff[i] / (alpha - qEff[i]);
    }

    return sum;
  }



  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static unsafe double SumCoeffOverAlphaMinusQVectorized(ReadOnlySpan<double> coeff, ReadOnlySpan<double> qEff, double alpha)
  {
    // Assumes: all coeff[i] > 0, all values finite.
    int n = coeff.Length;
    int vectorEnd = n & ~3; // Round down to multiple of 4

    Vector256<double> alphaVec = Vector256.Create(alpha);
    Vector256<double> sumVec = Vector256<double>.Zero;

    fixed (double* pCoeff = coeff)
    fixed (double* pQEff = qEff)
    {
      double* coeffPtr = pCoeff;
      double* qEffPtr = pQEff;
      double* coeffEnd = pCoeff + vectorEnd;

      while (coeffPtr < coeffEnd)
      {
        Vector256<double> coeffVec = Avx.LoadVector256(coeffPtr);
        Vector256<double> qEffVec = Avx.LoadVector256(qEffPtr);

        // denom = alpha - qEff, then coeff / denom
        sumVec = Avx.Add(sumVec, Avx.Divide(coeffVec, Avx.Subtract(alphaVec, qEffVec)));

        coeffPtr += 4;
        qEffPtr += 4;
      }
    }

    // Horizontal sum: add pairs then scalar
    Vector128<double> lo = sumVec.GetLower();
    Vector128<double> hi = sumVec.GetUpper();
    Vector128<double> sum128 = Sse2.Add(lo, hi);
    double sum = sum128.ToScalar() + sum128.GetElement(1);

    // Handle remaining elements with scalar loop.
    for (int i = vectorEnd; i < n; i++)
    {
      sum += coeff[i] / (alpha - qEff[i]);
    }

    return sum;
  }

  private static void GreedyOnQ(ReadOnlySpan<double> qEff, Span<double> posteriorPi)
  {
    int n = qEff.Length;

    int best = 0;
    double bestQ = qEff[0];

    for (int i = 1; i < n; i++)
    {
      double v = qEff[i];
      if (v > bestQ)
      {
        bestQ = v;
        best = i;
      }
    }

    for (int i = 0; i < n; i++)
    {
      posteriorPi[i] = 0.0;
    }

    posteriorPi[best] = 1.0;
  }

  private static void WriteNormalizedPrior(
      ReadOnlySpan<(double Q, double PriorP, double U)> actions,
      double minPrior,
      Span<double> posteriorPi)
  {
    int n = actions.Length;

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
      double p = actions[i].PriorP;
      if (!IsFinite(p) || p < 0.0)
      {
        p = 0.0;
      }

      if (minPrior > 0.0 && p < minPrior)
      {
        p = minPrior;
      }

      sum += p;
    }

    if (!(sum > 0.0) || !IsFinite(sum))
    {
      double v = 1.0 / (double)n;
      for (int i = 0; i < n; i++)
      {
        posteriorPi[i] = v;
      }
      return;
    }

    double inv = 1.0 / sum;
    for (int i = 0; i < n; i++)
    {
      double p = actions[i].PriorP;
      if (!IsFinite(p) || p < 0.0)
      {
        p = 0.0;
      }

      if (minPrior > 0.0 && p < minPrior)
      {
        p = minPrior;
      }

      posteriorPi[i] = p * inv;
    }
  }

  private static double DetermineFallbackQ(
      ReadOnlySpan<(double Q, double PriorP, double U)> actions,
      double rootQ,
      bool clampToUnit)
  {
    if (IsFinite(rootQ))
    {
      return clampToUnit ? Clamp(rootQ, -1.0, 1.0) : rootQ;
    }

    double sum = 0.0;
    int count = 0;

    for (int i = 0; i < actions.Length; i++)
    {
      double q = actions[i].Q;
      if (IsFinite(q))
      {
        sum += q;
        count++;
      }
    }

    if (count > 0)
    {
      double mean = sum / (double)count;
      return clampToUnit ? Clamp(mean, -1.0, 1.0) : mean;
    }

    return 0.0;
  }

  private static double DetermineFallbackU(ReadOnlySpan<(double Q, double PriorP, double U)> actions, double rootU)
  {
    if (IsFinite(rootU))
    {
      return rootU >= 0.0 ? rootU : -rootU;
    }

    double sum = 0.0;
    int count = 0;

    for (int i = 0; i < actions.Length; i++)
    {
      double u = actions[i].U;
      if (IsFinite(u))
      {
        if (u < 0.0)
        {
          u = -u;
        }

        sum += u;
        count++;
      }
    }

    if (count > 0)
    {
      return sum / (double)count;
    }

    // Conservative default uncertainty if nothing is known.
    return 0.25;
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static bool IsFinite(double x)
  {
    return !double.IsNaN(x) && !double.IsInfinity(x);
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double Clamp(double x, double lo, double hi)
  {
    if (x < lo)
    {
      return lo;
    }

    if (x > hi)
    {
      return hi;
    }

    return x;
  }
}










