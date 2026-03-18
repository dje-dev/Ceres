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
using System.Collections.Generic;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;


#endregion

namespace Ceres.MCGS.Utils;

/// <summary>
/// Utilities for:
///   - soft/forward-KL "reverse Boltzmann" calibration: Q ≈ τ log π + C(s)
///   - reverse-KL mappings between prior μ, improved y, q, and λ
///   - τ calibration helpers from per-node data or logs
///   
/// See paper by Grill et al. "Monte-Carlo Tree Search as Regularized Policy Optimization" 
/// (https://arxiv.org/abs/2007.12509).
/// </summary>
public static class BoltzmannCalibration
{
  public sealed record PolicyQSample(float[] Policy, float[] Q);

  /// <summary>
  /// Specifies the weighting scheme used when fitting τ (temperature) 
  /// in weighted regression of Q on log π.
  /// </summary>
  public enum NodeWeighting 
  {
    /// <summary>
    /// All actions are weighted equally (weight = 1).
    /// </summary>
    Uniform,

    /// <summary>
    /// Actions are weighted by their prior policy probability π(a).
    /// Emphasizes moves the network considers more likely.
    /// </summary>
    PriorPi,

    /// <summary>
    /// Actions are weighted by the improved policy y(a) (e.g., normalized visit counts).
    /// Emphasizes moves favored by search.
    /// </summary>
    ImprovedPolicy 
  }


  /// <summary>
  /// Forward-KL/entropy: calibrated child Q from parent value anchor.
  /// Q_i = v_parent + τ ( log π_i + H(π) ), ensuring E_π[Q]=v_parent.
  /// </summary>
  public static void ComputeQFromPolicy_MatchParentValue(ReadOnlySpan<float> pi,
                                                         float parentValue,
                                                         float tau,
                                                         Span<float> qOut,
                                                         bool renormalizeIfNeeded = true,
                                                         float epsilon = 1e-10f,
                                                         bool clipToRange = false,
                                                         float clipMin = -1f,
                                                         float clipMax = 1f)
                                                    {
    ThrowIfInvalidArgs(pi, qOut, tau, epsilon);
    float sum = TensorPrimitives.Sum(pi);
    if (renormalizeIfNeeded && (sum <= 0 || MathF.Abs(sum - 1f) > 1e-7f))
    {
      ScaleInPlace(pi, 1f / MathF.Max(sum, epsilon), out float[] piNorm);
      ComputeQFromPolicy_MatchParentValue(piNorm, parentValue, tau, qOut,
                                          renormalizeIfNeeded: false, epsilon, clipToRange, clipMin, clipMax);
      return;
    }

    // Compute entropy H = -Σ p log p (using double for accuracy)
    double H = 0.0;
    for (int i = 0; i < pi.Length; i++)
    {
      double p = ClampProb(pi[i], epsilon);
      H -= p * Math.Log(p);
    }

    for (int i = 0; i < pi.Length; i++)
    {
      double p = ClampProb(pi[i], epsilon);
      double qi = parentValue + tau * (Math.Log(p) + H);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = (float)qi;
    }
  }


  /// <summary>
  /// Forward-KL/entropy: calibrated child Q from anchoring a single child a*.
  /// 
  /// Given a policy prior π and one anchor child a* with known Q-value Q*, imputes
  /// Q-values for all siblings via Boltzmann inversion: Q̂(i) = Q* + τ·(log πᵢ − log π*).
  /// This assumes the prior is approximately a Boltzmann policy at temperature τ, so that
  /// log-probability ratios are proportional to Q-value differences — the anchor eliminates
  /// the unknown partition function, yielding a calibrated Q-vector from a single observation.
  /// </summary>
  public static void ComputeQFromPolicy_AnchorChild(ReadOnlySpan<float> pi,
                                                    int anchorIndex,
                                                    float anchorQ,
                                                    float tau,
                                                    Span<float> qOut,
                                                    bool renormalizeIfNeeded = true,
                                                    float epsilon = 1e-10f,
                                                    bool clipToRange = false,
                                                    float clipMin = -1f,
                                                    float clipMax = 1f)
  {
    ThrowIfInvalidArgs(pi, qOut, tau, epsilon);
    ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)anchorIndex, (uint)pi.Length, nameof(anchorIndex));

    if (renormalizeIfNeeded)
    {
      float sum = TensorPrimitives.Sum(pi);
      if (sum <= 0 || MathF.Abs(sum - 1f) > 1e-7f)
      {
        ScaleInPlace(pi, 1f / MathF.Max(sum, epsilon), out float[] piNorm);
        ComputeQFromPolicy_AnchorChild(piNorm, anchorIndex, anchorQ, tau, qOut,
                                       renormalizeIfNeeded: false, epsilon, clipToRange, clipMin, clipMax);
        return;
      }
    }

    // Scalar log for anchor
    float logAnchor = MathF.Log(ClampProbF(pi[anchorIndex], epsilon));

    int n = pi.Length;
    if (n == 0) return;

    // Work buffers for clamped pi and log(pi)
    float[] pBuf = ArrayPool<float>.Shared.Rent(n);
    float[] logBuf = ArrayPool<float>.Shared.Rent(n);

    try
    {
      Span<float> pSpan = new Span<float>(pBuf, 0, n);
      Span<float> logSpan = new Span<float>(logBuf, 0, n);

      // Copy and clamp: p <= eps => eps; p >= 1 => 1 - 1e-7f
      const float oneMinusTiny = 1.0f - 1e-7f;
      for (int i = 0; i < n; i++)
      {
        float v = pi[i];
        if (v <= epsilon) v = epsilon;
        else if (v >= 1.0f) v = oneMinusTiny;
        pSpan[i] = v;
      }

      // Vectorized log over the span
      TensorPrimitives.Log(pSpan, logSpan);

      // Vectorized affine transform and optional clipping
      int width = Vector<float>.Count;
      Vector<float> vTau = new Vector<float>(tau);
      Vector<float> vAnchorQ = new Vector<float>(anchorQ);
      Vector<float> vLogAnchor = new Vector<float>(logAnchor);

      int iVec = 0;
      if (!clipToRange)
      {
        for (; iVec <= n - width; iVec += width)
        {
          Vector<float> vLog = new Vector<float>(logSpan.Slice(iVec, width));
          Vector<float> v = vAnchorQ + vTau * (vLog - vLogAnchor);
          v.CopyTo(qOut.Slice(iVec, width));
        }
        for (; iVec < n; iVec++)
        {
          qOut[iVec] = anchorQ + tau * (logSpan[iVec] - logAnchor);
        }
      }
      else
      {
        Vector<float> vMin = new Vector<float>(clipMin);
        Vector<float> vMax = new Vector<float>(clipMax);

        for (; iVec <= n - width; iVec += width)
        {
          Vector<float> vLog = new Vector<float>(logSpan.Slice(iVec, width));
          Vector<float> v = vAnchorQ + vTau * (vLog - vLogAnchor);
          v = Vector.Min(Vector.Max(v, vMin), vMax);
          v.CopyTo(qOut.Slice(iVec, width));
        }
        for (; iVec < n; iVec++)
        {
          float qi = anchorQ + tau * (logSpan[iVec] - logAnchor);
          if (qi < clipMin) qi = clipMin;
          else if (qi > clipMax) qi = clipMax;
          qOut[iVec] = qi;
        }

        // Ensure strictly descending order by adding epsilon offsets to clipped duplicates on the right.
        // Since policy values are ordered descending, Q values should also be strictly descending.
        const float descendingEpsilon = 0.01f;
        for (int i = 1; i < n; i++)
        {
          if (qOut[i] >= qOut[i - 1])
          {
            qOut[i] = qOut[i - 1] - descendingEpsilon;
          }
        }
      }
    }
    finally
    {
      ArrayPool<float>.Shared.Return(pBuf);
      ArrayPool<float>.Shared.Return(logBuf);
    }
  }


  /// <summary>
  /// Reverse-KL (Grill et al. RPO): derive per-child Q from prior μ, improved y, parent value v, and λ.
  /// Formula: Q(a) = v + λ ( 1 - μ(a) / y(a) ).
  /// Typically y is the visit distribution (normalized) or the closed-form optimizer.
  /// </summary>
  public static void ComputeQFromPriorAndImprovedPolicy_ReverseKL(
      ReadOnlySpan<float> mu,               // prior policy π_θ
      ReadOnlySpan<float> improved,         // improved policy y (e.g., normalized visits)
      float parentValue,                    // v(s)
      float lambda,                         // λ_N
      Span<float> qOut,
      bool renormalizeIfNeeded = true,
      float epsilon = 1e-10f,
      bool clipToRange = false,
      float clipMin = -1f,
      float clipMax = 1f)
  {
    if (mu.Length != improved.Length || mu.Length != qOut.Length)
      throw new ArgumentException("All spans must have identical length.");
    if (!(lambda >= 0f)) throw new ArgumentOutOfRangeException(nameof(lambda), "λ must be ≥ 0.");
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = mu.Length;

    float sMu = TensorPrimitives.Sum(mu);
    float sY = TensorPrimitives.Sum(improved);
    if (renormalizeIfNeeded)
    {
      if (sMu <= 0) throw new ArgumentException("Prior μ has non-positive sum.");
      if (sY <= 0) throw new ArgumentException("Improved policy y has non-positive sum.");
    }
    float invMu = renormalizeIfNeeded ? 1f / sMu : 1f;
    float invY = renormalizeIfNeeded ? 1f / sY : 1f;

    for (int i = 0; i < n; i++)
    {
      double p = ClampProb(mu[i] * invMu, epsilon);
      double ya = ClampProb(improved[i] * invY, epsilon);
      double qi = parentValue + lambda * (1.0 - p / ya);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = (float)qi;
    }
  }

  /// <summary>
  /// Reverse-KL optimizer: given q, prior μ, and λ, compute y* = argmax_y { q^T y - λ KL[μ, y] }.
  /// Closed form: y*(a) = λ μ(a) / (α - q(a)), with α chosen so Σ y*(a)=1 and α > max_a q(a).
  /// Uses safe bisection for α.
  /// </summary>
  public static void ComputeImprovedPolicy_ReverseKL(ReadOnlySpan<float> q,
                                                     ReadOnlySpan<float> mu,
                                                     float lambda,
                                                     Span<float> yOut,
                                                     bool renormalizeMuIfNeeded = true,
                                                     float epsilon = 1e-10f,
                                                     int maxBisectionIters = 60)
  {
    if (q.Length != mu.Length || q.Length != yOut.Length)
    {
      throw new ArgumentException("q, μ, and yOut must have the same length.");
    }

    ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(lambda, 0f, nameof(lambda));
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = q.Length;

    // Normalize μ if requested
    float sumMu = TensorPrimitives.Sum(mu);
    if (renormalizeMuIfNeeded)
    {
      if (sumMu <= 0) throw new ArgumentException("μ has non-positive sum.");
    }
    float invMu = renormalizeMuIfNeeded ? 1f / sumMu : 1f;

    // α must be strictly greater than max q(a) to keep denominators positive
    float qMax = TensorPrimitives.Max(q);

    // Bracket α: lower just above qMax; upper grows until Σ y(α) <= 1
    double lower = qMax + 1e-12;
    double upper = lower * 2 + 1.0; // initial guess
    for (int guard = 0; guard < 60; guard++)
    {
      double sumY = 0.0;
      for (int i = 0; i < n; i++)
      {
        double denom = upper - q[i];
        if (denom <= 0) { sumY = double.PositiveInfinity; break; }
        double mui = ClampProb(mu[i] * invMu, epsilon);
        sumY += (lambda * mui) / denom;
      }
      if (sumY <= 1.0) break;
      upper = upper * 2 + 1.0;
    }

    // Bisection to solve Σ y(α) = 1
    for (int it = 0; it < maxBisectionIters; it++)
    {
      double mid = 0.5 * (lower + upper);
      double sumY = 0.0;
      for (int i = 0; i < n; i++)
      {
        double denom = mid - q[i];
        if (denom <= 0) { sumY = double.PositiveInfinity; break; }
        double mui = ClampProb(mu[i] * invMu, epsilon);
        sumY += (lambda * mui) / denom;
      }
      if (sumY > 1.0) lower = mid; else upper = mid;
    }
    double alpha = 0.5 * (lower + upper);

    // Compute y*(a)
    double sumYFinal = 0.0;
    for (int i = 0; i < n; i++)
    {
      double denom = alpha - q[i];
      double mui = ClampProb(mu[i] * invMu, epsilon);
      double yi = (lambda * mui) / denom;
      yOut[i] = (float)yi;
      sumYFinal += yi;
    }
    // Normalize to sum exactly 1 (small numerical drift)
    if (sumYFinal > 0)
    {
      double inv = 1.0 / sumYFinal;
      for (int i = 0; i < n; i++) yOut[i] = (float)(yOut[i] * inv);
    }
  }


  /// <summary>
  /// τ fit at a single node: regress Q on log π with a per-node intercept removed.
  /// Returns slope α ≈ τ. Choose weighting scheme: Uniform, PriorPi, or ImprovedPolicy.
  /// </summary>
  public static float FitTauAtNode(ReadOnlySpan<float> pi,
                                   ReadOnlySpan<float> q,
                                   NodeWeighting weighting = NodeWeighting.PriorPi,
                                   ReadOnlySpan<float> improvedPolicy = default,
                                   float epsilon = 1e-10f)
  {
    if (pi.Length != q.Length) throw new ArgumentException("pi and q must have the same length.");
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = pi.Length;

    // Normalize π and (optionally) y
    float sPi = TensorPrimitives.Sum(pi);
    if (sPi <= 0) throw new ArgumentException("π has non-positive sum.");
    float invPi = 1f / sPi;

    float invY = 1f;
    bool useY = weighting == NodeWeighting.ImprovedPolicy;
    if (useY)
    {
      if (improvedPolicy.Length != n) throw new ArgumentException("improvedPolicy length mismatch.");
      float sY = TensorPrimitives.Sum(improvedPolicy);
      if (sY <= 0) throw new ArgumentException("Improved policy has non-positive sum.");
      invY = 1f / sY;
    }

    // Weighted means of x = log π, y = Q (using double for accuracy)
    double wSum = 0.0, xBar = 0.0, yBar = 0.0;
    for (int i = 0; i < n; i++)
    {
      double p = ClampProb(pi[i] * invPi, epsilon);
      double w = weighting switch
      {
        NodeWeighting.Uniform => 1.0,
        NodeWeighting.PriorPi => p,
        NodeWeighting.ImprovedPolicy => ClampProb(improvedPolicy[i] * invY, epsilon),
        _ => 1.0
      };
      double x = Math.Log(p);
      double yy = q[i];
      wSum += w; xBar += w * x; yBar += w * yy;
    }
    if (wSum <= 0) throw new InvalidOperationException("No positive weights.");
    xBar /= wSum; yBar /= wSum;

    double sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; i++)
    {
      double p = ClampProb(pi[i] * invPi, epsilon);
      double w = weighting switch
      {
        NodeWeighting.Uniform => 1.0,
        NodeWeighting.PriorPi => p,
        NodeWeighting.ImprovedPolicy => ClampProb(improvedPolicy[i] * invY, epsilon),
        _ => 1.0
      };
      double x = Math.Log(p);
      double dx = x - xBar;
      double dy = q[i] - yBar;
      sxx += w * dx * dx;
      sxy += w * dx * dy;
    }
    if (sxx <= 0) throw new InvalidOperationException("Insufficient variance in log π.");
    return (float)(sxy / sxx); // α ≈ τ
  }


  /// <summary>
  /// τ fit at a node using RPO-implied Q as the target: 
  /// First compute Q_RPO from (μ, y, v, λ), then fit τ in Q_RPO ≈ τ log π + C.
  /// </summary>
  public static float FitTauAtNodeFromRPO(ReadOnlySpan<float> mu,
                                          ReadOnlySpan<float> improved,
                                          float parentValue,
                                          float lambda,
                                          NodeWeighting weighting = NodeWeighting.PriorPi,
                                          float epsilon = 1e-10f)
  {
    int n = mu.Length;

    float[] qTmp = new float[n];
    ComputeQFromPriorAndImprovedPolicy_ReverseKL(mu, improved, parentValue, lambda, qTmp,
                                                 renormalizeIfNeeded: true, epsilon: epsilon,
                                                 clipToRange: false);
    return FitTauAtNode(mu, qTmp, weighting, improved, epsilon);
  }


  /// <summary>
  /// Fit τ globally from many nodes.
  /// Q ≈ τ log π + C(s), with within-node demeaning; optionally weight by π.
  /// </summary>
  public static float FitTauFromSamples(IEnumerable<PolicyQSample> samples, bool weightByPi = true, float epsilon = 1e-10f)
  {
    if (samples is null) throw new ArgumentNullException(nameof(samples));

    double numer = 0.0, denom = 0.0;

    foreach (PolicyQSample state in samples)
    {
      float[] pi = state.Policy;
      float[] q = state.Q;

      int n = Math.Min(pi.Length, q.Length);
      if (n <= 1) continue;

      float sum = 0f;
      for (int i = 0; i < n; i++)
      {
        sum += pi[i];
      }

      if (sum <= 0) continue;

      float invSum = 1f / sum;

      double wSum = 0.0, xBar = 0.0, yBar = 0.0;
      for (int i = 0; i < n; i++)
      {
        double p = ClampProb(pi[i] * invSum, epsilon);
        double w = weightByPi ? p : 1.0;
        double x = Math.Log(p);
        double y = q[i];
        wSum += w; xBar += w * x; yBar += w * y;
      }
      if (wSum <= 0) continue;
      xBar /= wSum; yBar /= wSum;

      double sxx = 0.0, sxy = 0.0;
      for (int i = 0; i < n; i++)
      {
        double p = ClampProb(pi[i] * invSum, epsilon);
        double w = weightByPi ? p : 1.0;
        double x = Math.Log(p);
        double y = q[i];
        double dx = x - xBar;
        double dy = y - yBar;
        sxx += w * dx * dx;
        sxy += w * dx * dy;
      }
      if (sxx <= 0) continue;

      numer += sxy;
      denom += sxx;
    }

    if (denom <= 0)
    {
      throw new InvalidOperationException("Insufficient variance in log π to fit τ.");
    }

    return (float)(numer / denom);
  }


  /// <summary>
  /// Compute λ_N from visit counts according to the Grill et al.-style scaling:
  /// λ_N = c * sqrt(N_total) / ( |A| + N_total ).
  /// Tune c to your domain; larger c strengthens regularization.
  /// </summary>
  public static float ComputeLambdaFromCounts(ReadOnlySpan<int> visitCounts, float c = 1.0f)
  {
    long N = 0; for (int i = 0; i < visitCounts.Length; i++) N += Math.Max(visitCounts[i], 0);
    int A = Math.Max(visitCounts.Length, 1);
    return (float)(c * Math.Sqrt(Math.Max(0L, N)) / (A + Math.Max(0L, N)));
  }


  /// <summary>
  /// Normalize integer visit counts into a probability distribution (optionally with a tiny additive prior).
  /// </summary>
  public static void NormalizeCounts(ReadOnlySpan<int> counts, Span<float> yOut, float additivePrior = 0.0f)
  {
    if (counts.Length != yOut.Length) throw new ArgumentException("counts and yOut must have same length.");
    double sum = 0.0;
    for (int i = 0; i < counts.Length; i++)
    {
      double val = Math.Max(0, counts[i]) + additivePrior;
      yOut[i] = (float)val; sum += val;
    }
    if (sum <= 0) { float inv = 1f / counts.Length; for (int i = 0; i < counts.Length; i++) yOut[i] = inv; return; }
    float invSum = (float)(1.0 / sum);
    for (int i = 0; i < counts.Length; i++) yOut[i] *= invSum;
  }


  public static float ComputeLambdaFromCounts(ReadOnlySpan<float> visitCounts, float c = 1.0f)
  {
    double N = 0.0; for (int i = 0; i < visitCounts.Length; i++) N += Math.Max(visitCounts[i], 0.0f);
    int A = Math.Max(visitCounts.Length, 1);
    return (float)(c * Math.Sqrt(Math.Max(0.0, N)) / (A + Math.Max(0.0, N)));
  }



  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double ClampProb(double p, double eps) => p <= eps ? eps : (p >= 1.0 ? (1.0 - 1e-16) : p);

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static float ClampProbF(float p, float eps) => p <= eps ? eps : (p >= 1.0f ? (1.0f - 1e-7f) : p);

  private static void ScaleInPlace(ReadOnlySpan<float> x, float scale, out float[] y)
  {
    y = new float[x.Length];
    for (int i = 0; i < x.Length; i++) y[i] = x[i] * scale;
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static void ThrowIfInvalidArgs(ReadOnlySpan<float> pi, Span<float> qOut, float tau, float eps)
  {
    if (pi.Length > qOut.Length) throw new ArgumentException("pi <= qOut length");
    if (!(tau > 0f)) throw new ArgumentOutOfRangeException(nameof(tau), "τ must be > 0.");
    if (!(eps > 0f && eps < 1f)) throw new ArgumentOutOfRangeException(nameof(eps), "epsilon must be in (0,1).");
  }


  public static void TestBoltzmann()
  {
    //					public static float FitTauFromLogs(IEnumerable<StateLog> logs, bool weightByPi = true, float epsilon = 1e-10f)

    float[] newPol = new float[2];
    float[] q = [0.664f, 0.600f];
    float[] pi = [0.769f, 0.231f];
    ComputeImprovedPolicy_ReverseKL(q: q, mu: pi, lambda: 0.064f, newPol); // ==> [0.55, 0.45]

    //ComputeImprovedPolicy_ReverseKL([0.2f, 0.1f], [0.3f, 0.7f], 0.3f, newPol); // ==> [0.37, 0.63]

    // Example 1: derive child Q's by matching the parent value
    //float[] pi = [0.769f, 0.231f]; /* policy head at state s, length = number of legal moves */
    //float vParent = 0.20f;/* value head at s, e.g., in [-1,1] */

    float tau = 0.05f; // start with a guess, or fit from logs (below)
    float parentValue = q[0];
    ComputeQFromPolicy_MatchParentValue(pi, parentValue, tau, q,
                                        renormalizeIfNeeded: true, epsilon: 1e-10f,
                                        clipToRange: true, clipMin: -1f, clipMax: 1f);

    // q[i] is a calibrated estimate you can use to initialize unexpanded edges.

    // Example 2: anchor to a trusted best child
    int bestIdx = 0;/* index chosen by your move ordering or partial search */
    float qBest = q[0];/* current MCTS estimate for that child */
    float[] q2 = new float[pi.Length];
    ComputeQFromPolicy_AnchorChild(pi, bestIdx, qBest, tau, q2,
                                   renormalizeIfNeeded: true, epsilon: 1e-10f,
                                   clipToRange: true, clipMin: -1f, clipMax: 1f);

    // Example 3: fit τ from actual search
    // Populate samples with (policy, per-child Q) snapshots at roots (or selected nodes).
    // Q here should be current best estimates from search for children of s.
    List<PolicyQSample> samples = [];

    float fittedTau = FitTauFromSamples(samples, weightByPi: true, epsilon: 1e-10f);
    // Now reuse fittedTau in the calls above.
  }
}

#if NOT
      // Compute best fit for tau from some of the nodes in the graph.
      if (COUNT++ % 4777 == 32 && graph.GraphRootNode.N > 220)// && graph.GraphRootNode.Index.Index % 77 ==43)
      {
        List<BoltzmannValueCalibrator.StateLog> logs = new();
        for (int i = 1; i < graph.GraphRootNode.N; i++)
        {
          GNode testNode = graph[i];
//          using (new SpinLockByteBlock(ref testNode.NodeRef.LockRef))
          {
            if (testNode.N > 20)
            {
              double[] q = new double[testNode.NumEdgesExpanded];
              double[] policy = new double[testNode.NumEdgesExpanded];
              for (int j = 0; j < testNode.NumEdgesExpanded; j++)
              {
                if (testNode.ChildEdgeAtIndex(j).Type != GEdgeStruct.EdgeType.ChildEdge)
                {
                  break;
                }
                //              q[j] = testNode.ChildEdgeAtIndex(j).Q;
                //                q[j] = testNode.Q;
                q[j] = -testNode.ChildEdgeAtIndex(j).ChildNode.V;// .Q;
                policy[j] = testNode.ChildEdgeAtIndex(j).P;
              }

              logs.Add(new(policy, q));
            }
          }
        }

        double taux = BoltzmannValueCalibrator.FitTauFromLogs(logs);//, bool weightByPi = true
        if (!double.IsNaN(taux)) Console.WriteLine(logs.Count + " " + taux);        
      }
#endif
