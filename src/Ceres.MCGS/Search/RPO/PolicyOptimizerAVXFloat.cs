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
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Ceres.MCGS.Search.RPO;

public static class PolicyOptimizerAVXFloat
{
  private const int MaxActions = 64;
  public static Span<float> Optimize(float epsilonTol, Span<float> Q, float lambdaN, Span<float> P)
  {
    if (Q.Length == 0) { throw new ArgumentException("Q empty.", nameof(Q)); }
    if (Q.Length != P.Length) { throw new ArgumentException("Q and P must have same length."); }
    if (Q.Length > MaxActions) { throw new ArgumentException($"Supports at most {MaxActions} actions."); }
    if (lambdaN < 0.0f) { throw new ArgumentOutOfRangeException(nameof(lambdaN)); }

    int n = Q.Length;

    // Greedy fallback when epsilon close to zero
    if (lambdaN <= float.Epsilon)
    {
      return Greedy(Q);
    }

    float maxQ = MaxAvx(Q);
    float lower = maxQ + 1e-5f;                 // just above max(Q)
    float upper = ExpandUpper(lower, Q, P, lambdaN);
    float mu = Bisection(lower, upper, epsilonTol, Q, P, lambdaN);

    float[] pi = new float[n];
    ComputePi(Q, P, lambdaN, mu, pi);

    return pi;
  }

  //  Vectorised helpers
  private static float MaxAvx(ReadOnlySpan<float> data)
  {
    if (!Avx.IsSupported) { return MaxScalar(data); }

    Vector256<float> vMax;
    unsafe
    {
      ref float head = ref MemoryMarshal.GetReference(data);
      fixed (float* p = &head)
      {
        vMax = Avx.LoadVector256(p);                  // first 8
        int i = 8;
        for (; i + 7 < data.Length; i += 8)
        {
          vMax = Avx.Max(vMax, Avx.LoadVector256(p + i));
        }

        // Horizontal reduce vMax
        Span<float> tmp = stackalloc float[8];
        fixed (float* t = tmp)
        {
          Avx.Store(t, vMax);
        }
        float maxVal = tmp[0];
        for (int k = 1; k < 8; ++k) { maxVal = MathF.Max(maxVal, tmp[k]); }

        // Remainder
        for (; i < data.Length; ++i) { maxVal = MathF.Max(maxVal, p[i]); }

        return maxVal;
      }
    }
  }

  private static float MaxScalar(ReadOnlySpan<float> data)
  {
    float m = data[0];
    for (int i = 1; i < data.Length; ++i) { m = MathF.Max(m, data[i]); }
    return m;
  }

  private static float ExpandUpper
  (
      float lower,
      Span<float> Q,
      Span<float> P,
      float lambdaN
  )
  {
    float u = lower * 2.0f;
    while (ComputeF(u, Q, P, lambdaN) > 0.0f) { u *= 2.0f; }
    return u;
  }

  private static float Bisection(float lo, float hi, float tol, Span<float> Q, Span<float> P, float lambdaN)
  {
    float eps = MathF.Max(1e-7f, tol);
    while (hi - lo > eps)
    {
      float mid = 0.5f * (lo + hi);
      if (ComputeF(mid, Q, P, lambdaN) > 0.0f) { lo = mid; }
      else { hi = mid; }
    }
    return 0.5f * (lo + hi);
  }

  private static float ComputeF(float mu, Span<float> Q, Span<float> P, float lambdaN)
  {
    float sum = 0.0f;

    if (Avx.IsSupported)
    {
      Vector256<float> vMu = Vector256.Create(mu);
      Vector256<float> vLambda = Vector256.Create(lambdaN);
      Vector256<float> vSum = Vector256<float>.Zero;

      unsafe
      {
        ref float qRef = ref MemoryMarshal.GetReference(Q);
        ref float pRef = ref MemoryMarshal.GetReference(P);

        fixed (float* pQ = &qRef)
        fixed (float* pP = &pRef)
        {
          int i = 0;
          for (; i + 7 < Q.Length; i += 8)
          {
            Vector256<float> vQ = Avx.LoadVector256(pQ + i);
            Vector256<float> vP = Avx.LoadVector256(pP + i);

            Vector256<float> denom = Avx.Subtract(vMu, vQ);
            Vector256<float> frac = Avx.Divide(Avx.Multiply(vLambda, vP), denom);
            vSum = Avx.Add(vSum, frac);
          }

          Span<float> tmp = stackalloc float[8];
          fixed (float* t = tmp) { Avx.Store(t, vSum); }
          for (int k = 0; k < 8; ++k) { sum += tmp[k]; }

          for (; i < Q.Length; ++i) { sum += lambdaN * P[i] / (mu - Q[i]); }
        }
      }
    }
    else
    {
      for (int i = 0; i < Q.Length; ++i) { sum += lambdaN * P[i] / (mu - Q[i]); }
    }

    return sum - 1.0f;
  }


  private static void ComputePi(Span<float> Q, Span<float> P, float lambdaN, float mu, float[] piOut)
  {
    int n = Q.Length;
    Span<float> numer = stackalloc float[n];
    float sum = 0.0f;

    if (Avx.IsSupported)
    {
      Vector256<float> vMu = Vector256.Create(mu);
      Vector256<float> vLambda = Vector256.Create(lambdaN);
      Vector256<float> vSumAcc = Vector256<float>.Zero;

      unsafe
      {
        ref float qRef = ref MemoryMarshal.GetReference(Q);
        ref float pRef = ref MemoryMarshal.GetReference(P);
        ref float sRef = ref MemoryMarshal.GetReference(numer);

        fixed (float* pQ = &qRef)
        fixed (float* pP = &pRef)
        fixed (float* pN = &sRef)
        {
          int i = 0;
          for (; i + 7 < n; i += 8)
          {
            Vector256<float> vQ = Avx.LoadVector256(pQ + i);
            Vector256<float> vP = Avx.LoadVector256(pP + i);

            Vector256<float> denom = Avx.Subtract(vMu, vQ);
            Vector256<float> num = Avx.Divide(Avx.Multiply(vLambda, vP), denom);

            Avx.Store(pN + i, num);
            vSumAcc = Avx.Add(vSumAcc, num);
          }

          Span<float> tmp = stackalloc float[8];
          fixed (float* t = tmp) { Avx.Store(t, vSumAcc); }
          for (int k = 0; k < 8; ++k) { sum += tmp[k]; }

          for (; i < n; ++i)
          {
            float val = lambdaN * P[i] / (mu - Q[i]);
            numer[i] = val;
            sum += val;
          }
        }
      }
    }
    else
    {
      for (int i = 0; i < n; ++i)
      {
        float val = lambdaN * P[i] / (mu - Q[i]);
        numer[i] = val;
        sum += val;
      }
    }

    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) { piOut[i] = numer[i] * inv; }
  }


  //  Misc helpers
  private static Span<float> Greedy(ReadOnlySpan<float> Q)
  {
    int best = 0;
    float bestQ = Q[0];
    for (int i = 1; i < Q.Length; ++i)
    {
      if (Q[i] > bestQ) { best = i; bestQ = Q[i]; }
    }
    float[] pi = new float[Q.Length];
    pi[best] = 1.0f;
    return pi;
  }
}
