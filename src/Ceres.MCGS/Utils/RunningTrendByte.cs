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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.MCGS.Utils;

/// <summary>
/// Tiny (one byte) struct that maintains an (approximate) exponentially-weighted
/// mean of the INNOVATIONS folded into a node during backup - the signed drift of
/// incoming leaf values relative to the node's current Q. A persistently positive
/// value means recent evidence keeps arriving above the node's estimate (Q is
/// likely to rise with further visits); negative means the reverse; near zero
/// means the node is receiving value-confirming visits.
///
/// Companion to <see cref="RunningStdDevShort"/> (which tracks the second moment
/// of the same innovation stream); this tracks the first moment in a single byte.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 1)]
public record struct RunningTrendByte
{
  /// <summary>
  /// Encoded trend - the only per-instance state.
  /// Code 0 decodes to exactly zero so that arena zero-initialization yields a
  /// zero trend. Live codes 1..255 span [-RANGE, RANGE] linearly with 128 as the
  /// (equally exact) zero point.
  /// </summary>
  public byte Code;

  /// <summary>
  /// Representable trend magnitude. Innovation means larger than this saturate;
  /// resolution is RANGE/127 (about 0.0024).
  /// </summary>
  private const double RANGE = 0.3;

  private const int HalfLifeNumSamples = 32;
  private static readonly double Beta = 1.0 - Math.Pow(2.0, -1.0 / HalfLifeNumSamples);

  private const byte ZERO_CODE = 128;

  /// <summary>
  /// Folds a batch of <paramref name="count"/> samples (given their sum) into the EW
  /// innovation-mean estimate, measured about the reference mean <paramref name="mean"/>.
  /// Equivalent to applying count sequential single-sample updates about the same mean
  /// (aggregate decay), hence order-independent across merge rails like
  /// <see cref="RunningStdDevShort.AddBatch"/>.
  ///
  /// Encoding uses stochastic rounding driven by a DETERMINISTIC hash of the update
  /// inputs: per-sample EW deltas (Beta * innovation, often below half a grid step)
  /// would otherwise freeze the code permanently, while a hash-driven rounding keeps
  /// the stored grid value an unbiased estimator AND keeps identical searches
  /// byte-identical (same rationale as GNodeStruct.UpdateRepDrawFractionStochastic).
  /// </summary>
  /// <param name="mean">Reference mean the innovations are measured about (the node's Q).</param>
  /// <param name="sumV">Sum of the batch sample values (same perspective as <paramref name="mean"/>).</param>
  /// <param name="count">Number of samples in the batch.</param>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public void AddBatch(double mean, double sumV, int count)
  {
    if (count <= 0)
    {
      return;
    }

    double innovationMean = (sumV / count) - mean;

    // Parity with the RunningStdDevShort per-sample clamp (samples live in ~[-1.2, 1.2]).
    if (innovationMean < -2.4)
    {
      innovationMean = -2.4;
    }
    else if (innovationMean > 2.4)
    {
      innovationMean = 2.4;
    }

    double trend = DecodeTrend(Code);
    double decay = (count == 1) ? (1.0 - Beta) : Math.Pow(1.0 - Beta, count);
    trend = decay * trend + (1.0 - decay) * innovationMean;

    Code = EncodeTrend(trend, Code, sumV);
  }

  /// <summary>
  /// Current exponentially-weighted innovation mean. Seeded at zero, so it
  /// systematically under-reports until warmed up; prefer <see cref="TrendDebiased"/>
  /// when comparing across nodes with differing visit counts.
  /// </summary>
  public double Trend => DecodeTrend(Code);

  /// <summary>
  /// Bias-corrected trend removing the zero-initialization cold-start bias
  /// (Adam-style: divide by 1 - (1-Beta)^n). The struct stores no sample count,
  /// so the effective number of folded samples is supplied by the caller
  /// (e.g. the owning node's N). Noisy at very small n.
  /// </summary>
  /// <param name="sampleCount">Effective number of samples folded into this estimate (e.g. node N).</param>
  public double TrendDebiased(int sampleCount)
  {
    double trend = DecodeTrend(Code);
    if (sampleCount <= 0 || trend == 0)
    {
      return trend;
    }

    double biasCorrection = 1.0 - Math.Pow(1.0 - Beta, sampleCount);
    if (biasCorrection <= 0)
    {
      return trend;
    }

    double corrected = trend / biasCorrection;
    return Math.Clamp(corrected, -RANGE, RANGE);
  }


  /// <summary>
  /// Encoding: trend in [-RANGE, RANGE] -> byte code 1..255 (128 = exactly zero;
  /// code 0 is reserved for the zero-initialized state and never emitted),
  /// with deterministic-hash stochastic rounding (see <see cref="AddBatch"/>).
  /// </summary>
  private static byte EncodeTrend(double trend, byte priorCode, double hashSalt)
  {
    if (trend <= -RANGE)
    {
      return 1;
    }

    if (trend >= RANGE)
    {
      return 255;
    }

    double scaled = (trend / RANGE) * 127.0 + ZERO_CODE; // [1, 255]
    int lo = (int)Math.Floor(scaled);
    double frac = scaled - lo;

    // Deterministic "random" in [0,1): hash the exact update inputs so identical
    // searches reproduce identical rounding while distinct updates decorrelate.
    ulong bits = (ulong)BitConverter.DoubleToInt64Bits(trend) * 0x9E3779B97F4A7C15UL
               ^ (ulong)BitConverter.DoubleToInt64Bits(hashSalt) * 0xBF58476D1CE4E5B9UL
               ^ ((ulong)priorCode * 0xC2B2AE3D27D4EB4FUL);
    bits ^= bits >> 33; bits *= 0x62A9D9ED799705F5UL; bits ^= bits >> 28; bits *= 0xCB24D0A5C88C35B3UL; bits ^= bits >> 32;
    double u01 = (bits >> 11) * (1.0 / (1UL << 53));

    int q = lo + (u01 < frac ? 1 : 0);
    if (q < 1)
    {
      q = 1;
    }
    else if (q > 255)
    {
      q = 255;
    }

    return (byte)q;
  }

  private static double DecodeTrend(byte code)
      => code == 0 ? 0.0 : ((double)code - ZERO_CODE) * (RANGE / 127.0);
}
