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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.MCGS.Utils;

/// <summary>
/// Small (two byte) struct that maintain an (approximate)
/// exponentially-weighted average standard deviation.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 2)]
public record struct RunningStdDevShort
{
  /// <summary>
  /// Encoded standard deviation (0..65535) — the only per-instance state.
  /// </summary>
  public ushort Code;

  private const double RANGE = 1.2; // samples live in [-1.2, 1.2] -> sigma in [0, 1.2]
  private const int HalfLifeNumSamples = 50;
  private static readonly double Beta = 1.0 - Math.Pow(2.0, -1.0 / HalfLifeNumSamples); // TODO: convert constant

  /// <summary>
  /// Flips to true to enable stochastic rounding when encoding (slightly slower, deterministic across runs).
  /// </summary>
  public static bool UseStochasticRounding { get; set; } = false;

  /// <summary>
  /// Constructor. 
  /// </summary>
  public RunningStdDevShort(int halfLifeNumSamples)
  {
    Debug.Assert(halfLifeNumSamples == HalfLifeNumSamples);  // Not currently configurable on a per-instance basis
  }


  /// <summary>
  /// Add one sample using an externally supplied mean (running or otherwise).
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public void AddSample(double mean, double sample)
  {
    Console.WriteLine(Math.Round(mean - sample, 2));
    // Bound inputs defensively (sample is documented in [-1.2,1.2], mean should be close).
    if (sample < -RANGE) 
    { 
      sample = -RANGE;
    }
    else if (sample > RANGE)
    {
      sample = RANGE;
    }

    if (mean < -RANGE)
    {
      mean = -RANGE;
    }
    else if (mean > RANGE)
    {
      mean = RANGE;
    }

    // Decode -> update EW variance (about the supplied mean) -> re-encode.
    double sigma = DecodeSigma(Code);
    double variance = sigma * sigma;

    double diff = sample - mean;
    if (diff < -2 * RANGE) diff = -2 * RANGE;
    if (diff > 2 * RANGE) diff = 2 * RANGE;

    // Exponentially-weighted variance of the deviations
    variance = (1.0 - Beta) * variance + Beta * (diff * diff);

    // Clamp to the representable domain and pack back to 16 bits
    if (variance < 0)
    {
      variance = 0;
    }

    double newSigma = Math.Sqrt(variance);
    if (newSigma > RANGE)
    {
      newSigma = RANGE;
    }

    Code = EncodeSigma(newSigma, Beta);
  }


  /// <summary>
  /// Current exponentially-weighted standard deviation.
  /// </summary>
  public double RunningStdDev => DecodeSigma(Code);

  /// <summary>
  /// Encoding: sigma in [0, RANGE] -> ushort (linear).
  /// </summary>
  private static ushort EncodeSigma(double sigma, double beta)
  {
    if (sigma <= 0)
    {
      return 0;
    }

    if (sigma >= RANGE)
    { 
      return ushort.MaxValue;
    }

    double scaled = (sigma / RANGE) * ushort.MaxValue; // [0, 65535)
    if (!UseStochasticRounding)
    {
      return (ushort)Math.Round(scaled);
    }
    else
    {
      int lo = (int)Math.Floor(scaled);
      double frac = scaled - lo;

      // Cheap, deterministic “random” in [0,1): hash the mantissae of sigma and Beta.
      ulong bits = (ulong)BitConverter.DoubleToInt64Bits(sigma) * 0x9E3779B97F4A7C15UL
                 ^ (ulong)BitConverter.DoubleToInt64Bits(beta) * 0xBF58476D1CE4E5B9UL;
      bits ^= bits >> 33; bits *= 0x62A9D9ED799705F5UL; bits ^= bits >> 28; bits *= 0xCB24D0A5C88C35B3UL; bits ^= bits >> 32;
      double u01 = (bits >> 11) * (1.0 / (1UL << 53)); // uniform in [0,1)

      int q = lo + (u01 < frac ? 1 : 0);
      if (q < 0) q = 0; else if (q > ushort.MaxValue) q = ushort.MaxValue;
      return (ushort)q;
    }
  }

  private static double DecodeSigma(ushort code) => (RANGE * (double)code) / ushort.MaxValue;
}
