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

using Ceres.Base.Math;

#endregion

namespace Chess.Ceres.PlayEvaluation
{
  /// <summary>
  /// Holds the result of a pentanomial (paired-game) statistical analysis of a match.
  ///
  /// In a paired-game match each opening is played twice with reversed colors.
  /// The two games of a pair are scored together as a single observation taking one of
  /// five possible outcomes (pair points of 0, 0.5, 1, 1.5, or 2). Because this captures
  /// the correlation between the two games of a pair, the resulting error bars are
  /// generally tighter (more statistically powerful) than the equivalent trinomial
  /// (independent win/draw/loss) analysis.
  /// </summary>
  public readonly record struct PentanomialResult
  {
    /// <summary>
    /// Number of pairs with pair score 0 (both games lost; LL).
    /// </summary>
    public readonly long N0;

    /// <summary>
    /// Number of pairs with pair score 0.5 (one loss and one draw; LD + DL).
    /// </summary>
    public readonly long N1;

    /// <summary>
    /// Number of pairs with pair score 1.0 (one win and one loss, or two draws; WL + LW + DD).
    /// </summary>
    public readonly long N2;

    /// <summary>
    /// Number of pairs with pair score 1.5 (one win and one draw; WD + DW).
    /// </summary>
    public readonly long N3;

    /// <summary>
    /// Number of pairs with pair score 2.0 (both games won; WW).
    /// </summary>
    public readonly long N4;

    /// <summary>
    /// Total number of completed pairs included in the analysis.
    /// </summary>
    public readonly int NumPairs;

    /// <summary>
    /// Mean per-game score (in [0, 1]); identical to the trinomial mean score.
    /// </summary>
    public readonly float ScoreRate;

    /// <summary>
    /// Standard deviation (per normalized per-game score) across pairs.
    /// </summary>
    public readonly float StdDev;

    /// <summary>
    /// Standard error of the mean per-game score.
    /// </summary>
    public readonly float StdError;

    /// <summary>
    /// Relative Elo difference implied by the mean score.
    /// </summary>
    public readonly float Elo;

    /// <summary>
    /// Pentanomial Elo error margin (the "+/-" value), at the confidence level
    /// specified when computed (1 sigma by default).
    /// </summary>
    public readonly float EloErrorMargin;

    /// <summary>
    /// Pentanomial likelihood of superiority (probability the true score exceeds 0.5).
    /// </summary>
    public readonly float LOS;

    public PentanomialResult(long n0, long n1, long n2, long n3, long n4, int numPairs,
                             float scoreRate, float stdDev, float stdError,
                             float elo, float eloErrorMargin, float los)
    {
      N0 = n0;
      N1 = n1;
      N2 = n2;
      N3 = n3;
      N4 = n4;
      NumPairs = numPairs;
      ScoreRate = scoreRate;
      StdDev = stdDev;
      StdError = stdError;
      Elo = elo;
      EloErrorMargin = eloErrorMargin;
      LOS = los;
    }

    /// <summary>
    /// An empty result (no completed pairs), with all statistics set to NaN.
    /// </summary>
    public static PentanomialResult Empty =>
      new PentanomialResult(0, 0, 0, 0, 0, 0, float.NaN, float.NaN, float.NaN, float.NaN, float.NaN, float.NaN);
  }


  /// <summary>
  /// Static methods that compute pentanomial (paired-game) match statistics,
  /// including Elo confidence intervals and likelihood of superiority (LOS).
  /// </summary>
  public static class PentanomialCalculator
  {
    /// <summary>
    /// Normalized per-game score associated with each of the five pair outcomes
    /// (pair points of 0, 0.5, 1, 1.5, 2 divided by 2 games).
    /// </summary>
    static readonly double[] SCORE_FOR_BUCKET = { 0.0, 0.25, 0.5, 0.75, 1.0 };


    /// <summary>
    /// Computes the pentanomial statistics for a set of completed pairs.
    /// </summary>
    /// <param name="counts">
    /// Array of length 5 holding the number of completed pairs in each outcome bucket,
    /// indexed by (pair points * 2): [0] = LL, [1] = LD+DL, [2] = WL+LW+DD, [3] = WD+DW, [4] = WW.
    /// </param>
    /// <param name="mult">
    /// Number of standard errors used for the Elo error margin (1.0 = 1 sigma, matching
    /// the trinomial EloCalculator.EloConfidenceInterval convention).
    /// </param>
    /// <returns>The computed pentanomial result, or PentanomialResult.Empty if no pairs.</returns>
    public static PentanomialResult Compute(ReadOnlySpan<long> counts, float mult = 1.0f)
    {
      if (counts.Length != 5)
      {
        throw new ArgumentException("Pentanomial counts must have exactly 5 elements.", nameof(counts));
      }

      long numPairs = 0;
      for (int i = 0; i < 5; i++)
      {
        numPairs += counts[i];
      }

      if (numPairs == 0)
      {
        return PentanomialResult.Empty;
      }

      // Mean per-game score (identical to the trinomial mean).
      double mean = 0;
      for (int i = 0; i < 5; i++)
      {
        mean += counts[i] * SCORE_FOR_BUCKET[i];
      }
      mean /= numPairs;

      // Population variance across pairs (in normalized per-game score units).
      double variance = 0;
      for (int i = 0; i < 5; i++)
      {
        double diff = SCORE_FOR_BUCKET[i] - mean;
        variance += counts[i] * diff * diff;
      }
      variance /= numPairs;

      double stdDev = Math.Sqrt(variance);
      double stdError = stdDev / Math.Sqrt(numPairs);

      // Elo point estimate and error margin (let EloDiff yield +/-Inf or NaN on extreme
      // samples exactly as the trinomial path does, so the two are directly comparable).
      float elo = EloCalculator.EloDiff((float)mean);
      float eloMax = EloCalculator.EloDiff((float)(mean + mult * stdError));
      float eloErrorMargin = eloMax - elo;

      // Likelihood of superiority: probability the true mean score exceeds 0.5.
      float los = (float)(0.5 * (1 + ErrorFunction.Erf((mean - 0.5) / (stdError * Math.Sqrt(2)))));

      return new PentanomialResult(counts[0], counts[1], counts[2], counts[3], counts[4],
                                   (int)numPairs, (float)mean, (float)stdDev, (float)stdError,
                                   elo, eloErrorMargin, los);
    }
  }
}
