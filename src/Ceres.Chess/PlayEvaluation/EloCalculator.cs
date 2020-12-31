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
  /// A set of static methods that compute Elo chess ranking statistics,
  /// including confidence intervals and likelihood of superiority (LOS).
  /// </summary>
  public class EloCalculator
  {
    /// <summary>
    /// Returns relative Elo for a player having a specified win percentage.
    /// </summary>
    /// <param name="winPercentage"></param>
    /// <returns></returns>
    public static float EloDiff(float winPercentage)
    {
      float eloDiff = -400 * MathF.Log10(1 / winPercentage - 1);
      return eloDiff;
    }

    /// <summary>
    /// Returns relative Elo for a player scoring a specified win/draw/loss count.
    /// </summary>
    /// <param name="wins"></param>
    /// <param name="draws"></param>
    /// <param name="losses"></param>
    /// <returns></returns>
    public static float EloDiff(int wins, int draws, int losses)
    {
      float winPercentage = (float)(wins + 0.5 * draws) / (float)(wins + losses + draws);
      return EloDiff(winPercentage);
    }

    /// <summary>
    /// Returns likelihood of superiority of a player scoring a specified win/draw/loss count.
    /// </summary>
    /// <param name="wins"></param>
    /// <param name="draws"></param>
    /// <param name="losses"></param>
    /// <returns></returns>
    public static float LikelihoodSuperiority(int wins, int draws, int losses)
    {
      // See: https://www.chessprogramming.org/Match_Statistics;  
      return 0.5f * (1 + (float)ErrorFunction.Erf((wins - losses) / MathF.Sqrt(2 * wins + 2 * losses)));
    }


    /// <summary>
    /// Returns the mean and confidence intervals of relative Elo for a player scoring
    /// a specified win/draw/loss count.
    /// </summary>
    /// <param name="wins"></param>
    /// <param name="draws"></param>
    /// <param name="losses"></param>
    /// <param name="stdDev"></param>
    /// <returns></returns>
    public static (float min, float avg, float max) EloConfidenceInterval(int wins, int draws, int losses, float stdDev = 1.0f)
    {
      float count = wins + draws + losses;
      float score = (wins + (float)draws / 2) / count;
      float stdev = MathF.Sqrt((wins * MathF.Pow(1 - score, 2)
                               + losses * MathF.Pow(score, 2)
                               + draws * MathF.Pow(0.5f - score, 2)) / count);
      float min = score - stdDev * stdev / MathF.Sqrt(count);
      float max = score + stdDev * stdev / MathF.Sqrt(count);

      return (EloDiff(min), EloDiff(score), EloDiff(max));
    }

  }
}
