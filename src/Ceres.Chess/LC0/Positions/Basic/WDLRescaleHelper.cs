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

#endregion

namespace Ceres.Chess.LC0.Positions
{
  /// <summary>
  /// This rescaling logic is a transliteration of the code from The Leela Chess Zero project:
  ///   https://github.com/LeelaChessZero/lc0/src/mcts/search.cc%3E
  /// </summary>
  public static class WDLRescaleHelper
  {
    private static float Logistic(float x) => 1.0f / (1.0f + MathF.Exp(-x));


    /// <summary>
    /// Convert from WDL to centipawns.
    /// </summary>
    /// <param name="v"></param>
    /// <param name="d"></param>
    /// <param name="wdlRescaleRatio"></param>
    /// <param name="wdlRescaleDiff"></param>
    /// <param name="sign"></param>
    /// <param name="invert"></param>
    /// <param name="maxReasonableS">safeguard against unrealistically broad WDL distributions coming from the NN</param>
    /// <returns></returns>
    public static (float centipawns, float newV, float newD) WDLRescale(float v,
                                                                        float d,
                                                                        float wdlRescaleRatio = 1,
                                                                        float wdlRescaleDiff = 0,
                                                                        float sign = 1,
                                                                        bool invert = false,
                                                                        float maxReasonableS = 1.4f)

    {
      if (invert)
      {
        wdlRescaleDiff = -wdlRescaleDiff;
        wdlRescaleRatio = 1.0f / wdlRescaleRatio;
      }

      var w = (1 + v - d) / 2;
      var l = (1 - v - d) / 2;

      // Safeguard against numerical issues; skip WDL transformation if WDL is too extreme.
      const float eps = 0.0001f;
      if (w > eps
          && d > eps
          && l > eps
          && w < (1.0f - eps)
          && d < (1.0f - eps)
          && l < (1.0f - eps))
      {
        float a = MathF.Log(1 / l - 1);
        float b = MathF.Log(1 / w - 1);
        float s = 2 / (a + b);

        // Safeguard against unrealistically broad WDL distributions coming from the NN.
        if (!invert)
        {
          s = MathF.Min(maxReasonableS, s);
        }

        float mu = (a - b) / (a + b);
        float sNew = s * wdlRescaleRatio;

        if (invert)
        {
          float temp = s;
          s = sNew;
          sNew = temp;
          s = Math.Min(maxReasonableS, s);
        }

        float muNew = mu + sign * s * s * wdlRescaleDiff;
        float wNew = Logistic((-1.0f + muNew) / sNew);
        float lNew = Logistic((-1.0f - muNew) / sNew);

        v = wNew - lNew;
        d = MathF.Max(0.0f, 1.0f - wNew - lNew);

        return (100 * muNew, v, d);
      }

      return (float.NaN, float.NaN, float.NaN);
    }
  }
}
