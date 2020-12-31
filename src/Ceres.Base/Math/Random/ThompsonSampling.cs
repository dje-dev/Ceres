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
#endregion

namespace Ceres.Base.Math.Random
{
  /// <summary>
  /// Implements Thompson sampling method for choosing action 
  /// which provides optimal exploitation/exporation balance.
  /// </summary>
  public static class ThompsonSampling
  {
    static System.Random rand = new System.Random((int)DateTime.Now.Ticks);


    /// <summary>
    /// Returns a draw according to specified Thompson sampling parameters (with temperature applied).
    /// </summary>
    /// <param name="densities"></param>
    /// <param name="temperature"></param>
    /// <returns></returns>
    public static int Draw(float[] densities, float temperature)
    {
      float sum = StatUtils.Sum(densities);
      float adjust = 1.0f / sum;

      float[] fractionsWithTemperature = new float[densities.Length];
      for (int i = 0; i < densities.Length; i++)
        fractionsWithTemperature[i] = MathF.Pow((adjust * densities[i]), temperature);

      sum = StatUtils.Sum(fractionsWithTemperature);
      for (int i = 0; i < fractionsWithTemperature.Length; i++)
        fractionsWithTemperature[i] /= sum;

      return Draw(fractionsWithTemperature);
    }


    /// <summary>
    /// Returns a draw according to specified Thompson sampling parameters.
    /// </summary>
    /// <param name="densities"></param>
    /// <returns></returns>
    public static int Draw(float[] densities)
    {
      Debug.Assert(MathF.Abs(1.0f - StatUtils.Sum(densities)) < 0.001f);

      float[] cums = new float[densities.Length];
      float cum = 0;
      for (int i = 0; i < densities.Length; i++)
      {
        cums[i] = cum + densities[i];
        cum += densities[i];
      }

      float draw = (float)rand.NextDouble();
      for (int i = 0; i < densities.Length; i++)
        if (cums[i] >= draw)
          return i;

      return densities.Length - 1;
    }
  }

}