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
    [ThreadStatic]
    static System.Random rand;

    [ThreadStatic]
    static float[] fractionsScratch = null;

    [ThreadStatic]
    static float[] densitiesScratch = null;

    /// <summary>
    /// Returns a draw according to specified Thompson sampling parameters (with temperature applied).
    /// </summary>
    /// <param name="densities"></param>
    /// <param name="temperature"></param>
    /// <returns></returns>
    public static int Draw(Span<float> densities, int numDensities, float temperature)
    {
      if (fractionsScratch == null || fractionsScratch.Length < numDensities)
      {
        fractionsScratch = new float[System.Math.Max(64, numDensities)];
      }

      float sum = 0;
      for (int i = 0; i < numDensities; i++)
      {
        sum += densities[i];
      }

      float adjust = 1.0f / sum;

      float sumNew = 0;
      for (int i = 0; i < numDensities; i++)
      {
        float value = MathF.Pow((adjust * densities[i]), temperature);
        fractionsScratch[i] = value;
        sumNew += value;
      }

      for (int i = 0; i < numDensities; i++)
      {
        fractionsScratch[i] /= sumNew;
      }

      return Draw(fractionsScratch, numDensities);
    }


    /// <summary>
    /// Returns a draw according to specified Thompson sampling parameters.
    /// </summary>
    /// <param name="densities"></param>
    /// <returns></returns>
    public static int Draw(Span<float> densities, int numDensities)
    {
      if (densitiesScratch == null)
      {
        rand = new System.Random((int)DateTime.Now.Ticks);
        densitiesScratch = new float[64];
      }

      if (densitiesScratch.Length < numDensities)
      {
        densitiesScratch = new float[System.Math.Max(64, numDensities)];
      }

      float cum = 0;
      for (int i = 0; i < numDensities; i++)
      {
        densitiesScratch[i] = cum + densities[i];
        cum += densities[i];
      }

      float draw = (float)rand.NextDouble();
      for (int i = 0; i < numDensities; i++)
      {
        if (densitiesScratch[i] >= draw)
        {
          return i;
        }
      }

      return numDensities - 1;
    }
  }

}