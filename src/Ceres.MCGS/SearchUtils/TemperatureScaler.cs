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

namespace Ceres.MCGS.Search;

public static class TemperatureScaler
{
  /// <summary>
  /// Scales probabilities in place by the given temperature.
  /// Only the first probabilities.Length entries are assumed to be known; any remaining
  /// probability mass is spread uniformly over the unseen moves so that the output is
  /// properly normalized.
  /// </summary>
  public static void ApplyTemperature(int countAllProbabilities, Span<double> probabilitiesKnown, double sumProbabilities, double temperature)
  {
    int count = probabilitiesKnown.Length;

    double missingMass = 1.0 - sumProbabilities;
    int missingCount = countAllProbabilities - count;
    double alpha = 1.0f / temperature;

    // Raise to power (1/T) and accumulate denominator.
    Span<double> scaled = count <= 64 ? stackalloc double[count] : new double[count];

    double denominator = 0.0f;
    for (int i = 0; i < count; i++)
    {
      if (probabilitiesKnown[i] < 0.0f || probabilitiesKnown[i] > 1.0f)
      {
        throw new ArgumentOutOfRangeException(nameof(probabilitiesKnown), "Probabilities must be in [0,1]");
      }
      double value = (float)Math.Pow(probabilitiesKnown[i], alpha);
      scaled[i] = value;
      denominator += value;
    }

    //  Add the unseen moves' contribution (uniform assumption).
    if (missingCount > 0 && missingMass > 0.0f)
    {
      double uniformRest = missingMass / missingCount;
      double restContributionEach = (float)Math.Pow(uniformRest, alpha);
      denominator += restContributionEach * missingCount;
    }

    if (denominator == 0.0f)
    {
      throw new InvalidOperationException("Denominator is zero; check temperature and inputs.");
    }

    // Normalize and write back in place.
    for (int i = 0; i < count; i++)
    {
      probabilitiesKnown[i] = scaled[i] / denominator;
    }
  }
}
