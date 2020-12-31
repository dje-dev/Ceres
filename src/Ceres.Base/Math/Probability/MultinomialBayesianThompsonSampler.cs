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

namespace Ceres.Base.Math.Probability
{
  /// <summary>
  /// Collects statistics (sample counts) from a multinomial distribtion
  /// and then can return random draws from the multinomial distribution.
  /// </summary>
  [Serializable]
  public class MultinomialBayesianThompsonSampler
  {
    public delegate float SupplementalStandardDeviationDelegate(float baselineStdDev, int sampleCount);

    /// <summary>
    /// Underlying Bayesian normal estimator.
    /// </summary>
    public BayesianNormalEstimator[] childrenDistributions;

    private int numSamples;

    /// <summary>
    /// Returns number of samples added so far.
    /// </summary>
    public int NumSamples { get => numSamples; private set => numSamples = value; }

    /// <summary>
    /// Constructor for a multinomial distribution over a specified number of bins.
    /// </summary>
    /// <param name="numChildren"></param>
    public MultinomialBayesianThompsonSampler(int numChildren)
    {
      childrenDistributions = new BayesianNormalEstimator[numChildren];
      for (int i = 0; i < numChildren; i++)
        childrenDistributions[i] = new BayesianNormalEstimator(0, 1, 1.2, 0.01);
    }


    /// <summary>
    /// Recores a new sample in a given bin (or multiple in same bin).
    /// </summary>
    /// <param name="index"></param>
    /// <param name="sampleValue"></param>
    /// <param name="numSamples"></param>
    public void AddSample(int index, double sampleValue, int numSamples)
    {
      childrenDistributions[index].AddSample(sampleValue, numSamples);
      this.numSamples += numSamples;
    }

    /// <summary>
    /// Returns set of random samples.
    /// </summary>
    /// <param name="preferMax"></param>
    /// <param name="numIndicesToConsider"></param>
    /// <param name="numSamples"></param>
    /// <param name="sdScalingFactor"></param>
    /// <param name="supplementalSD"></param>
    /// <returns></returns>
    public int[] GetIndicesOfBestSamples(bool preferMax, int numIndicesToConsider, int numSamples, double sdScalingFactor, SupplementalStandardDeviationDelegate supplementalSD = null)
    {
      int[] sampleCounts = new int[childrenDistributions.Length];
      for (int i = 0; i < numSamples; i++)
        sampleCounts[GetIndexOfBestSample(preferMax, numIndicesToConsider, sdScalingFactor, supplementalSD)]++;
      return sampleCounts;
    }


    /// <summary>
    /// Draws s single random sample and returns it index.
    /// </summary>
    /// <param name="preferMax"></param>
    /// <param name="numIndicesToConsider"></param>
    /// <param name="sdScalingFactor"></param>
    /// <param name="supplementalSD"></param>
    /// <returns></returns>
    public int GetIndexOfBestSample(bool preferMax, int numIndicesToConsider, double sdScalingFactor, SupplementalStandardDeviationDelegate supplementalSD = null)
    {
      if (childrenDistributions.Length == 1) return 0;

      int bestIndex = -1;
      float bestValue = preferMax ? float.MinValue : float.MaxValue;
      for (int i = 0; i < numIndicesToConsider; i++)
      {
        Func<double, double> supplementalSDFunc = null;
        if (supplementalSD != null)
          supplementalSDFunc = (double baselineSD) => supplementalSD((float)baselineSD, childrenDistributions[i].NumSamples);
        double sample = childrenDistributions[i].GenerateSample(sdScalingFactor, supplementalSDFunc);
        if ((preferMax && sample > bestValue) 
        || (!preferMax && sample < bestValue))
        {
          bestValue = (float)sample;
          bestIndex = i;
        }
      }

      return bestIndex;
    }
  }
}
