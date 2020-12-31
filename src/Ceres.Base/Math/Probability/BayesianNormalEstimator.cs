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
using System.Collections.Generic;
using System.Text;

#endregion

namespace Ceres.Base.Math.Probability
{
  /// <summary>
  /// Computes Bayesian estimate of parameters of a Gaussian distribution 
  /// which is sampled online (sequential updates) as a Normal-Gamma distribution
  /// to model uncertainty in the mean and variance.
  /// </summary>
  public class BayesianNormalEstimator
  {
    /// <summary>
    /// Number of samples accumulated so far.
    /// </summary>
    public int NumSamples;

    /// <summary>
    /// Mu parameter of distribution.
    /// </summary>
    double Mu;

    /// <summary>
    /// Lambda parameter of distribution.
    /// </summary>
    double Lambda;

    /// <summary>
    /// Alpha distribution parameter of distribution.
    /// </summary>
    double Alpha;

    /// <summary>
    /// Beta parameter of distribution.
    /// </summary>
    double Beta;

    /// <summary>
    /// Constructor (specifying parameters of distribution).
    /// </summary>
    /// <param name="mu"></param>
    /// <param name="lambda"></param>
    /// <param name="alpha"></param>
    /// <param name="beta"></param>
    public BayesianNormalEstimator(double mu, double lambda, double alpha, double beta)
    {
      Mu = mu;
      Lambda = lambda;
      Alpha = alpha;
      Beta = beta;
    }

    /// <summary>
    /// Updates parameters of distribution based on a specified additional sample.
    /// </summary>
    /// <param name="sample"></param>
    /// <param name="numTimes"></param>
    public void AddSample(double sample, int numTimes = 1)
    {
      // Online Bayesian sequential update
      if (numTimes == 1)
      {
        Alpha += 0.5;
        double squaredDeviation = System.Math.Pow(sample - Mu, 2);
        Beta += (Lambda * squaredDeviation / (Lambda + 1)) / 2.0;
        Mu = (Lambda * Mu + sample) / (Lambda + 1f);
        Lambda += 1.0;

        NumSamples++;
      }
      else
      {
        // Online Bayesian sequential update
        Alpha += (numTimes * 0.5);
        double squaredDeviation = System.Math.Pow(sample - Mu, 2);

        for (int i = 0; i < numTimes; i++) // TODO: make more efficient
        {
          Beta += (Lambda * squaredDeviation / (Lambda + 1)) / 2.0;
          Mu = (Lambda * Mu + sample) / (Lambda + 1f);
        }
        Lambda += (numTimes * 1.0);

        NumSamples += numTimes;
      }
    }

    /// <summary>
    /// Generates a random sample from the distribution based on 
    /// current distribution parameters.
    /// </summary>
    /// <param name="sdScalingFactor"></param>
    /// <param name="incrementalStandardDeviation"></param>
    /// <returns></returns>
    public double GenerateSample(double sdScalingFactor = 1.0, Func<double, double> incrementalStandardDeviation = null)
    {
      double tau = DistributionRandomDraws.GammaRandomDraw(Alpha, Beta);
      double variance = Lambda / (Lambda * tau);
      double incrementalSDValue = 0;
      if (incrementalStandardDeviation != null) incrementalSDValue = incrementalStandardDeviation(System.Math.Sqrt(variance));
      double stdDev = System.Math.Sqrt(variance) + incrementalSDValue;

      return DistributionRandomDraws.GaussianRandomDraw(Mu, stdDev * sdScalingFactor);

    }

    /// <summary>
    /// Returns center of distribution.
    /// </summary>
    public double Mean => Mu;

    /// <summary>
    /// Returns standard deviation of the distribution.
    /// </summary>
    public double StdDev => System.Math.Sqrt(Lambda * Beta / (Lambda * (Alpha - 1)));


    /// <summary>
    /// Returns string summary of distribution.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      StringBuilder sb = new StringBuilder();
      sb.AppendLine($"Normal-gamma distribution with {NumSamples} samples (mean ={Mean,6:F3} std_dev ={StdDev,6:F3})");
      sb.AppendLine("Mu     " + Mu);
      sb.AppendLine("Lambda " + Lambda);
      sb.AppendLine("Alpha  " + Alpha);
      sb.AppendLine("Beta   " + Beta);
      return sb.ToString();
    }

    
    public static void Test()
    {
      List<float> vals = new List<float>();

      BayesianNormalEstimator bayes = new BayesianNormalEstimator(0, 1, 1.2, 0.01);
      for (int i = 0; i < 200; i++)
      {
        double r = DistributionRandomDraws.GaussianRandomDraw(-2, 0.3);
        bayes.AddSample(r);
      }

      Console.WriteLine(bayes.ToString());

      Console.WriteLine();
      for (int j = 0; j < 100; j++)
      {
        double x = bayes.GenerateSample();
        vals.Add((float)x);
      }

      Console.WriteLine(StatUtils.Average(vals));
      Console.WriteLine(StatUtils.StdDev(vals));
    }

  }
}
