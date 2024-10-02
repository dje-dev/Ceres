#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using Ceres.Base.DataTypes;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Numerics.Tensors;

#endregion

namespace Ceres.Base.Math
{
  /// <summary>
  /// Collection of miscellaneous statistical utility methods.
  /// </summary>
  public static class StatUtils
  {
    /// <summary>
    /// Returns the value bounded within the range [min, max].
    /// </summary>
    /// <param name="value"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public static float Bounded(float value, float min, float max)
    {
      if (value < min)
        return min;
      else if (value > max)
        return max;
      else
        return value;
    }

    /// <summary>
    /// Returns count of number of values in array which are not NaN.
    /// </summary>
    /// <param name="values"></param>
    /// <returns></returns>
    public static int CountNonNaN(float[] values)
    {
      int count = 0;
      for (int i=0; i<values.Length;i++)
      {
        if (!float.IsNaN(values[i]))
        {
          count++;
        }
      }
      return count;
    }


    /// <summary>
    /// Returns average of an array of floats.
    /// </summary>
    /// <param name="xs"></param>
    /// <returns></returns>
    public static float Average(params float[] xs)
    {
      float tot = 0;
      for (int i = 0; i < xs.Length; i++) tot += xs[i];
      return tot / xs.Length;
    }


    /// <summary>
    /// Returns sum of an IList of floats.
    /// </summary>
    /// <param name="xs"></param>
    /// <returns></returns>
    public static float Sum(IList<float> xs)
    {
      double tot = 0;
      for (int i = 0; i < xs.Count; i++) tot += xs[i];
      return (float)tot;
    }

    public static double Min(IList<float> xs)
    {
      float min = float.MaxValue;
      for (int i = 0; i < xs.Count; i++)
        if (xs[i] < min)
          min = xs[i];
      return min;
    }


    /// <summary>
    /// Returns the maximum value in an array of floats.
    /// </summary>
    /// <param name="xs"></param>
    /// <returns></returns>
    public static double Max(float[] xs)
    {
      float max = float.MinValue;
      for (int i = 0; i < xs.Length; i++)
        if (xs[i] > max)
          max = xs[i];
      return max;
    }


    /// <summary>
    /// Returns the index of the element with the maximum value in an array of floats.
    /// </summary>
    /// <param name="f"></param>
    /// <returns></returns>
    public static int IndexOfMax(Span<float> f)
    {
      int bestIndex = 0;
      float bestValue = float.MinValue;
      for (int i = 0; i < f.Length; i++)
      {
        if (f[i] > bestValue)
        {
          bestIndex = i;
          bestValue = f[i];
        }
      }
      return bestIndex;
    }


    /// <summary>
    /// Returns the average value in an IList of floats.
    /// </summary>
    /// <param name="xs"></param>
    /// <returns></returns>
    public static double Average(IList<float> xs)
    {
      float tot = 0;
      for (int i = 0; i < xs.Count; i++) tot += xs[i];
      return tot / xs.Count;
    }

    /// <summary>
    /// Normalizes (make sum be 1.0) a set of values.
    /// </summary>
    /// <param name="values"></param>
    public static void Normalize(Span<float> values)
    {
      float allSum = TensorPrimitives.Sum(values);

      // TODO: someday use Divide next
      //       but currently seems buggy!
      // TensorPrimitives.Divide(values, allSum, values);

      for (int i = 0; i < values.Length; i++)
      {
        values[i] /= allSum;
      }
    }


    /// <summary>
    /// Applies the Log transformation to values in an array, with a minimum value specified.
    /// </summary>
    /// <param name="vals"></param>
    /// <param name="minTruncationValue"></param>
    /// <returns></returns>
    public static float[] LogTruncatedWithMin(float[] vals, float minTruncationValue)
    {
      float[] ret = new float[vals.Length];
      for (int i = 0; i < vals.Length; i++)
      {
        if (vals[i] == 0)
          ret[i] = minTruncationValue;
        else
        {
          float logVal = (float)System.Math.Log(vals[i]);
          if (logVal < minTruncationValue)
            ret[i] = minTruncationValue;
          else
            ret[i] = logVal;
        }
      }
      return ret;
    }


    /// <summary>
    /// Returns the geometic average of an List of shorts.
    /// See: https://en.wikipedia.org/wiki/Geometric_mean
    /// </summary>
    /// <param name="xs"></param>
    /// <returns></returns>
    public static double AverageGeo(IList<short> xs)
    {
      // Compute in numerically stable way via logs
      double tot = 0.0;
      double totLogs = 0.0;
      double absTotLogs = 0.0;
      int numNonPositive = 0;
      for (int i = 0; i < xs.Count; i++)
      {
        tot += xs[i];
        totLogs += System.Math.Log(xs[i]);
        absTotLogs += System.Math.Log(System.Math.Abs(xs[i]));
        if (xs[i] <= 0) numNonPositive++;
      }

      //      Console.WriteLine(tot + " " + totLogs + " " + absTotLogs + " count:" + numNonPositive + " " + Math.Exp(absTotLogs / xs.Count));
      if (numNonPositive > 0)
        return System.Math.Pow(-1, (double)numNonPositive / (double)xs.Count) * System.Math.Exp(absTotLogs / xs.Count);
      else
        return System.Math.Exp(totLogs / xs.Count);
    }


    /// <summary>
    /// Returns the L2 norm of an IList of floats.
    /// </summary>
    /// <param name="vals"></param>
    /// <returns></returns>
    /// <summary>
    /// Computes the L2 norm of a vector.
    /// </summary>
    /// <param name="vals"></param>
    /// <returns></returns>
    public static float NormL2(ReadOnlySpan<float> vals)
    {
      int vectorSize = Vector<float>.Count;

      Vector<float> sum = Vector<float>.Zero;

      int i = 0;
      int lastBlockIndex = vals.Length - (vals.Length % Vector<float>.Count);

      // Process a vector at a time.
      while (i < lastBlockIndex)
      {
        Vector<float> thisSlice = new Vector<float>(vals.Slice(i, vectorSize));
        sum += thisSlice * thisSlice;
        i += vectorSize;
      }

      // Leftover that didn't fit into a vector.
      float totalSum = 0;
      for (; i < vals.Length; i++)
      {
        totalSum += vals[i] * vals[i];
      }

      // Horizontal sum.
      for (int j = 0; j < vectorSize; j++)
      {
        totalSum += sum[j];
      }

      return MathF.Sqrt(totalSum);
    }


    /// <summary>
    /// Returns the standard deviation of an IList of floats.
    /// </summary>
    /// <param name="vals"></param>
    /// <returns></returns>
    public static double StdDev(IList<float> vals)
    {
      double avg = Average(vals);

      double ss = 0;
      for (int i = 0; i < vals.Count; i++) ss += (vals[i] - avg) * (vals[i] - avg);

      return System.Math.Sqrt(ss / vals.Count);
    }

    /// <summary>
    /// Returns the kurtosis of an IList of floats.
    /// </summary>
    /// <param name="vals"></param>
    /// <returns></returns>
    public static double Kurtosis(IList<float> vals)
    {
      double avg = Average(vals);

      double sum2 = 0;
      double sum4 = 0;
      for (int i = 0; i < vals.Count; i++)
      {
        float diff = (float)(vals[i] - avg);
        sum2 += MathF.Pow(diff, 2);
        sum4 += MathF.Pow(diff, 4);
      }

      float variance = (float)sum2 / vals.Count;
      float avgPower4 = (float)sum4 / vals.Count;

      return avgPower4 / (variance * variance);
    }


    /// <summary>
    /// Returns the weighted average an array of doubles.
    /// </summary>
    /// <param name="d"></param>
    /// <param name="wts"></param>
    /// <returns></returns>
    public static double WtdAvg(double[] d, double[] wts)
    {
      double num = 0;
      double den = 0;
      for (int i = 0; i < d.Length; i++)
      {
        num += wts[i] * d[i];
        den += wts[i];
      }
      return num / den;
    }


    /// <summary>
    /// Returns the weighted average an array of floats.
    /// </summary>
    /// <param name="d"></param>
    /// <param name="wts"></param>
    /// <returns></returns>
    public static double WtdAvg(float[] d, double[] wts)
    {
      double num = 0;
      double den = 0;
      for (int i = 0; i < d.Length; i++)
      {
        num += wts[i] * d[i];
        den += wts[i];
      }
      return num / den;
    }


    /// <summary>
    /// Returns the weighted covariance of arrays of doubles.
    /// </summary>
    /// <param name="d1"></param>
    /// <param name="d2"></param>
    /// <param name="wts"></param>
    /// <returns></returns>
    public static double WtdCov(double[] d1, double[] d2, double[] wts)
    {
      double avg1 = WtdAvg(d1, wts);
      double avg2 = WtdAvg(d2, wts);
      double num = 0;
      double den = 0;
      for (int i = 0; i < d1.Length; i++)
      {
        num += wts[i] * (d1[i] - avg1) * (d2[i] - avg2);
        den += wts[i];
      }
      return num / den;
    }


    /// <summary>
    /// Returns the weighted covariance of arrays of floats.
    /// </summary>
    /// <param name="d1"></param>
    /// <param name="d2"></param>
    /// <param name="wts"></param>
    /// <returns></returns>
    public static double WtdCov(float[] d1, float[] d2, double[] wts)
    {
      double avg1 = WtdAvg(d1, wts);
      double avg2 = WtdAvg(d2, wts);
      double num = 0;
      double den = 0;
      for (int i = 0; i < d1.Length; i++)
      {
        num += wts[i] * (d1[i] - avg1) * (d2[i] - avg2);
        den += wts[i];
      }
      return num / den;
    }


    /// <summary>
    /// Returns the weighted correlation of arrays of doubles.
    /// </summary>
    /// <param name="d1"></param>
    /// <param name="d2"></param>
    /// <param name="wts"></param>
    /// <returns></returns>
    public static double CorrelationWeighted(double[] d1, double[] d2, double[] wts)
    {
      return WtdCov(d1, d2, wts) / (System.Math.Sqrt(WtdCov(d1, d1, wts) * WtdCov(d2, d2, wts)));
    }


    /// <summary>
    /// Returns the weighted correlation of arrays of floats.
    /// </summary>
    /// <param name="d1"></param>
    /// <param name="d2"></param>
    /// <param name="wts"></param>
    /// <returns></returns>
    public static double CorrelationWeighted(float[] d1, float[] d2, double[] wts)
    {
      return WtdCov(d1, d2, wts) / (System.Math.Sqrt(WtdCov(d1, d1, wts) * WtdCov(d2, d2, wts)));
    }


    /// <summary>
    /// Returns the mean average absolute deviation of two arrays of floats.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static double MeanAverageAbsoluteDeviation(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
      if (x.Length != y.Length)
      {
        throw new ArgumentException("x and y must be of the same length");
      }

      double sum = 0;
      for (int i = 0; i < x.Length; i++)
      {
        sum += System.Math.Abs(x[i] - y[i]);
      }

      return sum / x.Length;
    }

    /// <summary>
    /// Returns the mean squared difference of two arrays of floats.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static float MeanSquaredError(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
      if (x.Length != y.Length)
      {
        throw new ArgumentException("x and y must be of the same length");
      }

      double sum = 0;
      for (int i = 0; i < x.Length; i++)
      {
        sum += MathF.Pow(x[i] - y[i], 2);
      }

      return (float)sum / x.Length;
    }



    /// <summary>
    /// Returns the correlation of two spans of doubles.
    /// </summary>
    /// <param name="xs"></param>
    /// <param name="ys"></param>
    /// <returns></returns>
    public static double Correlation(ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)
    {
      //TODO: check here that arrays are not null, of the same length etc

      double sx = 0.0;
      double sy = 0.0;
      double sxx = 0.0;
      double syy = 0.0;
      double sxy = 0.0;

      int n = xs.Length;

      for (int i = 0; i < n; ++i)
      {
        double x = xs[i];
        double y = ys[i];

        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
      }

      // covariation
      double cov = sxy / n - sx * sy / n / n;

      // standard error of x
      double sigmax = System.Math.Sqrt(sxx / n - sx * sx / n / n);

      // standard error of y
      double sigmay = System.Math.Sqrt(syy / n - sy * sy / n / n);

      // correlation is just a normalized covariation
      return cov / sigmax / sigmay;
    }



    /// <summary>
    /// Returns the correlation of two Spans of FP16.
    /// </summary>
    /// <param name="xs"></param>
    /// <param name="ys"></param>
    /// <returns></returns>
    public static double Correlation(ReadOnlySpan<FP16> xs, ReadOnlySpan<FP16> ys)
    {
      //TODO: check here that arrays are not null, of the same length etc

      double sx = 0.0;
      double sy = 0.0;
      double sxx = 0.0;
      double syy = 0.0;
      double sxy = 0.0;

      int n = xs.Length;

      for (int i = 0; i < n; ++i)
      {
        double x = xs[i];
        double y = ys[i];

        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
      }

      // covariation
      double cov = sxy / n - sx * sy / n / n;

      // standard error of x
      double sigmax = System.Math.Sqrt(sxx / n - sx * sx / n / n);

      // standard error of y
      double sigmay = System.Math.Sqrt(syy / n - sy * sy / n / n);

      // correlation is just a normalized covariation
      return cov / sigmax / sigmay;
    }


    /// <summary>
    /// Returns the correlation of two Spans of floats.
    /// </summary>
    /// <param name="xs"></param>
    /// <param name="ys"></param>
    /// <returns></returns>
    public static double Correlation(ReadOnlySpan<float> xs, ReadOnlySpan<float> ys)
    {
      if (xs.Length == 0)
      {
        throw new Exception("No data provided to Correlation");
      }
      if (xs.Length != ys.Length)
      {
        throw new ArgumentException("Spans not of same length");
      }

      if (xs.Length == 1)
      {
        return 1;
      }

      double sx = 0.0;
      double sy = 0.0;
      double sxx = 0.0;
      double syy = 0.0;
      double sxy = 0.0;

      int n = xs.Length;

      for (int i = 0; i < n; ++i)
      {
        double x = xs[i];
        double y = ys[i];

        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
      }

      // covariation
      double cov = sxy / n - sx * sy / n / n;

      // standard error of x
      double sigmax = System.Math.Sqrt(sxx / n - sx * sx / n / n);

      // standard error of y
      double sigmay = System.Math.Sqrt(syy / n - sy * sy / n / n);

      if (cov == 0 || sigmax == 0 || sigmay == 0)
      {
        return 0;
      }

      // correlation is just a normalized covariation
      return cov / sigmax / sigmay;
    }


    /// <summary>
    /// Returns rank correlation of two arrays.
    /// </summary>
    /// <param name="values1"></param>
    /// <param name="values2"></param>
    /// <returns></returns>
    public static float RankCorrelation(Span<float> values1, Span<float> values2) => (float)Correlation(ToRanks(values1), ToRanks(values2));


    /// <summary>
    /// Returns array of ranks of values in specified array (starting with 0).
    /// </summary>
    /// <param name="values"></param>
    /// <returns></returns>
    public static float[] ToRanks(Span<float> values)
    {
      int N = values.Length;

      float[] ranks = new float[N];

      for (int i = 0; i < N; i++)
      {
        int rank1 = 0;
        int rank2 = 0;

        for (int j = 0; j < i; j++)
        {
          if (values[j] < values[i])
          {
            rank1++;
          }
          else if (values[j] == values[i])
          {
            rank2++;
          }
        }

        for (int j = i + 1; j < N; j++)
        {
          if (values[j] < values[i])
          {
            rank1++;
          }
          else if (values[j] == values[i])

          {
            rank2++;
          }
        }

        ranks[i] = rank1 + (rank2 - 1f) * 0.5f;
      }


      return ranks;
    }


    #region Probabilities

    /// <summary>
    /// Converts values to logits.
    /// </summary>
    public static float[] ToLogits(float[] f)
    {
      float[] ret = new float[f.Length];
      for (int i = 0; i < f.Length; i++)
      {
        ret[i] = MathF.Log(f[i] + 1E-10f);
      }
      return ret;
    }

    /// <summary>
    /// Computes softmax function.
    /// </summary>
    public static float[] Softmax(float[] f)
    {
      float acc = 0;
      for (int i = 0; i < f.Length; i++)
      {
        acc += MathF.Exp(f[i]);
      }

      float[] ret = new float[f.Length];
      for (int i = 0; i < f.Length; i++)
      {
        ret[i] = MathF.Exp(f[i]) / acc;
      }
      return ret;
    }


    /// <summary>
    /// Returns entropy of a probability distribution.
    /// </summary>
    /// <param name="f"></param>
    /// <returns></returns>
    public static float Entropy(Span<float> f)
    {
      float acc = 0;
      for (int i=0; i<f.Length;i++)
      {
        float value = f[i];
        if (value > 0)
        {
          acc += value * System.MathF.Log(value);
        }
      }

      return -acc;
    }


    /// <summary>
    /// Equivalent to Pytorch CrossEntropy function.
    /// </summary>
    public static float SoftmaxCrossEntropyWithLogits(float[] labels, float[] logits)
    {
      float[] probs = Softmax(logits);
      return SoftmaxCrossEntropy(labels, probs);
    }


    /// <summary>
    /// Computes cross entropy, where the labels are targets in probability space
    /// and are compared against logit values.
    /// </summary>
    public static float SoftmaxCrossEntropy(float[] labels, float[] logits)
    {
      float acc = 0;
      float sumLabels = 0;
      for (int i = 0; i < labels.Length; i++)
      {
        sumLabels += labels[i];
        acc += labels[i] * MathF.Log(logits[i] + 0.0000001f);
      }

#if DEBUG
      if (sumLabels < 0.98f || sumLabels > 1.02f)
      {
        Console.WriteLine($"WARNING: SoftmaxCrossEntropy passed labels which sum to {sumLabels} rather than add to 1 as expected.");
      }
#endif

      return -acc;
    }

    #endregion

    #region Regression

    /// <summary>
    /// Returns parameters of a univariate linear regresssion from specified data.
    /// Based closely upon https://gist.github.com/NikolayIT/d86118a3a0cb3f5ed63d674a350d75f2 (MIT License).
    /// </summary>
    /// <param name="xVals">The x-axis values.</param>
    /// <param name="yVals">The y-axis values.</param>
    /// <param name="rSquared">The r^2 value of the line.</param>
    /// <param name="yIntercept">The y-intercept value of the line (i.e. y = ax + b, yIntercept is b).</param>
    /// <param name="slope">The slope of the line (i.e. y = ax + b, slope is a).</param>
    public static (float rSquared, float yIntercept, float slope) LinearRegression(float[] xVals, float[] yVals)
    {
      if (xVals.Length != yVals.Length)
      {
        throw new Exception("Regression failure, X and Y not same cardinality.");
      }

      float sumOfX = 0;
      float sumOfY = 0;
      float sumOfXSq = 0;
      float sumOfYSq = 0;
      float sumCodeviates = 0;

      for (int i = 0; i < xVals.Length; i++)
      {
        float x = xVals[i];
        float y = yVals[i];
        sumCodeviates += x * y;
        sumOfX += x;
        sumOfY += y;
        sumOfXSq += x * x;
        sumOfYSq += y * y;
      }

      float count = xVals.Length;
      float ssX = sumOfXSq - ((sumOfX * sumOfX) / count);

      float rNumerator = (count * sumCodeviates) - (sumOfX * sumOfY);
      float rDenom = (count * sumOfXSq - (sumOfX * sumOfX)) * (count * sumOfYSq - (sumOfY * sumOfY));
      float sCo = sumCodeviates - ((sumOfX * sumOfY) / count);

      float meanX = sumOfX / count;
      float meanY = sumOfY / count;
      float dblR = rNumerator / MathF.Sqrt(rDenom);

      float rSquared = dblR * dblR;
      float yIntercept = meanY - ((sCo / ssX) * meanX);
      float slope = sCo / ssX;

      return (rSquared, yIntercept, slope);
    }


    /// <summary>
    /// Returns an array of the absolute differences between two spans of floats.
    /// </summary>
    /// <param name="finalQ1"></param>
    /// <param name="finalQ2"></param>
    /// <returns></returns>
    public static float[] AbsDiff(Span<float> finalQ1, Span<float> finalQ2)
    {
      float[] ret = new float[finalQ1.Length];
      for (int i = 0; i < finalQ1.Length; i++)
      {
        ret[i] = MathF.Abs(finalQ1[i] - finalQ2[i]);
      }

      return ret;
    }
    
  }

  #endregion
}


