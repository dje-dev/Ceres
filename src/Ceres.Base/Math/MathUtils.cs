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

using Ceres.Base.DataType;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

#endregion

namespace Ceres.Base.Math
{
  /// <summary>
  /// Collection of miscellaneous mathematical utility methods.
  /// </summary>
  public static class MathUtils
  {
    #region Half conversions

    /// <summary>
    /// Returns array of Half converted from array of float.
    /// </summary>
    /// <param name="floats"></param>
    /// <returns></returns>
    public static Half[] ToHalf(float[] floats)
    {
      if (floats == null) return null;

      Half[] ret = new Half[floats.Length];
      for (int i=0; i<ret.Length;i++)
      {
        ret[i] = (Half)floats[i];
      }
      return ret;
    }

    /// <summary>
    /// Returns array of float converted from array of Half.
    /// </summary>
    /// <param name="floats"></param>
    /// <param name="maxElements"></param>
    /// <returns></returns>
    public static float[] ToFloat(float[] halfs, int? maxElements = null)
    {
      if (halfs == null) return null;
      
      int length = System.Math.Min(halfs.Length, maxElements.HasValue ? maxElements.Value : int.MaxValue);

      float[] ret = new float[length];
      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = (float)halfs[i];
      }
      return ret;
    }

    /// <summary>
    /// Returns array of float converted from Span of Half.
    /// </summary>
    /// <param name="floats"></param>
    /// <param name="maxElements"></param>
    /// <returns></returns>
    public static float[] ToFloat(Span<Half> halfs, int? maxElements = null)
    {
      if (halfs == null) return null;

      int length = System.Math.Min(halfs.Length, maxElements.HasValue ? maxElements.Value : int.MaxValue);

      float[] ret = new float[length];
      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = (float)halfs[i];
      }
      return ret;
    }


    /// <summary>
    /// Returns float[] converted from array of Half.
    /// </summary>
    /// <param name="floats"></param>
    /// <returns></returns>
    public unsafe static float[,] ToFloat2D(Span<Half> data, int numRows, int numColumns)
    {
      float[,] ret = new float[numRows, numColumns];
      for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numColumns; j++)
          ret[i, j] = (float)data[i * numColumns + j];
      return ret;
    }

    #endregion


    /// <summary>
    /// Returns maximum of two positive integers
    /// in a branchless way (using a bit hack).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int MaxOfPositivesFast(int x, int y)
    {
      // Only works of most significant bit zero
      Debug.Assert(x >= 0 && y >= 0);

      int sub = x - y;
      int shift = sub >> 31;
      return x - (sub & shift);
    }

    /// <summary>
    /// Returns minimum of two positive integers
    /// in a branchless way (using a bit hack).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int MinOfPositivesFast(int x, int y)
    {
      // Only works of most significant bit zero
      Debug.Assert(x >= 0 && y >= 0);

      int sub = x - y;
      int shift = sub >> 31;
      return y + (sub & shift);
    }


    /// <summary>
    /// Returns the index of the first nonzero short in a span of shorts.
    /// </summary>
    /// <param name="ints"></param>
    /// <param name="numToCheck"></param>
    /// <returns></returns>
    public static int IndexOfFirstNonzero(Span<short> ints, int numToCheck)
    {
      for (int i = 0; i < numToCheck; i++)
        if (ints[i] > 0)
          return i;
      throw new NotImplementedException();

    }

    /// <summary>
    /// Rounds up a value, returning the first long which is at least minSize
    /// and also a multiple of toMultiplieOf.
    /// </summary>
    /// <param name="minSize"></param>
    /// <param name="toMultipleOf"></param>
    /// <returns></returns>
    public static long RoundedUp(long minSize, long toMultipleOf)
    {
      if (toMultipleOf > 1)
      {
        long modulo = (minSize % toMultipleOf);
        if (modulo != 0) minSize += toMultipleOf - modulo;
      }

      return minSize;
    }


    /// <summary>
    /// Returns a uniform discrete distribution.
    /// </summary>
    /// <param name="length"></param>
    /// <returns></returns>
    public static float[] Uniform(int length)
    {
      float[] ret = new float[length];
      Array.Fill<float>(ret, 1.0f / length);
      return ret;
    }


    /// <summary>
    /// Percent difference between two values.
    /// </summary>
    /// <param name="d1"></param>
    /// <param name="d2"></param>
    /// <returns></returns>
    public static double PctDifference(double d1, double d2)
    {
      if ((d1 != 0.0) && (d2 == 0.0)) return 100.0;
      if ((d2 != 0.0) && (d1 == 0.0)) return 100.0;

      return ((d1 / d2) - 1.0) * 100;
    }

    /// <summary>
    /// Computes the sigmoid function in an numerically stable manner.
    /// </summary>
    /// <param name="v"></param>
    /// <returns></returns>
    public static float SigmoidNumericallyStable(float v)
    {
      if (v >= 0)
      {
        double z = System.Math.Exp(-v);
        return (float)(1.0 / (1.0 + z));
      }
      else
      {
        double z = System.Math.Exp(v);
        return (float)(z / (1.0 + z));
      }
    }


    /// <summary>
    /// Returns a copy of the input matrix which is reblocked
    /// </summary>
    /// <param name="aIn"></param>
    /// <param name="aOut"></param>
    /// <param name="numIn"></param>
    /// <param name="numOutAtATime"></param>
    /// <param name="numInAtATime"></param>
    public static float[] BlockedMatrix(float[] aIn, int numOut, int numIn, int numOutAtATime, int numInAtATime)
    {
      Span<float> aOut = new SpanAligned<float>(aIn.Length, 128).Span;

      int dstIndex = 0;
      for (int i = 0; i < numOut; i += numOutAtATime)
        for (int j = 0; j < numIn; j += numInAtATime)
        {
          for (int ofsI = 0; ofsI < numOutAtATime; ofsI++)
          {
            for (int ofsJ = 0; ofsJ < numInAtATime; ofsJ++)
            {
              int col = i + ofsI;
              int row = j + ofsJ;

              aOut[dstIndex++] = aIn[col * numIn + row];
            }
          }
        }
      return aOut.ToArray();
    }


    /// <summary>
    /// Computes the exponential function in very efficient
    /// (and only very slightly inaccurate) way.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]    
    public static double FastExp(double x)
    {
      // The below not well behaved for very small x
      if (x < -10) return 0;

      long tmp = (long)(1512775 * x + 1072632447);
      int index = (int)(tmp >> 12) & 0xFF;
      return BitConverter.Int64BitsToDouble(tmp << 32) * ExpAdjustment[index];
    }
    

    static readonly double[] ExpAdjustment = new double[256] {
            1.040389835,
            1.039159306,
            1.037945888,
            1.036749401,
            1.035569671,
            1.034406528,
            1.033259801,
            1.032129324,
            1.031014933,
            1.029916467,
            1.028833767,
            1.027766676,
            1.02671504,
            1.025678708,
            1.02465753,
            1.023651359,
            1.022660049,
            1.021683458,
            1.020721446,
            1.019773873,
            1.018840604,
            1.017921503,
            1.017016438,
            1.016125279,
            1.015247897,
            1.014384165,
            1.013533958,
            1.012697153,
            1.011873629,
            1.011063266,
            1.010265947,
            1.009481555,
            1.008709975,
            1.007951096,
            1.007204805,
            1.006470993,
            1.005749552,
            1.005040376,
            1.004343358,
            1.003658397,
            1.002985389,
            1.002324233,
            1.001674831,
            1.001037085,
            1.000410897,
            0.999796173,
            0.999192819,
            0.998600742,
            0.998019851,
            0.997450055,
            0.996891266,
            0.996343396,
            0.995806358,
            0.995280068,
            0.99476444,
            0.994259393,
            0.993764844,
            0.993280711,
            0.992806917,
            0.992343381,
            0.991890026,
            0.991446776,
            0.991013555,
            0.990590289,
            0.990176903,
            0.989773325,
            0.989379484,
            0.988995309,
            0.988620729,
            0.988255677,
            0.987900083,
            0.987553882,
            0.987217006,
            0.98688939,
            0.98657097,
            0.986261682,
            0.985961463,
            0.985670251,
            0.985387985,
            0.985114604,
            0.984850048,
            0.984594259,
            0.984347178,
            0.984108748,
            0.983878911,
            0.983657613,
            0.983444797,
            0.983240409,
            0.983044394,
            0.982856701,
            0.982677276,
            0.982506066,
            0.982343022,
            0.982188091,
            0.982041225,
            0.981902373,
            0.981771487,
            0.981648519,
            0.981533421,
            0.981426146,
            0.981326648,
            0.98123488,
            0.981150798,
            0.981074356,
            0.981005511,
            0.980944219,
            0.980890437,
            0.980844122,
            0.980805232,
            0.980773726,
            0.980749562,
            0.9807327,
            0.9807231,
            0.980720722,
            0.980725528,
            0.980737478,
            0.980756534,
            0.98078266,
            0.980815817,
            0.980855968,
            0.980903079,
            0.980955475,
            0.981017942,
            0.981085714,
            0.981160303,
            0.981241675,
            0.981329796,
            0.981424634,
            0.981526154,
            0.981634325,
            0.981749114,
            0.981870489,
            0.981998419,
            0.982132873,
            0.98227382,
            0.982421229,
            0.982575072,
            0.982735318,
            0.982901937,
            0.983074902,
            0.983254183,
            0.983439752,
            0.983631582,
            0.983829644,
            0.984033912,
            0.984244358,
            0.984460956,
            0.984683681,
            0.984912505,
            0.985147403,
            0.985388349,
            0.98563532,
            0.98588829,
            0.986147234,
            0.986412128,
            0.986682949,
            0.986959673,
            0.987242277,
            0.987530737,
            0.987825031,
            0.988125136,
            0.98843103,
            0.988742691,
            0.989060098,
            0.989383229,
            0.989712063,
            0.990046579,
            0.990386756,
            0.990732574,
            0.991084012,
            0.991441052,
            0.991803672,
            0.992171854,
            0.992545578,
            0.992924825,
            0.993309578,
            0.993699816,
            0.994095522,
            0.994496677,
            0.994903265,
            0.995315266,
            0.995732665,
            0.996155442,
            0.996583582,
            0.997017068,
            0.997455883,
            0.99790001,
            0.998349434,
            0.998804138,
            0.999264107,
            0.999729325,
            1.000199776,
            1.000675446,
            1.001156319,
            1.001642381,
            1.002133617,
            1.002630011,
            1.003131551,
            1.003638222,
            1.00415001,
            1.004666901,
            1.005188881,
            1.005715938,
            1.006248058,
            1.006785227,
            1.007327434,
            1.007874665,
            1.008426907,
            1.008984149,
            1.009546377,
            1.010113581,
            1.010685747,
            1.011262865,
            1.011844922,
            1.012431907,
            1.013023808,
            1.013620615,
            1.014222317,
            1.014828902,
            1.01544036,
            1.016056681,
            1.016677853,
            1.017303866,
            1.017934711,
            1.018570378,
            1.019210855,
            1.019856135,
            1.020506206,
            1.02116106,
            1.021820687,
            1.022485078,
            1.023154224,
            1.023828116,
            1.024506745,
            1.025190103,
            1.02587818,
            1.026570969,
            1.027268461,
            1.027970647,
            1.02867752,
            1.029389072,
            1.030114973,
            1.030826088,
            1.03155163,
            1.032281819,
            1.03301665,
            1.033756114,
            1.034500204,
            1.035248913,
            1.036002235,
            1.036760162,
            1.037522688,
            1.038289806,
            1.039061509,
            1.039837792,
            1.040618648
        };

  }
}
