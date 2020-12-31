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
#endregion

// <copyright file="Gamma.cs" company="Math.NET">
// Math.NET Numerics, part of the Math.NET Project
// http://numerics.mathdotnet.com
// http://github.com/mathnet/mathnet-numerics
//
// Copyright (c) 2009-2014 Math.NET
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// </copyright>


/// <summary>
/// <para>Sampling implementation based on:
/// "A Simple Method for Generating Gamma Variables" - Marsaglia &amp; Tsang
/// ACM Transactions on Mathematical Software, Vol. 26, No. 3, September 2000, Pages 363–372.</para>
/// <para>This method performs no parameter checks.</para>
/// </summary>
/// <param name="rnd">The random number generator to use.</param>
/// <param name="shape">The shape (k, α) of the Gamma distribution. Range: α ≥ 0.</param>
/// <param name="rate">The rate or inverse scale (β) of the Gamma distribution. Range: β ≥ 0.</param>
/// <returns>A sample from a Gamma distributed random variable.</returns>
namespace Ceres.Base.Math.Random
{
  public static class GammaDistribution
  {
    public static double RandomDraw(System.Random rnd, double shape, double rate)
    {
      if (double.IsPositiveInfinity(rate))
      {
        return shape;
      }

      var a = shape;
      var alphafix = 1.0;

      // Fix when alpha is less than one.
      if (shape < 1.0)
      {
        a = shape + 1.0;
        alphafix = System.Math.Pow(rnd.NextDouble(), 1.0 / shape);
      }

      var d = a - (1.0 / 3.0);
      var c = 1.0 / System.Math.Sqrt(9.0 * d);
      while (true)
      {
        var x = NormalRandom(rnd, 0.0, 1.0);
        var v = 1.0 + (c * x);
        while (v <= 0.0)
        {
          x = NormalRandom(rnd, 0.0, 1.0);
          v = 1.0 + (c * x);
        }

        v = v * v * v;
        var u = rnd.NextDouble();
        x = x * x;
        if (u < 1.0 - (0.0331 * x * x))
        {
          return alphafix * d * v / rate;
        }

        if (System.Math.Log(u) < (0.5 * x) + (d * (1.0 - v + System.Math.Log(v))))
        {
          return alphafix * d * v / rate;
        }
      }

      // --------------------------------------------------------------------------------------------
      static double NormalRandom(System.Random rng, double mean = 0, double stdDev = 1)
      {
        double draw1 = 1.0 - rng.NextDouble();
        double draw2 = 1.0 - rng.NextDouble();
        double rStd = System.Math.Sqrt(-2.0
                                      * System.Math.Log(draw1)) 
                                      * System.Math.Sin(2.0 * System.Math.PI * draw2);
        return mean + stdDev * rStd;
      }

    }
  }

}