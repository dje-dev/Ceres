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

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.Math
{
  /// <summary>
  /// Efficient routines for computing logarithms in various bases.
  /// Performance seems about 20% improved over MathF class.
  /// 
  /// From: https://gist.github.com/r1pper/b8f4620a8f7718b16df9 (Hessam Jalali)
  /// </summary>
  public static class FastLog
  {
    private static float[] MantissaLogs;
    private const float Base10 = 3.321928F;
    private const float BaseE = 1.442695F;

    /// <summary>
    /// Returns logarithm (base 2) of specified value.
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Log2(float value)
    {
      if (value == 0F) return float.NegativeInfinity;

      Ieee754 number = new Ieee754 { Single = value };

      if (number.UnsignedBits >> 31 == 1) //NOTE: didn't call Sign property for higher performance
        return float.NaN;

      return (((number.SignedBits >> 23) & 0xFF) - 127) + MantissaLogs[number.UnsignedBits & 0x007FFFFF];
      //NOTE: didn't call Exponent and Mantissa properties for higher performance
    }


    /// <summary>
    /// Returns fast Log (base 10) of specified value.
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    public static float Log10(float value) => Log2(value) / Base10;


    /// <summary>
    /// Returns natural logarithm of specified value.
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    public static float Ln(float value) => Log2(value) / BaseE;


    /// <summary>
    /// Returns logarithm of specified value with respect to a specified base.
    /// </summary>
    /// <param name="value"></param>
    /// <param name="valueBase"></param>
    /// <returns></returns>
    public static float Log(float value, float valueBase) => Log2(value) / Log2(valueBase);

    /// <summary>
    /// Module intializer to initialize internal table data structures.
    /// </summary>
    [ModuleInitializer]
    internal static void ClassInitialize()
    {
      MantissaLogs = new float[(int)System.Math.Pow(2, 23)];

      // Initialize a lookup table
      // Size is about 838k and initialization time is approxiately 0.25 seconds
      for (uint i = 0; i < MantissaLogs.Length; i++)
      {
        Ieee754 n = new Ieee754 { UnsignedBits = i | 0x3F800000 }; //added the implicit 1 leading bit
        MantissaLogs[i] = (float)System.Math.Log(n.Single, 2);
      }
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct Ieee754
    {
      [FieldOffset(0)] public float Single;
      [FieldOffset(0)] public uint UnsignedBits;
      [FieldOffset(0)] public int SignedBits;

      public uint Sign => UnsignedBits >> 31;
      public int Exponent => (SignedBits >> 23) & 0xFF;
      public uint Mantissa => UnsignedBits & 0x007FFFFF;
    }


  }
}
