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
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// FP16 (half precision) data type.
  /// 
  /// Based on code by Ladislav Lang (2009), Joannes Vermorel (2017)
  /// which is "code is free to use for any reason without any restrictions", see:
  /// https://gist.github.com/vermorel/1d5c0212752b3e611faf84771ad4ff0d
  /// </summary>
  /// <remarks>
  /// References:
  ///     - Fast FP16 Float Conversions, Jeroen van der Zijp, link: http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
  ///     - IEEE 754 revision, link: http://grouper.ieee.org/groups/754/
  ///     
  /// TODO: consider using emerging .NET support, possibly with hardware intrinsics:
  ///   - System.Numerics.Experimental.Half (but where is the NuGet package?)
  ///   - AVX2 access to vcvtph2ps/vcvtps2ph (?)
  /// </remarks>
  [Serializable]
  public readonly struct FP16 : IComparable, IFormattable, IConvertible, IComparable<FP16>, IEquatable<FP16>
  {
    public static FP16[,] ToFP16(float[,] data)
    {
      FP16[,] ret = new FP16[data.GetLength(0), data.GetLength(1)];
      for (int i = 0; i < ret.GetLength(0); i++)
        for (int j = 0; j < ret.GetLength(1); j++)
        {
          float value = data[i, j];
          if (value != 0.0f)
          {
            ret[i, j] = (FP16)value;
          }
        }
      return ret;
    }

    public static FP16[] ToFP16(float[] data)
    {
      FP16[] ret = new FP16[data.Length];
      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = (FP16)data[i];
      }

      return ret;
    }

    public static FP16[] ToFP16Approx(float[] data)
    {
      if (DISABLE_APPROX_FP16_CONVERSIONS)
      {
        return ToFP16(data);
      }
      else
      {
        FP16[] ret = new FP16[data.Length];
        for (int i = 0; i < ret.Length; i++)
        {
          ret[i] = FP16.FromFloatApprox(data[i]);
        }

        return ret;
      }
    }

    public static float[,] ToFloat(FP16[,] data)
    {
      float[,] ret = new float[data.GetLength(0), data.GetLength(1)];
      for (int i = 0; i < ret.GetLength(0); i++)
      {
        for (int j = 0; j < ret.GetLength(1); j++)
        {
          ret[i, j] = data[i, j];
        }
      }

      return ret;
    }

    public unsafe static float[,] ToFloat(FP16* data, int numRows, int numColumns)
    {
      int sourceOffset = 0;
      float[,] ret = new float[numRows, numColumns];
      for (int i = 0; i < numRows; i++)
      {
        for (int j = 0; j < numColumns; j++)
        {
          ret[i, j] = data[sourceOffset++];
        }
      }
      return ret;
    }

    public unsafe static void ToFloat(Span<FP16> source, float[,] dest, int numRows, int numColumns)
    {
      int sourceOffset = 0;
      for (int i = 0; i < numRows; i++)
      {
        for (int j = 0; j < numColumns; j++)
        {
          dest[i, j] = source[sourceOffset++];
        }
      }
    }

    public unsafe static float[,] ToFloat(Span<FP16> data, int numRows, int numColumns)
    {
      float[,] ret = new float[numRows, numColumns];
      ToFloat(data, ret, numRows, numColumns);
      return ret;
    }

    public static unsafe float[] ToFloat(FP16* data, int numElements)
    {
      float[] ret = new float[numElements];
      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = data[i];
      }

      return ret;
    }


    public static unsafe void ToFloat(Span<FP16> data, float[] dest, int numElements)
    {
      for (int i = 0; i < data.Length; i++)
      {
        dest[i] = data[i];
      }
    }

    public static unsafe float[] ToFloat(Span<FP16> data, int numElements)
    {
      float[] ret = new float[numElements];
      ToFloat(data, ret, numElements);
      return ret;
    }

    public static float[] ToFloat(FP16[] data)
    {
      float[] ret = new float[data.Length];
      for (int i = 0; i < ret.Length; i++)
        ret[i] = data[i];
      return ret;
    }

    public static float[] ToFloat(Span<FP16> data)
    {
      float[] ret = new float[data.Length];
      for (int i = 0; i < ret.Length; i++)
        ret[i] = data[i];
      return ret;
    }

    /// <summary>
    /// Internal representation of the FP16-precision floating-point number.
    /// </summary>
    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    public readonly ushort Value;

    #region Constants
    /// <summary>
    /// Represents zero. This field is constant.
    /// </summary>
    public static readonly FP16 Zero = FP16.ToHalf(0);
    /// <summary>
    /// Represents the smallest positive FP16 value greater than zero. This field is constant.
    /// </summary>
    public static readonly FP16 Epsilon = FP16.ToHalf(0x0001);
    /// <summary>
    /// Represents the largest possible value of FP16. This field is constant.
    /// </summary>
    public static readonly FP16 MaxValue = FP16.ToHalf(0x7bff);
    /// <summary>
    /// Represents the smallest possible value of FP16. This field is constant.
    /// </summary>
    public static readonly FP16 MinValue = FP16.ToHalf(0xfbff);
    /// <summary>
    /// Represents not a number (NaN). This field is constant.
    /// </summary>
    public static readonly FP16 NaN = FP16.ToHalf(0xfe00);
    /// <summary>
    /// Represents negative infinity. This field is constant.
    /// </summary>
    public static readonly FP16 NegativeInfinity = FP16.ToHalf(0xfc00);
    /// <summary>
    /// Represents positive infinity. This field is constant.
    /// </summary>
    public static readonly FP16 PositiveInfinity = FP16.ToHalf(0x7c00);
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes a new instance of FP16 to the value of the specified single-precision floating-point number.
    /// </summary>
    /// <param name="value">The value to represent as a FP16.</param>
    public FP16(float value) { this = FP16Helper.SingleToHalf(value); }
    /// <summary>
    /// Initializes a new instance of FP16 to the value of the specified 32-bit signed integer.
    /// </summary>
    /// <param name="value">The value to represent as a FP16.</param>
    public FP16(int value) : this((float)value) { }
    /// <summary>
    /// Initializes a new instance of FP16 to the value of the specified 64-bit signed integer.
    /// </summary>
    /// <param name="value">The value to represent as a FP16.</param>
    public FP16(long value) : this((float)value) { }
    /// <summary>
    /// Initializes a new instance of FP16 to the value of the specified double-precision floating-point number.
    /// </summary>
    /// <param name="value">The value to represent as a FP16.</param>
    public FP16(double value) : this((float)value) { }
    /// <summary>
    /// Initializes a new instance of FP16 to the value of the specified decimal number.
    /// </summary>
    /// <param name="value">The value to represent as a FP16.</param>
    public FP16(decimal value) : this((float)value) { }
    /// <summary>
    /// Initializes a new instance of FP16 to the value of the specified 32-bit unsigned integer.
    /// </summary>
    /// <param name="value">The value to represent as a FP16.</param>
    public FP16(uint value) : this((float)value) { }
    /// <summary>
    /// Initializes a new instance of FP16 to the value of the specified 64-bit unsigned integer.
    /// </summary>
    /// <param name="value">The value to represent as a FP16.</param>
    public FP16(ulong value) : this((float)value) { }

    internal  FP16(ushort value, bool directAssigment = true) =>  Value = value;

    /// <summary>
    /// Returns an FP16 created from its raw (bits) value.
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    public static FP16 FromRaw(ushort value) => new FP16(value, true);

    /// <summary>
    /// 
    /// Warning: it is not clear this is faster, and results in memory cache pollution.    
    /// </summary>    
    public float ToSingleViaLookup => FP16Helper.HalfToSingleLookup(this);

    // WARNING: Do not enable these approximations.
    //          They seem sufficiently imprecise that
    //          they negatively impact qualitY (e.g. play quality at 3 nodes/move).
    const bool DISABLE_APPROX_FP16_CONVERSIONS = true;

    /// <summary>
    /// Fast approximate conversion from FP16 to float.
    /// Note that this does not support all FP16, for example NaNs.
    /// </summary>
    /// <param name="fp16"></param>
    /// <returns></returns>
    public float ToFloatApprox
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        if (DISABLE_APPROX_FP16_CONVERSIONS)
        {
          return (FP16)this;
        }

        if (Value == Zero.Value)
        {
          return 0.0f;
        }
        else
        {
          int conv = (((Value & 0x8000) << 16) | (((Value & 0x7c00) + 0x1C000) << 13) | ((Value & 0x03FF) << 13));
          float ret = Unsafe.As<int, float>(ref conv);
          Debug.Assert(MathF.Abs(ret - (FP16)this) < 0.01f); // TO DO: be more precise in testing expected precision
          return ret;
        }
      }
    }

    /// <summary>
    /// Fast approximate conversion from float to FP16.
    /// Note that this does not support all FP16, for example NaNs.
    /// </summary>
    /// <param name="f"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FP16 FromFloatApprox(float f)
    {
      if (DISABLE_APPROX_FP16_CONVERSIONS)
      {
        return (FP16)f;
      }

      const float MIN_FP16 = 0.0000610352f; // 0.00006103515625f;
      if (MathF.Abs(f) < MIN_FP16)
      {
        return FP16.Zero;
      }
      else
      {
        int x = Unsafe.As<float, int>(ref f);
        short xx = (short)(((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff));

        FP16 ret = Unsafe.As<short, FP16>(ref xx);
        Debug.Assert(MathF.Abs((float)ret - f) < 0.01f); // TO DO: be more precise in testing expected precision
        return ret;
      }
    }

#endregion

#region Numeric operators

    /// <summary>
    /// Returns the result of multiplying the specified FP16 value by negative one.
    /// </summary>
    /// <param name="FP16">A FP16.</param>
    /// <returns>A FP16 with the value of FP16, but the opposite sign. -or- Zero, if FP16 is zero.</returns>
    public static FP16 Negate(FP16 FP16) { return -FP16; }
    /// <summary>
    /// Adds two specified FP16 values.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>A FP16 value that is the sum of half1 and half2.</returns>
    public static FP16 Add(FP16 half1, FP16 half2) { return half1 + half2; }
    /// <summary>
    /// Subtracts one specified FP16 value from another.
    /// </summary>
    /// <param name="half1">A FP16 (the minuend).</param>
    /// <param name="half2">A FP16 (the subtrahend).</param>
    /// <returns>The FP16 result of subtracting half2 from half1.</returns>
    public static FP16 Subtract(FP16 half1, FP16 half2) { return half1 - half2; }
    /// <summary>
    /// Multiplies two specified FP16 values.
    /// </summary>
    /// <param name="half1">A FP16 (the multiplicand).</param>
    /// <param name="half2">A FP16 (the multiplier).</param>
    /// <returns>A FP16 that is the result of multiplying half1 and half2.</returns>
    public static FP16 Multiply(FP16 half1, FP16 half2) { return half1 * half2; }
    /// <summary>
    /// Divides two specified FP16 values.
    /// </summary>
    /// <param name="half1">A FP16 (the dividend).</param>
    /// <param name="half2">A FP16 (the divisor).</param>
    /// <returns>The FP16 that is the result of dividing half1 by half2.</returns>
    /// <exception cref="System.DivideByZeroException">half2 is zero.</exception>
    public static FP16 Divide(FP16 half1, FP16 half2) { return half1 / half2; }

    /// <summary>
    /// Returns the value of the FP16 operand (the sign of the operand is unchanged).
    /// </summary>
    /// <param name="FP16">The FP16 operand.</param>
    /// <returns>The value of the operand, FP16.</returns>
    public static FP16 operator +(FP16 FP16) { return FP16; }
    /// <summary>
    /// Negates the value of the specified FP16 operand.
    /// </summary>
    /// <param name="FP16">The FP16 operand.</param>
    /// <returns>The result of FP16 multiplied by negative one (-1).</returns>
    public static FP16 operator -(FP16 FP16) { return FP16Helper.Negate(FP16); }
    /// <summary>
    /// Increments the FP16 operand by 1.
    /// </summary>
    /// <param name="FP16">The FP16 operand.</param>
    /// <returns>The value of FP16 incremented by 1.</returns>
    public static FP16 operator ++(FP16 FP16) { return (FP16)(FP16 + 1f); }
    /// <summary>
    /// Decrements the FP16 operand by one.
    /// </summary>
    /// <param name="FP16">The FP16 operand.</param>
    /// <returns>The value of FP16 decremented by 1.</returns>
    public static FP16 operator --(FP16 FP16) { return (FP16)(FP16 - 1f); }
    /// <summary>
    /// Adds two specified FP16 values.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>The FP16 result of adding half1 and half2.</returns>
    public static FP16 operator +(FP16 half1, FP16 half2) { return (FP16)((float)half1 + (float)half2); }
    /// <summary>
    /// Subtracts two specified FP16 values.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>The FP16 result of subtracting half1 and half2.</returns>        
    public static FP16 operator -(FP16 half1, FP16 half2) { return (FP16)((float)half1 - (float)half2); }
    /// <summary>
    /// Multiplies two specified FP16 values.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>The FP16 result of multiplying half1 by half2.</returns>
    public static FP16 operator *(FP16 half1, FP16 half2) { return (FP16)((float)half1 * (float)half2); }
    /// <summary>
    /// Divides two specified FP16 values.
    /// </summary>
    /// <param name="half1">A FP16 (the dividend).</param>
    /// <param name="half2">A FP16 (the divisor).</param>
    /// <returns>The FP16 result of half1 by half2.</returns>
    public static FP16 operator /(FP16 half1, FP16 half2) { return (FP16)((float)half1 / (float)half2); }
    /// <summary>
    /// Returns a value indicating whether two instances of FP16 are equal.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>true if half1 and half2 are equal; otherwise, false.</returns>
    public static bool operator ==(FP16 half1, FP16 half2) { return (!IsNaN(half1) && (half1.Value == half2.Value)); }
    /// <summary>
    /// Returns a value indicating whether two instances of FP16 are not equal.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>true if half1 and half2 are not equal; otherwise, false.</returns>
    public static bool operator !=(FP16 half1, FP16 half2) { return !(half1.Value == half2.Value); }
    /// <summary>
    /// Returns a value indicating whether a specified FP16 is less than another specified FP16.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>true if half1 is less than half1; otherwise, false.</returns>
    public static bool operator <(FP16 half1, FP16 half2) { return (float)half1 < (float)half2; }
    /// <summary>
    /// Returns a value indicating whether a specified FP16 is greater than another specified FP16.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>true if half1 is greater than half2; otherwise, false.</returns>
    public static bool operator >(FP16 half1, FP16 half2) { return (float)half1 > (float)half2; }
    /// <summary>
    /// Returns a value indicating whether a specified FP16 is less than or equal to another specified FP16.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>true if half1 is less than or equal to half2; otherwise, false.</returns>
    public static bool operator <=(FP16 half1, FP16 half2) { return (half1 == half2) || (half1 < half2); }
    /// <summary>
    /// Returns a value indicating whether a specified FP16 is greater than or equal to another specified FP16.
    /// </summary>
    /// <param name="half1">A FP16.</param>
    /// <param name="half2">A FP16.</param>
    /// <returns>true if half1 is greater than or equal to half2; otherwise, false.</returns>
    public static bool operator >=(FP16 half1, FP16 half2) { return (half1 == half2) || (half1 > half2); }
#endregion

#region Type casting operators
    /// <summary>
    /// Converts an 8-bit unsigned integer to a FP16.
    /// </summary>
    /// <param name="value">An 8-bit unsigned integer.</param>
    /// <returns>A FP16 that represents the converted 8-bit unsigned integer.</returns>
    public static implicit operator FP16(byte value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a 16-bit signed integer to a FP16.
    /// </summary>
    /// <param name="value">A 16-bit signed integer.</param>
    /// <returns>A FP16 that represents the converted 16-bit signed integer.</returns>
    public static implicit operator FP16(short value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a Unicode character to a FP16.
    /// </summary>
    /// <param name="value">A Unicode character.</param>
    /// <returns>A FP16 that represents the converted Unicode character.</returns>
    public static implicit operator FP16(char value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a 32-bit signed integer to a FP16.
    /// </summary>
    /// <param name="value">A 32-bit signed integer.</param>
    /// <returns>A FP16 that represents the converted 32-bit signed integer.</returns>
    public static implicit operator FP16(int value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a 64-bit signed integer to a FP16.
    /// </summary>
    /// <param name="value">A 64-bit signed integer.</param>
    /// <returns>A FP16 that represents the converted 64-bit signed integer.</returns>
    public static implicit operator FP16(long value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a single-precision floating-point number to a FP16.
    /// </summary>
    /// <param name="value">A single-precision floating-point number.</param>
    /// <returns>A FP16 that represents the converted single-precision floating point number.</returns>
    public static explicit operator FP16(float value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a double-precision floating-point number to a FP16.
    /// </summary>
    /// <param name="value">A double-precision floating-point number.</param>
    /// <returns>A FP16 that represents the converted double-precision floating point number.</returns>
    public static explicit operator FP16(double value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a decimal number to a FP16.
    /// </summary>
    /// <param name="value">decimal number</param>
    /// <returns>A FP16 that represents the converted decimal number.</returns>
    public static explicit operator FP16(decimal value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a FP16 to an 8-bit unsigned integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>An 8-bit unsigned integer that represents the converted FP16.</returns>
    public static explicit operator byte(FP16 value) { return (byte)(float)value; }
    /// <summary>
    /// Converts a FP16 to a Unicode character.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A Unicode character that represents the converted FP16.</returns>
    public static explicit operator char(FP16 value) { return (char)(float)value; }
    /// <summary>
    /// Converts a FP16 to a 16-bit signed integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A 16-bit signed integer that represents the converted FP16.</returns>
    public static explicit operator short(FP16 value) { return (short)(float)value; }
    /// <summary>
    /// Converts a FP16 to a 32-bit signed integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A 32-bit signed integer that represents the converted FP16.</returns>
    public static explicit operator int(FP16 value) { return (int)(float)value; }
    /// <summary>
    /// Converts a FP16 to a 64-bit signed integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A 64-bit signed integer that represents the converted FP16.</returns>
    public static explicit operator long(FP16 value) { return (long)(float)value; }
    /// <summary>
    /// Converts a FP16 to a single-precision floating-point number.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A single-precision floating-point number that represents the converted FP16.</returns>
    public static implicit operator float(FP16 value) { return (float)FP16Helper.HalfToSingle(value); }
    /// <summary>
    /// Converts a FP16 to a double-precision floating-point number.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A double-precision floating-point number that represents the converted FP16.</returns>
    public static implicit operator double(FP16 value) { return (double)(float)value; }
    /// <summary>
    /// Converts a FP16 to a decimal number.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A decimal number that represents the converted FP16.</returns>
    public static explicit operator decimal(FP16 value) { return (decimal)(float)value; }
    /// <summary>
    /// Converts an 8-bit signed integer to a FP16.
    /// </summary>
    /// <param name="value">An 8-bit signed integer.</param>
    /// <returns>A FP16 that represents the converted 8-bit signed integer.</returns>
    public static implicit operator FP16(sbyte value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a 16-bit unsigned integer to a FP16.
    /// </summary>
    /// <param name="value">A 16-bit unsigned integer.</param>
    /// <returns>A FP16 that represents the converted 16-bit unsigned integer.</returns>
    public static implicit operator FP16(ushort value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a 32-bit unsigned integer to a FP16.
    /// </summary>
    /// <param name="value">A 32-bit unsigned integer.</param>
    /// <returns>A FP16 that represents the converted 32-bit unsigned integer.</returns>
    public static implicit operator FP16(uint value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a 64-bit unsigned integer to a FP16.
    /// </summary>
    /// <param name="value">A 64-bit unsigned integer.</param>
    /// <returns>A FP16 that represents the converted 64-bit unsigned integer.</returns>
    public static implicit operator FP16(ulong value) { return new FP16((float)value); }
    /// <summary>
    /// Converts a FP16 to an 8-bit signed integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>An 8-bit signed integer that represents the converted FP16.</returns>
    public static explicit operator sbyte(FP16 value) { return (sbyte)(float)value; }
    /// <summary>
    /// Converts a FP16 to a 16-bit unsigned integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A 16-bit unsigned integer that represents the converted FP16.</returns>
    public static explicit operator ushort(FP16 value) { return (ushort)(float)value; }
    /// <summary>
    /// Converts a FP16 to a 32-bit unsigned integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A 32-bit unsigned integer that represents the converted FP16.</returns>
    public static explicit operator uint(FP16 value) { return (uint)(float)value; }
    /// <summary>
    /// Converts a FP16 to a 64-bit unsigned integer.
    /// </summary>
    /// <param name="value">A FP16 to convert.</param>
    /// <returns>A 64-bit unsigned integer that represents the converted FP16.</returns>
    public static explicit operator ulong(FP16 value) { return (ulong)(float)value; }
#endregion

    /// <summary>
    /// Compares this instance to a specified FP16 object.
    /// </summary>
    /// <param name="other">A FP16 object.</param>
    /// <returns>
    /// A signed number indicating the relative values of this instance and value.
    /// Return Value Meaning Less than zero This instance is less than value. Zero
    /// This instance is equal to value. Greater than zero This instance is greater than value.
    /// </returns>
    public int CompareTo(FP16 other)
    {
      int result = 0;
      if (this < other)
      {
        result = -1;
      }
      else if (this > other)
      {
        result = 1;
      }
      else if (this != other)
      {
        if (!IsNaN(this))
        {
          result = 1;
        }
        else if (!IsNaN(other))
        {
          result = -1;
        }
      }

      return result;
    }

    /// <summary>
    /// Compares this instance to a specified System.Object.
    /// </summary>
    /// <param name="obj">An System.Object or null.</param>
    /// <returns>
    /// A signed number indicating the relative values of this instance and value.
    /// Return Value Meaning Less than zero This instance is less than value. Zero
    /// This instance is equal to value. Greater than zero This instance is greater
    /// than value. -or- value is null.
    /// </returns>
    /// <exception cref="System.ArgumentException">value is not a FP16</exception>
    public int CompareTo(object obj)
    {
      int result;
      if (obj == null)
      {
        result = 1;
      }
      else
      {
        if (obj is FP16 fP)
        {
          result = CompareTo(fP);
        }
        else
        {
          throw new ArgumentException("Object must be of type FP16.");
        }
      }

      return result;
    }
    /// <summary>
    /// Returns a value indicating whether this instance and a specified FP16 object represent the same value.
    /// </summary>
    /// <param name="other">A FP16 object to compare to this instance.</param>
    /// <returns>true if value is equal to this instance; otherwise, false.</returns>
    public bool Equals(FP16 other)
    {
      return ((other == this) || (IsNaN(other) && IsNaN(this)));
    }
    /// <summary>
    /// Returns a value indicating whether this instance and a specified System.Object
    /// represent the same type and value.
    /// </summary>
    /// <param name="obj">An System.Object.</param>
    /// <returns>true if value is a FP16 and equal to this instance; otherwise, false.</returns>
    public override bool Equals(object obj)
    {
      bool result = false;
      if (obj is FP16 FP16)
      {
        if ((FP16 == this) || (IsNaN(FP16) && IsNaN(this)))
        {
          result = true;
        }
      }

      return result;
    }
    /// <summary>
    /// Returns the hash code for this instance.
    /// </summary>
    /// <returns>A 32-bit signed integer hash code.</returns>
    public override int GetHashCode()
    {
      return Value.GetHashCode();
    }
    /// <summary>
    /// Returns the System.TypeCode for value type FP16.
    /// </summary>
    /// <returns>The enumerated constant (TypeCode)255.</returns>
    public static TypeCode GetTypeCode() => (TypeCode)255;
    

#region BitConverter & Math methods for FP16
    /// <summary>
    /// Returns the specified FP16-precision floating point value as an array of bytes.
    /// </summary>
    /// <param name="value">The number to convert.</param>
    /// <returns>An array of bytes with length 2.</returns>
    public static byte[] GetBytes(FP16 value) => BitConverter.GetBytes(value.Value);
    
    /// <summary>
    /// Converts the value of a specified instance of FP16 to its equivalent binary representation.
    /// </summary>
    /// <param name="value">A FP16 value.</param>
    /// <returns>A 16-bit unsigned integer that contain the binary representation of value.</returns>        
    public static ushort GetBits(FP16 value) => value.Value;
    
    /// <summary>
    /// Returns a FP16-precision floating point number converted from two bytes
    /// at a specified position in a byte array.
    /// </summary>
    /// <param name="value">An array of bytes.</param>
    /// <param name="startIndex">The starting position within value.</param>
    /// <returns>A FP16-precision floating point number formed by two bytes beginning at startIndex.</returns>
    /// <exception cref="System.ArgumentException">
    /// startIndex is greater than or equal to the length of value minus 1, and is
    /// less than or equal to the length of value minus 1.
    /// </exception>
    /// <exception cref="System.ArgumentNullException">value is null.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">startIndex is less than zero or greater than the length of value minus 1.</exception>
    public static FP16 ToHalf(byte[] value, int startIndex) => FP16.ToHalf((ushort)BitConverter.ToInt16(value, startIndex));
    
    /// <summary>
    /// Returns a FP16-precision floating point number converted from its binary representation.
    /// </summary>
    /// <param name="bits">Binary representation of FP16 value</param>
    /// <returns>A FP16-precision floating point number formed by its binary representation.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FP16 ToHalf(ushort bits) => new FP16(bits, true);
    

    /// <summary>
    /// Returns a value indicating the sign of a FP16-precision floating-point number.
    /// </summary>
    /// <param name="value">A signed number.</param>
    /// <returns>
    /// A number indicating the sign of value. Number Description -1 value is less
    /// than zero. 0 value is equal to zero. 1 value is greater than zero.
    /// </returns>
    /// <exception cref="System.ArithmeticException">value is equal to FP16.NaN.</exception>
    public static int Sign(FP16 value)
    {
      if (value < 0)
      {
        return -1;
      }
      else if (value > 0)
      {
        return 1;
      }
      else
      {
        if (value != 0)
        {
          throw new ArithmeticException("Function does not accept floating point Not-a-Number values.");
        }
      }

      return 0;
    }
    /// <summary>
    /// Returns the absolute value of a FP16-precision floating-point number.
    /// </summary>
    /// <param name="value">A number in the range FP16.MinValue ≤ value ≤ FP16.MaxValue.</param>
    /// <returns>A FP16-precision floating-point number, x, such that 0 ≤ x ≤FP16.MaxValue.</returns>
    public static FP16 Abs(FP16 value) => FP16Helper.Abs(value);
    
    /// <summary>
    /// Returns the larger of two FP16-precision floating-point numbers.
    /// </summary>
    /// <param name="value1">The first of two FP16-precision floating-point numbers to compare.</param>
    /// <param name="value2">The second of two FP16-precision floating-point numbers to compare.</param>
    /// <returns>
    /// Parameter value1 or value2, whichever is larger. If value1, or value2, or both val1
    /// and value2 are equal to FP16.NaN, FP16.NaN is returned.
    /// </returns>
    public static FP16 Max(FP16 value1, FP16 value2) => (value1 < value2) ? value2 : value1;
    
    /// <summary>
    /// Returns the smaller of two FP16-precision floating-point numbers.
    /// </summary>
    /// <param name="value1">The first of two FP16-precision floating-point numbers to compare.</param>
    /// <param name="value2">The second of two FP16-precision floating-point numbers to compare.</param>
    /// <returns>
    /// Parameter value1 or value2, whichever is smaller. If value1, or value2, or both val1
    /// and value2 are equal to FP16.NaN, FP16.NaN is returned.
    /// </returns>
    public static FP16 Min(FP16 value1, FP16 value2) =>  (value1 < value2) ? value1 : value2;
    
#endregion

    /// <summary>
    /// Returns a value indicating whether the specified number evaluates to not a number (FP16.NaN).
    /// </summary>
    /// <param name="FP16">A FP16-precision floating-point number.</param>
    /// <returns>true if value evaluates to not a number (FP16.NaN); otherwise, false.</returns>
    public static bool IsNaN(FP16 FP16) => FP16Helper.IsNaN(FP16);
    
    /// <summary>
    /// Returns a value indicating whether the specified number evaluates to negative or positive infinity.
    /// </summary>
    /// <param name="FP16">A FP16-precision floating-point number.</param>
    /// <returns>true if FP16 evaluates to FP16.PositiveInfinity or FP16.NegativeInfinity; otherwise, false.</returns>
    public static bool IsInfinity(FP16 FP16) => FP16Helper.IsInfinity(FP16);
    
    /// <summary>
    /// Returns a value indicating whether the specified number evaluates to negative infinity.
    /// </summary>
    /// <param name="FP16">A FP16-precision floating-point number.</param>
    /// <returns>true if FP16 evaluates to FP16.NegativeInfinity; otherwise, false.</returns>
    public static bool IsNegativeInfinity(FP16 FP16)=>  FP16Helper.IsNegativeInfinity(FP16);
    
    /// <summary>
    /// Returns a value indicating whether the specified number evaluates to positive infinity.
    /// </summary>
    /// <param name="FP16">A FP16-precision floating-point number.</param>
    /// <returns>true if FP16 evaluates to FP16.PositiveInfinity; otherwise, false.</returns>
    public static bool IsPositiveInfinity(FP16 FP16) =>FP16Helper.IsPositiveInfinity(FP16);
    

#region String operations (Parse and ToString)

    /// <summary>
    /// Converts the string representation of a number to its FP16 equivalent.
    /// </summary>
    /// <param name="value">The string representation of the number to convert.</param>
    /// <returns>The FP16 number equivalent to the number contained in value.</returns>
    /// <exception cref="System.ArgumentNullException">value is null.</exception>
    /// <exception cref="System.FormatException">value is not in the correct format.</exception>
    /// <exception cref="System.OverflowException">value represents a number less than FP16.MinValue or greater than FP16.MaxValue.</exception>
    public static FP16 Parse(string value) => (FP16)float.Parse(value, CultureInfo.InvariantCulture);
    
    /// <summary>
    /// Converts the string representation of a number to its FP16 equivalent 
    /// using the specified culture-specific format information.
    /// </summary>
    /// <param name="value">The string representation of the number to convert.</param>
    /// <param name="provider">An System.IFormatProvider that supplies culture-specific parsing information about value.</param>
    /// <returns>The FP16 number equivalent to the number contained in s as specified by provider.</returns>
    /// <exception cref="System.ArgumentNullException">value is null.</exception>
    /// <exception cref="System.FormatException">value is not in the correct format.</exception>
    /// <exception cref="System.OverflowException">value represents a number less than FP16.MinValue or greater than FP16.MaxValue.</exception>
    public static FP16 Parse(string value, IFormatProvider provider) => (FP16)float.Parse(value, provider);
    
    /// <summary>
    /// Converts the string representation of a number in a specified style to its FP16 equivalent.
    /// </summary>
    /// <param name="value">The string representation of the number to convert.</param>
    /// <param name="style">
    /// A bitwise combination of System.Globalization.NumberStyles values that indicates
    /// the style elements that can be present in value. A typical value to specify is
    /// System.Globalization.NumberStyles.Number.
    /// </param>
    /// <returns>The FP16 number equivalent to the number contained in s as specified by style.</returns>
    /// <exception cref="System.ArgumentNullException">value is null.</exception>
    /// <exception cref="System.ArgumentException">
    /// style is not a System.Globalization.NumberStyles value. -or- style is the
    /// System.Globalization.NumberStyles.AllowHexSpecifier value.
    /// </exception>
    /// <exception cref="System.FormatException">value is not in the correct format.</exception>
    /// <exception cref="System.OverflowException">value represents a number less than FP16.MinValue or greater than FP16.MaxValue.</exception>
    public static FP16 Parse(string value, NumberStyles style) => (FP16)float.Parse(value, style, CultureInfo.InvariantCulture);
  
    /// <summary>
    /// Converts the string representation of a number to its FP16 equivalent 
    /// using the specified style and culture-specific format.
    /// </summary>
    /// <param name="value">The string representation of the number to convert.</param>
    /// <param name="style">
    /// A bitwise combination of System.Globalization.NumberStyles values that indicates
    /// the style elements that can be present in value. A typical value to specify is 
    /// System.Globalization.NumberStyles.Number.
    /// </param>
    /// <param name="provider">An System.IFormatProvider object that supplies culture-specific information about the format of value.</param>
    /// <returns>The FP16 number equivalent to the number contained in s as specified by style and provider.</returns>
    /// <exception cref="System.ArgumentNullException">value is null.</exception>
    /// <exception cref="System.ArgumentException">
    /// style is not a System.Globalization.NumberStyles value. -or- style is the
    /// System.Globalization.NumberStyles.AllowHexSpecifier value.
    /// </exception>
    /// <exception cref="System.FormatException">value is not in the correct format.</exception>
    /// <exception cref="System.OverflowException">value represents a number less than FP16.MinValue or greater than FP16.MaxValue.</exception>
    public static FP16 Parse(string value, NumberStyles style, IFormatProvider provider) => (FP16)float.Parse(value, style, provider);
    

    /// <summary>
    /// Converts the string representation of a number to its FP16 equivalent.
    /// A return value indicates whether the conversion succeeded or failed.
    /// </summary>
    /// <param name="value">The string representation of the number to convert.</param>
    /// <param name="result">
    /// When this method returns, contains the FP16 number that is equivalent
    /// to the numeric value contained in value, if the conversion succeeded, or is zero
    /// if the conversion failed. The conversion fails if the s parameter is null,
    /// is not a number in a valid format, or represents a number less than FP16.MinValue
    /// or greater than FP16.MaxValue. This parameter is passed uninitialized.
    /// </param>
    /// <returns>true if s was converted successfully; otherwise, false.</returns>
    public static bool TryParse(string value, out FP16 result)
    {
      if (float.TryParse(value, out float f))
      {
        result = (FP16)f;
        return true;
      }

      result = new FP16();
      return false;
    }
    /// <summary>
    /// Converts the string representation of a number to its FP16 equivalent
    /// using the specified style and culture-specific format. A return value indicates
    /// whether the conversion succeeded or failed.
    /// </summary>
    /// <param name="value">The string representation of the number to convert.</param>
    /// <param name="style">
    /// A bitwise combination of System.Globalization.NumberStyles values that indicates
    /// the permitted format of value. A typical value to specify is System.Globalization.NumberStyles.Number.
    /// </param>
    /// <param name="provider">An System.IFormatProvider object that supplies culture-specific parsing information about value.</param>
    /// <param name="result">
    /// When this method returns, contains the FP16 number that is equivalent
    /// to the numeric value contained in value, if the conversion succeeded, or is zero
    /// if the conversion failed. The conversion fails if the s parameter is null,
    /// is not in a format compliant with style, or represents a number less than
    /// FP16.MinValue or greater than FP16.MaxValue. This parameter is passed uninitialized.
    /// </param>
    /// <returns>true if s was converted successfully; otherwise, false.</returns>
    /// <exception cref="System.ArgumentException">
    /// style is not a System.Globalization.NumberStyles value. -or- style 
    /// is the System.Globalization.NumberStyles.AllowHexSpecifier value.
    /// </exception>
    public static bool TryParse(string value, NumberStyles style, IFormatProvider provider, out FP16 result)
    {
      bool parseResult = false;
      if (float.TryParse(value, style, provider, out float f))
      {
        result = (FP16)f;
        parseResult = true;
      }
      else
      {
        result = new FP16();
      }

      return parseResult;
    }
    /// <summary>
    /// Converts the numeric value of this instance to its equivalent string representation.
    /// </summary>
    /// <returns>A string that represents the value of this instance.</returns>
    public override string ToString()
    {
      return ((float)this).ToString(CultureInfo.InvariantCulture);
    }
    /// <summary>
    /// Converts the numeric value of this instance to its equivalent string representation
    /// using the specified culture-specific format information.
    /// </summary>
    /// <param name="formatProvider">An System.IFormatProvider that supplies culture-specific formatting information.</param>
    /// <returns>The string representation of the value of this instance as specified by provider.</returns>
    public string ToString(IFormatProvider formatProvider)
    {
      return ((float)this).ToString(formatProvider);
    }
    /// <summary>
    /// Converts the numeric value of this instance to its equivalent string representation, using the specified format.
    /// </summary>
    /// <param name="format">A numeric format string.</param>
    /// <returns>The string representation of the value of this instance as specified by format.</returns>
    public string ToString(string format)
    {
      return ((float)this).ToString(format, CultureInfo.InvariantCulture);
    }
    /// <summary>
    /// Converts the numeric value of this instance to its equivalent string representation 
    /// using the specified format and culture-specific format information.
    /// </summary>
    /// <param name="format">A numeric format string.</param>
    /// <param name="formatProvider">An System.IFormatProvider that supplies culture-specific formatting information.</param>
    /// <returns>The string representation of the value of this instance as specified by format and provider.</returns>
    /// <exception cref="System.FormatException">format is invalid.</exception>
    public string ToString(string format, IFormatProvider formatProvider)
    {
      return ((float)this).ToString(format, formatProvider);
    }
#endregion

#region IConvertible Members
    float IConvertible.ToSingle(IFormatProvider provider)
    {
      return (float)this;
    }
    TypeCode IConvertible.GetTypeCode()
    {
      return GetTypeCode();
    }
    bool IConvertible.ToBoolean(IFormatProvider provider)
    {
      return Convert.ToBoolean((float)this);
    }
    byte IConvertible.ToByte(IFormatProvider provider)
    {
      return Convert.ToByte((float)this);
    }
    char IConvertible.ToChar(IFormatProvider provider)
    {
      throw new InvalidCastException(string.Format(CultureInfo.CurrentCulture, "Invalid cast from '{0}' to '{1}'.", "FP16", "Char"));
    }
    DateTime IConvertible.ToDateTime(IFormatProvider provider)
    {
      throw new InvalidCastException(string.Format(CultureInfo.CurrentCulture, "Invalid cast from '{0}' to '{1}'.", "FP16", "DateTime"));
    }
    decimal IConvertible.ToDecimal(IFormatProvider provider)
    {
      return Convert.ToDecimal((float)this);
    }
    double IConvertible.ToDouble(IFormatProvider provider)
    {
      return Convert.ToDouble((float)this);
    }
    short IConvertible.ToInt16(IFormatProvider provider)
    {
      return Convert.ToInt16((float)this);
    }
    int IConvertible.ToInt32(IFormatProvider provider)
    {
      return Convert.ToInt32((float)this);
    }
    long IConvertible.ToInt64(IFormatProvider provider)
    {
      return Convert.ToInt64((float)this);
    }
    sbyte IConvertible.ToSByte(IFormatProvider provider)
    {
      return Convert.ToSByte((float)this);
    }
    string IConvertible.ToString(IFormatProvider provider)
    {
      return Convert.ToString((float)this, CultureInfo.InvariantCulture);
    }
    object IConvertible.ToType(Type conversionType, IFormatProvider provider)
    {
      return (((float)this) as IConvertible).ToType(conversionType, provider);
    }
    ushort IConvertible.ToUInt16(IFormatProvider provider)
    {
      return Convert.ToUInt16((float)this);
    }
    uint IConvertible.ToUInt32(IFormatProvider provider)
    {
      return Convert.ToUInt32((float)this);
    }
    ulong IConvertible.ToUInt64(IFormatProvider provider)
    {
      return Convert.ToUInt64((float)this);
    }
#endregion
  }
}
