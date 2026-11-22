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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Helper class for FP16 conversions and some low level operations.
  /// Based on code by Ladislav Lang (2009), Joannes Vermorel (2017)
  /// which is "code is free to use for any reason without any restrictions", see:
  /// https://gist.github.com/vermorel/1d5c0212752b3e611faf84771ad4ff0d
  /// </summary>
  /// <remarks>
  /// References:
  ///     - Fast FloatHalfPrecision Float Conversions, Jeroen van der Zijp, link: http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
  ///
  /// The half to single direction no longer uses van der Zijp's lookup tables (see HalfToSingle);
  /// the single to half direction still does.
  /// </remarks>
  internal static class FP16Helper
  {
    private static ushort[] baseTable;
    private static sbyte[] shiftTable;

    /// <summary>All ones half precision exponent, shifted into single precision position.</summary>
    private const uint SHIFTED_EXPONENT_MASK = 0x7c00u << 13;

    /// <summary>Difference between the single and half precision exponent biases (127 - 15).</summary>
    private const uint EXPONENT_REBIAS = (127u - 15u) << 23;

    /// <summary>Remainder of the rebias for infinity and NaN, whose exponent saturates (128 - 16).</summary>
    private const uint INF_NAN_REBIAS = (128u - 16u) << 23;

    /// <summary>2^-14, the implied leading one introduced when renormalizing a subnormal.</summary>
    private const float SUBNORMAL_LEADING_ONE = 1.0f / (1 << 14);


    /// <summary>
    /// Converts a half precision value to single precision.
    /// </summary>
    /// <remarks>
    /// Deliberately not the shorter formulation which multiplies by 2^112: that one feeds a
    /// subnormal float into the multiply and would therefore return zero for every half precision
    /// subnormal on any thread running with the MXCSR denormals-are-zero flag set, which native
    /// libraries such as CUDA are free to do. Every intermediate below is a normal float.
    /// </remarks>
    internal static float HalfToSingle(FP16 floatHalfPrecision)
    {
      uint bits = floatHalfPrecision.Value;

      // Shift exponent and mantissa into single precision position, then rebias the exponent
      // from the half precision bias of 15 to the single precision bias of 127.
      uint result = (bits & 0x7fffu) << 13;
      uint exponent = SHIFTED_EXPONENT_MASK & result;
      result += EXPONENT_REBIAS;

      if (exponent == SHIFTED_EXPONENT_MASK)
      {
        // Infinity and NaN saturate the exponent in both formats, so finish rebiasing to all ones.
        // Mantissa bits pass through untouched, preserving NaN payloads (and so leaving signalling
        // NaNs signalling, exactly as the tables did).
        result += INF_NAN_REBIAS;
      }
      else if (exponent == 0)
      {
        // Zero and subnormals. One further increment of the exponent makes this a normal float
        // carrying an implied leading one, which the subtraction then removes, leaving
        // mantissa * 2^-24 (and exactly zero when the mantissa is zero).
        result = BitConverter.SingleToUInt32Bits(
                   BitConverter.UInt32BitsToSingle(result + (1u << 23)) - SUBNORMAL_LEADING_ONE);
      }

      return BitConverter.UInt32BitsToSingle(result | ((bits & 0x8000u) << 16));
    }


    internal static unsafe FP16 SingleToHalf(float single)
    {
      uint value = *((uint*)&single);

      ushort result = (ushort)(baseTable[(value >> 23) & 0x1ff] + ((value & 0x007fffff) >> shiftTable[value >> 23]));
      return FP16.ToHalf(result);
    }


    private static ushort[] GenerateBaseTable()
    {
      ushort[] baseTable = new ushort[512];
      for (int i = 0; i < 256; ++i)
      {
        sbyte e = (sbyte)(127 - i);
        if (e > 24)
        { // Very small numbers map to zero
          baseTable[i | 0x000] = 0x0000;
          baseTable[i | 0x100] = 0x8000;
        }
        else if (e > 14)
        { // Small numbers map to denorms
          baseTable[i | 0x000] = (ushort)(0x0400 >> (18 + e));
          baseTable[i | 0x100] = (ushort)((0x0400 >> (18 + e)) | 0x8000);
        }
        else if (e >= -15)
        { // Normal numbers just lose precision
          baseTable[i | 0x000] = (ushort)((15 - e) << 10);
          baseTable[i | 0x100] = (ushort)(((15 - e) << 10) | 0x8000);
        }
        else if (e > -128)
        { // Large numbers map to Infinity
          baseTable[i | 0x000] = 0x7c00;
          baseTable[i | 0x100] = 0xfc00;
        }
        else
        { // Infinity and NaN's stay Infinity and NaN's
          baseTable[i | 0x000] = 0x7c00;
          baseTable[i | 0x100] = 0xfc00;
        }
      }

      return baseTable;
    }

    private static sbyte[] GenerateShiftTable()
    {
      sbyte[] shiftTable = new sbyte[512];
      for (int i = 0; i < 256; ++i)
      {
        sbyte e = (sbyte)(127 - i);
        if (e > 24)
        { // Very small numbers map to zero
          shiftTable[i | 0x000] = 24;
          shiftTable[i | 0x100] = 24;
        }
        else if (e > 14)
        { // Small numbers map to denorms
          shiftTable[i | 0x000] = (sbyte)(e - 1);
          shiftTable[i | 0x100] = (sbyte)(e - 1);
        }
        else if (e >= -15)
        { // Normal numbers just lose precision
          shiftTable[i | 0x000] = 13;
          shiftTable[i | 0x100] = 13;
        }
        else if (e > -128)
        { // Large numbers map to Infinity
          shiftTable[i | 0x000] = 24;
          shiftTable[i | 0x100] = 24;
        }
        else
        { // Infinity and NaN's stay Infinity and NaN's
          shiftTable[i | 0x000] = 13;
          shiftTable[i | 0x100] = 13;
        }
      }

      return shiftTable;
    }


    internal static FP16 Negate(FP16 floatHalfPrecision) => FP16.ToHalf((ushort)(floatHalfPrecision.Value ^ 0x8000));
    
    internal static FP16 Abs(FP16 floatHalfPrecision) => FP16.ToHalf((ushort)(floatHalfPrecision.Value & 0x7fff));

    internal static bool IsNaN(FP16 floatHalfPrecision) => (floatHalfPrecision.Value & 0x7fff) > 0x7c00;
    
    internal static bool IsInfinity(FP16 floatHalfPrecision) => (floatHalfPrecision.Value & 0x7fff) == 0x7c00;

    internal static bool IsPositiveInfinity(FP16 floatHalfPrecision) => floatHalfPrecision.Value == 0x7c00;

    internal static bool IsNegativeInfinity(FP16 floatHalfPrecision) => floatHalfPrecision.Value == 0xfc00;    


    /// <summary>
    /// Proves HalfToSingle correct over its complete input domain, all 65,536 bit patterns,
    /// against the runtime's own half precision conversion. With no separate test project this
    /// is what guards the table free implementation against regression; it costs well under a
    /// millisecond and so simply runs at startup in debug builds.
    /// </summary>
    /// <remarks>
    /// One documented deviation, which is why the comparison is not unconditionally on bits:
    /// the runtime quiets signalling NaNs whereas this (like the lookup tables it replaced)
    /// preserves the payload, so for those 1,022 inputs only NaN-ness is asserted. Ceres itself
    /// never produces a signalling NaN; FP16.NaN is 0xFE00, which is quiet.
    /// </remarks>
    internal static void VerifyDecodeExhaustive()
    {
      for (int i = 0; i <= ushort.MaxValue; i++)
      {
        ushort bits = (ushort)i;
        float ours = HalfToSingle(FP16.ToHalf(bits));
        float expected = (float)BitConverter.UInt16BitsToHalf(bits);

        bool isSignallingNaN = (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0 && (bits & 0x0200) == 0;
        bool ok = isSignallingNaN
                ? float.IsNaN(ours) && float.IsNaN(expected)
                : BitConverter.SingleToUInt32Bits(ours) == BitConverter.SingleToUInt32Bits(expected);

        if (!ok)
        {
          throw new Exception($"FP16Helper.HalfToSingle disagrees with Half at 0x{bits:X4}: "
                            + $"0x{BitConverter.SingleToUInt32Bits(ours):X8} versus 0x{BitConverter.SingleToUInt32Bits(expected):X8}");
        }
      }
    }


    [ModuleInitializer]
    internal static void ClassInitialize()
    {
      baseTable = GenerateBaseTable();
      shiftTable = GenerateShiftTable();

#if DEBUG
      VerifyDecodeExhaustive();
#endif
    }

  }
}
