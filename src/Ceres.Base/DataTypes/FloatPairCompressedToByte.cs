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

namespace Ceres.Base.DataTypes;

/// <summary>
/// Stores two float-like values in a single byte (4 bits each).
/// Numeric range is [0 … 0.62]; the code 0xF encodes NaN.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 1, Pack = 1)]
public struct FloatPairCompressedToByte : IEquatable<FloatPairCompressedToByte>
{
  public const float MIN_VALUE = 0.0f;
  public const float MAX_VALUE = 0.62f;

  private const int NumericMaxCode = 14; // 0 … 14 encode real numbers
  private const int NaNCode = 15; // 0xF encodes NaN
  private const float EncodeScale = NumericMaxCode / MAX_VALUE;
  private const float DecodeStep = MAX_VALUE / NumericMaxCode;

  /// <summary>
  /// Returns the given value clamped to the representable range.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static float Clamped(float value) =>
    value switch
    {
      < MIN_VALUE => MIN_VALUE,
      > MAX_VALUE => MAX_VALUE,
      _ => value
    };


  /// <summary>
  /// Packed storage (nibble pair).
  /// </summary>
  [FieldOffset(0)]
  private byte packedData;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="v1"></param>
  /// <param name="v2"></param>
  public FloatPairCompressedToByte(float v1, float v2)
  {
    packedData = 0;
    V1 = v1;
    V2 = v2;
  }


  /// <summary>
  /// Returns first value (V1).
  /// </summary>
  public float V1
  {
    readonly get
    {
      int code = packedData & 0x0F;
      return Decode(code);
    }
    set
    {
      int code = Encode(value);
      packedData = (byte)((packedData & 0xF0) | code);
    }
  }


  /// <summary>
  /// Returns second value (V2).
  /// </summary>
  public float V2
  {
    readonly get
    {
      int code = (packedData >> 4) & 0x0F;
      return Decode(code);
    }
    set
    {
      int code = Encode(value);
      packedData = (byte)((packedData & 0x0F) | (code << 4));
    }
  }


  /// <summary>
  /// Returns the encoded 4-bit code for the given float value.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static int Encode(float value)
  {
    if (float.IsNaN(value))
    {
      return NaNCode;
    }

    if (value < MIN_VALUE || value > MAX_VALUE)
    {
      throw new ArgumentOutOfRangeException(nameof(value), $"Value {value} must be in [{MIN_VALUE}, {MAX_VALUE}] or NaN.");
    }

    // Round to nearest representable code (0-14), then clamp.
    return System.Math.Clamp((int)MathF.Round(value * EncodeScale), 0, NumericMaxCode);
  }


  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static float Decode(int code) => code == NaNCode ? float.NaN : code * DecodeStep;

  /// <summary>
  /// Returns string representation.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    int c1 = packedData & 0x0F;
    int c2 = (packedData >> 4) & 0x0F;
    return $"V1: ({c1}) ? {Decode(c1):F5}, " +
           $"V2: ({c2}) ? {Decode(c2):F5}";
  }

  #region Equality

  /// <inheritdoc />
  public readonly bool Equals(FloatPairCompressedToByte other) => packedData == other.packedData;

  /// <inheritdoc />
  public override readonly bool Equals(object obj) => obj is FloatPairCompressedToByte other && packedData == other.packedData;

  /// <inheritdoc />
  public override readonly int GetHashCode() => packedData;

  public static bool operator ==(FloatPairCompressedToByte left, FloatPairCompressedToByte right) => left.packedData == right.packedData;

  public static bool operator !=(FloatPairCompressedToByte left, FloatPairCompressedToByte right) => left.packedData != right.packedData;

  #endregion
}
