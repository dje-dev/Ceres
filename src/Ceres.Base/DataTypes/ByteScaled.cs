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
using System.Diagnostics;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.DataType
{
  /// <summary>
  /// A one byte structure containing a value that is
  /// transparently scaled to/from float based on a constant SCALING_FACTOR.
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 1, Size = 1)]
  public struct ByteScaled
  {
    /// <summary>
    /// Maximum value that can be stored in a ByteScaled.
    /// </summary>
    public const float MAX_VALUE = (byte.MaxValue - 1) / SCALING_FACTOR;

    /// <summary>
    /// Floating point values will be stored as bytes
    /// via this implicit fixed scaling factor.
    /// </summary>
    public const float SCALING_FACTOR = 100; // N.B. assumed to be an integer by code below

    /// <summary>
    /// The value which will be persisted
    /// (but represents the true value scaled up by SCALING_FACTOR).
    /// </summary>
    private byte scaledValue;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="value"></param>
    public ByteScaled(int value) => Value = value;

    /// <summary>
    /// Returns raw underlying value, before scaling.
    /// </summary>
    public readonly byte RawValue => scaledValue;


    /// <summary>
    /// Accessor to underlying value (in its true scale).
    /// </summary>
    public float Value
    {
      readonly get => scaledValue / SCALING_FACTOR;
      set
      {
        if (value == 0)
        {
          scaledValue = 0; // common case
        }
        else if (value == 1)
        {
          scaledValue = (byte)SCALING_FACTOR; // common case
        }
        else
        {
          float rawVal = value * SCALING_FACTOR;
          Debug.Assert(rawVal <= byte.MaxValue, "Requested value for ByteScaled would overflow byte range (255).");
          scaledValue = (byte)MathF.Round(rawVal);
        }
      }
    }


    /// <summary>
    /// Returns if value is exactly 0 or 1.
    /// </summary>
    public readonly bool IsZeroOrOne => scaledValue == 0 || scaledValue == SCALING_FACTOR;


    /// <summary>
    /// Verifies that all members of a span of ByteScaled have value of either 1 or 0.
    /// </summary>
    /// <param name="bytes"></param>
    /// <exception cref="NotImplementedException"></exception>
    public static void ValidateZeroOrOne(ReadOnlySpan<ByteScaled> bytes)
    {
      foreach (ByteScaled b in bytes)
      {
        if (!b.IsZeroOrOne)
        {
          throw new NotImplementedException("Invalid binary value found inside ByteScaled: " + b.scaledValue);
        }
      }
    }


    #region Conversion operators

    /// <summary>
    /// Implicit conversion from float to ByteScaled.
    /// </summary>
    /// <param name="value">Float value to convert.</param>
    public static implicit operator ByteScaled(float value) => new ByteScaled { Value = value };

    /// <summary>
    /// Implicit conversion from ByteScaled to float.
    /// </summary>
    /// <param name="b">ByteScaled value to convert.</param>
    public static implicit operator float(ByteScaled b) => b.Value;

    #endregion


    #region Static helper methods

    /// <summary>
    /// Determines if two spans of ByteScaled are equal.
    /// </summary>
    /// <param name="s1"></param>
    /// <param name="s2"></param>
    /// <returns></returns>
    public static bool SpansEqual(ReadOnlySpan<ByteScaled> s1, ReadOnlySpan<ByteScaled> s2)
    {
      for (int i = 0; i < s1.Length; i++)
      {
        if (s1[i].Value != s2[i].Value)
        {
          return false;
        }
      }
      return true;
    }

    #endregion


    #region Equality comparer

    public class ByteScaledComparer : IEqualityComparer<ByteScaled>
    {
      public bool Equals(ByteScaled x, ByteScaled y) => x.scaledValue == y.scaledValue;
      public int GetHashCode(ByteScaled obj) => obj.scaledValue;
    }

    #endregion


    public override string ToString()
    {
      return $"({Value} as {scaledValue})";
    }
  }

}
