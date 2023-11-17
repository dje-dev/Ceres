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
    /// Floating point values will be stored as bytes
    /// via this implicit fixed scaling factor.
    /// </summary>
    public const float SCALING_FACTOR = 100;

    /// <summary>
    /// The value which will be persisted
    /// (but represents the true value scaled up by SCALING_FACTOR).
    /// </summary>
    private byte value;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="value"></param>
    public ByteScaled(int value) => Value = value;

    /// <summary>
    /// Returns raw underlying value, before scaling.
    /// </summary>
    public byte RawValue => value;


    /// <summary>
    /// Accessor to underlying value (in its true scale).
    /// </summary>
    public float Value
    {
      get => value / SCALING_FACTOR;
      set
      {
        if (value == 0)
        {
          this.value = 0; // common case
        }
        else if (value == SCALING_FACTOR)
        {
          this.value = 1; // common case
        }
        else
        {
          float rawVal = value * SCALING_FACTOR;
          if (rawVal > byte.MaxValue)
          {
            throw new Exception("Requested value for ByteScaled would overflow byte range (255).");
          }

          this.value = (byte)MathF.Round(rawVal);
        }
      }
    }


    /// <summary>
    /// Returns if value is exactly 0 or 1.
    /// </summary>
    public readonly bool IsZeroOrOne => value == 0 || value == SCALING_FACTOR;


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
          throw new NotImplementedException("Invalid binary value found inside ByteScaled: " + b.value);
        }
      }
    }


    #region Static helper methods

    /// <summary>
    /// Implicitly converts an integer value to ByteScaled.
    /// </summary>
    /// <param name="value">Integer value to be converted.</param>
    public static implicit operator ByteScaled(int value) => new ByteScaled(value);

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
      public bool Equals(ByteScaled x, ByteScaled y) => x.value == y.value;
      public int GetHashCode(ByteScaled obj) => obj.value;
    }

    #endregion


    public override string ToString()
    {
      return $"({Value} as {value})";
    }
  }

}
