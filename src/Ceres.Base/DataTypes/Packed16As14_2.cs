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

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// A 16-bit value that packs two logical fields:
/// * 14-bit unsigned integer  (bits 0-13)
/// * 2-bit  unsigned integer  (bits 14-15)
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 2)]
public struct Packed16As14_2
{
  /// <summary>
  /// Raw 16 bits.
  /// </summary>
  [FieldOffset(0)]
  private ushort bits;

  private const int SHIFT_2BITS = 14;
  private const ushort MASK_14BITS = 0x3FFF; // 14 bits (0011 1111 1111 1111)
  private const ushort MASK_2BITS = 0x03;    // 2 bits after shifting


  /// <summary>
  /// Constructor to initialize both fields at once.
  /// </summary>
  /// <param name="value14Bits">14-bit value (0-16383).</param>
  /// <param name="value2Bits">2-bit value (0-3).</param>
  public Packed16As14_2(ushort value14Bits, byte value2Bits)
  {
    Debug.Assert(value14Bits <= MASK_14BITS);
    Debug.Assert(value2Bits <= MASK_2BITS);
    bits = (ushort)(value14Bits | (value2Bits << SHIFT_2BITS));
  }


  /// <summary>
  /// Returns or sets the 14-bit unsigned value (0-16383).
  /// </summary>
  public ushort Value14BitsUShort
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get => (ushort)(bits & MASK_14BITS);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      Debug.Assert(value <= MASK_14BITS);
      bits = (ushort)((bits & ~MASK_14BITS) | value);
    }
  }


  /// <summary>
  /// Returns or sets the 2-bit unsigned value (0-3).
  /// </summary>
  public byte Value2BitsByte
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get => (byte)((bits >> SHIFT_2BITS) & MASK_2BITS);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      Debug.Assert(value <= MASK_2BITS);
      bits = (ushort)((bits & MASK_14BITS) | (value << SHIFT_2BITS));
    }
  }


  /// <summary>
  /// Returns string representation.
  /// </summary>
  public override readonly string ToString()
  {
    return $"{{ Value14BitsUShort={Value14BitsUShort}, Value2BitsByte={Value2BitsByte} }}";
  }
}
