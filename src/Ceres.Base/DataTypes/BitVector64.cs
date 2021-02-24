#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion


// System.Collections.Specialized.BitVector32.cs
//
// Author:
//   Miguel de Icaza 
//   Lawrence Pit 
//   Andrew Birkett 
//   Andreas Nahr 
//
//   WonYoung(Brad) Jeong : converted 32bits to 64bits
//
//
// (C) Ximian, Inc.  <a href="http://www.ximian.com">http://www.ximian.com
// Copyright (C) 2005 Novell, Inc (<a href="http://www.novell.com">http://www.novell.com</a>)
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//

using System;
using System.Buffers.Binary;
using System.Collections.Specialized;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;


namespace Ceres.Base.DataTypes // TO DO: get rid of this in favor of BitVector64_DJE
{
  [Serializable]
  public struct BitVector64
  {
    long data;

    public static implicit operator BitVector64(long value) => new BitVector64(value);

    #region Constructors
    public BitVector64(BitVector64 source)
    {
      this.data = source.data;
    }
    public BitVector64(BitVector32 source)
    {
      this.data = source.Data;
    }

    public BitVector64(ulong source) => this.data = (long)source;
    public BitVector64(long source) =>  this.data = source;
    public BitVector64(int init) =>  this.data = init;

    public BitVector64 Reversed => new BitVector64(Reverse((ulong)data));

    #endregion


    #region Properties
    public long Data
    {
      get { return this.data; }
    }

    public void SetData(long value) => data = value;
    

    public bool this[long mask]
    {
      get
      {
#if NET_2_0
                return (this.data & mask) == mask;
#else
        long tmp = /*(uint)*/this.data;
        return (tmp & (long)mask) == (long)mask;
#endif
      }

      set
      {
        if (value)
          this.data |= mask;
        else
          this.data &= ~mask;
      }
    }
    #endregion

    public override bool Equals(object o)
    {
      if (!(o is BitVector64))
        return false;

      return data == ((BitVector64)o).data;
    }

    public override int GetHashCode() => data.GetHashCode();
    
    public override string ToString()
    {
      return ToString(this);
    }

    /// <summary>
    /// Returns a BitVector64 assembled from an array of bytes
    /// which contain the expanded (each bit to a byte) representation.
    /// </summary>
    /// <param name="bytes"></param>
    /// <param name="startIndex"></param>
    /// <returns></returns>
    public static BitVector64 FromExpandedBytes(byte[] bytes, int startIndex)
    {
      BitVector64 bits = new BitVector64();
      for (int i = startIndex; i < startIndex + 64; i++)
        if (bytes[i] == 1)
          bits.SetBit(i - startIndex);
      return bits;
    }


    public static string ToString(BitVector64 value)
    {
      StringBuilder sb = new StringBuilder(0x2d);
      sb.Append("BitVector64{");
      ulong data = (ulong)value.Data;
      for (int i = 0; i < 0x40; i++)
      {
        sb.Append(((data & 0x8000000000000000) == 0) ? '0' : '1');
        data = data << 1;
      }

      sb.Append("}");
      return sb.ToString();

      //StringBuilder b = new StringBuilder();
      //b.Append("BitVector64{");
      //ulong mask = (ulong)Convert.ToInt64(0x8000000000000000);
      //while (mask > 0)
      //{
      //    b.Append((((ulong)value.Data & mask) == 0) ? '0' : '1');
      //    mask >>= 1;
      //}
      //b.Append('}');
      //return b.ToString();
    }

    public int GetSetBitIndices(Span<byte> indices, int startIndexOffset, int maxBitsToReturn)
    {
      int count = 0;
      for (int bit = 0; bit < 64; bit++)
      {
        if (BitIsSet(bit))
        {
          if (count >= maxBitsToReturn)
            return count;
          indices[startIndexOffset + count] = (byte)bit;
          count++;
        }
      }

      return count;
    }


    // Private utilities

    public int NumberBitsSet => BitOperations.PopCount((ulong)this.data);

    public bool BitIsSet(int i) => this[1L << i];

    public void SetBit(int i) => this[1L << i] = true;
    public void SetBit(int i, float value) => this[1L << i] = value == 0 ? false : true;

    private static int HighestSetBit(int i)
    {
      for (int bit = 63; bit >= 0; bit--)
      {
        long mask = 1L << bit;
        if ((mask & i) != 0)
        {
          return bit;
        }
      }
      return -1;
    }

    // --------------------------------------------------------------------------------------------
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Reverse(ulong value) => BinaryPrimitives.ReverseEndianness(value);


    /// <summary>
    /// Table that facilitates bit reversal.
    /// Note that use of ReadOnlySpan allows compiler optimization to refer directly to data
    /// (see https://github.com/dotnet/roslyn/pull/24621).
    /// </summary>
    static ReadOnlySpan<byte> BitReverseTable256 => new[]
    {
      (byte)0x00, (byte)0x80, (byte)0x40, (byte)0xC0, (byte)0x20, (byte)0xA0, (byte)0x60, (byte)0xE0, (byte)0x10, (byte)0x90, (byte)0x50, (byte)0xD0, (byte)0x30, (byte)0xB0, (byte)0x70, (byte)0xF0,
      (byte)0x08, (byte)0x88, (byte)0x48, (byte)0xC8, (byte)0x28, (byte)0xA8, (byte)0x68, (byte)0xE8, (byte)0x18, (byte)0x98, (byte)0x58, (byte)0xD8, (byte)0x38, (byte)0xB8, (byte)0x78, (byte)0xF8,
      (byte)0x04, (byte)0x84, (byte)0x44, (byte)0xC4, (byte)0x24, (byte)0xA4, (byte)0x64, (byte)0xE4, (byte)0x14, (byte)0x94, (byte)0x54, (byte)0xD4, (byte)0x34, (byte)0xB4, (byte)0x74, (byte)0xF4,
      (byte)0x0C, (byte)0x8C, (byte)0x4C, (byte)0xCC, (byte)0x2C, (byte)0xAC, (byte)0x6C, (byte)0xEC, (byte)0x1C, (byte)0x9C, (byte)0x5C, (byte)0xDC, (byte)0x3C, (byte)0xBC, (byte)0x7C, (byte)0xFC,
      (byte)0x02, (byte)0x82, (byte)0x42, (byte)0xC2, (byte)0x22, (byte)0xA2, (byte)0x62, (byte)0xE2, (byte)0x12, (byte)0x92, (byte)0x52, (byte)0xD2, (byte)0x32, (byte)0xB2, (byte)0x72, (byte)0xF2,
      (byte)0x0A, (byte)0x8A, (byte)0x4A, (byte)0xCA, (byte)0x2A, (byte)0xAA, (byte)0x6A, (byte)0xEA, (byte)0x1A, (byte)0x9A, (byte)0x5A, (byte)0xDA, (byte)0x3A, (byte)0xBA, (byte)0x7A, (byte)0xFA,
      (byte)0x06, (byte)0x86, (byte)0x46, (byte)0xC6, (byte)0x26, (byte)0xA6, (byte)0x66, (byte)0xE6, (byte)0x16, (byte)0x96, (byte)0x56, (byte)0xD6, (byte)0x36, (byte)0xB6, (byte)0x76, (byte)0xF6,
      (byte)0x0E, (byte)0x8E, (byte)0x4E, (byte)0xCE, (byte)0x2E, (byte)0xAE, (byte)0x6E, (byte)0xEE, (byte)0x1E, (byte)0x9E, (byte)0x5E, (byte)0xDE, (byte)0x3E, (byte)0xBE, (byte)0x7E, (byte)0xFE,
      (byte)0x01, (byte)0x81, (byte)0x41, (byte)0xC1, (byte)0x21, (byte)0xA1, (byte)0x61, (byte)0xE1, (byte)0x11, (byte)0x91, (byte)0x51, (byte)0xD1, (byte)0x31, (byte)0xB1, (byte)0x71, (byte)0xF1,
      (byte)0x09, (byte)0x89, (byte)0x49, (byte)0xC9, (byte)0x29, (byte)0xA9, (byte)0x69, (byte)0xE9, (byte)0x19, (byte)0x99, (byte)0x59, (byte)0xD9, (byte)0x39, (byte)0xB9, (byte)0x79, (byte)0xF9,
      (byte)0x05, (byte)0x85, (byte)0x45, (byte)0xC5, (byte)0x25, (byte)0xA5, (byte)0x65, (byte)0xE5, (byte)0x15, (byte)0x95, (byte)0x55, (byte)0xD5, (byte)0x35, (byte)0xB5, (byte)0x75, (byte)0xF5,
      (byte)0x0D, (byte)0x8D, (byte)0x4D, (byte)0xCD, (byte)0x2D, (byte)0xAD, (byte)0x6D, (byte)0xED, (byte)0x1D, (byte)0x9D, (byte)0x5D, (byte)0xDD, (byte)0x3D, (byte)0xBD, (byte)0x7D, (byte)0xFD,
      (byte)0x03, (byte)0x83, (byte)0x43, (byte)0xC3, (byte)0x23, (byte)0xA3, (byte)0x63, (byte)0xE3, (byte)0x13, (byte)0x93, (byte)0x53, (byte)0xD3, (byte)0x33, (byte)0xB3, (byte)0x73, (byte)0xF3,
      (byte)0x0B, (byte)0x8B, (byte)0x4B, (byte)0xCB, (byte)0x2B, (byte)0xAB, (byte)0x6B, (byte)0xEB, (byte)0x1B, (byte)0x9B, (byte)0x5B, (byte)0xDB, (byte)0x3B, (byte)0xBB, (byte)0x7B, (byte)0xFB,
      (byte)0x07, (byte)0x87, (byte)0x47, (byte)0xC7, (byte)0x27, (byte)0xA7, (byte)0x67, (byte)0xE7, (byte)0x17, (byte)0x97, (byte)0x57, (byte)0xD7, (byte)0x37, (byte)0xB7, (byte)0x77, (byte)0xF7,
      (byte)0x0F, (byte)0x8F, (byte)0x4F, (byte)0xCF, (byte)0x2F, (byte)0xAF, (byte)0x6F, (byte)0xEF, (byte)0x1F, (byte)0x9F, (byte)0x5F, (byte)0xDF, (byte)0x3F, (byte)0xBF, (byte)0x7F, (byte)0xFF
    };


    [Serializable]
    [StructLayout(LayoutKind.Explicit, Size = 8, Pack = 1)]
    struct MirrorBytesStructHelper
    {
      [FieldOffset(0)] public ulong B;

      [FieldOffset(0)] public byte B0;
      [FieldOffset(1)] public byte B1;
      [FieldOffset(2)] public byte B2;
      [FieldOffset(3)] public byte B3;
      [FieldOffset(4)] public byte B4;
      [FieldOffset(5)] public byte B5;
      [FieldOffset(6)] public byte B6;
      [FieldOffset(7)] public byte B7;

      /// <summary>
      /// Runtime: 200 million per second, seems somewhat faster than possible bit twiddling options
      /// </summary>
      internal void Mirror()
      {
        B0 = BitReverseTable256[B0];
        B1 = BitReverseTable256[B1];
        B2 = BitReverseTable256[B2];
        B3 = BitReverseTable256[B3];
        B4 = BitReverseTable256[B4];
        B5 = BitReverseTable256[B5];
        B6 = BitReverseTable256[B6];
        B7 = BitReverseTable256[B7];
      }
    }
    // --------------------------------------------------------------------------------------------
    public static UInt64 Mirror(UInt64 v)
    {
      MirrorBytesStructHelper helper = default;

      helper.B = v;
      helper.Mirror();
      return helper.B;
    }


  }

}