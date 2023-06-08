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
using System.Text;
using System.Buffers.Binary;
using System.Collections.Specialized;
using System.Numerics;
using System.Runtime.CompilerServices;


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
      {
        return false;
      }

      return data == ((BitVector64)o).data;
    }

    public override int GetHashCode() => data.GetHashCode();
    
    public override string ToString()
    {
      return ToString(this);
    }


    /// <summary>
    /// Expands the 64 bits into 64 bytes within a specified array
    /// starting at specified index.
    /// </summary>
    /// <param name="bytes"></param>
    /// <param name="startIndex"></param>
    /// <returns></returns>
    public void SetExpandedBytes(byte[] bytes, int startIndex)
    {
      if (data == 0)
      {
        // Optimization for common case of all zeros.
        Array.Clear(bytes, startIndex, 64);
      }
      else
      {
        BitVector64 bits = new BitVector64(data);
        for (int i = 0; i < 64; i++)
        {
          bytes[i + startIndex] = bits.BitIsSet(i) ? (byte)1 : (byte)0;
        }
      }
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
      {
        if (bytes[i] == 1)
        {
          bits.SetBit(i - startIndex);
        }
      }
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
          {
            return count;
          }
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


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Reverse(ulong value) => BinaryPrimitives.ReverseEndianness(value);



    /// <summary>
    /// Fast 64 bit reversal (2x faster than table lookup).
    /// 
    /// With thanks to "steveu" (https://www.dsprelated.com/showthread/comp.dsp/131817-1.php)
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static UInt64 Mirror(UInt64 x)
    {
      x = ((x & 0xF0F0F0F0F0F0F0F0UL) >> 4) | ((x & 0x0F0F0F0F0F0F0F0FUL) << 4);
      x = ((x & 0xCCCCCCCCCCCCCCCCUL) >> 2) | ((x & 0x3333333333333333UL) << 2);
      return ((x & 0xAAAAAAAAAAAAAAAAUL) >> 1) | ((x &  0x5555555555555555UL) << 1);
    }

  }

}