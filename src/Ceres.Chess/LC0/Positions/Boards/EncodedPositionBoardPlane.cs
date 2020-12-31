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

using Ceres.Base;
using Ceres.Base.DataTypes;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.LC0.Boards
{
  /// <summary>
  /// A single plane (for a specific piece type or metadata) within an encoded board.
  /// </summary>
  [Serializable()]
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  public readonly struct EncodedPositionBoardPlane : IEquatable<EncodedPositionBoardPlane>
  {
    #region Raw structure data

    public readonly BitVector64 Bits;

    #endregion
    
    public enum PlanesType
    {
      PlaneOurPawns, PlaneOurKnights, PlaneOurBishops, PlaneOurRooks, PlaneOurQueens, PlaneOurKing,
      PlaneTheirPawns, PlaneTheirKnights, PlaneTheirBishops, PlaneTheirRooks, PlaneTheirQueens, PlaneTheirKing,
      Repetitions
    }


    public int NumberBitsSet => Bits.NumberBitsSet;
    public void SetBit(int index) => Bits.SetBit(index);
    public bool BitIsSet(int index) => Bits.BitIsSet(index);
    public ulong Data => (ulong)Bits.Data;


    
    /// <summary>
    /// Constructor from long.
    /// </summary>
    /// <param name="data"></param>
    public EncodedPositionBoardPlane(long data)  
    {
      Bits = data;
    }

    /// <summary>
    /// Constructor from bitvector.
    /// </summary>
    /// <param name="bits"></param>
    public EncodedPositionBoardPlane(BitVector64 bits)
    {
      Bits = bits;
    }

    /// <summary>
    /// Returns mirrored board.
    /// </summary>
    public EncodedPositionBoardPlane Mirrored => new EncodedPositionBoardPlane((long)BitVector64.Mirror((ulong)this.Bits.Data));

    /// <summary>
    /// Returns reversed board.
    /// </summary>
    public EncodedPositionBoardPlane Reversed
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return new EncodedPositionBoardPlane((long)BitVector64.Reverse((ulong)this.Bits.Data));
      }
    }

    public override string ToString()
    {
      if (Bits.Data == 0)
        return "<LZBoardPlane [EMPTY]>";
      else
      return $"<LZBoardPlane {Bits.Data} { Bits.ToString() }>";
    }


    public void SetBytesRepresentation(byte[] bytes, int floatsStartIndex)
    {
      if (Bits.Data == 0)
      {
        Array.Clear(bytes, floatsStartIndex, 64);
      }
      else
      {
        int c = floatsStartIndex;
        for (int i = 0; i < 64; i++)
        {
          bytes[c++] = Bits.BitIsSet(i) ? (byte)1 : (byte)0;
        }
      }
    }


    #region Operator overloads
    public override int GetHashCode() => Bits.GetHashCode();

    public static bool operator ==(EncodedPositionBoardPlane lhs, EncodedPositionBoardPlane rhs)
    {
      return lhs.Bits.Data == rhs.Bits.Data;
    }

    public static bool operator !=(EncodedPositionBoardPlane lhs, EncodedPositionBoardPlane rhs)
    {
      return lhs.Bits.Data != rhs.Bits.Data;
    }

    public override bool Equals(object obj)
    {
      return obj is EncodedPositionBoardPlane plane && Equals(plane);
    }

    public bool Equals(EncodedPositionBoardPlane other)
    {
      return Bits.Equals(other.Bits);
    }

    #endregion

  }

}
