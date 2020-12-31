#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region License

/* 
License Note
   
This code originated from Github repository from Judd Niemann
and is licensed with the MIT License.

This version is modified by David Elliott, including a translation to C# and 
some moderate modifications to improve performance and modularity.
*/

/*

MIT License

Copyright(c) 2016-2017 Judd Niemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#endregion

#region Using directives

using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

using BitBoard = System.UInt64;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Static helper functions related to move generation.
  /// </summary>
  internal static class MGMoveGenFillFunctions
  {

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static byte GetSquareIndex(BitBoard b)
    {
      Debug.Assert(b != 0);

      return (byte)(63 - BitOperations.LeadingZeroCount(b));
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillStraightAttacksOccluded(BitBoard g, BitBoard p)
    {
      BitBoard a;
      a = FillRightOccluded(g, p);
      a |= FillLeftOccluded(g, p);
      a |= FillUpOccluded(g, p);
      a |= FillDownOccluded(g, p);
      a &= ~g; // exclude attacking pieces 
      return a;

    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillDiagonalAttacksOccluded(BitBoard g, BitBoard p)
    {
      BitBoard a;
      a = FillUpRightOccluded(g, p);
      a |= FillDownRightOccluded(g, p);
      a |= FillDownLeftOccluded(g, p);
      a |= FillUpLeftOccluded(g, p);
      a &= ~g; // exclude attacking pieces
      return a;

    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillKingAttacksOccluded(BitBoard g, BitBoard p)
    {
      BitBoard a, b;
      BitBoard t, u;
      a = g; t = g; t <<= 1; t &= 0xfefefefefefefefe; a |= t;
      b = a; b <<= 8; a |= b; u = a; u >>= 1; u &= 0x7f7f7f7f7f7f7f7f; a |= u;
      b = a; b >>= 8; a |= b; a &= ~g; a &= p;
      return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillKingAttacks(BitBoard g)
    {
      BitBoard a, b;
      BitBoard t, u;
      a = g; t = g; t <<= 1; t &= 0xfefefefefefefefe; a |= t;
      b = a; b <<= 8; a |= b; u = a; u >>= 1; u &= 0x7f7f7f7f7f7f7f7f; a |= u;
      b = a; b >>= 8; a |= b; a &= ~g;
      return a;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillKnightAttacksOccluded(BitBoard g, BitBoard p)
    {
      BitBoard l1 = (g >> 1) & 0x7f7f7f7f7f7f7f7f;
      BitBoard l2 = (g >> 2) & 0x3f3f3f3f3f3f3f3f;
      BitBoard r1 = (g << 1) & 0xfefefefefefefefe;
      BitBoard r2 = (g << 2) & 0xfcfcfcfcfcfcfcfc;
      BitBoard h1 = l1 | r1;
      BitBoard h2 = l2 | r2;
      return p & ((h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8));
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillKnightAttacks(BitBoard g)
    {
      BitBoard l1 = (g >> 1) & 0x7f7f7f7f7f7f7f7f;
      BitBoard l2 = (g >> 2) & 0x3f3f3f3f3f3f3f3f;
      BitBoard r1 = (g << 1) & 0xfefefefefefefefe;
      BitBoard r2 = (g << 2) & 0xfcfcfcfcfcfcfcfc;
      BitBoard h1 = l1 | r1;
      BitBoard h2 = l2 | r2;
      return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillUpOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      g |= p & (g << 8);
      p &= (p << 8);
      g |= p & (g << 16);
      p &= (p << 16);
      g |= p & (g << 32);
      return g;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillDownOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      g |= p & (g >> 8);
      p &= (p >> 8);
      g |= p & (g >> 16);
      p &= (p >> 16);
      g |= p & (g >> 32);
      return g;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillLeftOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      p &= 0xfefefefefefefefe;
      g |= p & (g << 1);
      p &= (p << 1);
      g |= p & (g << 2);
      p &= (p << 2);
      g |= p & (g << 4);
      return g;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillRightOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      p &= 0x7f7f7f7f7f7f7f7f;
      g |= p & (g >> 1);
      p &= (p >> 1);
      g |= p & (g >> 2);
      p &= (p >> 2);
      g |= p & (g >> 4);
      return g;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillUpRightOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      p &= 0x7f7f7f7f7f7f7f7f; // left wall
      g |= p & (g << 7);
      p &= (p << 7);
      g |= p & (g << 14);
      p &= (p << 14);
      g |= p & (g << 28);
      return g;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillDownRightOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      p &= 0x7f7f7f7f7f7f7f7f; // left wall
      g |= p & (g >> 9);
      p &= (p >> 9);
      g |= p & (g >> 18);
      p &= (p >> 18);
      g |= p & (g >> 36);
      return g;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillDownLeftOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      p &= 0xfefefefefefefefe; // right wall
      g |= p & (g >> 7);
      p &= (p >> 7);
      g |= p & (g >> 14);
      p &= (p >> 14);
      g |= p & (g >> 28);
      return g;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard FillUpLeftOccluded(BitBoard g, BitBoard p)
    {
      // Note: Fill includes pieces.
      p &= 0xfefefefefefefefe; // right wall
      g |= p & (g << 9);
      p &= (p << 9);
      g |= p & (g << 18);
      p &= (p << 18);
      g |= p & (g << 36);
      return g;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveUpSingleOccluded(BitBoard g, BitBoard p) => (p & (g << 8));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveUpRightSingleOccluded(BitBoard g, BitBoard p) => p & 0x7f7f7f7f7f7f7f7f & (g << 7);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveRightSingleOccluded(BitBoard g, BitBoard p) => p & 0x7f7f7f7f7f7f7f7f & (g >> 1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveDownRightSingleOccluded(BitBoard g, BitBoard p) => p & 0x7f7f7f7f7f7f7f7f & (g >> 9);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveDownSingleOccluded(BitBoard g, BitBoard p) => p & (g >> 8);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveDownLeftSingleOccluded(BitBoard g, BitBoard p)
    {
      p &= 0xfefefefefefefefe;
      return (p & (g >> 7));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveLeftSingleOccluded(BitBoard g, BitBoard p) => p & 0xfefefefefefefefe & (g << 1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveUpLeftSingleOccluded(BitBoard g, BitBoard p) => p & 0xfefefefefefefefe & (g << 9);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveDownLeftRightSingle(BitBoard g)
    {
      return (0xfefefefefefefefe & (g >> 7)) |  // DownLeft
             (0x7f7f7f7f7f7f7f7f & (g >> 9));    // DownRight
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static BitBoard MoveUpLeftRightSingle(BitBoard g)
    {
      return (0xfefefefefefefefe & (g << 9)) |  // UpLeft
             (0x7f7f7f7f7f7f7f7f & (g << 7));    // UpRight
    }

  }
}
