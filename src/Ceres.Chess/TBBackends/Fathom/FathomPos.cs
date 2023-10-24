#region License notice

// NOTE: This file is substantially a transliteration from C to C# 
//       of code from the Fathom project.
//       Both Fathom and Ceres copyrights are included below.

/*
Copyright (c) 2015 basil00
Modifications Copyright (c) 2016-2020 by Jon Dart
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
using Ceres.Base.DataTypes;

#endregion


[assembly: InternalsVisibleTo("Ceres.Chess.Test")]

namespace Ceres.Chess.TBBackends.Fathom
{
  /// <summary>
  /// Represents a board position for use internally with Fathom methods.
  /// </summary>
  internal record struct FathomPos
  {
    public ulong White;
    public ulong Black;
    public ulong Kings;
    public ulong Queens;
    public ulong Rooks;
    public ulong Bishops;
    public ulong Knights;
    public ulong Pawns;
    public byte Rule50;
    public byte EnPassant;
    public bool Turn;
    //public short move;
    public int Castling;

    internal FathomPos(ulong white, ulong black, ulong kings, ulong queens, ulong rooks, ulong bishops, 
                       ulong knights, ulong pawns, byte rule50, byte ep, bool turn, int castling)
    {
      White = white;
      Black = black;
      Kings = kings;
      Queens = queens;
      Rooks = rooks;
      Bishops = bishops;
      Knights = knights;
      Pawns = pawns;
      Rule50 = rule50;
      EnPassant = ep;
      Turn = turn;
      Castling = castling;
    }


    /// <summary>
    /// Converts a Position into corresponding FathomPos.
    /// </summary>
    public static FathomPos FromPosition(in Position pos)
    {
      FathomPos fp2 = default;

      if (pos.MiscInfo.EnPassantRightsPresent)
      {
        // The Fathom logic removes the ep indicator if
        // the move list is not changed by it.
#if NOT
        // To avoid complexity of checking this
        // in the infrequent occurrence of en passant rights in tablebase probes.
        // we fallback to the original implementation in this case.
        FathomFENParsing.parse_FEN(ref fp2, pos.FEN);
        return fp2;

        // TODO: consider someday implementing this directly.
        //       Note that looking at positions of opponent pawns
        //       is not completely correct due to possible pins,
        //       therefore a full movegen would be necessary
        //       in cases where an opponent pawn is on a possible move square.
#endif
        int rank = pos.MiscInfo.SideToMove == SideType.White ? 5 : 2;
        int file = (int)pos.MiscInfo.EnPassantFileIndex;
        fp2.EnPassant = (byte)FathomMoveGen.square(rank, file);
      }

      Span<BitVector64> bv = stackalloc BitVector64[16];
      pos.InitializeBitmaps(bv, false);

      fp2.Rule50 = pos.MiscInfo.Move50Count;
      fp2.Turn = pos.MiscInfo.SideToMove == SideType.White;
      //fp2.move = Math.Min((byte)255, pos.MiscInfo.MoveNum);

      
      if (pos.MiscInfo.WhiteCanOO) fp2.Castling |= FathomFENParsing.TB_CASTLING_K;
      if (pos.MiscInfo.WhiteCanOOO) fp2.Castling |= FathomFENParsing.TB_CASTLING_Q;
      if (pos.MiscInfo.BlackCanOO) fp2.Castling |= FathomFENParsing.TB_CASTLING_k;
      if (pos.MiscInfo.BlackCanOOO) fp2.Castling |= FathomFENParsing.TB_CASTLING_q;

      fp2.White = (ulong)(bv[1].Data | bv[2].Data | bv[3].Data | bv[4].Data | bv[5].Data | bv[6].Data);
      fp2.Black = (ulong)(bv[9].Data | bv[10].Data | bv[11].Data | bv[12].Data | bv[13].Data | bv[14].Data);

      fp2.Pawns = (ulong)(bv[1].Data | bv[9].Data);
      fp2.Knights = (ulong)(bv[2].Data | bv[10].Data);
      fp2.Bishops = (ulong)(bv[3].Data | bv[11].Data);
      fp2.Rooks = (ulong)(bv[4].Data | bv[12].Data);
      fp2.Queens = (ulong)(bv[5].Data | bv[13].Data);
      fp2.Kings = (ulong)(bv[6].Data | bv[14].Data);

      return fp2;
    }

  }

}
