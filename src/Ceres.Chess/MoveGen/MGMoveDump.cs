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

using ManagedCuda.CudaBlas;
using System;
using System.Numerics;
using System.Text;

using BitBoard = System.UInt64;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Emumeration of notation style for string representation of an MGMove.
  /// </summary>
  public enum MGMoveNotationStyle
  {
    /// <summary>
    /// Long algebraic style, e.g. e7xf8(q)
    /// </summary>
    LongAlgebraic,

    /// <summary>
    /// Short algebraic, e.g. d4, 0-0 or 0-0-0.
    /// </summary>
    ShortAlgebraic,

    /// <summary>
    /// Coordinate style, e.g. Ng1f3 and e7f8q.
    /// </summary>
    PieceAndCoordinates,

    /// <summary>
    /// Ceres coordinate format, e.g. e2e4 and e1h1 indicates king takes rook for castling
    /// </summary>
    Coordinates
  };



  /// <summary>
  /// Methods on MGMove related to dumping to strings (diagnostic).
  /// </summary>
  public partial struct MGMove : IEquatable<MGMove>
  {
    /// <summary>
    /// Converts a move to ASCII and dumps it to stdout, unless a character buffer is supplied, in which case the ASCII is written to pBuffer instead.
    /// A number of notation styles are supported. Usually CoOrdinate should be used for WinBoard.
    /// </summary>
    /// <param name="style">The notation style for the string representation of the MGMove.</param>
    /// <returns>The string representation of the MGMove.</returns>
    public string MoveStr(MGMoveNotationStyle style = MGMoveNotationStyle.LongAlgebraic, bool isChess960 = false)
    {
      if (FromSquareIndex == 0 && ToSquareIndex == 0)
      {
        return "(none)";
      }

      if (!isChess960 && IsLegacyCastle)
      {
        bool isWhite = Piece == MGPositionConstants.MCChessPositionPieceEnum.WhiteKing;
        if (isWhite && FromSquareIndex == 3)
        {
          return CastleLong ? "e1c1" : "e1g1";
        }
        else if (!isWhite && FromSquareIndex == 59)
        {
          return CastleLong ? "e8c8" : "e8g8";
        }
      }

      if (style == MGMoveNotationStyle.ShortAlgebraic)
      {
        if (CastleShort)
        {
          return "O-O";
        }

        if (CastleLong)
        {
          return "O-O-O";
        }
      }

      BitBoard bbFrom = 1UL << FromSquareIndex;
      BitBoard bbTo = 1UL << ToSquareIndex;

      int from = BitOperations.TrailingZeroCount(bbFrom);
      int to = BitOperations.TrailingZeroCount(bbTo);

      char c1 = (char)('h' - (from % 8));
      char c2 = (char)('h' - (to % 8));

      // Determine piece and assign character p accordingly:
      string p = ((byte)Piece & 7) switch
      {
        MGPositionConstants.WPAWN => "",
        MGPositionConstants.WKNIGHT => "N",
        MGPositionConstants.WBISHOP => "B",
        MGPositionConstants.WROOK => "R",
        MGPositionConstants.WQUEEN => "Q",
        MGPositionConstants.WKING => "K",
        _ => "?",
      };

      if ((style == MGMoveNotationStyle.LongAlgebraic)
       || (style == MGMoveNotationStyle.ShortAlgebraic)
       || (style == MGMoveNotationStyle.Coordinates))
      {
        // Determine if move is a capture
        char capChar = (Capture || EnPassantCapture) ? 'x' : '-';

        StringBuilder sb = new StringBuilder();
        if ((style == MGMoveNotationStyle.ShortAlgebraic)
         || (style == MGMoveNotationStyle.Coordinates))
        {
          sb.Append(c1);
          sb.Append(1 + (from >> 3));
          sb.Append(c2);
          sb.Append(1 + (to >> 3));
        }
        else
        {
          sb.Append(p);
          sb.Append(c1);
          sb.Append(1 + (from >> 3));
          sb.Append(capChar);
          sb.Append(c2);
          sb.Append(1 + (to >> 3));
        }

        string s = sb.ToString();

        // Promotions:
        s = AddPromotions(s);

        return s;
      }

      if (style == MGMoveNotationStyle.PieceAndCoordinates)
      {
        StringBuilder sb = new StringBuilder();
        sb.Append(p);
        sb.Append(c1);
        sb.Append(1 + (from >> 3));
        sb.Append(c2);
        sb.Append(1 + (to >> 3));

        return AddPromotions(sb.ToString());
      }

      return "";
    }
    
    private string AddPromotions(string s)
    {
      if (PromoteBishop)
      {
        s += "b";
      }

      if (PromoteKnight)
      {
        s += "n";
      }

      if (PromoteRook)
      {
        s += "r";
      }

      if (PromoteQueen)
      {
        s += "q";
      }

      return s;
    }
  }

}