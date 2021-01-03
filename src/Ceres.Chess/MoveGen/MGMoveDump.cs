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
    /// Short algebraic
    /// </summary>
    ShortAlgebraic,

    /// <summary>
    /// Standard algebraic, e.g. exf(q)
    /// </summary>
    StandardAlgebraic,

    /// <summary>
    /// Coordinate style, e.g. e7f8q.
    /// </summary>
    CoOrdinate,

    /// <summary>
    /// Leela chess zero coordinate format.
    /// </summary>
    LC0Coordinate,

    /// <summary>
    /// Leela chess zero coordinate format, with 960 extension for castling moves.
    /// </summary>
    LC0Coordinate960Format
  };


  
  /// <summary>
  /// Methods on MGMove related to dumping to strings (diagnostic).
  /// </summary>
  public partial struct MGMove : IEquatable<MGMove>
  {
    /// <summary>
    /// Converts a move to ascii and dumps it to stdout, unless a character 
    /// buffer is supplied, in which case the ascii is written to pBuffer instead.
    /// A number of notation styles are supported. Usually CoOrdinate should be used
    /// for WinBoard.
    /// </summary>
    /// <param name="M"></param>
    /// <param name="style"></param>
    public string MoveStr(MGMoveNotationStyle style = MGMoveNotationStyle.LongAlgebraic)
    {
      if (FromSquareIndex == 0b0 && ToSquareIndex == 0b0) return "(none)";

      BitBoard bbFrom = 1UL << FromSquareIndex;
      BitBoard bbTo = 1UL << ToSquareIndex;

      int from = BitOperations.TrailingZeroCount(bbFrom);
      int to = BitOperations.TrailingZeroCount(bbTo);

      if (style == MGMoveNotationStyle.LC0Coordinate960Format)
      {
        if (IsCastle)
        {
          // Castling moves always indicate target file at edge of board
          bool isWhite = ToSquare.Rank == 0;
          if (isWhite) 
            return CastleLong ? "e1a1" : "e1h1";
          else
            return CastleLong ? "e8a8" : "e8h8";
        }
        else
        {
          // Not castle, so in that case the format will be identical to LC0 coordinate.
          style = MGMoveNotationStyle.LC0Coordinate;
        }
      }

      if (style != MGMoveNotationStyle.CoOrdinate && style != MGMoveNotationStyle.LC0Coordinate)
      {
        if (CastleShort) return "O-O";
        if (CastleLong) return "O-O-O";
      }

      char c1 = (char)('h' - (from % 8));
      char c2 = (char)('h' - (to % 8));

      // Determine piece and assign character p accordingly:
      string p;
      switch ((byte)Piece & 7)
      {
        case MGPositionConstants.WPAWN:
          p = "";
          break;
        case MGPositionConstants.WKNIGHT:
          p = "N";
          break;
        case MGPositionConstants.WBISHOP:
          p = "B";
          break;
        case MGPositionConstants.WROOK:
          p = "R";
          break;
        case MGPositionConstants.WQUEEN:
          p = "Q";
          break;
        case MGPositionConstants.WKING:
          p = "K";
          break;
        default:
          p = "?";
          break;
      }
      
      if ((style == MGMoveNotationStyle.LongAlgebraic) 
       || (style == MGMoveNotationStyle.ShortAlgebraic)
       || (style == MGMoveNotationStyle.StandardAlgebraic)
       || (style == MGMoveNotationStyle.LC0Coordinate))
      {
        // Determine if move is a capture
        char capChar = (Capture || EnPassantCapture) ? 'x' : '-';

        StringBuilder sb = new StringBuilder();
        if ((style == MGMoveNotationStyle.ShortAlgebraic) || (style == MGMoveNotationStyle.LC0Coordinate))
        {
          sb.Append(c1);
          sb.Append(1 + (from >> 3));
          sb.Append(c2);
          sb.Append(1 + (to >> 3));
        }
        else if (style == MGMoveNotationStyle.StandardAlgebraic)
        {
          sb.Append(p);
          sb.Append(1 + (from >> 3));
          sb.Append(capChar);
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

      if (style == MGMoveNotationStyle.CoOrdinate)
      {
        StringBuilder sb = new StringBuilder();
        //string s = $"{p}{c1}{1 + (from >> 3)}{c2}{1 + (to >> 3)}";
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
        s += "b";
      if (PromoteKnight)
        s += "n";
      if (PromoteRook)
        s += "r";
      if (PromoteQueen)
        s += "q";
      return s;
    }
  }

}


