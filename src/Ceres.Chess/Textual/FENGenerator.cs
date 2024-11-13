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
using System.Linq;
using System.Text;

#endregion

namespace Ceres.Chess.Textual
{
  /// <summary>
  /// Static helper methods relating to geneting FENs.
  /// </summary>
  internal static class FENGenerator
  {
    internal static char GetCastlingFileChar(int fileIndex, bool isWhite) =>
      isWhite ? Char.ToUpper((char)('h' - fileIndex)) : (char)('h' - fileIndex);    

    private static char[] normalCastlingCharsConverted = new[] { 'H', 'A', 'h', 'a' };

    internal static string GetFEN(Position pos)
    {
      // Determine if the side to move is white
      bool weAreWhite = pos.MiscInfo.SideToMove == 0;

      // Initialize the FEN string with the piece positions
      StringBuilder fen = new();
      fen.Append(GetFENPieces(pos));

      // Append the side to move
      fen.Append(weAreWhite ? " w" : " b");
      fen.Append(" ");

      // Initialize the castling rights string
      StringBuilder castlingSB = new();

      // Append castling rights for white kingside
      if (pos.MiscInfo.WhiteCanOO)
      {
        char file = GetCastlingFileChar(pos.MiscInfo.RookInfo.WhiteKRInitPlacement, true);
        castlingSB.Append(file);
      }

      // Append castling rights for white queenside
      if (pos.MiscInfo.WhiteCanOOO)
      {
        char file = GetCastlingFileChar(pos.MiscInfo.RookInfo.WhiteQRInitPlacement, true);
        castlingSB.Append(file);
      }

      // Append castling rights for black kingside
      if (pos.MiscInfo.BlackCanOO)
      {
        var file = GetCastlingFileChar(pos.MiscInfo.RookInfo.BlackKRInitPlacement, false);
        castlingSB.Append(file);
      }

      // Append castling rights for black queenside
      if (pos.MiscInfo.BlackCanOOO)
      {
        var file = GetCastlingFileChar(pos.MiscInfo.RookInfo.BlackQRInitPlacement, false);
        castlingSB.Append(file);
      }

      // Convert castling rights to standard notation if applicable
      string castling = castlingSB.ToString();
      var normalCastlingPerformed = castling.All(c => normalCastlingCharsConverted.Contains(c));
      //pos.ToMGPosition.BlackCanCastle
      // Keep the normal castling notation when in a normal castling position
      if (normalCastlingPerformed)
      {
        castling = castling.Replace('H', 'K').Replace('A', 'Q').Replace('h', 'k').Replace('a', 'q');
      }
      if (String.IsNullOrEmpty(castling))
      {
        castling = "-";
      }

      // Append castling rights to FEN
      fen.Append(castling + " ");

      // Determine en passant target square
      string epTarget = "-";
      if (pos.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
      {
        epTarget = pos.MiscInfo.EnPassantFileChar + (weAreWhite ? "6" : "3");
      }

      // Append en passant target square, halfmove clock, and fullmove number to FEN
      int moveNumToShow = pos.MiscInfo.MoveNum / 2; // move counter incremented only after each black move
      fen.Append(epTarget + " ");
      fen.Append(pos.MiscInfo.Move50Count + " ");
      fen.Append(moveNumToShow);

      // Return the complete FEN string
      return fen.ToString();
    }


    private static string GetFENPieces(Position pos)
    {
      StringBuilder fen = new StringBuilder();

      for (int r = 7; r >= 0; r--)
      {
        if (r != 7)
        {
          fen.Append("/");
        }
        int numBlank = 0;
        for (int c = 7; c >= 0; c--)
        {
          char thisChar = pos[Square.FromFileAndRank(7 - c, r)].Char;
          if (thisChar == Piece.EMPTY_SQUARE_CHAR)
          {
            numBlank++;
          }
          else
          {
            if (numBlank > 0)
            {
              fen.Append(numBlank);
            }
            numBlank = 0;
            fen.Append(thisChar);
          }
        }

        if (numBlank > 0)
        {
          fen.Append(numBlank);
        }
      }

      return fen.ToString();
    }

  }
}
