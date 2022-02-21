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
using System.Text;

#endregion

namespace Ceres.Chess.Textual
{
  /// <summary>
  /// Static helper methods relating to geneting FENs.
  /// </summary>
  internal static class FENGenerator
  {
    internal static string GetFEN(Position pos)
    {
      // KQkq - 0 1
      bool weAreWhite = pos.MiscInfo.SideToMove == 0;

      StringBuilder fen = new ();
      fen.Append(GetFENPieces(pos));

      fen.Append(weAreWhite ? " w" : " b");
      fen.Append(" ");

      StringBuilder castlingSB = new ();
      if (pos.MiscInfo.WhiteCanOO)
      {
        castlingSB.Append("K");
      }
      if (pos.MiscInfo.WhiteCanOOO)
      {
        castlingSB.Append("Q");
      }
      if (pos.MiscInfo.BlackCanOO)
      {
        castlingSB.Append("k");
      }
      if (pos.MiscInfo.BlackCanOOO)
      {
        castlingSB.Append("q");
      }

      string castling = castlingSB.ToString();
      if (castling == "")
      {
        castling = "-";
      }

      //rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
      string epTarget = "-";
      if (pos.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
      {
        epTarget = pos.MiscInfo.EnPassantFileChar + (weAreWhite ? "6" : "3");
      }

      int moveNumToShow = (1 + pos.MiscInfo.MoveNum) / 2; // move counter incremented only after each black move
      fen.Append(castling + " ");
      fen.Append(epTarget + " ");
      fen.Append(pos.MiscInfo.Move50Count + " ");
      fen.Append(moveNumToShow);

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
          char thisChar = pos[Square.FromFileAndRank(7-c, r)].Char;
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
