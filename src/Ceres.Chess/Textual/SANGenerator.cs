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
using System.Collections.Generic;

#endregion

namespace Ceres.Chess.Textual
{
  /// <summary>
  /// Manages generating move strings in SAN (standard algebrai notation).
  /// 
  /// See: http://cfajohnson.com/chess/SAN/.
  /// </summary>
  public static class SANGenerator
  {
    /// <summary>
    /// Returns the SAN string corresponding to specified move from specified position.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="move"></param>
    /// <returns></returns>
    public static string ToSANMove(in Position pos, Move move)
    {
      switch (move.Type)
      {
        case Move.MoveType.BlackWins:
          return "0-1";

        case Move.MoveType.WhiteWins:
          return "1-0";

        case Move.MoveType.Draw:
          return "=";

        case Move.MoveType.MoveCastleLong:
          return "O-O-O";

        case Move.MoveType.MoveCastleShort:
          return "O-O";

        case Move.MoveType.MoveNonCastle:
          return SANRegularMove(pos, move);

      }

      throw new Exception("Internal error: Unknown Move type");
    }


    /// <summary>
    /// Returns List of squares having pieces that can move to specified square
    /// (excluding the current move).
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="moves"></param>
    /// <param name="pieceType"></param>
    /// <param name="fromSquare"></param>
    /// <param name="toSquare"></param>
    /// <returns></returns>
    static List<Square> SquaresCanMoveToSquare(in Position pos, List<Move> moves, PieceType pieceType,
                                               Square fromSquare, Square toSquare)
    {
      List<Square> ret = new List<Ceres.Chess.Square>();
      foreach (Move move in moves)
      {
        if (move.Type == Move.MoveType.MoveNonCastle)
        {
          if (pos.PieceOnSquare(move.FromSquare).Type == pieceType
           && move.ToSquare == toSquare
           && move.FromSquare != fromSquare)
            ret.Add(move.FromSquare);
        }
      }


      return ret;
    }


    /// <summary>
    /// Generates SAN move for regular moves (non castle).
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="move"></param>
    /// <returns></returns>
    static string SANRegularMove(Position pos, Move move)
    {
      List<Move> moves = pos.Moves;
      Piece piece = pos.PieceOnSquare(move.FromSquare);
      Piece pieceOnToSquare = pos.PieceOnSquare(move.ToSquare);

      bool isEnPassantCapture = pieceOnToSquare.Type == PieceType.None
                                && piece.Type == PieceType.Pawn
                                && move.FromSquare.File != move.ToSquare.File;

      string part1Piece()
      {
        if (piece.Type == PieceType.Pawn)
        {
          return isEnPassantCapture ? "" + char.ToLower(move.FromSquare.FileChar) : "";
        }
        else
          return ("" + piece.Char).ToUpper(); ;
      }

      string part2From()
      {
        if (piece.Type == PieceType.Pawn)
          return (pieceOnToSquare.Type != PieceType.None) ? (move.FromSquare.FileChar + "").ToLower() : "";
        else
        {
          List<Square> fromSquares = SquaresCanMoveToSquare(pos, moves, piece.Type,
                                               move.FromSquare, move.ToSquare);
          if (fromSquares.Count > 0)
          {
            bool sameRank = move.FromSquare.Rank == fromSquares[0].Rank;
            bool sameFile = move.FromSquare.File == fromSquares[0].File;

            if (sameRank || (!sameRank && !sameFile))
              return ("" + move.FromSquare.FileChar).ToLower();
            else
              return "" + move.FromSquare.RankChar;
          }
          else
            return ""; // no ambiguity
        }

      }


      string part3Capture()
      {
        if (pieceOnToSquare.Type != PieceType.None || isEnPassantCapture)
          return "x";
        else
          return "";
      }

      string part4PTo() => move.ToSquare.ToString().ToLower();

      string part5Promo()
      {
        return move.PromoteTo switch
        {
          PieceType.Queen => "Q",
          PieceType.Rook => "R",
          PieceType.Bishop => "B",
          PieceType.Knight => "N",
          _ => ""
        };
      }


      return part1Piece() + part2From() + part3Capture() + part4PTo() + part5Promo();
    }

  }
}
