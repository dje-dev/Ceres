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

using Ceres.Chess.MoveGen.Converters;
using System.Collections.Generic;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Static helper for converting MGMove to a string representation (algebraic).
  /// </summary>
  public static class MGMoveToString
  {
    public static string AlgebraicMoveString(MGMove mgMove, Position pos)
    {
      string ret = null;

      bool white = pos.MiscInfo.SideToMove == SideType.White;

      MGPosition curPosition = MGChessPositionConverter.MGChessPositionFromFEN(pos.FEN);

      // Generate moves
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in curPosition, moves);

      //if (white) mgMove = new MGMove(mgMove, true);
      MGPosition newPosition = MGChessPositionConverter.MGChessPositionFromFEN(pos.FEN);
      newPosition.MakeMove(mgMove);

      bool isCheck = MGMoveGen.IsInCheck(newPosition, white);

      if (mgMove.CastleShort)
        ret = "O-O";
      else if (mgMove.CastleLong)
        ret = "O-O-O";

      else
      {
        Square fromSquare = mgMove.FromSquare;
        Square toSquare = mgMove.ToSquare;

        Piece fromType = pos.PieceOnSquare(fromSquare);
        Piece toPiece = pos.PieceOnSquare(toSquare);

        if (fromType.Type == PieceType.Pawn)
        {
          string promoteChar = "";
          if (mgMove.PromoteQueen) promoteChar = "Q";
          if (mgMove.PromoteBishop) promoteChar = "B";
          if (mgMove.PromoteRook) promoteChar = "R";
          if (mgMove.PromoteKnight) promoteChar = "N";

          if (mgMove.EnPassantCapture)
          {
            int newRank = white ? 6 : 3;
            char newFile = char.ToLower(toSquare.FileChar);
            ret = fromSquare.ToString().Substring(0, 1).ToLower() + "x" +
                  newFile + newRank;
          }
          else if (toPiece.Type == PieceType.None)
            ret = toSquare.ToString().ToLower() + promoteChar;
          else
            ret = fromSquare.ToString().Substring(0, 1).ToLower() + "x" +
                  toSquare.ToString().ToLower() + promoteChar;
        }
        else
        {
          string captureChar = toPiece.Type == PieceType.None ? "" : "x";

          List<MGMove> matchingCaptures = MovesByPieceTypeThatTakeOnSquare(mgMove.Piece, mgMove.ToSquareIndex, moves);
          if (matchingCaptures.Count == 1)
            ret = char.ToUpper(fromType.Char) + captureChar + toSquare.ToString().ToLower();
          else
          {
            // Disambiguate
            DifferBy differBy = MoveDifferFromAllOthersBy(matchingCaptures, mgMove);
            string fileChar = fromSquare.FileChar.ToString().ToLower();
            string rankChar = fromSquare.RankChar.ToString().ToLower();

            if (differBy == DifferBy.File)
              ret = char.ToUpper(fromType.Char) + fileChar + captureChar + toSquare.ToString().ToLower();
            else if (differBy == DifferBy.Rank)
              ret = char.ToUpper(fromType.Char) + rankChar + captureChar + toSquare.ToString().ToLower();
            else
              ret = char.ToUpper(fromType.Char) + fileChar + rankChar + captureChar + toSquare.ToString().ToLower();
          }

        }
      }

      if (isCheck)
        return ret + "+";
      else
        return ret;
    }


    static List<MGMove> MovesByPieceTypeThatTakeOnSquare(MGPositionConstants.MCChessPositionPieceEnum piece, byte toSquare, MGMoveList moves)
    {
      List<MGMove> matchingMoves = new List<MGMove>(0);
      foreach (MGMove move in moves.MovesArray)
        if (move.Piece == piece && move.ToSquareIndex == toSquare)
          matchingMoves.Add(move);
      return matchingMoves;
    }
    enum DifferBy { Rank, File, RankFileComo };


    static DifferBy MoveDifferFromAllOthersBy(List<MGMove> moves, MGMove thisMove)
    {
      bool differByRank = true;
      bool differByFile = true;

      foreach (MGMove move in moves)
      {
        if (move != thisMove)
        {
          if (move.FromSquare.File == thisMove.FromSquare.File)
            differByFile = false;

          if (move.FromSquare.Rank == thisMove.FromSquare.Rank)
            differByRank = false;
        }
      }

      if (differByFile)
        return DifferBy.File; // takes precedence
      else if (differByRank)
        return DifferBy.Rank;
      else
        return DifferBy.RankFileComo;
    }
#if NOT
    static void TestZZ(string fen)
    {
      MGPosition curPosition = MGChessPositionConverter.MGChessPositionFromFEN(fen);

      // Generate moves
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in curPosition, moves);
      foreach (var move in moves)
          Console.WriteLine(move + " " +  MoveStr(curPosition.AsPosition, move));
    }

    static void TestZZ()
    {
      // See: https://en.wikipedia.org/wiki/Algebraic_notation_(chess)
      TestZZ("3r3r/b5k1/3b4/R7/4Q2Q/8/8/R1K4Q b - - 0 31"); // Bdb8 Rdf8 
      // Bb8 would be ambiguous, as either of the bishops on a7 and d6 could legally move to b8. The move of the d6 bishop is therefore specified as Bdb8, 
      // indicating that it was the bishop on the d file which moved. 
      // Although they could also be differentiated by their ranks, the file letter takes precedence.
      // For the black rooks both on the 8th rank, both could potentially move to f8, so the move of the d8 rook to f8 is disambiguated as Rdf8.
      Console.WriteLine();
      TestZZ("3r3r/b5k1/3b4/R7/4Q2Q/8/8/R1K4Q w - - 0 31"); // R1a3 Qh4e1
      // For the white rooks both on the a file which could both move to a3, it is necessary to provide the rank of the moving piece, i.e., R1a3.

      // In the case of the white queen on h4 moving to e1, neither the rank nor file alone are sufficient to disambiguate from the other white queens.
      //As such, this move is written Qh4e1. 
    }
#endif

  }


}
