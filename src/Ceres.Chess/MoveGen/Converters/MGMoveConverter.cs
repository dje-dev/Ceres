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

using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.Textual;
using System;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.MoveGen.Converters
{
  public static class MGMoveConverter
  {
    // --------------------------------------------------------------------------------------------
    /// <summary>
    /// Converts from MGChessMove.
    /// TO DO: shouldn't this be in an Extension class instead?
    /// </summary>
    /// <param name="mgMove"></param>
    /// <returns></returns>
    public static Move ToMove(MGMove mgMove)
    {
      if (mgMove.CastleShort) return new Move(Move.MoveType.MoveCastleShort);
      if (mgMove.CastleLong) return new Move(Move.MoveType.MoveCastleLong);

      PieceType promoPiece = PieceType.None;
      if (mgMove.PromoteQueen)
        promoPiece = PieceType.Queen;
      else if (mgMove.PromoteRook)
        promoPiece = PieceType.Rook;
      else if (mgMove.PromoteBishop)
        promoPiece = PieceType.Bishop;
      else if (mgMove.PromoteKnight)
        promoPiece = PieceType.Knight;

      Square fromSquare = new Square((int)mgMove.FromSquareIndex, Square.SquareIndexType.BottomToTopRightToLeft);
      Square toSquare = new Square((int)mgMove.ToSquareIndex, Square.SquareIndexType.BottomToTopRightToLeft);

      return new Move(fromSquare, toSquare, promoPiece);
    }


    /// <summary>
    /// Returns the MGMove corresponding to a given Move.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="move"></param>
    /// <returns></returns>
    public static MGMove MGMoveFromPosAndMove(in Position pos, Move move)
    {
      PositionWithMove moveAndPos = new PositionWithMove(pos, move);
      MGPosition mgPos = MGPosition.FromPosition(in moveAndPos.Position);

      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in mgPos, moves);
      for (int i=0; i<moves.NumMovesUsed;i++)
      {
        if (moves.MovesArray[i].EqualsMove(move))
          return moves.MovesArray[i];
      }
      throw new Exception("Move not found");
    }

    public static MGMove ToMGMove(in Position position, EncodedMove encodedMove) => ToMGMove(MGPosition.FromPosition(in position), encodedMove);

    public static MGMove ToMGMove(in MGPosition mgPos, EncodedMove encodedMove)
    {
      MGMoveList movesLegal = new MGMoveList();
      MGMoveGen.GenerateMoves(in mgPos, movesLegal);

      int indexLegalMove = MoveInMGMovesArrayLocator.FindMoveInMGMoves(in mgPos, movesLegal.MovesArray, encodedMove, 0, movesLegal.NumMovesUsed, mgPos.BlackToMove);
      if (indexLegalMove == -1) throw new Exception($"Move not found {encodedMove}");
      return movesLegal.MovesArray[indexLegalMove];
      //  Move move = MGMoveConverter.ToMove(theMove);
    }

  }
}