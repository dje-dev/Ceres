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
using System.Diagnostics;

using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

using static Ceres.Chess.PieceType;
using static Ceres.Chess.SideType;

#endregion

namespace Ceres.Chess.Textual
{
  /// <summary>
  /// Manages parsing of chess moves expressed in SAN (standard algebraic notation).
  /// 
  /// See: http://cfajohnson.com/chess/SAN/.
  /// </summary>
  public static class SANParser
  {
    /// <summary>
    /// 
    /// TO DO: look back only one or two files
    /// </summary>
    /// <param name="square"></param>
    /// <param name="pos"></param>
    /// <returns></returns>
    static (bool, Square) PawnOnFileBeforeSquare(Square square, Position pos)
    {
      int increment = pos.MiscInfo.SideToMove == SideType.White ? -1 : 1;

      Square nextSquare = square;
      do
      {
        int nextRank = nextSquare.Rank + increment;
        if (nextRank <= 0 || nextRank >= 7) return (false, default);
        nextSquare = Square.FromFileAndRank(square.File, nextRank);
        Piece pieceOnSquare = pos.PieceOnSquare(nextSquare);
        if (pieceOnSquare.Type == PieceType.Pawn && pieceOnSquare.Side == pos.MiscInfo.SideToMove)
          return (true, nextSquare);

      } while (true);

      return (false, default);
    }

    static (bool, Square) SquareWithPieceThatCanMoveToSquare(Piece piece, Square targetSquare, Position pos, bool withCapture, string originRankOrFileOrSquare)
    {
      MGMoveList moves = new MGMoveList(); // TO DO: move to our move list representation
      MGMoveGen.GenerateMoves(MGPosition.FromPosition(in pos), moves);

      for (int i = 0; i < moves.NumMovesUsed; i++)
      {
        MGMove move = moves.MovesArray[i];
        Square thisFromSquare = new Square((int)move.FromSquareIndex, Square.SquareIndexType.BottomToTopRightToLeft);
        Square thisTargetSquare = new Square((int)move.ToSquareIndex, Square.SquareIndexType.BottomToTopRightToLeft); // TO DO: make a converter

        bool originRankOrFileOk = originRankOrFileOrSquare == default || thisFromSquare.ToString().Contains(originRankOrFileOrSquare);
        if (thisTargetSquare == targetSquare && pos.PieceOnSquare(thisFromSquare) == piece && originRankOrFileOk)
        {
          return (true, thisFromSquare);
        }
      }

      return (false, default);
    }


    static bool ParsePieceThenSquare(char pieceChar, string targetSquareStr, Position pos, string originRankOrFileOrSquare, bool withCapture,
                                     PieceType promotionType, out Move move)
    {
      move = default;

      if (originRankOrFileOrSquare != null) originRankOrFileOrSquare = originRankOrFileOrSquare.ToUpperInvariant();

      Piece sourcePiece;
      sourcePiece = new Piece(pos.MiscInfo.SideToMove, Piece.PieceTypeFromChar(pieceChar));

      Square targetSquare = targetSquareStr;
      (bool sourceSquareOk, Square sourceSquare) = SquareWithPieceThatCanMoveToSquare(sourcePiece, targetSquare, pos, withCapture, originRankOrFileOrSquare);
      if (sourceSquareOk)
      {
        string thisSourceSquare = sourceSquare.ToString();
        if (originRankOrFileOrSquare == null || thisSourceSquare.Contains(originRankOrFileOrSquare))
        {
          move = new Move(sourceSquare, targetSquare, promotionType);
          return true;
        }
      }

      return false;
    }


    static Move ParseNormalSANMove(string str, Position pos)
    {
      string orgStr = str;

      // Remove spaces
      str = str.Replace(" ", null) // TODO:  can't we use String.Extensions.ReplaceIfExists?
               .Replace("+", null)
               .Replace("#", null)
               .Replace("=", null)
               .Replace("!", null)
               .Replace("?", null)
               .Replace("e.p.", null);

      char lastChar = str[^1];
      PieceType promoPiece = lastChar switch
      {
        'Q' => PieceType.Queen,
        'R' => PieceType.Rook,
        'B' => PieceType.Bishop,
        'N' => PieceType.Knight,
        _ => PieceType.None
      };
      if (promoPiece != PieceType.None) str = str[0..^1];

      if (str.Length == 2) // e.g. e4
      {
        var (found, square) = PawnOnFileBeforeSquare(str, pos);
        if (!found) throw new ArgumentException($"No eligible pawn found in SAN {str}");
        return new Move(square, str, promoPiece);
      }
      else if (str.Length == 3) // e.g. Nf3
      {
        if (ParsePieceThenSquare(str[0], str.Substring(1), pos, null, false, promoPiece, out Move retMove)) return retMove;
      }
      else if (str.Length == 4 && str[1] == 'x') // e.g. Qxf6 or exc3
      {
        if (char.IsUpper(str[0])) // Qxf3
        {
          if (ParsePieceThenSquare(str[0], str.Substring(2), pos, null, true, promoPiece, out Move retMove))
            return retMove;
        }
        else // e.g. exc3
        {
          if (ParsePieceThenSquare('P', str.Substring(2), pos, str.Substring(0, 1), true, promoPiece, out Move retMove))
            return retMove;
        }
      }
      else if (str.Length == 4 && str[1] != 'x') // e.g. Nbc5
      {
        if (ParsePieceThenSquare(str[0], str.Substring(2), pos, str.Substring(1, 1), true, promoPiece, out Move retMove)) return retMove;
      }
      else if (str.Length == 5 && str[2] == 'x') // e.g. Ncxd5
      {
        if (ParsePieceThenSquare(str[0], str.Substring(3), pos, str.Substring(1, 1), true, promoPiece, out Move retMove)) return retMove;
      }
      else if (str.Length == 5 && str[2] != 'x') // e.g.Bg7f8
      {
        if (ParsePieceThenSquare(str[0], str.Substring(3), pos, str.Substring(1, 2), false, promoPiece, out Move retMove)) return retMove;
      }

      throw new Exception($"Invalid SAN move {orgStr}");

    }


    /// <summary>
    /// Returns the MoveAndPosition from a parsed SAN string and starting position.
    /// </summary>
    /// <param name="sanString"></param>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static PositionWithMove FromSAN(string sanString, in Position pos)
    {
      // strip off possible "++" or "+"
      int indexFirstPlus = sanString.IndexOf("+", StringComparison.Ordinal);
      if (indexFirstPlus != -1) sanString = sanString.Substring(0, indexFirstPlus);

      // strip off possible "#"
      int indexFirstHash = sanString.IndexOf("#", StringComparison.Ordinal);
      if (indexFirstHash != -1) sanString = sanString.Substring(0, indexFirstHash);

      switch (sanString)
      {
        case null: throw new ArgumentNullException();
        case "O-O": return new PositionWithMove(pos, new Move(Move.MoveType.MoveCastleShort));
        case "O-O-O": return new PositionWithMove(pos, new Move(Move.MoveType.MoveCastleLong));
        case "1-0": return new PositionWithMove(pos, new Move(Move.MoveType.WhiteWins));
        case "0-1": return new PositionWithMove(pos, new Move(Move.MoveType.BlackWins));
        case "1/2-1/2": return new PositionWithMove(pos, new Move(Move.MoveType.Draw));
        default: return new PositionWithMove(pos, ParseNormalSANMove(sanString, pos));
      }
    }

  }
}
