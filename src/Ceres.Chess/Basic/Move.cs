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

using Ceres.Chess.Textual;
using System;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Represents a move on a chessboard (or alternately a game termination of a specific type).
  /// 
  /// The minimal information possible is encoded (from/to square and possible promotion type).
  ///
  /// The associated position is not recorded, so it is not possible to know (for example):
  ///   - the side to move,
  ///   - the piece being moved
  ///   
  /// The PositionWithMove class can alternately be used to associate a position.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  [Serializable]
  public readonly struct Move : IEquatable<Move>
  {
    public enum MoveType
    {
      MoveNonCastle,
      MoveCastleShort, MoveCastleLong,
      Draw, WhiteWins, BlackWins
    };

    /// <summary>
    /// The type of move (either move of an actual chess piece, or indication of game termination)
    /// </summary>
    public readonly MoveType Type;

    /// <summary>
    /// Origin square
    /// </summary>
    public readonly Square FromSquare;

    /// <summary>
    /// Target square
    /// </summary>
    public readonly Square ToSquare;

    /// <summary>
    /// If a promotion type, the new piece chosen
    /// </summary>
    public readonly PieceType PromoteTo;

    /// <summary>
    /// If this is an empty (uninitialized) move)
    /// </summary>
    public bool IsNull => this == default;


    /// <summary>
    /// Constructor (for castling or moves indicating game termination)
    /// </summary>
    /// <param name="type"></param>
    public Move(MoveType type)
    {
      if (type == MoveType.MoveNonCastle) throw new ArgumentException("Use alternate constructor for non-castling regular moves");

      Type = type;
      FromSquare = ToSquare = default;
      PromoteTo = default;
    }


    /// <summary>
    /// Constructor (for actual moves)
    /// </summary>
    /// <param name="type"></param>
    /// <param name="startSquare"></param>
    /// <param name="destSquare"></param>
    /// <param name="promoteTo"></param>
    public Move(Square startSquare, Square destSquare, PieceType promoteTo = PieceType.None)
    {
      Type = MoveType.MoveNonCastle;
      FromSquare = startSquare;
      ToSquare = destSquare;
      PromoteTo = promoteTo;
    }

    public bool IsTerminal => Type >= MoveType.Draw;

    string PromoStr => PromoteTo != PieceType.None ? $"{(PromoteTo == PieceType.Knight ? "N" : PromoteTo.ToString()[0])}" : "";

    #region Conversion


    /// <summary>
    /// Converts a move string in UCI format (long algebraic notation) into a Move.
    /// Examples: e2e4, e1g1 (possible castling), a7a8r (promotion to rook)
    /// </summary>
    /// <param name="uciMoveStr"></param>
    /// <returns></returns>
    public static Move FromUCI(string uciMoveStr)
    {
      if (uciMoveStr.Length < 4 || uciMoveStr.Length > 5) throw new Exception("Invalid move " + uciMoveStr);

      Square startSquare = uciMoveStr.Substring(0, 2);
      Square destSquare = uciMoveStr.Substring(2, 2);
      PieceType promoPiece = (uciMoveStr.Length == 5) ? Piece.PieceTypeFromChar(uciMoveStr[4]) : PieceType.None;

      return new Move(startSquare, destSquare, promoPiece);
    }


    /// <summary>
    /// Converts to a Move from a move string in SAN (Standard Algebraic Notation)
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="san"></param>
    /// <returns></returns>
    public static Move FromSAN(in Position pos, string san) => SANParser.FromSAN(san, in pos).Move;

    public string ToSAN(in Position position) => SANGenerator.ToSANMove(in position, this);

    #endregion

    #region Overrides

    // --------------------------------------------------------------------------------------------
    /// <summary>
    /// Returns string representation
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return Type switch
      {
        MoveType.MoveNonCastle => FromSquare.ToString() + ToSquare.ToString() + PromoStr,
        MoveType.MoveCastleShort => "O-O",
        MoveType.MoveCastleLong => "O-O-O",
        MoveType.Draw => "1/2-1/2",
        MoveType.WhiteWins => "1-0",
        MoveType.BlackWins => "0-1",
        _ => throw new Exception("Invalid move type")
      };
    }

    public override bool Equals(object obj) => obj is Move move && Equals(move);


    public bool Equals(Move other)
    {
      return Type == other.Type &&
             FromSquare.Equals(other.FromSquare) &&
             ToSquare.Equals(other.ToSquare) &&
             PromoteTo == other.PromoteTo;
    }

    public override int GetHashCode() => HashCode.Combine(Type, FromSquare, ToSquare, PromoteTo);


    public static bool operator ==(Move left, Move right) => left.Equals(right);


    public static bool operator !=(Move left, Move right) => !(left == right);

#endregion
  }

}
