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


#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Static members for each type of Piece
  /// </summary>
  public static class Pieces
  {
    public static readonly Piece None = new Piece(SideType.White, PieceType.None);

    public static readonly Piece WhitePawn = new Piece(SideType.White, PieceType.Pawn);
    public static readonly Piece WhiteKnight = new Piece(SideType.White, PieceType.Knight);
    public static readonly Piece WhiteBishop = new Piece(SideType.White, PieceType.Bishop);
    public static readonly Piece WhiteRook = new Piece(SideType.White, PieceType.Rook);
    public static readonly Piece WhiteQueen = new Piece(SideType.White, PieceType.Queen);
    public static readonly Piece WhiteKing = new Piece(SideType.White, PieceType.King);

    public static readonly Piece BlackPawn = new Piece(SideType.Black, PieceType.Pawn);
    public static readonly Piece BlackKnight = new Piece(SideType.Black, PieceType.Knight);
    public static readonly Piece BlackBishop = new Piece(SideType.Black, PieceType.Bishop);
    public static readonly Piece BlackRook = new Piece(SideType.Black, PieceType.Rook);
    public static readonly Piece BlackQueen = new Piece(SideType.Black, PieceType.Queen);
    public static readonly Piece BlackKing = new Piece(SideType.Black, PieceType.King);
  }

}



