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
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Represents the combination of a piece identifier and the square on which it sits.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly struct PieceOnSquare : IEquatable<PieceOnSquare>
  {
    /// <summary>
    /// Square on which piece is placed
    /// </summary>
    public readonly Square Square;

    /// <summary>
    /// Piece on square
    /// </summary>
    public readonly Piece Piece;

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="square"></param>
    /// <param name="piece"></param>
    public PieceOnSquare(Square square, Piece piece)
    {
      Square = square;
      Piece = piece;
    }

    /// <summary>
    /// Implicit conversion operator from a tuple of (Square, Piece)
    /// </summary>
    /// <param name="sqPiece"></param>
    public static implicit operator PieceOnSquare((Square square, Piece piece) sqPiece) => new PieceOnSquare(sqPiece.square, sqPiece.piece);

    public static implicit operator PieceOnSquare((Square square, SideType side, PieceType piece) sqPiece) => new PieceOnSquare(sqPiece.square, (sqPiece.side, sqPiece.piece));

    /// <summary>
    /// Deconstruction method to tuple of (Square, Piece)
    /// </summary>
    /// <param name="square"></param>
    /// <param name="piece"></param>
    public void Deconstruct(out Square square, out Piece piece)
    {
      piece = Piece;
      square = Square;
    }

    #region  Overrides
    public bool Equals(PieceOnSquare other) => Square == other.Square && Piece == other.Piece;

    public static bool operator ==(PieceOnSquare sp1, PieceOnSquare sp2) => sp1.Equals(sp2);
    public static bool operator !=(PieceOnSquare sp1, PieceOnSquare sp2) => !sp1.Equals(sp2);

    public override bool Equals(object obj) => obj is PieceOnSquare square && Equals((PieceOnSquare)obj);

    public override int GetHashCode() => HashCode.Combine(Square.GetHashCode(), Piece.GetHashCode());

    public override string ToString() => $"{Piece } on { Square }";

    #endregion
  }



}
