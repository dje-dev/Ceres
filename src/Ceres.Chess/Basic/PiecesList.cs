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


namespace Ceres.Chess
{

  /// <summary>
  /// Defines specific set of pieces, which can be provided as a string such as "KPPkpp".
  /// </summary>
  public record class PieceList
  {
    /// <summary>
    /// Pieces string such as "KRPkr"
    /// </summary>
    public readonly string PiecesStr;

    /// <summary>
    /// Array of corresponding pieces
    /// </summary>
    public readonly Piece[] Pieces;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="pieceListString"></param>
    public PieceList(string pieceListString)
    {
      PiecesStr = pieceListString;
      Pieces = FromPieces(pieceListString);
    }

    /// <summary>
    /// Converts to a predicate over Position that returns true if the position matches the piece list.
    /// </summary>
    public Predicate<Position> ToPredicate => pos => PositionMatches(in pos);


    /// <summary>
    /// Returns if specified position exactly matches the piece list.
    /// </summary>
    /// <param name="position"></param>
    /// <returns></returns>
    public bool PositionMatches(in Position position)
    {
      string pieceMask = PiecesStr;

      if (position.PieceCount != pieceMask.Length)
      {
        return false;
      }

      foreach (PieceOnSquare pieceSquare in position.PiecesEnumeration)
      {
        Piece piece = pieceSquare.Piece;
        string pieceChar = piece.Char.ToString();
        int index = pieceMask.IndexOf(pieceChar);
        if (index == -1)
        {
          return false;
        }

        pieceMask = pieceMask.Remove(index, 1);
      }

      return true;
    }

    #region Internal helpers


    /// <summary>
    /// Returns array of Piece based on string (such as "KRPkrp").
    /// </summary>
    /// <param name="piecesString">string listing pieces such as "KRPkrp"</param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    static Piece[] FromPieces(string piecesString)
    {
      if (!piecesString.Contains("K") || !piecesString.Contains("k"))
      {
        throw new ArgumentException("String must contain white and black kings (K and k).", nameof(piecesString));
      }

      List<Piece> pieces = new(piecesString.Length);
      foreach (char c in piecesString)
      {
        PieceType type;
        SideType side;

        switch (c)
        {
          case 'K':
            type = PieceType.King;
            side = SideType.White;
            break;
          case 'Q':
            type = PieceType.Queen;
            side = SideType.White;
            break;
          case 'R':
            type = PieceType.Rook;
            side = SideType.White;
            break;
          case 'B':
            type = PieceType.Bishop;
            side = SideType.White;
            break;
          case 'N':
            type = PieceType.Knight;
            side = SideType.White;
            break;
          case 'P':
            type = PieceType.Pawn;
            side = SideType.White;
            break;
          case 'k':
            type = PieceType.King;
            side = SideType.Black;
            break;
          case 'q':
            type = PieceType.Queen;
            side = SideType.Black;
            break;
          case 'r':
            type = PieceType.Rook;
            side = SideType.Black;
            break;
          case 'b':
            type = PieceType.Bishop;
            side = SideType.Black;
            break;
          case 'n':
            type = PieceType.Knight;
            side = SideType.Black;
            break;
          case 'p':
            type = PieceType.Pawn;
            side = SideType.Black;
            break;
          default:
            throw new ArgumentException($"Invalid character in composition string: {c}");
        }

        pieces.Add(new Piece(side, type));
      }

      return pieces.ToArray();
    }


    #endregion
  }

}
