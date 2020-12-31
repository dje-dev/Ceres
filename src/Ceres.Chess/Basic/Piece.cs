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
using System.Diagnostics;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Represents a chess piece (with associated side). For example, black Knight.
  /// 
  /// Packed internally as a single half-byte (4 bits):
  ///   - 3 bits for the PieceType
  ///   - 1 bit for the SideType
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly struct Piece : IEquatable<Piece>
  {
    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="piece"></param>
    /// <param name="side"></param>
    public Piece(SideType side, PieceType piece) => data = (byte)(((int)side << 3) | (int)piece);


    /// <summary>
    /// Implicit conversion operator from a tuple of (Square, Piece)
    /// </summary>
    /// <param name="sqPiece"></param>
    public static implicit operator Piece((SideType side, PieceType pieceType) piece) => new Piece(piece.side, piece.pieceType);

    /// <summary>
    /// Deconstruction method to tuple of (Square, Piece)
    /// </summary>
    /// <param name="square"></param>
    /// <param name="piece"></param>
    public void Deconstruct(out SideType side, out PieceType pieceType)
    {
      side = Side;
      pieceType = Type;
    }

    internal Piece(byte rawValue)
    {
      Debug.Assert(rawValue < 16);
      data = rawValue;
    }

    public byte RawValue => data;

    public PieceType Type => (PieceType)(data & maskPiece);
    public SideType Side => (SideType)(data >> 3);

    public static PieceType PieceTypeFromChar(char pieceChar)
    {
      return char.ToUpper(pieceChar) switch
      {
        'P' => PieceType.Pawn,
        'N' => PieceType.Knight,
        'B' => PieceType.Bishop,
        'R' => PieceType.Rook,
        'Q' => PieceType.Queen,
        'K' => PieceType.King,
        _ => throw new Exception($"Invalid piece type {pieceChar}")
      };
    }

    #region Internals

    readonly byte data;

    const byte maskPiece = 0b0000_0111;
    const byte maskSide = 0b0000_1000;

    public const char EMPTY_SQUARE_CHAR = '.';

    static char[] charsWhite = { EMPTY_SQUARE_CHAR, 'P', 'N', 'B', 'R', 'Q', 'K' };
    static char[] charsBlack = { EMPTY_SQUARE_CHAR, 'p', 'n', 'b', 'r', 'q', 'k' };

    #endregion

    public string Name => Side.ToString() + "_" + Type.ToString();
    public char Char => (Side == SideType.White ? charsWhite[(byte)Type] : charsBlack[(byte)Type]);

    public override string ToString() => Name;


    public override bool Equals(object obj) => obj is Piece piece && Equals(piece);

    public bool Equals(Piece other) => data == other.data;

    public override int GetHashCode() => data;

    public static bool operator ==(Piece left, Piece right) => left.Equals(right);

    public static bool operator !=(Piece left, Piece right) => !(left == right);
  }

}



