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

#endregion

namespace Ceres.Chess.EncodedPositions.Basic
{
  /// <summary>
  /// Square as represented by Leela Zero:
  /// a single byte representing board index from left to right and bottom to top (A1 is 0)
  /// 
  /// TODO: This is same representation as our ChessSquare, we could eliminate this class (?).
  /// </summary>
  [Serializable]
  public struct EncodedSquare : IEquatable<EncodedSquare>
  {
    public readonly byte Value;

    /// <summary>
    /// Constructor (from specified index).
    /// </summary>
    /// <param name="value"></param>
    public EncodedSquare(int value)
    {
      Debug.Assert(value >= 0 && value <= 255);
      Value = (byte)value;
    }

    /// <summary>
    /// Constructor (from specified rank and file).
    /// </summary>
    /// <param name="rank"></param>
    /// <param name="file"></param>
    public EncodedSquare(int rank, int file)
    {
      Debug.Assert(rank >= 0 && rank < 8 && file >= 0 && file < 8);

      Value = (byte)((rank * 8) + file);
    }

    internal bool IsRank8 => Value >= 56;
    internal bool IsRank2 => Value >= 8 && Value < 16;
    internal bool IsRank4 => Value >= 24 && Value < 32;
    public int Rank => Value / 8;
    public int File => Value % 8; // ?/

    public int Index => Rank * 8 + File;

    public EncodedSquare Mirrored => new EncodedSquare(8 * Rank + 7 - File);
    public EncodedSquare Flipped => new EncodedSquare((byte)(Value ^ 0b111000));

    private const string Rows = "12345678";
    private const string Columns = "abcdefgh";

    Lazy<string[]> SquareNames => new Lazy<string[]>(() => GenerateSquareNames());


    static string[] GenerateSquareNames()
    {
      string[] squareStrs = new string[64];
      for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
          squareStrs[r * 8 + c] = new String(new char[] { Columns[c], Rows[r] });
      return squareStrs;
    }


    /// <summary>
    /// Returns raw value as a byte.
    /// </summary>
    public byte AsByte => Value;


    /// <summary>
    /// Constructor from string coordinate reprsentation.
    /// </summary>
    /// <param name="squareStr"></param>
    /// <param name="flipped"></param>
    public EncodedSquare(string squareStr, bool flipped = false)
    {
      if (squareStr.Length != 2) throw new Exception("Invalid square " + squareStr);

      int col = Columns.IndexOf(char.ToLower(squareStr[0]));
      if (col == -1) throw new Exception("Invalid col " + squareStr[0]);

      int row = Rows.IndexOf(char.ToLower(squareStr[1]));
      if (row == -1) throw new Exception("Invalid row " + squareStr[1]);

      int raw = (row * 8 + col);
      Value = (byte)( flipped ? (64 - raw) : raw);
    }


    #region Overrides and operators

    // --------------------------------------------------------------------------------------------
    public override string ToString()
    {
      return SquareNames.Value[Value];
    }

    // --------------------------------------------------------------------------------------------
    public static implicit operator EncodedSquare(string str) => new EncodedSquare(str);

    public static bool operator ==(EncodedSquare left, EncodedSquare right) =>  left.Equals(right);
    

    public static bool operator !=(EncodedSquare left, EncodedSquare right) => !(left == right);

    public override bool Equals(object obj) => obj is EncodedSquare square && Equals(square);

    public bool Equals(EncodedSquare other) => Value == other.Value;

    public override int GetHashCode() => Value;

    #endregion
  }

}
