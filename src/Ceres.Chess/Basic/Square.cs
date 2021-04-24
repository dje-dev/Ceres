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
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Square on chess board.
  /// 
  /// Represented as a single byte representing board index from left to right and bottom to top (A1 is 0)
  /// </summary>
  public readonly struct Square : IEquatable<Square>
  {
    /// <summary>
    /// Square index (using convention that A1 is index 0).
    /// </summary>
    public readonly byte SquareIndexStartA1;

    /// <summary>
    /// Returns index using convention that H1 is index 0.
    /// </summary>
    public byte SquareIndexStartH1 => (byte)(Rank * 8 + (7 - File));


    /// <summary>
    /// Constructor (from specified file and rank indices).
    /// </summary>
    /// <param name="file"></param>
    /// <param name="rank"></param>
    /// <returns></returns>
    public static Square FromFileAndRank(int file, int rank)
    {
      Debug.Assert(file >= 0 && file < 8);
      Debug.Assert(rank >= 0 && rank < 8);

      return new Square((byte)(rank * 8 + file));
    }


    /// <summary>
    /// Constructor (from a string representation such as "A3").
    /// </summary>
    /// <param name="squareStr"></param>
    public Square(string squareStr)
    {
      Debug.Assert(squareStr.Length == 2);
      squareStr = squareStr.ToUpperInvariant();

      SquareIndexStartA1 = (byte)(FileCharToFileIndex(squareStr[0]) + 8 * RankCharToRankIndex(squareStr[1]));
    }


    /// <summary>
    /// Constructor from a board square index, running from 0 (A1) to 63 (H8)
    /// </summary>
    /// <param name="index"></param>
    public Square(int index, SquareIndexType indexType = SquareIndexType.BottomToTopLeftToRight)
    {
      Debug.Assert(index >= 0 && index < 64);

      if (indexType == SquareIndexType.BottomToTopRightToLeft)
        index = (byte)((index / 8) * 8 + (7 - index % 8));
      SquareIndexStartA1 = (byte)index;
    }

    /// <summary>
    /// Method of indexing squares.
    /// </summary>
    public enum SquareIndexType { BottomToTopLeftToRight, BottomToTopRightToLeft };


    /// <summary>
    /// Returns index of rank of square.
    /// </summary>
    public int Rank => SquareIndexStartA1 / 8;

    /// <summary>
    /// Returns index of file of square.
    /// </summary>
    public int File => SquareIndexStartA1 % 8;

    /// <summary>
    /// Returns file of square as a character.
    /// </summary>
    public char FileChar => ToString()[0];

    /// <summary>
    /// Returns rank of square as a character.
    /// </summary>
    public char RankChar => ToString()[1];

    /// <summary>
    /// Implicit conversion operator form SquareNames to Square.
    /// </summary>
    /// <param name="square"></param>
    public static implicit operator Square(SquareNames square) => new Square((byte)square);


    /// <summary>
    /// Implicit conversion from board string (file and rank, such as "B3")
    /// </summary>
    /// <param name="squareStr"></param>
    public static implicit operator Square(string squareStr)
    {
      return new Square(squareStr);
    }

    /// <summary>
    /// Returns position mirrored about the vertical divide of the board.
    /// </summary>
    /// <param name="sq"></param>
    /// <returns></returns>
    public Square Mirrored => Square.FromFileAndRank(7 - File, Rank);


    /// <summary>
    /// Helper method to convert a character representing a rank into corresponding index.
    /// </summary>
    /// <param name="rankChar"></param>
    /// <returns></returns>
    static int RankCharToRankIndex(char rankChar)
    {
      rankChar = Char.ToUpper(rankChar);
      if (rankChar < '1' || rankChar > '8') throw new ArgumentException($"Invalid rank character {rankChar}");
      return (int)rankChar - (int)'1';
    }


    /// <summary>
    /// Helper method to convert a character representing a file into corresponding index.
    /// </summary>
    /// <param name="fileChar"></param>
    /// <returns></returns>
    static int FileCharToFileIndex(char fileChar)
    {
      if (fileChar < 'A' || fileChar > 'H') throw new ArgumentException($"Invalid file character {fileChar}");
      return (int)fileChar - (int)'A';
    }


    #region Square names

    static string[] squareNames;

    /// <summary>
    /// Helper method to initialize the array of 
    /// square names over all squares on board.
    /// </summary>
    /// <returns></returns>
    static string[] GenerateSquareNames()
    {
      squareNames = new string[64];
      for (int i = 0; i < 64; i++)
        squareNames[i] = ((SquareNames)i).ToString();
      return squareNames;
    }

    /// <summary>
    /// Returns string representation of square.
    /// </summary>
    /// <returns></returns>
    public override string ToString() => squareNames[SquareIndexStartA1];

    #endregion

    static List<Square> allSquares = null;

    /// <summary>
    /// Returns list of all squares on board.
    /// </summary>
    public static List<Square> AllSquares
    {
      get
      {
        if (allSquares == null)
        {
          allSquares = new List<Square>(64);
          for (int i = 0; i < 64; i++)
            allSquares.Add(new Square(((SquareNames)i).ToString()));
        }
        return allSquares;
      }
    }


    #region Overrides

    public bool IsLightSquare => this.SquareIndexStartA1 % 2 == 1;

    public static bool operator ==(Square left, Square right) => left.Equals(right);

    public static bool operator !=(Square left, Square right) => !(left == right);

    public override bool Equals(object obj) => obj is Square square && Equals(square);


    public bool Equals(Square other) => SquareIndexStartA1 == other.SquareIndexStartA1;

    public override int GetHashCode() => SquareIndexStartA1;

    #endregion

    #region Initialization

    [ModuleInitializer]
    internal static void Init()
    {
      GenerateSquareNames();
    }

    #endregion
  }

}
