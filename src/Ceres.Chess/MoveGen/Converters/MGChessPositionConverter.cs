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
using Ceres.Chess.Textual;

#endregion

#region License
/*

MIT License

Copyright(c) 2016-2017 Judd Niemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#endregion

namespace Ceres.Chess.MoveGen.Converters
{
  /// <summary>
  /// Static class which can convert between MGChessPosition and Position representations.
  /// </summary>
  public static class MGChessPositionConverter
  {
    static readonly int[] PieceMapWhite = { 0, 1, 5, 2, 4, 6, 7 };
    static readonly int[] PieceMapBlack = { 0, 9, 13, 10, 12, 14, 15 };

    const int EnPassantWhite = 3;
    const int EnPassantBlack = 11;


    /// <summary>
    /// Converts Position into corresponding MGPosition.
    /// </summary>
    /// <param name="position"></param>
    /// <returns></returns>
    internal static MGPosition MCChessPositionFromPosition(in Position position)
    {
      // TO DO: someday a custom low-level converter directly between the two board representations
      //        could be written to improve performance (currently about 400,000 per second)
      MGPosition pos = default;

      Span<(Piece, Square)> arrayPieces = stackalloc (Piece, Square)[position.PieceCount];
      arrayPieces = position.GetPiecesOnSquares(arrayPieces);

      foreach ((Piece, Square) ps in arrayPieces)
      {
        pos.SetPieceAtBitboardSquare((ulong)MGPieceFromPiece(ps.Item1), MGPosition.MGBitBoardFromSquare(ps.Item2));
      }
      pos.Rule50Count = position.MiscInfo.Move50Count;
      pos.MoveNumber = position.MiscInfo.MoveNum;
      pos.BlackToMove = position.MiscInfo.SideToMove == SideType.Black;
      pos.WhiteCanCastle = position.MiscInfo.WhiteCanOO;
      pos.BlackCanCastle = position.MiscInfo.BlackCanOO;
      pos.WhiteCanCastleLong = position.MiscInfo.WhiteCanOOO;
      pos.BlackCanCastleLong = position.MiscInfo.BlackCanOOO;

      if (position.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
      {
        bool whiteToMove = position.MiscInfo.SideToMove == SideType.White;
        int rank = whiteToMove ? 5 : 2;
        int file = (int)position.MiscInfo.EnPassantFileIndex;
        Square square = Square.FromFileAndRank(file, rank);
        pos.SetPieceAtBitboardSquare((ulong)(whiteToMove ? EnPassantBlack : EnPassantWhite), MGPosition.MGBitBoardFromSquare(square));

      }

#if MG_USE_HASH
      pos.CalculateHash();
#endif

      return pos;
    }


    // Returns piece at specified square index.
    internal static Piece PieceAt(in MGPosition mgPos, int i)
    {
      int pieceCode = mgPos.GetPieceAtBitboardSquare(1UL << i);
      if (pieceCode == 0) return default;

      // Get side
      SideType side = pieceCode >= 9 ? SideType.Black : SideType.White;

      if (pieceCode == MGPositionConstants.BENPASSANT || pieceCode == MGPositionConstants.WENPASSANT)
      {
        return default;
      }
      else
      {
        // Get piece type
        int pieceIndex = side == SideType.Black ? Array.IndexOf(PieceMapBlack, pieceCode)
                                                : Array.IndexOf(PieceMapWhite, pieceCode);
        return new Piece(side, (PieceType)pieceIndex);
      }
    }

    const int MAX_PIECES = 33; // 32 actual pieces, and one possible en passant

    static readonly int[] pieceIndexMap = new int[] { 0, (int)PieceType.Pawn, (int)PieceType.Bishop, -1, (int)PieceType.Rook, (int)PieceType.Knight, (int)PieceType.Queen, (int)PieceType.King,
                                                      0, (int)PieceType.Pawn, (int)PieceType.Bishop, -1, (int)PieceType.Rook, (int)PieceType.Knight, (int)PieceType.Queen, (int)PieceType.King, };


    /// <summary>
    /// Converts from MGPosition to Position.
    /// Note that [SkipLocalsInitAttribute] only makes 2% faster.
    /// </summary>
    /// <param name="mgPos"></param>
    /// <returns></returns>
    [SkipLocalsInit]
    public static Position PositionFromMGChessPosition(in MGPosition mgPos)
    {
      Span<PieceOnSquare> piecesArrayBufferLocal = stackalloc PieceOnSquare[MAX_PIECES];

      int pieceCount = 0;

      bool whiteCanCastleOO = mgPos.WhiteCanCastle;
      bool whiteCanCastleOOO = mgPos.WhiteCanCastleLong;
      bool blackCanCastleOO = mgPos.BlackCanCastle;
      bool blackCanCastleOOO = mgPos.BlackCanCastleLong;

      SideType sideToMove = mgPos.BlackToMove ? SideType.Black : SideType.White;
      int move50Count = mgPos.Rule50Count;
      int repetitionCount = 0; // unknowable
      int moveNum = mgPos.MoveNumber;
      PositionMiscInfo.EnPassantFileIndexEnum enPassantColIndex = PositionMiscInfo.EnPassantFileIndexEnum.FileNone;

      ulong occupied = mgPos.A | mgPos.B | mgPos.C; // all squares occupied by something
      byte thisSquare;
      do
      {
        thisSquare = (byte)System.Numerics.BitOperations.TrailingZeroCount(occupied);
        if (thisSquare < 64)
        {
          ulong squareMask = 1UL << (int)thisSquare;
          occupied ^= squareMask;

          int pieceCode = mgPos.GetPieceAtBitboardSquare(squareMask);
          Debug.Assert(pieceCode != 0);

          // Get square
          int rank = thisSquare / 8;
          int file = (thisSquare % 8) ^ 0b111; // equivalent to (7 - thisSquare % 8) but faster

          Square square = Square.FromFileAndRank(file, rank);

          // Get side
          SideType side = pieceCode >= 9 ? SideType.Black : SideType.White;

          if (pieceCode == MGPositionConstants.BENPASSANT || pieceCode == MGPositionConstants.WENPASSANT)
          {
            enPassantColIndex = (PositionMiscInfo.EnPassantFileIndexEnum)file;
          }
          else
          {
            // Get piece type
            int pieceIndex = pieceIndexMap[pieceCode];
            piecesArrayBufferLocal[pieceCount++] = new PieceOnSquare(square, new Piece(side, (PieceType)pieceIndex));
          }
        }
        else
          break;

      } while (true);

      // Mark end of populated array if we did not fill it up
      if (pieceCount < MAX_PIECES) piecesArrayBufferLocal[pieceCount] = default;

      PositionMiscInfo miscInfo = new PositionMiscInfo(whiteCanCastleOO, whiteCanCastleOOO,
                              blackCanCastleOO, blackCanCastleOOO,
                              sideToMove, move50Count, repetitionCount,
                              moveNum, enPassantColIndex);

      return new Position(piecesArrayBufferLocal, in miscInfo);
    }


    /// <summary>
    /// Converts FEN directly into MGPosition.
    /// </summary>
    /// <param name="fen"></param>
    /// <returns></returns>
    public static MGPosition MGChessPositionFromFEN(string fen)
    {
      FENParseResult fenParsed = FENParser.ParseFEN(fen);

      MGPosition pos = default;
      foreach (PieceOnSquare ps in fenParsed.Pieces)
        pos.SetPieceAtBitboardSquare((ulong)MGPieceFromPiece(ps.Piece), MGPosition.MGBitBoardFromSquare(ps.Square));
      pos.Rule50Count = fenParsed.MiscInfo.Move50Count;
      pos.MoveNumber = fenParsed.MiscInfo.MoveNum;
      pos.BlackToMove = fenParsed.MiscInfo.SideToMove == SideType.Black;
      pos.WhiteCanCastle = fenParsed.MiscInfo.WhiteCanOO;
      pos.BlackCanCastle = fenParsed.MiscInfo.BlackCanOO;
      pos.WhiteCanCastleLong = fenParsed.MiscInfo.WhiteCanOOO;
      pos.BlackCanCastleLong = fenParsed.MiscInfo.BlackCanOOO;

      if (fenParsed.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
      {
        bool whiteToMove = fenParsed.MiscInfo.SideToMove == SideType.White;
        int rank = whiteToMove ? 5 : 2;
        int file = (int)fenParsed.MiscInfo.EnPassantFileIndex;
        Square square = Square.FromFileAndRank(file, rank);
        pos.SetPieceAtBitboardSquare((ulong)(whiteToMove ? EnPassantBlack : EnPassantWhite), MGPosition.MGBitBoardFromSquare(square));

      }

#if MG_USE_HASH
      pos.CalculateHash();
#endif

      return pos;
    }

    /// <summary>
    /// Static helper to convert between Piece and MG piece.
    /// </summary>
    /// <param name="piece"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int MGPieceFromPiece(Piece piece)
      => piece.Side == SideType.White
                     ? PieceMapWhite[(int)piece.Type]
                     : PieceMapBlack[(int)piece.Type];

  }
}


