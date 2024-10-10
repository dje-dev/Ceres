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
using System.Linq;
using System.Runtime.ConstrainedExecution;
using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using static Ceres.Chess.PieceType;
using static Ceres.Chess.SideType;

#endregion

namespace Ceres.Chess.Textual
{
  /// <summary>
  /// Parser which converts chess positions represented 
  /// as FEN string into FENParseResult
  /// </summary>
  public static partial class FENParser
  {
    /// <summary>
    /// FEN string corresponding to the starting position in chess.
    /// </summary>
    public const string StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /// <summary>
    /// Static factory method to create a FENParseResult from a FEN string.
    /// 
    /// NOTE: performance could be improved by passing in the Piece[] preallocated
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="repetitionCount"></param>
    /// <returns></returns>
    public static FENParseResult ParseFEN(string fen, int repetitionCount = 0)
    {
      try
      {
        return DoParseFEN(fen, repetitionCount);
      }
      catch (Exception exc)
      {
        throw new Exception($"Unable to parse the FEN: {fen}");
      }
    }

    /// <summary>
    /// Get the file index from a character.
    /// </summary>
    /// <param name="c"></param>
    /// <param name="white"></param>
    /// <returns></returns>
    public static int CharToNumber(char c, bool white)
    {
      char lowerCase = char.ToLower(c);
      char baseChar = 'a';
      int file = 7 - (lowerCase - baseChar);

      if (white)
      {
        return file;
      }
      else
      {
        return 56 + file;
      }
    }

    //create a char array from h..a as a helper for parsing Chess960 fens
    private static readonly char[] FileArrayLower = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'];

    //create a char array from H..A as a helper for parsing Chess960 fens
    private static readonly char[] FileArrayUpper = ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'];

    /// <summary>
    /// Worker method to do FEN parsing.
    /// NOTE: performance could be improved by passing in the Piece[] preallocated
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="repetitionCount"></param>
    /// <returns></returns>
    static FENParseResult DoParseFEN(string fen, int repetitionCount = 0)
    {
      int charIndex = 0;

      void SkipAnySpaces() { while (charIndex < fen.Length && char.IsWhiteSpace(fen[charIndex])) charIndex++; }
      void SkipAnySpacesOrDash() { while (charIndex < fen.Length && (fen[charIndex] == '-' || char.IsWhiteSpace(fen[charIndex]))) charIndex++; }

      List<PieceOnSquare> pieces = new List<PieceOnSquare>(32);

      int curRank = 0;
      int curFile = 0;
      int wKRsquare = -1;
      int wQRsquare = -1;
      int bKRsquare = -1;
      int bQRsquare = -1;

      // Parse pieces
      while (true)
      {
        char thisChar = fen[charIndex++];

        if (thisChar == ' ') break;

        if (thisChar == '/')
        {
          if (curRank >= 7) throw new Exception("Illegal FEN, too many squares");
          curRank++;
          curFile = 0;
        }
        else if (char.IsDigit(thisChar))
        {
          curFile += (thisChar - '0');
          if (curFile > 8) throw new Exception("Illegal FEN, too many files");
        }
        else
        {
          Piece thisPiece = byteToPieces[thisChar];
          if (thisPiece.Type == PieceType.None) throw new Exception("Illegal FEN, found piece character " + thisChar);
          pieces.Add(new PieceOnSquare(Square.FromFileAndRank(curFile, 7 - curRank), thisPiece));
          curFile++;
        }
      }

      SkipAnySpaces();

      // 2. Side to move character
      char sideMoveChar = char.ToLower(fen[charIndex++]);
      SideType sideToMove;
      if (sideMoveChar == 'b')
      {
        sideToMove = SideType.Black;
      }
      else if (sideMoveChar == 'w')
      {
        sideToMove = SideType.White;
      }
      else
        throw new Exception($"Illegal FEN, side to move character {sideMoveChar}");

      SkipAnySpaces();

      // 3. Castling availability
      bool whiteCanOO = false;
      bool whiteCanOOO = false;
      bool blackCanOO = false;
      bool blackCanOOO = false;

      // Extract the castling rights portion from the FEN string
      int nextSpaceIndex = fen.IndexOf(' ', charIndex);
      string castlingRights;

      if (nextSpaceIndex != -1)
      {
        // There is a space, so extract the substring up to that space
        castlingRights = fen.Substring(charIndex, nextSpaceIndex - charIndex);
      }
      else
      {
        // No space found, assume castling rights go to the end of the FEN string
        castlingRights = fen.Substring(charIndex);
      }

      // Only proceed if there are castling rights specified (i.e., the string is not "-")
      if (castlingRights == "-")
      {
        charIndex++;
      }
      else
      {
        //error here - needs to use another approach to get the castling rights
        MGPosition pos = default;
        Square whiteKingSquare = default;
        Square blackKingSquare = default;
        List<Square> whiteRookSquares = new();
        List<Square> blackRookSquares = new();

        foreach (PieceOnSquare ps in pieces)
        {
          int mgPiece = MGChessPositionConverter.MGPieceFromPiece(ps.Piece);
          ulong mgSquare = MGPosition.MGBitBoardFromSquare(ps.Square);
          pos.SetPieceAtBitboardSquare((ulong)mgPiece, mgSquare);

          // Identify king and rook positions
          if (ps.Piece.Type == PieceType.King)
          {
            if (ps.Piece.Side == SideType.White)
            {
              whiteKingSquare = ps.Square;
            }
            else
            {
              blackKingSquare = ps.Square;
            }
          }
          else if (ps.Piece.Type == PieceType.Rook)
          {
            if (ps.Piece.Side == SideType.White)
            {
              whiteRookSquares.Add(ps.Square);
            }
            else
            {
              blackRookSquares.Add(ps.Square);
            }
          }
        }

        int whiteKingSq = whiteKingSquare.SquareIndexStartH1;
        int blackKingSq = blackKingSquare.SquareIndexStartH1;

        for (int i = 0; i < whiteRookSquares.Count; i++)
        {
          byte rook = whiteRookSquares[i].SquareIndexStartH1;
          if (rook < whiteKingSq && rook < 8)
          {
            wKRsquare = whiteRookSquares[i].SquareIndexStartH1;
          }
          else if (rook > whiteKingSq && rook < 8)
          {
            wQRsquare = whiteRookSquares[i].SquareIndexStartH1;
          }
        }

        for (int i = 0; i < blackRookSquares.Count; i++)
        {
          byte rook = blackRookSquares[i].SquareIndexStartH1;
          if (rook < blackKingSq && rook > 55)
          {
            bKRsquare = blackRookSquares[i].SquareIndexStartH1;
          }
          else if (rook > blackKingSq && rook > 55)
          {
            bQRsquare = blackRookSquares[i].SquareIndexStartH1;
          }
        }

        // Variables to hold the rook positions - crucial in chess960
        char whiteKingSideRook = ' ';
        char whiteQueenSideRook = ' ';
        char blackKingSideRook = ' ';
        char blackQueenSideRook = ' ';

        foreach (char c in castlingRights)
        {
          int idx = Array.IndexOf(FileArrayLower, c);
          int idxUpper = Array.IndexOf(FileArrayUpper, c);
          if (char.IsUpper(c) && idxUpper == wKRsquare && wKRsquare < whiteKingSq) // White's castling rights
          {
            whiteKingSideRook = c;
          }
          else if (char.IsUpper(c) && idxUpper == wQRsquare && wQRsquare > whiteKingSq)
          {
            whiteQueenSideRook = c;
          }
          else if (char.IsLower(c) && (idx + 56) == bKRsquare && bKRsquare < blackKingSq) // Black's castling rights
          {
            blackKingSideRook = c;
          }
          else if (char.IsLower(c) && (idx + 56) == bQRsquare && bQRsquare > blackKingSq)
          {
            blackQueenSideRook = c;
          }

          else if (c == 'K' && wKRsquare < whiteKingSq)
          {
            whiteKingSideRook = c;
          }

          else if (c == 'Q' && wQRsquare > whiteKingSq)
          {
            whiteQueenSideRook = c;
          }

          else if (c == 'k' && bKRsquare < blackKingSq)
          {
            blackKingSideRook = c;
          }

          else if (c == 'q' && bQRsquare > blackKingSq)
          {
            blackQueenSideRook = c;
          }
        }

        // Process the castling rights in the FEN string
        foreach (char thisChar in castlingRights)
        {
          if (thisChar == 'K' || thisChar == whiteKingSideRook)
          {
            whiteCanOO = true;
          }
          else if (thisChar == 'Q' || thisChar == whiteQueenSideRook)
          {
            whiteCanOOO = true;
          }
          else if (thisChar == 'k' || thisChar == blackKingSideRook)
          {
            blackCanOO = true;
          }
          else if (thisChar == 'q' || thisChar == blackQueenSideRook)
          {
            blackCanOOO = true;
          }
        }

      }
      charIndex += castlingRights.Length;

      SkipAnySpaces();

      // 4. En passant target square
      int numEPChars = 0;
      PositionMiscInfo.EnPassantFileIndexEnum epColIndex = PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
      int epFile = 0;
      int epRank = 0;
      while (charIndex < fen.Length)
      {
        // Try to work around error in FEN whereby dash is missing, and move number immediately follows
        if (numEPChars == 0 && char.IsDigit(fen[charIndex]))
          break;

        char thisChar = fen[charIndex++];
        if (thisChar == '-' || thisChar == ' ')
          break;
        else
        {
          if (numEPChars == 0)
            epFile = char.ToLower(thisChar) - 'a';
          else if (numEPChars == 1)
            epRank = char.ToLower(thisChar) - '0';
          else
            throw new Exception("too many en passant characters");

          numEPChars++;
        }
      }

      SkipAnySpaces();

      if (numEPChars > 0)
      {
        if (numEPChars == 2)
        {
          if (epFile > (byte)PositionMiscInfo.EnPassantFileIndexEnum.FileH)
            throw new Exception("Invalid en passant file in FEN");
          else
            epColIndex = (PositionMiscInfo.EnPassantFileIndexEnum)epFile;
        }
        else
          throw new Exception("Invalid en passant in FEN");
      }

      SkipAnySpacesOrDash(); // Sometimes we see ill-formed FENs that end with extaneous dash, e.g. "w q - -" 

      // 5. Halfmove clock: This is the number of halfmoves since the last capture or pawn advance. 
      int move50Count = 0;
      while (charIndex < fen.Length && charIndex < fen.Length)
      {
        char thisChar = fen[charIndex++];
        if (thisChar == ' ')
          break;
        else
          move50Count = move50Count * 10 + (thisChar - '0');
      }

      SkipAnySpaces();

      // Fullmove number
      int fullmoveCount = 0;
      while (charIndex < fen.Length && charIndex < fen.Length)
      {
        char thisChar = fen[charIndex++];
        if (!char.IsDigit(thisChar))
        {
          break;
        }
        else
        {
          fullmoveCount = fullmoveCount * 10 + (thisChar - '0');
        }
      }

      int plyCount;
      if (fullmoveCount == 0)
      {
        // Fill in move count with 1 if none found
        plyCount = 1;
      }
      else
      {
        // Convert from moves to ply
        plyCount = fullmoveCount * 2;
      }

      if (sideToMove == SideType.Black)
      {
        plyCount++;
      }

      if (charIndex < fen.Length - 1)
      {
        string remainder = fen.Substring(charIndex);
        if (StringUtils.WhitespaceRemoved(remainder) != "")
        {
          throw new Exception($"Unexpected characters after FEN {remainder} ");
        }
      }

      PositionMiscInfo miscInfo = new PositionMiscInfo(whiteCanOO, whiteCanOOO, blackCanOO, blackCanOOO,
                                                        sideToMove, move50Count, repetitionCount, plyCount, epColIndex);

      return new FENParseResult(pieces, miscInfo);
    }

    #region Internal statics

    static Piece[] byteToPieces;

    /// <summary>
    /// Static initializer to initialize bytesToPieces array.
    /// </summary>
    static FENParser()
    {
      byteToPieces = new Piece[byte.MaxValue - 1];

      byteToPieces['K'] = (White, King);
      byteToPieces['Q'] = (White, Queen);
      byteToPieces['R'] = (White, Rook);
      byteToPieces['B'] = (White, Bishop);
      byteToPieces['N'] = (White, Knight);
      byteToPieces['P'] = (White, Pawn);

      byteToPieces['k'] = (Black, King);
      byteToPieces['q'] = (Black, Queen);
      byteToPieces['r'] = (Black, Rook);
      byteToPieces['b'] = (Black, Bishop);
      byteToPieces['n'] = (Black, Knight);
      byteToPieces['p'] = (Black, Pawn);
    }

    #endregion


  }
}
