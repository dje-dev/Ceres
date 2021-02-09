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
using Ceres.Base.Misc;
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
    /// <returns></returns>
    public static FENParseResult ParseFEN(string fen)
    {
      try
      {
        return DoParseFEN(fen);
      }
      catch (Exception exc)
      {
        throw new Exception($"Unable to parse the FEN: {fen}");
      }
    }

    /// <summary>
    /// Worker method to do FEN parsing.
    /// NOTE: performance could be improved by passing in the Piece[] preallocated
    /// </summary>
    /// <param name="fen"></param>
    /// <returns></returns>
    static FENParseResult DoParseFEN(string fen)
    {
      int charIndex = 0;

      void SkipAnySpaces() { while (charIndex < fen.Length && char.IsWhiteSpace(fen[charIndex])) charIndex++; }
      void SkipAnySpacesOrDash() { while (charIndex < fen.Length && (fen[charIndex] == '-' || char.IsWhiteSpace(fen[charIndex]))) charIndex++; }

      List<PieceOnSquare> pieces = new List<PieceOnSquare>(32);

      int curRank = 0;
      int curFile = 0;

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
        sideToMove = SideType.Black;
      else if (sideMoveChar == 'w')
        sideToMove = SideType.White;
      else
        throw new Exception($"Illegal FEN, side to move character {sideMoveChar}");

      SkipAnySpaces();

      // 3. Castling availability
      bool whiteCanOO = false;
      bool whiteCanOOO = false;
      bool blackCanOO = false;
      bool blackCanOOO = false;
      while (charIndex < fen.Length)
      {
        char thisChar = fen[charIndex++];
        if (thisChar == 'K' || thisChar == 'H')
          whiteCanOO = true;
        else if (thisChar == 'Q' || thisChar == 'A')
          whiteCanOOO = true;
        else if (thisChar == 'k' || thisChar == 'h')
          blackCanOO = true;
        else if (thisChar == 'q' || thisChar == 'a')
          blackCanOOO = true;
        else if (thisChar == ' ')
          break;
      }

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
          break;
        else
          fullmoveCount = fullmoveCount * 10 + (thisChar - '0');
      }

      // Fill in move count with 1 if none found
      if (fullmoveCount == 0)
        fullmoveCount = 1;

      if (charIndex < fen.Length - 1)
      {
        string remainder = fen.Substring(charIndex);
        if (StringUtils.WhitespaceRemoved(remainder) != "")
          throw new Exception($"Unexpected characters after FEN {remainder} ");
      }

      PositionMiscInfo miscInfo = new PositionMiscInfo(whiteCanOO, whiteCanOOO, blackCanOO, blackCanOOO, sideToMove, move50Count, 0, fullmoveCount, epColIndex);

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
