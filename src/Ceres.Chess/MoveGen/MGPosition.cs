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
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen.Converters;
using static Ceres.Chess.MoveGen.MGPosition;
using static Ceres.Chess.Position;
using BitBoard = System.UInt64;

#endregion

namespace Ceres.Chess.MoveGen
{
#if NOT
  
// Build Options:
//   #define _USE_HASH 1								        // if undefined, entire hash table system will be excluded from build
//   #define _FLAG_CHECKS_IN_MOVE_GENERATION 1	// Move generator will set "Check" flag in moves which put enemy in check (not needed for perft)

#endif

  [Serializable]
  public partial struct MGPosition : IEquatable<MGPosition>
  {
    /* --------------------------------------------------------------
        Explanation of Bit-Representation used in positions:

        D (msb):	Colour (1=Black) 
        C			1=Straight-Moving Piece (Q or R)
        B			1=Diagonal-moving piece (Q or B)
        A (lsb):	1=Pawn

        D	C	B	A

    (0)		0	0	0	0	Empty Square
    (1)		0	0	0	1	White Pawn	
    (2)		0	0	1	0	White Bishop
    (3)		0	0	1	1	White En passant Square
    (4)		0	1	0	0	White Rook
    (5)		0	1	0	1	White Knight
    (6)		0	1	1	0	White Queen
    (7)		0	1	1	1	White King
    (8)		1	0	0	0	Reserved-Don't use (Black empty square)
    (9)		1	0	0	1	Black Pawn	
    (10)	1	0	1	0	Black Bishop
    (11)	1	0	1	1	Black En passant Square
    (12)	1	1	0	0	Black Rook
    (13)	1	1	0	1	Black Knight
    (14)	1	1	1	0	Black Queen
    (15)	1	1	1	1	Black King
    ---------------------------------------------------------------*/

    public BitBoard A;
    public BitBoard B;
    public BitBoard C;
    public BitBoard D;

    public FlagsEnum Flags;

    public short MoveNumber;
    public RookPlacementInfo rookInfo;

    /// <summary>
    /// Rule 50 count.
    /// </summary>
    public byte Rule50Count;

    /// <summary>
    /// Repetition count.
    /// N.B. This is populated only in certain situations.
    /// </summary>
    public byte RepetitionCount;

#if MG_USE_HASH
    public ulong HK;
#endif

    public MGPosition(in MGPosition copyPos)
    {
      this = copyPos;
    }

    [Flags]
    public enum FlagsEnum : short
    {
      None = 0,
      BlackToMove = 1 << 0,
      WhiteCanCastle = 1 << 1,
      WhiteCanCastleLong = 1 << 2,
      BlackCanCastle = 1 << 3,
      BlackCanCastleLong = 1 << 4,
      WhiteForfeitedCastle = 1 << 5,
      WhiteForfeitedCastleLong = 1 << 6,
      BlackForfeitedCastle = 1 << 7,
      BlackForfeitedCastleLong = 1 << 8,
      WhiteDidCastle = 1 << 9,
      WhiteDidCastleLong = 1 << 10,
      BlackDidCastle = 1 << 11,
      BlackDidCastleLong = 1 << 12,
      CheckmateWhite = 1 << 13,
      CheckmateBlack = 1 << 14,
    }

    public static ulong MGBitBoardFromSquare(Square square) => SquareMap[square.SquareIndexStartA1];

    /// <summary>
    /// Returns MGPosition corresponding to specified FEN string.
    /// </summary>
    /// <param name="fen"></param>
    /// <returns></returns>
    public static MGPosition FromFEN(string fen) => MGChessPositionConverter.MGChessPositionFromFEN(fen);


    /// <summary>
    /// Returns the Move50Category classification for this position.
    /// </summary>
    public readonly Move50CategoryEnum Move50Category => Rule50Count switch
    {
      < 75 => Move50CategoryEnum.LessThan75,
      >= 76 and <= 90 => Move50CategoryEnum.From76Thru90,
      >= 91 and <= 97 => Move50CategoryEnum.From91Thru97,
      _ => Move50CategoryEnum.Above97,
    };


    /// <summary>
    /// Converts Position to MGPosition.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static MGPosition FromPosition(in Position pos) => MGChessPositionConverter.MGChessPositionFromPosition(in pos);


    /// <summary>
    /// Returns Position corresponding to this MGPosition.
    /// </summary>
    public readonly Position ToPosition => MGChessPositionConverter.PositionFromMGChessPosition(in this);


    /// <summary>
    /// Returns which side has their turn to move.
    /// </summary>
    public readonly SideType SideToMove => BlackToMove ? SideType.Black : SideType.White;


    /// <summary>
    /// Returns if Black is to move.
    /// </summary>
    public bool BlackToMove
    {
      readonly get => (Flags & FlagsEnum.BlackToMove) != 0;
      set { if (value) Flags |= FlagsEnum.BlackToMove; else Flags &= ~FlagsEnum.BlackToMove; }
    }

    public bool WhiteCanCastle
    {
      readonly get => (Flags & FlagsEnum.WhiteCanCastle) != 0;
      set { if (value) Flags |= FlagsEnum.WhiteCanCastle; else Flags &= ~FlagsEnum.WhiteCanCastle; }
    }

    public bool WhiteCanCastleLong
    {
      readonly get => (Flags & FlagsEnum.WhiteCanCastleLong) != 0;
      set { if (value) Flags |= FlagsEnum.WhiteCanCastleLong; else Flags &= ~FlagsEnum.WhiteCanCastleLong; }
    }


    public bool BlackCanCastle
    {
      readonly get => (Flags & FlagsEnum.BlackCanCastle) != 0;
      set { if (value) Flags |= FlagsEnum.BlackCanCastle; else Flags &= ~FlagsEnum.BlackCanCastle; }
    }

    public bool BlackCanCastleLong
    {
      readonly get => (Flags & FlagsEnum.BlackCanCastleLong) != 0;
      set { if (value) Flags |= FlagsEnum.BlackCanCastleLong; else Flags &= ~FlagsEnum.BlackCanCastleLong; }
    }

    public bool WhiteForfeitedCastle
    {
      readonly get => (Flags & FlagsEnum.WhiteForfeitedCastle) != 0;
      set { if (value) Flags |= FlagsEnum.WhiteForfeitedCastle; else Flags &= ~FlagsEnum.WhiteForfeitedCastle; }
    }

    public bool WhiteForfeitedCastleLong
    {
      readonly get => (Flags & FlagsEnum.WhiteForfeitedCastleLong) != 0;
      set { if (value) Flags |= FlagsEnum.WhiteForfeitedCastleLong; else Flags &= ~FlagsEnum.WhiteForfeitedCastleLong; }
    }


    public bool BlackForfeitedCastle
    {
      readonly get => (Flags & FlagsEnum.BlackForfeitedCastle) != 0;
      set { if (value) Flags |= FlagsEnum.BlackForfeitedCastle; else Flags &= ~FlagsEnum.BlackForfeitedCastle; }
    }

    public bool BlackForfeitedCastleLong
    {
      readonly get => (Flags & FlagsEnum.BlackForfeitedCastleLong) != 0;
      set { if (value) Flags |= FlagsEnum.BlackForfeitedCastleLong; else Flags &= ~FlagsEnum.BlackForfeitedCastleLong; }
    }

    public bool WhiteDidCastle
    {
      readonly get => (Flags & FlagsEnum.WhiteDidCastle) != 0;
      set { if (value) Flags |= FlagsEnum.WhiteDidCastle; else Flags &= ~FlagsEnum.WhiteDidCastle; }
    }

    public bool WhiteDidCastleLong
    {
      readonly get => (Flags & FlagsEnum.WhiteDidCastleLong) != 0;
      set { if (value) Flags |= FlagsEnum.WhiteDidCastleLong; else Flags &= ~FlagsEnum.WhiteDidCastleLong; }
    }


    public bool BlackDidCastle
    {
      readonly get => (Flags & FlagsEnum.BlackDidCastle) != 0;
      set { if (value) Flags |= FlagsEnum.BlackDidCastle; else Flags &= ~FlagsEnum.BlackDidCastle; }
    }

    public bool BlackDidCastleLong
    {
      readonly get => (Flags & FlagsEnum.BlackDidCastleLong) != 0;
      set { if (value) Flags |= FlagsEnum.BlackDidCastleLong; else Flags &= ~FlagsEnum.BlackDidCastleLong; }
    }


    public bool CheckmateWhite
    {
      readonly get => (Flags & FlagsEnum.CheckmateWhite) != 0;
      set { if (value) Flags |= FlagsEnum.CheckmateWhite; else Flags &= ~FlagsEnum.CheckmateWhite; }
    }


    public bool CheckmateBlack
    {
      readonly get => (Flags & FlagsEnum.CheckmateBlack) != 0;
      set { if (value) Flags |= FlagsEnum.CheckmateBlack; else Flags &= ~FlagsEnum.CheckmateBlack; }
    }




#if MG_USE_HASH
    // TODO: make use of the 
    public void CalculateHash()
    {
      HK = 0;

      // Scan the squares:
      for (int q = 0; q < 64; q++)
      {
        ulong piece = (ulong)GetPieceAtBitboardSquare(1UL << q);
        if ((piece & 0x7) != 0)
          HK ^= MGZobristKeySet.zkPieceOnSquare[piece][q];
      }

      if (BlackToMove) HK ^= MGZobristKeySet.zkBlackToMove;

      if (WhiteCanCastle) HK ^= MGZobristKeySet.zkWhiteCanCastle;
      if (WhiteCanCastleLong) HK ^= MGZobristKeySet.zkWhiteCanCastleLong;
      if (BlackCanCastle) HK ^= MGZobristKeySet.zkBlackCanCastle;
      if (BlackCanCastleLong) HK ^= MGZobristKeySet.zkBlackCanCastleLong;
    }
#endif


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly PieceType PieceMoving(EncodedMove thisMove)
    {
      EncodedSquare square = thisMove.FromSquare;
      return BlackToMove ? GetPieceAtSquare(new Square(square.Flipped.AsByte)).Type
                         : GetPieceAtSquare(new Square(square.AsByte)).Type;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly PieceType PieceCapturing(EncodedMove thisMove)
    {
      EncodedSquare square = thisMove.ToSquare;
      return BlackToMove ? GetPieceAtSquare(new Square(square.Flipped.AsByte)).Type
                         : GetPieceAtSquare(new Square(square.AsByte)).Type;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal readonly MGPositionConstants.MCChessPositionPieceEnum PieceCapturingRaw(EncodedMove thisMove)
    {
      EncodedSquare square = thisMove.ToSquare;
      return BlackToMove ? RawPieceAtSquare(new Square(square.Flipped.AsByte))
                         : RawPieceAtSquare(new Square(square.AsByte));
    }


    /// <summary>
    /// Set a specified square to have a specified piece.
    /// </summary>
    /// <param name="piece"></param>
    /// <param name="square"></param>
    internal void SetPieceAtBitboardSquare(ulong piece, BitBoard square)
    {
      // Clear each square and install new piece.
      A &= ~square;
      if ((piece & 1) != 0) A |= square;
      B &= ~square;
      if ((piece & 2) != 0) B |= square;
      C &= ~square;
      if ((piece & 4) != 0) C |= square;
      D &= ~square;
      if ((piece & 8) != 0) D |= square;
    }


    /// <summary>
    /// Returns piece on specified square.
    /// </summary>
    /// <param name="square"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal readonly int GetPieceAtBitboardSquare(BitBoard square)
    {
      int V;
      V = (D & square) != 0 ? 8 : 0;
      V |= (C & square) != 0 ? 4 : 0;
      V |= (B & square) != 0 ? 2 : 0;
      V |= (A & square) != 0 ? 1 : 0;
      return V;
    }


    /// <summary>
    /// Modifies position so that en passant is assumed allowable in every file.
    /// </summary>
    public void SetAllEnPassantAllowed()
    {
      bool whiteToMove = SideToMove == SideType.White;
      int rank = whiteToMove ? 5 : 2;
      int rankPriorFrom = whiteToMove ? 6 : 1;
      int rankPriorTo = whiteToMove ? 4 : 3;

      // Check en passant square on every file for empty.
      for (int file = 0; file < 8; file++)
      {
        Square square = Square.FromFileAndRank(file, rank);
        if (GetPieceAtSquare(square).Type == PieceType.None)
        {
          Piece priorFrom = GetPieceAtSquare(Square.FromFileAndRank(file, rankPriorFrom));
          Piece priorTo = GetPieceAtSquare(Square.FromFileAndRank(file, rankPriorTo));

          if (priorFrom.Type == PieceType.None
           && priorTo.Type == PieceType.Pawn
           && priorTo.Side == (whiteToMove ? SideType.Black : SideType.White)
           )
          {

            // TODO: Could be more selective and only switch to en passant if the pawn is in expected position.
            SetPieceAtBitboardSquare((ulong)(whiteToMove ? MGChessPositionConverter.EnPassantBlack : MGChessPositionConverter.EnPassantWhite), MGBitBoardFromSquare(square));
          }
        }
      }
    }


    // TO DO: see if this is duplicated elsewhere
    static PieceType[] MGPieceCodeToPieceType;


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Piece GetPieceAtSquare(Square square)
    {
      int pieceCode = GetPieceAtBitboardSquare(MGBitBoardFromSquare(square));
      PieceType pieceType = MGPieceCodeToPieceType[pieceCode];
      SideType side = pieceCode >= MGPositionConstants.BPAWN ? SideType.Black : SideType.White;
      return new Piece(side, pieceType);
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly MGPositionConstants.MCChessPositionPieceEnum RawPieceAtSquare(Square square)
    {
      int pieceCode = GetPieceAtBitboardSquare(MGBitBoardFromSquare(square));
      return (MGPositionConstants.MCChessPositionPieceEnum)pieceCode;
    }


    /// <summary>
    /// Returns if the specified move is legal from this position.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public readonly bool IsLegalMove(MGMove move)
    {
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in this, moves);
      for (int i = 0; i < moves.NumMovesUsed; i++)
      {
        if (moves.MovesArray[i] == move)
        {
          return true;
        }
      }
      return false;
    }


    /// <summary>
    /// Returns count of total number of pieces on the board.
    /// </summary>
    public readonly int PieceCount
    {
      get
      {
        // Identify squares that are definitely NOT actual pieces.
        // Nibble values we exclude:
        //    0  => (A=0, B=0, C=0, D=0)  => empty squares
        //    3  => (A=1, B=1, C=0, D=0)  => white en-passant
        //    8  => (A=0, B=0, C=0, D=1)  => reserved/unused
        //    11 => (A=1, B=1, C=0, D=1)  => black en-passant

        // 1) Squares with nibble = 0 (empty squares):
        ulong emptySquares = ~(A | B | C | D);

        // 2) Squares with nibble = 3 (white en-passant):
        ulong whiteEnPassant = A & B & ~C & ~D;

        // 3) Squares with nibble = 8 (reserved):
        ulong reserved = ~A & ~B & ~C & D;

        // 4) Squares with nibble = 11 (black en-passant):
        ulong blackEnPassant = A & B & ~C & D;

        // Combine all invalid squares (where there is no real piece):
        ulong invalidMask = emptySquares | whiteEnPassant | reserved | blackEnPassant;

        // The complement contains all squares with valid pieces.
        ulong pieceMask = ~invalidMask;

        return BitOperations.PopCount(pieceMask);
      }
    }


    /// <summary>
    /// Returns number of white pawns on 7th rank.
    /// </summary>
    public readonly int NumPawnsRank7(bool white)
    {
      ulong whitePawns = ~D & ~C & ~B & A;
      ulong blackPawns = D & ~C & ~B & A;

      const ulong BOARD_RANK_2 = 0x000000000000FF00UL;
      const ulong BOARD_RANK_7 = 0x00FF000000000000UL;

      ulong whitePawnsRank7 = whitePawns & BOARD_RANK_7;
      ulong blackPawnsRank7 = blackPawns & BOARD_RANK_2;

      return white ? BitOperations.PopCount(whitePawnsRank7)
                    : BitOperations.PopCount(blackPawnsRank7);
    }


    /// <summary>
    /// Returns number of pawns still on their second rank (not yet advanced).
    /// </summary>
    public readonly int NumPawnsRank2
    {
      get
      {
        ulong whitePawns = ~D & ~C & ~B & A;
        ulong blackPawns = D & ~C & ~B & A;

        const ulong BOARD_RANK_2 = 0x000000000000FF00UL;
        const ulong BOARD_RANK_7 = 0x00FF000000000000UL;

        ulong whitePawnsRank2 = whitePawns & BOARD_RANK_2;
        ulong blackPawnsRank2 = blackPawns & BOARD_RANK_7;

        return BitOperations.PopCount(whitePawnsRank2) + BitOperations.PopCount(blackPawnsRank2);
      }
    }


    /// <summary>
    /// Calculates if the position is terminal.
    /// </summary>
    /// <param name="knownMoveList"></param>
    /// <returns></returns>
    public readonly GameResult CalcTerminalStatus(MGMoveList knownMoveList = null)
    {
      // Generate moves to check for checkmate
      MGMoveList moves;
      if (knownMoveList != null)
      {
        moves = knownMoveList;
      }
      else
      {
        // Move list not already known, generate
        moves = new MGMoveList();
        MGMoveGen.GenerateMoves(in this, moves);
      }

      if (moves.NumMovesUsed > 0)
      {
        return GameResult.Unknown;
      }
      else if (MGMoveGen.IsInCheck(in this, BlackToMove))
      {
        return GameResult.Checkmate;
      }
      else
      {
        return GameResult.Draw; // stalemate
      }
    }


    /// <summary>
    /// Calculates to draw status for this position.
    /// </summary>
    public readonly PositionDrawStatus CheckDrawBasedOnMaterial
    {
      get
      {
        // TODO: consider a short circuited version of PieceCount for improved efficiency
        int pieceCount = PieceCount;

        if (pieceCount == 2)
        {
          return PositionDrawStatus.DrawByInsufficientMaterial;
        }
        else if (pieceCount > 4)
        {
          // Our special material rules only apply of 4 or less pieces
          return PositionDrawStatus.NotDraw;
        }
        else
        {
          // Fallback, convert to Position use method there.
          // TODO: For performance, implement directly here without conversion
          return ToPosition.CheckDrawBasedOnMaterial;
        }
      }
    }


    #region Overrides

    public static bool operator ==(MGPosition pos1, MGPosition pos2) => pos1.Equals(pos2);

    public static bool operator !=(MGPosition pos1, MGPosition pos2) => !pos1.Equals(pos2);


    public override bool Equals(object obj) => obj is MGPosition && Equals(obj);

    /// <summary>
    /// Returns if equal to another position with respect to location and position of all pieces
    /// (ignores miscellaneous flags like castling, Move50Count, etc.).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public readonly bool EqualPiecePositionsIncludingEnPassant(in MGPosition other)
    {
      return A == other.A &&
             B == other.B &&
             C == other.C &&
             D == other.D;
    }

    /// <summary>
    /// Returns if equal to another position with respect to location and position of all pieces
    /// (ignores miscellaneous flags like castling, Move50Count, etc.).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public readonly bool EqualPiecePositionsExcludingEnPassant(in MGPosition other)
    {
      // Make copies of position with en passant allowed on all files.
      MGPosition posThis = this;
      posThis.SetAllEnPassantAllowed();
      MGPosition posOther = other;
      posOther.SetAllEnPassantAllowed();

      return posThis.EqualPiecePositionsIncludingEnPassant(in posOther);
    }


    /// <summary>
    /// Returns <c>true</c> if <paramref name="move"/> produces an
    /// irreversible change when going from the current position (<c>this</c>)
    /// to <paramref name="nextPosition"/>.
    /// </summary>
    public readonly bool IsIrreversibleMove(MGMove move, in MGPosition nextPosition)
    {
      // Material changes
      if (move.Capture || move.EnPassantCapture)
      {
        return true;
      }

      // Pawn activity.
      byte pc = (byte)move.Piece;
      bool pawnMove = pc == MGPositionConstants.WPAWN || pc == MGPositionConstants.BPAWN;

      if (pawnMove || move.DoublePawnMove || move.IsPromotion)
      {
        return true;
      }

      // Castling
      if (move.CastleShort || move.CastleLong)
      {
        return true;
      }

      // Change in castling rights (king/rook moves).
      if ((WhiteCanCastle && !nextPosition.WhiteCanCastle) ||
          (WhiteCanCastleLong && !nextPosition.WhiteCanCastleLong) ||
          (BlackCanCastle && !nextPosition.BlackCanCastle) ||
          (BlackCanCastleLong && !nextPosition.BlackCanCastleLong))
      {
        return true;
      }

      return false;
    }


    /// <summary>
    /// Tests for equality with another MGChessMove
    /// (using chess semantics which implies that the MoveNum is irrelevant).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public readonly bool Equals(MGPosition other)
    {
      return EqualPiecePositionsIncludingEnPassant(in other) &&
             Flags == other.Flags &&
             Rule50Count == other.Rule50Count;
    }

    public readonly override int GetHashCode()
    {
      return HashCode.Combine(A, B, C, D, Flags, MoveNumber, Rule50Count);
    }

    #endregion

    #region Initialization

    static ulong[] SquareMap;


    static void InitializeSquareMap()
    {
      SquareMap = new BitBoard[64];
      for (int i = 0; i < 64; i++)
      {
        Square square = new Square(i);
        SquareMap[i] = 1UL << (square.Rank * 8 + (7 - square.File));
      }
    }



    [ModuleInitializer]
    internal static void ClassInitialize()
    {
      InitializeSquareMap();

      MGPieceCodeToPieceType = [ PieceType.None, PieceType.Pawn, PieceType.Bishop, PieceType.None, PieceType.Rook, PieceType.Knight, PieceType.Queen, PieceType.King,
                                 PieceType.None, PieceType.Pawn, PieceType.Bishop, PieceType.None, PieceType.Rook, PieceType.Knight, PieceType.Queen, PieceType.King ];

    }

    #endregion

  }

  public enum Move50CategoryEnum
  {
    LessThan75,
    From76Thru90,
    From91Thru97,
    Above97,
  }


  public static class MGPositionHashing
  {
    public static PosHash64 Hash64(in MGPosition mgPos)
    {
      static ulong Fmix64(ulong k)
      {
        k ^= k >> 33; k *= 0xff51afd7ed558ccdUL;
        k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53UL;
        k ^= k >> 33;
        return k;
      }

      // xxHash-style odd primes
      const ulong P1 = 0x9E3779B185EBCA87UL;
      const ulong P2 = 0xC2B2AE3D27D4EB4FUL;
      const ulong P3 = 0x165667B19E3779F9UL;
      const ulong P4 = 0x27D4EB2F165667C5UL;

      static void Mix(ref ulong a, ulong k, ulong prime)
      {
        a ^= Fmix64(k * prime); // per-word avalanche
        a = BitOperations.RotateLeft(a, 27);
        a *= P1;
        a += P4;
      }

      ulong acc = 0;

      Mix(ref acc, mgPos.A, P1);
      Mix(ref acc, mgPos.B, P2);
      Mix(ref acc, mgPos.C, P3);
      Mix(ref acc, mgPos.D, P4);

      // Misc bundle: side-to-move, castling rights, rook info.
      const FlagsEnum KEEP =
            FlagsEnum.BlackToMove
          | FlagsEnum.WhiteCanCastle | FlagsEnum.WhiteCanCastleLong
          | FlagsEnum.BlackCanCastle | FlagsEnum.BlackCanCastleLong;

      ulong misc = ((ulong)(ushort)(mgPos.Flags & KEEP) << 48)
                 | ((ulong)(ushort)mgPos.rookInfo.RawValue << 16);

      Mix(ref acc, misc, P2);

      return new PosHash64(acc);
    }


    /// <summary>
    /// Add 50-move / repetition info to an existing base hash.
    /// </summary>
    /// <param name="hash"></param>
    /// <param name="repetitionCount"></param>
    /// <param name="move50Category"></param>
    /// <returns></returns>
    public static PosHash64WithMove50AndReps Hash64WithMove50AndRepsAdded(in MGPosition position,
                                                                          int repetitionCount,
                                                                          Move50CategoryEnum move50Category)
      => Hash64WithMove50AndRepsAdded(Hash64(position), repetitionCount, move50Category);


    /// <summary>
    /// Add 50-move / repetition info to an existing base hash.
    /// </summary>
    /// <param name="hash"></param>
    /// <param name="repetitionCount"></param>
    /// <param name="move50Category"></param>
    /// <returns></returns>
    public static PosHash64WithMove50AndReps Hash64WithMove50AndRepsAdded(PosHash64 hash,
                                                                          int repetitionCount,
                                                                          Move50CategoryEnum move50Category)
    {
      static ulong Fmix64(ulong k)
      {
        k ^= k >> 33; k *= 0xff51afd7ed558ccdUL;
        k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53UL;
        k ^= k >> 33;
        return k;
      }

      const ulong P1 = 0x9E3779B185EBCA87UL;   // same odd primes
      const ulong P4 = 0x27D4EB2F165667C5UL;

      // Bits 32-47 : Move-50 bucket (4 values fit in 2 bits)
      // Bit      0 : "has repetition" flag (any non-zero -> 1)
      ulong extra = ((ulong)move50Category << 32)
                  | ((ulong)(repetitionCount > 0 ? 1 : 0));

      hash.Hash ^= Fmix64(extra * 0xC2B2AE3D27D4EB4FUL); // prime P2
      hash.Hash = BitOperations.RotateLeft(hash.Hash, 27);
      hash.Hash *= P1;
      hash.Hash += P4;

      return new PosHash64WithMove50AndReps(hash.Hash);
    }


    /// <summary>
    /// Computes a 96-bit position hash for mgPos.
    /// </summary>
    public static PosHash96 Hash96(in MGPosition mgPos)
    {
      // Gather extra, non-board information into one 64-bit
      // word (side-to-move, castling rights, rook file info).
      const FlagsEnum KEEP =
            FlagsEnum.BlackToMove
          | FlagsEnum.WhiteCanCastle | FlagsEnum.WhiteCanCastleLong
          | FlagsEnum.BlackCanCastle | FlagsEnum.BlackCanCastleLong;

      ulong misc = ((ulong)(ushort)(mgPos.Flags & KEEP) << 48)
                 | ((ulong)(ushort)mgPos.rookInfo.RawValue << 16);

      // MurmurHash3 128-bit mix (x64 variant).
      const ulong C1 = 0x87C37B91114253D5UL;
      const ulong C2 = 0x4CF5AD432745937FUL;

      // 40 bytes total = 5 blocks --> seed incorporates that length.
      const ulong LEN = 40UL;

      ulong h1 = 0UL;
      ulong h2 = 0UL;

      static void MixBlock(ref ulong h1, ref ulong h2, ulong k)
      {
        // Mix into h1.
        ulong k1 = k * C1;
        k1 = BitOperations.RotateLeft(k1, 31);
        k1 *= C2;
        h1 ^= k1;
        h1 = BitOperations.RotateLeft(h1, 27);
        h1 += h2;
        h1 = h1 * 5UL + 0x52DCE729UL;

        // Mix into h2.
        ulong k2 = k * C2;
        k2 = BitOperations.RotateLeft(k2, 33);
        k2 *= C1;
        h2 ^= k2;
        h2 = BitOperations.RotateLeft(h2, 31);
        h2 += h1;
        h2 = h2 * 5UL + 0x38495AB5UL;
      }

      // Four 64-bit board words
      MixBlock(ref h1, ref h2, mgPos.A);
      MixBlock(ref h1, ref h2, mgPos.B);
      MixBlock(ref h1, ref h2, mgPos.C);
      MixBlock(ref h1, ref h2, mgPos.D);

      // Fifth block: misc info
      MixBlock(ref h1, ref h2, misc);

      // Finalization (fmix) – avalanche & collapse to 96 bits.
      static ulong Fmix(ulong k)
      {
        k ^= k >> 33;
        k *= 0xFF51AFD7ED558CCDUL;
        k ^= k >> 33;
        k *= 0xC4CEB9FE1A85EC53UL;
        k ^= k >> 33;
        return k;
      }

      h1 ^= LEN;
      h2 ^= LEN;

      h1 += h2;
      h2 += h1;

      h1 = Fmix(h1);
      h2 = Fmix(h2);

      h1 += h2;
      h2 += h1;

      // Collapse 128 --> 96  (low 64 bits + high 32 bits)
      return new PosHash96(High: (uint)(h2 >> 32), Low: h1);
    }
  }
}

public record struct PosHash64(ulong Hash);
public readonly record struct PosHash64WithMove50AndReps(ulong Hash);

[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 12)]
public readonly record struct PosHash96(uint High, ulong Low)
{
  public readonly string ShortStr() => $"{High % 10_000}/{Low % 10_000}";
  public override readonly string ToString() => $"0x{High:X8}/{Low:X16}";
}


[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 12)]
public readonly record struct PosHash96MultisetFinalized(uint High, ulong Low) : IEqualityComparer<PosHash96MultisetFinalized>
{
  public readonly bool Equals(PosHash96MultisetFinalized other)
    => High == other.High && Low == other.Low;

  public readonly bool Equals(PosHash96MultisetFinalized x, PosHash96MultisetFinalized y)
    => x.High == y.High && x.Low == y.Low;


  public override int GetHashCode()
  {
    // Mix the 96 bits (High:32, Low:64) into a strong 32-bit hash.
    // SplitMix64-style avalanching: fast, excellent diffusion.
    ulong x = Low ^ ((ulong)High << 32);

    x ^= 0x9E3779B97F4A7C15ul;               // add a large odd constant
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ul;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBul;
    x ^= x >> 31;

    return (int)(x ^ (x >> 32));
  }

  public int GetHashCode(PosHash96MultisetFinalized obj)
  {
    return obj.GetHashCode();
  }

  public readonly string ShortStr() => $"{High % 10_000}/{Low % 10_000}";

  public override readonly string ToString() => $"0x{High:X8}/{Low:X16}";
}


/// <summary>
/// Immutable 96-bit value (High = upper 32 bits, Low = lower 64 bits).
/// `record struct` already supplies
///   • value-based equality / != / ==  
///   • GetHashCode()  
///   • Deconstruct(out uint High, out ulong Low)  
///   • Printable ToString() – we override for hex formatting.
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 12)]
public record struct PosHash96MultisetRunning(uint High, ulong Low)
{
  /// <summary>
  /// Adds <paramref name="delta"/> to this (a running 96-bit hash).
  /// Operation is commutative (order-insensitive) because it is
  /// simple 96-bit modular addition.
  public void Add(PosHash96 delta)
  {
    // 96-bit modular addition with carry propagation.
    // (this = this + delta)
    ulong newLow = unchecked(Low + delta.Low);
    uint carry = (uint)(newLow < Low ? 1u : 0u);

    Low = newLow;
    High = unchecked(High + delta.High + carry);
  }


  /// <summary>
  /// Adds hash value of a "final "position to a running 96-bit hash in an
  /// order-sensitive manner and returns the new accumulated value.
  /// * Not commutative.
  /// * One fused multiply --> rotate --> xor step (FNV/xxHash flavor).  
  /// </summary>
  public readonly PosHash96MultisetFinalized Finalized(PosHash96 finalHash)
  {
    // 64- & 32-bit odd primes (xxHash / Murmur lineage)
    const ulong P_LOW = 0x9E3779B97F4A7C15UL; // 64-bit
    const uint P_HIGH = 0x85EBCA77u;          // 32-bit

    // Low 64 bits. 
    // FNV-like: H = (H * P) rotl r  ^  delta
    var low = BitOperations.RotateLeft(Low * P_LOW, 27);
    low ^= finalHash.Low;

    // High 32 bits.
    uint high = BitOperations.RotateLeft(High * P_HIGH, 15);
    high ^= finalHash.High;

    // Cross-feed a few high bits of ‘low’ for extra diffusion
    high += (uint)(Low >> 32);

    return new PosHash96MultisetFinalized(High: high, Low: low);
  }

  public readonly string ShortStr() => $"{High % 10_000}/{Low % 10_000}";

  public override string ToString() => $"0x{High:X8}{Low:X16}";
}

