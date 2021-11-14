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
using System.Runtime.CompilerServices;
using BitBoard = System.UInt64;

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen.Converters;
using System.Numerics;

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
    public short Rule50Count;
    //    public short material;

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
    /// Converts Position to MGPosition.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static MGPosition FromPosition(in Position pos) => MGChessPositionConverter.MCChessPositionFromPosition(in pos);


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
    // --------------------------------------------------------------------------------------------
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


    public void SetPieceAtBitboardSquare(ulong piece, BitBoard square)
    {
      // clear the square
      A &= ~square;
      B &= ~square;
      C &= ~square;
      D &= ~square;

      // 'install' the piece
      if ((piece & 1) != 0) A |= square;
      if ((piece & 2) != 0) B |= square;
      if ((piece & 4) != 0) C |= square;
      if ((piece & 8) != 0) D |= square;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly int GetPieceAtBitboardSquare(BitBoard square)
    {
      int V;
      V = (D & square) != 0 ? 8 : 0;
      V |= (C & square) != 0 ? 4 : 0;
      V |= (B & square) != 0 ? 2 : 0;
      V |= (A & square) != 0 ? 1 : 0;
      return V;
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
    public bool IsLegalMove(MGMove move)
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
    /// Returns number of pawns still on their second rank (not yet advanced).
    /// </summary>
    public byte NumPawnsRank2
    {
      get
      {
        ulong whitePawns = ~D & ~C & ~B & A;
        ulong blackPawns = D & ~C & ~B & A;

        const ulong BOARD_RANK_2 = 0x000000000000FF00UL;
        const ulong BOARD_RANK_7 = 0x00FF000000000000UL;

        ulong whitePawnsRank2 = whitePawns & BOARD_RANK_2;
        ulong blackPawnsRank2 = blackPawns & BOARD_RANK_7;

        return (byte)(BitOperations.PopCount(whitePawnsRank2) + BitOperations.PopCount(blackPawnsRank2));
      }
    }

    #region Overrides

    public static bool operator ==(MGPosition pos1, MGPosition pos2) =>  pos1.Equals(pos2);
    
    public static bool operator !=(MGPosition pos1, MGPosition pos2) => !pos1.Equals(pos2);
    

    public override bool Equals(object obj) => obj is MGPosition && Equals(obj);

    /// <summary>
    /// Tests for equality with another MGChessMove
    /// (using chess semantics which implies that the MoveNum is irrelevant).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public readonly bool Equals(MGPosition other)
    {
      return A == other.A &&
             B == other.B &&
             C == other.C &&
             D == other.D &&
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

      MGPieceCodeToPieceType = new[] { PieceType.None, PieceType.Pawn, PieceType.Bishop, PieceType.None, PieceType.Rook, PieceType.Knight, PieceType.Queen, PieceType.King,
                                       PieceType.None, PieceType.Pawn, PieceType.Bishop, PieceType.None, PieceType.Rook, PieceType.Knight, PieceType.Queen, PieceType.King };

    }

    #endregion

  }

}
