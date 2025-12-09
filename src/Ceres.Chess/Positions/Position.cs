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

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Textual;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using static Ceres.Chess.PieceType;
using static Ceres.Chess.PositionMiscInfo;
using static Ceres.Chess.SideType;
using static Ceres.Chess.SquareNames;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Chess position represented as array "piece on of squares"
  /// along with miscellaneous info such as castling rights.
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly struct Position : IEquatable<Position>
  {
    #region Raw squares data

    /// <summary>
    /// To facilitate quick tests for non-equality,
    /// a hash value over all the pieces (but not MiscInfo)
    /// is calculated at construction
    /// </summary>
    readonly public byte PiecesShortHash;

    /// <summary>
    /// Total number of pieces on the board
    /// </summary>
    readonly public byte PieceCount;

    /// <summary>
    /// Miscellaneous position information (castling rights, etc.)
    /// </summary>
    public readonly PositionMiscInfo MiscInfo;


    readonly byte Square_0_1;
    readonly byte Square_2_3;
    readonly byte Square_4_5;
    readonly byte Square_6_7;
    readonly byte Square_8_9;

    readonly byte Square_10_11;
    readonly byte Square_12_13;
    readonly byte Square_14_15;
    readonly byte Square_16_17;
    readonly byte Square_18_19;

    readonly byte Square_20_21;
    readonly byte Square_22_23;
    readonly byte Square_24_25;
    readonly byte Square_26_27;
    readonly byte Square_28_29;

    readonly byte Square_30_31;
    readonly byte Square_32_33;
    readonly byte Square_34_35;
    readonly byte Square_36_37;
    readonly byte Square_38_39;

    readonly byte Square_40_41;
    readonly byte Square_42_43;
    readonly byte Square_44_45;
    readonly byte Square_46_47;
    readonly byte Square_48_49;

    readonly byte Square_50_51;
    readonly byte Square_52_53;
    readonly byte Square_54_55;
    readonly byte Square_56_57;
    readonly byte Square_58_59;

    readonly byte Square_60_61;
    readonly byte Square_62_63;

    #endregion

    /// <summary>
    /// Returns a Position corresponding to specified FEN string.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="repetitionCount"></param>
    /// <returns></returns>
    public static Position FromFEN(string fen, int repetitionCount = 0) => FENParser.ParseFEN(fen, repetitionCount).AsPosition;

    /// <summary>
    /// Returns FEN string corresponding to this position.
    /// </summary>
    public readonly string FEN => FENGenerator.GetFEN(this);


    /// <summary>
    /// Enumeration indicating the draw status of a position.
    /// </summary>
    public enum PositionDrawStatus
    {
      NotDraw,
      DrawByInsufficientMaterial,
      DrawCanBeClaimed
    };

    /// <summary>
    /// Returns the side to move.
    /// </summary>
    public readonly SideType SideToMove => MiscInfo.SideToMove;

    /// <summary>
    /// If the white player is to move.
    /// </summary>
    public readonly bool IsWhite => SideToMove == SideType.White;


    /// <summary>
    /// Calculates to draw status for this position.
    /// </summary>
    public readonly PositionDrawStatus CheckDrawBasedOnMaterial
    {
      get
      {
        int pieceCount = PieceCount;

        if (pieceCount == 2) return PositionDrawStatus.DrawByInsufficientMaterial;

        // Our special material rules only apply of 4 or less pieces
        if (pieceCount > 4)
        {
          return PositionDrawStatus.NotDraw;
        }

        // Count number of bishops and knights, 
        // and immediately return false if Rook or Queen seen
        int bishopCountUs = 0;
        int bishopCountThem = 0;
        int knightCountUs = 0;
        int knightCountThem = 0;
        int otherPieceCount = 0;
        foreach ((Piece piece, Square square) piece in this)
        {
          if (piece.piece.Type == PieceType.Bishop)
          {
            if (piece.piece.Side == MiscInfo.SideToMove)
            {
              bishopCountUs++;
            }
            else
            {
              bishopCountThem++;
            }
          }
          else if (piece.piece.Type == PieceType.Knight)
          {
            if (piece.piece.Side == MiscInfo.SideToMove)
            {
              knightCountUs++;
            }
            else
            {
              knightCountThem++;
            }
          }
          else
          {
            if (piece.piece.Type != PieceType.King)
            {
              otherPieceCount++;
            }
          }
        }

        // Not draw if pawn, rook or queen present
        if (otherPieceCount > 0)
        {
          return PositionDrawStatus.NotDraw;
        }

        if (bishopCountThem == 2 || (bishopCountThem == 1 && knightCountThem == 1))
        {
          return PositionDrawStatus.NotDraw;
        }
        else if (bishopCountUs == 2 || (bishopCountUs == 1 && knightCountUs == 1))
        {
          return PositionDrawStatus.NotDraw;
        }

        return PositionDrawStatus.DrawByInsufficientMaterial;
      }
    }


    /// <summary>
    /// Calculate the draw status for this position.
    /// </summary>
    public readonly PositionDrawStatus CheckDrawCanBeClaimed
    {
      get
      {
        // Check for 50 move rule
        if (MiscInfo.Move50Count > 99)
        {
          return PositionDrawStatus.DrawCanBeClaimed;
        }

        // Two repetitions would mean the position occurred at least 3 times
        return MiscInfo.RepetitionCount >= 2 ? PositionDrawStatus.DrawCanBeClaimed : PositionDrawStatus.NotDraw;
      }
    }

    #region Set fields

    // Note: These methods abuse the readonly marking on this structure by making direct changes.

    internal unsafe void SetMiscInfo(PositionMiscInfo miscInfo)
    {
      fixed (PositionMiscInfo* infoPtr = &MiscInfo)
      {
        *infoPtr = miscInfo;
      }
    }

    internal unsafe void SetPieceCount(byte pieceCount)
    {
      fixed (byte* countPtr = &PieceCount)
      {
        *countPtr = pieceCount;
      }
    }

    internal unsafe void SetShortHash()
    {
      fixed (byte* hashPtr = &PiecesShortHash)
      {
        *hashPtr = CalcShortHash();
      }
    }

    #endregion

    #region Constructors

    /// <summary>
    /// Constructs a position from another Position, optionally reversing.
    /// 
    /// TODO: The semantics of "reverse" are unclear. Should it include castling rights, and side to move?
    ///       Fortunately most callers of this care only about the pieces being reversed, which is unambiguous.
    /// </summary>
    /// <param name="copyPosition"></param>
    /// <param name="reversedPieces"></param>
    /// <param name="reversedSide"></param>
    public Position(in Position copyPosition, bool reversedPieces = false, bool reversedSide = false)
    {
      PieceCount = 0;
      MiscInfo = reversedSide ? new(copyPosition.MiscInfo.BlackCanOO, copyPosition.MiscInfo.BlackCanOOO,
                                copyPosition.MiscInfo.WhiteCanOO, copyPosition.MiscInfo.WhiteCanOOO,
                                copyPosition.MiscInfo.SideToMove.Reversed(), copyPosition.MiscInfo.Move50Count,
                                copyPosition.MiscInfo.RepetitionCount, copyPosition.MiscInfo.MoveNum, copyPosition.MiscInfo.EnPassantFileIndex, copyPosition.MiscInfo.RookInfo)
                          : copyPosition.MiscInfo;

      for (int i = 0; i < 64; i++)
      {
        Square sq = new Square(i);
        Piece piece = copyPosition[sq];
        if (piece.Type != PieceType.None)
        {
          if (reversedPieces)
          {
            sq = sq.Reversed;
            piece = new Piece(reversedSide ? piece.Side.Reversed() : piece.Side, piece.Type);
          }

          int pieceCount = (byte)SetPieceOnSquare(sq.SquareIndexStartA1, piece);

          PieceCount += (byte)pieceCount;
        }
      }

      PiecesShortHash = CalcShortHash();
    }


    /// <summary>
    /// Constructor ffrom a set of piece bitmaps.
    /// </summary>
    /// <param name="whiteKingBitmap"></param>
    /// <param name="whiteQueenBitmap"></param>
    /// <param name="whiteRookBitmap"></param>
    /// <param name="whiteBishopBitmap"></param>
    /// <param name="whiteKnightBitmap"></param>
    /// <param name="whitePawnBitmap"></param>
    /// <param name="blackKingBitmap"></param>
    /// <param name="blackQueenBitmap"></param>
    /// <param name="blackRookBitmap"></param>
    /// <param name="blackBishopBitmap"></param>
    /// <param name="blackKnightBitmap"></param>
    /// <param name="blackPawnBitmap"></param>
    /// <param name="miscInfo"></param>
    public Position(ulong whiteKingBitmap, ulong whiteQueenBitmap, ulong whiteRookBitmap,
                    ulong whiteBishopBitmap, ulong whiteKnightBitmap, ulong whitePawnBitmap,
                    ulong blackKingBitmap, ulong blackQueenBitmap, ulong blackRookBitmap,
                    ulong blackBishopBitmap, ulong blackKnightBitmap, ulong blackPawnBitmap,
                    in PositionMiscInfo miscInfo)
    {
      // Obviate requirement of definite assignment to all fields
      Unsafe.SkipInit<Position>(out this);

      MiscInfo = miscInfo;

      PieceCount += SetFromBitmap(whiteKingBitmap, (White, King));
      PieceCount += SetFromBitmap(whiteQueenBitmap, (White, Queen));
      PieceCount += SetFromBitmap(whiteRookBitmap, (White, Rook));
      PieceCount += SetFromBitmap(whiteBishopBitmap, (White, Bishop));
      PieceCount += SetFromBitmap(whiteKnightBitmap, (White, Knight));
      PieceCount += SetFromBitmap(whitePawnBitmap, (White, Pawn));

      PieceCount += SetFromBitmap(blackKingBitmap, (Black, King));
      PieceCount += SetFromBitmap(blackQueenBitmap, (Black, Queen));
      PieceCount += SetFromBitmap(blackRookBitmap, (Black, Rook));
      PieceCount += SetFromBitmap(blackBishopBitmap, (Black, Bishop));
      PieceCount += SetFromBitmap(blackKnightBitmap, (Black, Knight));
      PieceCount += SetFromBitmap(blackPawnBitmap, (Black, Pawn));

      PiecesShortHash = CalcShortHash();
    }


    /// <summary>
    /// Initializes from IEnumerable of pieces.
    /// 
    /// Note that processing of the IEnumerable stops immediately if a PieceType.None is seen.
    /// 
    /// TODO: Some callers may construct an array as the first Span argument, these probably could be made more efficient.
    /// </summary>
    /// <param name="pieces"></param>
    /// <param name="miscInfo"></param>
    public unsafe Position(Span<PieceOnSquare> pieces, in PositionMiscInfo miscInfo)
    {
      PieceCount = 0;
      MiscInfo = miscInfo;

      foreach (PieceOnSquare pieceSquare in pieces)
      {
        if (pieceSquare.Piece.Type == PieceType.None)
        {
          break;
        }

        int pieceCount = (byte)SetPieceOnSquare(pieceSquare.Square.SquareIndexStartA1, pieceSquare.Piece);

        PieceCount += (byte)pieceCount;
      }

      PiecesShortHash = CalcShortHash();
    }


    /// <summary>
    /// Computes the short hash for this position.
    /// </summary>
    /// <returns></returns>
    unsafe readonly byte CalcShortHash()
    {
      // For efficiency we iterate over 4 longs (32 bytes)
      // instead of 32 byte fields.
      long hash = 0;
      fixed (void* pieceSquares = &Square_0_1)
      {
        long* pieceSquaresUint = (long*)pieceSquares;
        for (int i = 0; i < 4; i++)
        {
          hash = (long)hash * -1521134295L + pieceSquaresUint[i];
        }
      }
      return (byte)(hash.GetHashCode());
    }




    /// <summary>
    /// Returns the position with en passant rights which
    /// existed before the pawn move that created these rights.
    /// </summary>
    /// <returns></returns>
    internal readonly Position PosWithEnPassantUndone()
    {
      if (MiscInfo.EnPassantFileIndex == PositionMiscInfo.EnPassantFileIndexEnum.FileNone) throw new Exception("Expected en passant");

      // Build new position, with side reversed and prior move number
      // TODO: could we also decrement the Move50Count, repetition, etc?
      short moveNum = (short)(MiscInfo.MoveNum > 0 ? MiscInfo.MoveNum - 1 : 0);
      PositionMiscInfo newMiscInfo;
      if (MiscInfo.SideToMove == SideType.White)
      {
        newMiscInfo = new PositionMiscInfo(MiscInfo.WhiteCanOO, MiscInfo.WhiteCanOOO, MiscInfo.BlackCanOO, MiscInfo.BlackCanOOO,
                                           MiscInfo.SideToMove.Reversed(), MiscInfo.Move50Count, MiscInfo.RepetitionCount, moveNum,
                                           PositionMiscInfo.EnPassantFileIndexEnum.FileNone, MiscInfo.RookInfo);
      }
      else
      {
        newMiscInfo = new PositionMiscInfo(MiscInfo.BlackCanOO, MiscInfo.BlackCanOOO, MiscInfo.WhiteCanOO, MiscInfo.WhiteCanOOO,
                                           MiscInfo.SideToMove.Reversed(), MiscInfo.Move50Count, MiscInfo.RepetitionCount, moveNum,
                                           PositionMiscInfo.EnPassantFileIndexEnum.FileNone, MiscInfo.RookInfo);
      }

      // TODO: for efficiency, make a [ThreadStatic] PieceOnSquare[] for the temporary here
      List<PieceOnSquare> pieces = new List<PieceOnSquare>(32);
      bool found = false;
      int expectedPawnRank = MiscInfo.SideToMove == SideType.Black ? 3 : 4;
      foreach (PieceOnSquare piece in PiecesEnumeration)
      {
        if (piece.Piece.Type == PieceType.Pawn
         && piece.Square.Rank == expectedPawnRank
         && piece.Square.File == (int)MiscInfo.EnPassantFileIndex
         && piece.Piece.Side != MiscInfo.SideToMove)
        {
          int rankIncrement = piece.Piece.Side == SideType.White ? -2 : 2;
          pieces.Add(new PieceOnSquare(Square.FromFileAndRank(piece.Square.File, piece.Square.Rank + rankIncrement), piece.Piece));
          found = true;
        }
        else
        {
          pieces.Add(piece);
        }
      }

      if (!found)
      {
        throw new Exception("En passant pawn was not found");
      }

      return new Position(pieces.ToArray(), in newMiscInfo);
    }


    #endregion

    /// <summary>
    /// Indexer that returns the piece on a given square.
    /// </summary>
    /// <param name="square"></param>
    /// <returns></returns>
    public readonly Piece this[Square square] => PieceOnSquare(square);


    /// <summary>
    /// Returns position with each square modified according to a specified function.
    /// </summary>
    internal Position Modified(Func<PieceOnSquare, Piece> modifyFunc, PositionMiscInfo miscInfo)
    {
      Span<PieceOnSquare> ps = stackalloc PieceOnSquare[PieceCount];
      int count = 0;
      foreach (Square square in Square.AllSquares)
      {
        Piece newPiece = modifyFunc(new PieceOnSquare(square, this[square]));
        if (newPiece.Type != PieceType.None)
        {
          ps[count++] = new PieceOnSquare(square, newPiece);
        }
      }

      Position newPos = new Position(ps, miscInfo);
      return newPos;
    }


    /// <summary>
    /// Returns position mirrored about the vertical divide of the board.
    /// </summary>
    public readonly Position Mirrored
    {
      get
      {
        Span<PieceOnSquare> ps = stackalloc PieceOnSquare[PieceCount];
        int count = 0;
        foreach (PieceOnSquare pieceOnSquare in PiecesEnumeration)
        {
          ps[count++] = new PieceOnSquare(pieceOnSquare.Square.Mirrored, pieceOnSquare.Piece);
        }
        Position newPos = new Position(ps, MiscInfo.Mirrored);
        return newPos;
      }
    }


    /// <summary>
    /// Returns if this position is equivalent to another position
    /// as far as repetition counting is concerned.
    /// </summary>
    /// <param name="otherPos"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool EqualAsRepetition(in Position otherPos)
    {
      if (PiecesShortHash != otherPos.PiecesShortHash
       || SideToMove != otherPos.SideToMove)
      {
        // Fast path check.
        // NOTE: FIDE rules (9.2.1.2) state "Positions are considered the same if and only if the same player has the move."
        //       however some software (e.g. Arena) fail to implement this correctly.
        return false;
      }
      else
      {
        // Slow infrequent path.
        return EqualAsRepetitionFullCheck(in otherPos);
      }
    }

    /// <summary>
    /// Returns if this position is equivalent to another position
    /// as far as repetition counting is concerned.
    /// </summary>
    /// <param name="otherPos"></param>
    /// <returns></returns>
    readonly bool EqualAsRepetitionFullCheck(in Position otherPos)
    {
      if (!PiecesEqual(in otherPos)) return false;

      // NOTE: FIDE rules (9.2.1.2) state "Positions are considered the same if and only if the same player has the move."
      //       however some software (e.g. Arena) fail to implement this correctly.
      if (MiscInfo.SideToMove != otherPos.MiscInfo.SideToMove
       || MiscInfo.WhiteCanOOO != otherPos.MiscInfo.WhiteCanOOO
       || MiscInfo.WhiteCanOO != otherPos.MiscInfo.WhiteCanOO
       || MiscInfo.BlackCanOOO != otherPos.MiscInfo.BlackCanOOO
       || MiscInfo.BlackCanOO != otherPos.MiscInfo.BlackCanOO
       || MiscInfo.EnPassantFileIndex != otherPos.MiscInfo.EnPassantFileIndex)
      {
        return false;
      }

      return true;
    }


    /// <summary>
    /// Returns if this position is the same as another position
    /// with respect to piece placement.
    /// </summary>
    /// <param name="otherPos"></param>
    /// <returns></returns>
    public unsafe readonly bool PiecesEqual(in Position otherPos)
    {
      if (PieceCount != otherPos.PieceCount) return false;

      unsafe
      {
        fixed (byte* pieceSquares = &Square_0_1)
        fixed (byte* pieceSquaresOther = &otherPos.Square_0_1)
        {
          return new Span<byte>(pieceSquares, 32).SequenceEqual(
                 new Span<byte>(pieceSquaresOther, 32));
        }
      }
    }

    #region Conversion to bitmaps


    /// <summary>
    /// Sets the bit in a bitmap corresponding to specified square.
    /// </summary>
    /// <param name="bitmaps"></param>
    /// <param name="targetSquare"></param>
    /// <param name="bitmapIndex"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void DoApply(Span<BitVector64> bitmaps, int targetSquare, int bitmapIndex)
    {
      bitmaps[bitmapIndex].SetBit(targetSquare);
    }

    /// <summary>
    /// Sets the bit in a bitmap corresponding to specified square (board reversed).
    /// </summary>
    /// <param name="bitmaps"></param>
    /// <param name="targetSquare"></param>
    /// <param name="bitmapIndex"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void DoApplyReversed(Span<BitVector64> bitmaps, int targetSquare, int bitmapIndex)
    {
      int newIndex = ((63 - targetSquare) & 0b11111000) + targetSquare % 8;
      bitmaps[bitmapIndex].SetBit(newIndex);
    }


    /// <summary>
    /// Initialize 16 bitmaps extracted from position:
    ///   (unused, P, N, B, R, Q, K, unused, unused, p, n, b, r, q, k, unused)
    /// </summary>
    /// <param name="bitmaps"></param>
    public readonly void InitializeBitmaps(Span<BitVector64> bitmaps, bool reversed)
    {
      if (reversed)
      {
        DoInitializeBitmapsReversed(bitmaps);
      }
      else
      {
        DoInitializeBitmaps(bitmaps);
      }
    }


    /// <summary>
    /// Set the bits in a multi-bitmap representation of the 
    /// board corresponding to this position.
    /// </summary>
    /// <param name="bitmaps"></param>
    readonly void DoInitializeBitmaps(Span<BitVector64> bitmaps)
    {
      unsafe
      {
        fixed (byte* pieceSquares = &Square_0_1)
        {
          for (int i = 0; i < 32; i++)
          {
            byte rawValue = pieceSquares[i];
            if (rawValue == 0) continue;

            int squareIndex = i * 2;

            byte rawValueLeft = (byte)(rawValue & 0b0000_1111);
            if (rawValueLeft != 0)
            {
              DoApply(bitmaps, squareIndex, rawValueLeft);
            }

            byte rawValueRight = (byte)(rawValue >> 4);
            if (rawValueRight != 0)
            {
              DoApply(bitmaps, squareIndex + 1, rawValueRight);
            }
          }
        }
      }
    }

    /// <summary>
    /// Set the bits in a multi-bitmap representation of the 
    /// board corresponding to this position (reversed board).
    /// </summary>
    /// <param name="bitmaps"></param>
    void DoInitializeBitmapsReversed(Span<BitVector64> bitmaps)
    {
      unsafe
      {
        fixed (byte* pieceSquares = &Square_0_1)
        {
          for (int i = 0; i < 32; i++)
          {
            byte rawValue = pieceSquares[i];
            if (rawValue == 0) continue;

            int squareIndex = i * 2;

            byte rawValueLeft = (byte)(rawValue & 0b0000_1111);
            if (rawValueLeft != 0) DoApplyReversed(bitmaps, squareIndex, rawValueLeft);

            byte rawValueRight = (byte)(rawValue >> 4);
            if (rawValueRight != 0) DoApplyReversed(bitmaps, squareIndex + 1, rawValueRight);
          }
        }
      }
    }

    #endregion

    static ulong[][] FastHashTable; // size [32,256] = 8192*8=128k


    static void InitFastHashTable()
    {
      ulong[][][] keys = EncodedBoardZobrist.Keys;

      FastHashTable = new ulong[32][];

      for (int i = 0; i < 32; i++)
      {
        FastHashTable[i] = new ulong[256];

        for (int rawValue = 0; rawValue < 256; rawValue++)
        {
          ulong thisHashLeft = 0;
          ulong thisHashRight = 0;

          byte valueLeft = (byte)(rawValue & 0b0000_1111);
          int squareIndex = i * 2;
          Piece pieceLeft = new Piece(valueLeft);
          if (pieceLeft.Type != PieceType.None && pieceLeft.Type <= PieceType.King)
          {
            thisHashLeft = keys[(int)pieceLeft.Side][(int)pieceLeft.Type][squareIndex];
          }

          byte valueRight = (byte)(rawValue >> 4);
          squareIndex = i * 2 + 1;
          Piece pieceRight = new Piece(valueRight);
          if (pieceRight.Type != PieceType.None && pieceRight.Type <= PieceType.King)
          {
            thisHashRight = keys[(int)pieceRight.Side][(int)pieceRight.Type][squareIndex];
          }

          FastHashTable[i][rawValue] = thisHashLeft ^ thisHashRight;
        }
      }
    }

    /// <summary>
    /// Computes the Zobrist hash of this position
    /// 
    /// NOTE: For efficiency, the Zobrist hashing logic and position 
    /// interpretation logic is combined together here.
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public readonly ulong CalcZobristHash(PositionMiscInfo.HashMove50Mode hashMode, bool includeRepetitions = true)
    {
      if (FastHashTable == null) InitFastHashTable();

      ulong[][][] keys = EncodedBoardZobrist.Keys;

      ulong hash = 0;
      if (false) // ok but 20% slower
      {
        unsafe
        {
          fixed (byte* pieceSquares = &Square_0_1)
          {
            for (int i = 0; i < 32; i++)
            {
              byte rawValue = pieceSquares[i];
              hash ^= FastHashTable[i][rawValue];
            }
          }
        }
      }
      else
      {
        hash ^= FastHashTable[0][Square_0_1];
        hash ^= FastHashTable[1][Square_2_3];
        hash ^= FastHashTable[2][Square_4_5];
        hash ^= FastHashTable[3][Square_6_7];
        hash ^= FastHashTable[4][Square_8_9];

        hash ^= FastHashTable[5][Square_10_11];
        hash ^= FastHashTable[6][Square_12_13];
        hash ^= FastHashTable[7][Square_14_15];
        hash ^= FastHashTable[8][Square_16_17];
        hash ^= FastHashTable[9][Square_18_19];

        hash ^= FastHashTable[10][Square_20_21];
        hash ^= FastHashTable[11][Square_22_23];
        hash ^= FastHashTable[12][Square_24_25];
        hash ^= FastHashTable[13][Square_26_27];
        hash ^= FastHashTable[14][Square_28_29];

        hash ^= FastHashTable[15][Square_30_31];
        hash ^= FastHashTable[16][Square_32_33];
        hash ^= FastHashTable[17][Square_34_35];
        hash ^= FastHashTable[18][Square_36_37];
        hash ^= FastHashTable[19][Square_38_39];

        hash ^= FastHashTable[20][Square_40_41];
        hash ^= FastHashTable[21][Square_42_43];
        hash ^= FastHashTable[22][Square_44_45];
        hash ^= FastHashTable[23][Square_46_47];
        hash ^= FastHashTable[24][Square_48_49];

        hash ^= FastHashTable[25][Square_50_51];
        hash ^= FastHashTable[26][Square_52_53];
        hash ^= FastHashTable[27][Square_54_55];
        hash ^= FastHashTable[28][Square_56_57];
        hash ^= FastHashTable[29][Square_58_59];

        hash ^= FastHashTable[30][Square_60_61];
        hash ^= FastHashTable[31][Square_62_63];
      }

      hash ^= (ulong)MiscInfo.HashPosition(hashMode, includeRepetitions);

      return MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone
          ? (hash = hash * 31 + (ulong)MiscInfo.EnPassantFileIndex)
          : hash;
    }


    /// <summary>
    /// Computes the Zobrist hash of this position.
    /// 
    /// NOTE: This version no longer used, replaced by the faster version above.
    /// 
    /// NOTE: For efficiency, the Zobrist hashing logic and position interpretation logic is combined together here.
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public readonly ulong CalcZobristHashSlow(PositionMiscInfo.HashMove50Mode hashMode)
    {
      ulong[][][] keys = EncodedBoardZobrist.Keys;

      ulong hash = 0;
      unsafe
      {
        fixed (byte* pieceSquares = &Square_0_1)
        {
          for (int i = 0; i < 32; i++)
          {
            byte rawValue = pieceSquares[i];

            byte valueLeft = (byte)(rawValue & 0b0000_1111);

            if (valueLeft != 0)
            {
              int squareIndex = i * 2;
              Piece piece = new Piece(valueLeft);
              hash ^= keys[(int)piece.Side][(int)piece.Type][squareIndex];
            }

            byte valueRight = (byte)(rawValue >> 4);
            if (valueRight != 0)
            {
              int squareIndex = i * 2 + 1;
              Piece piece = new Piece(valueRight);
              hash ^= keys[(int)piece.Side][(int)piece.Type][squareIndex];
            }
          }
        }
      }


      hash ^= (ulong)MiscInfo.HashPosition(hashMode);

      return hash;
    }



    /// <summary>
    /// Custom enumerator struct for allocation-free enumeration.
    /// </summary>
    public PiecesOnSquaresEnumerator GetEnumerator()
    {
      // Create a span that covers the 32 bytes holding the 64 nibbles
      ReadOnlySpan<byte> squares = MemoryMarshal.CreateReadOnlySpan(ref Unsafe.AsRef(in Square_0_1), 32);

      return new PiecesOnSquaresEnumerator(squares);
    }

    public ref struct PiecesOnSquaresEnumerator
    {
      private readonly ReadOnlySpan<byte> _pieceSquares;
      private int _index;

      public PiecesOnSquaresEnumerator(ReadOnlySpan<byte> pieceSquares)
      {
        _pieceSquares = pieceSquares;
        _index = -1;
        Current = default;
      }

      public (Piece, Square) Current { get; private set; }

      public bool MoveNext()
      {
        while (++_index < 64)
        {
          int half = _index >> 1;              // which byte
          int nibbleShift = (_index & 1) * 4;  // left or right nibble
          byte nibble = (byte)((_pieceSquares[half] >> nibbleShift) & 0xF);

          if (nibble != 0)
          {
            Current = (new Piece(nibble), new Square(_index));
            return true;
          }
        }
        return false;
      }
    }


    /// <summary>
    /// Returns the piece residing on a specified square.
    /// </summary>
    /// <param name="square"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Piece PieceOnSquare(Square square)
    {
      int idx = square.SquareIndexStartA1;

      unsafe
      {
        fixed (byte* pieceSquares = &Square_0_1)
        {
          byte raw = pieceSquares[idx >> 1];                 // / 2
          int shift = (idx & 1) << 2;                        // 0 or 4
          raw = (byte)((raw >> shift) & 0x0F);               // select lower/upper nibble

          return new Piece(raw);
        }
      }
    }



    /// <summary>
    /// Enumerates all the PieceOnSquares on the board.
    /// 
    /// NOTE: This is much slower than the Span version above.
    /// TODO: Remove usages, switch to the Span version above
    /// </summary>
    public readonly IEnumerable<PieceOnSquare> PiecesEnumeration
    {
      get
      {
        for (int i = 0; i < 64; i++)
        {
          Square sq = new Square(i);
          Piece piece = this[sq];
          if (piece.Type != PieceType.None)
          {
            yield return (sq, piece);
          }
        }
      }
    }



    /// <summary>
    /// Calculates if the position is terminal.
    /// </summary>
    /// <param name="knownMoveList"></param>
    /// <returns></returns>
    public readonly GameResult CalcTerminalStatus(MGMoveList knownMoveList = null)
    {
      MGPosition posMG = MGPosition.FromPosition(in this);

      // Generate moves to check for checkmake
      MGMoveList moves;
      if (knownMoveList != null)
      {
        moves = knownMoveList;
      }
      else
      {
        // Move list not already known, generate
        moves = new MGMoveList();
        MGMoveGen.GenerateMoves(in posMG, moves);
      }

      if (moves.NumMovesUsed > 0)
      {
        return GameResult.Unknown;
      }
      else if (MGMoveGen.IsInCheck(in posMG, posMG.BlackToMove))
      {
        return GameResult.Checkmate;
      }
      else
      {
        return GameResult.Draw; // stalemate
      }
    }


    /// <summary>
    /// Returns string graphic reprsentation of pieces on board.
    /// </summary>
    public readonly string BoardPicture
    {
      get
      {
        StringBuilder sb = new StringBuilder();

        for (int rank = 7; rank >= 0; rank--)
        {
          for (int file = 0; file < 8; file++)
          {
            sb.Append(this[Square.FromFileAndRank(file, rank)].Char);
            sb.Append(' ');
          }
          sb.AppendLine();
        }

        sb.AppendLine();
        sb.Append(MiscInfo.ToString());
        return sb.ToString();
      }
    }


    /// <summary>
    /// Updates the position to have a specified piece on a specified square.
    /// </summary>
    /// <param name="squareIndex"></param>
    /// <param name="piece"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal int SetPieceOnSquare(int squareIndex, Piece piece)
    {
      Debug.Assert(piece.RawValue < 15);

      unsafe
      {
        int thisIndex = squareIndex / 2;
        fixed (byte* pieceSquares = &Square_0_1)
        {
          byte byteValue = pieceSquares[thisIndex];
          byteValue = squareIndex % 2 == 1
              ? (byte)((byteValue & 0b0000_1111) | piece.RawValue << 4)
              : (byte)((byteValue & 0b1111_0000) | piece.RawValue);
          pieceSquares[thisIndex] = byteValue;
        }
      }

      return piece.Type == PieceType.None ? 0 : 1;
    }



    /// <summary>
    /// Returns if any instances of a given piece exist in the position.
    /// </summary>
    /// <param name="piece"></param>
    /// <returns></returns>
    public readonly bool PieceExists(Piece piece)
    {
      unsafe
      {
        fixed (byte* pieceSquares = &Square_0_1)
        {
          for (int i = 0; i < 64; i++)
          {
            byte byteValue = pieceSquares[i / 2];
            byteValue = i % 2 == 1 ? (byte)(byteValue >> 4) : (byte)(byteValue & 0b0000_1111);

            if (byteValue == piece.RawValue)
            {
              return true;
            }
          }
        }
        return false;
      }
    }


    /// <summary>
    /// Returns the number of pieces of a specified type that exist in the position.
    /// </summary>
    /// <param name="piece"></param>
    /// <returns></returns>
    public readonly int PieceCountOfType(Piece piece)
    {
      int count = 0;
      unsafe
      {
        fixed (byte* pieceSquares = &Square_0_1)
        {
          for (int i = 0; i < 64; i++)
          {
            byte byteValue = pieceSquares[i / 2];
            byteValue = i % 2 == 1 ? (byte)(byteValue >> 4) : (byte)(byteValue & 0b0000_1111);

            if (byteValue == piece.RawValue)
            {
              count++;
            }
          }
        }
      }
      return count;
    }


    /// <summary>
    /// Updates the board from a bitmap indicating locations of a specified piece.
    /// </summary>
    /// <param name="bits"></param>
    /// <param name="piece"></param>
    /// <param name="scratchIndices"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    byte SetFromBitmap(ulong bits, Piece piece)
    {
      byte numPieces = 0;
      byte thisSquare;
      while (true)
      {
        thisSquare = (byte)System.Numerics.BitOperations.TrailingZeroCount(bits);
        if (thisSquare < 64)
        {
          numPieces += (byte)SetPieceOnSquare(thisSquare, piece);
          bits ^= 1UL << thisSquare;
        }
        else
        {
          return numPieces;
        }
      }
    }



    /// <summary>
    /// Returns string representation (via FEN).
    /// </summary>
    /// <returns></returns>
    public override readonly string ToString() => $"<Position {FEN} REP={MiscInfo.RepetitionCount} {CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98)}>";


    #region Linqpad dump support

    /// <summary>
    /// Dumps an HTML string with embedded SVG description of board,
    /// suitable for rendering in a browser or Linqpad results window.
    /// </summary>
    /// <returns></returns>
    public readonly string DumpHTML(string label)
    {
      PositionDocument pd = new PositionDocument();
      pd.WriteStartSection();
      pd.WritePos(this, 18);//, viewboxSize: 130);
      pd.WriteEndSection();
      for (int i = 0; i < 4; i++)
      {
        label += "<br></br>";
      }
      return pd.FinalText() + label;
    }

    #endregion

    #region Start position

    public static Position StartPosition => startPos;

    static readonly Position startPos = MakeStartPos();

    static Position MakeStartPos()
    {
      PieceOnSquare[] pieces = new PieceOnSquare[]
      {
          (A1, White, Rook), (B1, White, Knight), (C1, White, Bishop), (D1, White, Queen),
          (E1, White,King), (F1, White,Bishop), (G1, White,Knight), (H1, White,Rook),
          (A2, White,Pawn), (B2, White,Pawn), (C2, White,Pawn), (D2, White,Pawn),
          (E2, White,Pawn), (F2, White,Pawn), (G2, White,Pawn), (H2, White,Pawn),

          (A8, Black,Rook), (B8, Black,Knight), (C8, Black,Bishop), (D8, Black,Queen),
          (E8, Black,King), (F8, Black,Bishop), (G8, Black,Knight), (H8, Black,Rook),
          (A7, Black,Pawn), (B7, Black,Pawn), (C7, Black,Pawn), (D7, Black,Pawn),
          (E7, Black,Pawn), (F7, Black,Pawn), (G7, Black,Pawn), (H7, Black,Pawn),
      };

      RookPlacementInfo rookInfo = default;
      rookInfo.WhiteKRInitPlacement = 0;
      rookInfo.WhiteQRInitPlacement = 7;
      rookInfo.BlackKRInitPlacement = 0;
      rookInfo.BlackQRInitPlacement = 7;
      // NOTE: use 2 for ply number so translates to 1 for move number
      PositionMiscInfo miscInfo = new PositionMiscInfo(true, true, true, true, SideType.White,
                                                       0, 0, 2, PositionMiscInfo.EnPassantFileIndexEnum.FileNone, rookInfo);
      return new Position(pieces, in miscInfo);
    }

    #endregion

    #region Moves

    /// <summary>
    /// Returns a list of all legal Moves from this position.
    /// </summary>
    public readonly List<Move> Moves
    {
      get
      {
        MGPosition mgPos = MGPosition.FromPosition(in this);
        MGMoveList moves = new MGMoveList();
        MGMoveGen.GenerateMoves(in mgPos, moves);

        List<Move> ret = new(moves.NumMovesUsed);
        foreach (MGMove move in moves)
        {
          ret.Add(MGMoveConverter.ToMove(move));
        }

        return ret;
      }
    }

    /// <summary>
    /// Returns the Move corresponding to a specified SAN string from this starting position.
    /// </summary>
    /// <param name="sanMoveString"></param>
    /// <returns></returns>
    public readonly Move MoveSAN(string sanMoveString) => Move.FromSAN(in this, sanMoveString);


    /// <summary>
    /// Returns the Move corresponding to a specified UCI string from this starting position.
    /// </summary>
    /// <param name="ucImoveString"></param>
    /// <returns></returns>
    public readonly Move MoveUCI(string ucImoveString) => Move.FromUCI(in this, ucImoveString);


    /// <summary>
    /// Returns the Position resulting from making a sequence of moves specified as SAN strings.
    /// </summary>
    /// <param name="sanMoveStrings"></param>
    /// <returns></returns>
    public readonly Position AfterMovesSAN(params string[] sanMoveStrings)
    {
      Position pos = this;
      foreach (string sanMoveString in sanMoveStrings)
      {
        pos = pos.AfterMove(pos.MoveSAN(sanMoveString));
      }

      return pos;
    }


    /// <summary>
    /// Returns a new Position which results from making specified move on this position.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public readonly Position AfterMove(Move move)
    {
      // TODO: improve efficiency by implementing directly (instead of via MGMove)
      MGMove mgMove = MGMoveConverter.MGMoveFromPosAndMove(in this, move);
      MGPosition mgPos = MGPosition.FromPosition(in this);
      mgPos.MakeMove(mgMove);
      return MGChessPositionConverter.PositionFromMGChessPosition(in mgPos);
    }


    /// <summary>
    /// Returns MGPosition equivalen to this Position.
    /// </summary>
    public readonly MGPosition ToMGPosition => MGPosition.FromPosition(in this);

    #endregion


    #region Make move

    /// <summary>
    /// Mutates this position to reflect the new board position after the speciifed move.
    /// </summary>
    /// <param name="move"></param>
    /// <exception cref="Exception"></exception>
    public unsafe void MakeMove(Move move)
    {
      // Mostly working, but needs more testing (for hours...) and also with FRC.
      // Also needs more optimization, e.g. the loop that counts number of pieces at the end of the method.
      throw new NotImplementedException("MakeMove disabled pending more testing and optimization.");

      // Extract current PositionMiscInfo fields into locals.
      bool newWhiteCanOO = this.MiscInfo.WhiteCanOO;
      bool newWhiteCanOOO = this.MiscInfo.WhiteCanOOO;
      bool newBlackCanOO = this.MiscInfo.BlackCanOO;
      bool newBlackCanOOO = this.MiscInfo.BlackCanOOO;
      int newMove50Count = this.MiscInfo.Move50Count;
      int newRepetitionCount = this.MiscInfo.RepetitionCount;
      int newMoveNum = this.MiscInfo.MoveNum;
      EnPassantFileIndexEnum newEnPassant = this.MiscInfo.EnPassantFileIndex;
      SideType newSide = this.MiscInfo.SideToMove;

      // Save the moving piece before any mutations.
      Piece movingPiece = (move.Type == Move.MoveType.MoveCastleShort || move.Type == Move.MoveType.MoveCastleLong)
          ? // For castling, we need to find the king on the board.
            new Piece(newSide, PieceType.King) // assume the side to move is correct.
          : this.PieceOnSquare(move.FromSquare);

      // Determine if the move is a pawn move.
      bool pawnMoved = (movingPiece.Type == PieceType.Pawn);

      bool isCapture = false;

      if (move.Type == Move.MoveType.MoveCastleShort || move.Type == Move.MoveType.MoveCastleLong)
      {
        // --- CASTLING MOVE ---
        // For castling moves, FromSquare and ToSquare in Move are invalid.
        // We must determine the king and rook positions based on the board.

        // Find the king for the side to move.
        Square kingSquare = FindKing(this.MiscInfo.SideToMove);

        // Define standard castling destination squares.
        Square kingDest, rookDest;
        if (this.MiscInfo.SideToMove == SideType.White)
        {
          if (move.Type == Move.MoveType.MoveCastleShort)
          {
            kingDest = new Square("G1");
            rookDest = new Square("F1");
          }
          else // MoveCastleLong
          {
            kingDest = new Square("C1");
            rookDest = new Square("D1");
          }
        }
        else // Black
        {
          if (move.Type == Move.MoveType.MoveCastleShort)
          {
            kingDest = new Square("G8");
            rookDest = new Square("F8");
          }
          else // MoveCastleLong
          {
            kingDest = new Square("C8");
            rookDest = new Square("D8");
          }
        }

        // Identify the appropriate rook:
        // For kingside, choose the rook that is on a file greater than the king.
        // For queenside, choose the rook on a file less than the king.
        Square rookFrom = default(Square);
        if (move.Type == Move.MoveType.MoveCastleShort)
        {
          int bestFile = -1;
          for (int i = 0; i < 64; i++)
          {
            Square sq = new Square(i);
            Piece p = this.PieceOnSquare(sq);
            if (p.Type == PieceType.Rook && p.Side == this.MiscInfo.SideToMove && sq.File > kingSquare.File)
            {
              if (sq.File > bestFile)
              {
                bestFile = sq.File;
                rookFrom = sq;
              }
            }
          }
          if (bestFile == -1)
          {
            throw new Exception("Kingside rook not found for castling.");
          }
        }
        else
        {
          int bestFile = 8;
          for (int i = 0; i < 64; i++)
          {
            Square sq = new Square(i);
            Piece p = this.PieceOnSquare(sq);
            if (p.Type == PieceType.Rook && p.Side == this.MiscInfo.SideToMove && sq.File < kingSquare.File)
            {
              if (sq.File < bestFile)
              {
                bestFile = sq.File;
                rookFrom = sq;
              }
            }
          }

          if (bestFile == 8)
          {
            throw new Exception("Queenside rook not found for castling.");
          }
        }

        // Move the king.
        SetPieceOnSquare(kingSquare.SquareIndexStartA1, new Piece(this.MiscInfo.SideToMove, PieceType.None));
        SetPieceOnSquare(kingDest.SquareIndexStartA1, new Piece(this.MiscInfo.SideToMove, PieceType.King));

        // Move the rook.
        Piece rookPiece = this.PieceOnSquare(rookFrom);
        SetPieceOnSquare(rookFrom.SquareIndexStartA1, new Piece(rookPiece.Side, PieceType.None));
        SetPieceOnSquare(rookDest.SquareIndexStartA1, new Piece(rookPiece.Side, PieceType.Rook));

        // Remove castling rights for the moving side.
        if (this.MiscInfo.SideToMove == SideType.White)
        {
          newWhiteCanOO = false;
          newWhiteCanOOO = false;
        }
        else
        {
          newBlackCanOO = false;
          newBlackCanOOO = false;
        }

        // Clear en passant.
        newEnPassant = PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
      }
      else
      {
        // --- REGULAR MOVE (including promotion and en passant) ---
        Piece targetPiece = this.PieceOnSquare(move.ToSquare);
        if (targetPiece.Type != PieceType.None)
        {
          isCapture = true;
        }
        // Handle en passant capture: pawn moves diagonally into an empty square.
        if (movingPiece.Type == PieceType.Pawn &&
            move.FromSquare.File != move.ToSquare.File &&
            targetPiece.Type == PieceType.None)
        {
          isCapture = true;
          int captureRank = (movingPiece.Side == SideType.White) ? move.ToSquare.Rank - 1 : move.ToSquare.Rank + 1;
          Square epSquare = Square.FromFileAndRank(move.ToSquare.File, captureRank);
          SetPieceOnSquare(epSquare.SquareIndexStartA1, new Piece(
              (movingPiece.Side == SideType.White) ? SideType.Black : SideType.White, PieceType.None));
        }

        // Remove the moving piece from its source.
        SetPieceOnSquare(move.FromSquare.SquareIndexStartA1, new Piece(movingPiece.Side, PieceType.None));

        // Handle promotion.
        Piece pieceToPlace = movingPiece;
        if (movingPiece.Type == PieceType.Pawn && move.PromoteTo != PieceType.None)
        {
          pieceToPlace = new Piece(movingPiece.Side, move.PromoteTo);
        }

        // Place the (possibly promoted) piece on the destination.
        SetPieceOnSquare(move.ToSquare.SquareIndexStartA1, pieceToPlace);

        // Update en passant: if a pawn moves two squares forward.
        int rankDiff = move.ToSquare.Rank - move.FromSquare.Rank;
        if (movingPiece.Type == PieceType.Pawn && Math.Abs(rankDiff) == 2)
        {
          newEnPassant = (EnPassantFileIndexEnum)move.FromSquare.File;
        }
        else
        {
          newEnPassant = PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
        }

        // Update castling rights if a king or rook moved.
        if (movingPiece.Type == PieceType.King)
        {
          if (movingPiece.Side == SideType.White)
          {
            newWhiteCanOO = false;
            newWhiteCanOOO = false;
          }
          else
          {
            newBlackCanOO = false;
            newBlackCanOOO = false;
          }
        }
        else if (movingPiece.Type == PieceType.Rook)
        {
          // Standard chess: white rooks start at A1 (file 0) and H1 (file 7),
          // black rooks at A8 (file 0) and H8 (file 7).
          if (movingPiece.Side == SideType.White)
          {
            if (move.FromSquare.File == 7 && move.FromSquare.Rank == 0)
              newWhiteCanOO = false;
            else if (move.FromSquare.File == 0 && move.FromSquare.Rank == 0)
              newWhiteCanOOO = false;
          }
          else
          {
            if (move.FromSquare.File == 7 && move.FromSquare.Rank == 7)
              newBlackCanOO = false;
            else if (move.FromSquare.File == 0 && move.FromSquare.Rank == 7)
              newBlackCanOOO = false;
          }
        }
        // If a rook is captured, update opponent's castling rights.
        if (isCapture && targetPiece.Type == PieceType.Rook)
        {
          if (targetPiece.Side == SideType.White)
          {
            if (move.ToSquare.File == 7 && move.ToSquare.Rank == 0)
            {
              newWhiteCanOO = false;
            }
            else if (move.ToSquare.File == 0 && move.ToSquare.Rank == 0)
            {
              newWhiteCanOOO = false;
            }
          }
          else
          {
            if (move.ToSquare.File == 7 && move.ToSquare.Rank == 7)
            {
              newBlackCanOO = false;
            }
            else if (move.ToSquare.File == 0 && move.ToSquare.Rank == 7)
            {
              newBlackCanOOO = false;
            }
          }
        }
      }

      // Toggle the side to move.
      newSide = (this.MiscInfo.SideToMove == SideType.White) ? SideType.Black : SideType.White;

      // Move number stored is actually ply number; always increment it
      newMoveNum = this.MiscInfo.MoveNum + 1;

      // Reset the 50-move counter if the move is a pawn move, a capture, or a castling move.
      if (pawnMoved || isCapture)
      {
        newMove50Count = 0;
      }
      else
      {
        newMove50Count = this.MiscInfo.Move50Count + 1;
      }

      // Construct a new, immutable PositionMiscInfo.
      PositionMiscInfo updatedMiscInfo = new PositionMiscInfo(
          newWhiteCanOO,
          newWhiteCanOOO,
          newBlackCanOO,
          newBlackCanOOO,
          newSide,
          newMove50Count,
          newRepetitionCount,
          newMoveNum,
          newEnPassant
      );
      SetMiscInfo(updatedMiscInfo);

      // Recalculate the piece count.
      int newPieceCount = 0;
      for (int i = 0; i < 64; i++)
      {
        Square sq = new Square(i);
        if (PieceOnSquare(sq).Type != PieceType.None)
        {
          newPieceCount++;
        }
      }
      SetPieceCount((byte)newPieceCount);
      SetShortHash();
    }


    /// <summary>
    // Helper method to locate the king of a given side.
    /// </summary>
    /// <param name="side"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    private readonly Square FindKing(SideType side)
    {
      for (int i = 0; i < 64; i++)
      {
        Square sq = new Square(i);
        Piece p = PieceOnSquare(sq);
        if (p.Type == PieceType.King && p.Side == side)
        {
          return sq;
        }
      }
      throw new Exception("King not found on the board.");
    }

    #endregion


    #region Equality 

    /// <summary>
    /// Equals operator.
    /// </summary>
    /// <param name="ths"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static bool operator ==(Position ths, Position other) => ths.Equals(other);


    /// <summary>
    /// Not equals operator.
    /// </summary>
    /// <param name="ths"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static bool operator !=(Position ths, Position other) => !ths.Equals(other);

    /// <summary>
    /// Returns new Position being the reverse of this.
    /// </summary>
    public Position Reversed => new Position(this, true, true);


    /// <summary>
    ///  Returns if two positions are equal (in all respects).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public bool Equals(Position other)
    {
      // Quick first check
      if (PiecesShortHash != other.PiecesShortHash)
      {
        return false;
      }

      if (!PiecesEqual(other))
      {
        return false;
      }

      if (!MiscInfo.Equals(other.MiscInfo))
      {
        return false;
      }

      return true;
    }

    #endregion

    #region Initialization

    [ModuleInitializer]
    internal static void Init()
    {
      InitFastHashTable();
    }

    #endregion
  }
}
