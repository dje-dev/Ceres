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
using static Ceres.Chess.MoveGen.MGPositionConstants;

using Ceres.Chess.EncodedPositions.Basic;
using System.Runtime.CompilerServices;
using Ceres.Chess.EncodedPositions;

#endregion


namespace Ceres.Chess.MoveGen.Converters
{
  /// <summary>
  /// Static methods to facilitate conversion between MGMove and EncodedMove.
  /// </summary>
  public static class ConverterMGMoveEncodedMove
  {
    static readonly PieceType[] PIECE_CONV = new[] { PieceType.None, PieceType.Pawn, PieceType.Bishop, PieceType.None,
                                       PieceType.Rook, PieceType.Knight, PieceType.Queen, PieceType.King,
                                       PieceType.None,
                                       PieceType.None, PieceType.Pawn, PieceType.Bishop, PieceType.None,
                                       PieceType.Rook, PieceType.Knight, PieceType.Queen, PieceType.King
                                      };


    static byte CalcMGSquareFromLC0Square(byte square) => (byte)(square ^ 0b111); // same as (byte)(8 * (square / 8) + (7 - (square % 8)))

    static byte CalcLC0SquareFromMGSquare(byte square) => CalcMGSquareFromLC0Square(square); // symmetric

    /// Unique index packed [0...16383] computed by bitwise concatenation of from/to squares and promotion code
    /// </summary>
    //private readonly ushort rawValue;
    [StructLayout(LayoutKind.Explicit, Pack = 1)]
    public struct FromTo
    {
      [FieldOffset(0)]
      public readonly byte From;

      [FieldOffset(1)]
      public readonly byte To;

      /// <summary>
      /// Alias of first two bytes together (used in some performance sensitive code to do quick compare)
      /// </summary>
      [FieldOffset(0)]
      internal short FromAndToCombined;

      public FromTo(byte from, byte to)
      {
        From = from;
        To = to;

        // Suppress C# definite assignment rule
        unsafe { fixed (void* fromAndToPtr = &FromAndToCombined) { } }
      }
    }

    static readonly MCChessPositionPieceEnum[] whitePieceToMGPieceCode
  = new MCChessPositionPieceEnum[] { 0, MCChessPositionPieceEnum.WhitePawn, MCChessPositionPieceEnum.WhiteKnight, MCChessPositionPieceEnum.WhiteBishop,
                                            MCChessPositionPieceEnum.WhiteRook, MCChessPositionPieceEnum.WhiteQueen, MCChessPositionPieceEnum.WhiteKing };
    static readonly MCChessPositionPieceEnum[] blackPieceToMGPieceCode
      = new MCChessPositionPieceEnum[] { 0, MCChessPositionPieceEnum.BlackPawn, MCChessPositionPieceEnum.BlackKnight, MCChessPositionPieceEnum.BlackBishop,
                                            MCChessPositionPieceEnum.BlackRook, MCChessPositionPieceEnum.BlackQueen, MCChessPositionPieceEnum.BlackKing };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static FromTo CalcFromTo(int i)
    {
      EncodedMove lzMove = new EncodedMove((ushort)i);

      return new ConverterMGMoveEncodedMove.FromTo(
                  (byte)(((lzMove.RawValue & EncodedMove.MaskFrom) >> 6) ^ 0b111),
                  (byte)((lzMove.RawValue & EncodedMove.MaskTo) ^ 0b111));

      // The above code is an unrolled (much faster) version than the following:
      // return new FromTo(CalcLZSquareFromMGSquare(lzMove.FromSquare.AsByte),
      //                   CalcLZSquareFromMGSquare(lzMove.ToSquare.AsByte));
    }

    public static FromTo EncodedMoveToMGChessMoveFromTo(EncodedMove thisMove, bool flipped)
    {
      if (flipped)
      {
        FromTo raw = CalcFromTo(thisMove.RawValue & (16384 - 1));
        return new FromTo(MGMove.FlipSquare((byte)raw.From), MGMove.FlipSquare((byte)raw.To));
      }
      else
        return CalcFromTo(thisMove.RawValue & (16384 - 1));
    }


    public static MGMove EncodedMoveToMGChessMove(EncodedMove thisMove, in MGPosition position, 
                                                      bool setFlags = true)
    {
      MGMove moveMG = DoEncodedMoveToMGChessMove(thisMove, in position, setFlags);
      if (position.SideToMove == SideType.Black)
      {
        moveMG = moveMG.Reversed;
      }

      return moveMG;
    }

    /// <summary>
    /// 
    /// Speed: approximately 70,000,000 per second
    /// </summary>
    /// <param name="thisMove"></param>
    /// <param name="position"></param>
    /// <param name="setFlags">if the flags should be computed and stored (this is expensive)</param>
    /// <returns></returns>
    static MGMove DoEncodedMoveToMGChessMove(EncodedMove thisMove, in MGPosition position, bool setFlags = true)
    {
      Debug.Assert(position != default);

      FromTo fromTo = CalcFromTo(thisMove.RawValue & (16384 - 1));
      byte fromSquare = fromTo.From;
      byte toSquare = fromTo.To;

      // All done if we don't have to set the flags
      if (!setFlags)
      {
        return new MGMove(fromSquare, toSquare, MGMove.MGChessMoveFlags.None);
      }

      PieceType pieceMoving = position.PieceMoving(thisMove);
      PieceType pieceCapture = position.PieceCapturing(thisMove);

      MCChessPositionPieceEnum rawPieceAtToSquare = position.PieceCapturingRaw(thisMove);

      bool blackToMove = position.BlackToMove;
      MCChessPositionPieceEnum pieceMG = blackToMove ? blackPieceToMGPieceCode[(int)pieceMoving] : whitePieceToMGPieceCode[(int)pieceMoving];
      int pieceMGFlags = (int)pieceMG << MGMove.PIECE_SHIFT;
      int captureFlag = 0;
      if (pieceCapture != PieceType.None)
        {
        captureFlag = (int)MGMove.MGChessMoveFlags.Capture;
      }
      else if (pieceMoving == PieceType.Pawn && (rawPieceAtToSquare == MCChessPositionPieceEnum.WhiteEnPassant || rawPieceAtToSquare == MCChessPositionPieceEnum.BlackEnPassant))
      {
        captureFlag = (int)MGMove.MGChessMoveFlags.EnPassantCapture;
      }

      bool isMovingToRank8 = thisMove.ToSquare.IsRank8;
      if (isMovingToRank8 && pieceMoving == PieceType.Pawn)
      {
        int promotionPart = thisMove.Promotion switch
        {
          // In the special of Knight promotions, we can't count on the LZPositionMove flags for Promotion already reflecting this
          // because the NN index used for ordinary move from 7th to 8th rank is the same with and without promotion
          EncodedMove.PromotionType.None => (int)MGMove.MGChessMoveFlags.PromoteKnight,

          EncodedMove.PromotionType.Knight => (int)MGMove.MGChessMoveFlags.PromoteKnight, // note: possibly this will never happen, Knight promotion not encoded
          EncodedMove.PromotionType.Bishop => (int)MGMove.MGChessMoveFlags.PromoteBishop,
          EncodedMove.PromotionType.Rook => (int)MGMove.MGChessMoveFlags.PromoteRook,
          EncodedMove.PromotionType.Queen => (int)MGMove.MGChessMoveFlags.PromoteQueen,
          _ => 0
        };

        Debug.Assert(promotionPart != 0);
        return new MGMove(fromSquare, toSquare, (MGMove.MGChessMoveFlags)(promotionPart | pieceMGFlags | captureFlag));
      }
      else
      {
        bool isCastling = false;

        int thisNNIndex = thisMove.IndexNeuralNet;
        bool isCastlingFromAndToSquares = thisNNIndex == 103 || thisNNIndex == 97;
        if (isCastlingFromAndToSquares)
        {
          isCastling = (pieceMoving == PieceType.King);
        }

        if (isCastling)
        {
          int castlingPart;
          if (toSquare <= 4)
          {
            castlingPart = (int)MGMove.MGChessMoveFlags.CastleShort;
            toSquare = 1;
          }
          else
          {
            castlingPart = (int)MGMove.MGChessMoveFlags.CastleLong;
            toSquare = 5;
          }

          return new MGMove(fromSquare, toSquare, (MGMove.MGChessMoveFlags)(castlingPart | pieceMGFlags));
        }

        // If the LZPositionMove happens to have the castling flag set,  make sure we agree
        Debug.Assert(!(thisMove.IsCastling && !isCastling));

        // Check for double pawn move
        bool isDoublePawnMove = thisMove.FromSquare.IsRank2 && thisMove.ToSquare.IsRank4 && pieceMoving == PieceType.Pawn;
        MGMove.MGChessMoveFlags flagsDefault = isDoublePawnMove ? MGMove.MGChessMoveFlags.DoublePawnMove : 0;

        return new MGMove(fromSquare, toSquare, (MGMove.MGChessMoveFlags)((int)flagsDefault | pieceMGFlags | captureFlag));
      }
    }


    static Square[] squareMap;

    static void InitializeStatics()
    {
      if (squareMap == null)
      {
        Square[] temp = new Square[64];
        for (int i = 0; i < 64; i++)
        {
          temp[i] = Square.FromFileAndRank(7 - (i % 8), i / 8);
        }
        squareMap = temp;
      }

    }

    static readonly EncodedMove moveCastle = new EncodedMove("E1", "H1", EncodedMove.PromotionType.None, true);
    static readonly EncodedMove moveCastleLong = new EncodedMove("E1", "A1", EncodedMove.PromotionType.None, true);


    public static EncodedMove MGChessMoveToEncodedMove(MGMove thisMove)
    {
      // LZPositionMove always from perspective of white to move
      if (thisMove.BlackToMove)
      {
        thisMove = thisMove.Reversed;
      }
      return MGChessMoveToEncodedMoveWhite(thisMove);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static EncodedMove MGChessMoveToEncodedMoveBlack(MGMove thisMove)
    {
      Debug.Assert(thisMove.BlackToMove);
      return MGChessMoveToEncodedMoveWhite(thisMove.Reversed);
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static EncodedMove MGChessMoveToEncodedMoveWhite(MGMove thisMove)
    {
      Debug.Assert(!thisMove.BlackToMove);

      EncodedMove.PromotionType promotion = EncodedMove.PromotionType.None;
      if (thisMove.IsCastleOrPromotion)
      {
        if (thisMove.CastleShort)
        {
          return moveCastle;
        }
        else if (thisMove.CastleLong)
        {
          return moveCastleLong;
        }
        else if (thisMove.IsPromotion)
        {
          if (thisMove.PromoteQueen)
          {
            promotion = EncodedMove.PromotionType.Queen;
          }
          else if (thisMove.PromoteRook)
          {
            promotion = EncodedMove.PromotionType.Rook;
          }
          else if (thisMove.PromoteKnight)
          {
            promotion = EncodedMove.PromotionType.Knight;
          }
          else if (thisMove.PromoteBishop)
          {
            promotion = EncodedMove.PromotionType.Bishop;
          }
          else
            throw new Exception("Internal error, unknown promotion type");
        }
      }

      Square squareFrom = squareMap[thisMove.FromSquareIndex];
      Square squareTo = squareMap[thisMove.ToSquareIndex];
      return new EncodedMove(new EncodedSquare(squareFrom.SquareIndexStartA1), 
                             new EncodedSquare(squareTo.SquareIndexStartA1), promotion, false);
    }

    #region Table

    static int MGInfoToTableIndex(byte from, byte to, int promoIndex) => (promoIndex * 64 * 64 * 5) + (from * 64) + to;

    static int[] cachedLC0NNIndexToMGIndex;

    public static int[] LC0NNIndexToMGIndex
    {
      get
      {
        if (cachedLC0NNIndexToMGIndex == null)
        {
          cachedLC0NNIndexToMGIndex = BuildLC0NNIndexToMGIndexTable();
        }

        return cachedLC0NNIndexToMGIndex;
      }
    }

    static int[] BuildLC0NNIndexToMGIndexTable()
    {
      int[] table = new int[EncodedPolicyVector.POLICY_VECTOR_LENGTH];
      int numAdded = 0;

      for (byte from = 0; from < 64; from++)
      {
        for (byte to = 0; to < 64; to++)
        {
          int rankFrom = from / 8;
          int rankTo = to / 8;
          for (int promo = 0; promo < 5; promo++)
          {
            if (from != to && promo != 1) // We skip 0 (no promotion) and also 1 (promote to knight, because this is the default)
            {
              // Promotions only encoded when moving from penultimate to last rank
              if (promo > 1 && ((rankFrom != 6 || rankTo != 7))) continue;

              MGMove.MGChessMoveFlags flags = (MGMove.MGChessMoveFlags)(1 << (6 + promo));

              MGMove moveMG = new MGMove(from, to, flags);
              EncodedMove moveLZ = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(moveMG);
              int indexNN = moveLZ.IndexNeuralNet;
              if (indexNN != -1)
              {
                if (table[indexNN] != 0)
                {
                  throw new Exception("already set");
                }
                table[indexNN] = MGInfoToTableIndex(from, to, promo);
                numAdded++;
              }
            }
          }
        }
      }

      Debug.Assert(numAdded == EncodedPolicyVector.POLICY_VECTOR_LENGTH);
      Console.WriteLine("Successful BuildLZNNIndexToMGIndexTable, but NOT CURRENTLY USED");
      return table;
    }

    [ModuleInitializer]
    internal static void ClassInitialize()
    {
      InitializeStatics();
    }

    #endregion
  }

}
