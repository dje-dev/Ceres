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
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Representation of a move (as used by the move generator subsystem).
  /// </summary>
  /// 
  using BitBoard = System.UInt64;

  [Serializable]
  [StructLayout(LayoutKind.Explicit, Pack = 1, Size=4)]
  public partial struct MGMove
  {
    #region Field

    /// <summary>
    /// Square from which piece is moving (h1 is index 0)
    /// </summary>
    [FieldOffset(0)]
    public byte FromSquareIndex;

    /// <summary>
    /// Square to which piece is moving (h1 is index 0)
    /// </summary>
    [FieldOffset(1)]
    public byte ToSquareIndex;

    /// <summary>
    /// Flags indicating characteristics, such as if capture
    /// </summary>
    [FieldOffset(2)]
    public MGChessMoveFlags Flags;

    /// <summary>
    /// Alias of first two bytes together (used in some performance sensitive code to do quick compare)
    /// </summary>
    [FieldOffset(0)]
    internal short FromAndToCombined;

    [FieldOffset(0)]
    internal uint AllFieldsCombined;

    #endregion


    /// <summary>
    /// Returns the origin (from) square of the move.
    /// </summary>
    public Square FromSquare => new Square(FromSquareIndex, Square.SquareIndexType.BottomToTopRightToLeft);

    /// <summary>
    /// Returns the destination (to) square of the move.
    /// </summary>
    public Square ToSquare => new Square(ToSquareIndex, Square.SquareIndexType.BottomToTopRightToLeft);


    /// <summary>
    /// Constructor (with piece specified).
    /// </summary>
    /// <param name="from"></param>
    /// <param name="to"></param>
    /// <param name="piece"></param>
    /// <param name="flags">optional flags other than piece</param>
    public MGMove(byte from, byte to, MGPositionConstants.MCChessPositionPieceEnum piece, MGChessMoveFlags flags)
    {
      Debug.Assert(from < 64);
      Debug.Assert(to < 64);
      Debug.Assert(from != to);
      Debug.Assert((flags & MGChessMoveFlags.Piece) == 0); // not expected to have already set

      FromSquareIndex = from;
      ToSquareIndex = to;
      Flags = flags;

      // Obviate requirement of definite assignment to all fields
      Unsafe.SkipInit<short>(out FromAndToCombined);
      Unsafe.SkipInit<uint>(out AllFieldsCombined);

      Piece = piece;
  }


  /// <summary>
  /// Constructor
  /// </summary>
  /// <param name="from"></param>
  /// <param name="to"></param>
  /// <param name="flags"></param>
  public MGMove(byte from, byte to, MGChessMoveFlags flags)
    {
      Debug.Assert(from != to);
      Debug.Assert(from < 64);
      Debug.Assert(to < 64);

      FromSquareIndex = from;
      ToSquareIndex = to;
      Flags = flags;

      // Obviate requirement of definite assignment to all fields
      Unsafe.SkipInit<short>(out FromAndToCombined);
      Unsafe.SkipInit<uint>(out AllFieldsCombined);
    }


    /// <summary>
    /// Copy constructor.
    /// </summary>
    /// <param name="from"></param>
    /// <param name="to"></param>
    /// <param name="flags"></param>
    public MGMove(MGMove otherMove, bool flip)
    {
      if (!flip)
      {
        FromSquareIndex = otherMove.FromSquareIndex;
        ToSquareIndex = otherMove.ToSquareIndex;
        Flags = otherMove.Flags;
      }
      else
      {
        FromSquareIndex = FlipSquare(otherMove.FromSquareIndex);
        ToSquareIndex = FlipSquare(otherMove.ToSquareIndex);
        Flags = otherMove.Flags ^ MGChessMoveFlags.BlackToMove;
      }

      // Obviate requirement of definite assignment to all fields
      Unsafe.SkipInit<short>(out FromAndToCombined);
      Unsafe.SkipInit<uint>(out AllFieldsCombined);
    }


    internal const int PIECE_SHIFT = 12;
    internal const int MOVE_COUNT_SHIFT = 17;

    public const MGChessMoveFlags PromotionFlags = (MGChessMoveFlags.PromoteKnight | MGChessMoveFlags.PromoteBishop 
                                                 |  MGChessMoveFlags.PromoteRook   | MGChessMoveFlags.PromoteQueen);

    /// <summary>
    /// Returns a new MGMove initialized from a specified raw value.
    /// </summary>
    /// <param name="rawValue"></param>
    /// <returns></returns>
    public static MGMove FromRaw(uint rawValue) => new MGMove() { AllFieldsCombined = rawValue };


    /// <summary>
    /// Helper method to flip a square.
    /// </summary>
    /// <param name="b"></param>
    /// <returns></returns>
    internal static byte FlipSquare(byte b) => (byte)(b ^ 0b111_000 | b & 0b111);// equivalent to: (byte)((7 - b / 8) * 8 + b % 8);

    public MGMove Reversed => new MGMove(FlipSquare(FromSquareIndex), FlipSquare(ToSquareIndex), Flags ^= MGChessMoveFlags.BlackToMove);

    /// <summary>
    /// Returns if the move is null (uninitialized).
    /// </summary>
    public bool IsNull => FromSquareIndex == ToSquareIndex;

    static readonly short[] promoFlagsToEnum = new short[] { 0, 1, 2, -1, 3, -1, -1, -1, 4 };

    /// <summary>
    /// Returns a value (0, 1, 2, 4 or 8) corresonding to none, Knight, Bishop, Rook, or Queen.
    /// </summary>
    public short PromotionValue => promoFlagsToEnum[(short)((int)(Flags & PromotionFlags)>>7)];
    
    
    /// <summary>
    /// Returns if the move is a promotion.
    /// </summary>
    public bool IsPromotion => (Flags & PromotionFlags) != 0;


    /// <summary>
    /// Returns if white is to move.
    /// </summary>
    public bool WhiteToMove => !BlackToMove;


    /// <summary>
    /// Returns if black is to move.
    /// </summary>
    public bool BlackToMove
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get => (Flags & MGChessMoveFlags.BlackToMove) != 0;
      set { if (value) Flags |= MGChessMoveFlags.BlackToMove; else Flags &= ~MGChessMoveFlags.BlackToMove; }
    }


    /// <summary>
    /// Resets the move 50 counter.
    /// </summary>
    public bool ResetsMove50Count
    {
      get
      {
        // Update Rule 50 count (upon any pawn move or capture)
        return (Capture || (byte)Piece == MGPositionConstants.WPAWN || (byte)Piece == MGPositionConstants.BPAWN);
      }
    }

    /// <summary>
    /// Returns if the move is a check.
    /// </summary>
    public bool Check
    {
      get => (Flags & MGChessMoveFlags.Check) != 0;
      set { if (value) Flags |= MGChessMoveFlags.Check; else Flags &= ~MGChessMoveFlags.Check; }
    }

    /// <summary>
    /// Returns if the move is a capture.
    /// </summary>
    public bool Capture
    {
      get => (Flags & MGChessMoveFlags.Capture) != 0;
      set { if (value) Flags |= MGChessMoveFlags.Capture; else Flags &= ~MGChessMoveFlags.Capture; }
    }

    /// <summary>
    /// Returns if the move is an en passant capture.
    /// </summary>
    public bool EnPassantCapture
    {
      get => (Flags & MGChessMoveFlags.EnPassantCapture) != 0;
      set { if (value) Flags |= MGChessMoveFlags.EnPassantCapture; else Flags &= ~MGChessMoveFlags.EnPassantCapture; }
    }

    /// <summary>
    /// Returns if the move is a double square pawn move (second to fourth rank).
    /// </summary>
    public bool DoublePawnMove
    {
      get => (Flags & MGChessMoveFlags.DoublePawnMove) != 0;
      set { if (value) Flags |= MGChessMoveFlags.DoublePawnMove; else Flags &= ~MGChessMoveFlags.DoublePawnMove; }
    }

    /// <summary>
    /// Returns if the move is either a castle or promotion move.
    /// </summary>
    public bool IsCastleOrPromotion
      => (Flags & (MGChessMoveFlags.CastleShort 
                 | MGChessMoveFlags.CastleLong
                 | MGChessMoveFlags.PromoteQueen
                 | MGChessMoveFlags.PromoteRook
                 | MGChessMoveFlags.PromoteBishop
                 | MGChessMoveFlags.PromoteKnight)) != 0;

    /// <summary>
    /// Returns if the move is either a short or long castle.
    /// </summary>
    public bool IsCastle => (Flags & (MGChessMoveFlags.CastleShort | MGChessMoveFlags.CastleLong)) != 0;

    /// <summary>
    /// Returns if the move is a castle short move.
    /// </summary>
    public bool CastleShort
    {
      get => (Flags & MGChessMoveFlags.CastleShort) != 0;
      set { if (value) Flags |= MGChessMoveFlags.CastleShort; else Flags &= ~MGChessMoveFlags.CastleShort; }
    }

    /// <summary>
    /// Returns if the move is a castle long move.
    /// </summary>
    public bool CastleLong
    {
      get => (Flags & MGChessMoveFlags.CastleLong) != 0;
      set { if (value) Flags |= MGChessMoveFlags.CastleLong; else Flags &= ~MGChessMoveFlags.CastleLong; }
    }

    /// <summary>
    /// Returns if the move is an underpromotion to knight.
    /// </summary>
    public bool PromoteKnight
    {
      get => (Flags & MGChessMoveFlags.PromoteKnight) != 0;
      set { if (value) Flags |= MGChessMoveFlags.PromoteKnight; else Flags &= ~MGChessMoveFlags.PromoteKnight; }
    }

    /// <summary>
    /// Returns if the move is an underpromotion to bishop.
    /// </summary>
    public bool PromoteBishop
    {
      get => (Flags & MGChessMoveFlags.PromoteBishop) != 0;
      set { if (value) Flags |= MGChessMoveFlags.PromoteBishop; else Flags &= ~MGChessMoveFlags.PromoteBishop; }
    }

    /// <summary>
    /// Returns if the move is an underpromotion to rook.
    /// </summary>
    public bool PromoteRook
    {
      get => (Flags & MGChessMoveFlags.PromoteRook) != 0;
      set { if (value) Flags |= MGChessMoveFlags.PromoteRook; else Flags &= ~MGChessMoveFlags.PromoteRook; }
    }

    /// <summary>
    /// Returns if the move is an promotion to queen.
    /// </summary>
    public bool PromoteQueen
    {
      get => (Flags & MGChessMoveFlags.PromoteQueen) != 0;
      set { if (value) Flags |= MGChessMoveFlags.PromoteQueen; else Flags &= ~MGChessMoveFlags.PromoteQueen; }
    }

    /// <summary>
    /// NOTE: To save storage (bits), this is specially encoded as a combination of PromoteBishop and PromoteKnight
    ///       This flag is only used in one type of situation, and is only ever set to true.
    /// </summary>
    public bool IllegalMove
    {
      get => (PromoteBishop && PromoteKnight);
#if DEBUG
      set { if (value) { PromoteBishop = true; PromoteKnight = true; } else throw new NotImplementedException(); }
#else
      set { PromoteBishop = true; PromoteKnight = true; }
#endif
    }


    public bool NoMoreMoves
    {
      get => (Flags & MGChessMoveFlags.NoMoreMoves) != 0;
      set { if (value) Flags |= MGChessMoveFlags.NoMoreMoves; else Flags &= ~MGChessMoveFlags.NoMoreMoves; }
    }

    public MGPositionConstants.MCChessPositionPieceEnum Piece
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      { 
        return (MGPositionConstants.MCChessPositionPieceEnum)(((int)(Flags & MGChessMoveFlags.Piece)) >> PIECE_SHIFT); 
      }

      set
      {
        MGChessMoveFlags val = (Flags & ~(MGChessMoveFlags.Piece)); // clear bits from any prior value
        Flags = (MGChessMoveFlags)((int)val | ((int)value << PIECE_SHIFT)); // set bits from this new value
      }
    }

#region Overrides

    public override int GetHashCode()
    {
      return HashCode.Combine(FromSquareIndex, ToSquareIndex, Flags & PromotionFlags);
    }
    public override string ToString() => MoveStr(MGMoveNotationStyle.LongAlgebraic);

    public static bool operator ==(MGMove move1, MGMove move2) => move1.Equals(move2);    

    public static bool operator !=(MGMove move1, MGMove move2) => !move1.Equals(move2);

    public override bool Equals(object obj) => obj is MGMove && Equals((MGMove)obj);

#endregion

    /// <summary>
    /// Helper comparer class for MGMove which uses FromAndToCombined.
    /// </summary>
    internal class MGMoveComparerByFromAndTo : IComparer<MGMove>
    {
      public int Compare(MGMove x, MGMove y) => x.FromAndToCombined < y.FromAndToCombined ? 1 : (x.FromAndToCombined > y.FromAndToCombined ? -1 : 0);
    }


    /// <summary>
    /// Tests for equality with a Move object.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public bool EqualsMove(Move move)
    {
      if (move.Type == Move.MoveType.MoveNonCastle)
      {
        if (FromSquare == move.FromSquare && ToSquare == move.ToSquare)
        {
          if (move.PromoteTo == PieceType.None) return true;

          if (PromoteQueen && (move.PromoteTo == PieceType.Queen)) return true;
          if (PromoteRook && (move.PromoteTo == PieceType.Rook)) return true;
          if (PromoteKnight && (move.PromoteTo == PieceType.Knight)) return true;
          if (PromoteBishop && (move.PromoteTo == PieceType.Bishop)) return true;
        }
      }
      else if (CastleLong && move.Type == Move.MoveType.MoveCastleLong)
      {
        return true;
      }
      else if (CastleShort && move.Type == Move.MoveType.MoveCastleShort)
      {
        return true;
      }
      
      return false;
    }

    /// <summary>
    /// Returns if two MGMoves are equal or not.
    /// 
    /// Note that we don't require exact equality on all the Flags.
    /// Two moves are equivalent if the from and to squares agree and the promotion type (if any) was the same.
    /// We can't test all the other flags for equality, since they are optionally populated because:
    ///   - possibly the move was translated from another source that did not provide full information (e.g. which piece being moved), or
    ///   - possibly the code did not bother to fully track and populate these flags because they were not helpful for the computation being done
    /// </summary>
    /// <param name="m"></param>
    /// <returns></returns>
    public bool Equals(MGMove m) =>
                               FromSquareIndex == m.FromSquareIndex
                            && ToSquareIndex == m.ToSquareIndex
                            && ((Flags & PromotionFlags) == (m.Flags & PromotionFlags));


  }

}