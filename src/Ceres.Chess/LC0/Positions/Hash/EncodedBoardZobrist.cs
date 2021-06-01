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

using Ceres.Base;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Chess.LC0.Boards;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Manages generation of Zobrist hashes for boards.
  /// </summary>
  public static class EncodedBoardZobrist
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void ApplyZobrist(ulong[] keys, BitVector64 bv, ref ulong hash)
    {
        long bv1 = bv.Data;
        while (bv1 != 0)
        {
          long lsbMask = bv1 & -bv1;
          hash ^= keys[BitOperations.TrailingZeroCount(bv1)];
          bv1 ^= lsbMask;
        }
    }


    /// <summary>
    /// Arrays of Zobrist Keys:
    ///   - first index is 0 or 1 (White or Black)
    ///   - second index is the piece type index
    ///   - third index holds actual Zobrist keys
    /// </summary>
    static readonly ulong[][][] keys = new ulong[2][][];
    static byte FlipSquare(byte b) => (byte)((7 - b / 8) * 8 + b % 8);

    internal static ulong[][][] Keys => keys;

    /// <summary>
    /// 
    /// NOTE: This method was replaced by a version in Position class which combines together 
    /// logic for position interpretation and hashing logic (to improve performance).

    /// This version is retained for testing purposes.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static ulong ZobristHashSlow(in Position pos, PositionMiscInfo.HashMove50Mode hashMode)
    {
      Span<(Piece, Square)> arrayPieces = stackalloc (Piece, Square)[32];
      arrayPieces = pos.GetPiecesOnSquares(arrayPieces);

      ulong hash = 0;
      foreach ((Piece, Square) kvp in arrayPieces)
      {
        int squareIndex =  kvp.Item2.SquareIndexStartA1;
        hash ^= keys[(int)kvp.Item1.Side][(int)kvp.Item1.Type][squareIndex];
      }

      hash ^= (ulong)pos.MiscInfo.HashPosition(hashMode);

      return hash;
    }


    /// <summary>
    /// 
    /// NOTE: Our convention is that the hashes are applied in reverse order
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="hashMode"></param>
    /// <returns></returns>
    public static ulong ZobristHash(Span<Position> positions, PositionMiscInfo.HashMove50Mode hashMode)
    {
      ulong hash = 0;
      for (int i = positions.Length - 1; i>=0; i--)
          hash ^= positions[i].CalcZobristHash(hashMode) << i;

      return hash;
    }



    public static ulong ZobristHash(in EncodedPositionBoard board)
    {
      // TO DO: Optimize, for example only one bit can possibly be set for Kings
      ulong hash = 0;

      ApplyZobrist(OurPawnSquareKeys, board.OurPawns.Bits, ref hash);
      ApplyZobrist(OurKnightSquareKeys, board.OurKnights.Bits, ref hash);
      ApplyZobrist(OurBishopSquareKeys, board.OurBishops.Bits, ref hash);
      ApplyZobrist(OurRookSquareKeys, board.OurRooks.Bits, ref hash);
      ApplyZobrist(OurQueenSquareKeys, board.OurQueens.Bits, ref hash);
      ApplyZobrist(OurKingSquareKeys, board.OurKing.Bits, ref hash);

      ApplyZobrist(TheirPawnSquareKeys, board.TheirPawns.Bits, ref hash);
      ApplyZobrist(TheirKnightSquareKeys, board.TheirKnights.Bits, ref hash);
      ApplyZobrist(TheirBishopSquareKeys, board.TheirBishops.Bits, ref hash);
      ApplyZobrist(TheirRookSquareKeys, board.TheirRooks.Bits, ref hash);
      ApplyZobrist(TheirQueenSquareKeys, board.TheirQueens.Bits, ref hash);
      ApplyZobrist(TheirKingSquareKeys, board.TheirKing.Bits, ref hash);

      if (board.Repetitions.Data > 0) hash ^= EncodedBoardZobrist.RepetitionsKey;
      return hash;
    }

    #region Generating random keys

    static readonly ulong[] OurPawnSquareKeys;
    static readonly ulong[] OurKnightSquareKeys;
    static readonly ulong[] OurBishopSquareKeys;
    static readonly ulong[] OurRookSquareKeys;
    static readonly ulong[] OurQueenSquareKeys;
    static readonly ulong[] OurKingSquareKeys;

    static readonly ulong[] TheirPawnSquareKeys;
    static readonly ulong[] TheirKnightSquareKeys;
    static readonly ulong[] TheirBishopSquareKeys;
    static readonly ulong[] TheirRookSquareKeys;
    static readonly ulong[] TheirQueenSquareKeys;
    static readonly ulong[] TheirKingSquareKeys;

    internal readonly static ulong RepetitionsKey;

    const int ZOBRIST_RAND_SEED = 784_759_373; // Need to use fixed seed to keep hashes stable with persisted data structures
    static readonly Random rand = new Random(ZOBRIST_RAND_SEED);

    static readonly byte[] bytes = new byte[sizeof(ulong)];

    static ulong RandULong()
    {
      rand.NextBytes(bytes);

      return (ulong)BitConverter.ToInt64(bytes);
    }

    static void FillRand(ref ulong[] array)
    {
      array = new ulong[64];
      for (int i = 0; i < 64; i++)
      {
        array[i] = RandULong();
      }
    }

    static EncodedBoardZobrist()
    {
      FillRand(ref OurPawnSquareKeys);
      FillRand(ref OurKnightSquareKeys);
      FillRand(ref OurBishopSquareKeys);
      FillRand(ref OurRookSquareKeys);
      FillRand(ref OurQueenSquareKeys);
      FillRand(ref OurKingSquareKeys);

      FillRand(ref TheirPawnSquareKeys);
      FillRand(ref TheirKnightSquareKeys);
      FillRand(ref TheirBishopSquareKeys);
      FillRand(ref TheirRookSquareKeys);
      FillRand(ref TheirQueenSquareKeys);
      FillRand(ref TheirKingSquareKeys);

      RepetitionsKey = RandULong();

      keys[0] = new ulong[][] { null, OurPawnSquareKeys, OurKnightSquareKeys, OurBishopSquareKeys, OurRookSquareKeys, OurQueenSquareKeys, OurKingSquareKeys };
      keys[1] = new ulong[][] { null, TheirPawnSquareKeys, TheirKnightSquareKeys, TheirBishopSquareKeys, TheirRookSquareKeys, TheirQueenSquareKeys, TheirKingSquareKeys };
    }

    #endregion

  }

}
