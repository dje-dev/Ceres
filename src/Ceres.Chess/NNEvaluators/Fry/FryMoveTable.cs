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

using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.NNEvaluators.Fry
{
  /// <summary>
  /// The Fry network's 1968-entry policy move table (a port of model/encoding.py _build_all_possible_moves,
  /// whose order fixes the policy indices): for each from-square a1..h8, every queen and knight destination
  /// on an empty board in ascending square order (1792 moves); then promotions from rank 7 to 8 and from
  /// rank 2 to 1, per from-file, to-file and piece in the order knight, rook, bishop, queen (176 moves).
  /// Moves are in the side-to-move frame (Black's squares mirrored by square ^ 56), castling is the king's
  /// two-square move, and queen promotions are explicit.
  /// </summary>
  public static class FryMoveTable
  {
    public const int NUM_MOVES = 1968;

    // Promotion codes in the index: 0 none, 1 knight, 2 bishop, 3 rook, 4 queen.
    const int NUM_PROMO_CODES = 5;
    static readonly char[] PROMO_CHARS = [' ', 'n', 'b', 'r', 'q'];

    /// <summary>
    /// Policy index by (from, to, promotion code), -1 if the move is not in the table.
    /// </summary>
    static readonly short[] indexByMove;

    /// <summary>
    /// The table's moves as UCI strings in the side-to-move frame (as moves.json of the Fry explorer).
    /// </summary>
    public static readonly string[] MovesUCI;


    static FryMoveTable()
    {
      List<(int from, int to, int promo)> moves = new(NUM_MOVES);

      for (int from = 0; from < 64; from++)
      {
        int fromFile = from % 8;
        int fromRank = from / 8;
        for (int to = 0; to < 64; to++) // ascending, as python-chess iterates a SquareSet
        {
          if (to == from)
          {
            continue;
          }
          int df = to % 8 - fromFile;
          int dr = to / 8 - fromRank;
          bool queenMove = df == 0 || dr == 0 || Math.Abs(df) == Math.Abs(dr);
          bool knightMove = (Math.Abs(df) == 1 && Math.Abs(dr) == 2) || (Math.Abs(df) == 2 && Math.Abs(dr) == 1);
          if (queenMove || knightMove)
          {
            moves.Add((from, to, 0));
          }
        }
      }

      int[] promoOrder = [1, 3, 2, 4]; // knight, rook, bishop, queen
      foreach ((int fromRank, int toRank) in new[] { (6, 7), (1, 0) })
      {
        for (int fromFile = 0; fromFile < 8; fromFile++)
        {
          for (int toFile = Math.Max(fromFile - 1, 0); toFile < Math.Min(fromFile + 2, 8); toFile++)
          {
            foreach (int promo in promoOrder)
            {
              moves.Add((fromRank * 8 + fromFile, toRank * 8 + toFile, promo));
            }
          }
        }
      }

      if (moves.Count != NUM_MOVES)
      {
        throw new Exception($"Internal error: Fry move table has {moves.Count} moves, expected {NUM_MOVES}");
      }

      indexByMove = new short[64 * 64 * NUM_PROMO_CODES];
      Array.Fill(indexByMove, (short)-1);
      MovesUCI = new string[NUM_MOVES];
      for (int i = 0; i < moves.Count; i++)
      {
        (int from, int to, int promo) = moves[i];
        indexByMove[(from * 64 + to) * NUM_PROMO_CODES + promo] = (short)i;
        MovesUCI[i] = SquareName(from) + SquareName(to) + (promo == 0 ? "" : PROMO_CHARS[promo].ToString());
      }
    }


    static string SquareName(int square) => $"{(char)('a' + square % 8)}{square / 8 + 1}";


    /// <summary>
    /// Returns the policy index of a legal move (any side to move), or -1 if not in the table
    /// (only possible for Chess960 castling).
    /// </summary>
    public static int IndexOf(MGMove move)
    {
      int from = move.FromSquare.SquareIndexStartA1;
      int to = move.ToSquare.SquareIndexStartA1;

      // Ceres encodes castling as king takes own rook; Fry as the king's two-square move.
      if (move.IsCastle)
      {
        to = (from & ~7) | (move.CastleShort ? 6 : 2);
      }

      if (move.BlackToMove)
      {
        from ^= 56;
        to ^= 56;
      }

      int promo = move.PromoteKnight ? 1
                : move.PromoteBishop ? 2
                : move.PromoteRook ? 3
                : move.PromoteQueen ? 4
                : 0;

      return indexByMove[(from * 64 + to) * NUM_PROMO_CODES + promo];
    }
  }
}
