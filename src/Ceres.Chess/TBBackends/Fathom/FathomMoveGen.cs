#region License notice

// NOTE: This file is substantially a transliteration from C to C# 
//       of code from the Fathom project.
//       Both Fathom and Ceres copyrights are included below.

/*
Copyright (c) 2015 basil00
Modifications Copyright (c) 2016-2020 by Jon Dart
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
using System.Numerics;

#endregion

namespace Ceres.Chess.TBBackends.Fathom
{
  internal static unsafe class FathomMoveGen
  {
    public const int TB_PIECES = 7;
    public enum WDL { WDL, DTM, DTZ };


    internal const int TB_MAX_MOVES = (192 + 1); 

    const int TB_PROMOTES_NONE = 0;
    const int TB_PROMOTES_QUEEN = 1;
    const int TB_PROMOTES_ROOK = 2;
    const int TB_PROMOTES_BISHOP = 3;
    const int TB_PROMOTES_KNIGHT = 4;

    // ==========

    static bool initialized = false;
    public static void Init()
    {
      if (!initialized)
      {
        king_attacks_init();
        knight_attacks_init();
        bishop_attacks_init();
        rook_attacks_init();
        pawn_attacks_init();
        initialized = true;
      }

    }

    const int TB_PAWN = 1;
    const int TB_KNIGHT = 2;
    const int TB_BISHOP = 3;
    const int TB_ROOK = 4;
    const int TB_QUEEN = 5;
    const int TB_KING = 6;

    const int TB_WPAWN = TB_PAWN;
    const int TB_BPAWN = (TB_PAWN | 8);

    const int WHITE_KING = (TB_WPAWN + 5);
    const int WHITE_QUEEN = (TB_WPAWN + 4);
    const int WHITE_ROOK = (TB_WPAWN + 3);
    const int WHITE_BISHOP = (TB_WPAWN + 2);
    const int WHITE_KNIGHT = (TB_WPAWN + 1);
    const int WHITE_PAWN = TB_WPAWN;
    const int BLACK_KING = (TB_BPAWN + 5);
    const int BLACK_QUEEN = (TB_BPAWN + 4);
    const int BLACK_ROOK = (TB_BPAWN + 3);
    const int BLACK_BISHOP = (TB_BPAWN + 2);
    const int BLACK_KNIGHT = (TB_BPAWN + 1);
    const int BLACK_PAWN = TB_BPAWN;

    const ulong PRIME_WHITE_QUEEN = 11811845319353239651UL;
    const ulong PRIME_WHITE_ROOK = 10979190538029446137UL;
    const ulong PRIME_WHITE_BISHOP = 12311744257139811149UL;
    const ulong PRIME_WHITE_KNIGHT = 15202887380319082783UL;
    const ulong PRIME_WHITE_PAWN = 17008651141875982339UL;
    const ulong PRIME_BLACK_QUEEN = 15484752644942473553UL;
    const ulong PRIME_BLACK_ROOK = 18264461213049635989UL;
    const ulong PRIME_BLACK_BISHOP = 15394650811035483107UL;
    const ulong PRIME_BLACK_KNIGHT = 13469005675588064321UL;
    const ulong PRIME_BLACK_PAWN = 11695583624105689831UL;

    const ulong BOARD_RANK_EDGE = 0x8181818181818181UL;
    const ulong BOARD_FILE_EDGE = 0xFF000000000000FFUL;
    const ulong BOARD_EDGE = (BOARD_RANK_EDGE | BOARD_FILE_EDGE);
    const ulong BOARD_RANK_1 = 0x00000000000000FFUL;
    const ulong BOARD_FILE_A = 0x8080808080808080UL;

    const ulong KEY_KvK = 0;

    internal const int BEST_NONE = 0xFFFF;
    internal const int SCORE_ILLEGAL = 0x7FFF;

    // Note: WHITE, BLACK values are reverse of Stockfish
    internal enum FathomColor { BLACK, WHITE };
    internal enum FathomPieceType { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };
    internal enum FathomPiece
    {
      W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
      B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
    };

    internal static FathomColor ColorOfPiece(int piece)
    {
      //dje    return (Color)(!(piece >> 3));
      return (FathomColor)((piece >> 3) == 0 ? 1 : 0);
    }

    internal static FathomPieceType TypeOfPiece(int piece)
    {
      return (FathomPieceType)(piece & 7);
    }

    //  typedef int32_t Value;


    internal static ulong pieces_by_type(in FathomPos pos, FathomColor c, FathomPieceType p)
    {

      ulong mask = (c == FathomColor.WHITE) ? pos.white : pos.black;
      switch (p)
      {
        case FathomPieceType.PAWN:
          return pos.pawns & mask;
        case FathomPieceType.KNIGHT:
          return pos.knights & mask;
        case FathomPieceType.BISHOP:
          return pos.bishops & mask;
        case FathomPieceType.ROOK:
          return pos.rooks & mask;
        case FathomPieceType.QUEEN:
          return pos.queens & mask;
        case FathomPieceType.KING:
          return pos.kings & mask;
        default:
          throw new Exception("fathom.pieces_by_type: internal error, unknown piece type " + p);
      }
    }

    static string piece_to_char = " PNBRQK  pnbrqk";

    // map upper-case characters to piece types
    static FathomPieceType char_to_piece_type(char c)
    {
      for (int i = (int)FathomPieceType.PAWN; i <= (int)FathomPieceType.KING; i++)
        if (c == piece_to_char[i])
        {
          return (FathomPieceType)i;
        }
      return (FathomPieceType)0;
    }

    internal static int rank(int s) => ((s) >> 3);
    internal static int file(int s) => ((s) & 0x07);
    static ulong board(int s) => ((ulong)1 << (s));
    internal static int square(int r, int f) => (8 * (r) + (f));

    static ulong[] king_attacks_table = new ulong[64];

    internal static ulong king_attacks(int square) => king_attacks_table[(square)];

    static void king_attacks_init()
    {
      for (int s = 0; s < 64; s++)
      {
        int r = rank(s);
        int f = file(s);
        ulong b = 0;
        if (r != 0 && f != 0)
          b |= board(square(r - 1, f - 1));
        if (r != 0)
          b |= board(square(r - 1, f));
        if (r != 0 && f != 7)
          b |= board(square(r - 1, f + 1));
        if (f != 7)
          b |= board(square(r, f + 1));
        if (r != 7 && f != 7)
          b |= board(square(r + 1, f + 1));
        if (r != 7)
          b |= board(square(r + 1, f));
        if (r != 7 && f != 0)
          b |= board(square(r + 1, f - 1));
        if (f != 0)
          b |= board(square(r, f - 1));
        king_attacks_table[s] = b;
      }
    }




    static ulong[] knight_attacks_table = new ulong[64];

    internal static ulong knight_attacks(int square) => knight_attacks_table[(square)];

    static void knight_attacks_init()
    {
      for (int s = 0; s < 64; s++)
      {
        int r1, r = rank(s);
        int f1, f = file(s);
        ulong b = 0;
        r1 = r - 1; f1 = f - 2;
        if (r1 >= 0 && f1 >= 0)
          b |= board(square(r1, f1));
        r1 = r - 1; f1 = f + 2;
        if (r1 >= 0 && f1 <= 7)
          b |= board(square(r1, f1));
        r1 = r - 2; f1 = f - 1;
        if (r1 >= 0 && f1 >= 0)
          b |= board(square(r1, f1));
        r1 = r - 2; f1 = f + 1;
        if (r1 >= 0 && f1 <= 7)
          b |= board(square(r1, f1));
        r1 = r + 1; f1 = f - 2;
        if (r1 <= 7 && f1 >= 0)
          b |= board(square(r1, f1));
        r1 = r + 1; f1 = f + 2;
        if (r1 <= 7 && f1 <= 7)
          b |= board(square(r1, f1));
        r1 = r + 2; f1 = f - 1;
        if (r1 <= 7 && f1 >= 0)
          b |= board(square(r1, f1));
        r1 = r + 2; f1 = f + 1;
        if (r1 <= 7 && f1 <= 7)
          b |= board(square(r1, f1));
        knight_attacks_table[s] = b;
      }
    }

    static ulong[][] MakeAOA(int dim1, int dim2)
    {
      ulong[][] ret = new ulong[dim1][];
      for (int i = 0; i < dim1; i++)
        ret[i] = new ulong[dim2];
      return ret;
    }
    static ulong[][] diag_attacks_table = MakeAOA(64, 64);
    static ulong[][] anti_attacks_table = MakeAOA(64, 64);


    static int[] square2diag_table = new[]
    {
    0,  1,  2,  3,  4,  5,  6,  7,
    14, 0,  1,  2,  3,  4,  5,  6,
    13, 14, 0,  1,  2,  3,  4,  5,
    12, 13, 14, 0,  1,  2,  3,  4,
    11, 12, 13, 14, 0,  1,  2,  3,
    10, 11, 12, 13, 14, 0,  1,  2,
    9,  10, 11, 12, 13, 14, 0,  1,
    8,  9,  10, 11, 12, 13, 14, 0
};

    static int[] square2anti_table = new[]
    {
    8,  9,  10, 11, 12, 13, 14, 0,
    9,  10, 11, 12, 13, 14, 0,  1,
    10, 11, 12, 13, 14, 0,  1,  2,
    11, 12, 13, 14, 0,  1,  2,  3,
    12, 13, 14, 0,  1,  2,  3,  4,
    13, 14, 0,  1,  2,  3,  4,  5,
    14, 0,  1,  2,  3,  4,  5,  6,
    0,  1,  2,  3,  4,  5,  6,  7
};

    static ulong[] diag2board_table = new[]
    {
    0x8040201008040201UL,
    0x0080402010080402UL,
    0x0000804020100804UL,
    0x0000008040201008UL,
    0x0000000080402010UL,
    0x0000000000804020UL,
    0x0000000000008040UL,
    0x0000000000000080UL,
    0x0100000000000000UL,
    0x0201000000000000UL,
    0x0402010000000000UL,
    0x0804020100000000UL,
    0x1008040201000000UL,
    0x2010080402010000UL,
    0x4020100804020100UL,
};

    static ulong[] anti2board_table = new[]
    {
    0x0102040810204080UL,
    0x0204081020408000UL,
    0x0408102040800000UL,
    0x0810204080000000UL,
    0x1020408000000000UL,
    0x2040800000000000UL,
    0x4080000000000000UL,
    0x8000000000000000UL,
    0x0000000000000001UL,
    0x0000000000000102UL,
    0x0000000000010204UL,
    0x0000000001020408UL,
    0x0000000102040810UL,
    0x0000010204081020UL,
    0x0001020408102040UL,
};

    static ulong diag2index(ulong b)
    {
      b *= 0x0101010101010101UL;
      b >>= 56;
      b >>= 1;
      return (ulong)b;
    }

    static ulong anti2index(ulong b)
    {
      return diag2index(b);
    }

    static int diag(int s) => square2diag_table[(s)];
    static int anti(int s) => square2anti_table[(s)];
    static ulong diag2board(int d) => diag2board_table[(d)];
    static ulong anti2board(int a) => anti2board_table[(a)];

    internal static ulong bishop_attacks(int sq, ulong occ)
    {
      occ &= ~board(sq);
      int d = diag(sq), a = anti(sq);
      ulong d_occ = occ & (diag2board(d) & ~BOARD_EDGE);
      ulong a_occ = occ & (anti2board(a) & ~BOARD_EDGE);
      ulong d_idx = diag2index(d_occ);
      ulong a_idx = anti2index(a_occ);
      ulong d_attacks = diag_attacks_table[sq][d_idx];
      ulong a_attacks = anti_attacks_table[sq][a_idx];
      return d_attacks | a_attacks;
    }

    static void bishop_attacks_init()
    {
      for (int idx = 0; idx < 64; idx++)
      {
        int idx1 = idx << 1;
        for (int s = 0; s < 64; s++)
        {
          int r = rank(s);
          int f = file(s);
          ulong b = 0;
          for (int i = -1; f + i >= 0 && r + i >= 0; i--)
          {
            int occ = (1 << (f + i));
            b |= board(square(r + i, f + i));
            if ((idx1 & occ) != 0)
              break;
          }
          for (int i = 1; f + i <= 7 && r + i <= 7; i++)
          {
            int occ = (1 << (f + i));
            b |= board(square(r + i, f + i));
            if ((idx1 & occ) != 0)
              break;
          }
          diag_attacks_table[s][idx] = b;
        }
      }

      for (uint idx = 0; idx < 64; idx++)
      {
        uint idx1 = idx << 1;
        for (int s = 0; s < 64; s++)
        {
          int r = rank(s);
          int f = file(s);
          ulong b = 0;
          for (int i = -1; f + i >= 0 && r - i <= 7; i--)
          {
            int occ = (1 << (f + i));
            b |= board(square(r - i, f + i));
            if ((idx1 & occ) != 0)
              break;
          }
          for (int i = 1; f + i <= 7 && r - i >= 0; i++)
          {
            int occ = (1 << (f + i));
            b |= board(square(r - i, f + i));
            if ((idx1 & occ) != 0)
              break;
          }
          anti_attacks_table[s][idx] = b;
        }
      }
    }


    static ulong[][] rank_attacks_table = MakeAOA(64, 64);
    static ulong[][] file_attacks_table = MakeAOA(64, 64);

    static ulong rank2index(ulong b, int r)
    {
      b >>= (8 * r);
      b >>= 1;
      return b;
    }

    static ulong file2index(ulong b, int f)
    {
      b >>= f;
      b *= 0x0102040810204080UL;
      b >>= 56;
      b >>= 1;
      return b;
    }

    static ulong rank2board(int r) => (0xFFUL << (8 * ((int)r)));
    static ulong file2board(int f) => (0x0101010101010101UL << ((int)f));

    internal static ulong rook_attacks(int sq, ulong occ)
    {
      occ &= ~board(sq);
      int r = rank(sq), f = file(sq);
      ulong r_occ = occ & (rank2board(r) & ~BOARD_RANK_EDGE);
      ulong f_occ = occ & (file2board(f) & ~BOARD_FILE_EDGE);
      ulong r_idx = rank2index(r_occ, r);
      ulong f_idx = file2index(f_occ, f);
      ulong r_attacks = rank_attacks_table[sq][r_idx];
      ulong f_attacks = file_attacks_table[sq][f_idx];
      return r_attacks | f_attacks;
    }

    static void rook_attacks_init()
    {
      for (int idx = 0; idx < 64; idx++)
      {
        int idx1 = idx << 1, occ;
        for (int f = 0; f <= 7; f++)
        {
          ulong b = 0;
          if (f > 0)
          {
            int i = f - 1;
            do
            {
              occ = (1 << i);
              b |= board(square(0, i));
              i--;
            }
            while (((idx1 & occ) == 0) && i >= 0);
          }
          if (f < 7)
          {
            int i = f + 1;
            do
            {
              occ = (1 << i);
              b |= board(square(0, i));
              i++;
            }
            while (((idx1 & occ) == 0) && i <= 7);
          }
          for (int r = 0; r <= 7; r++)
          {
            rank_attacks_table[square(r, f)][idx] = b;
            b <<= 8;
          }
        }
      }
      for (int idx = 0; idx < 64; idx++)
      {
        int idx1 = idx << 1, occ;
        for (int r = 0; r <= 7; r++)
        {
          ulong b = 0;
          if (r > 0)
          {
            int i = r - 1;
            do
            {
              occ = (1 << i);
              b |= board(square(i, 0));
              i--;
            }
            while (((idx1 & occ) == 0) && i >= 0);
          }
          if (r < 7)
          {
            int i = r + 1;
            do
            {
              occ = (1 << i);
              b |= board(square(i, 0));
              i++;
            }
            while (((idx1 & occ) == 0) && i <= 7);
          }
          for (int f = 0; f <= 7; f++)
          {
            file_attacks_table[square(r, f)][idx] = b;
            b <<= 1;
          }
        }
      }
    }



    internal static ulong queen_attacks(int s, ulong occ) => (rook_attacks((s), (occ)) | bishop_attacks((s), (occ)));


    internal static ulong[][] pawn_attacks_table = MakeAOA(2, 64);

    internal static ulong pawn_attacks(int s, int c) => pawn_attacks_table[c][s];

    static void pawn_attacks_init()
    {
      for (int s = 0; s < 64; s++)
      {
        int r = rank(s);
        int f = file(s);

        ulong b = 0;
        if (r != 7)
        {
          if (f != 0)
            b |= board(square(r + 1, f - 1));
          if (f != 7)
            b |= board(square(r + 1, f + 1));
        }
        pawn_attacks_table[1][s] = b;

        b = 0;
        if (r != 0)
        {
          if (f != 0)
            b |= board(square(r - 1, f - 1));
          if (f != 7)
            b |= board(square(r - 1, f + 1));
        }
        pawn_attacks_table[0][s] = b;
      }
    }


    /*
     * Given a position, produce a 64-bit material signature key.
     */
    internal static ulong calc_key(in FathomPos pos, bool mirror)
    {
      ulong white = pos.white, black = pos.black;
      if (mirror)
      {
        ulong tmp = white;
        white = black;
        black = tmp;
      }
      return (ulong)popcount(white & pos.queens) * PRIME_WHITE_QUEEN +
             (ulong)popcount(white & pos.rooks) * PRIME_WHITE_ROOK +
             (ulong)popcount(white & pos.bishops) * PRIME_WHITE_BISHOP +
             (ulong)popcount(white & pos.knights) * PRIME_WHITE_KNIGHT +
             (ulong)popcount(white & pos.pawns) * PRIME_WHITE_PAWN +
             (ulong)popcount(black & pos.queens) * PRIME_BLACK_QUEEN +
             (ulong)popcount(black & pos.rooks) * PRIME_BLACK_ROOK +
             (ulong)popcount(black & pos.bishops) * PRIME_BLACK_BISHOP +
             (ulong)popcount(black & pos.knights) * PRIME_BLACK_KNIGHT +
             (ulong)popcount(black & pos.pawns) * PRIME_BLACK_PAWN;
    }

    // Produce a 64-bit material key corresponding to the material combination
    // defined by pcs[16], where pcs[1], ..., pcs[6] are the number of white
    // pawns, ..., kings and pcs[9], ..., pcs[14] are the number of black
    // pawns, ..., kings.
    internal static ulong calc_key_from_pcs(int[] pcs, int mirror)
    {
      mirror = (mirror > 0 ? 8 : 0);
      return (ulong)pcs[WHITE_QUEEN ^ mirror] * PRIME_WHITE_QUEEN +
             (ulong)pcs[WHITE_ROOK ^ mirror] * PRIME_WHITE_ROOK +
             (ulong)pcs[WHITE_BISHOP ^ mirror] * PRIME_WHITE_BISHOP +
             (ulong)pcs[WHITE_KNIGHT ^ mirror] * PRIME_WHITE_KNIGHT +
             (ulong)pcs[WHITE_PAWN ^ mirror] * PRIME_WHITE_PAWN +
             (ulong)pcs[BLACK_QUEEN ^ mirror] * PRIME_BLACK_QUEEN +
             (ulong)pcs[BLACK_ROOK ^ mirror] * PRIME_BLACK_ROOK +
             (ulong)pcs[BLACK_BISHOP ^ mirror] * PRIME_BLACK_BISHOP +
             (ulong)pcs[BLACK_KNIGHT ^ mirror] * PRIME_BLACK_KNIGHT +
             (ulong)pcs[BLACK_PAWN ^ mirror] * PRIME_BLACK_PAWN;
    }


    static readonly ulong[] keys = new ulong[] {0,PRIME_WHITE_PAWN,PRIME_WHITE_KNIGHT,
                                      PRIME_WHITE_BISHOP,PRIME_WHITE_ROOK,
                                      PRIME_WHITE_QUEEN,0,0,PRIME_BLACK_PAWN,
                                      PRIME_BLACK_KNIGHT,PRIME_BLACK_BISHOP,
                                     PRIME_BLACK_ROOK,PRIME_BLACK_QUEEN,0};

    // Produce a 64-bit material key corresponding to the material combination
    // piece[0], ..., piece[num - 1], where each value corresponds to a piece
    // (1-6 for white pawn-king, 9-14 for black pawn-king).
    internal static ulong calc_key_from_pieces(Span<byte> piece, int num)
    {
      ulong key = 0;
      for (int i = 0; i < num; i++)
      {
        Debug.Assert(piece[i] < 16);
        key += keys[piece[i]];
      }
      return key;
    }

    static ushort make_move(int promote, int from, int to) => (ushort)((((promote) & 0x7) << 12) | (((from) & 0x3F) << 6) | ((to) & 0x3F));

    internal static ushort move_from(ushort move) => (ushort)(((move) >> 6) & 0x3F);
    internal static ushort move_to(ushort move) => (ushort)((move) & 0x3F);
    internal static ushort move_promotes(ushort move) => (ushort)(((move) >> 12) & 0x7);

    internal static int type_of_piece_moved(in FathomPos pos, ushort move)
    {
      for (int i = (int)FathomPieceType.PAWN; i <= (int)FathomPieceType.KING; i++)
      {
        if ((pieces_by_type(pos, ((pos.turn ? 1 : 0) == (int)FathomColor.WHITE) ? FathomColor.WHITE : FathomColor.BLACK, (FathomPieceType)i) & board(move_from(move))) != 0)
        {
          return i;
        }
      }

      throw new Exception("Internal error");
    }

    internal const int MAX_MOVES = TB_MAX_MOVES;
    internal const int MOVE_STALEMATE = 0xFFFF;
    internal const int MOVE_CHECKMATE = 0xFFFE;

    static void add_move(TBMoveList moves, bool promotes, int from, int to)
    {
      if (!promotes)
        moves.AddMove(make_move(TB_PROMOTES_NONE, from, to));
      else
      {
        moves.AddMove(make_move(TB_PROMOTES_QUEEN, from, to));
        moves.AddMove(make_move(TB_PROMOTES_KNIGHT, from, to));
        moves.AddMove(make_move(TB_PROMOTES_ROOK, from, to));
        moves.AddMove(make_move(TB_PROMOTES_BISHOP, from, to));
      }

    }

    /*
     * Generate all captures, including all underpomotions
     */
    internal static TBMoveList gen_captures(in FathomPos pos)
    {
      TBMoveList moves = new TBMoveList();

      ulong occ = pos.white | pos.black;
      ulong us = (pos.turn ? pos.white : pos.black),
           them = (pos.turn ? pos.black : pos.white);
      ulong b, att;
      {
        int from = lsb(pos.kings & us);
        Debug.Assert(from < 64);
        for (att = king_attacks(from) & them; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.queens; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = queen_attacks(from, occ) & them; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.rooks; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = rook_attacks(from, occ) & them; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.bishops; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = bishop_attacks(from, occ) & them; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.knights; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = knight_attacks(from) & them; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.pawns; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        att = pawn_attacks(from, pos.turn ? 1 : 0);
        if (pos.ep != 0 && ((att & board(pos.ep)) != 0))
        {
          int to = pos.ep;
          add_move(moves, false, from, to);
        }
        for (att = att & them; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, (rank(to) == 7 || rank(to) == 0), from, to);
        }
      }
      return moves;
    }

    /*
     * Generate all moves.
     */
    static internal TBMoveList gen_moves(in FathomPos pos)
    {
      TBMoveList moves = new();
      ulong occ = pos.white | pos.black;
      ulong us = (pos.turn ? pos.white : pos.black),
               them = (pos.turn ? pos.black : pos.white);
      ulong b, att;

      {
        int from = lsb(pos.kings & us);
        for (att = king_attacks(from) & ~us; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.queens; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = queen_attacks(from, occ) & ~us; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.rooks; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = rook_attacks(from, occ) & ~us; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.bishops; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = bishop_attacks(from, occ) & ~us; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.knights; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        for (att = knight_attacks(from) & ~us; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, false, from, to);
        }
      }
      for (b = us & pos.pawns; b != 0; b = poplsb(b))
      {
        int from = lsb(b);
        int next = from + (pos.turn ? 8 : -8);
        att = pawn_attacks(from, pos.turn ? 1 : 0);
        if (pos.ep != 0 && ((att & board(pos.ep)) != 0))
        {
          int to = pos.ep;
          add_move(moves, false, from, to);
        }
        att &= them;
        if ((board(next) & occ) == 0)
        {
          att |= board(next);
          int next2 = from + (pos.turn ? 16 : -16);
          if ((pos.turn ? rank(from) == 1 : rank(from) == 6) &&
                  ((board(next2) & occ) == 0))
            att |= board(next2);
        }
        for (; att != 0; att = poplsb(att))
        {
          int to = lsb(att);
          add_move(moves, (rank(to) == 7 || rank(to) == 0), from, to);
        }
      }
      return moves;
    }

    /*
     * Test if the given move is an en passant capture.
     */
    internal static bool is_en_passant(in FathomPos pos, ushort move)
    {
      ushort from = move_from(move);
      ushort to = move_to(move);
      ulong us = (pos.turn ? pos.white : pos.black);
      if (pos.ep == 0)
        return false;
      if (to != pos.ep)
        return false;
      if ((board(from) & us & pos.pawns) == 0)
        return false;
      return true;
    }


    /*
     * Test if the given move is a capture.
     */
    internal static bool is_capture(in FathomPos pos, ushort move)
    {
      ushort to = move_to(move);
      ulong them = (pos.turn ? pos.black : pos.white);
      return (them & board(to)) != 0 || is_en_passant(pos, move);
    }


    /*
     * Test if the given position is legal.
     * (Pawns on backrank? Can the king be captured?)
     */
    static bool is_legal(in FathomPos pos)
    {
      ulong occ = pos.white | pos.black;
      ulong us = (pos.turn ? pos.black : pos.white),
               them = (pos.turn ? pos.white : pos.black);
      ulong king = pos.kings & us;
      if (king == 0)
        return false;
      int sq = lsb(king);
      if ((king_attacks(sq) & (pos.kings & them)) != 0)
        return false;
      ulong ratt = rook_attacks(sq, occ);
      ulong batt = bishop_attacks(sq, occ);
      if ((ratt & (pos.rooks & them)) != 0)
        return false;
      if ((batt & (pos.bishops & them)) != 0)
        return false;
      if (((ratt | batt) & (pos.queens & them)) != 0)
        return false;
      if ((knight_attacks(sq) & (pos.knights & them)) != 0)
        return false;
      if ((pawn_attacks(sq, pos.turn ? 0 : 1) & (pos.pawns & them)) != 0)
        return false;
      return true;
    }

    static readonly int[] index64 = new[]{
    0,  1, 48,  2, 57, 49, 28,  3,
   61, 58, 50, 42, 38, 29, 17,  4,
   62, 55, 59, 36, 53, 51, 43, 22,
   45, 39, 33, 30, 24, 18, 12,  5,
   63, 47, 56, 27, 60, 41, 37, 16,
   54, 35, 52, 21, 44, 32, 23, 11,
   46, 26, 40, 15, 34, 20, 31, 10,
   25, 14, 19,  9, 13,  8,  7,  6
};



    /**
     * @author Matt Taylor
     * @return index 0..63
     * @param bb a 64-bit word to bitscan, should not be zero
     */
    static int[] foldedTable = {
     63,30, 3,32,59,14,11,33,
     60,24,50, 9,55,19,21,34,
     61,29, 2,53,51,23,41,18,
     56,28, 1,43,46,27, 0,35,
     62,31,58, 4, 5,49,54, 6,
     15,52,12,40, 7,42,45,16,
     25,57,48,13,10,39, 8,44,
     20,47,38,22,17,37,36,26,
    };

    static public int bitScanForwardMatt(long b)
    {
      b ^= (b - 1);
      int folded = ((int)b) ^ ((int)(b >> 32));
      return foldedTable[(folded * 0x78291ACF) >> 26];
    }

    internal static int lsb(ulong b) => BitOperations.TrailingZeroCount(b);
    //static int lsb(int b) => throw new NotImplementedException();

    internal static ulong poplsb(ulong b) => ((b) & ((b) - 1));

    internal static int popcount(ulong b) => BitOperations.PopCount(b);

    /*
     * Test if the king is in check.
     */
    internal static bool is_check(in FathomPos pos)
    {
      ulong occ = pos.white | pos.black;
      ulong us = (pos.turn ? pos.white : pos.black),
               them = (pos.turn ? pos.black : pos.white);
      ulong king = pos.kings & us;

      Debug.Assert(king != 0);
      int sq = lsb(king);
      ulong ratt = rook_attacks(sq, occ);
      ulong batt = bishop_attacks(sq, occ);
      if ((ratt & (pos.rooks & them)) != 0)
        return true;
      if ((batt & (pos.bishops & them)) != 0)
        return true;
      if (((ratt | batt) & (pos.queens & them)) != 0)
        return true;
      if ((knight_attacks(sq) & (pos.knights & them)) != 0)
        return true;
      if ((pawn_attacks(sq, pos.turn ? 1 : 0) & (pos.pawns & them)) != 0)
        return true;
      return false;
    }

    /*
     * Test if the position is valid.
     */
    internal static bool is_valid(in FathomPos pos)
    {
      if (popcount(pos.kings) != 2)
        return false;
      if (popcount(pos.kings & pos.white) != 1)
        return false;
      if (popcount(pos.kings & pos.black) != 1)
        return false;
      if ((pos.white & pos.black) != 0)
        return false;
      if ((pos.kings & pos.queens) != 0)
        return false;
      if ((pos.kings & pos.rooks) != 0)
        return false;
      if ((pos.kings & pos.bishops) != 0)
        return false;
      if ((pos.kings & pos.knights) != 0)
        return false;
      if ((pos.kings & pos.pawns) != 0)
        return false;
      if ((pos.queens & pos.rooks) != 0)
        return false;
      if ((pos.queens & pos.bishops) != 0)
        return false;
      if ((pos.queens & pos.knights) != 0)
        return false;
      if ((pos.queens & pos.pawns) != 0)
        return false;
      if ((pos.rooks & pos.bishops) != 0)
        return false;
      if ((pos.rooks & pos.knights) != 0)
        return false;
      if ((pos.rooks & pos.pawns) != 0)
        return false;
      if ((pos.bishops & pos.knights) != 0)
        return false;
      if ((pos.bishops & pos.pawns) != 0)
        return false;
      if ((pos.knights & pos.pawns) != 0)
        return false;
      if ((pos.pawns & BOARD_FILE_EDGE) > 0)
        return false;
      if ((pos.white | pos.black) !=
          (pos.kings | pos.queens | pos.rooks | pos.bishops | pos.knights |
           pos.pawns))
        return false;

      return is_legal(pos);
    }

    static ulong do_bb_move(ulong b, int from, int to) =>
      (((b) & (~board(to)) & (~board(from))) |
            ((((b) >> (from)) & 0x1) << (to)));

    internal static bool do_move(ref FathomPos pos, in FathomPos pos0, ushort move)
    {
      int from = move_from(move);
      int to = move_to(move);
      int promotes = move_promotes(move);
      pos.turn = !pos0.turn;
      pos.white = do_bb_move(pos0.white, from, to);
      pos.black = do_bb_move(pos0.black, from, to);
      pos.kings = do_bb_move(pos0.kings, from, to);
      pos.queens = do_bb_move(pos0.queens, from, to);
      pos.rooks = do_bb_move(pos0.rooks, from, to);
      pos.bishops = do_bb_move(pos0.bishops, from, to);
      pos.knights = do_bb_move(pos0.knights, from, to);
      pos.pawns = do_bb_move(pos0.pawns, from, to);
      pos.ep = 0;
      if (promotes != TB_PROMOTES_NONE)
      {
        pos.pawns &= ~board(to);       // Promotion
        switch (promotes)
        {
          case TB_PROMOTES_QUEEN:
            pos.queens |= board(to); break;
          case TB_PROMOTES_ROOK:
            pos.rooks |= board(to); break;
          case TB_PROMOTES_BISHOP:
            pos.bishops |= board(to); break;
          case TB_PROMOTES_KNIGHT:
            pos.knights |= board(to); break;
        }
        pos.rule50 = 0;
      }
      else if ((board(from) & pos0.pawns) != 0)
      {
        pos.rule50 = 0;                // Pawn move
        if (rank(from) == 1 && rank(to) == 3 &&
            (pawn_attacks(from + 8, 1) & pos0.pawns & pos0.black) != 0)
          pos.ep = (byte)(from + 8);
        else if (rank(from) == 6 && rank(to) == 4 &&
            (pawn_attacks(from - 8, 0) & pos0.pawns & pos0.white) != 0)
          pos.ep = (byte)(from - 8);
        else if (to == pos0.ep)
        {
          int ep_to = (pos0.turn ? to - 8 : to + 8);
          ulong ep_mask = ~board(ep_to);
          pos.white &= ep_mask;
          pos.black &= ep_mask;
          pos.pawns &= ep_mask;
        }
      }
      else if ((board(to) & (pos0.white | pos0.black)) != 0)
        pos.rule50 = 0;                // Capture
      else
        pos.rule50 = (byte)(pos0.rule50 + 1); // Normal move
      if (!is_legal(pos))
        return false;
      return true;
    }

    internal static bool legal_move(in FathomPos pos, ushort move)
    {
      FathomPos pos1 = default;
      return do_move(ref pos1, pos, move);
    }


    /// <summary>
    /// Return if the king is in checkmate.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    internal static bool is_mate(in FathomPos pos)
    {
      if (!is_check(pos))
      {
        return false;
      }

      TBMoveList moves = gen_moves(pos);
      for (int i = 0; i < moves.NumMoves; i++)
      {
        FathomPos pos1 = default;
        if (do_move(ref pos1, in pos, moves.Moves[i]))
        {
          return false;
        }
      }

      return true;
    }


    /// <summary>
    /// Generates all legal moves.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    internal static TBMoveList gen_legal(in FathomPos pos)
    {
      TBMoveList ret = new TBMoveList();

      TBMoveList moves = gen_moves(pos);
      for (int i = 0; i < moves.NumMoves; i++)
      {
        if (legal_move(pos, moves.Moves[i]))
        {
          ret.AddMove(moves.Moves[i]);
        }
      }
      return ret;
    }

  }

}