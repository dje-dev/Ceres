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

using static Ceres.Chess.TBBackends.Fathom.FathomMoveGen;

#endregion

namespace Ceres.Chess.TBBackends.Fathom
{
  /// <summary>
  /// Manages parsing of string FENs into FathomPos.
  /// </summary>
  internal static class FathomFENParsing
  {
    internal const int TB_CASTLING_K = 0x1;     /* White king-side. */
    internal const int TB_CASTLING_Q = 0x2;     /* White queen-side. */
    internal const int TB_CASTLING_k = 0x4;     /* Black king-side. */
    internal const int TB_CASTLING_q = 0x8;     /* Black queen-side. */

    static ulong board(int s) => ((ulong)1 << (s));

    internal static bool parse_FEN(ref FathomPos pos, string fen)
    {
      ulong white = 0, black = 0;
      ulong kings, queens, rooks, bishops, knights, pawns;
      kings = queens = rooks = bishops = knights = pawns = 0;
      bool turn;
      int rule50 = 0, move = 1;
      int ep = 0;
      int castling = 0;
      char c;
      int r, f;

      int fenIndex = 0;
      if (fen == null)
      {
        goto fen_parse_error;
      }

      for (r = 7; r >= 0; r--)
      {
        for (f = 0; f <= 7; f++)
        {
          int s = (r * 8) + f;
          ulong b = board(s);
          c = fen[fenIndex++];
          switch (c)
          {
            case 'k':
              kings |= b;
              black |= b;
              continue;
            case 'K':
              kings |= b;
              white |= b;
              continue;
            case 'q':
              queens |= b;
              black |= b;
              continue;
            case 'Q':
              queens |= b;
              white |= b;
              continue;
            case 'r':
              rooks |= b;
              black |= b;
              continue;
            case 'R':
              rooks |= b;
              white |= b;
              continue;
            case 'b':
              bishops |= b;
              black |= b;
              continue;
            case 'B':
              bishops |= b;
              white |= b;
              continue;
            case 'n':
              knights |= b;
              black |= b;
              continue;
            case 'N':
              knights |= b;
              white |= b;
              continue;
            case 'p':
              pawns |= b;
              black |= b;
              continue;
            case 'P':
              pawns |= b;
              white |= b;
              continue;
            default:
              break;
          }
          if (c >= '1' && c <= '8')
          {
            int jmp = (int)c - '0';
            f += jmp - 1;
            continue;
          }
          goto fen_parse_error;
        }
        if (r == 0)
          break;

        c = fen[fenIndex++];
        if (c != '/')
        {
          goto fen_parse_error;
        }
      }
      c = fen[fenIndex++];
      if (c != ' ')
      {
        goto fen_parse_error;
      }

      c = fen[fenIndex++];
      if (c != 'w' && c != 'b')
      {
        goto fen_parse_error;
      }

      turn = (c == 'w');
      c = fen[fenIndex++];
      if (c != ' ')
      {
        goto fen_parse_error;
      }

      c = fen[fenIndex++];
      if (c != '-')
      {
        do
        {
          switch (c)
          {
            case 'K':
              castling |= TB_CASTLING_K; break;
            case 'Q':
              castling |= TB_CASTLING_Q; break;
            case 'k':
              castling |= TB_CASTLING_k; break;
            case 'q':
              castling |= TB_CASTLING_q; break;
            default:
              goto fen_parse_error;
          }
          c = fen[fenIndex++];
        } while (c != ' ');
        fenIndex--;
      }
      c = fen[fenIndex++];
      if (c != ' ')
        goto fen_parse_error;
      c = fen[fenIndex++];
      if (c >= 'a' && c <= 'h')
      {
        int file = c - 'a';
        c = fen[fenIndex++];
        if (c != '3' && c != '6')
        {
          goto fen_parse_error;
        }

        int rank = c - '1';
        ep = FathomMoveGen.square(rank, file);
        if (rank == 2 && turn)
        {
          goto fen_parse_error;
        }

        if (rank == 5 && !turn)
        {
          goto fen_parse_error;
        }

        if (rank == 2 && ((pawn_attacks(ep, 1) & (black & pawns)) == 0))
        {
          ep = 0;
        }

        if (rank == 5 && ((pawn_attacks(ep, 0) & (white & pawns)) == 0))
        {
          ep = 0;
        }
      }
      else if (c != '-')
      {
        goto fen_parse_error;
      }

      c = fen[fenIndex++];
      if (c != ' ')
      {
        goto fen_parse_error;
      }

      string clk = ""; //char clk[4];
      clk += fen[fenIndex++];
      if (clk[0] < '0' || clk[0] > '9')
      {
        goto fen_parse_error;
      }

      clk += fen[fenIndex++];
      if (clk[1] != ' ')
      {
        if (clk[1] < '0' || clk[1] > '9')
        {
          goto fen_parse_error;
        }

        clk += fen[fenIndex++];
        if (clk[2] != ' ')
        {
          if (clk[2] < '0' || clk[2] > '9')
          {
            goto fen_parse_error;
          }

          c = fen[fenIndex++];
          if (c != ' ')
          {
            goto fen_parse_error;
          }
        }
      }

      rule50 = byte.Parse(clk);
      int intMove = int.Parse(fen.Substring(fenIndex));

      move = (byte)System.Math.Min(byte.MaxValue, intMove);

      pos.white = white;
      pos.black = black;
      pos.kings = kings;
      pos.queens = queens;
      pos.rooks = rooks;
      pos.bishops = bishops;
      pos.knights = knights;
      pos.pawns = pawns;

      pos.castling = castling;
      pos.rule50 = (byte)rule50;
      pos.ep = (byte)ep;
      pos.turn = turn;
      pos.move = (byte)move;
      return true;

      fen_parse_error:
      return false;
    }

  }

}
