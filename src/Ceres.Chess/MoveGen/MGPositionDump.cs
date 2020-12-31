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
using System.Text;


#endregion

#region License
/*

MIT License

Copyright(c) 2016-2017 Judd Niemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#endregion

using BitBoard = System.UInt64;

namespace Ceres.Chess.MoveGen
{
  public partial struct MGPosition
  {
#if MG_USE_HASH
    public override int GetHashCode() => (int)HK;
#endif

    public override string ToString() => $"<MGChessPosition A= {A}, B= {B}, C= {C}, D= {D}, Flags={Flags}>";

    // --------------------------------------------------------------------------------------------
    public readonly void Dump()
    {
      Console.WriteLine(BoardString);
    }

    // --------------------------------------------------------------------------------------------
    public readonly string BoardString
    {
      get
      {
        StringBuilder ret = new StringBuilder();

        ret.AppendLine("\n---------------------------------");
        ulong M;
        BitBoard V;
        for (int q = 63; q >= 0; q--)
        {
          M = 1UL << q;
          V = ((D) & M) >> q;
          V <<= 1;
          V |= ((C) & M) >> q;
          V <<= 1;
          V |= ((B) & M) >> q;
          V <<= 1;
          V |= ((A) & M) >> q;

#if SOMEDAY_CSHARP_LATER_VERSION
        string chars = V.Bits switch
        {
          PositionConstants.WPAWN => " | P ";
          _ => "|   ");
    };
#endif

          switch (V)
          {
            case MGPositionConstants.WPAWN:
              ret.Append("| P ");
              break;
            case MGPositionConstants.WBISHOP:
              ret.Append("| B ");
              break;
            case MGPositionConstants.WROOK:
              ret.Append("| R ");
              break;
            case MGPositionConstants.WQUEEN:
              ret.Append("| Q ");
              break;
            case MGPositionConstants.WKING:
              ret.Append("| K ");
              break;
            case MGPositionConstants.WKNIGHT:
              ret.Append("| N ");
              break;
            case MGPositionConstants.WENPASSANT:
              ret.Append("| EP");
              break;
            case MGPositionConstants.BPAWN:
              ret.Append("| p ");
              break;
            case MGPositionConstants.BBISHOP:
              ret.Append("| b ");
              break;
            case MGPositionConstants.BROOK:
              ret.Append("| r ");
              break;
            case MGPositionConstants.BQUEEN:
              ret.Append("| q ");
              break;
            case MGPositionConstants.BKING:
              ret.Append("| k ");
              break;
            case MGPositionConstants.BKNIGHT:
              ret.Append("| n ");
              break;
            case MGPositionConstants.BENPASSANT:
              ret.Append("| ep");
              break;
            default:
              ret.Append("|   ");
              break;
          }

          if ((q % 8) == 0)
          {
            ret.Append("|   ");
            if ((q == 56) && (BlackCanCastle))
              ret.Append("Black can Castle");
            if ((q == 48) && (BlackCanCastleLong))
              ret.Append("Black can Castle Long");
            if ((q == 40) && (WhiteCanCastle))
              ret.Append("White can Castle");
            if ((q == 32) && (WhiteCanCastleLong))
              ret.Append("White can Castle Long");
            //if (q == 8) ret.Append($"material={ p.material}");
            if (q == 0) ret.Append(BlackToMove ? "Black" : "White" + " to move");
            ret.AppendLine("\r\n---------------------------------");
          }
        }

        return ret.ToString();
      }
    }

  }
}


