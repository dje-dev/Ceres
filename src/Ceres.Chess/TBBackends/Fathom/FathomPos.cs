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


#endregion

namespace Ceres.Chess.TBBackends.Fathom
{
  /// <summary>
  /// Represents a board position for use internally with Fathom methods.
  /// </summary>
  public struct FathomPos
  {
    public ulong white;
    public ulong black;
    public ulong kings;
    public ulong queens;
    public ulong rooks;
    public ulong bishops;
    public ulong knights;
    public ulong pawns;
    public byte rule50;
    public byte ep;
    public bool turn;
    public short move;
    public int castling;

    public FathomPos(ulong white, ulong black, ulong kings, ulong queens, ulong rooks, ulong bishops, ulong knights, ulong pawns, byte rule50, byte ep, bool turn, short move, int castling)
    {
      this.white = white;
      this.black = black;
      this.kings = kings;
      this.queens = queens;
      this.rooks = rooks;
      this.bishops = bishops;
      this.knights = knights;
      this.pawns = pawns;
      this.rule50 = rule50;
      this.ep = ep;
      this.turn = turn;
      this.move = move;
      this.castling = castling;
    }

  }




}