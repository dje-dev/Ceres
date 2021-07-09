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

using static Ceres.Chess.TBBackends.Fathom.FathomMoveGen;

#endregion

namespace Ceres.Chess.TBBackends.Fathom
{
  /// <summary>
  /// Type of entry used for pawns to encode metadata and index information.
  /// </summary>
  internal unsafe class PieceEntry : BaseEntry
  {
    public EncInfo[] ei = new EncInfo[2 + 2 + 1];

    public ushort* dtmMap;

    public ushort[,,] dtmMapIdx = new ushort[1, 2, 2];

    public void* dtzMap;

    public ushort[,] dtzMapIdx = new ushort[1, 4];

    public byte[] dtzFlags = new byte[1];

    public override Span<EncInfo> first_ei(int type)
    {
      int start = type == (int)WDL.WDL ? 0 : type == (int)WDL.DTM ? 2 : 4;
      return new Span<EncInfo>(ei, start, 5 - start);
    }

    public override string ToString()
    {
      return $"<PieceEntry>";
    }
  }

}