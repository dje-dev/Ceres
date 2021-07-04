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
using System.Runtime.InteropServices;

using static Ceres.Chess.TBBackends.Fathom.FathomMoveGen;

#endregion

namespace Ceres.Chess.TBBackends.Fathom
{
  /// <summary>
  /// Piece encoding info structure.
  /// </summary>
  internal unsafe struct EncInfo
  {
    /// <summary>
    /// Pointer to associated pairs data.
    /// </summary>
    public PairsDataPtr precomp;

    fixed ulong _factor[TB_PIECES];
    fixed byte _pieces[TB_PIECES];
    fixed byte _norm[TB_PIECES];

    public Span<ulong> factor => MemoryMarshal.CreateSpan<ulong>(ref _factor[0], TB_PIECES);
    public Span<byte> pieces => MemoryMarshal.CreateSpan<byte>(ref _pieces[0], TB_PIECES);
    public Span<byte> norm => MemoryMarshal.CreateSpan<byte>(ref _norm[0], TB_PIECES);

    public override string ToString()
    {
      return $"<EncInfo "
           + $"factor[{factor[0]},{factor[1]},{factor[2]},{factor[3]},{factor[4]},{factor[5]},{factor[6]}] "
           + $"pieces[{pieces[0]},{pieces[1]},{pieces[2]},{pieces[3]},{pieces[4]},{pieces[5]},{pieces[6]}] "
           + $"norm[{norm[0]},{norm[1]},{norm[2]},{norm[3]},{norm[4]},{norm[5]},{norm[6]}] "
           + ">";
    }

  };

}