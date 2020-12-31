#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region using directives

using System.Collections.Generic;
using System.IO;

#endregion

namespace Ceres.Chess.Textual.PgnFileTools.Extensions
{
  public static class TextReaderExtensions
  {
    public static IEnumerable<char> GenerateFrom(this TextReader source)
    {
      while (source.Peek() != -1)
      {
        yield return (char)source.Read();
      }
    }
  }
}
