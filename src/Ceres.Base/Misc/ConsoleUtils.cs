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

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods for working with the system Console.
  /// </summary>
  public static class ConsoleUtils
  {
    /// <summary>
    /// Writes a line of output to Conole in specified color.
    /// </summary>
    /// <param name="str"></param>
    /// <param name="color"></param>
    public static void WriteLineColored(ConsoleColor color, string str)
    {
      ConsoleColor priorColor = Console.ForegroundColor;
      Console.ForegroundColor = color;
      Console.WriteLine(str);
      Console.ForegroundColor = priorColor;
    }
  }
}
