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


#endregion


namespace Ceres.Chess
{
  /// <summary>
  /// Set of miscellaneous extensions methods on various enumeration types.
  /// </summary>
  public static class ChessEnumExtensions
  {
    public static bool IsTerminal(this GameResult terminal) =>  terminal >= GameResult.Draw;
    public static SideType Reversed(this SideType side) => (SideType)(((int)side) ^ 0b1);

  }
}


