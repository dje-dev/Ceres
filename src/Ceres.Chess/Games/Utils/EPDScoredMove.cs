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

namespace Ceres.Chess.Games.Utils
{
  /// <summary>
  /// Represnts a single scored move within an EPD entry.
  /// </summary>
  public readonly struct EPDScoredMove
  {
    /// <summary>
    /// Move (as a string).
    /// </summary>
    public readonly string MoveStr;

    /// <summary>
    /// Score of move (number of points of "goodness").
    /// </summary>
    public readonly int Score;


    /// <summary>
    /// Constructor from specified move and score.
    /// </summary>
    /// <param name="moveStr"></param>
    /// <param name="score"></param>
    public EPDScoredMove(string moveStr, int score)
    {
      MoveStr = moveStr;
      Score = score;
    }
  }


}

