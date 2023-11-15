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

namespace Ceres.Chess.NNEvaluators.LC0DLL
{
  /// <summary>
  /// WDL (win/draw/loss) score returned from tablebase probe.
  /// </summary>
  public enum SyzygyWDLScore
  {
    /// <summary>
    /// Loss
    /// </summary>
    WDLLoss = -2,

    /// <summary>
    /// Loss, but draw under 50-move rule
    /// </summary>
    WDLBlessedLoss = -1,

    /// <summary>
    /// Draw
    /// </summary>
    WDLDraw = 0,

    /// <summary>
    /// Win, but draw under 50-move rule
    /// </summary>
    WDLCursedWin = 1,

    /// <summary>
    /// Win
    /// </summary>
    WDLWin = 2
  }

}
