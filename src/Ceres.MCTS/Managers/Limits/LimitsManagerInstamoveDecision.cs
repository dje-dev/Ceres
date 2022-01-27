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


namespace Ceres.MCTS.Managers.Limits
{
  /// <summary>
  /// Decision made by an IManagerGameLimit
  /// if an instamove should be made in a given search state.
  /// </summary>
  public enum LimitsManagerInstamoveDecision
  {
    /// <summary>
    /// The manager does not request any particular decision.
    /// </summary>
    NoDecision,

    /// <summary>
    /// The manager requests an instamove.
    /// </summary>
    Instamove,

    /// <summary>
    /// The manager requests instamove not be made.
    /// </summary>
    DoNotInstamove
  }
}
