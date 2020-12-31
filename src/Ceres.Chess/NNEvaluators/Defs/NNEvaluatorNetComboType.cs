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

namespace Ceres.Chess.NNEvaluators.Defs
{
  /// <summary>
  /// Methodology for combining the output of (possibly) multiple neural networks
  /// </summary>
  public enum NNEvaluatorNetComboType
  {
    /// <summary>
    /// Single neural network
    /// </summary>
    Single,

    /// <summary>
    /// Weighted average of several neural neworks
    /// </summary>
    WtdAverage,

    /// <summary>
    /// Multiple networks with the output from the first returned but compared against another with large deviations reported
    /// </summary>
    Compare,

    /// <summary>
    /// Multiple neural networks, only one of which is selected for each evaluation (dynamically selected)
    /// </summary>
    Dynamic
  };
}
