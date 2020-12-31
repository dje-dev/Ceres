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
  /// Methodology for aggregating (possibly) multiple devices
  /// </summary>
  public enum NNEvaluatorDeviceComboType
  {
    /// <summary>
    /// Single device
    /// </summary>
    Single,

    /// <summary>
    /// Multiple devices with each batch split across the devices
    /// </summary>
    Split,

    /// <summary>
    /// Multiple devices with each batch being sent to one of the devices in a round-robin fashion
    /// </summary>
    RoundRobin,

    /// <summary>
    /// Evaluator is shared and batches consiste of positions pooled across possibly multiple clients.
    /// </summary>
    Pooled,

    /// <summary>
    /// Multiple devices with the output from the first returned but compared against another with large deviations reported
    /// </summary>
    Compare
  };
}
