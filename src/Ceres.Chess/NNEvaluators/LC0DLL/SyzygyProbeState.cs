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
  /// Result of an Syzyzy probe
  /// </summary>
  public enum SyzygyProbeState
  {
    /// <summary>
    /// DTZ should check other side
    /// </summary>
    ChangeSTM = -1,

    /// <summary>
    /// Fail
    /// </summary>
    Fail = 0,

    /// <summary>
    /// Ok
    /// </summary>
    Ok = 1,

    /// <summary>
    /// Best move zeros DTZ
    /// </summary>
    ZeroingBestMove = 2
  }

}
