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

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using System;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.PositionEvalCaching
{
  /// <summary>
  /// Singl eposition with a PositionEvalCache.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  [Serializable]
  public struct PositionEvalCacheEntry
  {
    /// <summary>
    /// 
    /// </summary>
    public FP16 WinP;

    public FP16 LossP;
    public FP16 M;

    /// <summary>
    /// Cached policy value
    /// </summary>
    public CompressedPolicyVector Policy;

    public GameResult TerminalStatus;


    public PositionEvalCacheEntry(GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m, in CompressedPolicyVector policy)
    {
      TerminalStatus = terminalStatus;
      WinP = winP;
      LossP = lossP;
      M = m;
      Policy = policy;
    }
  }


}
