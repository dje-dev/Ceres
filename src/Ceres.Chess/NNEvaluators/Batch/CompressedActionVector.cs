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

using System.Runtime.CompilerServices;

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// Array of action probabilities for each move.
  /// This structure is typically stored alongside a CompressedPolicyVector
  /// for the same  position, with the ordering of the moves matching.
  /// 
  /// TODO: consider switching from WDL to just V to save memory.
  /// TODO: make this readonly
  /// </summary>
  [InlineArray(CompressedPolicyVector.NUM_MOVE_SLOTS)]
  public struct CompressedActionVector
  {
    /// <summary>
    /// Win/loss probabilities returned by the action head.
    /// </summary>
    (FP16 W, FP16 L) WL;

    /// <summary>
    /// Win probability.
    /// </summary>
    public float W => WL.W;
    
    /// <summary>
    /// Loss probability.
    /// </summary>
    public float L => WL.L;

    /// <summary>
    /// Draw probability.
    /// </summary>
    public float D => 1 - (WL.W + WL.L);  

    /// <summary>
    /// Tuple of win, draw, loss probabilities.
    /// </summary>
    public (float w, float d, float l) WDL => (WL.W, D, WL.L);

    /// <summary>
    /// Net win probability.
    /// </summary>
    public float V => W - L;
  }
}
