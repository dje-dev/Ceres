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
    /// Constructor.
    /// </summary>
    /// <param name="w"></param>
    /// <param name="l"></param>
    public CompressedActionVector(Half w, Half l) => WL = (w, l);

    /// <summary>
    /// Win/loss probabilities returned by the action head.
    /// </summary>
    private (Half W, Half L) WL;

    /// <summary>
    /// Win - loss probability.
    /// </summary>
    public readonly float V => (float)WL.W - (float)WL.L;

    /// <summary>
    /// Win probability.
    /// </summary>
    public readonly float W => (float)WL.W;
    
    /// <summary>
    /// Loss probability.
    /// </summary>
    public readonly float L => (float)WL.L;

    /// <summary>
    /// Draw probability.
    /// </summary>
    public readonly float D => 1 - ((float)WL.W + (float)WL.L);  

    /// <summary>
    /// Tuple of win, draw, loss probabilities.
    /// </summary>
    public readonly (float w, float d, float l) WDL => ((float)WL.W, D, (float)WL.L);

    /// <summary>
    /// Net win probability.
    /// </summary>
    public readonly float V => W - L;


    /// <summary>
    /// Returns string repersentation.
    /// </summary>
    /// <returns></returns>
    public override readonly string ToString() => $"W={W,6:F3}, D={D,6:F3}, L={L,6:F3}";
  }
}
