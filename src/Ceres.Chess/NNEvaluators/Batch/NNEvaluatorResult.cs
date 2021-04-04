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

using Ceres.Chess.EncodedPositions;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// Represents the a evaluation from a neural network
  /// of a single position.
  /// </summary>
  public readonly struct NNEvaluatorResult
  {
    #region Private data

    private readonly float winP;
    private readonly float lossP;

    #endregion

    /// <summary>
    /// Moves left head output.
    /// </summary>
    public readonly float M;

    /// <summary>
    /// Policy head output.
    /// </summary>
    public readonly CompressedPolicyVector Policy;

    /// <summary>
    /// Activations from certain hidden layers (optional).
    /// </summary>
    public readonly NNEvaluatorResultActivations Activations;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    /// <param name="policy"></param>
    /// <param name="activations"></param>
    public NNEvaluatorResult(float winP, float lossP, float m, 
                             CompressedPolicyVector policy, 
                             NNEvaluatorResultActivations activations)
    {
      this.winP = winP;
      this.lossP = lossP;
      M = m;
      Policy = policy;
      Activations = activations;
    }


    /// <summary>
    /// Value (win minus loss probability).
    /// </summary>
    public readonly float V => float.IsNaN(lossP) ? winP : (winP - lossP);


    /// <summary>
    /// Draw probability.
    /// </summary>
    public readonly float D => 1.0f - (winP + lossP);


    /// <summary>
    /// Win probability.
    /// </summary>
    public readonly float W => float.IsNaN(lossP) ? float.NaN : winP;


    /// <summary>
    /// Draw probability.
    /// </summary>
    public readonly float L => float.IsNaN(lossP) ? float.NaN : lossP;


    /// <summary>
    /// Returns string description.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<NNPositionEvaluation V={V,6:F2} Policy={Policy}>";
    }
  }
}
