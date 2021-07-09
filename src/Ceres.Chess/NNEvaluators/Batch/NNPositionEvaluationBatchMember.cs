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
using System;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// A single member of an evaluation batch.
  /// </summary>
  public class NNPositionEvaluationBatchMember
  {
    /// <summary>
    /// Win probability.
    /// </summary>
    public float WinP => batch.GetWinP(index);

    /// <summary>
    /// Loss probability.
    /// </summary>
    public float LossP => batch.GetLossP(index);

    /// <summary>
    /// Moves left value.
    /// </summary>
    public float M => batch.GetM(index);

    /// <summary>
    /// Policy vector.
    /// </summary>
    public CompressedPolicyVector Policy
    {
      get
      {
        (Memory<CompressedPolicyVector> policies, int index) pol = batch.GetPolicy(index);
        return pol.policies.Span[pol.index];
      }
    }

    /// <summary>
    /// Position value head output.
    /// </summary>
    public float V => batch.GetV(index);

    IPositionEvaluationBatch batch;
    int index;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="index"></param>
    public NNPositionEvaluationBatchMember(IPositionEvaluationBatch batch, int index)
    {
      this.batch = batch;
      this.index = index;
    }    

  }
}
