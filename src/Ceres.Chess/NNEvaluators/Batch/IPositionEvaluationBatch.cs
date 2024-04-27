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
using System.Collections.Generic;

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// Interface defining common operations supported by 
  /// batches of neural network evaluations of positions.
  /// </summary>
  public interface IPositionEvaluationBatch : IEnumerable<NNPositionEvaluationBatchMember>
  {
    int NumPos { get; }
    bool IsWDL { get; }
    bool HasM { get; }
    bool HasUncertaintyV { get; }
    bool HasAction { get; }
    bool HasState { get; }

    bool HasValueSecondary { get; }

    FP16 GetWinP(int index);
    FP16 GetLossP(int index);

    FP16 GetWin2P(int index);
    FP16 GetLoss2P(int index);

    FP16 GetM(int index);
    FP16 GetUncertaintyV(int index);
    FP16 GetExtraStat0(int index);
    FP16 GetExtraStat1(int index);

    Half[] GetState(int index);

    (Memory<CompressedPolicyVector> policies, int index) GetPolicy(int index);
    (Memory<CompressedActionVector> actions, int index) GetAction(int index);

  
    public NNEvaluatorResultActivations GetActivations(int index);

    public float GetV(int index) => GetWinP(index) - GetLossP(index);

    public ref readonly CompressedPolicyVector PolicyRef(int index)
    {
      (Memory<CompressedPolicyVector> policies, _) = GetPolicy(index);
      return ref policies.Span[index];
    }
  }
}
