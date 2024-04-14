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
using System.Collections;
using System.Collections.Generic;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// An implementation of IPositionEvaluationBatch which 
  /// efficiently merges together several consecutive sub-batches
  /// (using indexing into sub-arrays).
  /// </summary>
  internal class PositionsEvaluationBatchMerged : IPositionEvaluationBatch
  {
    internal readonly IPositionEvaluationBatch[] Batches;
    internal readonly int[] BatchSizes;

    /// <summary>
    /// Index of element last requested. We track this to allow 
    /// efficient traversal of items in order (the typical use case).
    /// </summary>
    int lastIndex = 0;

    /// <summary>
    /// The index of the sub-batch to which the last requested item belongs
    /// </summary>
    int lastEvaluatorIndex = 0;

    /// <summary>
    /// The index within the last evalutor at which last element was found
    /// </summary>
    int lastEvaluatorInnerIndex = 0;

    bool isWDL;
    bool hasM;
    bool hasUncertaintyV;
    bool hasAction;
    bool hasValueSecondary;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="batches"></param>
    /// <param name="batchSizes"></param>
    internal PositionsEvaluationBatchMerged(IPositionEvaluationBatch[] batches, int[] batchSizes)
    {
      Batches = batches;
      BatchSizes = batchSizes;
      isWDL = batches[0].IsWDL;
      hasM  = batches[0].HasM;
      hasUncertaintyV = batches[0].HasUncertaintyV;
      hasAction = batches[0].HasAction;
      hasValueSecondary = batches[0].HasValueSecondary;
    }

    bool IPositionEvaluationBatch.IsWDL => isWDL;

    bool IPositionEvaluationBatch.HasM => hasM;

    bool IPositionEvaluationBatch.HasUncertaintyV => hasUncertaintyV;

    bool IPositionEvaluationBatch.HasAction => hasAction;
    bool IPositionEvaluationBatch.HasValueSecondary => hasValueSecondary;

    /// <summary>
    /// Number of positions in batch.
    /// </summary>
    int IPositionEvaluationBatch.NumPos
    {
      get
      {
        int count = 0;
        for (int i = 0; i < Batches.Length; i++)
        {
          count += BatchSizes[i];
        }
        return count;
      }
    }


    /// <summary>
    /// Next available set of indices.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    (int, int) GetIndices(int index)
    {
      (int, int) nextIndices = DoGetIndices(index);
      lastIndex = index;
      return nextIndices;
    }


    (int, int) DoGetIndices(int index)
    {
      if (index == lastIndex)
      {
        return (lastEvaluatorIndex, lastEvaluatorInnerIndex);
      }
      else if (index == 0)
      {
        // Support restart from beginning
        lastEvaluatorIndex = 0;
        lastEvaluatorInnerIndex = 0;
        return (0, 0);
      }
      else if (index == lastIndex + 1)
      {
        int nextEvaluatorInnerIndex = lastEvaluatorInnerIndex + 1;
        if (nextEvaluatorInnerIndex == BatchSizes[lastEvaluatorIndex])
        {
          // Advance to next evaluator
          lastEvaluatorIndex++;
          lastEvaluatorInnerIndex = 0;
          return (lastEvaluatorIndex, 0);
        }
        else
        {
          // Just advance to next slog within same evaluator
          lastEvaluatorInnerIndex = nextEvaluatorInnerIndex;
          return (lastEvaluatorIndex, lastEvaluatorInnerIndex);
        }
      }
      else
        throw new Exception("Internal error: implementation currently only supports sequential acccess");
    }


    FP16 IPositionEvaluationBatch.GetLossP(int index)
    {
      (int, int) indices = GetIndices(index);
      return Batches[indices.Item1].GetLossP(indices.Item2);
    }

    FP16 IPositionEvaluationBatch.GetLoss2P(int index)
    {
      (int, int) indices = GetIndices(index);
      return Batches[indices.Item1].GetLoss2P(indices.Item2);
    }


    (Memory<CompressedPolicyVector> policies, int index) IPositionEvaluationBatch.GetPolicy(int index)
    {
      (int, int) indicies = GetIndices(index);
      return Batches[indicies.Item1].GetPolicy(indicies.Item2);
    }


    (Memory<CompressedActionVector> actions, int index) IPositionEvaluationBatch.GetAction(int index)
    {
      (int, int) indicies = GetIndices(index);
      return Batches[indicies.Item1].GetAction(indicies.Item2);
    }


    FP16 IPositionEvaluationBatch.GetM(int index)
    {
      (int, int) indicies = GetIndices(index);
      return Batches[indicies.Item1].GetM(indicies.Item2);
    }


    FP16 IPositionEvaluationBatch.GetWinP(int index)
    {
      (int, int) indicies = GetIndices(index);
      return Batches[indicies.Item1].GetWinP(indicies.Item2);
    }

    FP16 IPositionEvaluationBatch.GetWin2P(int index)
    {
      (int, int) indicies = GetIndices(index);
      return Batches[indicies.Item1].GetWin2P(indicies.Item2);
    }

    FP16 IPositionEvaluationBatch.GetUncertaintyV(int index)
    {
      (int, int) indicies = GetIndices(index);
      return Batches[indicies.Item1].GetUncertaintyV(indicies.Item2);
    }

    public NNEvaluatorResultActivations GetActivations(int index)
    {
      (int, int) indicies = GetIndices(index);
      return Batches[indicies.Item1].GetActivations(indicies.Item2);
    }

    public FP16 GetExtraStat0(int index)
    {
      (int, int) indicies = GetIndices(index);

      return Batches[indicies.Item1].GetExtraStat0(indicies.Item2);
    }

    public FP16 GetExtraStat1(int index)
    {
      (int, int) indicies = GetIndices(index);

      return Batches[indicies.Item1].GetExtraStat1(indicies.Item2);
    }

    public IEnumerator<NNPositionEvaluationBatchMember> GetEnumerator()
    {
      int index = 0;
      for (int i = 0; i < ((IPositionEvaluationBatch)this).NumPos; i++)
      {
        // TODO: make more efficient by tracking batch/pos indices directly rather than use GetIndices.
        (int, int) indicies = GetIndices(index);
        yield return new NNPositionEvaluationBatchMember(Batches[indicies.Item1], indicies.Item2);
        index++;
      }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
      throw new NotImplementedException();
    }

  }
}
