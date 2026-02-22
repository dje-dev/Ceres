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

namespace Ceres.Chess.NetEvaluation.Batch;

/// <summary>
/// An implementation of IPositionEvaluationBatch which efficiently 
/// merges together several consecutive sub-batches (using indexing into sub-arrays).
/// </summary>
internal class PositionsEvaluationBatchMerged : IPositionEvaluationBatch
{
  internal readonly IPositionEvaluationBatch[] Batches;
  internal readonly int[] BatchSizes;

  /// <summary>
  /// Cumulative offsets for each batch to enable O(log n) lookup.
  /// CumulativeOffsets[i] = sum of BatchSizes[0..i-1]
  /// </summary>
  private readonly int[] cumulativeOffsets;

  /// <summary>
  /// Total number of positions across all batches.
  /// </summary>
  private readonly int totalNumPos;

  private readonly bool isWDL;
  private readonly bool hasM;
  private readonly bool hasUncertaintyV;
  private readonly bool hasUncertaintyP;
  private readonly bool hasAction;
  private readonly bool hasValueSecondary;
  private readonly bool hasState;
  private readonly bool hasPlyBinOutputs;

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="batches"></param>
  /// <param name="batchSizes"></param>
  internal PositionsEvaluationBatchMerged(IPositionEvaluationBatch[] batches, int[] batchSizes)
  {
    Batches = batches;
    BatchSizes = batchSizes;

    // Pre-compute cumulative offsets for efficient lookup
    cumulativeOffsets = new int[batches.Length];
    int cumulativeSum = 0;
    for (int i = 0; i < batches.Length; i++)
    {
      cumulativeOffsets[i] = cumulativeSum;
      cumulativeSum += batchSizes[i];
    }
    totalNumPos = cumulativeSum;

    // Cache batch properties from first batch
    isWDL = batches[0].IsWDL;
    hasM = batches[0].HasM;
    hasUncertaintyV = batches[0].HasUncertaintyV;
    hasUncertaintyP = batches[0].HasUncertaintyP;
    hasAction = batches[0].HasAction;
    hasValueSecondary = batches[0].HasValueSecondary;
    hasState = batches[0].HasState;
    hasPlyBinOutputs = batches[0].HasPlyBinOutputs;
  }


  bool IPositionEvaluationBatch.IsWDL => isWDL;

  bool IPositionEvaluationBatch.HasM => hasM;

  bool IPositionEvaluationBatch.HasUncertaintyV => hasUncertaintyV;

  bool IPositionEvaluationBatch.HasUncertaintyP => hasUncertaintyP;

  bool IPositionEvaluationBatch.HasAction => hasAction;

  bool IPositionEvaluationBatch.HasValueSecondary => hasValueSecondary;

  bool IPositionEvaluationBatch.HasState => hasState;

  bool IPositionEvaluationBatch.HasPlyBinOutputs => hasPlyBinOutputs;

  /// <summary>
  /// Number of positions in batch.
  /// </summary>
  int IPositionEvaluationBatch.NumPos => totalNumPos;


  /// <summary>
  /// Maps a global index to (batchIndex, localIndex) tuple using binary search.
  /// </summary>
  /// <param name="index">Global index across all batches</param>
  private (int batchIndex, int localIndex) GetIndices(int index)
  {
    // Binary search to find which batch contains this index
    int left = 0;
    int right = cumulativeOffsets.Length - 1;
    int batchIndex = 0;

    while (left <= right)
    {
      int mid = left + ((right - left) >> 1);

      if (cumulativeOffsets[mid] <= index)
      {
        batchIndex = mid;
        left = mid + 1;
      }
      else
      {
        right = mid - 1;
      }
    }

    // Calculate local index within the batch
    int localIndex = index - cumulativeOffsets[batchIndex];

    return (batchIndex, localIndex);
  }


  FP16 IPositionEvaluationBatch.GetLossP(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetLossP(localIndex);
  }

  FP16 IPositionEvaluationBatch.GetLoss1P(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetLoss1P(localIndex);
  }

  FP16 IPositionEvaluationBatch.GetLoss2P(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetLoss2P(localIndex);
  }


  (Memory<CompressedPolicyVector> policies, int index) IPositionEvaluationBatch.GetPolicy(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetPolicy(localIndex);
  }


  (Memory<CompressedActionVector> actions, int index) IPositionEvaluationBatch.GetAction(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetAction(localIndex);
  }


  FP16 IPositionEvaluationBatch.GetM(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetM(localIndex);
  }


  FP16 IPositionEvaluationBatch.GetWinP(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetWinP(localIndex);
  }

  FP16 IPositionEvaluationBatch.GetWin1P(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetWin1P(localIndex);
  }

  FP16 IPositionEvaluationBatch.GetWin2P(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetWin2P(localIndex);
  }

  FP16 IPositionEvaluationBatch.GetUncertaintyV(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetUncertaintyV(localIndex);
  }

  FP16 IPositionEvaluationBatch.GetUncertaintyP(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetUncertaintyP(localIndex);
  }

  public NNEvaluatorResultActivations GetActivations(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetActivations(localIndex);
  }

  public FP16 GetExtraStat0(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetExtraStat0(localIndex);
  }

  public FP16 GetExtraStat1(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetExtraStat1(localIndex);
  }

  public Half[] GetState(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetState(localIndex);
  }

  public ReadOnlySpan<Half> GetPlyBinMoveProbs(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetPlyBinMoveProbs(localIndex);
  }

  public ReadOnlySpan<Half> GetPlyBinCaptureProbs(int index)
  {
    (int batchIndex, int localIndex) = GetIndices(index);
    return Batches[batchIndex].GetPlyBinCaptureProbs(localIndex);
  }

  public IEnumerator<NNPositionEvaluationBatchMember> GetEnumerator()
  {
    int globalIndex = 0;
    for (int batchIndex = 0; batchIndex < Batches.Length; batchIndex++)
    {
      int batchSize = BatchSizes[batchIndex];
      IPositionEvaluationBatch batch = Batches[batchIndex];

      for (int localIndex = 0; localIndex < batchSize; localIndex++)
      {
        yield return new NNPositionEvaluationBatchMember(batch, localIndex);
        globalIndex++;
      }
    }
  }

  IEnumerator IEnumerable.GetEnumerator()
  {
    return GetEnumerator();
  }


  /// <summary>
  /// Disposes of resources held by this batch.
  /// </summary>
  public void Dispose()
  {
    foreach (IPositionEvaluationBatch batch in Batches)
    {
      batch?.Dispose();
    }
  }
}
