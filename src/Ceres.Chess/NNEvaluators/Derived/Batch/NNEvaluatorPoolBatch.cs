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
using System.Threading;

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Aggregates together positions for possibly many sub-batches
  /// before executing them as a single large batch and
  /// returning back as disaggregated sub-batches.
  /// </summary>
  internal class NNEvaluatorPoolBatch
  {
    /// <summary>
    /// Event which is triggered when batch is done processing.
    /// </summary>
    internal ManualResetEventSlim batchesDoneEvent = new ManualResetEventSlim(false);

    /// <summary>
    /// List of batches which have thus far been added.
    /// </summary>
    internal List<IEncodedPositionBatchFlat> pendingBatches = new List<IEncodedPositionBatchFlat>();

    /// <summary>
    /// Holds the disaggregated results (one for each requesting search thread)
    /// after evaluation is completed (to be picked up by the waiting search threads).
    /// </summary>
    internal PositionEvaluationBatch[] completedBatches;


    /// <summary>
    /// Processes the current set of batches by:
    ///   - aggregating them into one big batch
    ///   - evaluating that big batch all at once
    ///   - disaggregating the returned evaluations into sub-batch-results
    /// </summary>
    /// <param name="evaluator"></param>
    /// <param name="retrieveSupplementalResults"></param>
    internal void ProcessPooledBatch(NNEvaluator evaluator, bool retrieveSupplementalResults)
    {
      // Combine together the pending batches.
      IEncodedPositionBatchFlat fullBatch = null;
      if (pendingBatches.Count == 1)
      {
        // Handle the special and easy case of exactly one batch.
        fullBatch = pendingBatches[0];
      }
      else
      {
        fullBatch = AggregateBatches();
      }

      // Evaluate the big batch
      IPositionEvaluationBatch fullBatchResult = evaluator.EvaluateIntoBuffers(fullBatch, retrieveSupplementalResults);
      PositionEvaluationBatch batchDirect = (PositionEvaluationBatch)fullBatchResult;

      completedBatches = DisaggregateBatches(retrieveSupplementalResults, batchDirect, pendingBatches);
    }

    /// <summary>
    /// Aggregates together all batches in the pending set into a single big batch.
    /// </summary>
    /// <returns></returns>
    private IEncodedPositionBatchFlat AggregateBatches()
    {
      IEncodedPositionBatchFlat fullBatch;

      // Concatenate all batches together into one big batch.
      // TODO: could we allocate these arrays once and then reuse for efficiency?
      int numPositions = NumPendingPositions;
      ulong[] posPlaneBitmaps = new ulong[numPositions * EncodedPositionWithHistory.NUM_PLANES_TOTAL];
      byte[] posPlaneValuesEncoded = new byte[numPositions * EncodedPositionWithHistory.NUM_PLANES_TOTAL];

      bool hasPositions = pendingBatches[0].Positions != null;
      bool hasMoves = pendingBatches[0].Moves != null;
      bool hasHashes = pendingBatches[0].PositionHashes != null;

      MGPosition[] positions = hasPositions ? new MGPosition[numPositions] : null;
      ulong[] positionHashes = hasHashes ? new ulong[numPositions] : null;
      MGMoveList[] moves = hasMoves ? new MGMoveList[numPositions] : null;

      int nextSourceBitmapIndex = 0;
      int nextSourceValueIndex = 0;
      int nextPositionIndex = 0;
      foreach (EncodedPositionBatchFlat thisBatch in pendingBatches)
      {
        int skipCount = thisBatch.NumPos * EncodedPositionWithHistory.NUM_PLANES_TOTAL;
        Array.Copy(thisBatch.PosPlaneBitmaps, 0, posPlaneBitmaps, nextSourceBitmapIndex, skipCount);
        nextSourceBitmapIndex += skipCount;
        Array.Copy(thisBatch.PosPlaneValues, 0, posPlaneValuesEncoded, nextSourceValueIndex, skipCount);
        nextSourceValueIndex += skipCount;

        if (hasPositions)
        {
          Array.Copy(thisBatch.Positions, 0, positions, nextPositionIndex, thisBatch.NumPos);
        }

        if (hasHashes)
        {
          Array.Copy(thisBatch.PositionHashes, 0, positionHashes, nextPositionIndex, thisBatch.NumPos);
        }

        if (hasMoves)
        {
          Array.Copy(thisBatch.Moves, 0, moves, nextPositionIndex, thisBatch.NumPos);
        }

        nextPositionIndex += thisBatch.NumPos;
      }

      fullBatch = new EncodedPositionBatchFlat(posPlaneBitmaps, posPlaneValuesEncoded, null, null, null, numPositions);

      if (hasPositions)
      {
        fullBatch.Positions = positions;
      }

      if (hasHashes)
      {
        fullBatch.PositionHashes = positionHashes;
      }

      if (hasMoves)
      {
        fullBatch.Moves = moves;
      }

      return fullBatch;
    }


    /// <summary>
    /// Splits up positions in an aggregated batch back into sub-batches.
    /// </summary>
    /// <param name="retrieveSupplementalResults"></param>
    /// <param name="fullBatchResult"></param>
    /// <param name="pendingBatches"></param>
    /// <returns></returns>
    internal static PositionEvaluationBatch[] DisaggregateBatches(bool retrieveSupplementalResults, 
                                                                  PositionEvaluationBatch fullBatchResult, 
                                                                  List<IEncodedPositionBatchFlat> pendingBatches)
    {
      Span<CompressedPolicyVector> fullPolicyValues = fullBatchResult.Policies.Span;
      Span<FP16> fullW = fullBatchResult.W.Span;

      Span<FP16> fullL = fullBatchResult.IsWDL ? fullBatchResult.L.Span : default;
      Span<FP16> fullM = fullBatchResult.IsWDL ? fullBatchResult.M.Span : default;
      Span<NNEvaluatorResultActivations> fullActivations = fullBatchResult.Activations.IsEmpty ? null : fullBatchResult.Activations.Span;

      // Finally, disaggregate the big batch back into a set of individual subbatch results
      PositionEvaluationBatch[] completedBatches = new PositionEvaluationBatch[pendingBatches.Count];

      int subBatchIndex = 0;
      int nextPosIndex = 0;
      foreach (EncodedPositionBatchFlat thisBatch in pendingBatches)
      {
        if (retrieveSupplementalResults) throw new NotImplementedException();

        int numPos = thisBatch.NumPos;
        PositionEvaluationBatch thisResultSubBatch =
          new PositionEvaluationBatch(fullBatchResult.IsWDL, fullBatchResult.HasM, thisBatch.NumPos,
                                fullPolicyValues.Slice(nextPosIndex, numPos).ToArray(),
                                fullW.Slice(nextPosIndex, numPos).ToArray(),
                                fullBatchResult.IsWDL ? fullL.Slice(nextPosIndex, numPos).ToArray() : null,
                                fullBatchResult.IsWDL ? fullM.Slice(nextPosIndex, numPos).ToArray() : null,
                                fullBatchResult.Activations.IsEmpty ? null : fullActivations.Slice(nextPosIndex, numPos).ToArray(),
                                fullBatchResult.Stats);

        nextPosIndex += numPos;
        completedBatches[subBatchIndex++] = thisResultSubBatch;
      }
      return completedBatches;
    }


    /// <summary>
    /// Returns the number of positions in use across all sub-batches.
    /// </summary>
    internal int NumPendingPositions
    {
      get
      {
        int count = 0;
        foreach (EncodedPositionBatchFlat batch in pendingBatches)
          count += batch.NumPos;
        return count;

      }
    }

  }
}
