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
    /// If non-null, an exception that occurred while processing this pooled batch.
    /// The waiting caller(s) rethrow this rather than the worker thread terminating the process.
    /// </summary>
    internal Exception ProcessingException;


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

      // When pooling across multiple independent searches the sub-batches may be heterogeneous
      // (e.g. some carry optional auxiliary inputs while others do not, or some are empty), so the
      // first batch is not necessarily representative. Detect each optional field across ALL batches,
      // allocate it if any batch supplies it, and copy per-batch only where present.
      bool hasPositions = false;
      bool hasMoves = false;
      bool hasHashes = false;
      bool hasPliesSincePerSquare = false;
      bool hasPositionsBuffer = false;
      bool hasCompactHistories = false;
      foreach (EncodedPositionBatchFlat scanBatch in pendingBatches)
      {
        if (scanBatch.Positions != null) { hasPositions = true; }
        if (scanBatch.Moves != null) { hasMoves = true; }
        if (scanBatch.PositionHashes != null) { hasHashes = true; }
        if (scanBatch.LastMovePlies != null) { hasPliesSincePerSquare = true; }
        if (scanBatch.PositionsBuffer != null && scanBatch.PositionsBuffer.Length > 0) { hasPositionsBuffer = true; }
        if (scanBatch.CompactHistories != null && scanBatch.CompactHistories.Length > 0) { hasCompactHistories = true; }
      }

      MGPosition[] positions = hasPositions ? new MGPosition[numPositions] : null;
      ulong[] positionHashes = hasHashes ? new ulong[numPositions] : null;
      byte[] pliesSinceLastSquare = hasPliesSincePerSquare ? new byte[numPositions * 64] : null;

      MGMoveList[] moves = hasMoves ? new MGMoveList[numPositions] : null;

      // The raw encoded positions (needed by TPG/Ceres nets that re-convert positions at eval time, e.g.
      // when EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS is set) must also be carried through the
      // aggregation, otherwise the combined batch's PositionsBuffer is empty and TPG conversion throws.
      EncodedPositionWithHistory[] positionsBuffer = hasPositionsBuffer ? new EncodedPositionWithHistory[numPositions] : null;

      // Compact history records are likewise carried through; rows originating from batches
      // without them remain default (NumPositions == 0) and consumers fall back to
      // that row's PositionsBuffer entry.
      MGPositionHistoryCompact[] compactHistories = hasCompactHistories ? new MGPositionHistoryCompact[numPositions] : null;

      int nextSourceBitmapIndex = 0;
      int nextSourceValueIndex = 0;
      int nextPositionIndex = 0;
      foreach (EncodedPositionBatchFlat thisBatch in pendingBatches)
      {
        int numPos = thisBatch.NumPos;
        if (numPos == 0)
        {
          continue;
        }

        int skipCount = numPos * EncodedPositionWithHistory.NUM_PLANES_TOTAL;
        Array.Copy(thisBatch.PosPlaneBitmaps, 0, posPlaneBitmaps, nextSourceBitmapIndex, skipCount);
        nextSourceBitmapIndex += skipCount;
        Array.Copy(thisBatch.PosPlaneValues, 0, posPlaneValuesEncoded, nextSourceValueIndex, skipCount);
        nextSourceValueIndex += skipCount;

        if (hasPositions && thisBatch.Positions != null)
        {
          Array.Copy(thisBatch.Positions, 0, positions, nextPositionIndex, numPos);
        }

        if (hasHashes && thisBatch.PositionHashes != null)
        {
          Array.Copy(thisBatch.PositionHashes, 0, positionHashes, nextPositionIndex, numPos);
        }

        if (hasPliesSincePerSquare && thisBatch.LastMovePlies != null)
        {
          Array.Copy(thisBatch.LastMovePlies, 0, pliesSinceLastSquare, nextPositionIndex * 64, numPos * 64);
        }

        if (hasMoves && thisBatch.Moves != null)
        {
          Array.Copy(thisBatch.Moves, 0, moves, nextPositionIndex, numPos);
        }

        if (hasPositionsBuffer && thisBatch.PositionsBuffer != null && thisBatch.PositionsBuffer.Length > 0)
        {
          Array.Copy(thisBatch.PositionsBuffer, 0, positionsBuffer, nextPositionIndex, numPos);
        }

        if (hasCompactHistories && thisBatch.CompactHistories != null && thisBatch.CompactHistories.Length > 0)
        {
          Array.Copy(thisBatch.CompactHistories, 0, compactHistories, nextPositionIndex, numPos);
        }

        nextPositionIndex += numPos;
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

      if (hasPliesSincePerSquare)
      {
        fullBatch.LastMovePlies = pliesSinceLastSquare;
      }

      if (hasMoves)
      {
        fullBatch.Moves = moves;
      }

      if (hasPositionsBuffer)
      {
        // PositionsBuffer is a settable field on the concrete type (get-only on the interface).
        ((EncodedPositionBatchFlat)fullBatch).PositionsBuffer = positionsBuffer;
      }

      if (hasCompactHistories)
      {
        ((EncodedPositionBatchFlat)fullBatch).CompactHistories = compactHistories;
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

      if (fullBatchResult.HasAction || fullBatchResult.HasState)
      {
        throw new NotImplementedException("action code needs remediation below, multiple heads including action/state");
      }

      Span<CompressedActionVector> fullActionValues = default;// fullBatchResult.HasAction ? fullBatchResult.ActionProbabilities.Span : default;

      Span<FP16> fullW = fullBatchResult.W.Span;
      Span<FP16> fullL = fullBatchResult.IsWDL ? fullBatchResult.L.Span : default;
      Span<FP16> fullM = fullBatchResult.HasM ? fullBatchResult.M.Span : default;

      Span<FP16> fullW2 = fullBatchResult.HasValueSecondary ? fullBatchResult.W2.Span : default;
      Span<FP16> fullL2 = fullBatchResult.HasValueSecondary ? fullBatchResult.L2.Span : default;

      Span<FP16> fullUncertaintyV = fullBatchResult.HasUncertaintyV ? fullBatchResult.UncertaintyV.Span : default;
      Span<FP16> fullUncertaintyP = fullBatchResult.HasUncertaintyP ? fullBatchResult.UncertaintyP.Span : default;
      Span<Half> fullState = default;
      Span<FP16> fullExtraStat0 = fullBatchResult.ExtraStat0.Span;
      Span<FP16> fullExtraStat1 = fullBatchResult.ExtraStat1.Span;
      Span<NNEvaluatorResultActivations> fullActivations = fullBatchResult.Activations.IsEmpty ? null : fullBatchResult.Activations.Span;

      // Finally, disaggregate the big batch back into a set of individual subbatch results
      PositionEvaluationBatch[] completedBatches = new PositionEvaluationBatch[pendingBatches.Count];

      int subBatchIndex = 0;
      int nextPosIndex = 0;
      foreach (EncodedPositionBatchFlat thisBatch in pendingBatches)
      {
        if (retrieveSupplementalResults) throw new NotImplementedException();

        if (fullBatchResult.HasState)
        {
          throw new NotImplementedException("State not implemented below");
        }
        int numPos = thisBatch.NumPos;
        PositionEvaluationBatch thisResultSubBatch =
          new PositionEvaluationBatch(fullBatchResult.IsWDL, fullBatchResult.HasM, 
                                      fullBatchResult.HasUncertaintyV, fullBatchResult.HasUncertaintyP,
                                      fullBatchResult.HasAction, fullBatchResult.HasValueSecondary, fullBatchResult.HasState,
                                      thisBatch.NumPos,
                                      fullPolicyValues.Slice(nextPosIndex, numPos).ToArray(),
                                      fullBatchResult.HasAction ? fullActionValues.Slice(nextPosIndex, numPos).ToArray() : default,

                                      fullW.Slice(nextPosIndex, numPos).ToArray(),
                                      fullBatchResult.IsWDL ? fullL.Slice(nextPosIndex, numPos).ToArray() : null,

                                      fullBatchResult.HasValueSecondary ? fullW2.Slice(nextPosIndex, numPos).ToArray() : null,
                                      fullBatchResult.HasValueSecondary && fullBatchResult.IsWDL ? fullL2.Slice(nextPosIndex, numPos).ToArray() : null,

                                      fullBatchResult.HasM ? fullM.Slice(nextPosIndex, numPos).ToArray() : null,
                                      fullBatchResult.HasUncertaintyV ? fullUncertaintyV.Slice(nextPosIndex, numPos).ToArray() : null,
                                      fullBatchResult.HasUncertaintyP ? fullUncertaintyP.Slice(nextPosIndex, numPos).ToArray() : null,
                                      fullBatchResult.HasState ? default : default, // ** TO DO
                                      fullBatchResult.Activations.IsEmpty ? null : fullActivations.Slice(nextPosIndex, numPos).ToArray(),
                                      fullBatchResult.Stats,
                                      fullExtraStat0.IsEmpty ? default : fullExtraStat0.Slice(nextPosIndex, numPos).ToArray(),
                                      fullExtraStat1.IsEmpty ? default : fullExtraStat1.Slice(nextPosIndex, numPos).ToArray());

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
