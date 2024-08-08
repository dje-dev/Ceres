#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Static helper class to convert EncodedTrainingPosition to corresponding TPGRecord.
  /// </summary>
  public static class TPGRecordConverter
  {
    static bool IsInvalid(float f) => float.IsNaN(f) || float.IsInfinity(f);
    static bool IsInvalid((float w, float d, float l) item) => float.IsNaN(item.w + item.d + item.l)
                                                            || float.IsInfinity(item.w + item.d + item.l)
                                                            || item.w + item.d + item.l < 0.999
                                                            || item.w + item.d + item.l > 1.001;


    /// <summary>
    /// Converter from EncodedTrainingPosition to TPGRecord (fast version for inference, not supporting setting of training target info).
    /// </summary>
    /// <param name="encodedPosToConvert"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static TPGRecord ConvertedToTPGRecord(in EncodedPositionWithHistory encodedPosToConvert,
                                                 bool includeHistory,
                                                 Span<byte> pliesSinceLastPieceMoveBySquare = default,
                                                 float qNegativeBlunders = 0, float qPositiveBlunders = 0,
                                                 float priorPosWinP = 0, float priorPosDrawP = 0, float priorPosLossP = 0)
    {
#if MOVES
        throw new NotImplementedException();
#endif

      // N.B. Some logic here is the same as in method below (ConvertToTPGRecord) and should be kept in sync.
      TPGRecord tpgRecord = default;

      // Write squares.
      ConvertToTPGRecordSquares(in encodedPosToConvert, includeHistory, 
                                ref tpgRecord, pliesSinceLastPieceMoveBySquare, pliesSinceLastPieceMoveBySquare != default,
                                qNegativeBlunders, qPositiveBlunders,
                                priorPosWinP, priorPosDrawP, priorPosLossP); 

      return tpgRecord;
    }


    public static unsafe void ConvertToTPGRecord(in EncodedPositionWithHistory encodedPosToConvert,
                                                 bool includeHistory,
                                                 Memory<MGMoveList> moves,
                                                 TPGTrainingTargetNonPolicyInfo? targetInfo,
                                                 CompressedPolicyVector? policyVector,
                                                 float minLegalMoveProbability,
                                                 ref TPGRecord tpgRecord,
                                                 Span<byte> pliesSinceLastPieceMoveBySquare,
                                                 bool emitPlySinceLastMovePerSquare,
                                                 float qNegativeBlunders = 0,
                                                 float qPositiveBlunders = 0,
                                                 float priorPosWinP = 0, 
                                                 float priorPosDrawP = 0,
                                                 float priorPosLossP = 0,
                                                 bool validate = true)
    {
      // N.B. Some logic here is the same as in method above (ConvertedToTPGRecord) and should be kept in sync.

      // Clear out any prior values.
      tpgRecord = default;
      
      // Write squares.
#if DISABLED_MIRROR
      // TODO: we mirror the position here to match expectation of trained net based on 
      //       LC0 training data (which is mirrored). Someday undo this mirroring in training and then can remove here.
      EncodedPositionWithHistory encodedPosToConvertMirrored = encodedPosToConvert.Mirrored;
#endif
      ConvertToTPGRecordSquares(in encodedPosToConvert, includeHistory, ref tpgRecord, 
                                pliesSinceLastPieceMoveBySquare, emitPlySinceLastMovePerSquare,
                                qNegativeBlunders, qPositiveBlunders,
                                priorPosWinP, priorPosDrawP, priorPosLossP);

      // Convert the values unrelated to moves and squares
      if (targetInfo != null)
      {
        ConvertToTPGEvalInfo(targetInfo.Value, ref tpgRecord, validate);
      }

      if (policyVector is not null)
      {
        ConvertToTPGRecordPolicies(in policyVector, minLegalMoveProbability, ref tpgRecord);
      }

#if MOVES
      if (emitMoves)
      {
        Position finalPos = encodedPosToConvert.FinalPosition;
        if (!finalPos.IsWhite)
        {
          finalPos = finalPos.Reversed;
        }

        TPGMoveRecord.WriteMoves(finalPos.Mirrored, tpgRecord.MovesSetter, default, default);
      }
#endif

#if DEBUG
      if (validate)
      {
        TPGRecordValidation.Validate(in encodedPosToConvert, in tpgRecord, policyVector is not null);
      }
#endif

    }


    static bool haveInitialized = false;

    const int MAX_TPG_RECORDS_PER_BUFFER = 4096;

    [ThreadStatic]
    static TPGRecord[] tempTPGRecords;

    static int countConversions = 0;


    /// <summary>
    /// Converts input positions defined as IEncodedPositionFlat 
    /// into raw square and move bytes used by the TPGRecord.
    /// </summary>
    /// <param name="positions">underlying batch of positions</param>
    /// <param name="includeHistory">if history planes should be filled in if necessary</param>
    /// <param name="moves">list of legal moves for each position</param>
    /// <param name="lastMovePliesEnabled"></param>
    /// <param name="qNegativeBlunders">value for expected forward downside blunder to inject</param>
    /// <param name="qPositiveBlunders">value for expected forward upside blunder to inject</param>
    /// <param name="mgPos">current position</param>
    /// <param name="squareBytesAll">byte array to receive converted encoded positions in TPGSquareRecord format</param>
    /// <param name="legalMoveIndices">optional array to recieve indices of legal moves in position</param>
    /// <exception cref="Exception"></exception>
    public static void ConvertPositionsToRawSquareBytes(IEncodedPositionBatchFlat positions,
                                                        bool includeHistory,
                                                        Memory<MGMoveList> moves,
                                                        bool lastMovePliesEnabled,
                                                        float qNegativeBlunders,
                                                        float qPositiveBlunders,
                                                        out MGPosition[] mgPos,
                                                        out byte[] squareBytesAll,
                                                        short[] legalMoveIndices)
    {
#if DEBUG
      short[] legalMoveIndicesAlternate = null;
      const int FREQUENCY_VERIFY_MOVE_INDICES = 100;
      bool verifyMoveIndices = countConversions++ % FREQUENCY_VERIFY_MOVE_INDICES == 0;

      if (verifyMoveIndices) 
      {
        // Old slow code, replaced below with help of the passed in moves argument.
        legalMoveIndicesAlternate = new short[positions.NumPos * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];
      }
#endif

      if (legalMoveIndices != null)
      {
        // Reset legalMoveIndices back to 0.
        Array.Clear(legalMoveIndices, 0, positions.NumPos * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST);
      }

      // Get all positions from input batch.
      // TODO: Improve efficiency, these array materializations are expensive.
      Memory<EncodedPositionWithHistory> positionsFlat = positions.PositionsBuffer;
      if (positions.PositionsBuffer.IsEmpty)
      {
        throw new Exception("PositionsBuffer not initialized, EncodedPositionBatchFlat.RETAIN_POSITIONS_INTERNALS needs to be set true");
      }
      mgPos = positions.Positions.ToArray();
      byte[] pliesSinceLastMoveAllPositions = positions.LastMovePlies.ToArray();

      squareBytesAll = new byte[positions.NumPos * Marshal.SizeOf<TPGSquareRecord>() * 64];
      byte[] squareBytesAllCopy = squareBytesAll;

      // Determine each position and copy converted raw board bytes into rawBoardBytesAll.
      // TODO: for efficiency, avoid doing this if the NN evaluator does not need raw bytes
      const int MAX_THREADS = 8;
      Parallel.For(0, positions.NumPos, new ParallelOptions() { MaxDegreeOfParallelism = MAX_THREADS }, i =>

      //for (int i = 0; i < positions.NumPos; i++)
      {
        if (tempTPGRecords == null)
        {
          tempTPGRecords = new TPGRecord[MAX_TPG_RECORDS_PER_BUFFER]; // ThreadStatic
        }

        if (!lastMovePliesEnabled)
        {
          // Disable any values possibly passed for last used plies since they are not to be used.
          pliesSinceLastMoveAllPositions = null;
        }

        TPGRecord tpgRecord = default;
        Span<byte> thesePliesSinceLastMove = pliesSinceLastMoveAllPositions == null ? default : new Span<byte>(pliesSinceLastMoveAllPositions, i * 64, 64);

        float w = TPGRecordEncoding.ENABLE_PRIOR_VALUE_POSITION ? positions.PositionsBuffer.Span[i].MiscInfo.InfoTraining.OriginalQ : float.NaN;
        float d = TPGRecordEncoding.ENABLE_PRIOR_VALUE_POSITION ? positions.PositionsBuffer.Span[i].MiscInfo.InfoTraining.OriginalD : float.NaN;
        float l = TPGRecordEncoding.ENABLE_PRIOR_VALUE_POSITION ? positions.PositionsBuffer.Span[i].MiscInfo.InfoTraining.OriginalM : float.NaN;

        ConvertToTPGRecord(in positionsFlat.Span[i], includeHistory, moves, null, null, float.NaN,
                           ref tpgRecord, thesePliesSinceLastMove, lastMovePliesEnabled, 
                           qNegativeBlunders, qPositiveBlunders,
                           w, d, l);
                           
        tempTPGRecords[i] = tpgRecord;

        const bool VALIDITY_CHECK = true;
        if (VALIDITY_CHECK && pliesSinceLastMoveAllPositions != null)
        {
          tpgRecord = CheckPliesSinceLastMovedCorrect(tpgRecord);
        }

        // Extract as bytes.
        tpgRecord.CopySquares(squareBytesAllCopy, i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);

        if (legalMoveIndices != null)
        {
          int numMoves = Math.Min(TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST, moves.Span[i].NumMovesUsed);
          int indexOfMoveWithNNIndex0 = -1;
          Span<short> thisPositionLegalMoveIndices = new Span<short>(legalMoveIndices, i * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST, TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST);

          for (int moveNum = 0; moveNum < numMoves; moveNum++)
          {
            MGMove thisMove = moves.Span[i].MovesArray[moveNum];
            short nnIndex = (short)ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(thisMove).IndexNeuralNet;
            thisPositionLegalMoveIndices[moveNum] = (short)nnIndex;

            if (nnIndex == 0)
            {
              indexOfMoveWithNNIndex0 = moveNum;
            }
          }

          // Since we use index 0 as a sentinel (unless appears in first slot),
          // if index 0 actually is present in the legal moves, swap it into the first slot.
          if (indexOfMoveWithNNIndex0 != -1)
          {
            // Move index 0 must be first.
            short temp = thisPositionLegalMoveIndices[0];
            thisPositionLegalMoveIndices[0] = thisPositionLegalMoveIndices[indexOfMoveWithNNIndex0];
            thisPositionLegalMoveIndices[indexOfMoveWithNNIndex0] = temp;
          }

#if DEBUG
          if (verifyMoveIndices)
          {
            // Old slow code, replaced below with help of the passed in moves argument.
            TPGRecordMovesExtractor.ExtractLegalMoveIndicesForIndex(tempTPGRecords, moves.Span[i], legalMoveIndicesAlternate, i);
          }
#endif
        }
      });

#if DEBUG
      if (verifyMoveIndices && legalMoveIndices != null)
      {
        DebugVerifyCorrectLegalMoveIndices(positions, legalMoveIndices, legalMoveIndicesAlternate);
      }
#endif

    }


    /// <summary>
    /// Diagnostic method to verify that two methods of extracting legal move indices are consistent.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="legalMoveIndices"></param>
    /// <param name="legalMoveIndicesAlternate"></param>
    /// <exception cref="Exception"></exception>
    private static void DebugVerifyCorrectLegalMoveIndices(IEncodedPositionBatchFlat positions, 
                                                           short[] legalMoveIndices,
                                                           short[] legalMoveIndicesAlternate)
    {
      short[] copyLegalMoveIndices = new short[TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];
      short[] copyLegalMoveIndicesAlternate = new short[TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];

      // Because 0 is reserved sentinel, don't expect it in first slot
      // (unless we already see that in the correct copy, indicating no legal moves).
      for (int i = 0; i < positions.NumPos; i++)
      {
        ReadOnlySpan<short> thisPositionLegalMoveIndices = new Span<short>(legalMoveIndices, i * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST, TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST);
        ReadOnlySpan<short> thisPositionLegalMoveIndicesAlternate = new Span<short>(legalMoveIndicesAlternate, i * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST, TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST);

        // This test disabled. This not actually required for Lc0 nets
        // because they don't use gather/scatter on legal move indices.
        //Debug.Assert(!(thisPositionLegalMoveIndices[0] == 0 && legalMoveIndicesAlternate[0] != 0));

        // Compare the two methods of extracting legal move indices.
        // Create copies and sort them to facilitate comparison.
        thisPositionLegalMoveIndices.CopyTo(copyLegalMoveIndices);
        thisPositionLegalMoveIndicesAlternate.CopyTo(copyLegalMoveIndicesAlternate);
        Array.Sort(copyLegalMoveIndices);
        Array.Sort(copyLegalMoveIndicesAlternate);

        for (int ix = 0; ix < TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST; ix++)
        {
          if (copyLegalMoveIndices[ix] != copyLegalMoveIndicesAlternate[ix])
          {
            throw new Exception("Mismatch in legal move indices");
          }
        }
      }
    }

    private static TPGRecord CheckPliesSinceLastMovedCorrect(TPGRecord tpgRecord)
    {
      throw new NotImplementedException();
    }


    public static unsafe void ConvertToTPGRecord(in EncodedTrainingPosition trainingPos,
                                                 bool includeHistory,
                                                 in TPGTrainingTargetNonPolicyInfo targetInfo,
                                                 CompressedPolicyVector? overridePolicyVector,
                                                 float minLegalMoveProbability,
                                                 ref TPGRecord tpgRecord,
                                                 Span<byte> pliesSinceLastPieceMoveBySquare,
                                                 bool emitPlySinceLastMovePerSquare,
                                                 float qNegativeBlunders, float qPositiveBlunders,
                                                 float priorPosWinP, float priorPosDrawP, float priorPosLossP,
                                                 bool validate)
                                                 
    {
      if (validate)
      {
        trainingPos.ValidateIntegrity("Validate in ConvertToTPGRecord");
      }

      // Clear out any prior values.
      tpgRecord = default;

      // Convert policies.
      if (overridePolicyVector is not null)
      {
        throw new NotImplementedException(); // see the else below
        ConvertToTPGRecordPolicies(in overridePolicyVector, minLegalMoveProbability, ref tpgRecord);
      }
      else
      {
        // Note that ConvertToTPGRecordPolicies is called first.
        // This will initialize the tpgRecord.Policy which is then referenced in WriteMoves below.
        ConvertToTPGRecordPolicies(in trainingPos, minLegalMoveProbability, ref tpgRecord);

#if MOVES
        if (emitMoves)
        {
          TPGMoveRecord.WriteMoves(in thisPosition, tpgRecord.MovesSetter, tpgRecord.Policy, tpgRecord.PolicyForMoves);
        }
#endif

      }

      // Convert the values unrelated to moves and squares
      ConvertToTPGEvalInfo(in targetInfo, ref tpgRecord, validate);

#if DEBUG
      Debug.Assert(qNegativeBlunders == targetInfo.ForwardSumNegativeBlunders);
      Debug.Assert(qPositiveBlunders == targetInfo.ForwardSumPositiveBlunders);
#endif

      // Write squares.
      ConvertToTPGRecordSquares(trainingPos.PositionWithBoards, includeHistory, ref tpgRecord,
                                pliesSinceLastPieceMoveBySquare, emitPlySinceLastMovePerSquare,
                                qNegativeBlunders, qPositiveBlunders,
                                priorPosWinP, priorPosDrawP, priorPosLossP);

#if DEBUG
      TPGRecordValidation.Validate(in trainingPos.PositionWithBoards, in tpgRecord, overridePolicyVector is not null);
#endif   
    }


    internal static unsafe void ConvertToTPGRecordPolicies(in CompressedPolicyVector? policyVector,
                                                           float minLegalMoveProbability,
                                                           ref TPGRecord tpgRecord)
    {
      if (policyVector is null)
      {
        throw new ArgumentNullException(nameof(policyVector));
      }

      int count = 0;
      foreach ((EncodedMove move, float probability) in policyVector.Value.ProbabilitySummary())
      {
        if (count < TPGRecord.MAX_MOVES)
        {
          float probAdjusted = MathF.Max(minLegalMoveProbability, probability);
          tpgRecord.PolicyValues[count] = (Half)probAdjusted;
          tpgRecord.PolicyIndices[count] = (short)move.IndexNeuralNet;
          count++;
        }
      }

      PadPolicies(ref tpgRecord, count);
    }


    /// <summary>
    /// Postprocesses policy array in a TPGRecord to set unused slots
    /// such that future scatter operations across the full array will be correct
    /// (replicate last entry).
    /// </summary>
    /// <param name="tpgRecord"></param>
    /// <param name="count"></param>
    unsafe static void PadPolicies(ref TPGRecord tpgRecord, int count)
    {
      if (count > 0)
      {
        if (count < TPGRecord.MAX_MOVES)
        {
          // Replicate the last entry into every remaining position
          // so that scattering across all slots will result in correct policy.
          short lastIndex = tpgRecord.PolicyIndices[count - 1];
          Half lastValue = tpgRecord.PolicyValues[count - 1];
          while (count < TPGRecord.MAX_MOVES)
          {
            tpgRecord.PolicyValues[count] = lastValue;
            tpgRecord.PolicyIndices[count] = lastIndex;
            count++;
          }
        }
      }
    }


    internal static unsafe void ConvertToTPGRecordPolicies(in EncodedTrainingPosition trainingPos, float minLegalMoveProbability, ref TPGRecord tpgRecord)
    {
      // TODO: speed up. Check two at a time within loop for better parallelism?
      float* probabilitiesSource = &trainingPos.Policies.ProbabilitiesPtr[0];

      float incrementFromValuesSetAtMin = 0;
      int count = 0;
      for (int i = 0; i < 1858; i++)
      {
        if (count < TPGRecord.MAX_MOVES)
        {
          float prob = probabilitiesSource[i];
          if (prob >= 0)
          {
            if (prob < minLegalMoveProbability)
            {
              incrementFromValuesSetAtMin += minLegalMoveProbability - prob;
              prob = minLegalMoveProbability;
            }

            tpgRecord.PolicyIndices[count] = (short)i;
            tpgRecord.PolicyValues[count] = (Half)prob;
            count++;
          }
        }
      }

      // Restore "sum to 1.0" property if necessary.
      const float MAX_INCREMENT_ALLOWED_BEFORE_RENORMALIZE = 0.005f; // for efficiency ignore if very small deviation from 1.0
      if (incrementFromValuesSetAtMin > MAX_INCREMENT_ALLOWED_BEFORE_RENORMALIZE)
      {
        float multiplier = 1.0f / (1 + incrementFromValuesSetAtMin);
        for (int i = 0; i < count; i++)
        {
          float current = (float)tpgRecord.PolicyValues[i];
          tpgRecord.PolicyValues[i] = (Half)(current * multiplier);
        }
      }

      PadPolicies(ref tpgRecord, count);
    }

    static int convertCount;

    public static unsafe void ConvertToTPGRecordSquares(in EncodedPositionWithHistory posWithHistory,
                                                        bool includeHistory,
                                                        ref TPGRecord tpgRecord,
                                                        Span<byte> pliesSinceLastPieceMoveBySquare,
                                                        bool emitPlySinceLastMovePerSquare,
                                                        float qNegativeBlunders, float qPositiveBlunders,
                                                        float priorPosWinP, float priorPosDrawP, float priorPosLossP)
    {
      static Position GetHistoryPosition(in EncodedPositionWithHistory historyPos, int index, in Position? fillInIfEmpty)
      {
        Position pos = historyPos.HistoryPositionIsEmpty(index) ? default
                                                                : historyPos.HistoryPosition(index);

#if NOT
        // NOTE: This is only to keep compatability with previously written TPG files.
        //       Someday consider backing this out (also in decode methods).
        pos = pos.Mirrored; 
#endif

        if (pos.PieceCount == 0 && fillInIfEmpty != null)
        {
          pos = fillInIfEmpty.Value;
        }

        // Put position in same perspective as final position (index 0).
        if (pos.IsWhite != (index % 2 == 0))
        {
          pos = pos.Reversed;
        }
        return pos;
      }

      tpgRecord.IsWhiteToMove = posWithHistory.MiscInfo.WhiteToMove ? (byte)1 : (byte)0;

      const bool FILL_IN = true;
      Position thisPosition = GetHistoryPosition(in posWithHistory, 0, null);
      Position historyPos1 = GetHistoryPosition(in posWithHistory, 1, FILL_IN ? thisPosition : null);
      Position historyPos2 = GetHistoryPosition(in posWithHistory, 2, FILL_IN ? historyPos1 : null);
      Position historyPos3 = GetHistoryPosition(in posWithHistory, 3, FILL_IN ? historyPos2 : null);
      Position historyPos4 = GetHistoryPosition(in posWithHistory, 4, FILL_IN ? historyPos3 : null);
      Position historyPos5 = GetHistoryPosition(in posWithHistory, 5, FILL_IN ? historyPos4 : null);
      Position historyPos6 = GetHistoryPosition(in posWithHistory, 6, FILL_IN ? historyPos5 : null);
      Position historyPos7 = GetHistoryPosition(in posWithHistory, 7, FILL_IN ? historyPos6 : null);

      if (!includeHistory)
      {
        // TODO: make more efficient
        if (FILL_IN)
        {
          historyPos1 = historyPos2 = historyPos3 = historyPos4 = historyPos5 = historyPos6 = historyPos6 = historyPos7 = thisPosition;
        }
        else
        {
          historyPos1 = historyPos2 = historyPos3 = historyPos4 = historyPos5 = historyPos6 = historyPos6 = historyPos7 = default;
        }
      }

      // Write squares.
      TPGSquareRecord.WritePosPieces(in thisPosition, in historyPos1, in historyPos2, in historyPos3,
                                     in historyPos4, in historyPos5, in historyPos6, in historyPos7,
                                     tpgRecord.Squares, pliesSinceLastPieceMoveBySquare, emitPlySinceLastMovePerSquare,
                                     qNegativeBlunders, qPositiveBlunders,
                                     priorPosWinP, priorPosDrawP, priorPosLossP);

#if DEBUG
      const int VALIDATE_FREQUENCY = 100; // too slow to do every time
      bool validate = convertCount++ % VALIDATE_FREQUENCY == 0;
      if (validate)
      {
        TPGRecordValidation.ValidateHistoryReachability(in tpgRecord);
        TPGRecordValidation.ValidateSquares(in posWithHistory, ref tpgRecord);
      }
#endif
    }



    /// <summary>
    /// Extracts TPGWriterNonPolicyTargetInfo into a TPGRecord.
    /// </summary>
    /// <param name="targetInfo"></param>
    /// <param name="tpgRecord"></param>
    /// <exception cref="Exception"></exception>
    internal static unsafe void ConvertToTPGEvalInfo(in TPGTrainingTargetNonPolicyInfo targetInfo, ref TPGRecord tpgRecord, bool validate)
    {
      if (validate)
      {
        if (IsInvalid(targetInfo.ResultDeblunderedWDL)) throw new Exception("Bad ResultDeblunderedWDL " + targetInfo.ResultDeblunderedWDL);
        if (IsInvalid(targetInfo.ResultNonDeblunderedWDL)) throw new Exception("Bad ResultNonDeblunderedWDL " + targetInfo.ResultNonDeblunderedWDL);
        if (IsInvalid(targetInfo.BestWDL)) throw new Exception("Bad BestWDL " + targetInfo.BestWDL);
        if (IsInvalid(targetInfo.MLH)) throw new Exception("Bad MLH " + targetInfo.MLH);
        if (IsInvalid(targetInfo.DeltaQVersusV)) throw new Exception("Bad UNC " + targetInfo.DeltaQVersusV);
        if (IsInvalid(targetInfo.KLDPolicy)) throw new Exception("Bad KLDPolicy " + targetInfo.KLDPolicy); 
      }

      tpgRecord.WDLResultNonDeblundered[0] = targetInfo.ResultNonDeblunderedWDL.w;
      tpgRecord.WDLResultNonDeblundered[1] = targetInfo.ResultNonDeblunderedWDL.d;
      tpgRecord.WDLResultNonDeblundered[2] = targetInfo.ResultNonDeblunderedWDL.l;

      tpgRecord.WDLResultDeblundered[0] = targetInfo.ResultDeblunderedWDL.w;
      tpgRecord.WDLResultDeblundered[1] = targetInfo.ResultDeblunderedWDL.d;
      tpgRecord.WDLResultDeblundered[2] = targetInfo.ResultDeblunderedWDL.l;

      tpgRecord.WDLQ[0] = targetInfo.BestWDL.w;
      tpgRecord.WDLQ[1] = targetInfo.BestWDL.d;
      tpgRecord.WDLQ[2] = targetInfo.BestWDL.l;

      tpgRecord.MLH = targetInfo.MLH;
      tpgRecord.DeltaQVersusV = targetInfo.DeltaQVersusV;

      // Note that suboptimality will almost always be positive, but take absolute value to be sure.
      // (to compensate for numerical rounding or the chosen best N move not being quite best Q).
      tpgRecord.PlayedMoveQSuboptimality = MathF.Abs(MathF.Round(targetInfo.PlayedMoveQSuboptimality, 3));

      tpgRecord.KLDPolicy = targetInfo.KLDPolicy;

      tpgRecord.QDeviationLower = (Half)targetInfo.ForwardMinQDeviation;
      tpgRecord.QDeviationUpper = (Half)targetInfo.ForwardMaxQDeviation;


      tpgRecord.PolicyIndexInParent = targetInfo.PolicyIndexInParent;

    }
  }
}
