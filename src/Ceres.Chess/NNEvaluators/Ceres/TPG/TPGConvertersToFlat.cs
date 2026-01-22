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
using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
//using CeresTrain.NNEvaluators;
using Ceres.Chess.NNEvaluators.Ceres.TPG;

#endregion 

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Static helper methods to convert TPGRecord to flat square format.
  /// </summary>
  public static class TPGConvertersToFlat
  {
    /// <summary>
    /// Converts a batch of TPGRecord[] into TPG flat square values.
    /// </summary>
    /// <param name="records"></param>
    /// <param name="flatValues"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static int ConvertToFlatTPGFromTPG(NNEvaluatorOptions options, object records, Span<byte> flatValues)
    {
      NNEvaluatorOptionsCeres optionsCeres = (NNEvaluatorOptionsCeres)options;

      // TODO: Requiring the converter to take a materialized array could be inefficient, can we use Memory instead?
      TPGRecord[] tpgRecords = records as TPGRecord[];
      if (tpgRecords == null)
      {
        throw new NotImplementedException("Expected input to be TPGRecord[]");
      }

      byte[] squareBytesAll = new byte[tpgRecords.Length * Marshal.SizeOf<TPGSquareRecord>() * 64];

      for (int i = 0; i < tpgRecords.Length; i++)
      {
        int offsetSquares = i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD;

        // Extract as bytes.
        tpgRecords[i].CopySquares(squareBytesAll, offsetSquares);
      }

      // N.B. Scaling (by 100) already done in ONXNRuntimeExecutor (but probably doesn't belong there?).
      // TODO: This could be made more efficient, fold into the above loop.
      for (int i = 0; i < squareBytesAll.Length; i++)
      {
        flatValues[i] = squareBytesAll[i];
      }

      return squareBytesAll.Length;
    }


    /// <summary>
    /// Copies sourceBytes into targetFloats, also dividing by divisor.
    /// </summary>
    /// <param name="sourceBytes"></param>
    /// <param name="targetHalves"></param>
    static void CopyAndDivide(Memory<byte> sourceBytes, Memory<Half> targetHalves, float divisor)
    {
      CopyAndDivideSIMD(sourceBytes, targetHalves, divisor);
#if NOT
      // Disabled. Incompatible with the MemoryHandle.Pin used in CopyAndDivideSIMD that objects to re-pinning.
      // Also the parallelism may not be very helpful or needed given that the common path now uses
      // byte inputs and completely avoids this code path.
      const int CHUNK_SIZE = 2 * 1024 * 128;
      if (sourceBytes.Length >= CHUNK_SIZE * 2)
      {
        Parallel.For(0, sourceBytes.Length / CHUNK_SIZE + 1,
                     new ParallelOptions()
                     {
                       MaxDegreeOfParallelism = 5 // limit parallelism because already threaded if multiple GPUs
                     },
          (chunkIndex) =>
          {
            int startIndex = chunkIndex * CHUNK_SIZE;
            int numThisBlock = Math.Min(CHUNK_SIZE, sourceBytes.Length - startIndex);

            if (numThisBlock > 0)
            {
              Memory<byte> sourceBytesThisBlock = sourceBytes.Slice(startIndex, numThisBlock);
              Memory<Half> targetHalvesThisBlock = targetHalves.Slice(startIndex, numThisBlock);
              CopyAndDivideSIMD(sourceBytesThisBlock, targetHalvesThisBlock, divisor);
            }
          });
      }
#endif
    }


    /// <summary>
    /// Copies sourceBytes into targetFloats, also dividing by divisor.
    /// </summary>
    /// <param name="sourceBytes"></param>
    /// <param name="targetHalfs"></param>
    internal static unsafe void CopyAndDivideSIMD(Memory<byte> sourceBytes, Memory<Half> targetHalfs, float divisor)
    {
      int vectorSize = Vector256<byte>.Count;
      int i = 0;

      if (Avx2.IsSupported)
      {
        Vector256<float> divisorVec = Vector256.Create(divisor);
        using MemoryHandle sourceHandle = sourceBytes.Pin();
        using MemoryHandle targetHandle = targetHalfs.Pin();

        byte* squareBytesAllPtr = (byte*)sourceHandle.Pointer;
        ushort* targetFlatValuesPrimaryPtr = (ushort*)targetHandle.Pointer;

        // Process in chunks of 32 bytes
        for (i = 0; i <= sourceBytes.Length - vectorSize; i += vectorSize)
        {
          // Load 32 bytes from the byte array
          Vector256<byte> byteVec = Avx.LoadVector256(&squareBytesAllPtr[i]);

          // Convert the 32 bytes to 32 floats
          Vector256<short> ushortLow = Avx2.ConvertToVector256Int16(byteVec.GetLower());
          Vector256<float> floatLowLow = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortLow.GetLower()));
          Vector256<float> floatLowHigh = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortLow.GetUpper()));
          Vector256<uint> r1 = SingleToHalfAsWidenedUInt32_Vector256(floatLowLow, divisorVec);
          Vector256<uint> r2 = SingleToHalfAsWidenedUInt32_Vector256(floatLowHigh, divisorVec);
          Vector256<ushort> source2 = Vector256.Narrow(r1, r2);
          Avx.Store(&targetFlatValuesPrimaryPtr[i], source2);

          Vector256<short> ushortHigh = Avx2.ConvertToVector256Int16(byteVec.GetUpper());
          Vector256<float> floatHighLow = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortHigh.GetLower()));
          Vector256<float> floatHighHigh = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortHigh.GetUpper()));
          Vector256<uint> r1H = SingleToHalfAsWidenedUInt32_Vector256(floatHighLow, divisorVec);
          Vector256<uint> r2H = SingleToHalfAsWidenedUInt32_Vector256(floatHighHigh, divisorVec);
          Vector256<ushort> source2H = Vector256.Narrow(r1H, r2H);

          // Store the results
          Avx.Store(&targetFlatValuesPrimaryPtr[i + 16], source2H);
        }
      }
#if NOT
      else if (AdvSimd.IsSupported)
      {
        throw new NotImplementedException("this code is not yet tested");

        Vector128<float> divisorVec = Vector128.Create(divisor);
        Vector128<float> reciprocalDivisorVec = AdvSimd.ReciprocalEstimate(divisorVec);
        fixed (byte* sourceBytesPtr = sourceBytes)
        fixed (Half* targetFloatsPtr = targetFloats)
        {
          for (i = 0; i <= sourceBytes.Length - vectorSize; i += vectorSize)
          {
            // Load 16 bytes from the byte array
            Vector128<byte> byteVec = AdvSimd.LoadVector128(&sourceBytesPtr[i]);

            // Split the vector into two 64-bit parts
            Vector64<byte> byteVecLower = byteVec.GetLower();
            Vector64<byte> byteVecUpper = byteVec.GetUpper();

            // Zero extend the lower and upper parts
            Vector128<ushort> ushortVecLower = AdvSimd.ZeroExtendWideningLower(byteVecLower);
            Vector128<ushort> ushortVecUpper = AdvSimd.ZeroExtendWideningLower(byteVecUpper);

            // Zero extend the widened vectors to 32-bit integers
            Vector128<uint> uintVecLower1 = AdvSimd.ZeroExtendWideningLower(ushortVecLower.GetLower());
            Vector128<uint> uintVecLower2 = AdvSimd.ZeroExtendWideningUpper(ushortVecLower);
            Vector128<uint> uintVecUpper1 = AdvSimd.ZeroExtendWideningLower(ushortVecUpper.GetLower());
            Vector128<uint> uintVecUpper2 = AdvSimd.ZeroExtendWideningUpper(ushortVecUpper);

            // Convert the 32-bit integers to single-precision floats
            Vector128<float> floatVecLower1 = AdvSimd.ConvertToSingle(uintVecLower1);
            Vector128<float> floatVecLower2 = AdvSimd.ConvertToSingle(uintVecLower2);
            Vector128<float> floatVecUpper1 = AdvSimd.ConvertToSingle(uintVecUpper1);
            Vector128<float> floatVecUpper2 = AdvSimd.ConvertToSingle(uintVecUpper2);

            // Divide by the divisor using reciprocal approximation
            Vector128<float> resultVecLower1 = AdvSimd.Multiply(floatVecLower1, reciprocalDivisorVec);
            Vector128<float> resultVecLower2 = AdvSimd.Multiply(floatVecLower2, reciprocalDivisorVec);
            Vector128<float> resultVecUpper1 = AdvSimd.Multiply(floatVecUpper1, reciprocalDivisorVec);
            Vector128<float> resultVecUpper2 = AdvSimd.Multiply(floatVecUpper2, reciprocalDivisorVec);

            // Store the results
            AdvSimd.Store(&targetFloatsPtr[i], resultVecLower1);
            AdvSimd.Store(&targetFloatsPtr[i + 4], resultVecLower2);
            AdvSimd.Store(&targetFloatsPtr[i + 8], resultVecUpper1);
            AdvSimd.Store(&targetFloatsPtr[i + 12], resultVecUpper2);
          }
        }
      }
#endif
      // Process remaining elements (15x slower than vectorized).
      Span<byte> sourceSpan = sourceBytes.Span;
      Span<Half> targetSpan = targetHalfs.Span;
      for (; i < sourceBytes.Length; i++)
      {
        targetSpan[i] = (Half)(sourceSpan[i] / divisor);
      }

    }

    /// <summary>
    /// Converts a float (after a division by a constant) to a half-precision float using vectorized instructions.
    /// 
    /// Code heavily based on .NET runtime (System.Numerics.Tensors.TensorPrimitives)
    /// </summary>
    /// <param name="value"></param>
    /// <param name="divisorVec"></param>
    /// <returns></returns>
    static Vector256<uint> SingleToHalfAsWidenedUInt32_Vector256(Vector256<float> value, Vector256<float> divisorVec)
    {
      value = Avx.Divide(value, divisorVec);
      Vector256<uint> vector8 = value.AsUInt32();
      Vector256<uint> vector9 = Vector256.ShiftRightLogical(vector8 & Vector256.Create(2147483648u), 16);
      Vector256<uint> vector10 = Vector256.Equals(value, value).AsUInt32();
      value = Vector256.Abs(value);
      value = Vector256.Min(Vector256.Create(65520f), value);
      Vector256<uint> vector11 = Vector256.Max(value, Vector256.Create(947912704u).AsSingle()).AsUInt32();
      vector11 &= Vector256.Create(2139095040u);
      vector11 += Vector256.Create(109051904u);
      value += vector11.AsSingle();
      vector8 = value.AsUInt32();
      Vector256<uint> vector12 = ~vector10 & Vector256.Create(31744u);
      vector8 -= Vector256.Create(1056964608u);
      Vector256<uint> vector13 = Vector256.ShiftRightLogical(vector8, 13);
      vector8 &= vector10;
      vector8 += vector13;
      vector8 &= ~vector12;
      Vector256<uint> vector14 = vector12 | vector9;
      return vector8 | vector14;
    }


    [ThreadStatic] static byte[] squareValuesByteTemporary;

    /// <summary>
    /// Converts a IEncodedPositionBatchFlat of encoded positions into TPG flat square values.
    /// </summary>
    /// <param name="options"></param>
    /// <param name="batch"></param>
    /// <param name="includeHistory"></param>
    /// <param name="squareValues"></param>
    /// <exception cref="NotImplementedException"></exception>
    public static void ConvertToFlatTPG(NNEvaluatorOptions options,
                                        IEncodedPositionBatchFlat batch,
                                        bool includeHistory, Memory<byte> squareValuesByte, Memory<Half> squareValues, short[] legalMoveIndices)
    {
      if (TPGRecord.EMIT_PLY_SINCE_LAST_MOVE_PER_SQUARE)
      {
        throw new NotImplementedException();
      }

      const bool EMIT_PLY_SINCE = false;
      TPGRecord tpgRecord = default;
      EncodedPositionBatchFlat ebf = batch as EncodedPositionBatchFlat;

      // TODO: Consider possibly restoring the commented out code below 
      //       to efficiently decode the two top positions into TPGRecord
      //       instead of having to setting EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true
      //       and incurring all that overhead.
      //       If do this, the regression/equivalency test can to be to compare
      //       this version computed here against the new more efficient code.
      // TODO: someday handle since ply, does that need to be passed in from the search engine?

      byte[] moveBytesAll;

      int numConvertedElements = TPGRecord.BYTES_PER_SQUARE_RECORD * 64 * batch.NumPos;
      bool useTemporarySqureValuesByte = squareValuesByte.IsEmpty;
      if (useTemporarySqureValuesByte)
      {
        squareValuesByteTemporary ??= new byte[137 * 64 * 1024];   // intial guess for max batch size 1024
        if (squareValuesByteTemporary.Length < numConvertedElements)
        {
          squareValuesByteTemporary ??= new byte[numConvertedElements];
        }
        squareValuesByte = squareValuesByteTemporary;
      }

      NNEvaluatorOptionsCeres optionsCeres = options as NNEvaluatorOptionsCeres;
      // TODO: consider pushing the CopyAndDivide below into this next method
      TPGRecordConverter.ConvertPositionsToRawSquareBytes(batch, includeHistory, batch.Moves, EMIT_PLY_SINCE,
                                                          optionsCeres.QNegativeBlunders, optionsCeres.QPositiveBlunders,
                                                          out _, squareValuesByte, legalMoveIndices);

      // If we are providing float inputs, then it is necessary to do the
      // (slow) convertion from Half to float (and also divide by 100).
      if (useTemporarySqureValuesByte)
      {
        CopyAndDivide(new Memory<byte>(squareValuesByteTemporary, 0, numConvertedElements),
                      squareValues, TPGSquareRecord.SQUARE_BYTES_DIVISOR);
      }


#if OLD_TPG_COMBO_DIRECT_CONVER
      static bool HAVE_WARNED = false;

      int offsetAttentionInput = 0;
      int offsetBoardInput = 0;
      for (int i = 0; i < batch.NumPos; i++)
      {
        Position pos = batch.Positions[i].ToPosition;

        TPGRecordCombo tpgRecordCombo = default;

        // NOTE: history (prior move to square) not passed here
        //        var lastMoveInfo = batch[i].PositionWithBoards.LastMoveInfoFromSideToMovePerspective();
        //        Console.WriteLine(lastMoveInfo.pieceType + " " + lastMoveInfo.fromSquare + " " + lastMoveInfo.toSquare + " " + (lastMoveInfo.wasCastle ? " ************ " : ""));
        //        int? targetSquareFromPriorMoveFromOurPerspective = lastMoveInfo.pieceType == PieceType.None ? null : lastMoveInfo.toSquare.SquareIndexStartA1;
        int? targetSquareFromPriorMoveFromOurPerspective = null;
        if (!HAVE_WARNED)
        {
          HAVE_WARNED = true;
          Console.WriteLine("WARNING: ConvertToFlatTPG does not yet set history (via targetSquareFromPriorMoveFromOurPerspective), someday pass in IEncodedPositionBatchFlat somehow");
        }


        // Get first board
        int startOffset = i * 112;
        Span<BitVector64> bvOurs0   = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset, 6));
        Span<BitVector64> bvTheirs0 = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset + 6, 6));
        EncodedPositionBoard eb0 = new EncodedPositionBoard(bvOurs0, bvTheirs0, false).Mirrored;

        // Get second board
        startOffset = i * 112 + 13;
        Span<BitVector64> bvOurs1 = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset, 6));
        Span<BitVector64> bvTheirs1 = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset + 6, 6));
        EncodedPositionBoard eb1 = new EncodedPositionBoard(bvOurs1, bvTheirs1, false).Mirrored;

        (PieceType pieceType, Square fromSquare, Square toSquare, bool wasCastle) = EncodedPositionWithHistory.LastMoveInfoFromSideToMovePerspective(in eb0, in eb1);
//Console.WriteLine("decode_LAST_MOVE " + pieceType + " " + fromSquare + " " + toSquare + " " + wasCastle + " " + pos.FEN + " " + eb1.GetFEN(pos.IsWhite) + " " + eb1.GetFEN(!pos.IsWhite));



        if (pieceType != PieceType.None)
        {
          targetSquareFromPriorMoveFromOurPerspective = pos.IsWhite ? toSquare.SquareIndexStartA1
                                                                    : toSquare.Reversed.SquareIndexStartA1;
        }

        EncodedPositionBatchFlat batchFlat = batch as EncodedPositionBatchFlat; 
        TPGRecordConverter.ConvertToTPGCombo(in pos, targetSquareFromPriorMoveFromOurPerspective, false, default, ref tpgRecordCombo);

        // TODO: Consider if we could simplify and avoid code duplication, use this method
        //   TPGRecordConverter.ConvertToTPGCombo(in EncodedTrainingPosition trainingPos, ref TPGRecordCombo tpgRecordCombo)
        // like
        //   TPGRecordConverter.ConvertToTPGCombo(in batchFlat.PositionsBuffer[i].BoardsHistory, ref tpgRecordCombo)

        float[] rawDataSquaresAndMoves = tpgRecordCombo.SquareAndMoveRawValues;
        for (int j = 0; j < rawDataSquaresAndMoves.Length; j++)
        {
          flatValuesPrimary[offsetAttentionInput++] = rawDataSquaresAndMoves[j];
        }

        if (TPGRecordCombo.NUM_RAW_BOARD_BYTES_TOTAL > 0)
        {
          TPGWriter.ExtractRawBoard(batchFlat.PositionsBuffer[i].Mirrored, ref tpgRecordCombo);
          float[] rawBoardValues = tpgRecordCombo.BoardRawValues;
          for (int j = 0; j < rawBoardValues.Length; j++)
          {
            flatValuesSecondary[offsetBoardInput++] = rawBoardValues[j];
          }
        }

      }
#endif
    }


#if BUGGY
    /// <summary>
    /// Converts a batch of encoded positions into TPG combo format values (floats)
    /// ready to then be sent into neural network.
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="flatValues"></param>
    static void ConvertToFlatTPG(IEncodedPositionBatchFlat batch, float[] flatValues)
    {
      Parallel.For(0, batch.NumPos, i =>
      {
        int offset = i * TPGRecordCombo.BYTES_PER_MOVE_AND_SQUARE_RECORD;

        Position pos = batch.Positions[i].ToPosition;

        TPGRecordCombo tpgRecordCombo = default;
        TPGRecordConverter.ConvertToTPGCombo(in pos, false, default, ref tpgRecordCombo);

        float[] rawData = tpgRecordCombo.RawBoardInputs;
        for (int j = 0; j < rawData.Length; j++)
        {
          flatValues[offset + j] = rawData[j];
        }
      });

    }
#endif

  }
}


#if NOT
          //          if (bestMoveNNEncoded.Move.IndexNeuralNet == 97 || bestMoveNNEncoded.Move.IndexNeuralNet == 103)
          if (pos.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
          {
            Console.WriteLine("................................ " + bestMoveTraining + " " + bestMoveNN);
          }

          //          Console.WriteLine("\r\n" + pos.FEN + "  " + bestMoveTraining + " " + bestMoveNN.Move);
          //          Console.WriteLine(evalResult.Policy);

          int promotionCount = 0;
          int epCount = 0;
          for (int ix = 0; ix < 64; ix++)
          {
            if (rawPosBuffer[i].SquaresAndMoves[ix].MoveRecord.PromotionBytes[0] > 0
             || rawPosBuffer[i].SquaresAndMoves[ix].MoveRecord.PromotionBytes[1] > 0)
              promotionCount++;
            if (rawPosBuffer[i].SquaresAndMoves[ix].SquareRecord.IsEnPassant > 0)
            {
              epCount++;
            }
          }

          if (promotionCount > 0)
          {
            Console.WriteLine(i + " found promotion " + pos.FEN);
          }
          if (epCount > 0)
          {
            Console.WriteLine(i + " found en passant " + pos.FEN);
          }
}
#endif




