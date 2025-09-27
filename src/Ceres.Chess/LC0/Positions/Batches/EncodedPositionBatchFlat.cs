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
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.LC0.Batches
{
  /// <summary>
  /// Encoded neural network input batch in "flat" (expanded)
  /// in the Leela Chess Zero format.
  /// </summary>
  [Serializable]
  public class EncodedPositionBatchFlat : IEncodedPositionBatchFlat
  {
    /// <summary>
    /// Retains some additional data structures related to positions,
    /// consumes a lot of memory but may be needed for some formats (TPG?).
    /// </summary>
    public static bool RETAIN_POSITION_INTERNALS = false;

    // These constants probably belong elsewhere
    public const int NUM_PIECE_PLANES_PER_POS = 13;
    public const int NUM_MISC_PLANES_PER_POS = 8; // constants holding miscellaneous values
    public const int NUM_HISTORY_POSITIONS = 8;
    public const int NUM_CACHE_HISTORY_POSITIONS = 7; // Number used for memory cache hashing

    public const int TOTAL_NUM_PLANES_ALL_HISTORIES = (NUM_HISTORY_POSITIONS * NUM_PIECE_PLANES_PER_POS)
                                                     + NUM_MISC_PLANES_PER_POS; // 112
    public const int TOTAL_NUM_PLANE_BYTES_ALL_HISTORIES = TOTAL_NUM_PLANES_ALL_HISTORIES * 64; // 60 * 64 = 3840
    public int MaxBatchSize { get; }


    // Bitmaps for the multiple board planes
    public ulong[] PosPlaneBitmaps;

    /// <summary>
    /// One byte for each bitmap with corresponding value.
    /// </summary>
    public byte[] PosPlaneValues;

    /// <summary>
    /// Optionally the set of state information assoicated with these positions.
    /// </summary>
    public Half[][] States;

    /// <summary>
    /// Optionally the associated MGPositions
    /// </summary>
    public MGPosition[] Positions;

    /// <summary>
    /// Optionally the associated hashes of the positions
    /// </summary>
    public ulong[] PositionHashes;

    /// <summary>
    /// Optionally the arrays of "plies since last move on square."
    /// </summary>
    public byte[] LastMovePlies;

    /// <summary>
    /// Optionally the set of moves from this position
    /// </summary>
    public MGMoveList[] Moves;

    /// <summary>
    /// Array of win probabilities. 
    /// </summary>
    public float[] W;

    /// <summary>
    /// Array of loss probabilities.
    /// </summary>
    public float[] L;

    /// <summary>
    /// Array of policies.
    /// </summary>
    public FP16[] Policy; // These are probabilities, e.g. 0.1 for 10% probability of moving being made

    /// <summary>
    /// Number of positions in the batch.
    /// </summary>
    public int NumPos;

    /// <summary>
    /// Type of training (if position only or includes training data).
    /// </summary>
    public EncodedPositionType TrainingType;

    /// <summary>
    /// Optionally (if multiple evaluators are configured) 
    /// the index of which executor should be used for this batch
    /// </summary>
    public short PreferredEvaluatorIndex;

    /// <summary>
    /// If originated from EncodedPositionWithHistory then
    /// this field optionally holds the origin data array.
    /// </summary>
    public EncodedPositionWithHistory[] PositionsBuffer;

    public bool PositionsUseSecondaryEvaluator { get; set; }

    public IEncodedPositionBatchFlat GetSubBatchSlice(int startIndex, int count)
    {
      return new EncodedPositionBatchFlatSlice(this, startIndex, count);
    }


    /// <summary>
    /// Returns a new EncodedPositionBatchFlat which contains a specified 
    /// subset of positions in this batch (by copying the underlying values).
    /// </summary>
    /// <param name="startIndex"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    public IEncodedPositionBatchFlat GetSubBatchCopied(int startIndex, int count)
    {
      float[] w = null;
      float[] l = null;
      if (W != null)
      {
        w = new float[count];
        Array.Copy(W, startIndex, w, 0, count);
      }
      if (L != null)
      {
        l = new float[count];
        Array.Copy(L, startIndex, l, 0, count);
      }

      byte[] posPlaneValuesEncoded = new byte[count * EncodedPositionWithHistory.NUM_PLANES_TOTAL];
      Array.Copy(PosPlaneValues, startIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, posPlaneValuesEncoded, 0, count * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

      ulong[] posPlaneBitmaps = new ulong[count * EncodedPositionWithHistory.NUM_PLANES_TOTAL];
      Array.Copy(PosPlaneBitmaps, startIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, posPlaneBitmaps, 0, count * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

      EncodedPositionBatchFlat ret = new EncodedPositionBatchFlat(posPlaneBitmaps, posPlaneValuesEncoded, w, l, null, count);

      if (States != null)
      {
        // For safety, make a deep copy.
        Half[][] states = new Half[count][];
        for (int i = 0; i < count; i++)
        {
          states[i] = new Half[States[i].Length];
          Array.Copy(States[i], states[i], States[i].Length);
        }
        ret.States = states;
      }

      if (Positions != null)
      {
        ulong[] hashes = new ulong[count];
        MGPosition[] positionsMG = new MGPosition[count];
        MGMoveList[] moves = new MGMoveList[count];

        Array.Copy(PositionHashes, startIndex, hashes, 0, count);
        Array.Copy(Positions, startIndex, positionsMG, 0, count);
        Array.Copy(Moves, startIndex, moves, 0, count);

        ret.PositionHashes = hashes;
        ret.Positions = positionsMG;
        ret.Moves = moves;
      }

      if (LastMovePlies != null)
      {
        byte[] lastPlies = new byte[count * 64];
        Array.Copy(LastMovePlies, startIndex * 64, lastPlies, 0, count * 64);
        ret.LastMovePlies = lastPlies;
      }

      return ret;
    }


    /// <summary>
    /// Sets the Position field with positions converted from a Span of EncodedTrainingPosition.
    /// </summary>
    /// <param name="positions"></param>
    public void SetPositions(ReadOnlySpan<EncodedTrainingPosition> positions)
    {
      NumPos = positions.Length;

      if (positions.Length != NumPos)
      {
        throw new ArgumentException("Number of positions does not match.");
      }

      Positions = new MGPosition[NumPos];
      for (int i = 0; i < Positions.Length; i++)
      {
        Positions[i] = positions[i].FinalPosition.ToMGPosition;
      }
    }


    /// <summary>
    /// Sets the Position field with positions converted from a Span of EncodedPositionWithHistory.
    /// </summary>
    /// <param name="positions"></param>
    private void SetPositions(ReadOnlySpan<EncodedPositionWithHistory> positions)
    {
      if (positions.Length != NumPos)
      {
        throw new ArgumentException("Number of positions does not match.");
      }

      Positions = new MGPosition[NumPos];
      for (int i = 0; i < Positions.Length; i++)
      {
        Positions[i] = positions[i].FinalPosition.ToMGPosition;
      }
    }


    /// <summary>
    /// Copies a set of training positions into the batch.
    /// </summary>
    /// <param name="positions"></param>
    void SetTrainingData(ReadOnlySpan<EncodedTrainingPosition> positions)
    {
      // Initialize and copy over game results, policy vectors
      int nextPolicyIndex = 0;
      for (int i = 0; i < NumPos; i++)
      {
        W = new float[NumPos];
        L = new float[NumPos];

        throw new NotImplementedException("result needs remediation, probably take from ResultD");
        sbyte result = 0;// (sbyte)positions[i].Position.MiscInfo.InfoTraining.ResultFromOurPerspective;

        switch (result)
        {
          case (sbyte)EncodedPositionMiscInfo.ResultCode.Win:
            W[i] = 1.0f;
          case (sbyte)EncodedPositionMiscInfo.ResultCode.Loss:
            L[i] = 1.0f;
          default:
            throw new Exception("Internal error: Unknown result code");
        }


        ref readonly EncodedTrainingPosition posRef = ref positions[i];
        unsafe
        {
          for (int j = 0; j < EncodedPolicyVector.POLICY_VECTOR_LENGTH; j++)
          {
            float val = posRef.Policies.ProbabilitiesPtr[j];
            // NOTE: We convert NaN to 0.0 (in later V4 data this indicates illegal move)
            if (val != 0 && !float.IsNaN(val)) Policy[nextPolicyIndex] = (FP16)val;
            nextPolicyIndex++;
          }
        }
      }
    }


    // TODO: The following two versions of Set basically duplciates, but operating on slightly different data structures.
    //       There is no highly efficient way to centralize this. 
    // Since this is a performance-critical method, we leave both of these in place.

    public void Set(ReadOnlySpan<EncodedPositionWithHistory> positions, int numToProcess, bool setPositions, bool fillInHistoryPlanes = false)
    {
      NumPos = numToProcess;
      TrainingType = EncodedPositionType.PositionOnly;
      if (RETAIN_POSITION_INTERNALS)
      {
        PositionsBuffer = positions.ToArray();
      }

      int nextOutPlaneIndex = 0;

      if (setPositions)
      {
        SetPositions(positions);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      void WritePairWithValue1(ulong bitmap)
      {
        PosPlaneBitmaps[nextOutPlaneIndex] = bitmap;
        PosPlaneValues[nextOutPlaneIndex] = 1;
        nextOutPlaneIndex++;
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      void WritePair(ulong bitmap, byte value)
      {
        PosPlaneBitmaps[nextOutPlaneIndex] = bitmap;
        PosPlaneValues[nextOutPlaneIndex] = value;
        nextOutPlaneIndex++;
      }

      const int PLANES_WRITTEN = EncodedPositionBoard.NUM_PLANES_PER_BOARD * EncodedPositionBoards.NUM_MOVES_HISTORY;

      //      Unsafe.InitBlock(ref PosPlaneValues[0], 1, (uint)(numToProcess * PLANES_WRITTEN));

      // Initialize planes
      for (int i = 0; i < numToProcess; i++)
      {
        // Set planes (NOTE: we move all 8 history planes)
        if (fillInHistoryPlanes)
        {
          EncodedPositionWithHistory positionCopy = positions[i];
          positionCopy.FillInEmptyPlanes();
          positionCopy.ExtractPlanesValuesIntoArray(EncodedPositionBoards.NUM_MOVES_HISTORY, PosPlaneBitmaps, nextOutPlaneIndex);
        }
        else
        {
          positions[i].ExtractPlanesValuesIntoArray(EncodedPositionBoards.NUM_MOVES_HISTORY, PosPlaneBitmaps, nextOutPlaneIndex);
        }

        // Start by setting all plane values to 1.
        Unsafe.InitBlock(ref PosPlaneValues[nextOutPlaneIndex], 1, (uint)PLANES_WRITTEN);
        //        Array.Fill<byte>(PosPlaneValues, 1, nextOutPlaneIndex, PLANES_WRITTEN);

        // Advance
        nextOutPlaneIndex += PLANES_WRITTEN;

        // Copy in special plane values
        EncodedPositionMiscInfo miscInfo = positions[i].MiscInfo.InfoPosition;
        WritePairWithValue1(miscInfo.Castling_US_OOO > 0 ? ulong.MaxValue : 0);
        WritePairWithValue1(miscInfo.Castling_US_OO > 0 ? ulong.MaxValue : 0);
        WritePairWithValue1(miscInfo.Castling_Them_OOO > 0 ? ulong.MaxValue : 0);
        WritePairWithValue1(miscInfo.Castling_Them_OO > 0 ? ulong.MaxValue : 0);

        WritePairWithValue1((ulong)miscInfo.SideToMove > 0 ? ulong.MaxValue : 0);
        WritePair(ulong.MaxValue, miscInfo.Rule50Count);
        WritePairWithValue1(0);// "used to be movecount plane, now it's all zeros" // index 110
        WritePairWithValue1(ulong.MaxValue); // "all ones to help NN find board edges" // index 111
      }
    }

    public void Set(ReadOnlySpan<EncodedTrainingPosition> positions, int numToProcess, EncodedPositionType trainingType, bool setPositions)
    {
      NumPos = numToProcess;
      TrainingType = trainingType;

      if (setPositions)
      {
        // Note that the SetPositions method will take care of the necessary unmirroring itself.
        SetPositions(positions.Slice(0, numToProcess));
      }

      EncodedTrainingPosition[] positionsCopy = positions.Slice(0, numToProcess).ToArray();
      for (int i = 0; i < NumPos; i++)
      {
        positionsCopy[i] = positions[i];
      }
      positions = default; // Make sure the original is not subsequently used.

      if (trainingType == EncodedPositionType.PositionAndTrainingData)
      {
        SetTrainingData(positionsCopy);
      }

      int nextOutPlaneIndex = 0;

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      void WritePairWithValue1(ulong bitmap)
      {
        PosPlaneBitmaps[nextOutPlaneIndex] = bitmap;
        PosPlaneValues[nextOutPlaneIndex] = 1;
        nextOutPlaneIndex++;
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      void WritePair(ulong bitmap, byte value)
      {
        PosPlaneBitmaps[nextOutPlaneIndex] = bitmap;
        PosPlaneValues[nextOutPlaneIndex] = value;
        nextOutPlaneIndex++;
      }

      // Initialize planes
      for (int i = 0; i < numToProcess; i++)
      {
        // Set planes (NOTE: we move all 8 history planes)
        positionsCopy[i].PositionWithBoards.ExtractPlanesValuesIntoArray(EncodedPositionBoards.NUM_MOVES_HISTORY, PosPlaneBitmaps, nextOutPlaneIndex);
        const int PLANES_WRITTEN = EncodedPositionBoard.NUM_PLANES_PER_BOARD * EncodedPositionBoards.NUM_MOVES_HISTORY;

        // Set all values to 1.0f
        Array.Fill<byte>(PosPlaneValues, 1, nextOutPlaneIndex, PLANES_WRITTEN);

        // Advance
        nextOutPlaneIndex += PLANES_WRITTEN;

        // Copy in special plane values
        EncodedPositionMiscInfo miscInfo = positionsCopy[i].PositionWithBoards.MiscInfo.InfoPosition;
        WritePairWithValue1(miscInfo.Castling_US_OOO > 0 ? ulong.MaxValue : 0);
        WritePairWithValue1(miscInfo.Castling_US_OO > 0 ? ulong.MaxValue : 0);
        WritePairWithValue1(miscInfo.Castling_Them_OOO > 0 ? ulong.MaxValue : 0);
        WritePairWithValue1(miscInfo.Castling_Them_OO > 0 ? ulong.MaxValue : 0);

        WritePairWithValue1((ulong)miscInfo.SideToMove > 0 ? ulong.MaxValue : 0);
        WritePair(ulong.MaxValue, miscInfo.Rule50Count);
        WritePairWithValue1(0);// "used to be movecount plane, now it's all zeros" // index 110
        WritePairWithValue1(ulong.MaxValue); // "all ones to help NN find board edges" // index 111
      }
    }


    /// <summary>
    /// Destructor.
    /// </summary>
    ~EncodedPositionBatchFlat()
    {
      Shutdown();
    }

    /// <summary>
    /// Release resources associated with the batch.
    /// </summary>
    public void Shutdown()
    {
      PosPlaneBitmaps = null;
      PosPlaneValues = null;
    }

    void Init(EncodedPositionType trainingType)
    {
      PosPlaneBitmaps = GC.AllocateUninitializedArray<ulong>(MaxBatchSize * EncodedPositionWithHistory.NUM_PLANES_TOTAL);
      PosPlaneValues = GC.AllocateUninitializedArray<byte>(MaxBatchSize * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

#if NOT
      // Actually these allocations below are unnecessary, we only (rarely) use W/L/Policy 
      // if the SetTrainingData method is called.
      if (trainingType == EncodedPositionType.PositionAndTrainingData)
      {
        W = GC.AllocateUninitializedArray<float>(MaxBatchSize);
        L = GC.AllocateUninitializedArray<float>(MaxBatchSize);
        Policy = GC.AllocateUninitializedArray<FP16>(MaxBatchSize * EncodedPolicyVector.POLICY_VECTOR_LENGTH);
      }
#endif
    }

    const int BATCH_SIZE_ALIGNMENT = 4;


    public EncodedPositionBatchFlat(EncodedPositionType trainingType, int maxBatchSize)
    {
      MaxBatchSize = (int)MathUtils.RoundedUp(maxBatchSize, BATCH_SIZE_ALIGNMENT);
      Init(trainingType);
    }

    public EncodedPositionBatchFlat(ReadOnlySpan<EncodedPositionWithHistory> positions, int numToProcess,
                                    bool setPositions, bool fillInHistoryPlanes = false)
    {
      MaxBatchSize = (int)MathUtils.RoundedUp(numToProcess, BATCH_SIZE_ALIGNMENT);
      Init(EncodedPositionType.PositionOnly);
      Set(positions, numToProcess, setPositions, fillInHistoryPlanes);
    }

    public EncodedPositionBatchFlat(ReadOnlySpan<EncodedTrainingPosition> positions, int numToProcess, EncodedPositionType trainingType, bool setPositions)
    {
      MaxBatchSize = (int)MathUtils.RoundedUp(numToProcess, BATCH_SIZE_ALIGNMENT);
      Init(trainingType);

      Set(positions, numToProcess, trainingType, setPositions);
    }

    public EncodedPositionBatchFlat(ulong[] posPlaneBitmaps, byte[] posPlaneValuesEncoded, float[] w, float[] l, float[] policy, int numPos)
    {
      NumPos = numPos;
      W = w;
      L = l;
      PosPlaneBitmaps = posPlaneBitmaps;
      PosPlaneValues = posPlaneValuesEncoded;
      MaxBatchSize = numPos;

      if (policy != null)
      {
        Policy = FP16.ToFP16(policy);
      }
    }


    /// <summary>
    /// Zero out the history planes for all positions in the batch.
    /// </summary>
    public void ZeroHistoryPlanes()
    {
      Span<ulong> bitmaps = PosPlaneBitmaps;
      Span<byte> values = PosPlaneValues;

      for (int i = 0; i < NumPos; i++)
      {
        for (int j = NUM_PIECE_PLANES_PER_POS;
                 j < TOTAL_NUM_PLANES_ALL_HISTORIES
                   - NUM_MISC_PLANES_PER_POS; j++)
        {
          int index = i * TOTAL_NUM_PLANES_ALL_HISTORIES + j;
          bitmaps[index] = 0;
          values[index] = 0;
        }
      }
    }



    public void DumpPlanes(int positionIndex)
    {
      int baseOffset = positionIndex * TOTAL_NUM_PLANES_ALL_HISTORIES;
      Console.WriteLine();
      for (int i = 0; i < 112; i++)
      {
        Console.WriteLine($"({i},{PosPlaneBitmaps[baseOffset + i]}, {PosPlaneValues[baseOffset + i]}),");
      }
    }

    /// <summary>
    /// Static factor method to return a batch from a span of EncodedPositionWithHistory.
    /// </summary>
    /// <param name="fillInHistoryPlanes"></param>
    /// <param name="fens"></param>
    /// <returns></returns>
    public static EncodedPositionBatchFlat FromTrainingPositionRaw(ReadOnlySpan<EncodedPositionWithHistory> pos, bool setPositions)
    {
      return new EncodedPositionBatchFlat(pos, pos.Length, setPositions);
    }


    /// <summary>
    /// Static factor method to return a batch from an EncodedPositionWithHistory.
    /// </summary>
    /// <param name="fillInHistoryPlanes"></param>
    /// <param name="fens"></param>
    /// <returns></returns>
    public static EncodedPositionBatchFlat FromTrainingPositionRaw(EncodedPositionWithHistory pos, bool setPositions)
    {
      EncodedPositionWithHistory[] array = new EncodedPositionWithHistory[] { pos };
      return new EncodedPositionBatchFlat(array, 1, setPositions);
    }

    /// <summary>
    /// Static factor method to return a batch from a LZTrainingPositionRaw
    /// </summary>
    /// <param name="fillInHistoryPlanes"></param>
    /// <param name="fens"></param>
    /// <returns></returns>
    public static EncodedPositionBatchFlat FromTrainingPositionRaw(ReadOnlySpan<EncodedTrainingPosition> pos, bool setPositions)
    {
      return new EncodedPositionBatchFlat(pos, pos.Length, EncodedPositionType.PositionOnly, setPositions);
    }


    /// <summary>
    /// NOTE: not fully tested, and may not be needed anywhere
    /// </summary>
    /// <param name="rawData"></param>
    /// <param name="channels"></param>
    /// <returns></returns>
    static float[] ToNHWC(float[] rawData, int channels)
    {
      // Convert NCHW to NHWC (n, C,64) to (n, 64, C)
      float[] remapped = new float[rawData.Length];
      int numPositions = rawData.Length / (channels * 8 * 8);
      for (int i = 0; i < numPositions; i++)
      {
        int baseOffsetSource = i * 64 * channels;
        int baseOffsetTarget = i * 64 * channels;
        for (int c = 0; c < channels; c++)
        {
          for (int k = 0; k < 64; k++)
          {
            int sourceOffset = baseOffsetSource + (c * 64) + k;
            int destOffset = baseOffsetTarget + (k * channels) + c;
            remapped[destOffset] = rawData[sourceOffset];
          }
        }
      }
      return remapped;
    }


    public float[] ValuesFlatFromPlanesSubsetHistoryPositions(int numHistoryPositions, bool convertToNHWC = false, bool scale50MoveCounter = true)
    {
      throw new NotImplementedException("Please verify correctness of arguemnt scale50MoveCounter");

      int numChannels = 8 + 13 * numHistoryPositions;

      float[] ret;

      if (numHistoryPositions == NUM_HISTORY_POSITIONS)
      {
        throw new NotImplementedException();
        //ret = ValuesFlatFromPlanes(null, convertToNHWC, scale50MoveCounter);
      }
      else
      {
        ret = new float[numChannels * NumPos * 64];

        // TO DO: clean up, similar to code in ConvertToFlaat
        int nextRetPos = 0;
        for (int i = 0; i < 112 * NumPos; i++)
        {
          int modulo112 = i % 112;
          bool shouldCopy = false;

          // Is it in the postfix miscellaneous planes?
          if (modulo112 > 104)
          {
            shouldCopy = true;
          }

          // Is this plane encoding info from within the first numHistoryPositions positions
          if (modulo112 < numHistoryPositions * 13)
          {
            shouldCopy = true;
          }

          if (shouldCopy)
          {
            BitVector64 bits = new BitVector64((long)PosPlaneBitmaps[i]);
            float value = PosPlaneValues[i];

            for (int b = 0; b < 64; b++)
            {
              ret[nextRetPos++] = bits.BitIsSet(b) ? value : 0;
            }
          }
        }
      }

      return convertToNHWC ? ToNHWC(ret, numChannels) : ret;
    }


    public Memory<Half> ValuesFlatFromPlanes(Memory<Half> preallocatedBuffer, bool nhwc, bool scale50MoveCounter)
    {
      Debug.Assert(!nhwc); // not implemented

      Memory<Half> ret;

      int length = TOTAL_NUM_PLANES_ALL_HISTORIES * NumPos * 64;
      if (!preallocatedBuffer.IsEmpty)
      {
        Debug.Assert(preallocatedBuffer.Length >= length);
        ret = preallocatedBuffer;
        ret.Slice(0, length).Span.Clear();
      }
      else
      {
        ret = new Half[length];
      }

      ConvertToFlat(ret, scale50MoveCounter);
      return ret;
    }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void BitmapRepresentationExpand(ulong[] thisLongs,
                                                         byte[] thisValues,
                                                         Memory<Half> targetArrayMemory,
                                                         int startIndex,
                                                         int numToConvert,
                                                         int totalElements,
                                                         bool scale50MoveCounter)
    {
      Span<Half> targetSpan = targetArrayMemory.Span;
      ref Half dst = ref MemoryMarshal.GetReference(targetSpan);
      ref ulong longsRef = ref MemoryMarshal.GetArrayDataReference(thisLongs);
      ref byte valsRef = ref MemoryMarshal.GetArrayDataReference(thisValues);

      const int SQUARES_PER_PLANE = 64;
      const int PLANES_PER_BLOCK = 112;
      const int MOVES50_PLANE_MOD = 109;
      const float INV_99 = 1.0f / 99.0f;

      int endIndex = startIndex + numToConvert;
      if (endIndex > totalElements) { endIndex = totalElements; }

      int targetOffset = startIndex * SQUARES_PER_PLANE;
      int rem112 = startIndex % PLANES_PER_BLOCK;

      for (int outer = startIndex; outer < endIndex; outer++)
      {
        ulong bits = Unsafe.Add(ref longsRef, outer);
        if (bits != 0UL)
        {
          float val = (float)Unsafe.Add(ref valsRef, outer);
          if (scale50MoveCounter && rem112 == MOVES50_PLANE_MOD)
          {
            val *= INV_99;
          }
          Half hval = (Half)val;

          // Base of the 64-wide stripe
          ref Half dstStripe = ref Unsafe.Add(ref dst, targetOffset);

          while (bits != 0UL)
          {
            int tz = BitOperations.TrailingZeroCount(bits);
            Unsafe.Add(ref dstStripe, tz) = hval;
            bits &= bits - 1; // clear lowest set bit
          }
        }

        targetOffset += SQUARES_PER_PLANE;
        rem112++;
        if (rem112 == PLANES_PER_BLOCK) { rem112 = 0; }
      }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="outBuffer">buffer to receive values. NOTE! This is assumed to start out cleared (all zeros)</param>
    /// <param name="encodingType"></param>
    void ConvertToFlat(Memory<Half> outBuffer, bool scale50MoveCounter)
    {
      // TODO: Somehow rework this performance-critical method
      //       by vectorization or putting expansion on GPU.
      //return ConvertToFlatSlow(outBuffer, encodingType); old slow version
      int numToConvert = NumPos * TOTAL_NUM_PLANES_ALL_HISTORIES;
      if (numToConvert < 64)
      {
        BitmapRepresentationExpand(PosPlaneBitmaps, PosPlaneValues, outBuffer,
                                   0, numToConvert, numToConvert, scale50MoveCounter);
      }
      else
      {
        // Do a Parallel.For with each thread converting a block of 48 positions
        const int NUM_PER_BLOCK = 48;
        int numBlocks = numToConvert / NUM_PER_BLOCK;
        if (numToConvert % NUM_PER_BLOCK != 0)
        {
          numBlocks++;
        }

        Parallel.For(0, numBlocks, i =>
        {
          BitmapRepresentationExpand(PosPlaneBitmaps, PosPlaneValues, outBuffer,
                                     i * NUM_PER_BLOCK, NUM_PER_BLOCK, numToConvert, scale50MoveCounter);
        });

      }
    }


    #region Dump diagostics

    public void DumpDecoded()
    {
      DumpDecoded(ValuesFlatFromPlanes(default, false, false).ToArray(), TOTAL_NUM_PLANES_ALL_HISTORIES);
    }

    public static void DumpDecoded(Memory<Half> encodedPos, int numPlanes)
    {
      ulong[] decoded = DecodedBits(encodedPos);
      DumpDecoded(decoded, numPlanes);
    }

    public static void DumpDecoded(ulong[] decodedPos, int numPlanes)
    {
      Console.WriteLine("\r\nRaw value of a planes of first position (can compare agaist values seen in debugger within encoder.cc");
      for (int i = 0; i < numPlanes; i++)
      {
        Console.WriteLine($"Plane {i,-4:##0} {decodedPos[i],-10:########0}");
      }

    }


    public static ulong[] DecodedBits(Memory<Half> rawMem)
    {
      Span<Half> raw = rawMem.Span;

      ulong[] ret = new ulong[raw.Length / 64];
      for (int i = 0; i < ret.Length; i++)
      {
        BitVector64 v = new BitVector64();
        for (int j = 0; j < 64; j++)
        {
          float val = (float)raw[i * 64 + j];
          if (val == 1)
          {
            v.SetBit(j);
          }
          else if (val == 0)
          {
            // nothing to do
          }
          else
          {
            // Tricky. A few planes are actually numbers, such as rule 50. Encode here directly as that number.
            if (j == 0) v.SetData((long)val);
          }
        }
        ret[i] = (ulong)v.Data;
      }
      return ret;
    }


    public static ulong[] DecodedBits(byte[] raw)
    {
      ulong[] ret = new ulong[raw.Length / 64];
      for (int i = 0; i < ret.Length; i++)
      {
        BitVector64 v = new BitVector64();
        for (int j = 0; j < 64; j++)
        {
          byte val = raw[i * 64 + j];
          if (val == 1)
          {
            v.SetBit(j);
          }
          else if (val == 0)
          {
            // Tricky. A few planes are actually numbers, such as rule 50. Encode here directly as that number.
            if (j == 0) v.SetData((long)val);
          }
          else
            throw new NotImplementedException(); // not expected
        }
        ret[i] = (ulong)v.Data;
      }
      return ret;
    }

    #endregion

    #region Utils

    static bool Equals<T>(T[] v1, T[] v2) where T : struct
    {
      if (v1.Length != v2.Length) return false;
      for (int i = 0; i < v1.Length; i++)
        if (!Equals(v1[i], v2[i]))
        {
          //Console.WriteLine("bad " + i + " " + (i%112) + " " + v1[i] + " " + v2[i]);
          return false;
        }
      return true;
    }

    public override bool Equals(object obj)
    {
      if (obj == null || GetType() != obj.GetType())
        return false;

      EncodedPositionBatchFlat other = (EncodedPositionBatchFlat)obj;

      if (!Equals(PosPlaneBitmaps, other.PosPlaneBitmaps)) return false;
      if (!Equals(PosPlaneValues, other.PosPlaneValues)) return false;
      if (!Equals(W, other.W)) return false;
      if (!Equals(L, other.L)) return false;
      if (!Equals(NumPos, other.NumPos)) return false;

      // Approx equality on policy
      for (int i = 0; i < EncodedPolicyVector.POLICY_VECTOR_LENGTH; i++)
        if (Policy[i] - other.Policy[i] > 0.001)
          return false;


      return true;
    }

    #endregion

    #region Interface

    Memory<ulong> IEncodedPositionBatchFlat.PosPlaneBitmaps => PosPlaneBitmaps.AsMemory();

    Memory<byte> IEncodedPositionBatchFlat.PosPlaneValues => PosPlaneValues.AsMemory();

    int IEncodedPositionBatchFlat.NumPos => NumPos;

    EncodedPositionType IEncodedPositionBatchFlat.TrainingType => TrainingType;

    short IEncodedPositionBatchFlat.PreferredEvaluatorIndex => PreferredEvaluatorIndex;

    Memory<MGPosition> IEncodedPositionBatchFlat.Positions { get => Positions.AsMemory(); set => Positions = value.ToArray(); }
    Memory<ulong> IEncodedPositionBatchFlat.PositionHashes { get => PositionHashes.AsMemory(); set => PositionHashes = value.ToArray(); }
    Memory<byte> IEncodedPositionBatchFlat.LastMovePlies { get => LastMovePlies.AsMemory(); set => LastMovePlies = value.ToArray(); }
    Memory<MGMoveList> IEncodedPositionBatchFlat.Moves { get => Moves.AsMemory(); set => Moves = value.ToArray(); }

    Memory<Half[]> IEncodedPositionBatchFlat.States { get => States.AsMemory(); set => States = value.ToArray(); }

    Memory<Half> IEncodedPositionBatchFlat.ValuesFlatFromPlanes(Memory<Half> preallocatedBuffer, bool nwhc, bool scaleMove50Counter) => ValuesFlatFromPlanes(preallocatedBuffer, nwhc, scaleMove50Counter);

    Memory<EncodedPositionWithHistory> IEncodedPositionBatchFlat.PositionsBuffer { get => PositionsBuffer; }

    #endregion

    #region Overrides

    /// <summary>
    /// Returns strimg summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<EncodedPositionBatchFlat of size {NumPos} of type {TrainingType}>";
    }

    #endregion
  }


}

