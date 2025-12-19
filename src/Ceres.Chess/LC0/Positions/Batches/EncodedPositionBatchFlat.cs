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
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
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
        PositionsBuffer = positions.Slice(0, numToProcess).ToArray();
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


    public void ConvertValuesToFlatFromPlanes(Memory<Half> destinationBuffer,
                                            bool nhwc, bool scale50MoveCounter)
    {
      Debug.Assert(!nhwc);
      ConvertToFlat(0, NumPos, destinationBuffer, scale50MoveCounter);
    }



    /// <summary>
    /// Non-optimized version of BitmapRepresentationExpand for reference.  
    /// </summary>
    /// <param name="thisLongs"></param>
    /// <param name="thisValues"></param>
    /// <param name="targetArray"></param>
    /// <param name="numToConvert"></param>
    /// <param name="scale50MoveCounter"></param>
    unsafe static void BitmapRepresentationExpandSLOW(ulong[] thisLongs, byte[] thisValues,
                                                      Memory<Half> targetArrayM, int numToConvert,
                                                      bool scale50MoveCounter)
    {
      int targetOffset = 0;
      Span<Half> targetArray = targetArrayM.Span;
      fixed (ulong* longs = &thisLongs[0])
      {
        for (int planeIndex = 0; planeIndex < numToConvert; planeIndex++)
        {
          // Apply the necessary scaling of this is the move counter (50 move rule)
          bool isMoves50Plane = planeIndex % 112 == 109;
          float multiplier = (scale50MoveCounter && isMoves50Plane) ? (1.0f / 99.0f) : 1.0f;
          float targetValue = multiplier * thisValues[planeIndex];

          byte* bytes = (byte*)&longs[planeIndex];

          for (int i = 0; i < 8; i++)
          {
            targetArray[targetOffset] = default;
            targetArray[targetOffset + 1] = default;
            targetArray[targetOffset + 2] = default;
            targetArray[targetOffset + 3] = default;
            targetArray[targetOffset + 4] = default;
            targetArray[targetOffset + 5] = default;
            targetArray[targetOffset + 6] = default;
            targetArray[targetOffset + 7] = default;

            byte val = bytes[i];
            if (val != 0)
            {
              if ((val & (1 << 0)) > 0) targetArray[targetOffset] = (Half)targetValue;
              if ((val & (1 << 1)) > 0) targetArray[targetOffset + 1] = (Half)targetValue;
              if ((val & (1 << 2)) > 0) targetArray[targetOffset + 2] = (Half)targetValue;
              if ((val & (1 << 3)) > 0) targetArray[targetOffset + 3] = (Half)targetValue;
              if ((val & (1 << 4)) > 0) targetArray[targetOffset + 4] = (Half)targetValue;
              if ((val & (1 << 5)) > 0) targetArray[targetOffset + 5] = (Half)targetValue;
              if ((val & (1 << 6)) > 0) targetArray[targetOffset + 6] = (Half)targetValue;
              if ((val & (1 << 7)) > 0) targetArray[targetOffset + 7] = (Half)targetValue;
            }
            targetOffset += 8;
          }

        }
      }
    }


    /// <summary>
    /// Lookup table: for each byte value (0-255), stores 8 Half values 
    /// representing the expanded bits (1.0 for set bit, 0.0 for unset).
    /// </summary>
    private static readonly Half[] ByteToHalfLUT = InitializeByteLUT();

    private static Half[] InitializeByteLUT()
    {
      // 256 possible byte values, each expands to 8 Half values
      Half[] lut = new Half[256 * 8];
      Half one = (Half)1.0f;
      Half zero = (Half)0.0f;

      for (int b = 0; b < 256; b++)
      {
        int baseIdx = b * 8;
        for (int bit = 0; bit < 8; bit++)
        {
          lut[baseIdx + bit] = ((b >> bit) & 1) != 0 ? one : zero;
        }
      }
      return lut;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void BitmapRepresentationExpand(ulong[] thisLongs,
                                                         byte[] thisValues,
                                                         Memory<Half> targetArrayMemory,
                                                         int startPlaneIndex,
                                                         int numPlanesToConvert,
                                                         int totalElements,
                                                         bool scale50MoveCounter)
    {
      const int SQUARES_PER_PLANE = 64;

      int endIndex = startPlaneIndex + numPlanesToConvert;
      if (endIndex > totalElements) throw new NotImplementedException();

      // Verify buffer sizes to prevent out of range accesses
      Debug.Assert(startPlaneIndex <= totalElements);
      Debug.Assert(startPlaneIndex + numPlanesToConvert <= thisLongs.Length);
      Debug.Assert(targetArrayMemory.Length >= numPlanesToConvert * SQUARES_PER_PLANE);

      // Choose vectorized path if AVX2 is available (x64)
      if (Avx2.IsSupported)
      {
        BitmapRepresentationExpandAVX2(thisLongs, thisValues, targetArrayMemory,
                                       startPlaneIndex, numPlanesToConvert, totalElements, scale50MoveCounter);
      }
      else
      {
        BitmapRepresentationExpandScalar(thisLongs, thisValues, targetArrayMemory,
                                         startPlaneIndex, numPlanesToConvert, totalElements, scale50MoveCounter);
      }
    }


    /// <summary>
    /// AVX2-optimized version of BitmapRepresentationExpand.
    /// Uses SIMD to expand 8 bits to 8 Half values at a time.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void BitmapRepresentationExpandAVX2(ulong[] thisLongs,
                                                               byte[] thisValues,
                                                               Memory<Half> targetArrayMemory,
                                                               int startPlaneIndex,
                                                               int numPlanesToConvert,
                                                               int totalElements,
                                                               bool scale50MoveCounter)
    {
      const int SQUARES_PER_PLANE = 64;
      const int PLANES_PER_BLOCK = 112;
      const int MOVES50_PLANE_MOD = 109;
      const float INV_99 = 1.0f / 99.0f;

      Span<Half> targetSpan = targetArrayMemory.Span;
      int endIndex = startPlaneIndex + numPlanesToConvert;
      int targetOffset = 0;
      int rem112 = startPlaneIndex % PLANES_PER_BLOCK;

      fixed (ulong* longsPtr = thisLongs)
      fixed (byte* valsPtr = thisValues)
      fixed (Half* dstPtr = targetSpan)
      fixed (Half* lutPtr = ByteToHalfLUT)
      {
        // Precompute bit masks for expansion: [1, 2, 4, 8, 16, 32, 64, 128]
        Vector128<byte> bitMasks = Vector128.Create((byte)1, 2, 4, 8, 16, 32, 64, 128,
                                                     1, 2, 4, 8, 16, 32, 64, 128);

        for (int outer = startPlaneIndex; outer < endIndex; outer++)
        {
          Half* dst = dstPtr + targetOffset;
          ulong bits = longsPtr[outer];

          if (bits == 0UL)
          {
            // Fast path: zero the entire plane using vectorized stores
            Vector256<short> zero256 = Vector256<short>.Zero;
            Avx.Store((short*)dst, zero256);
            Avx.Store((short*)(dst + 16), zero256);
            Avx.Store((short*)(dst + 32), zero256);
            Avx.Store((short*)(dst + 48), zero256);
          }
          else
          {
            float val = (float)valsPtr[outer];
            if (scale50MoveCounter && rem112 == MOVES50_PLANE_MOD)
            {
              val *= INV_99;
            }
            Half hval = (Half)val;
            ushort hvalBits = Unsafe.As<Half, ushort>(ref hval);

            if (bits == ulong.MaxValue)
            {
              // Fast path: all bits set - fill with hval using vectorized stores
              Vector256<ushort> valVec = Vector256.Create(hvalBits);
              Avx.Store((ushort*)dst, valVec);
              Avx.Store((ushort*)(dst + 16), valVec);
              Avx.Store((ushort*)(dst + 32), valVec);
              Avx.Store((ushort*)(dst + 48), valVec);
            }
            else
            {
              // General case: expand each byte of the ulong
              // Process using LUT + scalar multiplication for value scaling
              byte* bitsBytes = (byte*)&bits;

              for (int byteIdx = 0; byteIdx < 8; byteIdx++)
              {
                byte b = bitsBytes[byteIdx];
                Half* dstByte = dst + byteIdx * 8;

                if (b == 0)
                {
                  // Zero 8 Half values (16 bytes) - use 128-bit store
                  Sse2.Store((short*)dstByte, Vector128<short>.Zero);
                }
                else if (b == 0xFF)
                {
                  // All 8 bits set - fill with hval
                  Vector128<ushort> valVec128 = Vector128.Create(hvalBits);
                  Sse2.Store((ushort*)dstByte, valVec128);
                }
                else
                {
                  // Use LUT for bit pattern, then multiply by value
                  Half* lutEntry = lutPtr + b * 8;

                  // Load 8 Half values from LUT (0.0 or 1.0)
                  Vector128<short> lutVec = Sse2.LoadVector128((short*)lutEntry);

                  // Convert to float, multiply by hval, convert back
                  // Since LUT contains 0 or 1, we can use integer masking instead:
                  // Create mask where set bits become 0xFFFF
                  Vector128<byte> byteVec = Vector128.Create(b, b, b, b, b, b, b, b,
                                                              b, b, b, b, b, b, b, b);
                  Vector128<byte> expanded = Sse2.And(byteVec, bitMasks);
                  Vector128<byte> mask8 = Sse2.CompareEqual(expanded, bitMasks);

                  // Expand 8-bit mask to 16-bit mask for Half values
                  Vector128<short> maskLo = Sse2.UnpackLow(mask8, mask8).AsInt16();

                  // Apply mask to select hval or zero
                  Vector128<ushort> valVec128 = Vector128.Create(hvalBits);
                  Vector128<short> result = Sse2.And(valVec128.AsInt16(), maskLo);

                  Sse2.Store((short*)dstByte, result);
                }
              }
            }
          }

          targetOffset += SQUARES_PER_PLANE;
          rem112++;
          if (rem112 == PLANES_PER_BLOCK) { rem112 = 0; }
        }
      }
    }

    /// <summary>
    /// Optimized scalar fallback using lookup table.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void BitmapRepresentationExpandScalar(ulong[] thisLongs,
                                                                 byte[] thisValues,
                                                                 Memory<Half> targetArrayMemory,
                                                                 int startPlaneIndex,
                                                                 int numPlanesToConvert,
                                                                 int totalElements,
                                                                 bool scale50MoveCounter)
    {
      const int SQUARES_PER_PLANE = 64;
      const int PLANES_PER_BLOCK = 112;
      const int MOVES50_PLANE_MOD = 109;
      const float INV_99 = 1.0f / 99.0f;

      Span<Half> targetSpan = targetArrayMemory.Span;
      ref Half dst = ref MemoryMarshal.GetReference(targetSpan);
      ref ulong longsRef = ref MemoryMarshal.GetArrayDataReference(thisLongs);
      ref byte valsRef = ref MemoryMarshal.GetArrayDataReference(thisValues);

      int endIndex = startPlaneIndex + numPlanesToConvert;
      int targetOffset = 0;
      int rem112 = startPlaneIndex % PLANES_PER_BLOCK;

      // Pin LUT for fast access
      fixed (Half* lutPtr = ByteToHalfLUT)
      {
        for (int outer = startPlaneIndex; outer < endIndex; outer++)
        {
          ulong bits = Unsafe.Add(ref longsRef, outer);

          if (bits == 0UL)
          {
            // Zero the plane
            targetSpan.Slice(targetOffset, SQUARES_PER_PLANE).Clear();
          }
          else
          {
            float val = (float)Unsafe.Add(ref valsRef, outer);
            if (scale50MoveCounter && rem112 == MOVES50_PLANE_MOD)
            {
              val *= INV_99;
            }

            ref Half dstStripe = ref Unsafe.Add(ref dst, targetOffset);

            if (bits == ulong.MaxValue)
            {
              // All bits set - fill with value
              Half hval = (Half)val;
              for (int i = 0; i < 64; i++)
              {
                Unsafe.Add(ref dstStripe, i) = hval;
              }
            }
            else if (val == 1.0f)
            {
              // Value is 1.0 - use LUT directly without scaling
              byte* bitsBytes = (byte*)&bits;
              for (int byteIdx = 0; byteIdx < 8; byteIdx++)
              {
                byte b = bitsBytes[byteIdx];
                Half* lutEntry = lutPtr + b * 8;
                ref Half dstByte = ref Unsafe.Add(ref dstStripe, byteIdx * 8);

                // Copy 8 Half values from LUT
                Unsafe.Add(ref dstByte, 0) = lutEntry[0];
                Unsafe.Add(ref dstByte, 1) = lutEntry[1];
                Unsafe.Add(ref dstByte, 2) = lutEntry[2];
                Unsafe.Add(ref dstByte, 3) = lutEntry[3];
                Unsafe.Add(ref dstByte, 4) = lutEntry[4];
                Unsafe.Add(ref dstByte, 5) = lutEntry[5];
                Unsafe.Add(ref dstByte, 6) = lutEntry[6];
                Unsafe.Add(ref dstByte, 7) = lutEntry[7];
              }
            }
            else
            {
              // General case with value scaling
              Half hval = (Half)val;
              Half zero = default;
              byte* bitsBytes = (byte*)&bits;

              for (int byteIdx = 0; byteIdx < 8; byteIdx++)
              {
                byte b = bitsBytes[byteIdx];
                ref Half dstByte = ref Unsafe.Add(ref dstStripe, byteIdx * 8);

                if (b == 0)
                {
                  // Zero all 8
                  Unsafe.Add(ref dstByte, 0) = zero;
                  Unsafe.Add(ref dstByte, 1) = zero;
                  Unsafe.Add(ref dstByte, 2) = zero;
                  Unsafe.Add(ref dstByte, 3) = zero;
                  Unsafe.Add(ref dstByte, 4) = zero;
                  Unsafe.Add(ref dstByte, 5) = zero;
                  Unsafe.Add(ref dstByte, 6) = zero;
                  Unsafe.Add(ref dstByte, 7) = zero;
                }
                else if (b == 0xFF)
                {
                  // Fill all 8 with value
                  Unsafe.Add(ref dstByte, 0) = hval;
                  Unsafe.Add(ref dstByte, 1) = hval;
                  Unsafe.Add(ref dstByte, 2) = hval;
                  Unsafe.Add(ref dstByte, 3) = hval;
                  Unsafe.Add(ref dstByte, 4) = hval;
                  Unsafe.Add(ref dstByte, 5) = hval;
                  Unsafe.Add(ref dstByte, 6) = hval;
                  Unsafe.Add(ref dstByte, 7) = hval;
                }
                else
                {
                  // Expand each bit
                  Unsafe.Add(ref dstByte, 0) = (b & 0x01) != 0 ? hval : zero;
                  Unsafe.Add(ref dstByte, 1) = (b & 0x02) != 0 ? hval : zero;
                  Unsafe.Add(ref dstByte, 2) = (b & 0x04) != 0 ? hval : zero;
                  Unsafe.Add(ref dstByte, 3) = (b & 0x08) != 0 ? hval : zero;
                  Unsafe.Add(ref dstByte, 4) = (b & 0x10) != 0 ? hval : zero;
                  Unsafe.Add(ref dstByte, 5) = (b & 0x20) != 0 ? hval : zero;
                  Unsafe.Add(ref dstByte, 6) = (b & 0x40) != 0 ? hval : zero;
                  Unsafe.Add(ref dstByte, 7) = (b & 0x80) != 0 ? hval : zero;
                }
              }
            }
          }

          targetOffset += SQUARES_PER_PLANE;
          rem112++;
          if (rem112 == PLANES_PER_BLOCK) { rem112 = 0; }
        }
      }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="destinationBuffer">buffer to receive values. NOTE! This is assumed to start out cleared (all zeros)</param>
    /// <param name="encodingType"></param>
    internal void ConvertToFlat(int startPosToConvertIndex, int numPositionsToConvert,
                                Memory<Half> destinationBuffer, bool scale50MoveCounter)
    {
      int numPlanesTotal = NumPos * TOTAL_NUM_PLANES_ALL_HISTORIES;
      int startPlanesToConvert = startPosToConvertIndex * TOTAL_NUM_PLANES_ALL_HISTORIES;
      int numPlanesToConvert = numPositionsToConvert * TOTAL_NUM_PLANES_ALL_HISTORIES;

      if (false)
      {
        BitmapRepresentationExpandSLOW(PosPlaneBitmaps, PosPlaneValues, destinationBuffer,
                                       numPlanesToConvert, scale50MoveCounter); //old slow version
        return;
      }

      BitmapRepresentationExpand(PosPlaneBitmaps, PosPlaneValues, destinationBuffer,
                                 startPlanesToConvert, numPlanesToConvert, numPlanesTotal, scale50MoveCounter);
    }


    #region Dump diagostics

    public void DumpDecoded()
    {
      throw new NotImplementedException();
      //DumpDecoded(ValuesFlatFromPlanes(default, false, false).ToArray(), TOTAL_NUM_PLANES_ALL_HISTORIES);
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

