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
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Chess.LC0.Batches
{
  /// <summary>
  /// Facilitates assembling batches of encoded positions which
  /// can then be evaluated by neural network.
  /// </summary>
  public class EncodedPositionBatchBuilder
  {
    /// <summary>
    /// Maximum number of positions in batch.
    /// </summary>
    public readonly int MaxPositions;

    /// <summary>
    /// Number of positions currently in the batch.
    /// </summary>
    public int NumPositionsAdded;

    /// <summary>
    /// The types of ancillary information that should be retained about the batch inputs.
    /// </summary>
    public readonly NNEvaluator.InputTypes InputsRequired;


    /// <summary>
    /// If the batch is at maximum capacity.
    /// </summary>
    public bool IsFull => NumPositionsAdded == MaxPositions;

    /// <summary>
    /// Clears contents of batch.
    /// </summary>
    public void ResetBatch() => NumPositionsAdded = 0;

    /// <summary>
    /// Underlying raw encoded batch.
    /// </summary>
    EncodedPositionBatchFlat batch;

    /// <summary>
    /// Optional associated positions.
    /// </summary>
    EncodedPositionWithHistory[] pendingPositions;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxPositions"></param>
    /// <param name="inputsRequired"></param>
    public EncodedPositionBatchBuilder(int maxPositions, NNEvaluator.InputTypes inputsRequired)
    {
      MaxPositions = maxPositions;
      InputsRequired = inputsRequired;

      pendingPositions = new EncodedPositionWithHistory[MaxPositions];
      batch = new EncodedPositionBatchFlat(EncodedPositionType.PositionAndTrainingData, MaxPositions);

      if (InputsRequired.HasFlag(NNEvaluator.InputTypes.Positions)) batch.Positions = new MGPosition[MaxPositions];
      if (InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes)) batch.PositionHashes = new ulong[MaxPositions];
      if (InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves)) batch.Moves = new MGMoveList[MaxPositions];
    }


    /// <summary>
    /// Adds a position (with history) to the batch.
    /// </summary>
    /// <param name="positionEncoded"></param>
    public void Add(in EncodedPositionWithHistory positionEncoded)
    {
      ulong zobristHashForCaching = 0;
      MGPosition positionMG = default;
      MGMoveList moves = null;

      if (InputsRequired > NNEvaluator.InputTypes.Boards)
      {
        // Decode main position from encoded board
        Position position = EncodedPositionWithHistory.PositionFromEncodedPosition(in positionEncoded);
        positionMG = MGPosition.FromPosition(in position);

        // Compute hash
        // TODO: review this:
        //    - does it generate the same hash as the version used elsewhere in this class (based off of Positions)
        //      (and is it actually necessary to make the hash functions exactly the same?)
        //    - it seems to only hash the top board, without additional MiscInfo features such as en passant column
        zobristHashForCaching = EncodedBoardZobrist.ZobristHash(positionEncoded.GetPlanesForHistoryBoard(0));
        //ulong zobristHashForCaching = EncodedBoardZobrist.ZobristHash(posSeq, PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98);

        // Generate moves
        moves = new MGMoveList();
        MGMoveGen.GenerateMoves(in positionMG, moves);
      }

      Add(in positionEncoded, in positionMG, zobristHashForCaching, moves);
    }

    /// <summary>
    /// Adds a position to the batch.
    /// </summary>
    /// <param name="positionEncoded"></param>
    /// <param name="position"></param>
    /// <param name="positionHash"></param>
    /// <param name="moves"></param>
    public void Add(in EncodedPositionWithHistory positionEncoded, in MGPosition position, ulong positionHash, MGMoveList moves)
    {
      if (batch.Positions == null) throw new Exception("AddPosition requires that constructor argument hashesAndMovesRequired be true");

      lock (this)
      {
        if (NumPositionsAdded >= MaxPositions) throw new Exception($"PositionBatchBuilder overflow, already added maximum of {MaxPositions}");
        if (InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes) && positionHash == default) throw new ArgumentException(nameof(positionHash));
        if (InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves) && moves == default) throw new ArgumentException(nameof(moves));

        batch.Positions[NumPositionsAdded] = position;
        if (InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes)) batch.PositionHashes[NumPositionsAdded] = positionHash;
        if (InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves)) batch.Moves[NumPositionsAdded] = moves;

        pendingPositions[NumPositionsAdded] = positionEncoded;

        NumPositionsAdded++;
      }
    }


    /// <summary>
    /// Adds a position to the batch.
    /// </summary>
    /// <param name="position"></param>
    public void Add(in Position position, bool fillInMissingPlanes) => Add(new PositionWithHistory(position), fillInMissingPlanes);
    

    /// <summary>
    /// Adds a position with history to the batch.
    /// </summary>
    /// <param name="moveSequence"></param>
    public void Add(PositionWithHistory moveSequence, bool fillInMissingPlanes)
    {
      MGPosition finalPositionMG = moveSequence.FinalPosMG;

      Position[] posSeq = moveSequence.GetPositions();
      PositionRepetitionCalc.SetRepetitionsCount(posSeq);

      ref EncodedPositionWithHistory posRaw = ref pendingPositions[NumPositionsAdded];
      posRaw.SetFromSequentialPositions(posSeq, fillInMissingPlanes);

      // Compute hash
      ulong zobristHashForCaching = EncodedBoardZobrist.ZobristHash(posSeq, PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98);

      // Generate moves
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in finalPositionMG, moves);

      Add(in posRaw, in finalPositionMG, zobristHashForCaching, moves);
    }

    /// <summary>
    /// Returns the encoded batch.
    /// </summary>
    /// <returns></returns>
    public EncodedPositionBatchFlat GetBatch()
    {
      batch.Set(new Span<EncodedPositionWithHistory>(pendingPositions).Slice(0, NumPositionsAdded), NumPositionsAdded, true);
      return batch;
    }


    // TODO: relocate this
    public static void ExpandPositionsInfoFlatRepresentation(IEncodedPositionBatchFlat batch,
                                                             int numPositions, int numBoardsPerPosition, 
                                                             float[] flatValues, bool[] excludePlanes = null)
    {
      Debug.Assert(flatValues.Length >= numPositions * numBoardsPerPosition * EncodedPositionBoard.NUM_PLANES_PER_BOARD * 64);

      Span<byte> planeValues = batch.PosPlaneValues;
      Span<ulong> bitmaps = batch.PosPlaneBitmaps;

      int planesIndex = 0;
      int flatsIndex = 0;
      for (int posIndex = 0; posIndex < numPositions; posIndex++)
      {
        for (int boardIndex = 0; boardIndex < EncodedPositionBoards.NUM_MOVES_HISTORY; boardIndex++)
        {
          if (boardIndex < numBoardsPerPosition)
          {
            for (int planeInBoardIndex = 0; planeInBoardIndex < EncodedPositionBoard.NUM_PLANES_PER_BOARD; planeInBoardIndex++)
            {
              if (excludePlanes == null || !excludePlanes[planeInBoardIndex])
              {

                if (bitmaps[planesIndex] == 0)
                {
                  for (int i = 0; i < 64; i++)
                    flatValues[flatsIndex++] = 0;
                }
                else
                {
                  float planeValue = (float)planeValues[planesIndex];
                  BitVector64 bits = new BitVector64(bitmaps[planesIndex]);
                  for (int i = 0; i < 64; i++)
                  {
                    bool isSet = bits.BitIsSet(i);
                    flatValues[flatsIndex++] = isSet ? planeValue : 0f;
                  }
                }
              }

              planesIndex++;
            }
          }
          else
          {
            // Skip these planes appearing in the input (but not copied to output)
            planesIndex += EncodedPositionBoard.NUM_PLANES_PER_BOARD;
          }
        }

        // Now Misc Info
        for (int i = 0; i < 8; i++)
        {
          float planeValue = (float)planeValues[planesIndex];
          flatValues[flatsIndex++] = planeValue;

          planesIndex++;
        }
      }
    }


    public static void ExpandPositionsInfoFlatRepresentationPacked(IEncodedPositionBatchFlat batch, int numPositions, int numBoardsPerPosition, 
                                                                   long[] planesOut, float[] floatsOut,  bool[] excludePlanes = null)
    {
      Span<byte> planeValues = batch.PosPlaneValues;
      Span<ulong> bitmaps = batch.PosPlaneBitmaps;

      int planesIndexSource = 0;

      int planesIndexDest = 0;
      int floatsIndexDest = 0;
      for (int posIndex = 0; posIndex < numPositions; posIndex++)
      {
        for (int boardIndex = 0; boardIndex < EncodedPositionBoards.NUM_MOVES_HISTORY; boardIndex++)
        {
          if (boardIndex < numBoardsPerPosition)
          {
            for (int planeInBoardIndex = 0; planeInBoardIndex < EncodedPositionBoard.NUM_PLANES_PER_BOARD; planeInBoardIndex++)
            {
              if (excludePlanes == null || !excludePlanes[planeInBoardIndex])
              {
                float planeValue = (float)planeValues[planesIndexSource];
                if (planeValue == 0)
                  planesOut[planesIndexDest] = 0;
                else if (planeValue == 1)
                  planesOut[planesIndexDest] = (long)bitmaps[planesIndexSource];
                else
                  throw new Exception($"Unexpected {planeValue}");

                planesIndexDest++;
              }

              planesIndexSource++;
            }
          }
          else
          {
            // Skip these planes appearing in the input (but not copied to output)
            planesIndexSource += EncodedPositionBoard.NUM_PLANES_PER_BOARD;
          }
        }

        // Now Misc Info into floats
        for (int i = 0; i < EncodedPositionWithHistory.NUM_MISC_PLANES; i++)
        {
          float planeValue = (float)planeValues[planesIndexSource++];
          floatsOut[floatsIndexDest++] = planeValue;
        }

      }


    }
  }
}

#if NOTUSED
    public static void ExpandPositionsInfoFlatRepresentation2D(ILZTrainingPositionServerBatch batch, 
                                                               int numPositions, int numBoardsPerPosition, float[,] flatValues,
                                                               bool[] excludePlanes = null)
    {
      if (excludePlanes != null) throw new NotImplementedException();

      Debug.Assert(flatValues.Length >= numPositions * numBoardsPerPosition * LZBoard.NUM_PLANES_PER_BOARD * 64);

      Span<byte> planeValues = batch.PosPlaneValuesTimes;
      Span<ulong> bitmaps = batch.PosPlaneBitmaps;

      int planesIndex = 0;

      for (int posIndex = 0; posIndex < numPositions; posIndex++)
      {
        int flatsIndex = 0;
        for (int boardIndex = 0; boardIndex < LZBoardsHistory.NUM_MOVES_HISTORY; boardIndex++)
        {
          if (boardIndex < numBoardsPerPosition)
          {
            for (int planeInBoardIndex = 0; planeInBoardIndex < LZBoard.NUM_PLANES_PER_BOARD; planeInBoardIndex++)
            {
              float planeValue = (float)planeValues[planesIndex];
              BitVector64 bits = new BitVector64(bitmaps[planesIndex]);
              for (int bitIndex = 0; bitIndex < 64; bitIndex++)
              {
                bool isSet = bits.BitIsSet(bitIndex);
                flatValues[posIndex, flatsIndex++] = isSet ? planeValue : 0f;
              }

              planesIndex++;
            }
          }
          else
          {
            // Skip these planes appearing in the input (but not copied to output)
            planesIndex += LZBoard.NUM_PLANES_PER_BOARD;
          }
        }

        // Now Misc Info
        for (int i = 0; i < 8; i++)
        {
          float planeValue = (float)planeValues[planesIndex];
          flatValues[posIndex, flatsIndex++] = planeValue;

          planesIndex++; 
        }
      }

    }

    public static (long[,], float[]) ExpandPositionsInfoFlatRepresentationSparse(ILZTrainingPositionServerBatch batch, int numPositions, int numBoardsPerPosition, bool[] excludePlanes = null)
    {
      if (excludePlanes != null) throw new NotImplementedException();

      List<(int, int)> indices = new List<(int, int)>();
      List<float> values = new List<float>();
      int numFound = 0;

      Span<byte> planeValues = batch.PosPlaneValuesTimes;
      Span<ulong> bitmaps = batch.PosPlaneBitmaps;

      int planesIndex = 0;

      for (int posIndex = 0; posIndex < numPositions; posIndex++)
      {
        int colIndex = 0;
        for (int boardIndex = 0; boardIndex < LZBoardsHistory.NUM_MOVES_HISTORY; boardIndex++)
        {
          if (boardIndex < numBoardsPerPosition)
          {
            for (int planeInBoardIndex = 0; planeInBoardIndex < LZBoard.NUM_PLANES_PER_BOARD; planeInBoardIndex++)
            {
              float planeValue = (float)planeValues[planesIndex];
              BitVector64 bits = new BitVector64(bitmaps[planesIndex]);
              for (int bitIndex = 0; bitIndex < 64; bitIndex++)
              {
                bool isSet = bits.BitIsSet(bitIndex);
                if (isSet)
                {
                  indices.Add((posIndex, colIndex));
                  values.Add(planeValue);
                  numFound++;
                }
                colIndex++;
              }

              planesIndex++;
            }
          }
          else
          {
            // Skip these planes appearing in the input (but not copied to output)
            planesIndex += LZBoard.NUM_PLANES_PER_BOARD;
          }
        }

        // Now Misc Info (note that we write only 1 copy, not 64)
        for (int i = 0; i < 8; i++)
        {
          float planeValue = (float)planeValues[planesIndex];
          if (planeValue != 0)
          {
            indices.Add((posIndex, colIndex));
            values.Add(planeValue);
            numFound++;
          }
          colIndex++;

          planesIndex++;
        }
      }

      long[,] indicesA = new long[numFound, 2];
      int next = 0;
      foreach ((int, int) index in indices)
      {
        indicesA[next, 0] = index.Item1;
        indicesA[next, 1] = index.Item2;
        next++;
      }
      return (indicesA, values.ToArray());
    }
#endif
