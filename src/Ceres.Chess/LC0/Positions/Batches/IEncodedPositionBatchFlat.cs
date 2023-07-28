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
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.Base.Threading;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators;

#endregion

namespace Ceres.Chess.LC0.Batches
{
  /// <summary>
  /// Interface implemented by batches of encoded positions.
  /// </summary>
  public interface IEncodedPositionBatchFlat
  {
    /// <summary>
    // Bitmaps for the multiple board planes.
    /// </summary>
    Memory<ulong> PosPlaneBitmaps { get;  }

    /// <summary>
    /// One byte for each bitmap with corresopnding value.
    /// These values are generally 0 or 1, 
    /// except for Move50 plane which can be any integer from 0 to 99.
    /// </summary>
    Memory<byte> PosPlaneValues { get; }

    /// <summary>
    /// Optionally the associated MGPositions
    /// </summary>
    Memory<MGPosition> Positions { get; set; }

    /// <summary>
    /// Optionally the associated hashes of the positions
    /// </summary>
    Memory<ulong> PositionHashes { get; set; }

    /// <summary>
    /// Optionally the arrays of "plies since last move on square."
    /// </summary>
    Memory<byte> LastMovePlies{ get; set; }

    /// <summary>
    /// Optionally the set of moves from this position
    /// </summary>
    Memory<MGMoveList> Moves { get; set; }

    /// <summary>
    /// Memory of W (winning probability forecasts)
    /// </summary>
    Memory<float> W { get; }

    /// <summary>
    /// Memory of L (loss probability forecasts)
    /// </summary>
    Memory<float> L { get; }

    /// <summary>
    /// Memory of policy probabilities, e.g. 0.1 for 10% probability of a move being chosen
    /// </summary>
    Memory<FP16> Policy { get; }

    /// <summary>
    /// If originated from EncodedPositionWithHistory then
    /// this field optionally holds the origin data array.
    /// </summary>
    Memory<EncodedPositionWithHistory> PositionsBuffer 
    {
      get
      {
        return default;
      }
    }

    /// <summary>
    /// Number of positions actually used within the batch
    /// </summary>
    int NumPos { get; }

    EncodedPositionType TrainingType { get; }

    /// <summary>
    /// Optionally (if multiple evaluators are configured) 
    /// the index of which executor should be used for this batch
    /// </summary>
    short PreferredEvaluatorIndex { get; }

    bool PositionsUseSecondaryEvaluator { get; set; }

    #region Implmentation

    float[] ValuesFlatFromPlanes(float[] preallocatedBuffer, bool nhwc, bool scale50MoveCounter);

    
    public IEncodedPositionBatchFlat GetSubBatchSlice(int startIndex, int count)
    {
      return new EncodedPositionBatchFlatSlice(this, startIndex, count);
    }


    public EncodedPositionBatchFlat Mirrored(bool fillInMissingPlanes)
    {
      if (Positions.IsEmpty) throw new NotImplementedException("Implementation restriction: Mirrored only implemented when Positions is not null");

      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(NumPos, NNEvaluator.InputTypes.Boards);
      for (int i = 0; i < NumPos; i++)
      {
        Position pos = MGChessPositionConverter.PositionFromMGChessPosition(in Positions.Span[i]);

        // Mirror the position if it is equivalent to do so
        if (!pos.MiscInfo.BlackCanOO && !pos.MiscInfo.BlackCanOOO && !pos.MiscInfo.WhiteCanOO && !pos.MiscInfo.WhiteCanOOO)
        {
          pos = pos.Mirrored;
        }

        builder.Add(in pos, fillInMissingPlanes);
      }
      return builder.GetBatch();
    }

    /// <summary>
    /// If possible, generates moves for this position and assigns to Moves field
    /// if the Moves field is not already initialized.
    /// </summary>
    public void TrySetMoves()
    {
      if (Moves.IsEmpty && Positions.Length == NumPos)
      {
        MGMoveList[] moves = new MGMoveList[NumPos];
        ParallelOptions parallelOptions = ParallelUtils.ParallelOptions(NumPos, 96);
        Parallel.For(0, moves.Length, parallelOptions,
          delegate (int i)
          {
            MGMoveList thisPosMoves = new MGMoveList();
            MGMoveGen.GenerateMoves(in Positions.Span[i], thisPosMoves);
            moves[i] = thisPosMoves;
          });
        Moves = moves;
      }
    }

    /// <summary>
    /// Sets all entries in a specified policy array to -1E10
    /// if they are illegal moves for the corresponding position.
    /// </summary>
    /// <param name="policyValues"></param>
    public void MaskIllegalMovesInPolicyArray(Span<float> policyValues)
    {
      TrySetMoves();

      HashSet<int> legalIndices = new HashSet<int>(96);
      for (int pos = 0; pos < NumPos; pos++)
      {
        legalIndices.Clear();

        Span<MGMoveList> moves = Moves.Span;
        for (int i = 0; i < moves[pos].NumMovesUsed; i++)
        {
          EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(moves[pos].MovesArray[i]);
          legalIndices.Add(encodedMove.IndexNeuralNet);
        }

        int thisOffset = 1858 * pos;
        for (int i = 0; i < 1858; i++)
        {
          if (!legalIndices.Contains(i))
          {
            policyValues[thisOffset + i] = -1E10f;
          }
        }
      }

    }

    /// <summary>
    /// Returns BitArray of vaild moves for all positions and moves in batch.
    /// </summary>
    public BitArray ValidMovesMasks
    {
      get
      {
        TrySetMoves();

        Span<MGMoveList> moves = Moves.Span;
        BitArray mask = new BitArray(NumPos * 1858);
        for (int pos = 0; pos < NumPos; pos++)
        {
          int thisOffset = 1858 * pos;
          for (int i = 0; i < moves[pos].NumMovesUsed; i++)
          {
            EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(moves[pos].MovesArray[i]);
            mask[thisOffset + encodedMove.IndexNeuralNet] = true;
          }
        }

        return mask;
      }
    }

    #endregion
  }


}
