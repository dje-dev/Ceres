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

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators;
using System;
using System.Collections;
using System.Collections.Generic;

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
    Span<ulong> PosPlaneBitmaps { get;  }

    /// <summary>
    /// One byte for each bitmap with corresopnding value.
    /// These values are generally 0 or 1, 
    /// except for Move50 plane which can be any integer from 0 to 99.
    /// </summary>
    Span<byte> PosPlaneValues { get; }

    /// <summary>
    /// Optionally the associated MGPositions
    /// </summary>
    Span<MGPosition> Positions { get; set; }

    /// <summary>
    /// Optionally the associated hashes of the positions
    /// </summary>
    Span<ulong> PositionHashes { get; set; }

    /// <summary>
    /// Optionally the set of moves from this position
    /// </summary>
    Span<MGMoveList> Moves { get; set; }

    /// <summary>
    /// Span of W (winning probability forecasts)
    /// </summary>
    Span<float> W { get; }

    /// <summary>
    /// Span of L (loss probability forecasts)
    /// </summary>
    Span<float> L { get; }

    /// <summary>
    /// Span of policy probabilities, e.g. 0.1 for 10% probability of a move being chosen
    /// </summary>
    Span<FP16> Policy { get; }

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

    float[] ValuesFlatFromPlanes(float[] preallocatedBuffer = null);

    
    public IEncodedPositionBatchFlat GetSubBatchSlice(int startIndex, int count)
    {
      return new EncodedPositionBatchFlatSlice(this, startIndex, count);
    }


    public EncodedPositionBatchFlat Mirrored(bool fillInMissingPlanes)
    {
      if (Positions == null) throw new NotImplementedException("Implemenation restriction: Mirrored only implemented when Positions is not null");

      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(NumPos, NNEvaluator.InputTypes.Boards);
      for (int i = 0; i < NumPos; i++)
      {
        Position pos = MGChessPositionConverter.PositionFromMGChessPosition(in Positions[i]);

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
      if (Moves == null && Positions.Length == NumPos)
      {
        MGMoveList[] moves = new MGMoveList[NumPos];
        for (int i = 0; i < moves.Length; i++)
        {
          MGMoveList thisPosMoves = new MGMoveList();
          MGMoveGen.GenerateMoves(in Positions[i], thisPosMoves);
          moves[i] = thisPosMoves;
        }
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

        for (int i = 0; i < Moves[pos].NumMovesUsed; i++)
        {
          EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(Moves[pos].MovesArray[i]);
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

        BitArray mask = new BitArray(NumPos * 1858);
        for (int pos = 0; pos < NumPos; pos++)
        {
          int thisOffset = 1858 * pos;
          for (int i = 0; i < Moves[pos].NumMovesUsed; i++)
          {
            EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(Moves[pos].MovesArray[i]);
            mask[thisOffset + encodedMove.IndexNeuralNet] = true;
          }
        }

        return mask;
      }
    }

    #endregion
  }


}
