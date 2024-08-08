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

using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Static helper methods for extracting legal moves from TPGRecords
  /// into arrays of (index, value) pairs which are stored in TPG records.
  /// </summary>
  public static class TPGRecordMovesExtractor
  {
    /// <summary>
    /// Number of per-position slots in array of legal moves sent to remote evaluator.
    /// </summary>
    public const int NUM_MOVE_SLOTS_PER_REQUEST = TPGRecord.MAX_MOVES;


    /// <summary>
    /// Extracts legal move indices for a single TPGRecord within specified array,
    /// storing them in legalMoveIndices at an offset appropriate for that position
    /// (assuming each position has NUM_MOVE_SLOTS_PER_REQUEST slots).
    /// </summary>
    /// <param name="recs"></param>
    /// <param name="moves"></param>
    /// <param name="legalMoveIndices"></param>
    /// <param name="i"></param>
    /// <exception cref="Exception"></exception>
    public static void ExtractLegalMoveIndicesForIndex(ReadOnlySpan<TPGRecord> recs, MGMoveList moves, short[] legalMoveIndices, int i)
    {
      MGMoveList theseMoves;
      if (moves == null || moves.IsEmpty)
      {
        MGPosition mgPos = recs[i].FinalPosition.ToMGPosition;
        theseMoves = new MGMoveList();
        MGMoveGen.GenerateMoves(mgPos, theseMoves);
      }
      else
      {
        theseMoves = moves;
      }

      int numMovesToProcess = theseMoves.NumMovesUsed;
      if (numMovesToProcess > NUM_MOVE_SLOTS_PER_REQUEST)
      {
        Console.WriteLine($"Too many moves for position: {numMovesToProcess} > {NUM_MOVE_SLOTS_PER_REQUEST}, skip overflow moves");
        numMovesToProcess = NUM_MOVE_SLOTS_PER_REQUEST;
      }

      int indexOfIndexZero = -1;
      for (int j = 0; j < numMovesToProcess; j++)
      {
        MGMove thisMove = theseMoves.MovesArray[j];
        short nnIndex = (short)ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(thisMove).IndexNeuralNet;
        if (nnIndex == 0)
        {
          if (indexOfIndexZero != -1)
          {
            throw new Exception(" Duplicate move index 0 at: " + j + " " + indexOfIndexZero);
          }
          indexOfIndexZero = j;
        }
        legalMoveIndices[i * NUM_MOVE_SLOTS_PER_REQUEST + j] = nnIndex;
      }

      // Since we use index 0 as a sentinel (unless appears in first slot),
      // if index 0 actually is present in the legal moves, 
      // swap it into the first slot.
      if (indexOfIndexZero != -1)
      {
        SwapSlotIntoFirstSlot(i * NUM_MOVE_SLOTS_PER_REQUEST, indexOfIndexZero, legalMoveIndices);
      }
    }


    /// <summary>
    /// Swaps element at slotIndex into first slot,
    /// starting from a subarray of elements at index baseIndex.
    /// </summary>
    /// <param name="baseIndex"></param>
    /// <param name="slotIndex"></param>
    /// <param name="slots"></param>
    public static void SwapSlotIntoFirstSlot(int baseIndex, int slotIndex, short[] slots)
    {
      int temp = slots[baseIndex + 0];
      slots[baseIndex + 0] = slots[baseIndex + slotIndex];
      slots[baseIndex + slotIndex] = (short)temp;
    }
  }

}