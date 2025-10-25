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
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.LC0.Batches
{
  /// <summary>
  /// Represents a subset (slice) of positions from a flat batch.
  /// </summary>
  public readonly struct EncodedPositionBatchFlatSlice : IEncodedPositionBatchFlat
  {
    /// <summary>
    /// Parent batch from which this slice was taken.
    /// </summary>
    public readonly IEncodedPositionBatchFlat SliceParent;

    /// <summary>
    /// Starting index of the slice.
    /// </summary>
    public readonly int StartIndex;

    /// <summary>
    /// Length of the slice.
    /// </summary>
    public readonly int Length;


    /// <summary>
    /// Constructor which takes a slice from a specified flat batch.
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="startIndex"></param>
    /// <param name="length"></param>
    public EncodedPositionBatchFlatSlice(IEncodedPositionBatchFlat parent, int startIndex, int length)
    {
      SliceParent = parent;
      StartIndex = startIndex;
      Length = length;

    }

    public IEncodedPositionBatchFlat Parent => SliceParent;


    public Memory<ulong> PosPlaneBitmaps => SliceParent.PosPlaneBitmaps.Slice(StartIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, Length * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

    public Memory<byte> PosPlaneValues => SliceParent.PosPlaneValues.Slice(StartIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, Length * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

    public Memory<Half[]> States => SliceParent.States.IsEmpty ? default : SliceParent.States.Slice(StartIndex, Length);

    public Memory<MGPosition> Positions
    {
      get => SliceParent.Positions.IsEmpty ? default : SliceParent.Positions.Slice(StartIndex, Length);
      set => value.CopyTo(SliceParent.Positions);
    }

    public Memory<ulong> PositionHashes
    {
      get => SliceParent.PositionHashes.IsEmpty ? default : SliceParent.PositionHashes.Slice(StartIndex, Length);
      set => value.CopyTo(SliceParent.PositionHashes);
    }
    public Memory<MGMoveList> Moves
    {
      get => SliceParent.Moves.IsEmpty ? default : SliceParent.Moves.Slice(StartIndex, Length);
      set => value.CopyTo(SliceParent.Moves);
    }

    public Memory<byte> LastMovePlies
    {
      get => SliceParent.LastMovePlies.IsEmpty ? default : SliceParent.LastMovePlies.Slice(StartIndex * 64, Length * 64);
      set => value.CopyTo(SliceParent.LastMovePlies);
    }


    public int NumPos => Length;

    public EncodedPositionType TrainingType => SliceParent.TrainingType;

    public short PreferredEvaluatorIndex => SliceParent.PreferredEvaluatorIndex;

    public bool PositionsUseSecondaryEvaluator
    {
      get { return SliceParent.PositionsUseSecondaryEvaluator; }
      set { SliceParent.PositionsUseSecondaryEvaluator = value; }
    }

    public IEncodedPositionBatchFlat GetSubBatch(int startIndex, int count)
    {
      throw new NotImplementedException();
    }


    public void ConvertValuesToFlatFromPlanes(Memory<Half> targetBuffer, bool nhwc, bool scale50MoveCounter)
    {
      (SliceParent as EncodedPositionBatchFlat).ConvertToFlat(StartIndex, Length, targetBuffer, scale50MoveCounter);
    }



    public Memory<EncodedPositionWithHistory> PositionsBuffer
    {
      get
      {
        return SliceParent.PositionsBuffer.Slice(StartIndex, Length);
      }
    }

    Memory<Half[]> IEncodedPositionBatchFlat.States
    {
      get => SliceParent.States.IsEmpty ? default : SliceParent.States.Slice(StartIndex, Length);
      set => value.CopyTo(SliceParent.States);
    }


    /// <summary>
    /// Zero out the history planes for all positions in the batch.
    /// </summary>
    public void ZeroHistoryPlanes() => throw new NotImplementedException();
  }
}