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
using Ceres.Chess.MoveGen;
using System;

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
    public readonly IEncodedPositionBatchFlat Parent;

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
      Parent = parent;
      StartIndex = startIndex;
      Length = length;

    }

    public Memory<ulong> PosPlaneBitmaps => Parent.PosPlaneBitmaps.Slice(StartIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, Length * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

    public Memory<byte> PosPlaneValues => Parent.PosPlaneValues.Slice(StartIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, Length * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

    public Memory<MGPosition> Positions
    {
      get => Parent.Positions.IsEmpty ? default : Parent.Positions.Slice(StartIndex, Length);
      set => value.CopyTo(Parent.Positions);
    }

    public Memory<ulong> PositionHashes
    {
      get => Parent.PositionHashes.IsEmpty ? default : Parent.PositionHashes.Slice(StartIndex, Length);
      set => value.CopyTo(Parent.PositionHashes);
    }
    public Memory<MGMoveList> Moves
    {
      get => Parent.Moves.IsEmpty ? default : Parent.Moves.Slice(StartIndex, Length);
      set => value.CopyTo(Parent.Moves);
    }

    public Memory<byte> LastMovePlies
    {
      get => Parent.LastMovePlies.IsEmpty ? default : Parent.LastMovePlies.Slice(StartIndex, Length * 64);
      set => value.CopyTo(Parent.LastMovePlies);
    }



    public Memory<float> W => Parent.W.Slice(StartIndex, Length);

    public Memory<float> L => Parent.L.Slice(StartIndex, Length);

    public Memory<FP16> Policy => Parent.Policy.Slice(StartIndex, Length);

    public int NumPos => Length;

    public EncodedPositionType TrainingType => Parent.TrainingType;

    public short PreferredEvaluatorIndex => Parent.PreferredEvaluatorIndex;

    public bool PositionsUseSecondaryEvaluator 
    { 
      get {  return Parent.PositionsUseSecondaryEvaluator; } 
      set { Parent.PositionsUseSecondaryEvaluator = value; } 
    }

    public IEncodedPositionBatchFlat GetSubBatch(int startIndex, int count)
    {
      throw new NotImplementedException();
    }

    public float[] ValuesFlatFromPlanes(float[] preallocatedBuffer, bool nhwc, bool scale50MoveCounters)
    {
      throw new NotImplementedException();
    }

    public Memory<EncodedPositionWithHistory> PositionsBuffer
    {
      get
      {
        return Parent.PositionsBuffer.Slice(StartIndex, Length);
      }
    }

  }

}
