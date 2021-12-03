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
  public class EncodedPositionBatchFlatSlice : IEncodedPositionBatchFlat
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

    public Span<ulong> PosPlaneBitmaps => Parent.PosPlaneBitmaps.Slice(StartIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, Length * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

    public Span<byte> PosPlaneValues => Parent.PosPlaneValues.Slice(StartIndex * EncodedPositionWithHistory.NUM_PLANES_TOTAL, Length * EncodedPositionWithHistory.NUM_PLANES_TOTAL);

    public Span<MGPosition> Positions
    {
      get => Parent.Positions == default ? default : Parent.Positions.Slice(StartIndex, Length);
      set => value.CopyTo(Parent.Positions);
    }

    public Span<ulong> PositionHashes
    {
      get => Parent.PositionHashes == default ? default : Parent.PositionHashes.Slice(StartIndex, Length);
      set => value.CopyTo(Parent.PositionHashes);
    }
    public Span<MGMoveList> Moves
    {
      get => Parent.Moves == default ? default : Parent.Moves.Slice(StartIndex, Length);
      set => value.CopyTo(Parent.Moves);
    }

    public Span<float> W => Parent.W.Slice(StartIndex, Length);

    public Span<float> L => Parent.L.Slice(StartIndex, Length);

    public Span<FP16> Policy => Parent.Policy.Slice(StartIndex, Length);

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

    public float[] ValuesFlatFromPlanes(float[] preallocatedBuffer = null)
    {
      throw new NotImplementedException();
    }
  }

}
