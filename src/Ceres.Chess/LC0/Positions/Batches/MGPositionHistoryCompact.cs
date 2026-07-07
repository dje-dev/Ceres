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
using System.Runtime.CompilerServices;

using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.LC0.Batches
{
  /// <summary>
  /// Compact fixed-size record of the sequential MGPositions (up to 8, stored oldest first)
  /// backing one neural network evaluation position.
  ///
  /// Serves as a much smaller substitute for retaining EncodedPositionWithHistory in
  /// EncodedPositionBatchFlat.PositionsBuffer (~920 bytes/position, allocated fresh every batch)
  /// for consumers such as TPG conversion which only need the underlying position sequence
  /// (328 bytes/position, preallocated per batch, zero steady-state allocation).
  ///
  /// A record with NumPositions == 0 means "not populated" and consumers are expected
  /// to fall back to PositionsBuffer for that batch row (possible in merged pooled batches).
  /// </summary>
  public struct MGPositionHistoryCompact
  {
    /// <summary>
    /// Maximum number of history positions stored.
    /// </summary>
    public const int MAX_POSITIONS = EncodedPositionBatchFlat.NUM_HISTORY_POSITIONS; // 8

    /// <summary>
    /// Number of populated positions (1..MAX_POSITIONS), stored oldest to newest
    /// (index NumPositions - 1 is the current position).
    /// Zero indicates the record is not populated.
    /// </summary>
    public byte NumPositions;

    /// <summary>
    /// The history fill-in regime (ParamsSearch.HistoryFillIn) the producer encoded planes under.
    /// This governs how consumers reconstruct history slots beyond NumPositions
    /// (fill-in true: verbatim copy of the oldest normalized position;
    ///  fill-in false: alternating perspective-reversed cascade)
    /// and the prior-state usability threshold (3 versus 2 positions).
    /// Carried per record (not per batch) so that pooled merges of batches
    /// originating from searches with different settings remain correct.
    /// </summary>
    public bool FillInHistory;

    /// <summary>
    /// The positions, oldest first; only the first NumPositions entries are meaningful.
    /// </summary>
    public MGPositionArray8 Positions;

    /// <summary>
    /// Inline array of MAX_POSITIONS MGPositions.
    /// </summary>
    [InlineArray(MAX_POSITIONS)]
    public struct MGPositionArray8
    {
      private MGPosition element0;
    }


    /// <summary>
    /// If the record has been populated.
    /// </summary>
    public readonly bool IsPopulated => NumPositions > 0;


    /// <summary>
    /// The current (most recent) position.
    /// </summary>
    public readonly MGPosition CurrentPosition => Positions[NumPositions - 1];


    /// <summary>
    /// Populates the record from a span of sequential positions (oldest first).
    /// </summary>
    public void SetFrom(ReadOnlySpan<MGPosition> sequentialPositions, bool fillInHistory)
    {
      Debug.Assert(sequentialPositions.Length >= 1 && sequentialPositions.Length <= MAX_POSITIONS);

      sequentialPositions.CopyTo(Positions);
      NumPositions = (byte)sequentialPositions.Length;
      FillInHistory = fillInHistory;
    }
  }
}
