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

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.LC0.Batches
{
  /// <summary>
  /// Derives compact per-position history records (MGPositionHistoryCompact) from the
  /// representations legacy producers put on an evaluation batch: either a full
  /// EncodedPositionWithHistory, or the flat LC0 plane arrays.
  ///
  /// Both derivations reconstruct the same real (non-fill) position sequence and fill regime,
  /// so a batch carrying only planes can be made to feed CompactHistories-consuming (TPG/Ceres)
  /// evaluators. The from-planes variant is the exact structural inverse of the plane encoding
  /// (EncodedPositionBatchFlat.Set), so it reproduces byte-identical downstream conversions.
  /// </summary>
  public static class CompactHistoryDerivation
  {
    /// <summary>
    /// Reconstructs the real (non-fill) history positions of one encoded position and stores
    /// them (oldest first) into a compact record, mirroring the slot-detection rule of
    /// EncodedPositionWithHistory.ToPositionWithHistory: a history slot is an actual position
    /// iff its planes are non-empty AND differ from the next-more-recent board. The record's
    /// fill regime is inferred from the trailing planes themselves: empty trailing boards mean
    /// the producer ran with history fill-in disabled, non-empty (fill copies) mean enabled;
    /// with all 8 slots real the value is irrelevant (no filled slots, and the prior-state
    /// predicate thresholds only differ below 3 positions).
    /// </summary>
    public static void DeriveFromEncodedPosition(in EncodedPositionWithHistory encoded, ref MGPositionHistoryCompact record)
    {
      const int NUM_SLOTS = EncodedPositionBatchFlat.NUM_HISTORY_POSITIONS; // 8

      // Gather real positions, oldest first (slot 0 is the current position).
      Span<MGPosition> positionsMG = stackalloc MGPosition[NUM_SLOTS];
      int numReal = 0;
      for (int i = NUM_SLOTS - 1; i >= 0; i--)
      {
        bool isActualBoardPosition = (i == 0)
                                  || (!encoded.GetPlanesForHistoryBoard(i).IsEmpty
                                     && encoded.GetPlanesForHistoryBoard(i - 1) != encoded.GetPlanesForHistoryBoard(i));
        if (isActualBoardPosition)
        {
          // HistoryPosition decodes side to move, en passant (from board diff), castling and
          // repetition count (0/1 from the repetition plane) - the same information the
          // legacy plane-based conversion path had available.
          Position pos = encoded.HistoryPosition(i);
          MGPosition posMG = MGChessPositionConverter.MGChessPositionFromPosition(in pos);
          posMG.RepetitionCount = pos.MiscInfo.RepetitionCount; // not copied by the converter
          positionsMG[numReal++] = posMG;
        }
      }

      bool fillInHistory = numReal == NUM_SLOTS
                        || !encoded.GetPlanesForHistoryBoard(numReal).IsEmpty;

      record.SetFrom(positionsMG.Slice(0, numReal), fillInHistory);
    }


    /// <summary>
    /// Derives a compact record from one row of the flat batch plane representation
    /// (PosPlaneBitmaps / PosPlaneValues) by reconstructing a scratch EncodedPositionWithHistory
    /// (structural inverse of the plane encode) and then applying DeriveFromEncodedPosition.
    /// </summary>
    /// <param name="rowBitmaps">the 112 plane bitmaps for one position</param>
    /// <param name="rowValues">the 112 plane values for one position</param>
    /// <param name="record">record to populate</param>
    public static void DeriveFromPlanes(ReadOnlySpan<ulong> rowBitmaps, ReadOnlySpan<byte> rowValues,
                                        ref MGPositionHistoryCompact record)
    {
      EncodedPositionWithHistory scratch = default;
      scratch.SetFromPlanes(rowBitmaps, rowValues);
      DeriveFromEncodedPosition(in scratch, ref record);
    }
  }
}
