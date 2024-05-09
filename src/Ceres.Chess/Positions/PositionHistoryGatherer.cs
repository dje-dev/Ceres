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

#endregion

namespace Ceres.Chess.Positions
{
  /// <summary>
  /// Static helper methods related to collecting a history of positions 
  /// from sequence of positions (including en passant fill-in).
  /// </summary>
  public static class PositionHistoryGatherer
  {
    /// <summary>
    /// Returns sequence of positions, with the last move at the last position in the Span.
    /// </summary>
    /// <param name="depthOfLastNodeAdded"></param>
    /// <param name="priorMoves"></param>
    /// <param name="posSpan"></param>
    /// <param name="numPositionsFilled"></param>
    /// <param name="maxPositions"></param>
    /// <param name="doEnPassantPrefill"></param>
    /// <param name="setRepetitionCounts"></param>
    /// <returns></returns>
    public static Span<Position> DoGetHistoryPositions(int depthOfLastNodeAdded, 
                                                       PositionWithHistory priorMoves, Span<Position> posSpan,
                                                       int numPositionsFilled, int maxPositions, 
                                                       bool doEnPassantPrefill,
                                                       bool setRepetitionCounts)
    {
      Debug.Assert(priorMoves != null);

      // Try to do fill in of history from moves prior to the root node of the search, if available
      if (depthOfLastNodeAdded == 0)
      {
        if (priorMoves.Moves != null && priorMoves.Moves.Count > 0)
        {
          VerifyBeginningOfHistoryOverlapsWithPriorMoves(priorMoves, posSpan, numPositionsFilled);
        }

        Position[] priorPositions = priorMoves.GetPositions();
        int numTaken = 0;
        int lastPriorPositionIndex;
        if (numPositionsFilled == 0)
        {
          lastPriorPositionIndex = priorPositions.Length - 1;
        }
        else
        {
          // do not take last position (^1), this is root (already in tree)
          lastPriorPositionIndex = priorPositions.Length - 2; 
        }
        while (numPositionsFilled < maxPositions && numTaken <= lastPriorPositionIndex)
        {
          posSpan[numPositionsFilled++] = priorPositions[lastPriorPositionIndex - numTaken];
          numTaken++;
        }
      }

      // Do final fill in of implied prior position if the first position was en-passant 
      // and we have room for another position before this one insert the en-passant prior position
      if (doEnPassantPrefill && numPositionsFilled < maxPositions && posSpan[numPositionsFilled - 1].MiscInfo.EnPassantRightsPresent)
      {
        posSpan[numPositionsFilled] = posSpan[numPositionsFilled - 1].PosWithEnPassantUndone();
        numPositionsFilled++;
      }

      // Trim span to actual size
      if (numPositionsFilled != posSpan.Length)
      {
        posSpan = posSpan.Slice(0, numPositionsFilled);
      }

      // Reverse span
      // TODO: could this inefficiency be avoided?
      posSpan.Reverse();

      // Set repetition counts if requested.
      if (setRepetitionCounts)
      {
        PositionRepetitionCalc.SetRepetitionsCount(posSpan);
      }

      return posSpan;
    }
    

    [Conditional("DEBUG")]
    private static void VerifyBeginningOfHistoryOverlapsWithPriorMoves(PositionWithHistory priorMoves, Span<Position> posSpan, int count)
    {
      if (count > 0)
      {
        // We expect overlap of one move between the prior moves and priorPositionEnumerator
        // (but we ignore Move 50 and repetitions)
        // TODO: consider if we can/should also expect alignment of these as well
        Debug.Assert(priorMoves.FinalPosition.CalcZobristHash(PositionMiscInfo.HashMove50Mode.Ignore, false)
                  == posSpan[count - 1].CalcZobristHash(PositionMiscInfo.HashMove50Mode.Ignore, false));
      }
    }
  }


}
