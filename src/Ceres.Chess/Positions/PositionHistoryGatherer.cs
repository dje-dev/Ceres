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
    /// 
    /// </summary>
    /// <param name="posSpan"></param>
    /// <returns></returns>
    public static Span<Position> DoGetHistoryPositions(PositionWithHistory priorMoves, Span<Position> posSpan,
                                                       int numPositionsFilled, int maxPositions, bool setFinalPositionRepetitionCount)
    {
      // Try to do fill in of history from moves prior to the root node of the search, if available
      if (numPositionsFilled < maxPositions && priorMoves != null && priorMoves.Moves.Count > 0)
      {
        VerifyBeginningOfHistoryOverlapsWithPriorMoves(priorMoves, posSpan, numPositionsFilled);

        Position[] priorPositions = priorMoves.GetPositions();
        int numTaken = 0;
        while (numPositionsFilled < maxPositions && numTaken < priorPositions.Length - 1)
        {
          posSpan[numPositionsFilled++] = priorPositions[priorPositions.Length - 2 - numTaken];
          numTaken++;
        }
      }

      if (numPositionsFilled < maxPositions)
      {
        // Do final fill in of implied prior position if the first position was en-passant 
        // and we have room for another position before this one insert the en-passant prior position
        if (numPositionsFilled == 0)
        {
          if (priorMoves.FinalPosition.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
          {
            posSpan[numPositionsFilled++] = priorMoves.FinalPosition.PosWithEnPassantUndone();
          }
        }
        else
        {
          // Check end of already existing positions
          bool hasEnPassantRights = posSpan[numPositionsFilled - 1].MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
          if (hasEnPassantRights)
          {
            Position priorPos = posSpan[numPositionsFilled - 1];
            posSpan[numPositionsFilled++] = priorPos.PosWithEnPassantUndone();
          }
        }
      }

      if (numPositionsFilled != posSpan.Length) posSpan = posSpan.Slice(0, numPositionsFilled);

      posSpan.Reverse();

      if (setFinalPositionRepetitionCount) PositionRepetitionCalc.SetFinalPositionRepetitionCount(posSpan);

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
