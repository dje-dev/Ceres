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

using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Static helper methods for computing the count of 
  /// repetitions as of a specified position.
  /// </summary>
  public static class PositionRepetitionCalc
  {
    /// <summary>
    /// Calculates and sets the reptition count field 
    /// on the last position in a sequence of positions
    /// </summary>
    /// <param name="posSpan"></param>
    public static void SetFinalPositionRepetitionCount(Span<Position> posSpan)
    {
      ref readonly Position finalPosition = ref posSpan[^1];

      // Only need to check positions with same side to play
      int count = 0;
      for (int i = posSpan.Length - 3; i >= 0; i-=2)
      {
        ref readonly Position thisPosition = ref posSpan[i];
        if (thisPosition.EqualAsRepetition(in finalPosition))
          count++;
      }

      finalPosition.MiscInfo.SetRepetitionCount(count);
    }


    /// <summary>
    /// Sets the reptitions count in the MiscInfo substructure of each position within specified span,
    /// based on number of times it has already appeared in the sequence.
    /// </summary>
    /// <param name="posSpan"></param>
    [SkipLocalsInit]
    public static void SetRepetitionsCount(Span<Position> posSpan)
    {
      // Grow a sequence of positions, sorted by their hash
      Span<Position> sortedPositions = stackalloc Position[posSpan.Length];

      // Loop over each position
      for (int i = 0; i < posSpan.Length; i++)
      {
        ref readonly Position thisPos = ref posSpan[i];
        byte thisHash = thisPos.PiecesShortHash;

        // Find position of first hash entry greater than or equal on short hash
        int firstIndexHashSame = 0;
        while (firstIndexHashSame < i && sortedPositions[firstIndexHashSame].PiecesShortHash < thisHash)
          firstIndexHashSame++;

        // Count number of matches by looping over all values with the same hash and checking them for equality
        int equalCount = 0;
        int firstMatchIndex = firstIndexHashSame;
        for (int k = firstMatchIndex; sortedPositions[k].PiecesShortHash == thisHash && k < i; k++)
        {
          if (thisPos.EqualAsRepetition(in sortedPositions[k]))
            equalCount++;
        }

        // Save back the computed repetition count
        thisPos.MiscInfo.SetRepetitionCount(equalCount);

        // If necessary, shift up existing positions to make room for this one
        if (firstIndexHashSame < i)
        {
          for (int l = i; l > firstIndexHashSame; l--)
            sortedPositions[l] = sortedPositions[l - 1];
        }

        // Add this position to the sequence
        sortedPositions[firstIndexHashSame] = thisPos;
      }
    }


    /// <summary>
    /// Returns if a draw by repetition would be claimable by opponent after
    /// we make a specified move from a specified positions
    /// (claimable either immediately or after opponent makes one of the possible moves in response).
    /// </summary>
    /// <param name="ourPos"></param>
    /// <param name="ourMove"></param>
    /// <param name="historyPositions"></param>
    /// <returns></returns>
    public static bool DrawByRepetitionWouldBeClaimable(in Position ourPos, MGMove ourMove, IList<Position> historyPositions)
    {
      // Determine position after our move.
      MGPosition mgPos = ourPos.ToMGPosition;
      mgPos.MakeMove(ourMove);
      Position newPosAfterOurMove = mgPos.ToPosition;

      // Check for draw by repetition which opponent could claim after our move.
      if (NewPosWouldResultInDrawByRepetition(in newPosAfterOurMove, historyPositions))
      {
        return true;
      }

      // Build new List of Positions which includes position after our move.
      List<Position> positionsAll = new(historyPositions);
      positionsAll.Add(newPosAfterOurMove);

      // Loop thru all possible opponent moves.
      foreach ((MGMove _, Position newPosAfterOpponentMove) in PositionsGenerator1Ply.GenPositions(newPosAfterOurMove))
      {
        // Check for draw by repetition claimable after the opponent makes the candidate move.
        bool wouldBeDrawByRepetition = NewPosWouldResultInDrawByRepetition(in newPosAfterOpponentMove, positionsAll);
        if (wouldBeDrawByRepetition)
        {
          return true;
        }
      }


      return false;
    }


    private static bool NewPosWouldResultInDrawByRepetition(in Position newPos, IEnumerable<Position> positionsAll)
    {
      int countRepetitions = 0;
      foreach (Position position in positionsAll)
      {
        if (position.EqualAsRepetition(in newPos))
        {
          countRepetitions++;
        }
      }

      bool wouldBeDrawByRepetition = countRepetitions >= 2;
      return wouldBeDrawByRepetition;
    }



#if NOT
    /// <summary>
    /// Equivalent to SetRepetitionsCount but exploits fact that black and white positions can never be equal
    /// and splits them up for the calculations.
    /// 
    /// However runtime speed was actually slower.
    /// </summary>
    /// <param name="posSpan"></param>
    public static void SetRepetitionsCountBlackWhite(Span<Position> posSpan)
    {
      // Grow a sequence of positions, sorted by their hash
      Span<Position> sortedPositionsEven = stackalloc Position[posSpan.Length / 2 + (posSpan.Length % 2)];
      Span<Position> sortedPositionsOdd = stackalloc Position[posSpan.Length / 2];

      for (int i=0;i<posSpan.Length;i++)
      {
        if (i % 2 == 0)
          sortedPositionsEven[i % 2] = posSpan[i];
        else
          sortedPositionsOdd[i % 2] = posSpan[i];
      }

      SetRepetitionsCountFAST(sortedPositionsEven);
      SetRepetitionsCountFAST(sortedPositionsOdd);
    }
#endif

#if OLD_AND_BUGGY_ONLY_DETECTS_CONSECUTIVE_REPEITIONS

    // TODO: make a new class to encapsulate sequence of positions
    public static void SetRepetitionsCountSLOW(Span<Position> posSpan)
    {
      // Set repetitions for each position
      // (skip first two positions since they can't possibly already be repetitions)
      for (int i = 2; i < posSpan.Length; i += 1)
        SetRepetitionsCountSLOW(posSpan, i);
    }


    // TODO: make a new class to encapsulate sequence of positions
    static void SetRepetitionsCountSLOW(Span<Position> posSpan, int index)
    {
      ref Position thisPos = ref posSpan[index];

      int countRep = 0;
      for (int j = index - 2; j >= 0; j -= 2)
      {
        bool equals = thisPos.EqualAsRepetition(in posSpan[j]);
        if (equals)
        {
          countRep += 1;
          thisPos.MiscInfo.SetRepetitionCount(countRep);
        }
      }
    }
#endif

  }
}



