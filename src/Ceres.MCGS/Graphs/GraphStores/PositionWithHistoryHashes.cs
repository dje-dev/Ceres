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
using System.Collections.Generic;
using System.Diagnostics;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;


#endregion

namespace Ceres.MCGS.Graphs.GraphStores;

/// <summary>
/// Contains a set of hashes of various types (standalone, running, finalized)
/// corresponding to all positions which appeared in a given PositionWithHistory.
/// </summary>
public readonly record struct PositionWithHistoryHashes
{
  /// <summary>
  /// The sequence of positions for which hashes were computed.
  /// </summary>
  public readonly PositionWithHistory PositionHistory;

  /// <summary>
  /// The initial running hash that existed at the end
  /// of any sequence preceding this PositionWithHistory (if any).
  /// </summary>
  public readonly PosHash96MultisetRunning IntialRunningHash;

  /// <summary>
  /// Sequence of positions appearing in game prior to root.
  /// The repetition flag is set properly.
  /// </summary>
  public readonly MGPosition[] PriorPositionsMG;

  /// <summary>
  /// Sequence of standalone hashes corresponding to PriorPositionsMg.
  /// </summary>
  public readonly PosHash64[] PriorPositionsHashes64;

  /// <summary>
  /// Sequence of history-aware hashes corresponding to PriorPositionsMG.
  /// TODO: this is possibly not used/needed.
  /// </summary>
  public readonly PosHash96MultisetRunning[] PriorPositionsHashesRunning;

  /// <summary>
  /// Sequence of history-aware hashes corresponding to PriorPositionsMG.
  /// TODO: this is possibly not used/needed.
  /// </summary>
  public readonly PosHash96MultisetFinalized[] PriorPositionsHashesRunningFinalized;

  /// <summary>
  /// If the move made after the position was irreversible.
  /// </summary>
  public readonly bool[] MoveAfterPositionWasIrreversible;

  /// <summary>
  /// The running hash at the end of the position history.
  /// </summary>
  public readonly PosHash96MultisetRunning RunningHashAtEnd;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="positionWithHistory"></param>
  /// <param name="initialRunningHash"></param>
  public PositionWithHistoryHashes(PositionWithHistory positionWithHistory, PosHash96MultisetRunning initialRunningHash)
  {
    Debug.Assert(positionWithHistory.Moves.Count == positionWithHistory.Positions.Length - 1);

    PositionHistory = positionWithHistory;
    IntialRunningHash = initialRunningHash;

    // Initialize arrays to hold precomputed hashes (of various types) for each history position.
    PriorPositionsMG = new MGPosition[positionWithHistory.Count];
    PriorPositionsHashes64 = new PosHash64[positionWithHistory.Count];
    PriorPositionsHashesRunning = new PosHash96MultisetRunning[positionWithHistory.Count];
    PriorPositionsHashesRunningFinalized = new PosHash96MultisetFinalized[positionWithHistory.Count];
    MoveAfterPositionWasIrreversible = new bool[positionWithHistory.Count];

    HashSet<PosHash64> seenHashes = new(positionWithHistory.Count);
    PosHash96MultisetRunning runningHash = initialRunningHash;

    MGPosition priorPosition = default;
    for (int i = 0; i < positionWithHistory.Count; i++)
    {
      MGPosition thisPosition = positionWithHistory.Positions[i].ToMGPosition;
      PriorPositionsMG[i] = thisPosition;

      PosHash64 thisStandaloneHash64 = MGPositionHashing.Hash64(in thisPosition);
      PosHash96 thisStandaloneHash96 = MGPositionHashing.Hash96(in thisPosition);

      PriorPositionsHashes64[i] = thisStandaloneHash64;

      // Save the finalized and hash for this position.
      PriorPositionsHashesRunningFinalized[i] = runningHash.Finalized(thisStandaloneHash96);

      // Update and save the running hash.
      runningHash.Add(thisStandaloneHash96);
      PriorPositionsHashesRunning[i] = runningHash;

      // Test for repetition to set on the position.
      bool alreadySeen = !seenHashes.Add(thisStandaloneHash64);
      if (alreadySeen)
      {
        PriorPositionsMG[i].RepetitionCount = 1;
      }

      // Determine irreversibility status.
      if (i < positionWithHistory.Moves.Count)
      {
        MGMove moveAfterThisPosition = positionWithHistory.Moves[i];
        Debug.Assert(moveAfterThisPosition != default);
        bool moveWasIrreversible = priorPosition.IsIrreversibleMove(moveAfterThisPosition, thisPosition);

        MoveAfterPositionWasIrreversible[i] = moveWasIrreversible;
        if (moveWasIrreversible)
        {
          // Start a fresh hash since last move was irreversible
          runningHash = default;
        }
      }

      priorPosition = thisPosition;
    }

    RunningHashAtEnd = runningHash;
  }


  /// <summary>
  /// Dumps the contents of this PositionWithHistoryHashes to the console.
  /// </summary>
  public readonly void Dump()
  {
    Console.WriteLine("PositionWithHistoryHashes");
    Console.WriteLine(" PositionHistory " + PositionHistory);
    Console.WriteLine(" IntialRunningHash " + IntialRunningHash);
    for (int i = 0; i < PositionHistory.Count; i++)
    {
      Console.Write($" Pos {i}: {PriorPositionsMG[i].ToPosition.FEN} ");
      Console.Write($"  Hash64: {PriorPositionsHashes64[i].Hash%10_000} ");
      Console.Write($"  Hash96Running: {PriorPositionsHashesRunning[i].ShortStr()}");
      Console.Write($"  Hash96Finalized: {PriorPositionsHashesRunningFinalized[i].ShortStr()}");
      Console.WriteLine($"  Irrev: {MoveAfterPositionWasIrreversible[i]} ");
    }
    Console.WriteLine(" RunningHashAtEnd: " + RunningHashAtEnd.ShortStr());
  }
}
