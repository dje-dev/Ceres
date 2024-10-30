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

namespace Ceres.Chess.Games.Utils
{
  /// <summary>
  /// Represents a single entry from a Lichess puzzle database.
  /// 
  /// Courtesy of Lichess, see: https://database.lichess.org/#puzzles.
  /// 
  /// Example row:
  ///   ID,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,2032,75,91,297,crushing hangingPiece long middlegame,https://lichess.org/787zsVup/black#48
  /// </summary>
  public readonly record struct LichessDatabaseRecord
  {
    /// <summary>
    /// Unique ID number of the puzzle.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Starting FEN.
    /// </summary>
    public readonly string FEN;

    /// <summary>
    /// Sequence of moves which followed starting FEN.
    /// First move is pre-puzzle, second move is first puzzle move.
    /// </summary>
    public readonly string[] Moves;

    /// <summary>
    /// Difficulty rating assigned to the puzzle.
    /// </summary>
    public readonly int Rating;

    /// <summary>
    /// Change in the player's puzzle rating after they attempt the puzzle.
    /// </summary>
    public readonly int RatingDiff;

    /// <summary>
    /// One or more descriptive categories for the puzzle.
    /// </summary>
    public readonly string Description;

    /// <summary>
    /// Internet link to puzzle on Lichess site.
    /// </summary>
    public readonly string URL;


    /// <summary>
    /// Consructor.
    /// </summary>
    /// <param name="lichessRowText"></param>
    public LichessDatabaseRecord(string lichessRowText)
    { 
      string[] allParts = lichessRowText.Split(",");
      ID = allParts[0];
      FEN = allParts[1];
      Moves = allParts[2].Split(" ");
      Rating = int.Parse(allParts[3]);
      RatingDiff = int.Parse(allParts[4]);
      Description = allParts[7];
      URL = allParts[8];
    }


    /// <summary>
    /// Number of puzzle moves.
    /// </summary>
    public readonly int NumPuzzleMoves => Moves.Length / 2;


    /// <summary>
    /// Returns EPD entry for a puzzle move a specified index.
    /// </summary>
    /// <param name="puzzleMoveIndex"></param>
    /// <returns></returns>
    public readonly EPDEntry EPDForPuzzleMoveAtIndex(int puzzleMoveIndex)
    {
      int numMovesPrep = 2 * puzzleMoveIndex;

      string newStartMoves = string.Join(" ", Moves[..(numMovesPrep+1)]);
      string bmMove = Moves[numMovesPrep + 1];

      EPDEntry epd = new(ID, FEN, newStartMoves, [bmMove], EPDEntry.MovesFormatEnum.UCI);

      // If the move is a checkmate, we add all checkmates from the position as valid choices.
      // Rebuild the EPDEntry to include all checkmates.
      PositionWithHistory posAfterCorrectMove = PositionWithHistory.FromFENAndMovesUCI(FEN, newStartMoves + " " + bmMove);
      if (posAfterCorrectMove.FinalPosition.CalcTerminalStatus() == GameResult.Checkmate)
      {
        List<string> checkmateMoves = GetAllCheckmateMovesFromPosition(newStartMoves);

        Debug.Assert(checkmateMoves.Contains(bmMove));
        epd = new(ID, FEN, newStartMoves, checkmateMoves.ToArray(), EPDEntry.MovesFormatEnum.UCI);
      }

      return epd;
    }


    /// <summary>
    /// Returns list of all checkmate moves from a specified position (in UCI format).
    /// </summary>
    /// <param name="newStartMoves"></param>
    /// <returns></returns>
    private readonly List<string> GetAllCheckmateMovesFromPosition(string newStartMoves)
    {
      List<string> checkmateMoves = new();
      MGMoveList moves = new MGMoveList();
      MGPosition mgPos = MGPosition.FromPosition(PositionWithHistory.FromFENAndMovesUCI(FEN, newStartMoves).FinalPosition);
      MGMoveGen.GenerateMoves(in mgPos, moves);

      for (int i = 0; i < moves.NumMovesUsed; i++)
      {
        MGPosition newPos = new MGPosition(in mgPos);
        newPos.MakeMove(moves.MovesArray[i]);
        if (newPos.ToPosition.CalcTerminalStatus() == GameResult.Checkmate)
        {
          checkmateMoves.Add(moves.MovesArray[i].MoveStr(MGMoveNotationStyle.Coordinates));
        }
      }

//      Console.WriteLine($"Found {checkmateMoves.Count} checkmates from position {newStartMoves}");
      return checkmateMoves;
    }


    static void Test()
    {
      // LichessRow basicTest = new("ID,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,2032,75,91,297,crushing hangingPiece long middlegame,https://lichess.org/787zsVup/black#48");
      LichessDatabaseRecord multiCheckmatesTest = new("ID,7k/8/8/8/8/8/4q1PP/6K1 w - - 0 1,g1h1 e2e1,2032,75,91,297,multiway mate,(no URL)");

      Console.WriteLine(multiCheckmatesTest);
      for (int i = 0; i < multiCheckmatesTest.NumPuzzleMoves; i++)
      {
        EPDEntry ez0 = multiCheckmatesTest.EPDForPuzzleMoveAtIndex(i);
      };
    }

  }

}
