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
using System.Linq;
using System.Text;
using System.Collections.Generic;

using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.Games.Utils
{
  /// <summary>
  /// Static helper methods which assist with identification of 
  /// shared move sequences between two or more games.
  /// </summary>
  public static class PGNCommonMoves
  {
    /// <summary>
    /// Represents a sequence of moves which were all the same as played by a set of games (newGames)
    /// but where possibly these players deviated from some larger set of prior games (priorGames).
    /// </summary>
    public record PGNCommonMoveSequence(int id, List<PGNGame> priorGames, List<PGNGame> newGames,
                                        int startMoveNum, int numMoves, int depth)
    { }

    static int subsetID = 0; // unique ID assigned to each sequence

    /// <summary>
    /// Returns a sequence of MoveSubset each of which is a sub-sequence of identical moves 
    /// within a game being followed in sync by one or more players.
    /// </summary>
    public static IEnumerable<PGNCommonMoveSequence> PGNCommonMoveSequences(List<PGNGame> priorGames, 
                                                                            List<PGNGame> games, int startPosNum, int depth = 0)
    {
      int posNum = startPosNum;
      while (true)
      {
        Dictionary<Position, List<PGNGame>> dict = new();
        foreach (PGNGame game in games)
        {
          if (game.Moves.Moves.Count > posNum)
          {
            Position pos = game.Moves.Positions.ToArray()[posNum];
            if (!dict.TryGetValue(pos, out List<PGNGame> these))
            {
              these = new List<PGNGame>();
            }
            these.Add(game);
            dict[pos] = these;
          }
        }

        if (dict.Count == 1 && dict.First().Value.Count == games.Count)
        {
          posNum++;
        }
        else
        {
          yield return new PGNCommonMoveSequence(subsetID++, priorGames, games, startPosNum, posNum - startPosNum, depth); // xxx

          depth += 1;
          foreach (List<PGNGame> innerGames in dict.Values)
          {
            foreach (var inner in PGNCommonMoveSequences(games, innerGames, posNum, depth))
            {
              yield return inner;
            }
          }
          yield break;
        }
      }
    }

    #region Text dump 

    static string ToPlayerList(IEnumerable<PGNGame> games)
    {
      StringBuilder sb = new();
      foreach (PGNGame game in games)
      {
        sb.Append(game.WhitePlayer + " vs " + game.BlackPlayer + "  ");
      }
      return sb.ToString();
    }


    /// <summary>
    /// Dumps a summary to Console of all common subsets within specified List of games.
    /// </summary>
    /// <param name="games"></param>
    /// <param name="showFirstDeviationOnly"></param>
    public static void DumpTextOutline(List<PGNGame> games, bool showFirstDeviationOnly = false)
    {
      //  int indent = 2;
      foreach ((int id, List<PGNGame> priorGames, List<PGNGame> newGames, int startMoveNum, int numMoves, int depth) in PGNCommonMoveSequences(null, games, 0))
      {
        string indentStr = "";
        for (int i = 0; i < depth * 2; i++) indentStr += " ";

        string deviators = "";
        HashSet<string> deviatorsSet = new();
        if (priorGames != null)
        {
          deviators += " DEVIATORS: ";
          MGMove firstMoveThisSequence = newGames[0].Moves.Moves[startMoveNum];
          bool whiteDeviated = firstMoveThisSequence.BlackToMove;
          for (int i = 0; i < newGames.Count; i++)
          {
            deviatorsSet.Add(whiteDeviated ? newGames[i].WhitePlayer : newGames[i].BlackPlayer);
          }
          foreach (string deviatorInner in deviatorsSet)
          {
            deviators += deviatorInner + " ";
          }
        }

        if (!showFirstDeviationOnly || depth == 1)
        {
          Console.WriteLine($"ROUND:{newGames[0].Round} {indentStr} {depth}: Moves {startMoveNum}...{ startMoveNum + numMoves}  {deviators}");
          indentStr = indentStr + "  " + ToPlayerList(newGames);
        }
      }
    }

    #endregion
  }
}
