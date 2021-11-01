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
using System.Linq;
using Ceres.Chess.GameEngines;
using Chess.Ceres.PlayEvaluation;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Statistics relating to the outcome of a tournament.
  /// </summary>
  public record TournamentResultStats
  {
    /// <summary>
    /// Table to store Win-Draw-Loss statistics against each opponent
    /// </summary>
    public Dictionary<string, (int, int, int)> Opponents { get; set; } = new Dictionary<string, (int, int, int)>();

    public long Player1TotalNodes { set; get; }

    public float Player1TotalTime { set; get; }

    public long Player2TotalNodes { set; get; }

    public float Player2TotalTime { set; get; }

    /// <summary>
    /// Name of player 1.
    /// </summary>
    public string Player1 { init; get; }

    /// <summary>
    /// Name of player 2.
    /// </summary>
    public string Player2 { init; get; }

    /// <summary>
    /// Number of wins by player 1.
    /// </summary>
    public int Player1Wins { set; get; }

    /// <summary>
    /// Number of draws.
    /// </summary>
    public int Draws { set; get; }

    /// <summary>
    /// Number of losses by player 1.
    /// </summary>
    public int Player1Losses { set; get; }

    /// <summary>
    /// Short string summarizing games outcome.
    /// </summary>
    public string GameOutcomesString { set; get; }

    /// <summary>
    /// Total number of games played.
    /// </summary>
    public int NumGames => Player1Wins + Draws + Player1Losses;

    /// <summary>
    /// List of TournamentResultStats for every player
    /// </summary>
    public List<TournamentResultStats> Results { get; set; }

    public TournamentResultStats()
    {
      Results = new List<TournamentResultStats>();
    }

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="player1"></param>
    /// <param name="player2"></param>
    public TournamentResultStats(string player1, string player2)
    {
      Player1 = player1;
      Player2 = player2;
    }

    /// <summary>
    /// Dump tournament summary to console
    /// </summary>
    public void DumpTournamentSummary(string referenceId)
    {
      Console.WriteLine("Tournament summary:");
      DumpEngineTournamentSummary(referenceId);
      Console.WriteLine();
      DumpRoundRobinResultTable(referenceId);
      Console.WriteLine();
    }

    /// <summary>
    /// Dumps full engine summary table to Console
    /// </summary>
    void DumpEngineTournamentSummary(string referenceId)
    {
      var width = 180;
      PrintLine(width);
      var header = new List<string>() { "Player", "Rating", "Error +/-", "CFS (%)", "Points", "Played", "W-D-L", "D(%)", "Time", "Nodes" };
      PrintCenterAlignedRow(header, width);
      PrintLine(width);
      foreach (var engine in Results)
      {
        WriteEngineSummary(engine, width, referenceId);
      }
      PrintLine(width);
      Console.WriteLine();
    }

    /// <summary>
    /// Write summary row for player1
    /// </summary>
    /// <param name="engineStat"></param>
    /// <param name="width"></param>
    void WriteEngineSummary(TournamentResultStats engineStat, int width, string referenceId)
    {
      var playerInfo = engineStat.Player1 == referenceId ? engineStat.Player1 + "*" : engineStat.Player1;
      var score = engineStat.Player1Wins + (engineStat.Draws / 2.0);
      //var numberOfGames = engineStat.Player1Losses + engineStat.Player1Wins + engineStat.Draws;
      var wdl = $"{engineStat.Player1Wins}-{engineStat.Draws}-{engineStat.Player1Losses}";
      var cfs = EloCalculator.LikelihoodSuperiority(engineStat.Player1Wins, engineStat.Draws, engineStat.Player1Losses);
      var (min, avg, max) = EloCalculator.EloConfidenceInterval(engineStat.Player1Wins, engineStat.Draws, engineStat.Player1Losses);
      var error = $"{(max - avg):F0}";
      var draws = engineStat.Draws / (double)engineStat.NumGames;
      var nodes = engineStat.Player1TotalNodes;
      var time = engineStat.Player1TotalTime;
      var rowItems = new List<string>()
                { playerInfo, avg.ToString("F0"), error, cfs.ToString("P0"), score.ToString("F1"),
                engineStat.NumGames.ToString(), wdl, draws.ToString("P2"), {time.ToString("F2")}, {nodes.ToString("N0")} };
      PrintEngineRow(rowItems, width);
    }

    public void DumpRoundRobinResultTable(string referenceId)
    {
      //decide total width for table
      int totalWidth = 20 * Results.Count;
      Console.WriteLine("Tournament round robin table (W-D-L):");
      DumpHeadingTable(totalWidth);
      if (string.IsNullOrEmpty(referenceId))
      {
        for (int i = 0; i < Results.Count; i++)
        {
          var row = CreateRoundRobinRow(i);
          PrintCenterAlignedRow(row, totalWidth);
        }
      }
      else
      {
        var id = Results.FirstOrDefault(e => e.Player1 == referenceId);
        if (id == null)
        {
          throw new Exception("Reference engine not found in Result table");
        }
        var index = Results.IndexOf(id);
        var row = CreateRoundRobinRow(index);
        PrintCenterAlignedRow(row, totalWidth);
      }
      PrintLine(totalWidth);
    }

    /// <summary>
    /// Updates tournament statistics based on a game with specified result.
    /// </summary>
    /// <param name="thisResult"></param>
    /// <param name="opponent"></param>
    public void UpdateGameOutcome(TournamentGameInfo thisResult, string opponent)
    {
      var (win, draw, loss) = Opponents[opponent];
      switch (thisResult.Result)
      {
        case TournamentGameResult.Win:
          Player1Wins++;
          Opponents[opponent] = (win + 1, draw, loss);
          GameOutcomesString += "+";
          break;

        case TournamentGameResult.Loss:
          Player1Losses++;
          Opponents[opponent] = (win, draw, loss + 1);
          GameOutcomesString += "-";
          break;

        default:
          Draws++;
          Opponents[opponent] = (win, draw + 1, loss);
          GameOutcomesString += "=";
          break;
      }
    }

    /// <summary>
    /// Get TournamentResultStats for player1 and set opponent as player 2
    /// </summary>
    /// <param name="player1"></param>
    /// <param name="player2"></param>
    /// <returns></returns>
    public TournamentResultStats GetResultsForPlayer(string player1, string player2)
    {
      var whitePlayer = Results.FirstOrDefault(e => e.Player1 == player1);

      if (whitePlayer == null)
      {
        var entry = new TournamentResultStats(player1, player2);
        if (!entry.Opponents.ContainsKey(player2))
        {
          entry.Opponents.Add(player2, (0, 0, 0));
        }
        Results.Add(entry);
        return entry;
      }
      if (!whitePlayer.Opponents.ContainsKey(player2))
      {
        whitePlayer.Opponents.Add(player2, (0, 0, 0));
      }
      return whitePlayer;
    }

    /// <summary>
    /// Update tournament stat for player1 and opponent
    /// </summary>
    /// <param name="thisResult"></param>
    /// <param name="engine"></param>

    public void UpdateTournamentStats(TournamentGameInfo thisResult, GameEngine engine)
    {
      TournamentResultStats player1;
      player1 = GetResultsForPlayer(engine.ID, engine.OpponentEngine.ID);
      player1.UpdateGameOutcome(thisResult, engine.OpponentEngine.ID);
      var player2 = GetResultsForPlayer(engine.OpponentEngine.ID, engine.ID);
      var gameResultBlack =
          thisResult.Result == TournamentGameResult.Win ? TournamentGameResult.Loss :
          thisResult.Result == TournamentGameResult.Loss ? TournamentGameResult.Win :
          TournamentGameResult.Draw;

      var reverseResult = thisResult with { Result = gameResultBlack };
      player2.UpdateGameOutcome(reverseResult, engine.ID);
      player1.UpdateNodeCounterAndTimeUse(thisResult, player1, player2);
    }

    void UpdateNodeCounterAndTimeUse(TournamentGameInfo thisResult, TournamentResultStats player1, TournamentResultStats player2)
    {
      player1.Player1TotalNodes += thisResult.TotalNodesEngine1;
      player2.Player1TotalNodes += thisResult.TotalNodesEngine2;
      player1.Player1TotalTime += thisResult.TotalTimeEngine1;
      player2.Player1TotalTime += thisResult.TotalTimeEngine2;
    }

    /// <summary>
    /// Dumps Round Robin summary to Console
    /// </summary>


    /// <summary>
    /// Dump dashes to console for a certain width
    /// </summary>
    /// <param name="width"></param>
    void PrintLine(int width)
    {
      Console.WriteLine(new string('-', width));
    }

    /// <summary>
    /// Dump center aligned text row to console
    /// </summary>
    /// <param name="columns"></param>
    /// <param name="maxWidth"></param>

    void PrintCenterAlignedRow(IEnumerable<string> columns, int maxWidth)
    {
      int Columnwidth = (maxWidth - columns.Count()) / columns.Count();
      string row = "|";

      foreach (string column in columns)
      {
        row += AlignCentre(column, Columnwidth) + "|";
      }

      Console.WriteLine(row);
    }

    void PrintEngineRow(List<string> columns, int maxWidth)
    {
      var numberOfColumns = columns.Count();
      int Columnwidth = (maxWidth - columns.Count()) / columns.Count();
      string row = "|";

      for (int i = 0; i < numberOfColumns; i++)
      {
        if (i == numberOfColumns - 1)
          row += AlignRight(columns[i], Columnwidth) + "|";
        else
          row += AlignCentre(columns[i], Columnwidth) + "|";
      }

      Console.WriteLine(row);
    }

    /// <summary>
    /// Center align text in a column for a certain width
    /// </summary>
    /// <param name="text"></param>
    /// <param name="width"></param>
    /// <returns></returns>

    string AlignCentre(string text, int width)
    {
      text = text.Length > width ? text.Substring(0, width - 3) + "..." : text;

      if (string.IsNullOrEmpty(text))
      {
        return new string(' ', width);
      }
      else
      {
        return text.PadRight(width - (width - text.Length) / 2).PadLeft(width);
      }
    }

    /// <summary>
    /// Right align text in a column for a certain width
    /// </summary>
    /// <param name="text"></param>
    /// <param name="width"></param>
    /// <returns></returns>

    string AlignRight(string text, int width)
    {
      text = text.Length > width ? text.Substring(0, width - 3) + "..." : text;

      if (string.IsNullOrEmpty(text))
      {
        return new string(' ', width);
      }
      else
      {
        return text.PadLeft(width);
      }
    }

    /// <summary>
    /// Create Round Robin row for result table
    /// </summary>
    /// <param name="row"></param>
    /// <returns></returns>
    IEnumerable<string> CreateRoundRobinRow(int row)
    {
      const string empty = "-----";
      int counter = 0;
      var stat = Results[row];
      yield return stat.Player1;
      foreach (var opponent in stat.Opponents)
      {
        if (row == counter)
        {
          yield return empty;
        }

        var (win, draw, loss) = opponent.Value;
        yield return $"{win}-{draw}-{loss}";
        counter++;
      }

      if (row + 1 == Results.Count)
      {
        yield return empty;
      }
    }

    /// <summary>
    /// Dump Round Robin result header
    /// </summary>
    /// <param name="width"></param>

    void DumpHeadingTable(int width)
    {
      var players = Results.Select(e => e.Player1);
      var header = new List<string>();
      header.Add("Engine");
      header.AddRange(players);
      PrintLine(width);
      PrintCenterAlignedRow(header, width);
      PrintLine(width);
    }


  }
}
