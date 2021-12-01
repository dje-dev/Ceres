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
using System.IO;
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
    /// List of players in the tournament.
    /// </summary>
    public List<PlayerStat> Players { get; set; } = new List<PlayerStat>();

    /// <summary>
    /// List of detailed information about each tournament game.
    /// </summary>
    public List<TournamentGameInfo> GameInfos = new List<TournamentGameInfo>();

    /// <summary>
    /// Dump full tournament summary to console.
    /// </summary>
    public void DumpTournamentSummary(TextWriter writer, string referenceId)
    {
      //parameter for how many percent of items above and below median should be included in the average median calculation.
      //Navs summary normally use 20% av items above and below median value.
      double medianRangePercent = 0.20; 
      CalculateMedianNodes(medianRangePercent);
      writer.WriteLine();
      writer.WriteLine("Tournament summary:");
      DumpEngineTournamentSummary(writer, referenceId);
      writer.WriteLine("Tournament round robin score table (W-D-L):");
      DumpRoundRobinResultTable(writer, referenceId);
      writer.WriteLine();
      writer.WriteLine("Tournament round robin Elo table (W-D-L):");
      DumpRoundRobinEloTable(writer, referenceId);
      writer.WriteLine();
    }

    /// <summary>
    /// Dumps full engine summary table to console.
    /// </summary>
    void DumpEngineTournamentSummary(TextWriter writer, string referenceId)
    {
      int maxWidth = 151;
      PrintLine(writer, maxWidth);
      List<(string, int)> header = new List<(string, int)>
        { ("Player",25), ("Elo", 8), ("+/-",5), ("CFS(%)", 8), ("Points",8),
          ("Played", 8), ("W-D-L", 13), ("D(%)",5), ("Time",12), ("Nodes",18), ("NPS-avg", 14), ("NPS-median", 14) };
      PrintHeaderRow(writer, header, maxWidth);
      PrintLine(writer, maxWidth);
      foreach (PlayerStat engine in Players)
      {
        WriteEngineSummary(writer, engine, maxWidth, referenceId);
      }
      PrintLine(writer, maxWidth);
      writer.WriteLine();
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="writer"></param>
    /// <param name="player"></param>
    /// <param name="width"></param>
    /// <param name="referenceId"></param>
    void WriteEngineSummary(TextWriter writer, PlayerStat player, int width, string referenceId)
    {
      string playerInfo = player.Name == referenceId ? player.Name + "*" : player.Name;
      double score = player.PlayerWins + (player.Draws / 2.0);
      string wdl = $"+{player.PlayerWins}={player.Draws}-{player.PlayerLosses}";
      float cfs = EloCalculator.LikelihoodSuperiority(player.PlayerWins, player.Draws, player.PlayerLosses);
      var (_, avg, max) = EloCalculator.EloConfidenceInterval(player.PlayerWins, player.Draws, player.PlayerLosses);
      string error = $"{(max - avg):F0}";
      double draws = (player.Draws / (double)player.NumGames) * 100;
      long nodes = player.PlayerTotalNodes;
      float time = player.PlayerTotalTime;

      List<(string, int)> rowItems = new()
      {
        (playerInfo, 25),
        (player.Name == referenceId ? "0.0" : avg.ToString("F0"), 8),
        (player.Name == referenceId ? "---" : error, 5),
        (player.Name == referenceId ? "----" : cfs.ToString("P0"), 8),
        (score.ToString("F1"), 8),
        (player.NumGames.ToString(), 8),
        (wdl, 13),
        (draws.ToString("N0"), 5),
        { (time.ToString("F2"), 12) },
        { (nodes.ToString("N0") + " ", 18) },
        { ((nodes / time).ToString("N0") + " ", 14) },
        { ((player.MedianNPSAverage).ToString("N0") + " ", 14) }

      };
      PrintEngineRow(writer, rowItems, width);
    }


    /// <summary>
    /// Dump round robin score table to console.
    /// </summary>
    /// <param name="referenceId"></param>
    /// <exception cref="Exception"></exception>
    public void DumpRoundRobinResultTable(TextWriter writer, string referenceId)
    {
      //calculate total width for table based on number of players in the tournament.
      int totalWidth = 20 * Players.Count;
      DumpHeadingTable(writer, totalWidth);
      if (string.IsNullOrEmpty(referenceId))
      {
        for (int i = 0; i < Players.Count; i++)
        {
          IEnumerable<string> row = CreateRoundRobinRow(i);
          PrintCenterAlignedRow(writer, row, totalWidth);
        }
      }
      else
      {
        PlayerStat id = Players.FirstOrDefault(e => e.Name == referenceId);
        if (id == null)
        {
          throw new Exception("Reference engine not found in Result table");
        }
        int index = Players.IndexOf(id);
        IEnumerable<string> row = CreateRoundRobinRow(index);
        PrintCenterAlignedRow(writer, row, totalWidth);
      }
      PrintLine(writer, totalWidth);
    }

    /// <summary>
    /// Dump round robin Elo table to console.
    /// </summary>
    /// <param name="referenceId"></param>
    /// <exception cref="Exception"></exception>
    public void DumpRoundRobinEloTable(TextWriter writer, string referenceId)
    {
      //calculate total width for table based on number of players in the tournament.
      int totalWidth = 25 * Players.Count;
      DumpHeadingTable(writer, totalWidth);
      if (string.IsNullOrEmpty(referenceId))
      {
        for (int i = 0; i < Players.Count; i++)
        {
          IEnumerable<string> row = CreateRoundRobinEloStats(i);
          PrintCenterAlignedRow(writer, row, totalWidth);
        }
      }
      else
      {
        PlayerStat id = Players.FirstOrDefault(e => e.Name == referenceId);
        if (id == null)
        {
          throw new Exception("Reference engine not found in Result table");
        }
        int index = Players.IndexOf(id);
        IEnumerable<string> row = CreateRoundRobinEloStats(index);
        PrintCenterAlignedRow(writer, row, totalWidth);
      }
      PrintLine(writer, totalWidth);
    }

    /// <summary>
    /// Get PlayerStat for a player with a specific opponent.
    /// </summary>
    /// <param name="playerToFind"></param>
    /// <param name="opponent"></param>
    /// <returns></returns>

    public PlayerStat GetPlayer(string playerToFind, string opponent)
    {
      PlayerStat player = Players.FirstOrDefault(e => e.Name == playerToFind);

      if (player == null)
      {
        player = new PlayerStat() { Name = playerToFind };
        player.Opponents.Add(opponent, (0, 0, 0));
        Players.Add(player);
        return player;
      }
      if (!player.Opponents.ContainsKey(opponent))
      {
        player.Opponents.Add(opponent, (0, 0, 0));
      }
      return player;
    }

    /// <summary>
    /// Update tournament stat for both players in a game.
    /// </summary>
    /// <param name="thisResult"></param>
    /// <param name="engine"></param>
    public void UpdateTournamentStats(TournamentGameInfo thisResult, GameEngine engine)
    {
      PlayerStat playerWhite;
      PlayerStat playerBlack;
      TournamentGameResult whiteResult;
      TournamentGameResult blackResult;

      TournamentGameResult reverseResult =
          thisResult.Result == TournamentGameResult.Win ? TournamentGameResult.Loss :
          thisResult.Result == TournamentGameResult.Loss ? TournamentGameResult.Win :
          TournamentGameResult.Draw;

      if (thisResult.Engine2IsWhite)
      {
        playerWhite = GetPlayer(engine.OpponentEngine.ID, engine.ID);
        playerBlack = GetPlayer(engine.ID, engine.OpponentEngine.ID);
        whiteResult = reverseResult;
        blackResult = thisResult.Result;
      }
      else
      {
        playerWhite = GetPlayer(engine.ID, engine.OpponentEngine.ID);
        playerBlack = GetPlayer(engine.OpponentEngine.ID, engine.ID);
        whiteResult = thisResult.Result;
        blackResult = reverseResult;
      }

      playerWhite.UpdateGameOutcome(whiteResult, playerBlack.Name);
      playerBlack.UpdateGameOutcome(blackResult, playerWhite.Name);
      UpdateNodeCounterAndTimeUse(thisResult, playerWhite, playerBlack);
    }


    /// <summary>
    /// Update total number of nodes computed and time used in the game.
    /// </summary>
    /// <param name="thisResult"></param>
    /// <param name="playerWhite"></param>
    /// <param name="playerBlack"></param>
    void UpdateNodeCounterAndTimeUse(TournamentGameInfo thisResult, PlayerStat playerWhite, PlayerStat playerBlack)
    {
      if (thisResult.Engine2IsWhite)
      {
        playerWhite.PlayerTotalNodes += thisResult.TotalNodesEngine2;
        playerBlack.PlayerTotalNodes += thisResult.TotalNodesEngine1;
        playerWhite.PlayerTotalTime += thisResult.TotalTimeEngine2;
        playerBlack.PlayerTotalTime += thisResult.TotalTimeEngine1;
      }
      else
      {
        playerWhite.PlayerTotalNodes += thisResult.TotalNodesEngine1;
        playerBlack.PlayerTotalNodes += thisResult.TotalNodesEngine2;
        playerWhite.PlayerTotalTime += thisResult.TotalTimeEngine1;
        playerBlack.PlayerTotalTime += thisResult.TotalTimeEngine2;
      }     
    }

    /// <summary>
    /// Dump dashes to console for a certain width.
    /// </summary>
    /// <param name="width"></param>
    void PrintLine(TextWriter writer, int width)
    {
      writer.WriteLine(new string('-', width));
    }

    /// <summary>
    /// Dump center aligned header with variable width to console.
    /// </summary>
    /// <param name="columns"></param>
    /// <param name="maxWidth"></param>

    void PrintHeaderRow(TextWriter writer, IEnumerable<(string, int)> columns, int maxWidth)
    {
      string row = "|";

      foreach ((string txt, int width) in columns)
      {
        row += AlignCentre(txt, width) + "|";
      }

      writer.WriteLine(row);
    }

    /// <summary>
    /// Dump center aligned text with fixed width to console. 
    /// </summary>
    /// <param name="columns"></param>
    /// <param name="maxWidth"></param>
    void PrintCenterAlignedRow(TextWriter writer, IEnumerable<string> columns, int maxWidth)
    {
      int Columnwidth = (maxWidth - columns.Count()) / columns.Count();
      string row = "|";

      foreach (string column in columns)
      {
        row += AlignCentre(column, Columnwidth) + "|";
      }

      writer.WriteLine(row);
    }

    /// <summary>
    /// Dump player text with variable column width to console.
    /// </summary>
    /// <param name="columns"></param>
    /// <param name="maxWidth"></param>

    void PrintEngineRow(TextWriter writer, List<(string, int)> columns, int maxWidth)
    {
      int numberOfColumns = columns.Count();
      string row = "|";

      for (int i = 0; i < numberOfColumns; i++)
      {
        var (txt, width) = columns[i];
        if (i > numberOfColumns - 4)
          row += AlignRight(txt, width) + "|";
        else
          row += AlignCentre(txt, width) + "|";
      }

      writer.WriteLine(row);
    }

    /// <summary>
    /// Center align text in a column for a certain width.
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
    /// Right align text in a column for a certain width.
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

    IEnumerable<string> CreateRoundRobinRow(int row)
    {
      const string empty = "------";
      int counter = 0;
      PlayerStat stat = Players[row];
      yield return stat.Name;
      foreach (KeyValuePair<string, (int, int, int)> opponent in stat.Opponents)
      {
        if (row == counter)
        {
          yield return empty;
        }

        var (win, draw, loss) = opponent.Value;
        yield return $"+{win}={draw}-{loss}";
        counter++;
      }

      if (row + 1 == Players.Count)
      {
        yield return empty;
      }
    }


    /// <summary>
    /// Create Round Robin Elo stat for player based on index row in player list.
    /// </summary>
    /// <param name="row"></param>
    /// <returns></returns>

    IEnumerable<string> CreateRoundRobinEloStats(int row)
    {
      const string empty = "-------";
      int counter = 0;
      PlayerStat stat = Players[row];
      yield return stat.Name;
      foreach (KeyValuePair<string, (int, int, int)> opponent in stat.Opponents)
      {
        if (row == counter)
        {
          yield return empty;
        }

        var (win, draw, loss) = opponent.Value;
        string eloPerf = EloCalculator.EloDiff(win, draw, loss).ToString("F0");
        var (min, avg, max) = EloCalculator.EloConfidenceInterval(win, draw, loss);
        string error = (max - avg).ToString("F0");
        string msg = $"{eloPerf} +/- {error: F0}";
        yield return msg;
        counter++;
      }

      if (row + 1 == Players.Count)
      {
        yield return empty;
      }
    }

    /// <summary>
    /// Dump Round Robin table header to console.
    /// </summary>
    /// <param name="width"></param>

    void DumpHeadingTable(TextWriter writer, int width)
    {
      IEnumerable<string> players = Players.Select(e => e.Name);
      List<string> header = new List<string>();
      header.Add("Engine");
      header.AddRange(players);
      PrintLine(writer, width);
      PrintCenterAlignedRow(writer, header, width);
      PrintLine(writer, width);
    }

    /// <summary>
    /// Calculation of median node speed for each player in the tournament.
    /// </summary>
    /// <param name="rangeSelector"></param>
    public void CalculateMedianNodes(double rangeSelector)
    {
      foreach (var player in Players)
      {
        IEnumerable<float> speedTStat = ExtractNodeSpeedStat(player.Name);
        player.CalculateMedianNPS(rangeSelector, speedTStat);
      }
    }

    /// <summary>
    /// Extract a collection of nodes per second for all moves in all games played.
    /// </summary>
    /// <param name="player"></param>
    /// <returns></returns>
    IEnumerable<float> ExtractNodeSpeedStat(string player)
    {
      foreach (var info in GameInfos)
      {
        if (player == info.PlayerWhite || player == info.PlayerBlack)
        {
          foreach (var move in info.GameMoveHistory)
          {
            if (player == move.Id)
              yield return (move.NodesPerSecond);
          }
        }
      }
    }    
  }
}
