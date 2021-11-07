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
    /// List of players in the tournament
    /// </summary>
    public List<PlayerStat> Players { get; set; } = new List<PlayerStat>();

    /// <summary>
    /// List of detailed information about each tournament game.
    /// </summary>
    public List<TournamentGameInfo> GameInfos = new List<TournamentGameInfo>();

    /// <summary>
    /// Dump full tournament summary to console.
    /// </summary>
    public void DumpTournamentSummary(string referenceId)
    {
      //parameter for how many percent of items above and below median should be included in the average median calculation
      double medianPercent = 0.20; //use 20% av items above and below median
      CalculateMedianNodes(medianPercent);
      Console.WriteLine();
      Console.WriteLine("Tournament summary:");
      DumpEngineTournamentSummary(referenceId);
      Console.WriteLine("Tournament round robin score table (W-D-L):");
      DumpRoundRobinResultTable(referenceId);
      Console.WriteLine();
      Console.WriteLine("Tournament round robin Elo table (W-D-L):");
      DumpRoundRobinEloTable(referenceId);
      Console.WriteLine();
    }

    /// <summary>
    /// Dumps full engine summary table to console.
    /// </summary>
    void DumpEngineTournamentSummary(string referenceId)
    {
      int maxWidth = 150;
      PrintLine(maxWidth);
      List<(string, int)> header = new List<(string, int)>
        { ("Player",25), ("Elo", 8), ("+/-",5), ("CFS(%)", 8), ("Points",8),
          ("Played", 8), ("W-D-L", 13), ("D(%)",5), ("Time",12), ("Nodes",18), ("NPS-avg", 14), ("NPS-median", 14)  };
      PrintHeaderRow(header, maxWidth);
      PrintLine(maxWidth);
      foreach (PlayerStat engine in Players)
      {
        WriteEngineSummaryBeta(engine, maxWidth, referenceId);
      }
      PrintLine(maxWidth);
      Console.WriteLine();
    }

    /// <summary>
    /// Write summary row for player.
    /// </summary>
    /// <param name="player"></param>
    /// <param name="width"></param>
    /// <param name="referenceId"></param>
    void WriteEngineSummaryBeta(PlayerStat player, int width, string referenceId)
    {
      string playerInfo = player.Name == referenceId ? player.Name + "*" : player.Name;
      double score = player.PlayerWins + (player.Draws / 2.0);
      string wdl = $"{player.PlayerWins}-{player.Draws}-{player.PlayerLosses}";
      float cfs = EloCalculator.LikelihoodSuperiority(player.PlayerWins, player.Draws, player.PlayerLosses);
      var (_, avg, max) = EloCalculator.EloConfidenceInterval(player.PlayerWins, player.Draws, player.PlayerLosses);
      string error = $"{(max - avg):F0}";
      double draws = (player.Draws / (double)player.NumGames) * 100;
      long nodes = player.PlayerTotalNodes;
      float time = player.PlayerTotalTime;
      List<(string, int)> rowItems = new()
      {
        (playerInfo, 25),
        (avg.ToString("F0"), 8),
        (error, 5),
        (cfs.ToString("P0"), 8),
        (score.ToString("F1"), 8),
        (player.NumGames.ToString(), 8),
        (wdl, 13),
        (draws.ToString("N0"), 5),
        { (time.ToString("F2"), 12) },
        { (nodes.ToString("N0") + " ", 18) },
        { ((nodes / time).ToString("N0") + " ", 14) },
        { ((player.MedianNodeValue).ToString("N0") + " ", 14) }

      };
      PrintEngineRow(rowItems, width);
    }

    /// <summary>
    /// Dump round robin score table to console.
    /// </summary>
    /// <param name="referenceId"></param>
    /// <exception cref="Exception"></exception>
    public void DumpRoundRobinResultTable(string referenceId)
    {
      //decide total width for table
      int totalWidth = 20 * Players.Count;
      DumpHeadingTable(totalWidth);
      if (string.IsNullOrEmpty(referenceId))
      {
        for (int i = 0; i < Players.Count; i++)
        {
          IEnumerable<string> row = CreateRoundRobinRow(i);
          PrintCenterAlignedRow(row, totalWidth);
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
        PrintCenterAlignedRow(row, totalWidth);
      }
      PrintLine(totalWidth);
    }

    /// <summary>
    /// Dump round robin Elo table to console.
    /// </summary>
    /// <param name="referenceId"></param>
    /// <exception cref="Exception"></exception>
    public void DumpRoundRobinEloTable(string referenceId)
    {
      //decide total width for table
      int totalWidth = 25 * Players.Count;
      DumpHeadingTable(totalWidth);
      if (string.IsNullOrEmpty(referenceId))
      {
        for (int i = 0; i < Players.Count; i++)
        {
          IEnumerable<string> row = CreateRoundRobinEloStats(i);
          PrintCenterAlignedRow(row, totalWidth);
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
        PrintCenterAlignedRow(row, totalWidth);
      }
      PrintLine(totalWidth);
    }

    /// <summary>
    /// Get PlayerStat for a player with a specific opponent.
    /// </summary>
    /// <param name="player1"></param>
    /// <param name="opponent"></param>
    /// <returns></returns>

    public PlayerStat GetPlayer(string player1, string opponent)
    {
      PlayerStat player = Players.FirstOrDefault(e => e.Name == player1);

      if (player == null)
      {
        player = new PlayerStat() { Name = player1 };
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
      PlayerStat white;
      PlayerStat opponent;
      if (thisResult.Engine2IsWhite)
      {
        white = GetPlayer(engine.OpponentEngine.ID, engine.ID);
        opponent = GetPlayer(engine.ID, engine.OpponentEngine.ID);
      }
      else
      {
        white = GetPlayer(engine.ID, engine.OpponentEngine.ID);
        opponent = GetPlayer(engine.OpponentEngine.ID, engine.ID);
      }

      TournamentGameResult gameResultOpponent =
          thisResult.Result == TournamentGameResult.Win ? TournamentGameResult.Loss :
          thisResult.Result == TournamentGameResult.Loss ? TournamentGameResult.Win :
          TournamentGameResult.Draw;

      white.UpdatePlayerStat(thisResult.Result, opponent.Name);
      opponent.UpdatePlayerStat(gameResultOpponent, white.Name);
      UpdateNodeCounterAndTimeUse(thisResult, white, opponent);
    }

    void UpdateNodeCounterAndTimeUse(TournamentGameInfo thisResult, PlayerStat white, PlayerStat opponent)
    {
      if (thisResult.Engine2IsWhite)
      {
        white.PlayerTotalNodes += thisResult.TotalNodesEngine2;
        opponent.PlayerTotalNodes += thisResult.TotalNodesEngine1;
        white.PlayerTotalTime += thisResult.TotalTimeEngine2;
        opponent.PlayerTotalTime += thisResult.TotalTimeEngine1;
      }
      else
      {
        white.PlayerTotalNodes += thisResult.TotalNodesEngine1;
        opponent.PlayerTotalNodes += thisResult.TotalNodesEngine2;
        white.PlayerTotalTime += thisResult.TotalTimeEngine1;
        opponent.PlayerTotalTime += thisResult.TotalTimeEngine2;
      }

      //todo - update node and time stats for enabling median calculation of nps and time use
      foreach (var item in thisResult.GameMoveHistory)
      {
        if (item.Side == Chess.SideType.Black)
        {
          //float nps = item.FinalN / item.TimeElapsed;
          opponent.NodesPerMoveList.Add((item.FinalN,item.TimeElapsed));
          //opponent.TimeUsedPerMoveList.Add(item.TimeElapsed);
        }

        else
        {
          white.NodesPerMoveList.Add((item.FinalN,item.TimeElapsed));
          //white.TimeUsedPerMoveList.Add(item.TimeElapsed);
        }
      }

      ////debugging
      //float avgP1 = white.NodesPerMoveList.Average();
      //float avgP2 = opponent.NodesPerMoveList.Average();
      //string msg = $"Player1 avg nps: {avgP1:N0}, Player2 avg nps: {avgP2:N0}";
      ////Console.WriteLine(msg);
    }

    /// <summary>
    /// Dump dashes to console for a certain width.
    /// </summary>
    /// <param name="width"></param>
    void PrintLine(int width)
    {
      Console.WriteLine(new string('-', width));
    }

    /// <summary>
    /// Dump center aligned header with variable width to console.
    /// </summary>
    /// <param name="columns"></param>
    /// <param name="maxWidth"></param>

    void PrintHeaderRow(IEnumerable<(string, int)> columns, int maxWidth)
    {
      string row = "|";

      foreach ((string txt, int width) in columns)
      {
        row += AlignCentre(txt, width) + "|";
      }

      Console.WriteLine(row);
    }

    /// <summary>
    /// Dump center aligned text with fixed width to console. 
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

    /// <summary>
    /// Dump player text with variable column width to console.
    /// </summary>
    /// <param name="columns"></param>
    /// <param name="maxWidth"></param>

    void PrintEngineRow(List<(string, int)> columns, int maxWidth)
    {
      int numberOfColumns = columns.Count();
      string row = "|";

      for (int i = 0; i < numberOfColumns; i++)
      {
        var (txt, width) = columns[i];
        if (i > numberOfColumns - 3)
          row += AlignRight(txt, width) + "|";
        else
          row += AlignCentre(txt, width) + "|";
      }

      Console.WriteLine(row);
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

    /// <summary>
    /// Create Round Robin row for player based on index.
    /// </summary>
    /// <param name="row"></param>
    /// <returns></returns>
    IEnumerable<string> CreateRoundRobinRow(int row)
    {
      const string empty = "-----";
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
        yield return $"{win}-{draw}-{loss}";
        counter++;
      }

      if (row + 1 == Players.Count)
      {
        yield return empty;
      }
    }

    /// <summary>
    /// Create Round Robin Elo stat for player based on index row.
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
        var eloPerf = EloCalculator.EloDiff(win, draw, loss).ToString("F0");
        var (min, avg, max) = EloCalculator.EloConfidenceInterval(win, draw, loss);
        var error = (max - avg).ToString("F0");
        var msg = $"{eloPerf} +/- {error: F0}";
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

    void DumpHeadingTable(int width)
    {
      IEnumerable<string> players = Players.Select(e => e.Name);
      List<string> header = new List<string>();
      header.Add("Engine");
      header.AddRange(players);
      PrintLine(width);
      PrintCenterAlignedRow(header, width);
      PrintLine(width);
    }

    /// <summary>
    /// Calculation of median Node speed for each player in the tournament
    /// </summary>
    public void CalculateMedianNodes(double medianScaler)
    {

      //debugging
      foreach (var info in GameInfos)
      {
        foreach (var move in info.GameMoveHistory)
        {
          if (move.FinalN < 0)
          {

          }
        }
      }

      foreach (var player in Players)
      {
        player.CalculateMedianNPS(medianScaler);
      }
    }
  }
}
