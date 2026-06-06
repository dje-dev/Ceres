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
using Ceres.Chess.Textual.PgnFileTools;
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
    /// Headline pentanomial (paired-game) result for the tournament, computed for the
    /// first non-reference engine's aggregate matchup when the tournament summary is dumped.
    /// The trinomial statistics on PlayerStat remain available for backward compatibility.
    /// </summary>
    public PentanomialResult Pentanomial { get; set; } = PentanomialResult.Empty;

    /// <summary>
    /// Tracks the first-completed game of each pair (keyed by the global pair index,
    /// GameSequenceNum / 2) while awaiting its partner game to form a completed pair
    /// for pentanomial accounting. Only ever accessed under the statistics lock (i.e.
    /// from UpdateTournamentStats).
    /// </summary>
    private readonly Dictionary<int, (string engA, string engB, int aHalfGame1)> pendingPentanomialPairs = new();

    /// <summary>
    /// Dump full tournament summary to console.
    /// </summary>
    public void DumpTournamentSummary(TournamentDef def)
    {
      TextWriter writer = def.Logger;
      string referenceId = def.ReferenceEngineId;

      //parameter for how many percent of items above and below median should be included in the average median calculation.
      //Navs summary normally use 20% av items above and below median value.
      double medianRangePercent = 0.20;
      CalculateMedianNodes(medianRangePercent);
      writer.WriteLine();
      writer.WriteLine("Tournament summary:");
      DumpEngineTournamentSummary(def);
      writer.WriteLine("Simple summary:");
      DumpSimpleEngineTournamentSummary(def);
      writer.WriteLine("Tournament round robin score table (W-D-L):");
      DumpRoundRobinResultTable(writer, referenceId);
      writer.WriteLine();
      writer.WriteLine("Tournament round robin Elo table (W-D-L):");
      DumpRoundRobinEloTable(writer, referenceId);
      writer.WriteLine();
      writer.WriteLine("Pentanomial analysis (paired-game statistics):");
      DumpPentanomialSummary(def);
      writer.WriteLine();

      Console.WriteLine();
      if (def.ForceReferenceEngineDeterministic)
      {
        int totalForced = GameInfos.Sum(g => g.NumMovesForcedDeterministic);
        writer.WriteLine($"Number of reference engine moves overridden to force deterministic: {totalForced:N0}");
      }
    }

    /// <summary>
    /// Dumps full engine summary table to console.
    /// </summary>
    void DumpEngineTournamentSummary(TournamentDef def)
    {
      TextWriter writer = def.Logger;
      string referenceId = def.ReferenceEngineId;

      int maxWidth = 155;
      PrintLine(writer, maxWidth - 1);
      List<(string, int)> header = new List<(string, int)>
        { ("Player",25), ("Elo", 8), ("+/-",5), ("CFS(%)", 8),
          ("Played", 8), ("W-D-L", 13), ("D(%)",5), ("Time",12), ("Nodes",18), ("NPS-avg", 14), ("EPS-avg", 14), ("NPS-med", 11) };
      PrintHeaderRow(writer, header, maxWidth);
      PrintLine(writer, maxWidth - 1);
      bool twoPlayers = Players.Count == 2 && string.IsNullOrEmpty(referenceId);
      var sorted = Players.OrderByDescending(e => e.Name == referenceId ? 10000 : 0 + e.PlayerWins + (e.Draws * 0.5));
      foreach (PlayerStat engine in sorted)
      {
        var refEng = twoPlayers ? engine.Name : referenceId;
        WriteEngineSummary(writer, engine, maxWidth, refEng);
        twoPlayers = false;
      }
      PrintLine(writer, maxWidth - 1);
      writer.WriteLine();
    }

    void DumpSimpleEngineTournamentSummary(TournamentDef def)
    {
      TextWriter writer = def.Logger;
      string referenceId = def.ReferenceEngineId;

      int maxWidth = 108;
      PrintLine(writer, maxWidth);
      List<(string, int)> header = new List<(string, int)>
        { ("Player",25), ("Elo", 8), ("+/-",5), ("CFS(%)", 8),
           ("W-D-L", 13), ("Time",12), ("NPS-avg", 14), ("EPS-avg", 14) };
      PrintHeaderRow(writer, header, maxWidth);
      PrintLine(writer, maxWidth);
      bool twoPlayers = Players.Count == 2 && string.IsNullOrEmpty(referenceId);
      var sorted = Players.OrderByDescending(e => e.Name == referenceId ? 10000 : 0 + e.PlayerWins + (e.Draws * 0.5));
      foreach (PlayerStat engine in sorted)
      {
        var refEng = twoPlayers ? engine.Name : referenceId;
        WriteSimpleEngineSummary(writer, engine, maxWidth, refEng);
        twoPlayers = false;
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
      //double score = player.PlayerWins + (player.Draws / 2.0);
      string wdl = $"+{player.PlayerWins}={player.Draws}-{player.PlayerLosses}";
      // Error bar (+/-) and CFS (likelihood of superiority) are based on pentanomial
      // (paired-game) analysis; the Elo point estimate is the same as the trinomial mean.
      PentanomialResult penta = PentanomialAggregateFor(player.Name);
      float cfs = penta.LOS;
      var (_, avg, _) = EloCalculator.EloConfidenceInterval(player.PlayerWins, player.Draws, player.PlayerLosses);
      string error = $"{penta.EloErrorMargin:F0}";
      double draws = (player.Draws / (double)player.NumGames) * 100;
      long nodes = player.PlayerTotalNodes;
      float time = player.PlayerTotalTime;
      // Average evaluations per second (neural network position evaluations), shown where reported.
      string epsAvg = player.PlayerTotalEvaluations > 0 && time > 0
                    ? (player.PlayerTotalEvaluations / time).ToString("N0") + " " : "-";

      List<(string, int)> rowItems = new()
      {
        (playerInfo, 25),
        (player.Name == referenceId ? "0.0" : avg.ToString("F0"), 8),
        (player.Name == referenceId ? "---" : error, 5),
        (player.Name == referenceId ? "----" : cfs.ToString("P0"), 8),
        (player.NumGames.ToString(), 8),
        (wdl, 13),
        (draws.ToString("N0"), 5),
        { (time.ToString("F2"), 12) },
        { (nodes.ToString("N0") + " ", 18) },
        { ((nodes / time).ToString("N0") + " ", 14) },
        { (epsAvg, 14) },
        { ((player.MedianNPSAverage).ToString("N0") + " ", 11) }

      };
      PrintEngineRow(writer, rowItems, width);
    }

    void WriteSimpleEngineSummary(TextWriter writer, PlayerStat player, int width, string referenceId)
    {
      string playerInfo = player.Name == referenceId ? player.Name + "*" : player.Name;
      string wdl = $"+{player.PlayerWins}={player.Draws}-{player.PlayerLosses}";
      // Error bar (+/-) and CFS (likelihood of superiority) are based on pentanomial
      // (paired-game) analysis; the Elo point estimate is the same as the trinomial mean.
      PentanomialResult penta = PentanomialAggregateFor(player.Name);
      float cfs = penta.LOS;
      var (_, avg, _) = EloCalculator.EloConfidenceInterval(player.PlayerWins, player.Draws, player.PlayerLosses);
      string error = $"{penta.EloErrorMargin:F0}";
      long nodes = player.PlayerTotalNodes;
      float time = player.PlayerTotalTime;
      // Average evaluations per second (neural network position evaluations), shown where reported.
      string epsAvg = player.PlayerTotalEvaluations > 0 && time > 0
                    ? (player.PlayerTotalEvaluations / time).ToString("N0") + " " : "-";

      List<(string, int)> rowItems = new()
      {
        (playerInfo, 25),
        (player.Name == referenceId ? "0.0" : avg.ToString("F0"), 8),
        (player.Name == referenceId ? "---" : error, 5),
        (player.Name == referenceId ? "----" : cfs.ToString("P0"), 8),
        (wdl, 13),
        { (time.ToString("F2"), 12) },
        { ((nodes / time).ToString("N0") + " ", 14) },
        { (epsAvg, 14) },
      };
      PrintSimpleEngineRow(writer, rowItems, width);
    }


    /// <summary>
    /// Dumps the pentanomial (paired-game) analysis table to the writer, with one row per
    /// matchup showing the distribution of pair outcomes, the pentanomial Elo error bar
    /// (alongside the trinomial error bar for comparison), and the likelihood of superiority.
    /// Also populates the headline Pentanomial property (first non-reference engine aggregate).
    /// </summary>
    /// <param name="def"></param>
    void DumpPentanomialSummary(TournamentDef def)
    {
      TextWriter writer = def.Logger;
      string referenceId = def.ReferenceEngineId;

      // Populate the headline pentanomial result (first non-reference engine's aggregate).
      PlayerStat headline = Players.FirstOrDefault(p => p.Name != referenceId) ?? Players.FirstOrDefault();
      if (headline != null)
      {
        Pentanomial = PentanomialAggregateFor(headline.Name);
      }

      List<(string, int)> header = new List<(string, int)>
      {
        ("Matchup", 28), ("Pairs", 7), ("LL", 6), ("LD+DL", 8), ("WL+LW+DD", 11),
        ("WD+DW", 8), ("WW", 6), ("Score%", 8), ("Elo", 8), ("Penta +/-", 11),
        ("Tri +/-", 9), ("LOS%", 7)
      };
      int totalWidth = header.Sum(h => h.Item2) + header.Count + 1;

      PrintLine(writer, totalWidth);
      PrintHeaderRow(writer, header, totalWidth);
      PrintLine(writer, totalWidth);

      bool anyRows = false;
      for (int i = 0; i < Players.Count; i++)
      {
        for (int j = i + 1; j < Players.Count; j++)
        {
          PlayerStat a = Players[i];
          PlayerStat b = Players[j];
          if (!a.OpponentsPentanomial.TryGetValue(b.Name, out long[] counts))
          {
            continue; // these two engines did not play each other
          }

          anyRows = true;
          PentanomialResult penta = PentanomialCalculator.Compute(counts);

          // Trinomial error bar (1 sigma) for side-by-side comparison.
          string triError = "----";
          if (a.Opponents.TryGetValue(b.Name, out (int w, int d, int l) wdl))
          {
            var (_, triAvg, triMax) = EloCalculator.EloConfidenceInterval(wdl.w, wdl.d, wdl.l);
            triError = $"{(triMax - triAvg):F0}";
          }

          List<(string, int)> row = new List<(string, int)>
          {
            ($"{a.Name} vs {b.Name}", 28),
            (penta.NumPairs.ToString("N0"), 7),
            (counts[0].ToString("N0"), 6),
            (counts[1].ToString("N0"), 8),
            (counts[2].ToString("N0"), 11),
            (counts[3].ToString("N0"), 8),
            (counts[4].ToString("N0"), 6),
            ((penta.ScoreRate * 100).ToString("F1"), 8),
            (penta.Elo.ToString("F0"), 8),
            (penta.EloErrorMargin.ToString("F0"), 11),
            (triError, 9),
            ((penta.LOS * 100).ToString("F0"), 7)
          };
          PrintPentanomialRow(writer, row);
        }
      }

      if (!anyRows)
      {
        writer.WriteLine("(no completed game pairs)");
      }
      PrintLine(writer, totalWidth);
    }

    /// <summary>
    /// Dumps a single pentanomial table row (matchup label centered, numeric columns right aligned).
    /// </summary>
    void PrintPentanomialRow(TextWriter writer, List<(string, int)> columns)
    {
      string row = "|";
      for (int i = 0; i < columns.Count; i++)
      {
        var (txt, width) = columns[i];
        row += (i == 0 ? AlignCentre(txt, width) : AlignRight(txt, width)) + "|";
      }
      writer.WriteLine(row);
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
      int totalWidth = 20 * (Players.Count + 1);
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
    public void UpdateTournamentStats(TournamentGameInfo thisResult, string playerID, string opponentID)
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
        playerWhite = GetPlayer(opponentID, playerID);
        playerBlack = GetPlayer(playerID, opponentID);
        whiteResult = reverseResult;
        blackResult = thisResult.Result;
      }
      else
      {
        playerWhite = GetPlayer(playerID, opponentID);
        playerBlack = GetPlayer(opponentID, playerID);
        whiteResult = thisResult.Result;
        blackResult = reverseResult;
      }

      playerWhite.UpdateGameOutcome(whiteResult, playerBlack.Name);
      playerBlack.UpdateGameOutcome(blackResult, playerWhite.Name);
      UpdateNodeCounterAndTimeUse(thisResult, playerWhite, playerBlack);
      UpdatePentanomialStats(thisResult);
    }


    /// <summary>
    /// Reverses a game result (from one player's perspective to the opponent's).
    /// </summary>
    private static TournamentGameResult Reverse(TournamentGameResult result) =>
        result == TournamentGameResult.Win ? TournamentGameResult.Loss :
        result == TournamentGameResult.Loss ? TournamentGameResult.Win :
        TournamentGameResult.Draw;

    /// <summary>
    /// Half-points (out of 2) earned in a single game given a result, so that pair scores
    /// remain integers in 0..4. Matches the trinomial accounting which treats a non-decisive
    /// result (including None) as a draw.
    /// </summary>
    private static int HalfPoints(TournamentGameResult result) =>
        result == TournamentGameResult.Win ? 2 :
        result == TournamentGameResult.Loss ? 0 :
        1;

    /// <summary>
    /// Updates the pentanomial (paired-game) statistics for a completed game.
    ///
    /// The two games of a pair share the same pair index (GameSequenceNum / 2). The first
    /// game to complete is held pending; when its partner completes, the pair score is
    /// recorded for both engines (with mirrored buckets, since the pair is zero-sum).
    ///
    /// Must only be called while holding the statistics lock (it is invoked from
    /// UpdateTournamentStats), since it reads and mutates shared accumulation state.
    /// </summary>
    /// <param name="thisResult"></param>
    void UpdatePentanomialStats(TournamentGameInfo thisResult)
    {
      // Half-points earned by the engine playing White in this game (zero-sum: Black gets 2 - white).
      // thisResult.Result is from engine 1's (player1's) perspective, so White's result is the
      // reverse when engine 2 is White and Result directly when engine 1 is White. This mirrors
      // the whiteResult mapping in UpdateTournamentStats, keeping the pentanomial accounting
      // consistent with the trinomial accounting.
      int whiteHalf = thisResult.Engine2IsWhite
                        ? HalfPoints(Reverse(thisResult.Result))
                        : HalfPoints(thisResult.Result);

      int pairKey = thisResult.GameSequenceNum / 2;

      if (pendingPentanomialPairs.Remove(pairKey, out (string engA, string engB, int aHalfGame1) pending))
      {
        // Both games of the pair are now complete. Compute engine A's half-points in this
        // (second) game; A played White in the first game, so its color here may differ.
        int aHalfThisGame = thisResult.PlayerWhite == pending.engA ? whiteHalf : 2 - whiteHalf;
        int idxA = pending.aHalfGame1 + aHalfThisGame; // in 0..4

        GetPlayer(pending.engA, pending.engB).AddPentanomialPair(pending.engB, idxA);
        GetPlayer(pending.engB, pending.engA).AddPentanomialPair(pending.engA, 4 - idxA);
      }
      else
      {
        // First game of the pair: hold it pending until its partner completes.
        // Engine A is defined as the engine playing White in this first game.
        pendingPentanomialPairs[pairKey] = (thisResult.PlayerWhite, thisResult.PlayerBlack, whiteHalf);
      }
    }


    /// <summary>
    /// Returns the pentanomial result for a specific matchup (player versus a single opponent).
    /// </summary>
    /// <param name="player"></param>
    /// <param name="opponent"></param>
    /// <param name="mult">Number of standard errors for the Elo error margin (1.0 = 1 sigma).</param>
    /// <returns>The pentanomial result, or PentanomialResult.Empty if no completed pairs.</returns>
    public PentanomialResult PentanomialFor(string player, string opponent, float mult = 1.0f)
    {
      PlayerStat stat = Players.FirstOrDefault(e => e.Name == player);
      if (stat == null || !stat.OpponentsPentanomial.TryGetValue(opponent, out long[] counts))
      {
        return PentanomialResult.Empty;
      }
      return PentanomialCalculator.Compute(counts, mult);
    }


    /// <summary>
    /// Returns the pentanomial result for a player aggregated across all opponents.
    /// </summary>
    /// <param name="player"></param>
    /// <param name="mult">Number of standard errors for the Elo error margin (1.0 = 1 sigma).</param>
    /// <returns>The pentanomial result, or PentanomialResult.Empty if no completed pairs.</returns>
    public PentanomialResult PentanomialAggregateFor(string player, float mult = 1.0f)
    {
      PlayerStat stat = Players.FirstOrDefault(e => e.Name == player);
      if (stat == null)
      {
        return PentanomialResult.Empty;
      }
      return PentanomialCalculator.Compute(stat.AggregatePentanomialCounts(), mult);
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
        playerWhite.PlayerTotalEvaluations += thisResult.TotalEvaluationsEngine2;
        playerBlack.PlayerTotalEvaluations += thisResult.TotalEvaluationsEngine1;
        playerWhite.PlayerTotalTime += thisResult.TotalTimeEngine2;
        playerBlack.PlayerTotalTime += thisResult.TotalTimeEngine1;
      }
      else
      {
        playerWhite.PlayerTotalNodes += thisResult.TotalNodesEngine1;
        playerBlack.PlayerTotalNodes += thisResult.TotalNodesEngine2;
        playerWhite.PlayerTotalEvaluations += thisResult.TotalEvaluationsEngine1;
        playerBlack.PlayerTotalEvaluations += thisResult.TotalEvaluationsEngine2;
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

    void PrintSimpleEngineRow(TextWriter writer, List<(string, int)> columns, int maxWidth)
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

      writer.WriteLine(row);
    }

    void PrintEngineRow(TextWriter writer, List<(string, int)> columns, int maxWidth)
    {
      int numberOfColumns = columns.Count();
      string row = "|";

      for (int i = 0; i < numberOfColumns; i++)
      {
        var (txt, width) = columns[i];
        if (i > numberOfColumns - 5)
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

        string msg = ShortEloStrForOpponent(opponent);
        yield return msg;
        counter++;
      }

      if (row + 1 == Players.Count)
      {
        yield return empty;
      }
    }

    private static string ShortEloStrForOpponent(KeyValuePair<string, (int, int, int)> opponent)
    {
      var (win, draw, loss) = opponent.Value;
      string eloPerf = EloCalculator.EloDiff(win, draw, loss).ToString("F0");
      var (min, avg, max) = EloCalculator.EloConfidenceInterval(win, draw, loss);
      string error = (max - avg).ToString("F0");
      string msg = $"{eloPerf} +/- {error: F0}";
      return msg;
    }


    /// <summary>
    /// Returns a short string (W/D/L with error bars) for specified player versus opponent.
    /// </summary>
    /// <param name="playerName"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public string ShortEloSummaryStr(string playerName)
    {
      PlayerStat player = Players.FirstOrDefault(p => p.Name == playerName);
      if (player == default)
      {
        throw new Exception($"Player {playerName} not found in tournament.");
      }

      if (player.Opponents.Count != 1)
      {
        throw new Exception($"Player {playerName} has more than one opponent.");
      }

      return ShortEloStrForOpponent(player.Opponents.First());
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
