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

using Ceres.Base.Misc;

using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.Chess.Textual.PgnFileTools;
using Ceres.Chess.Games.Utils;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.Features.Visualization.AnalysisGraph;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Features.UCI
{
  /// <summary>
  /// Static helper methods useful for the UCIManager.
  /// </summary>
  public static class UCIManagerHelpers
  {
    /// <summary>
    /// Parses a specification of search time in UCI format into an equivalent SearchLimit.
    /// Returns null if parsing failed.
    /// </summary>
    /// <param name="command"></param>
    /// <returns></returns>
    static internal SearchLimit GetSearchLimit(string command, PositionWithHistory curPositionAndMoves, Action<string> uciWriteLine)
    {
      SearchLimit searchLimit;
      UCIGoCommandParsed goInfo = new UCIGoCommandParsed(command, curPositionAndMoves.FinalPosition);
      if (!goInfo.IsValid) return null;

      if (goInfo.Nodes.HasValue)
      {
        searchLimit = SearchLimit.NodesPerMove(goInfo.Nodes.Value);
      }
      else if (goInfo.MoveTime.HasValue)
      {
        searchLimit = SearchLimit.SecondsPerMove(goInfo.MoveTime.Value / 1000.0f);
      }
      else if (goInfo.Infinite)
      {
        searchLimit = SearchLimit.NodesPerMove(MCTSNodeStore.MAX_NODES);
      }
      else if (goInfo.BestValueMove)
      {
        searchLimit = SearchLimit.BestValueMove;
      }
      else if (goInfo.BestActionMove)
      {
        searchLimit = SearchLimit.BestValueMove;
      }
      else if (goInfo.TimeOurs.HasValue)
      {
        float increment = 0;
        if (goInfo.IncrementOurs.HasValue) increment = goInfo.IncrementOurs.Value / 1000.0f;

        int? movesToGo = null;
        if (goInfo.MovesToGo.HasValue) movesToGo = goInfo.MovesToGo.Value;

        searchLimit = SearchLimit.SecondsForAllMoves(goInfo.TimeOurs.Value / 1000.0f, increment, movesToGo, true);
      }
      else if (goInfo.NodesOurs.HasValue)
      {
        float increment = 0;
        if (goInfo.IncrementOurs.HasValue) increment = goInfo.IncrementOurs.Value;

        int? movesToGo = null;
        if (goInfo.MovesToGo.HasValue) movesToGo = goInfo.MovesToGo.Value;

        searchLimit = SearchLimit.NodesForAllMoves(goInfo.NodesOurs.Value, (int)increment, movesToGo, true);
      }
      else
      {
        uciWriteLine($"Unsupported time control in UCI go command {command}");
        return null;
      }

      // Add on possible search moves restriction.
      return searchLimit with { SearchMoves = goInfo.SearchMoves };
    }


    static public void ProcessDownloadCommand(string c, Action<string> uciWriteLine)
    {
      string[] partsDownload = c.Split(" ");
      if (partsDownload.Length != 2)
      {
        uciWriteLine("info string Invalid download command, expect Ceres network ID after download command");
        return;
      }

      string ceresNetID = partsDownload[1].ToUpper();
      if (!ceresNetID.StartsWith("C"))
      {
        uciWriteLine("info string Invalid Ceres network ID (expected to begin with C, see https://github.com/dje-dev/CeresNets)");
        return;
      }

      CeresNetDownloader downloader = new CeresNetDownloader();
      (bool alreadyDownloaded, string fullNetworkPath) downloadResults;
      downloadResults = downloader.DownloadCeresNetIfNeeded(ceresNetID, CeresUserSettingsManager.Settings.DirCeresNetworks, false);

      if (downloadResults.alreadyDownloaded)
      {
        uciWriteLine("info string Network previously downloaded: " + downloadResults.fullNetworkPath);
      }
      else
      {
        uciWriteLine("info string Network downloaded to: " + downloadResults.fullNetworkPath);
      }
    }


    /// <summary>
    /// Parses and process the game comparison feature command.
    /// </summary>
    /// <param name="c"></param>
    static internal void ProcessGameComp(string c, Action<string> uciWriteLine)
    {
      string[] parts = c.TrimEnd().Split(" ");
      if (parts.Length < 2)
      {
        uciWriteLine("Expected name of PGN file possibly followed by list of games (e.g. \"1,2\") or a round number \"e.g. r1\")");
        return;
      }
      string fn = parts[1];
      if (!System.IO.File.Exists(fn))
      {
        uciWriteLine($"Specified file not found {fn}");
        return;
      }

      List<PGNGame> games = PgnStreamReader.ReadGames(fn);
      if (parts.Length == 3)
      {
        string gamesList = parts[2].ToUpper();

        if (gamesList.StartsWith("R"))
        {
          // One round with specified index.
          int round = int.Parse(gamesList.Substring(1));
          uciWriteLine($"Generating game comparison graph of round {round} from {fn}");
          GameCompareGraphGenerator comp = new(games, s => s.Round == round, s => s.Round);
          comp.Write(launchWithBrowser: true);
        }
        else
        {
          // List of games by index.
          string[] gameIndices = gamesList.Split(",");
          uciWriteLine($"Generating game comparison graph of games {gamesList} from {fn}");
          GameCompareGraphGenerator comp = new(games, s => Array.IndexOf(gameIndices, s.GameIndex.ToString()) != -1, s => 1);
          comp.Write(launchWithBrowser: true);
        }
      }
      else
      {
        // All games by round.
        uciWriteLine($"Generating game comparison graph of all rounds from {fn}");
        GameCompareGraphGenerator comp = new(games, s => true, s => s.Round);
        comp.Write(launchWithBrowser: true);
      }
    }

  }
}
