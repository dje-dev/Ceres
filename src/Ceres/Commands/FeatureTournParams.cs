#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;
using System.Collections.Generic;

using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Base.DataTypes;
using Ceres.Features.Tournaments;

#endregion

namespace Ceres.Commands
{
  public record FeatureTournParams : FeaturePlayWithOpponent
  {
    /// <summary>
    /// The name of an EPD or PGN file containing the set of 
    /// positions to be used as opening positions in the tournament.
    /// </summary>
    public string Openings { init; get; }
    public bool ShowMoves { init; get; } = true;

    /// <summary>
    /// Optional device slots specification over which concurrent games are spread
    /// (see NNDeviceSlots), for example "GPU:0+GPU:(0,1)*2@HOP+GPU:3*3@station".
    /// </summary>
    public string Slots { init; get; }

    /// <summary>
    /// Optional number of game pairs to play (by default all openings are played).
    /// </summary>
    public int? GamePairs { init; get; }


    public static FeatureTournParams ParseTournCommand(string fen, string args)
    {
      List<string> validArgs = new List<string>(FEATURE_PLAY_COMMON_ARGS);
      validArgs.Add("OPENINGS");
      validArgs.Add("SHOW-MOVES");
      validArgs.Add("SLOTS");
      validArgs.Add("GAME-PAIRS");

      KeyValueSetParsed keys = new KeyValueSetParsed(args, validArgs);

      string gamePairsStr = keys.GetValueOrDefault("Game-Pairs", null, false);

      FeatureTournParams parms = new FeatureTournParams()
      {
        Openings = keys.GetValueOrDefault("Openings", null, false),
        ShowMoves = keys.GetValueOrDefaultMapped<bool>("Show-Moves", "true", false, str => bool.Parse(str)),
        Slots = keys.GetValueOrDefault("Slots", null, false),
        GamePairs = gamePairsStr == null ? null : int.Parse(gamePairsStr)
      };

      // Add in all the fields from the base class
      parms.ParseBaseFields(args, false);

      if (fen != null && parms.Openings != null)
      {
        throw new Exception("Only one of FEN Openings should be specified with the TOURN command");
      }

      return parms;
    }

    public static void RunTournament(FeatureTournParams tournParams, string fen)
    {
      (EnginePlayerDef player1, EnginePlayerDef player2) = tournParams.GetEngineDefs(true);

      TournamentDef def = new TournamentDef("Tournament", player1, player2);
      def.ShowGameMoves = tournParams.ShowMoves;

      if (tournParams.Openings != null)
      {
        def.OpeningsFileName = tournParams.Openings;
      }
      else if (fen != null)
      {
        def.StartingFEN = fen;
      }
      else
      {
        def.StartingFEN = Position.StartPosition.FEN;
      }

      if (tournParams.GamePairs.HasValue)
      {
        def.NumGamePairs = tournParams.GamePairs;
      }

      // If device slots were specified they determine the number of concurrent games
      // (and where each is played); otherwise games are played one at a time.
      def.DeviceSlots = tournParams.Slots;
      TournamentManager runner = tournParams.Slots != null ? new TournamentManager(def)
                                                           : new TournamentManager(def, 1);
      runner.RunTournament(null);
    }

  }
}
