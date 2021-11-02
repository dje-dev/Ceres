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
using Ceres.Base.DataTypes;
using Ceres.Features.Players;
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


    public static FeatureTournParams ParseTournCommand(string fen, string args)
    {
      List<string> validArgs = new List<string>(FEATURE_PLAY_COMMON_ARGS);
      validArgs.Add("OPENINGS");
      validArgs.Add("SHOW-MOVES");

      KeyValueSetParsed keys = new KeyValueSetParsed(args, validArgs);

      FeatureTournParams parms = new FeatureTournParams()
      {
        Openings = keys.GetValueOrDefault("Openings", null, false),
        ShowMoves = keys.GetValueOrDefaultMapped<bool>("Show-Moves", "true", false, str => bool.Parse(str))
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

      TournamentManager runner = new TournamentManager(def, 1);
      runner.RunTournament(null);
    }

  }
}
