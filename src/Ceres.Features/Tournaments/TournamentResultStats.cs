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
    /// Dumps summary to Console.
    /// </summary>
    public void Dump()
    {
      Console.WriteLine($"Tournament Results of {Player1} versus {Player2} in {NumGames} games");
      Console.WriteLine($"  {Player1} wins {Player1Wins}");
      Console.WriteLine($"  {Player1} draws {Draws}");
      Console.WriteLine($"  {Player1} loses {Player1Losses}");

      var eloInterval = EloCalculator.EloConfidenceInterval(Player1Wins, Draws, Player1Losses);
      Console.WriteLine($"  ELO Difference {eloInterval.avg,4:F0} +/- {eloInterval.max - eloInterval.avg,4:F0}");
    }
  }
}
