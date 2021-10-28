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
        //public int GamesPlayed => Results.Sum(e => e.NumGames);

        /// <summary>
        /// Names for engines in a round robin tournament
        /// </summary>
        public List<string> PlayerNames { get; set; }

        public List<TournamentResultStats> Results { get; set; }


        public TournamentResultStats(IEnumerable<string> engines)
        {
            PlayerNames = engines.ToList();
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
            Results = new List<TournamentResultStats>();
        }


        /// <summary>
        /// Updates tournament statistics based on a game with specified result.
        /// </summary>
        /// <param name="thisResult"></param>
        public void UpdateTournamentStats(TournamentGameInfo thisResult)
        {
            switch (thisResult.Result)
            {
                case TournamentGameResult.Win:
                    Player1Wins++;
                    GameOutcomesString += "+";
                    break;

                case TournamentGameResult.Loss:
                    Player1Losses++;
                    GameOutcomesString += "-";
                    break;

                default:
                    Draws++;
                    GameOutcomesString += "=";
                    break;
            }
        }

        void UpdateRRStats(TournamentResultStats stat, TournamentGameInfo result)
        {
            stat.UpdateTournamentStats(result);
        }

        public TournamentResultStats GetResultsForPlayer(string player1, string player2)
        {
            var white = Results.FirstOrDefault(e => e.Player1 == player1);

            if (white == null)
            {
                var entry = new TournamentResultStats(player1, player2);
                Results.Add(entry);
                return entry;
            }
            return white;
        }

        public void UpdateRRTournamentStats(TournamentGameInfo thisResult, GameEngine engine1, GameEngine engine2)
        {
            TournamentResultStats stats;
            stats = GetResultsForPlayer(engine1.ID, engine2.ID);
            stats.UpdateRRStats(stats, thisResult);
            var otherPlayer = GetResultsForPlayer(engine2.ID, engine1.ID);
            var gameResultBlack =
                thisResult.Result == TournamentGameResult.Win ? TournamentGameResult.Loss :
                thisResult.Result == TournamentGameResult.Loss ? TournamentGameResult.Win :
                TournamentGameResult.Draw;

            var reverseResult = thisResult with { Result = gameResultBlack };
            otherPlayer.UpdateRRStats(otherPlayer, reverseResult);
            //GamesPlayed++;
            
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

        public void DumpRoundRobin()
        {
            var numberOfGames = Results.Sum(e => e.NumGames / 2);
            Console.WriteLine($"Tournament Results from {numberOfGames} games");
            WriteResults();
        }

        void WriteResults()
        {
            foreach (var item in Results)
            {
                Console.WriteLine($"  {item.Player1} wins {item.Player1Wins}");
                Console.WriteLine($"  {item.Player1} draws {item.Draws}");
                Console.WriteLine($"  {item.Player1} loses {item.Player1Losses}");
                var eloInterval = EloCalculator.EloConfidenceInterval(item.Player1Wins, item.Draws, item.Player1Losses);
                Console.WriteLine($"  ELO Difference {eloInterval.avg,4:F0} +/- {eloInterval.max - eloInterval.avg,4:F0}");
                Console.WriteLine();
            }
        }
    }
}
