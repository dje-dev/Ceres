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

using System.Collections.Generic;
using System.Linq;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Statistics related to a player in a tournament.
  /// </summary>
  public class PlayerStat
  {
    /// <summary>
    /// Player name.
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// Number of wins by player.
    /// </summary>
    public int PlayerWins { set; get; }

    /// <summary>
    /// Number of draws.
    /// </summary>
    public int Draws { set; get; }

    /// <summary>
    /// Number of losses by player.
    /// </summary>
    public int PlayerLosses { set; get; }

    /// <summary>
    /// Short string summarizing games outcome.
    /// </summary>
    public string GameOutcomesString { set; get; }

    /// <summary>
    /// Total number of games played.
    /// </summary>
    public int NumGames => PlayerWins + Draws + PlayerLosses;

    /// <summary>
    /// Total number of nodes for player across all games.
    /// </summary>
    public long PlayerTotalNodes { set; get; }

    /// <summary>
    /// Total time spent in seconds for player across all games.
    /// </summary>
    public float PlayerTotalTime { get; set; }

    /// <summary>
    /// The median average for nodes per second within a specific range from median value - typically +/- 20% range for all moves in all games.
    /// </summary>
    public float MedianNPSAverage { get; set; }      

    /// <summary>
    /// Table to store Win-Draw-Loss statistics against each opponent.
    /// </summary>
    public Dictionary<string, (int, int, int)> Opponents { get; set; } = new Dictionary<string, (int, int, int)>();
  
    /// <summary>
    /// Update player statistics based on a game result.
    /// </summary>
    /// <param name="result"></param>
    /// <param name="opponent"></param>
    public void UpdateGameOutcome(TournamentGameResult result, string opponent)
    {
      var (win, draw, loss) = Opponents[opponent];
      switch (result)
      {
        case TournamentGameResult.Win:
          PlayerWins++;
          Opponents[opponent] = (win + 1, draw, loss);
          GameOutcomesString += "+";
          break;

        case TournamentGameResult.Loss:
          PlayerLosses++;
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
    /// Perform median calculation for nodes per second for all moves in all games played.
    /// A range selector is used to choose a range of values to include in the calculation.
    /// </summary>
    /// <param name="rangeSelector"></param>
    /// <param name="npsPerMove"></param>
    public void CalculateMedianNPS(double rangeSelector, IEnumerable<float> npsPerMove)
    {
      var npsOrdered = npsPerMove.OrderBy(e => e).ToArray();
      string name = Name;
      int length = npsOrdered.Length;
      int indexMedian = (int) length / 2;      
      int take = (int) (length * rangeSelector);
      int upperRange = indexMedian + take;
      int LowerRange = indexMedian - take;
      List<float> medianRangedValues = new();

      for (int i = 0; i < length - 1; i++)
      {
        if (i >= LowerRange && i <= upperRange)
        {
          float nps = npsOrdered[i];
          medianRangedValues.Add(nps);
        }
      }
      MedianNPSAverage = medianRangedValues.Average();
    }    
  }
}
