using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ceres.Features.Tournaments
{
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

    public float MedianNodeValue { get; set; }   

    public float TotalTimeUsed { get; set; }
    

    /// <summary>
    /// Table to store Win-Draw-Loss statistics against each opponent
    /// </summary>
    public Dictionary<string, (int, int, int)> Opponents { get; set; } = new Dictionary<string, (int, int, int)>();

    /// <summary>
    /// A list of nodes per move for all games played in tournament
    /// </summary>
    public List<(float,float)> NodesPerMoveList { get; set; } = new List<(float,float)>();

    public List<float> TimeUsedPerMoveList { get; set; } = new List<float>();


    public void UpdatePlayerStat(TournamentGameResult result, string opponent)
    {
      UpdateGameOutcome(result, opponent);
    }

    /// <summary>
    /// Updates player statistics based on a game with specified result.
    /// </summary>
    /// <param name="result"></param>
    /// <param name="opponent"></param>
    private void UpdateGameOutcome(TournamentGameResult result, string opponent)
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

    public void CalculateMedianNPS(double medianScaler)
    {
      var orderedArray = NodesPerMoveList.OrderBy(key => key.Item1/key.Item2).ToArray();
      var length = orderedArray.Length;
      var indexMedian = (int) length / 2;
      (float nodes,float time) = orderedArray[indexMedian];      
      var take = (int) (length * medianScaler);
      var upperRange = indexMedian + take;
      var LowerRange = indexMedian - take;
      List<(float,float)> medianRangedValues = new();
      for (int i = 0; i < length - 1; i++)
      {
        if (i >= LowerRange && i <= upperRange)
        {
          (float n, float t) = orderedArray[i];
          medianRangedValues.Add((n,t));
        }
      }
      var totalNodes = medianRangedValues.Sum(key => key.Item1);
      var totalTime = medianRangedValues.Sum(key => key.Item2);
      var medianNps = totalNodes / totalTime;
      MedianNodeValue = medianNps;

      //var medianNps = orderedArray.Average(); //medianRangedValues.Average();
      //var timeUsed = TimeUsedPerMoveList.Sum();
      //var total = orderedArray.Sum();
      //var avg = total / length;
      //if (PlayerTotalTime != timeUsed)
      //{

      //}
      //if (total != PlayerTotalNodes)
      //{

      //}
      //var avgSpeed = orderedArray.Average(key => key.Item1/key.Item2);
      //var nodesPersec = PlayerTotalNodes / PlayerTotalTime;
    }
  }
}
