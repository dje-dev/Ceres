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
using XPlot.Plotly;

using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.Charts;
using Ceres.Chess.GameEngines;

#endregion

namespace Ceres.Features.Tournaments
{
  public enum TournamentGameResult
  {
    /// <summary>
    /// Game not completed.
    /// </summary>
    None,

    /// <summary>
    /// Win.
    /// </summary>
    Win,

    /// <summary>
    /// Loss.
    /// </summary>
    Loss,

    /// <summary>
    /// Draw.
    /// </summary>
    Draw
  };


  public enum TournamentGameResultReason
  {
    /// <summary>
    /// Checkmate.
    /// </summary>
    Checkmate,

    /// <summary>
    /// Stalemate.
    /// </summary>
    Stalemate,

    /// <summary>
    /// Adjudicate tablebase known result.
    /// </summary>
    AdjudicateTB,

    /// <summary>
    /// Adjudicate insufficient mating material.
    /// </summary>
    AdjudicateMaterial,

    /// <summary>
    /// Aborted due to too excessive moves
    /// </summary>
    ExcessiveMoves,

    /// <summary>
    /// Draw by repetition.
    /// </summary>
    Repetition,

    /// <summary>
    /// Both players agree on advantage of one side of sufficient magnitude,
    /// or sustained draw-like score.
    /// </summary>
    AdjudicatedEvaluation,

    /// <summary>
    /// Player forfeited due to exceeding time/node limit.
    /// </summary>
    ForfeitLimits,
  }

  /// <summary>
  /// Record summarizing result of a tournament game.
  /// </summary>
  [Serializable]
  public record TournamentGameInfo
  {
    /// <summary>
    /// Engine with white pieces.
    /// </summary>
    public string PlayerWhite;

    /// <summary>
    /// Engine with black pieces.
    /// </summary>
    public string PlayerBlack;

    /// <summary>
    /// Search limit used by white player.
    /// </summary>
    public SearchLimit SearchLimitWhite;

    /// <summary>
    /// Search limit used by black player.
    /// </summary>
    public SearchLimit SearchLimitBlack;

    /// <summary>
    /// Sequence number within tournament.
    /// </summary>
    public int GameSequenceNum;

    /// <summary>
    /// Index of opening from opening book.
    /// </summary>
    public int OpeningIndex;

    /// <summary>
    /// If Engine2 is playing the white pieces.
    /// </summary>
    public bool Engine2IsWhite;

    /// <summary>
    /// Starting FEN of game.
    /// </summary>
    public string FEN;

    /// <summary>
    /// Result of game.
    /// </summary>
    public TournamentGameResult Result;

    /// <summary>
    /// Result of game as 1-0, 0-1 and draw.
    /// </summary>
    public string ResultString;

    public string Time;

    public float EndCP;

    public int WhiteNumberOfInstaMoves; // < 0.05
    public int BlackNumberOfInstaMoves; // < 0.05
    public int WhiteNumberOfInstaMovesLessThan10;
    public int BlackNumberOfInstaMovesLessThan10;
    public int WhiteNumberOfInstaMovesLessThan25;
    public int BlackNumberOfInstaMovesLessThan25;
    public int WhiteNumberOfInstaMovesLessThan50;
    public int BlackNumberOfInstaMovesLessThan50;
    public int BlackNumberOfInstaMovesLessThan1Sec;
    public int WhiteNumberOfInstaMovesLessThan1Sec;
    public int BlackNumberOfInstaMovesLessThan3Sec;
    public int WhiteNumberOfInstaMovesLessThan3Sec;

    /// <summary>
    /// Reason for game termination.
    /// </summary>
    public TournamentGameResultReason ResultReason;

    /// <summary>
    /// Number of ply in game.
    /// </summary>
    public int PlyCount;

    /// <summary>
    /// Total time used by engine 1 in seconds.
    /// </summary>
    public float TotalTimeEngine1;

    /// <summary>
    /// Total time used by engine 2 in seconds.
    /// </summary>
    public float TotalTimeEngine2;

    /// <summary>
    /// Remaining time on clock left used by engine 1 in seconds.
    /// </summary>
    public float RemainingTimeEngine1;

    /// <summary>
    /// Remaining time on clock left used by engine 2 in seconds.
    /// </summary>
    public float RemainingTimeEngine2;

    /// <summary>
    /// If engine 1 should have forfeited due to exceeding specified search limit at least once.
    /// </summary>
    public bool ShouldHaveForfeitedOnLimitsEngine1;

    /// <summary>
    /// If engine 2 should have forfeited due to exceeding specified search limit at least once.
    /// </summary>
    public bool ShouldHaveForfeitedOnLimitsEngine2;

    /// <summary>
    /// If check engine is in use, the number of moves for which engine 2 played a different move from check engine.
    /// </summary>
    public int NumEngine2MovesDifferentFromCheckEngine;

    /// <summary>
    /// Total number of nodes evaluated by engine 1.
    /// </summary>
    public long TotalNodesEngine1;

    /// <summary>
    /// Total number of nodes evaluated by engine 2.
    /// </summary>
    public long TotalNodesEngine2;

    /// <summary>
    /// Average score (in centipawns) across all moves.
    /// </summary>
    public float AvgScoreCentipawnsEngine1
    {
      get
      {
        float count = 0;
        float sum = 0;
        foreach (GameMoveStat stat in GameMoveHistory)
        {
          count++;
          if ((stat.Side == Chess.SideType.Black && Engine2IsWhite)
           || (stat.Side == Chess.SideType.White && !Engine2IsWhite))
          {
            sum += stat.ScoreCentipawns;
          }
        }
        return sum / count;
      }
    }

    /// <summary>
    /// List of descriptive information relating to all moves played.
    /// </summary>
    public List<GameMoveStat> GameMoveHistory;

    #region Limits aggressiveness

    public float TimeAggressivenessRatio(bool white)
    {
      const int MOVES_EACH_SIZE = 5;
      int midpointInconclusivePhase = PlyNumEndInconclusivePhase / 2;
      float firstQuartileN = AvgFinalNAtPly(white, (int)MathF.Round(0.25f * midpointInconclusivePhase), MOVES_EACH_SIZE);
      float thirdQuartileN = AvgFinalNAtPly(white, (int)MathF.Round(0.75f * midpointInconclusivePhase), MOVES_EACH_SIZE);

      return firstQuartileN / thirdQuartileN;
    }

    public float AvgFinalNAtPly(bool white, int plyNum, int numMovesEachSide)
    {
      // Must do 2x as many moves because each side only plays every other ply.
      int numPlyEachSide = numMovesEachSide * 2;
      List<float> n = new();

      for (int i = plyNum - numPlyEachSide; i <= plyNum + numPlyEachSide; i++)
      {
        if (i >= 0 && i < GameMoveHistory.Count)
        {
          if ((GameMoveHistory[i].Side == Chess.SideType.White) == white)
          {
            n.Add(GameMoveHistory[i].FinalN);
          }
        }
      }
      return StatUtils.Average(n.ToArray());
    }

    /// <summary>
    /// Returns the index of the first move which marked the beginning
    /// of the non-decisive phase of the game 
    /// (all subsequent moves were highly consistent with the final outcome).
    /// </summary>
    public int PlyNumEndInconclusivePhase
    {
      get
      {
        const float DRAW_THRESHOLD_CP = 10f;
        const float WIN_THRESHOLD_CP = 150f;

        bool wasDraw = Result == TournamentGameResult.Draw;

        for (int i = GameMoveHistory.Count - 1; i >= 0; i--)
        {
          GameMoveStat stat = GameMoveHistory[i];
          if (wasDraw && MathF.Abs(stat.ScoreCentipawns) > DRAW_THRESHOLD_CP)
          {
            return i;
          }
          else if (!wasDraw && MathF.Abs(stat.ScoreCentipawns) < WIN_THRESHOLD_CP)
          {
            return i;
          }
        }

        return 0;
      }
    }

    #endregion


    #region Charts

    public enum LimitsUsageChartType { Bar, Line };

    /// <summary>
    /// Returns a chart of the limits usage of two players throughout the game.
    /// </summary>
    /// <param name="chartType"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public PlotlyChart LimitsUsageChart(LimitsUsageChartType chartType)
    {
      string title = Result.ToString();

      // Populate the search limits (if available and per move).
      float? limitBase = null;
      float? limitIncrement = null;
      if (SearchLimitWhite.Equals(SearchLimitBlack) && SearchLimitWhite.IsPerGameLimit)
      {
        limitBase = SearchLimitWhite.Value;
        limitIncrement = SearchLimitWhite.ValueIncrement;
      };

      // Build lists of used search values.
      bool isTime = SearchLimitWhite.IsTimeLimit;
      List<float> limitsWhite = new();
      List<float> limitsBlack = new();
      foreach (GameMoveStat moveStat in GameMoveHistory)
      {
        if (moveStat.Side == Chess.SideType.White)
        {
          limitsWhite.Add(isTime ? moveStat.TimeElapsed : moveStat.NumNodesComputed);
        }
        else
        {
          limitsBlack.Add(isTime ? moveStat.TimeElapsed : moveStat.NumNodesComputed);
        }
      }

      return chartType switch
      {
        LimitsUsageChartType.Bar => PlayerLimitsUsageCharts.BarChart(title, PlayerWhite, PlayerBlack, limitsWhite.ToArray(), limitsBlack.ToArray()),
        LimitsUsageChartType.Line => PlayerLimitsUsageCharts.LineChart(title, PlayerWhite, PlayerBlack, limitsWhite.ToArray(), limitsBlack.ToArray(), limitBase, limitIncrement),
        _ => throw new NotImplementedException(chartType.ToString())
      };

    }

    public PlotData LimitsUsageChartData(LimitsUsageChartType chartType)
    {
      string title = Result.ToString();

      // Populate the search limits (if available and per move).
      float? limitBase = null;
      float? limitIncrement = null;
      if (SearchLimitWhite.Equals(SearchLimitBlack) && SearchLimitWhite.IsPerGameLimit)
      {
        limitBase = SearchLimitWhite.Value;
        limitIncrement = SearchLimitWhite.ValueIncrement;
      };

      // Build lists of used search values.
      bool isTime = SearchLimitWhite.IsTimeLimit;
      List<double> limitsWhite = new();
      List<double> limitsBlack = new();
      foreach (GameMoveStat moveStat in GameMoveHistory)
      {
        if (moveStat.Side == Chess.SideType.White)
        {
          limitsWhite.Add(isTime ? moveStat.TimeElapsed : moveStat.NumNodesComputed);
        }
        else
        {
          limitsBlack.Add(isTime ? moveStat.TimeElapsed : moveStat.NumNodesComputed);
        }
      }

      return chartType switch
      {
        LimitsUsageChartType.Bar => PlayerLimitsUsageCharts.BarChartData(title, PlayerWhite, PlayerBlack, limitsWhite.ToArray(), limitsBlack.ToArray()),
        LimitsUsageChartType.Line => PlayerLimitsUsageCharts.LineChartData(title, PlayerWhite, PlayerBlack, limitsWhite.ToArray(), limitsBlack.ToArray(), limitBase, limitIncrement),
        _ => throw new NotImplementedException(chartType.ToString())
      };

    }

    #endregion


  }
}
