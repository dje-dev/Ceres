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
using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Managers.Limits
{
  /// <summary>
  /// Inputs to the TimeManager, used to determine optimal time to spend on next move.
  /// </summary>
  public record ManagerGameLimitInputs
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="startPos"></param>
    /// <param name="searchParams"></param>
    /// <param name="priorMoveStats"></param>
    /// <param name="limitType"></param>
    /// <param name="rootN"></param>
    /// <param name="rootQ"></param>
    /// <param name="timeRemainingFixedSelf"></param>
    /// <param name="timeRemainingIncrementSelf"></param>
    /// <param name="maxTreeNodesSelf"></param>
    /// <param name="timeRemainingFixedOpponent"></param>
    /// <param name="timeRemainingIncrementOpponent"></param>
    /// <param name="maxMovesToGo"></param>
    /// <param name="isFirstMoveOfGame"></param>
    public ManagerGameLimitInputs(in Position startPos, ParamsSearch searchParams,
                                 List<GameMoveStat> priorMoveStats,
                                 SearchLimitType limitType,
                                 int rootN,
                                 float rootQ,
                                 float timeRemainingFixedSelf, float timeRemainingIncrementSelf,
                                 int? maxTreeNodesSelf,
                                 int? maxTreeVisitsSelf,
                                 float timeRemainingFixedOpponent, float timeRemainingIncrementOpponent,
                                 int? maxMovesToGo = null,
                                 bool isFirstMoveOfGame = false)
    {
      StartPos = startPos;
      SearchParams = searchParams;
      PriorMoveStats = priorMoveStats;
      TargetLimitType = limitType;
      RootN = rootN;
      RootQ = rootQ;
      RemainingFixedSelf = MathF.Max(0.001f, timeRemainingFixedSelf);
      IncrementSelf = timeRemainingIncrementSelf;
      MaxTreeNodesSelf = maxTreeNodesSelf;
      MaxTreeVisitsSelf = maxTreeVisitsSelf;
      RemainingFixedOpponent = MathF.Max(0.001f, timeRemainingFixedOpponent);
      IncrementOpponent = timeRemainingIncrementOpponent;
      MaxMovesToGo = maxMovesToGo;
      IsFirstMoveOfGame = isFirstMoveOfGame;
    }


    /// <summary>
    /// Starting position for this search.
    /// </summary>
    public readonly Position StartPos;

    /// <summary>
    /// The ParamsSearch to be used for this search.
    /// </summary>
    public readonly ParamsSearch SearchParams;

    /// <summary>
    /// Statistics relating to prior moves.
    /// </summary>
    public readonly List<GameMoveStat> PriorMoveStats;

    /// <summary>
    /// Type of limit (either NodesPerMove or SecondsPerMove).
    /// </summary>
    public readonly SearchLimitType TargetLimitType;

    /// <summary>
    /// The N at the start of search/
    /// </summary>
    public readonly int RootN;

    /// <summary>
    /// The Q value of the root node at the beginning of search 
    /// (if this is a continuation, else NaN).
    /// </summary>
    public readonly float RootQ;

    /// <summary>
    /// Amount of search units (time or nodes) remaining until end of game.
    /// </summary>
    public readonly float RemainingFixedSelf;

    /// <summary>
    /// Amount of incremental search units (time or nodes) which will be added to allowance upon each move.
    /// </summary>
    public readonly float IncrementSelf;

    /// <summary>
    /// Maximum size in nodes which the search tree is allowed to grow.
    /// </summary>
    public readonly int? MaxTreeNodesSelf;

    /// <summary>
    /// Maximum size in visits which the search tree is allowed to grow.
    /// </summary>
    public readonly int? MaxTreeVisitsSelf;

    /// <summary>
    /// Amount of remaining search units (time or nodes) until end of game.
    /// </summary>
    public readonly float RemainingFixedOpponent;

    /// <summary>
    /// Amount of incremental v which will be added to allowance upon each move.
    /// </summary>
    public readonly float IncrementOpponent;

    /// <summary>
    /// Optional maximum number of moves that must be made in limits.
    /// </summary>
    public int? MaxMovesToGo;

    /// <summary>
    /// If move is part of a game and this is the first move of that game.
    /// </summary>
    public readonly bool IsFirstMoveOfGame;

    /// <summary>
    /// Flag for testing/diagnostic purposes to indicate use of test mode.
    /// </summary>
    public bool TestMode;

    #region Utility methods

    /// <summary>
    /// Dumps description of inputs to a Textwriter.
    /// </summary>
    /// <param name="writer"></param>
    public void Dump(TextWriter writer)
    {
      writer.WriteLine($"  PieceCount                  {StartPos.PieceCount,14:N0}");
      writer.WriteLine($"  TargetLimitType             {TargetLimitType,14}");
      writer.WriteLine($"  RemainingFixedSelf          {RemainingFixedSelf,14:F2}");//  Opponent: {RemainingFixedOpponent,8:F2}");
      writer.WriteLine($"  IncrementSelf               {IncrementSelf,14:F2}");//       Opponent: { IncrementOpponent,8:F2}");
      writer.WriteLine($"  MaxMovesToGo                {MaxMovesToGo,14:F2}");
      writer.WriteLine($"  MaxTreeNodesSelf            {MaxTreeNodesSelf,14:N0}");
      writer.WriteLine($"  MaxTreeVisitsSelf           {MaxTreeVisitsSelf,14:N0}");
      writer.WriteLine($"  TrailingAvgFinalNodes       {TrailingAvgFinalNodes(3, StartPos.SideToMove),14:N0}");
      writer.WriteLine($"  TrailingAvgNPS              {TrailingAvgNPS(3, StartPos.SideToMove),14:N0}");
      writer.WriteLine();
    }


    /// <summary>
    /// Returns trailing average nodes per second (NPS).
    /// </summary>
    /// <param name="numPrior"></param>
    /// <param name="sideToMove"></param>
    /// <returns></returns>
    public float TrailingAvgNPS(int numPrior, SideType sideToMove)
    {
      return TrailingAvg(numPrior, sideToMove, m => m.NumNodesComputed > 10, m => m.NodesPerSecond);
    }

    /// <summary>
    /// Returns trailing average number of nodes.
    /// </summary>
    /// <param name="numPrior"></param>
    /// <param name="sideToMove"></param>
    /// <returns></returns>
    public float TrailingAvgFinalNodes(int numPrior, SideType sideToMove)
    {
      return TrailingAvg(numPrior, sideToMove, m => m.FinalN > 0, m => m.FinalN);
    }

    /// <summary>
    /// Returns the average of some statistic over some trailing window.
    /// </summary>
    /// <param name="numPrior"></param>
    /// <param name="sideToMove"></param>
    /// <param name="filterFunc"></param>
    /// <param name="metricFunc"></param>
    /// <returns></returns>
    float TrailingAvg(int numPrior, SideType sideToMove, Predicate<GameMoveStat> filterFunc, Func<GameMoveStat, float> metricFunc)
    {
      if (PriorMoveStats == null) return float.NaN;

      int count = 0;
      float acc = 0;
      for (int i = PriorMoveStats.Count - 1; i >= 0; i--)
      {
        GameMoveStat stat = PriorMoveStats[i];
        if (stat.Side == sideToMove)
        {
          if (filterFunc(stat))
          {
            acc += metricFunc(stat);
            count++;

            if (count == numPrior) break;
          }
        }
      }

      if (count > 0)
        return acc / count;
      else
        return float.NaN;
    }

    /// <summary>
    /// Dumps information about these niputs to the console as a single line.
    /// </summary>
    /// <param name="side"></param>
    public void Dump(SideType? side)
    {
      Console.WriteLine($"{"STARTING N " + RootN} ");
      float opponentTime = float.NaN;
      foreach (GameMoveStat move in PriorMoveStats)
      {
        if (side == null || move.Side == side)
        {
          Console.WriteLine($" {move.ClockSecondsAlreadyConsumed,6:F2}   "
                          + $"Allot= {move.SearchLimit,6:F2} [Diff= {move.SearchLimit.Value - opponentTime,6:F2}]  Used= {move.TimeElapsed,6:F2}  Q= {move.ScoreQ,5:F2} "
                          + $"MAvg={move.MAvg,6:F0} #Pc={move.NumPieces}  {move.PlyNum,5} { move.Side,10} StartN= { move.StartN,10} "
                          + $"ComputeN ={move.NumNodesComputed,9}  EndN={move.FinalN,10}  NPS={move.NodesPerSecond,8:F0} ");
        }
        else
          opponentTime = move.TimeElapsed;
      }
    }

    #endregion
  }


}
