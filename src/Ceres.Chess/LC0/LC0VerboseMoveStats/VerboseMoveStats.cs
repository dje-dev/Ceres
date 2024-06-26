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

using Ceres.Chess.ExternalPrograms.UCI;
using System;
using System.Collections.Generic;
using System.Linq;

#endregion

namespace Ceres.Chess.SearchResultVerboseMoveInfo
{
  /// <summary>
  /// Detailed information related to a search result from an MCTS engine
  /// including detail on top-level move choices and their evaluation
  /// (such as output by LC0  with verbose-move-stats enabled).
  /// </summary>
  public class VerboseMoveStats
  {
    /// <summary>
    /// Position at the root of the search.
    /// </summary>
    public readonly Position Position;

    /// <summary>
    /// Number of elapsed seconds during search.
    /// </summary>
    public readonly double ElapsedTime;

    /// <summary>
    /// Selected move.
    /// </summary>
    public readonly string BestMove;

    /// <summary>
    /// Number of visits to leaf nodes.
    /// </summary>
    public readonly ulong NumNodes;

    /// <summary>
    /// Final position evaluation in centipawns.
    /// </summary>
    public readonly float ScoreCentipawns;

    /// <summary>
    /// Results as a UCISearchInfo object.
    /// </summary>
    public UCISearchInfo UCIInfo;

    /// <summary>
    /// List of all moves in position with detailed information.
    /// </summary>
    public List<VerboseMoveStat> Moves;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="bestMove"></param>
    /// <param name="processorTime"></param>
    /// <param name="numNodes"></param>
    /// <param name="scoreCentipawns"></param>
    /// <param name="uciSearchInfo"></param>
    public VerboseMoveStats(Position position, string bestMove, 
                           double processorTime, ulong numNodes, float scoreCentipawns, 
                           UCISearchInfo uciSearchInfo)
    {
      Position = position;
      ElapsedTime = processorTime;
      BestMove = bestMove;
      NumNodes = numNodes;
      ScoreCentipawns = scoreCentipawns;
      UCIInfo = uciSearchInfo;
    }


    /// <summary>
    /// Sets the moves array.
    /// </summary>
    /// <param name="moves"></param>
    internal void SetMoves(List<VerboseMoveStat> moves)
    {
      Moves = moves.OrderBy(o => -o.VisitCount).ToList();
    }


    /// <summary>
    /// Returns the move having the largest N.
    /// </summary>
    public VerboseMoveStat MaxNMove => MaxMoveByMetric(m => m.VisitCount);


    /// <summary>
    /// Returns the move having the largest P (policy probability).
    /// </summary>
    public VerboseMoveStat MaxPMove => MaxMoveByMetric(m => m.P);


    /// <summary>
    /// Returns the mvoe having the largest Q.
    /// </summary>
    public VerboseMoveStat MaxQMove => MaxMoveByMetric(m => m.Q.LogisticValue);


    /// <summary>
    /// Returns the move statistics relating to a specified move.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public VerboseMoveStat Move(Move move)
    {
      foreach (VerboseMoveStat stat in Moves)
      {
        if (stat.Move == move)
        {
          return stat;
        }
      }

      throw new Exception("No such move " + move);
    }


    /// <summary>
    /// Returns the move having a specified neural network evaluation index.
    /// </summary>
    /// <param name="moveCode"></param>
    /// <returns></returns>
    public VerboseMoveStat MoveByCode(int moveCode)
    {
      foreach (VerboseMoveStat stat in Moves)
      {
        if (stat.MoveCode == moveCode)
        {
          return stat;
        }
      }

      throw new Exception("No such move " + moveCode);
    }


    /// <summary>
    /// Returns the move having largest value according to particular metric (specified via a Func).
    /// </summary>
    /// <param name="metric"></param>
    /// <returns></returns>
    VerboseMoveStat MaxMoveByMetric(Func<VerboseMoveStat, double> metric)
    {
      double best = int.MinValue;
      VerboseMoveStat bestStat = default;
      foreach (VerboseMoveStat stat in Moves)
      {
        if (metric(stat) > best)
        {
          bestStat = stat;
          best = metric(stat);
        }
      }

      return bestStat == default(VerboseMoveStat) ? throw new Exception("Internal error ,no moves?") : bestStat;
    }


    /// <summary>
    /// Returns the move having a speciifed string representation.
    /// </summary>
    /// <param name="moveStr"></param>
    /// <returns></returns>
    public VerboseMoveStat MoveByString(string moveStr)
    {
      foreach (VerboseMoveStat stat in Moves)
      {
        if (stat.MoveString == moveStr)
        {
          return stat;
        }
      }
      throw new Exception("Move not found " + moveStr);
    }

    /// <summary>
    /// Indexer returning the move having specified string representation.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public VerboseMoveStat this[string move]
    {
      get
      {
        foreach (VerboseMoveStat stat in Moves)
        {
          if (stat.MoveString == move)
          {
            return stat;
          }
        }

        throw new Exception("Move not found " + move);
      }
    }


    /// <summary>
    /// Dumps full information relating to all moves.
    /// </summary>
    public void Dump()
    {
      foreach (VerboseMoveStat mi in Moves)
      {
        Console.WriteLine(mi.ToString());
      }
    }

  }
}
