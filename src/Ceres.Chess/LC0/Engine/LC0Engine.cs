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

using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.LC0VerboseMoves;
using Ceres.Chess.Positions;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Chess.LC0.Engine
{
  /// <summary>
  /// Manages launching, issuing commands, and reading output from 
  /// an Leela Chess Zero (LC0) executable running in UCI mode.
  /// </summary>
  public class LC0Engine : IDisposable
  {
    /// <summary>
    /// Underlying UCI game runnter with connection to running engine process.
    /// </summary>
    public UCIGameRunner Runner;

    /// <summary>
    /// Returns the process ID of the LC0 engine process.
    /// </summary>
    public int ProcessID => Runner.EngineProcessID;


    /// <summary>
    /// Constructor given an executable and specified command line options.
    /// </summary>
    /// <param name="exePath"></param>
    /// <param name="options"></param>
    /// <param name="resetStateAndCachesBeforeMoves"></param>
    public LC0Engine(string exePath, string options, bool resetStateAndCachesBeforeMoves)
    {
      Runner = new UCIGameRunner(exePath, resetStateAndCachesBeforeMoves, options, null);
    }

  
    /// <summary>
    /// Executes any preparatory steps (that should not be counted in thinking time) before a search.
    /// </summary>
    public void DoSearchPrepare()
    {
      Runner.EvalPositionPrepare();
    }


    /// <summary>
    /// Analyzes a specified position until a specified limit is exhausted.
    /// 
    /// TODO: This method is highly redundant (and inferior to?) the next method AnalyzePositionFromFENAndMoves, delete it.
    /// </summary>
    /// <param name="fenOrFENAndMoves">a FEN</param>
    /// <param name="nodes"></param>
    /// <returns></returns>
    public UCISearchInfo AnalyzePositionFromFEN(string fenAndMovesString, SearchLimit searchLimit)
    {
      List<LC0VerboseMoveStat> moves = new List<LC0VerboseMoveStat>();
      Runner.EvalPositionPrepare();

      UCISearchInfo searchInfo;
      switch (searchLimit.Type)
      {
        case SearchLimitType.SecondsPerMove:
          searchInfo = Runner.EvalPositionToMovetime(fenAndMovesString, (int)(searchLimit.Value * 1000.0f));
          break;
        case SearchLimitType.NodesPerMove:
          searchInfo = Runner.EvalPositionToNodes(fenAndMovesString, (int)searchLimit.Value);
          break;
        case SearchLimitType.NodesPerTree:
          throw new NotImplementedException("NodesPerTree not currently supported for LC0 engines");

        default:
          throw new Exception("Unknown search limit " + searchLimit.Type);
      }

      double elapsed = searchInfo.EngineReportedSearchTime / 1000.0f;

      // no more, we now assume  win_percentages is requested     LeelaVerboseMoveStats ret = new LeelaVerboseMoveStats(positionEnd, searchInfo.BestMove, elapsed, searchInfo.Nodes, LZPositionEvalLogistic.CentipawnToLogistic2018(searchInfo.Score));

      PositionWithHistory pwh = PositionWithHistory.FromFENAndMovesUCI(fenAndMovesString);
      LC0VerboseMoveStats ret = new (pwh.FinalPosition, searchInfo.BestMove, elapsed, searchInfo.Nodes, searchInfo.ScoreCentipawns, searchInfo);

      foreach (string info in searchInfo.Infos)
      {
        if (info.Contains("P:"))
        {
          moves.Add(new LC0VerboseMoveStat(ret, info));
        }
      }

      ret.SetMoves(moves);

      // TODO: Someday perhaps make LeelaVerboseMoveStats a subclass of UCISearchInfo so this is more elegant
      UCISearchInfo uciInfo = new UCISearchInfo(null, ret.BestMove, null);
      uciInfo.Nodes = ret.NumNodes;
      uciInfo.EngineReportedSearchTime = (int)(1000.0f * ret.ElapsedTime);
      uciInfo.ExtraInfo = ret;
      uciInfo.BestMove = ret.BestMove;

      return uciInfo;
    }

    public LC0VerboseMoveStats LastAnalyzedPositionStats;


    /// <summary>
    /// Analyzes a position until a specified search limit is exhausted.
    /// </summary>
    /// <param name="fenAndMovesStr"></param>
    /// <param name="searchLimit"></param>
    /// <returns></returns>
    public LC0VerboseMoveStats AnalyzePositionFromFENAndMoves(string fenAndMovesStr, SearchLimit searchLimit)
    {
      List<LC0VerboseMoveStat> moves = new List<LC0VerboseMoveStat>();
      PositionWithHistory pwh = PositionWithHistory.FromFENAndMovesUCI(fenAndMovesStr);

      UCISearchInfo searchInfo;

      int searchValueMilliseconds = (int)((float)searchLimit.Value * 1000.0f);
      switch (searchLimit.Type)
      {
        case SearchLimitType.NodesPerMove:
          searchInfo = Runner.EvalPositionToNodes(fenAndMovesStr, (int)searchLimit.Value);
          break;

        case SearchLimitType.NodesPerTree:
          // TODO: someday this could be supported,
          //       omit "--nodes-as-playouts" option.
          throw new NotImplementedException("NodesPerTree not currently supported for LC0 engines");

        case SearchLimitType.SecondsPerMove:
          searchInfo = Runner.EvalPositionToMovetime(fenAndMovesStr, searchValueMilliseconds);
          break;

        case SearchLimitType.NodesForAllMoves:
          throw new Exception("NodesForAllMoves not supported for Leela Chess Zero");

        case SearchLimitType.SecondsForAllMoves:
          bool weAreWhite = pwh.FinalPosition.MiscInfo.SideToMove == SideType.White;

          searchInfo = Runner.EvalPositionRemainingTime(fenAndMovesStr,
                                                        weAreWhite,
                                                        searchLimit.MaxMovesToGo,
                                                        (int)(searchLimit.Value * 1000),
                                                        (int)(searchLimit.ValueIncrement * 1000));
          break;

        default:
          throw new Exception($"Unknown SeachLimit.Type {searchLimit.Type}");
      }

      double elapsed = 0;//engine.EngineProcess.TotalProcessorTime.TotalSeconds - startTime;
      LC0VerboseMoveStats ret = new(pwh.FinalPosition, searchInfo.BestMove, elapsed, searchInfo.Nodes, searchInfo.ScoreCentipawns, searchInfo);

      searchInfo.Infos.Reverse();
      foreach (string info in searchInfo.Infos)
      {
        if (info.Contains("P:"))
        {
          moves.Add(new LC0VerboseMoveStat(ret, info));
        }
      }

      ret.SetMoves(moves);

      return LastAnalyzedPositionStats = ret;
    }

    public void Dispose()
    {
      Runner.Shutdown();
      Runner = null;
    }
  }
}
