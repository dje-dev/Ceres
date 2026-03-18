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

using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Base.Benchmarking;

using Ceres.MCGS.Managers;
using Ceres.MCGS.Search;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Search.Coordination;


#endregion

namespace Ceres.MCGS.GameEngines;

public readonly record struct GameEngineSearchResultsStats // TODO: move this into Ceres project and GameEngine class, support in MCTS engine
{
  public float RatioVisitsToNodes { get; init; }
}

public record GameEngineSearchResultCeresMCGS : GameEngineSearchResult
{
  public MCGSEngine Engine => Search.Manager.Engine;

  /// <summary>
  /// Coordinator used to generate this result.
  /// </summary>
  public MCGSSearch Search { get; private set; }

  /// <summary>
  /// Information about the best (selected move) from the search.
  /// </summary>
  public readonly BestMoveInfoMCGS BestMoveInfo;

  public MGMove BestMoveMG { get; private set; }

  public float ScoreQRoot { get; private set; }

  public GameEngineSearchResultsStats Stats;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="search"></param>
  /// <param name="moveString"></param>
  /// <param name="scoreQ"></param>
  /// <param name="scoreCentipawns"></param>
  /// <param name="mAvg"></param>
  /// <param name="searchLimit"></param>
  /// <param name="timingStats"></param>
  /// <param name="startingN"></param>
  /// <param name="endingN"></param>
  /// <param name="depth"></param>
  /// <param name="bestMoveInfo"></param>
  public GameEngineSearchResultCeresMCGS(MCGSSearch search,
                                         string moveString, MGMove bestMoveMG, 
                                         float scoreQRoot, float scoreQBestMove, 
                                         float scoreCentipawns, float mAvg,
                                         SearchLimit searchLimit, TimingStats timingStats,
                                         int startingN, int endingN, 
                                         int eps, int depth, 
                                         BestMoveInfoMCGS bestMoveInfo, 
                                         float ratioVisitsToNodes)
    : base(moveString, scoreQBestMove, scoreCentipawns, mAvg, searchLimit, timingStats,
           startingN, endingN, (int)MathF.Round((endingN - startingN) / (float)timingStats.ElapsedTimeSecs, 1), eps, depth)
  {
    if (moveString == "(none)")
    {
      throw new Exception("Engine returned null move");
    }

    Search = search;
    BestMoveMG = bestMoveMG;
    BestMoveInfo = bestMoveInfo;
    ScoreQRoot = scoreQRoot;
    Stats = new GameEngineSearchResultsStats() with { RatioVisitsToNodes = ratioVisitsToNodes };
  }
}
