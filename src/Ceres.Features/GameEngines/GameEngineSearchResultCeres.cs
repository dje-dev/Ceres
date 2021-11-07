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

using Ceres.Base;
using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using System;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Sublcass of GameEngineSearchResult specialized for Ceres engine.
  /// </summary>
  public record GameEngineSearchResultCeres : GameEngineSearchResult
  {
    /// <summary>
    /// MCTSearch used to generate this result.
    /// </summary>
    public readonly MCTSearch Search;

    /// <summary>
    /// Information about the best (selected move) from teh search/
    /// </summary>
    public readonly BestMoveInfo BestMove;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="moveString"></param>
    /// <param name="scoreQ"></param>
    /// <param name="scoreCentipawns"></param>
    /// <param name="mAvg"></param>
    /// <param name="searchLimit"></param>
    /// <param name="timingStats"></param>
    /// <param name="startingN"></param>
    /// <param name="endingN"></param>
    /// <param name="nps"></param>
    /// <param name="depth"></param>
    /// <param name="search"></param>
    /// <param name="bestMoveInfo"></param>
    public GameEngineSearchResultCeres(string moveString, float scoreQ, float scoreCentipawns, float mAvg,
                                       SearchLimit searchLimit, TimingStats timingStats,
                                       int startingN, int endingN, int depth, MCTSearch search,
                                       BestMoveInfo bestMove)
      : base(moveString, scoreQ, scoreCentipawns, mAvg, searchLimit, timingStats,
             startingN, endingN, (int)MathF.Round((endingN - startingN) / (float)timingStats.ElapsedTimeSecs, 1), depth)
    {
      Search = search;
      BestMove = bestMove;
    }
  }
}

