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
using System;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Summary information about the the results of a search by a GameEngine.
  /// </summary>
  public record GameEngineSearchResult
  {
    #region Basic Values

    public string MoveString;
    public float ScoreQ;
    public float ScoreCentipawns;
    public int StartingN;
    public int FinalN;
    public int Depth;
    public float MAvg;
    public SearchLimit Limit;
    public TimingStats TimingStats;

    #endregion

    public GameEngineSearchResult(string moveString, float scoreQ, float scoreCentipawns, float mAvg, 
                                  SearchLimit searchLimit, TimingStats timingStats, 
                                  int startingN, int finalN, int depth)
    {
      MoveString = moveString ?? throw new ArgumentNullException(nameof(moveString));
      ScoreCentipawns = scoreCentipawns;
      ScoreQ = scoreQ;
      ScoreCentipawns = scoreCentipawns;
      MAvg = mAvg;
      Limit = searchLimit;
      TimingStats = timingStats;
      StartingN = startingN;
      FinalN = finalN;
      Depth = depth;
    }

    public override string ToString()
    {
      return $"<GameEngineSearchResult BestMove={MoveString} ScoreQ={ScoreQ,6:F2} ScoreCP={ScoreCentipawns,6:F2} " 
           + $"in {TimingStats.ElapsedTimeSecs}sec "
           + $"N={FinalN} Depth={Depth}>";
    }
  }
}
