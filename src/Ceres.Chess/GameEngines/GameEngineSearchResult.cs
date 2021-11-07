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
    
    /// <summary>
    /// String representation of the selected best move.
    /// </summary>
    public string MoveString;

    /// <summary>
    /// Score of best move (win - loss).
    /// </summary>
    public float ScoreQ;

    /// <summary>
    /// Score of best move (centipawns).
    /// </summary>
    public float ScoreCentipawns;

    /// <summary>
    /// The N of the tree at beginning of search.
    /// </summary>
    public int StartingN;

    /// <summary>
    /// The N of the tree at end of search.
    /// </summary>
    public int FinalN;

    /// <summary>
    /// Number of visits made during search.
    /// </summary>
    public int Visits => FinalN - StartingN;

    /// <summary>
    /// Average depth of search tree.
    /// </summary>
    public int Depth;

    /// <summary>
    /// Average MLH head value at root.
    /// </summary>
    public float MAvg;

    /// <summary>
    /// The SearchLimit used for the search.
    /// </summary>
    public SearchLimit Limit;

    /// <summary>
    /// Engine's reported NPS (nodes per move).
    /// </summary>
    public int NPS;

    /// <summary>
    /// Time required for search.
    /// </summary>
    public TimingStats TimingStats;

    #endregion

    public GameEngineSearchResult(string moveString, float scoreQ, float scoreCentipawns, float mAvg, 
                                  SearchLimit searchLimit, TimingStats timingStats, 
                                  int startingN, int finalN, int nps, int depth)
    {
      MoveString = moveString ?? throw new ArgumentNullException(nameof(moveString));
      ScoreQ = scoreQ;
      ScoreCentipawns = scoreCentipawns;
      MAvg = mAvg;
      Limit = searchLimit;
      TimingStats = timingStats;
      StartingN = startingN;
      FinalN = finalN;
      NPS = nps;
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
