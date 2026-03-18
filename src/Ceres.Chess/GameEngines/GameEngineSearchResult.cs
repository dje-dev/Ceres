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
using Ceres.Chess.SearchResultVerboseMoveInfo;
using System;
using System.Collections.Generic;

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
    /// Engine's reported EPS (neural network evaluator nodes per move).
    /// </summary>
    public int EPS;

    /// <summary>
    /// Time required for search.
    /// </summary>
    public TimingStats TimingStats;

    /// <summary>
    /// Optional full detailed verbose move statistics.
    /// </summary>
    public List<VerboseMoveStat> VerboseMoveStats;

    /// <summary>
    /// Number of nodes in tree when the eventual top-N move was first chosen.
    /// </summary>
    public int NumNodesWhenChoseTopNNode;

    /// <summary>
    /// Number of neural network evaluation batches during search.
    /// </summary>
    public int NumNNBatches;

    /// <summary>
    /// Number of neural network node evaluations during search.
    /// </summary>
    public int NumNNNodes;

    /// <summary>
    /// Visit count of the top-N child at end of search.
    /// </summary>
    public int TopNNodeN;

    /// <summary>
    /// Fraction of total nodes used when the top-N move was chosen.
    /// </summary>
    public float FractionNumNodesWhenChoseTopNNode;

    /// <summary>
    /// Average depth of nodes in the search tree.
    /// </summary>
    public float AvgDepth;

    /// <summary>
    /// Maximum depth reached in the search tree.
    /// </summary>
    public float MaxDepth;

    /// <summary>
    /// Fraction of node selection attempts that yielded a usable node.
    /// </summary>
    public float NodeSelectionYieldFrac;

    /// <summary>
    /// Indicator if the best move was not the top-N move ("!" if non-top-N, " " otherwise).
    /// </summary>
    public string PickedNonTopNMoveStr;

    #endregion

    public GameEngineSearchResult(string moveString, float scoreQ, float scoreCentipawns, float mAvg, 
                                  SearchLimit searchLimit, TimingStats timingStats, 
                                  int startingN, int finalN, int nps, int eps, int depth,
                                  List<VerboseMoveStat> verboseMoveStats = null)
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
      EPS = eps;
      Depth = depth;
      VerboseMoveStats = verboseMoveStats;
    }


    public override string ToString()
    {
      return $"<GameEngineSearchResult BestMove={MoveString} ScoreQ={ScoreQ,6:F2} ScoreCP={ScoreCentipawns,6:F2} " 
           + $"in {TimingStats.ElapsedTimeSecs}sec "
           + $"N={FinalN} Depth={Depth}>";
    }
  }
}
