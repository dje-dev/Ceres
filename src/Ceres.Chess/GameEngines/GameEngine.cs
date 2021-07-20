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

using Ceres.Base.Benchmarking;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.Positions;

using System;
using System.Collections.Generic;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Abstract base class for game engines which are capable of 
  /// evaluating chess positions and choosing their best moves.
  /// </summary>
  public abstract class GameEngine : IDisposable
  {
    /// <summary>
    /// Identifying string of the engine.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Delegate type that may be called (potentially many times) during search,
    /// allowing updates to progress or other actions.
    /// </summary>
    /// <param name="context"></param>
    public delegate void ProgressCallback(object context);

    /// <summary>
    /// Optional opponent engine.
    /// </summary>
    public GameEngine OpponentEngine { get; set; }

    /// <summary>
    /// Total search time in seconds.
    /// </summary>
    public float CumulativeSearchTimeSeconds = 0;

    /// <summary>
    /// Total number of nodes summed across across all search trees.
    /// </summary>
    public int CumulativeNodes = 0;


    protected DateTime lastSearchStartTime;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    public GameEngine(string id)
    {
      ID = id;
    }


    /// <summary>
    /// Resets all state between games.
    /// </summary>
    /// <param name="gameID">optional game descriptive string</param>
    public abstract void ResetGame(string gameID = null);

    volatile bool inSearch = false;

    /// <summary>
    /// Runs a search, calling DoSearch and adjusting the cumulative search time
    /// </summary>
    /// <param name="curPositionAndMoves"></param>
    /// <param name="searchLimit"></param>
    /// <param name="callback"></param>
    /// <returns></returns>
    public GameEngineSearchResult Search(PositionWithHistory curPositionAndMoves, 
                                         SearchLimit searchLimit, 
                                         List<GameMoveStat> gameMoveHistory = null, 
                                         ProgressCallback callback = null,
                                         bool verbose = false)
    {
      if (inSearch)
      {
        throw new Exception("GameEngine.Search cannot be called concurrently by more than one thread.");
      }

      inSearch = true;

      // Execute any preparation which should not be counted against thinking time
      // For example, Stockfish can require hundreds of milliseconds to process "ucinewgame"
      // which is used to reset state/hash table when the tree reuse option is enabled.
      DoSearchPrepare();

      // Officially start timing now.
      TimingStats stats = new TimingStats();
      lastSearchStartTime = DateTime.Now;
      GameEngineSearchResult result;
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        result = DoSearch(curPositionAndMoves, searchLimit, gameMoveHistory, callback, verbose);
      }

      CumulativeSearchTimeSeconds += (float)stats.ElapsedTimeSecs;
      CumulativeNodes += result.FinalN;

      result.TimingStats = stats;

      inSearch = false;
      return result;
    }

    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public abstract bool SupportsNodesPerGameMode { get; }


    /// <summary>
    /// Executes any preparatory steps (that should not be counted in thinking time) before a search.
    /// </summary>
    protected abstract void DoSearchPrepare();
    protected abstract GameEngineSearchResult DoSearch(PositionWithHistory curPositionAndMoves, 
                                                       SearchLimit searchLimit,
                                                       List<GameMoveStat> gameMoveHistory, 
                                                       ProgressCallback callback,
                                                       bool verbose);

    public virtual void DumpMoveHistory(List<GameMoveStat> gameMoveHistory, SideType? side = null)
    {

    }

    readonly object dumpLockObj = new();
    public void DumpFullMoveHistory(List<GameMoveStat> gameMoveHistory, bool weAreWhite)
    {
      lock (dumpLockObj)
      {
        Console.WriteLine();
        Console.WriteLine("Dump game from our perspective");
        DumpMoveHistory(gameMoveHistory, weAreWhite ? SideType.White : SideType.Black);

        Console.WriteLine();
        Console.WriteLine("Dump game from opponent perspective");
        DumpMoveHistory(gameMoveHistory, weAreWhite ? SideType.Black : SideType.White);
      }
    }

    /// <summary>
    /// Returns UCI search information 
    /// (such as would appear in a chess GUI describing search progress) 
    /// based on last state of search.
    /// </summary>
    public abstract UCISearchInfo UCIInfo { get; }


    public abstract void Dispose();


    /// <summary>
    /// Attepts to perform preliminary initialization of engine.
    /// </summary>
    /// <param name="knownMaxNumNodes">optional hint indicating known maximum number of nodes by any search</param>
    public virtual void Warmup(int? knownMaxNumNodes = null)
    {
      const int NUM_WARMUP_NODES = 1;
      Search(PositionWithHistory.StartPosition, SearchLimit.NodesPerMove(NUM_WARMUP_NODES));
      ResetGame();
    }

  }
}
