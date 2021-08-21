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
using Ceres.Base.Environment;
using Ceres.Base.Benchmarking;

using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.PositionEvalCaching;

using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;
using Ceres.MCTS.Environment;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Entry point for launching a search and capturing the results.

  /// Note that in most situations instead the class GameEngineCeresInProcess 
  /// is preferrable as the entry point for launching searches.
  /// </summary>
  public partial class MCTSearch
  {
    /// <summary>
    /// The underlying serach manager.
    /// </summary>
    public MCTSManager Manager { get; private set; }

    /// <summary>
    /// Selected best move from last search.
    /// </summary>
    public MGMove BestMove { get; private set; }

    /// <summary>
    /// Time statistics of last search.
    /// </summary>
    public TimingStats TimingInfo { get; private set; }


    /// <summary>
    /// The node in the tree which was the root for the last search,
    /// possible not the root of the whole tree if the
    /// the last search was satisifed directly from tree reuse.
    /// </summary>
    public MCTSNode SearchRootNode => continuationSubroot != null ? continuationSubroot : Manager.Root;


    /// <summary>
    /// Total number of searches conducted.
    /// </summary>
    public static int SearchCount { get; internal set; }

    #region Tree reuse related

    /// <summary>
    /// Cumulative count of number of instant moves made due to 
    /// tree at start of search combined with a best move well ahead of others.
    /// </summary>
    public static int InstamoveCount { get; private set; }

    /// <summary>
    /// If not null, the non-root node which was used
    /// to satisfy the last search request (out of tree reuse).
    /// </summary>
    private MCTSNode continuationSubroot;

    /// <summary>
    /// The number of times a search from this tree
    /// has been satisfied out of tree reuse (no actual search).
    /// </summary>
    public int CountSearchContinuations { get; private set; }

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    public MCTSearch()
    {
    }


    /// <summary>
    /// Runs a new search.
    /// </summary>
    /// <param name="nnEvaluators"></param>
    /// <param name="paramsSelect"></param>
    /// <param name="paramsSearch"></param>
    /// <param name="limitManager"></param>
    /// <param name="reuseOtherContextForEvaluatedNodes"></param>
    /// <param name="priorMoves"></param>
    /// <param name="searchLimit"></param>
    /// <param name="verbose"></param>
    /// <param name="startTime"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="progressCallback"></param>
    /// <param name="possiblyUsePositionCache"></param>
    /// <param name="isFirstMoveOfGame"></param>
    public void Search(NNEvaluatorSet nnEvaluators,
                       ParamsSelect paramsSelect,
                       ParamsSearch paramsSearch,
                       IManagerGameLimit limitManager,
                       MCTSIterator reuseOtherContextForEvaluatedNodes,
                       PositionWithHistory priorMoves,
                       SearchLimit searchLimit, bool verbose,
                       DateTime startTime,
                       List<GameMoveStat> gameMoveHistory,
                       MCTSManager.MCTSProgressCallback progressCallback = null,
                       PositionEvalCache positionEvalCache = null,
                       bool possiblyUsePositionCache = false,
                       bool isFirstMoveOfGame = false)
    {
      if (searchLimit == null)
      {
        throw new ArgumentNullException(nameof(searchLimit));
      }

      if (searchLimit.SearchCanBeExpanded)
      {
        if (!MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC)
        {
          throw new Exception("STORAGE_USE_INCREMENTAL_ALLOC must be true when SearchCanBeExpanded.");
        }
      }

      if (!MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC  && !searchLimit.IsNodesLimit)
      {
        throw new Exception("SearchLimit must be NodesPerMove or NodesPerGame when STORAGE_USE_INCREMENTAL_ALLOC is false");
      }


      searchLimit = AdjustedSearchLimit(searchLimit, paramsSearch);

      int maxNodes;
      if (!searchLimit.SearchCanBeExpanded && searchLimit.IsNodesLimit)
      {
        maxNodes = (int)(searchLimit.Value + searchLimit.ValueIncrement + 5000);
      }
      else
      {
        // In this mode, we are just reserving virtual address space
        // from a very large pool (e.g. 256TB for Windows).
        // Therefore it is safe to reserve a very large block.
        if (searchLimit.MaxTreeNodes != null && searchLimit.MaxTreeNodes > MCTSNodeStore.MAX_NODES)
        {
          maxNodes = searchLimit.MaxTreeNodes.Value + 100_000;
        }
        else
        {
          maxNodes = MCTSNodeStore.MAX_NODES;
        }
      }

      MCTSNodeStore store = new MCTSNodeStore(maxNodes, priorMoves);

      SearchLimit searchLimitToUse = ConvertedSearchLimit(priorMoves.FinalPosition, searchLimit, 0, 0,
                                                          paramsSearch, limitManager,
                                                          gameMoveHistory, isFirstMoveOfGame);

      Manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, positionEvalCache, null,
                                nnEvaluators, paramsSearch, paramsSelect,  searchLimitToUse, 
                                limitManager, startTime, gameMoveHistory, isFirstMoveOfGame);

      using (new SearchContextExecutionBlock(Manager.Context))
      {
        (BestMove, TimingInfo) = MCTSManager.Search(Manager, verbose, progressCallback, possiblyUsePositionCache);
      }
    }


    /// <summary>
    /// Returns a new SearchLimit, which is converted
    /// from a game limit to per-move limit if necessary.
    /// </summary>
    /// <param name="searchLimit"></param>
    /// <param name="store"></param>
    /// <param name="searchParams"></param>
    /// <param name="limitManager"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="isFirstMoveOfGame"></param>
    /// <returns></returns>
    SearchLimit ConvertedSearchLimit(in Position position,
                                     SearchLimit searchLimit, 
                                     int searchRootN, float searchRootQ,
                                     ParamsSearch searchParams,
                                     IManagerGameLimit limitManager,
                                     List<GameMoveStat> gameMoveHistory,
                                     bool isFirstMoveOfGame)
    {
      // Possibly convert time limit per game into time for this move.
      if (!searchLimit.IsPerGameLimit)
      {
        // Return a clone.
        return searchLimit with { };
      }
      else
      {
        SearchLimitType type = searchLimit.Type == SearchLimitType.SecondsForAllMoves
                                                       ? SearchLimitType.SecondsPerMove
                                                       : SearchLimitType.NodesPerMove;


        ManagerGameLimitInputs limitsManagerInputs = new(in position,
                                searchParams, gameMoveHistory,
                                type, searchRootN, searchRootQ,
                                searchLimit.Value, searchLimit.ValueIncrement, searchLimit.MaxTreeNodes,
                                float.NaN, float.NaN,
                                maxMovesToGo: searchLimit.MaxMovesToGo,
                                isFirstMoveOfGame: isFirstMoveOfGame);

        ManagerGameLimitOutputs timeManagerOutputs = limitManager.ComputeMoveAllocation(limitsManagerInputs);
        return timeManagerOutputs.LimitTarget;
      }
    }
   

    /// <summary>
    /// Runs a search, possibly continuing from node 
    /// nested in a prior search (tree reuse).
    /// </summary>
    /// <param name="priorSearch"></param>
    /// <param name="reuseOtherContextForEvaluatedNodes"></param>
    /// <param name="moves"></param>
    /// <param name="newPositionAndMoves"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="searchLimit"></param>
    /// <param name="verbose"></param>
    /// <param name="startTime"></param>
    /// <param name="progressCallback"></param>
    /// <param name="thresholdMinFractionNodesRetained"></param>
    /// <param name="isFirstMoveOfGame"></param>
    public void SearchContinue(MCTSearch priorSearch,
                               MCTSIterator reuseOtherContextForEvaluatedNodes,
                               IEnumerable<MGMove> moves, PositionWithHistory newPositionAndMoves,
                               List<GameMoveStat> gameMoveHistory,
                               SearchLimit searchLimit,
                               bool verbose,  DateTime startTime,
                               MCTSManager.MCTSProgressCallback progressCallback, 
                               float thresholdMinFractionNodesRetained,
                               bool isFirstMoveOfGame = false)
    {
      CountSearchContinuations = priorSearch.CountSearchContinuations;
      Manager = priorSearch.Manager;
      Manager.StartTimeThisSearch = startTime;
      Manager.RootNWhenSearchStarted = priorSearch.SearchRootNode.N;

      searchLimit = AdjustedSearchLimit(searchLimit, Manager.Context.ParamsSearch);

      MCTSIterator priorContext = Manager.Context;
      MCTSNodeStore store = priorContext.Tree.Store;
      int numNodesInitial = Manager == null ? 0 : Manager.Root.N;

      MCTSNodeStructIndex newRootIndex;
      using (new SearchContextExecutionBlock(priorContext))
      {
        MCTSNode newRoot = null;
        if (Manager.TablebaseImmediateBestMove.IsNull)
        {
          newRoot = Manager.Root.FollowMovesToNode(moves);
        }

        // New root is not useful if contained no search
        // (for example if it was resolved via tablebase)
        // thus in that case we pretend as if we didn't find it
        if (newRoot != null && (newRoot.N == 0 || newRoot.NumPolicyMoves == 0)) newRoot = null;

        // Update contempt manager (if any) based opponent's prior move
        UpdateContemptManager(newRoot);

        // Compute search limits
        ComputeSearchLimits(newPositionAndMoves.FinalPosition, 
                            newRoot == null ? 0 : newRoot.N,
                            newRoot == null ? 0 : (float)newRoot.Q,
                            gameMoveHistory, searchLimit, isFirstMoveOfGame, priorContext, 
                            out SearchLimit searchLimitTargetAdjusted, out SearchLimit searchLimitIncremental);

        // Check for possible instant move
        (MCTSManager, MGMove, TimingStats) instamove = CheckInstamove(Manager, searchLimitIncremental, newRoot);
        if (instamove != default)
        {
          // Modify in place to point to the new root
          continuationSubroot = newRoot;
          BestMove = instamove.Item2;
          Manager.StopStatus = MCTSManager.SearchStopStatus.Instamove;
          TimingInfo = new TimingStats();
          return;
        }
        else
        {
          CountSearchContinuations = 0;
        }

        const bool possiblyUsePositionCache = false; // TODO: could this be relaxed?

        PositionEvalCache positionEvalCache = null;

        bool storeIsAlmostFull = priorContext.Tree.Store.FractionInUse > 0.9f;
        bool newRootIsBigEnoughForReuse = newRoot != null && newRoot.N >= (priorContext.Root.N * thresholdMinFractionNodesRetained);
        if (priorContext.ParamsSearch.TreeReuseEnabled
         && newRootIsBigEnoughForReuse
         && !storeIsAlmostFull)
        {
          SearchContinueRetainTree(reuseOtherContextForEvaluatedNodes, newPositionAndMoves, gameMoveHistory, verbose, startTime, progressCallback, isFirstMoveOfGame, priorContext, store, numNodesInitial, newRoot, searchLimitTargetAdjusted, possiblyUsePositionCache);
        }
        else
        {
#if NOT
          // Although it works, there actually seems no benefit to this idea.
          // The time spent on tree rebuild is proportional to benefit (nodes extracted)
          // so it's always better to directly rebuild tree.

          if ( newRoot != null && 
               priorContext.ParamsSearch.TestFlag &&
               priorContext.ParamsSearch.TreeReuseEnabled) // ????????????
          {
//            using (new TimingBlock("ExtractCacheNodesInSubtree"))
            {
              // TODO: One idea is to not create a separate data structure for this cache.
              //       Instead, store these extracted values into a List and
              //       store the index of each position in the primary transposition table,
              //       but with the value equal to the negative of the index 
              //       (so it can be distinguished from a "normal" entry pointing to a node in the new tree)

              // TODO: Some of the entires in the subtree are transposition linked.
              //       Currently ExtractCacheNodesInSubtree isn't smart enough to recover them.
              positionEvalCache = MCTSNodeStructStorage.ExtractCacheNodesInSubtree(priorContext.Tree, ref newRoot.Ref);
              MCTSEventSource.TestMetric1 += positionEvalCache.Count;
//              Console.WriteLine("got " + positionEvalCache.Count + " from prior root " + priorContext.Tree.Root.N + " new root " + newRoot.N);
//              positionEvalCache = null;

            }
          }
#endif

          // We decided not to (or couldn't find) that path in the existing tree.
          // First immediately release the prior store to allow memory reclamation.
          priorContext.Tree.Store.Dispose();

          // Now just run the search from a new tree.
          Search(Manager.Context.NNEvaluators, Manager.Context.ParamsSelect,
                 Manager.Context.ParamsSearch, Manager.LimitManager,
                 reuseOtherContextForEvaluatedNodes, newPositionAndMoves, searchLimit, verbose,
                 startTime, gameMoveHistory, progressCallback, positionEvalCache, possiblyUsePositionCache, isFirstMoveOfGame);
        }
      }
    }

    private void SearchContinueRetainTree(MCTSIterator reuseOtherContextForEvaluatedNodes, PositionWithHistory newPositionAndMoves, List<GameMoveStat> gameMoveHistory, bool verbose, DateTime startTime, MCTSManager.MCTSProgressCallback progressCallback, bool isFirstMoveOfGame, MCTSIterator priorContext, MCTSNodeStore store, int numNodesInitial, MCTSNode newRoot, SearchLimit searchLimitTargetAdjusted, bool possiblyUsePositionCache)
    {
      TimingStats statsMaterialize = new();
      if (Manager.Context.ParamsSearch.Execution.TranspositionMode != TranspositionMode.None)
      {
        if (!MCTSNodeStructStorage.USE_FAST_TREE_REBUILD)
        {
          using (new TimingBlock(statsMaterialize, TimingBlock.LoggingType.None))
          {
            MaterializeAllTranspositionLinkages(newRoot);
          }
        }
      }

      // Now rewrite the tree nodes and children "in situ"
      PositionEvalCache reusePositionCache = null;
      if (Manager.Context.ParamsSearch.TreeReuseRetainedPositionCacheEnabled)
      {
        reusePositionCache = new PositionEvalCache(false, 0);
      }

      // Create a new dictionary to recieve the new transposition roots
      TranspositionRootsDict newTranspositionRoots = null;
      if (priorContext.Tree.TranspositionRoots != null)
      {
        newTranspositionRoots = new TranspositionRootsDict(newRoot.N);
      }

      // TODO: Consider sometimes or always skip rebuild via MakeChildNewRoot,
      //       instead just set a new root (move it into place as first node).
      //       Perhaps rebuild only if the MCTSNodeStore would become excessively large.
      TimingStats makeNewRootTimingStats = new TimingStats();
      bool wasSwap = false;
      using (new TimingBlock(makeNewRootTimingStats, TimingBlock.LoggingType.None))
      {
        const float THRESHOLD_RETAIN_TREE = 0.70f;

        float fracRetain = (float)newRoot.Ref.N / priorContext.Tree.Root.N;
        if (fracRetain < THRESHOLD_RETAIN_TREE || !priorContext.ParamsSearch.TreeReuseSwapRootEnabled)
        {
          //Console.WriteLine("New root " + fracRetain);
          MCTSNodeStructStorage.MakeChildNewRoot(Manager.Context.Tree, ref newRoot.Ref, newPositionAndMoves,
                                                 reusePositionCache, newTranspositionRoots,
                                                 priorContext.ParamsSearch.Execution.TranspositionMaximizeRootN);
        }
        else
        {
          wasSwap = true;
          //Console.WriteLine("Swap root " + fracRetain + " from index " + newRoot.Index);
          MCTSNodeStructStorage.DoMakeChildNewRootSwapRoot(Manager.Context.Tree, ref newRoot.Ref, newPositionAndMoves,
                                                           reusePositionCache, newTranspositionRoots,
                                                           priorContext.ParamsSearch.Execution.TranspositionMaximizeRootN);
        }
        MCTSManager.TotalTimeSecondsInMakeNewRoot += (float)(statsMaterialize.ElapsedTimeSecs + makeNewRootTimingStats.ElapsedTimeSecs);
        //if (wasSwap) MCTSEventSource.TestMetric1 += (float)(statsMaterialize.ElapsedTimeSecs + makeNewRootTimingStats.ElapsedTimeSecs);
        if (this.Manager.Context.ParamsSearch.TestFlag) MCTSEventSource.TestMetric1 += (float)(statsMaterialize.ElapsedTimeSecs + makeNewRootTimingStats.ElapsedTimeSecs);
        if (wasSwap) MCTSEventSource.TestCounter1++;
      }

      CeresEnvironment.LogInfo("MCTS", "MakeChildNewRoot", $"Select {newRoot.N:N0} from {numNodesInitial:N0} "
                              + $"in {(int)(makeNewRootTimingStats.ElapsedTimeSecs / 1000.0)}ms");

      // Construct a new search manager reusing this modified store and modified transposition roots.
      Manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, reusePositionCache, newTranspositionRoots,
                                priorContext.NNEvaluators, priorContext.ParamsSearch, priorContext.ParamsSelect,
                                searchLimitTargetAdjusted, Manager.LimitManager,
                                startTime, gameMoveHistory, isFirstMoveOfGame:isFirstMoveOfGame);
      Manager.Context.ContemptManager = priorContext.ContemptManager;

      (BestMove, TimingInfo) = MCTSManager.Search(Manager, verbose, progressCallback, possiblyUsePositionCache);
    }


    /// <summary>
    /// Returns new SearchLimit, possibly adjusted for time overhead.
    /// </summary>
    /// <param name="limit"></param>
    /// <param name="paramsSearch"></param>
    /// <returns></returns>
    SearchLimit AdjustedSearchLimit(SearchLimit limit, ParamsSearch paramsSearch)
    {
      if (limit.IsTimeLimit)
      {
        return limit with { Value = Math.Max(0.01f, limit.Value - paramsSearch.MoveOverheadSeconds) };
      }
      else
      {
        return limit with { };
      }
    }


    private void ComputeSearchLimits(in Position position,
                                     int searchRootN, float searchRootQ,
                                     List<GameMoveStat> gameMoveHistory, SearchLimit searchLimit, 
                                     bool isFirstMoveOfGame, MCTSIterator priorContext, 
                                     out SearchLimit searchLimitTargetAdjusted, out SearchLimit searchLimitIncremental)
    {
      switch (searchLimit.Type)
      {
        case SearchLimitType.NodesPerMove:
          searchLimitIncremental = searchLimit with { };

          // Nodes per move are treated as incremental beyond initial starting size of tree.
          searchLimitTargetAdjusted = new SearchLimit(SearchLimitType.NodesPerMove, searchLimit.Value + searchRootN, maxTreeNodes:searchLimit.MaxTreeNodes);
          break;

        case SearchLimitType.SecondsPerMove:
          searchLimitIncremental = searchLimit with { };
          searchLimitTargetAdjusted = searchLimit with { };
          break;

        case SearchLimitType.NodesForAllMoves:
          searchLimitIncremental = ConvertedSearchLimit(in position,  searchLimit, 
                                                        searchRootN, searchRootQ,
                                                        priorContext.ParamsSearch, Manager.LimitManager,
                                                        gameMoveHistory, isFirstMoveOfGame);
          // Nodes per move are treated as incremental beyond initial starting size of tree.
          searchLimitTargetAdjusted = new SearchLimit(SearchLimitType.NodesPerMove, searchLimitIncremental.Value + searchRootN, 
                                                      maxTreeNodes: searchLimit.MaxTreeNodes);
          break;

        case SearchLimitType.SecondsForAllMoves:
          searchLimitTargetAdjusted = ConvertedSearchLimit(in position, searchLimit,
                                                           searchRootN, searchRootQ,
                                                           priorContext.ParamsSearch, Manager.LimitManager,
                                                           gameMoveHistory, isFirstMoveOfGame);
          searchLimitIncremental = searchLimitTargetAdjusted with { };
          break;

        default:
          throw new Exception($"Unsupported limit type {searchLimit.Type}");
          break;
      }
    }

    private static void MaterializeAllTranspositionLinkages(MCTSNode newRoot)
    {
      // The MakeChildNewRoot method is not able to handle transposition linkages
      // (this would be complicated and could involve linkages to nodes no longer in the retained subtree).
      // Therefore we first materialize any transposition linked nodes in the subtree.
      // Since this is not currently multithreaded we can turn off tree node locking for the duration.
      newRoot.MaterializeAllTranspositionLinks();
    }

    private void UpdateContemptManager(MCTSNode newRoot)
    {
      newRoot?.Annotate();

      // Inform contempt manager about the opponents move
      // (compared to the move we believed was optimal)
      if (newRoot != null && newRoot.Depth == 2)
      {
        MCTSNode opponentsPriorMove = newRoot;
        MCTSNode bestMove = opponentsPriorMove.Parent.ChildrenSorted(n => (float)n.Q)[0];
        if (bestMove.N > opponentsPriorMove.N / 10)
        {
          float bestQ = (float)bestMove.Q;
          float actualQ = (float)opponentsPriorMove.Q;
          Manager.Context.ContemptManager.RecordOpponentMove(actualQ, bestQ);
          //Console.WriteLine("Record " + actualQ + " vs best " + bestQ + " target contempt " + priorManager.Context.ContemptManager.TargetContempt);
        }
      }
    }
  }
}
