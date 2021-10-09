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
using Ceres.MCTS.Managers;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.NodeCache;

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
    public MCTSNode SearchRootNode => continuationSubroot.IsNotNull ? continuationSubroot : Manager.Root;


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
                       IMCTSNodeCache reuseNodeCache = null,
                       bool possiblyUsePositionCache = false,
                       bool isFirstMoveOfGame = false,
                       bool moveImmediateIfOnlyOneMove = false)
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

      if (!MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC && !searchLimit.IsNodesLimit)
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
        if (searchLimit.MaxTreeNodes != null)
        {
          long max = Math.Min(MCTSNodeStore.MAX_NODES, searchLimit.MaxTreeNodes.Value + 100_000);
          maxNodes = (int)max;
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

      Manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, positionEvalCache, reuseNodeCache, null,
                                nnEvaluators, paramsSearch, paramsSelect, searchLimitToUse,
                                limitManager, startTime, gameMoveHistory, isFirstMoveOfGame);
      
      MCTSIterator context = Manager.Context;

      reuseNodeCache?.ResetCache(false);
      reuseNodeCache?.SetContext(context);

      using (new SearchContextExecutionBlock(Manager.Context))
      {
        (BestMove, TimingInfo) = MCTSManager.Search(Manager, verbose, progressCallback, 
                                                    possiblyUsePositionCache, moveImmediateIfOnlyOneMove);
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
                                searchLimit.Value, searchLimit.ValueIncrement,
                                searchLimit.MaxTreeNodes, searchLimit.MaxTreeVisits,
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
                               bool verbose, DateTime startTime,
                               MCTSManager.MCTSProgressCallback progressCallback,
                               float thresholdMinFractionNodesRetained,
                               bool isFirstMoveOfGame = false,
                               bool moveImmediateIfOnlyOneMove = false)
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
        MCTSNode newRoot = default;
        if (Manager.TablebaseImmediateBestMove.IsNull)
        {
          newRoot = Manager.Root.FollowMovesToNode(moves);
        }

        // New root is not useful if contained no search
        // (for example if it was resolved via tablebase)
        // thus in that case we pretend as if we didn't find it
        if (newRoot.IsNotNull && (newRoot.N == 0 || newRoot.NumPolicyMoves == 0))
        {
          newRoot = default;
        }

        // Update contempt manager (if any) based opponent's prior move
        UpdateContemptManager(newRoot);

        // Compute search limits
        ComputeSearchLimits(newPositionAndMoves.FinalPosition,
                            newRoot.IsNull ? 0 : newRoot.N,
                            newRoot.IsNull ? 0 : (float)newRoot.Q,
                            gameMoveHistory, searchLimit, isFirstMoveOfGame, priorContext,
                            out SearchLimit searchLimitTargetAdjusted, out SearchLimit searchLimitIncremental);


        ManagerTreeReuse.Method reuseMethod = ManagerTreeReuse.Method.NewStore;
        if (newRoot.IsNotNull)
        {
          MCTSNode candidateBestMove = newRoot.BestMove(false);
          reuseMethod = ManagerTreeReuse.ChooseMethod(priorContext.Tree.Root, candidateBestMove, searchLimitTargetAdjusted.MaxTreeNodes);
        }

        // Check for possible instant move
        bool instamove = CheckInstamove(Manager, searchLimitIncremental, newRoot, reuseMethod);
        if (instamove)
        {
          // Modify in place to point to the new root
          continuationSubroot = newRoot;
          BestMove = newRoot.BestMoveInfo(false).BestMove;
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

        //        bool storeIsAlmostFull = priorContext.Tree.Store.FractionInUse > 0.9f;
        //        bool newRootIsBigEnoughForReuse = newRoot != null && newRoot.N >= (priorContext.Root.N * thresholdMinFractionNodesRetained);
        if (priorContext.ParamsSearch.TreeReuseEnabled
          && (reuseMethod == ManagerTreeReuse.Method.KeepStoreRebuildTree || reuseMethod == ManagerTreeReuse.Method.KeepStoreSwapRoot))
        //         && newRootIsBigEnoughForReuse
        //         && !storeIsAlmostFull//         
        {
          SearchContinueRetainTree(reuseOtherContextForEvaluatedNodes, newPositionAndMoves, gameMoveHistory, verbose, startTime,
                                   progressCallback, isFirstMoveOfGame, priorContext, store, numNodesInitial, newRoot,
                                   searchLimitTargetAdjusted, possiblyUsePositionCache, reuseMethod, moveImmediateIfOnlyOneMove);
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
          priorContext.Tree.ClearNodeCache(false);

          // We decided not to (or couldn't find) that path in the existing tree.
          // First immediately release the prior store to allow memory reclamation.
          priorContext.Tree.Store.Dispose();

          // Now just run the search from a new tree.
          Search(Manager.Context.NNEvaluators, Manager.Context.ParamsSelect,
                 Manager.Context.ParamsSearch, Manager.LimitManager,
                 reuseOtherContextForEvaluatedNodes, newPositionAndMoves, searchLimit, verbose,
                 startTime, gameMoveHistory, progressCallback, positionEvalCache, 
                 priorContext.Tree.NodeCache, possiblyUsePositionCache, 
                 isFirstMoveOfGame, moveImmediateIfOnlyOneMove);
        }
      }
    }

    private void SearchContinueRetainTree(MCTSIterator reuseOtherContextForEvaluatedNodes, PositionWithHistory newPositionAndMoves,
                                          List<GameMoveStat> gameMoveHistory, bool verbose, DateTime startTime,
                                          MCTSManager.MCTSProgressCallback progressCallback, bool isFirstMoveOfGame,
                                          MCTSIterator priorContext, MCTSNodeStore store, int numNodesInitial, MCTSNode newRoot,
                                          SearchLimit searchLimitTargetAdjusted, bool possiblyUsePositionCache,
                                          ManagerTreeReuse.Method reuseMethod,
                                          bool moveImmediateIfOnlyOneMove)
    {
      IMCTSNodeCache reuseNodeCache = Manager.Context.Tree.NodeCache;

      // Now rewrite the tree nodes and children "in situ"
      PositionEvalCache reusePositionCache = null;
      if (Manager.Context.ParamsSearch.TreeReuseRetainedPositionCacheEnabled)
      {
        reusePositionCache = new PositionEvalCache(false, 0);
      }

      // We will either create a new transposition table (if tree rebuild) 
      // or modify existing one (if swap root).
      TranspositionRootsDict newTranspositionRoots = null;

      // TODO: Consider sometimes or always skip rebuild via MakeChildNewRoot,
      //       instead just set a new root (move it into place as first node).
      //       Perhaps rebuild only if the MCTSNodeStore would become excessively large.
      TimingStats makeNewRootTimingStats = new TimingStats();
      using (new TimingBlock(makeNewRootTimingStats, TimingBlock.LoggingType.None))
      {
        //        const float THRESHOLD_RETAIN_TREE = 0.70f;

        //        float fracRetain = (float)newRoot.Ref.N / priorContext.Tree.Root.N;
        if (newRoot.Index == 1)
        {
          // Root not changing. Reuse existing store and transposition roots. No need to clear node cache.
          newTranspositionRoots = Manager.Context.Tree.TranspositionRoots;
        }
        else if (reuseMethod == ManagerTreeReuse.Method.KeepStoreSwapRoot
         && priorContext.ParamsSearch.TreeReuseSwapRootEnabled)
        {
          // For efficiency, keep the entries in the node cache
          // so they may be reused in next search.
          if (!MCTSParamsFixed.NEW_ROOT_SWAP_RETAIN_NODE_CACHE)
          {
            Manager.Context.Tree.ClearNodeCache(false);
          }
          newTranspositionRoots = Manager.Context.Tree.TranspositionRoots;
          MCTSNodeStructStorage.DoMakeChildNewRootSwapRoot(Manager.Context.Tree, ref newRoot.StructRef, newPositionAndMoves,
                                                           reusePositionCache, newTranspositionRoots,
                                                           priorContext.ParamsSearch.Execution.TranspositionMaximizeRootN,
                                                           MCTSParamsFixed.NEW_ROOT_SWAP_RETAIN_NODE_CACHE);
        }
        else
        {
          // TODO: We could add logic (similar to swap root case above)
          //       so the node cache would not be flushed, instead
          //       the node cache information would be updated (e.g. the field with Index).
          //       However this might be slightly involved, and performance gain would probably be modest.
          Manager.Context.Tree.ClearNodeCache(false);

          // Create a new dictionary to recieve the new transposition roots
          if (priorContext.Tree.TranspositionRoots != null)
          {
            newTranspositionRoots = new TranspositionRootsDict(newRoot.N);
          }

          MCTSNodeStructStorage.MakeChildNewRoot(Manager.Context.Tree, ref newRoot.StructRef, newPositionAndMoves,
                                                 reusePositionCache, newTranspositionRoots,
                                                 priorContext.ParamsSearch.Execution.TranspositionMaximizeRootN);
        }
      }

      MCTSManager.TotalTimeSecondsInMakeNewRoot += (float) makeNewRootTimingStats.ElapsedTimeSecs;

      CeresEnvironment.LogInfo("MCTS", "MakeChildNewRoot", $"Select {newRoot.N:N0} from {numNodesInitial:N0} "
                              + $"in {(int)(makeNewRootTimingStats.ElapsedTimeSecs / 1000.0)}ms");

      // Construct a new search manager reusing this modified store and modified transposition roots.
      Manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, reusePositionCache, reuseNodeCache, newTranspositionRoots,
                                priorContext.NNEvaluators, priorContext.ParamsSearch, priorContext.ParamsSelect,
                                searchLimitTargetAdjusted, Manager.LimitManager,
                                startTime, gameMoveHistory, isFirstMoveOfGame: isFirstMoveOfGame);
      Manager.Context.ContemptManager = priorContext.ContemptManager;

      Manager.Context.Tree.NodeCache.SetContext(Manager.Context);

      (BestMove, TimingInfo) = MCTSManager.Search(Manager, verbose, progressCallback, 
                                                  possiblyUsePositionCache, moveImmediateIfOnlyOneMove);
    }


    /// <summary>
    /// Returns new SearchLimit, possibly adjusted for time overhead and max tree nodes.
    /// </summary>
    /// <param name="limit"></param>
    /// <param name="paramsSearch"></param>
    /// <returns></returns>
    SearchLimit AdjustedSearchLimit(SearchLimit limit, ParamsSearch paramsSearch)
    {
      // Determine maximum number of tree nodes to allow store to grow to.
      int? maxTreeNodes;
      if (limit.MaxTreeNodes is not null)
      {
        // Use explicit value specified.
        maxTreeNodes = limit.MaxTreeNodes;
      }
      else if (CeresUserSettingsManager.Settings.MaxTreeNodes is not null)
      {
        // Use value explicitly set in Ceres.json.
        maxTreeNodes = CeresUserSettingsManager.Settings.MaxTreeNodes;
      }
      else
      {
        // Use default value based on amount of physical memory.
        maxTreeNodes = MCTSNodeStore.MAX_NODES - 2500;
      }

      if (limit.IsTimeLimit)
      {
        return limit with
        {
          MaxTreeNodes = maxTreeNodes,
          Value = Math.Max(0.01f, limit.Value - paramsSearch.MoveOverheadSeconds)
        };
      }
      else
      {
        return limit with { MaxTreeNodes = maxTreeNodes };
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
          searchLimitTargetAdjusted = new SearchLimit(SearchLimitType.NodesPerMove, searchLimit.Value + searchRootN,
                                                      maxTreeNodes: searchLimit.MaxTreeNodes,
                                                      maxTreeVisits: searchLimit.MaxTreeVisits);
          break;

        case SearchLimitType.SecondsPerMove:
          searchLimitIncremental = searchLimit with { };
          searchLimitTargetAdjusted = searchLimit with { };
          break;

        case SearchLimitType.NodesForAllMoves:
          searchLimitIncremental = ConvertedSearchLimit(in position, searchLimit,
                                                        searchRootN, searchRootQ,
                                                        priorContext.ParamsSearch, Manager.LimitManager,
                                                        gameMoveHistory, isFirstMoveOfGame);
          // Nodes per move are treated as incremental beyond initial starting size of tree.
          searchLimitTargetAdjusted = new SearchLimit(SearchLimitType.NodesPerMove, searchLimitIncremental.Value + searchRootN,
                                                      maxTreeNodes: searchLimit.MaxTreeNodes,
                                                      maxTreeVisits: searchLimit.MaxTreeVisits);
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


    private void UpdateContemptManager(MCTSNode newRoot)
    {
      if (newRoot.IsNotNull)
      {
        newRoot.Annotate();
      }

      // Inform contempt manager about the opponents move
      // (compared to the move we believed was optimal)
      if (newRoot.IsNotNull && newRoot.Depth == 2)
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
