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
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.PositionEvalCaching;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.Chess.Positions;
using Ceres.MCTS.Params;
using Ceres.Base.Benchmarking;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Set of static helper methods which launch MCTS searches.
  /// </summary>
  public static partial class MCTSLaunch
  {
    /// <summary>
    /// Cumulative count of number of instant moves made due to 
    /// tree at start of search combined with a best move well ahead of others.
    /// </summary>
    public static int InstamoveCount { get; private set; }


    public static (MCTSManager, MGMove, TimingStats)
      Search(NNEvaluatorSet nnEvaluators,
             ParamsSelect paramsSelect,
             ParamsSearch paramsSearch,
             IManagerGameLimit timeManager,
             ParamsSearchExecutionModifier paramsSearchExecutionPostprocessor,
             MCTSIterator reuseOtherContextForEvaluatedNodes,
             PositionWithHistory priorMoves,
             SearchLimit searchLimit, bool verbose,
             DateTime startTime,
             List<GameMoveStat> gameMoveHistory,
             MCTSManager.MCTSProgressCallback progressCallback = null,
             bool possiblyUsePositionCache = false,
             bool isFirstMoveOfGame = false)
    {
      int maxNodes;
      if (MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC)
      {
        // In this mode, we are just reserving virtual address space
        // from a very large pool (e.g. 256TB for Windows).
        // Therefore it is safe to reserve a very large block.
        maxNodes = (int)(1.1f * MCTSNodeStore.MAX_NODES);
      }
      else
      {
        if (searchLimit.SearchCanBeExpanded)
          throw new Exception("STORAGE_USE_INCREMENTAL_ALLOC must be true when SearchCanBeExpanded.");

        if (searchLimit.Type != SearchLimitType.NodesPerMove)
          maxNodes = (int)searchLimit.Value + 5_000;
        else
          throw new Exception("STORAGE_USE_INCREMENTAL_ALLOC must be true when using time search limits.");
      }

      MCTSNodeStore store = new MCTSNodeStore(maxNodes, priorMoves);

      MCTSManager manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, null, null,
                                            nnEvaluators, paramsSearch, paramsSelect,
                                            searchLimit, paramsSearchExecutionPostprocessor, timeManager,
                                            startTime, null, gameMoveHistory, isFirstMoveOfGame);

      using (new SearchContextExecutionBlock(manager.Context))
      {
        (MGMove move, TimingStats stats) result = MCTSManager.Search(manager, verbose, progressCallback, possiblyUsePositionCache);
        return (manager, result.move, result.stats);
      }
    }



    public static (MCTSManager, MGMove, TimingStats)
      SearchContinue(MCTSManager priorManager,
                     MCTSIterator reuseOtherContextForEvaluatedNodes,
                     IEnumerable<MGMove> moves, PositionWithHistory newPositionAndMoves,
                     List<GameMoveStat> gameMoveHistory,
                     SearchLimit searchLimit,
                     bool verbose,  DateTime startTime,
                     MCTSManager.MCTSProgressCallback progressCallback, 
                     float thresholdMinFractionNodesRetained,
                     bool isFirstMoveOfGame = false)
    {
      MCTSIterator priorContext = priorManager.Context;
      MCTSNodeStore store = priorContext.Tree.Store;
      int numNodesInitial = priorManager == null ? 0 : priorManager.Root.N;

      MCTSNodeStructIndex newRootIndex;
      using (new SearchContextExecutionBlock(priorContext))
      {
        MCTSNode newRoot = FollowMovesToNode(priorManager.Root, moves);

        // New root is not useful if contained no search
        // (for example if it was resolved via tablebase)
        // thus in that case we pretend as if we didn't find it
        if (newRoot != null && (newRoot.N == 0 || newRoot.NumPolicyMoves == 0)) newRoot = null;

        // Check for possible instant move
        (MCTSManager, MGMove, TimingStats) instamove = CheckInstamove(priorManager, searchLimit, newRoot);

        if (instamove != default) return instamove;

        // TODO: don't reuse tree if it would cause the nodes in use
        //       to exceed a reasonable value for this machine
#if NOT
// NOTE: abandoned, small subtrees will be fast to rewrite so we can always do this
        // Only rewrite the store with the subtree reused
        // if it is not tiny relative to the current tree
        // (otherwise the scan/rewrite is not worth it
        float fracTreeReuse = newRoot.N / store.Nodes.NumUsedNodes;
        const float THRESHOLD_REUSE_TREE = 0.02f;
#endif
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
            priorManager.Context.ContemptManager.RecordOpponentMove(actualQ, bestQ);
            //Console.WriteLine("Record " + actualQ + " vs best " + bestQ + " target contempt " + priorManager.Context.ContemptManager.TargetContempt);
          }
        }

        bool storeIsAlmostFull = priorContext.Tree.Store.FractionInUse > 0.9f;
        bool newRootIsBigEnoughForReuse = newRoot != null && newRoot.N >= (priorContext.Root.N * thresholdMinFractionNodesRetained);
        if (priorContext.ParamsSearch.TreeReuseEnabled && newRootIsBigEnoughForReuse && !storeIsAlmostFull)
        {
          SearchLimit searchLimitAdjusted = searchLimit;

          if (priorManager.Context.ParamsSearch.Execution.TranspositionMode != TranspositionMode.None)
          {
            // The MakeChildNewRoot method is not able to handle transposition linkages
            // (this would be complicated and could involve linkages to nodes no longer in the retained subtree).
            // Therefore we first materialize any transposition linked nodes in the subtree.
            // Since this is not currently multithreaded we can turn off tree node locking for the duration.
            newRoot.Tree.ChildCreateLocks.LockingActive = false;
            newRoot.MaterializeAllTranspositionLinks();
            newRoot.Tree.ChildCreateLocks.LockingActive = true;
          }

          // Now rewrite the tree nodes and children "in situ"
          PositionEvalCache reusePositionCache = null;
          if (priorManager.Context.ParamsSearch.TreeReuseRetainedPositionCacheEnabled)
            reusePositionCache = new PositionEvalCache(0);

          TranspositionRootsDict newTranspositionRoots = null;
          if (priorContext.Tree.TranspositionRoots != null)
          {
            int estNumNewTranspositionRoots = newRoot.N + newRoot.N / 3; // somewhat oversize to allow for growth in subsequent search
            newTranspositionRoots = new TranspositionRootsDict(estNumNewTranspositionRoots);
          }

          // TODO: Consider sometimes or always skip rebuild via MakeChildNewRoot,
          //       instead just set a new root (move it into place as first node).
          //       Perhaps rebuild only if the MCTSNodeStore would become excessively large.
          TimingStats makeNewRootTimingStats = new TimingStats();
          using (new TimingBlock(makeNewRootTimingStats, TimingBlock.LoggingType.None))
          {
            MCTSNodeStructStorage.MakeChildNewRoot(store, priorManager.Context.ParamsSelect.PolicySoftmax, ref newRoot.Ref, newPositionAndMoves,
                                                   reusePositionCache, newTranspositionRoots);
          }
          MCTSManager.TotalTimeSecondsInMakeNewRoot += (float)makeNewRootTimingStats.ElapsedTimeSecs;

          CeresEnvironment.LogInfo("MCTS", "MakeChildNewRoot", $"Select {newRoot.N:N0} from {numNodesInitial:N0} " 
                                  + $"in {(int)(makeNewRootTimingStats.ElapsedTimeSecs/1000.0)}ms");

          // Finally if nodes adjust based on current nodes
          if (searchLimit.Type == SearchLimitType.NodesPerMove)
            searchLimitAdjusted = new SearchLimit(SearchLimitType.NodesPerMove, searchLimit.Value + store.RootNode.N);

          // Construct a new search manager reusing this modified store and modified transposition roots
          MCTSManager manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, reusePositionCache, newTranspositionRoots,
                                                    priorContext.NNEvaluators, priorContext.ParamsSearch, priorContext.ParamsSelect,
                                                    searchLimitAdjusted, priorManager.ParamsSearchExecutionPostprocessor, priorManager.LimitManager, 
                                                    startTime, priorManager, gameMoveHistory, isFirstMoveOfGame: isFirstMoveOfGame);
          manager.Context.ContemptManager = priorContext.ContemptManager;

          (MGMove move, TimingStats stats) result = MCTSManager.Search(manager, verbose, progressCallback, false);
          return (manager, result.move, result.stats);
        }

        else
        {
          // We decided not to (or couldn't find) that path in the existing tree
          // Just run the search from scratch
          if (verbose) Console.WriteLine("\r\nFailed nSearchFollowingMoves.");

          return Search(priorManager.Context.NNEvaluators, priorManager.Context.ParamsSelect,
                        priorManager.Context.ParamsSearch, priorManager.LimitManager,
                        null, reuseOtherContextForEvaluatedNodes, newPositionAndMoves, searchLimit, verbose, 
                        startTime, gameMoveHistory, progressCallback, false);
          //        priorManager.Context.StartPosAndPriorMoves, searchLimit, verbose, progressCallback, false);
        }
      }


#if NOT
      // This code partly or completely works
      // We don't rely upon it because it could result in uncontained growth of the store, 
      // since detached nodes are left
      // But if the subtree chosen is almost the whole tree, maybe we could indeed use this techinque as an alternate in these cases
      if (store.ResetRootAssumingMovesMade(moves, thresholdFractionNodesRetained))
      {

        SearchManager manager = new SearchManager(store, priorContext.ParamsNN,
                                                  priorContext.ParamsSearch, priorContext.ParamsSelect,
                                                  null, limit);
        manager.Context.TranspositionRoots = priorContext.TranspositionRoots;

        return Search(manager, false, verbose, progressCallback, false);
      }
#endif
    }

    static MCTSNode FollowMovesToNode(MCTSNode priorRoot, IEnumerable<MGMove> movesMade)
    {
      PositionWithHistory startingPriorMove = priorRoot.Context.StartPosAndPriorMoves;
      MGPosition position = startingPriorMove.FinalPosMG;
      MCTSIterator context = priorRoot.Context;

      // Advance root node and update prior moves
      MCTSNode newRoot = priorRoot;
      foreach (MGMove moveMade in movesMade)
      {
        bool foundChild = false;

        // Find this new root node (after these moves)
        foreach (MCTSNodeStructChild child in newRoot.Ref.Children)
        {
          if (child.IsExpanded)
          {
            MGMove thisChildMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(child.Move, in position);
            if (thisChildMove == moveMade)
            {
              // Advance new root to reflect this move
              newRoot = context.Tree.GetNode(child.ChildIndex, newRoot);

              // Advance position
              position.MakeMove(thisChildMove);

              // Done looking for match
              foundChild = true;
              break;
            }
          }
        }

        if (!foundChild)
          return null;
      }

      // Found it
      return newRoot;
    }


  }
}


