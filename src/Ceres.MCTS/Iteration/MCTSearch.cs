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
using Ceres.Base.DataType.Trees;
using System.Diagnostics;
using Ceres.MCTS.Evaluators;

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
    /// Undoes specified number of draw visits to node.
    /// </summary>
    public void BackupRevertDrawVisits(ref MCTSNodeStruct nodeRef, int numToRevert)
    {
      MCTSEventSource.TestCounter1++; // ****************************

      Span<MCTSNodeStruct> nodes = Manager.Context.Tree.Root.Store.Nodes.Span;

      // TODO: actually evaluate this node?
      nodeRef.Terminal = GameResult.Unknown;
      if (!nodeRef.IsRoot && nodeRef.ParentRef.DrawKnownToExistAmongChildren)
      {
        nodeRef.ParentRef.DrawKnownToExistAmongChildren = false;
      }

      ref MCTSNodeStruct node = ref nodeRef;
      while (true)
      {
        node.N -= numToRevert;
        if (nodeRef.dSum > 0)
        {
          nodeRef.dSum -= numToRevert;
        }

        if (node.IsRoot)
        {
          return;
        }
        else
        {
          node = ref nodes[node.ParentRef.Index.Index];
        }
      }

    }



    /// <summary>
    /// Possibly some of the positions marked as draws by two or three fold repetition
    /// are no longer repetitions if the prior moves changed with the tree reuse.
    /// Therefore possibly back these out (similar to LC0).
    /// </summary>
    public void FixupDrawsInvalidatedByTreeReuse()
    {
      Manager.Context.Tree.Root.StructRef.Traverse(Manager.Context.Tree.Store,
                       (ref MCTSNodeStruct nodeRef) =>
                       {
                         if (!nodeRef.IsOldGeneration && nodeRef.Terminal.IsTerminal() && nodeRef.HasRepetitions)
                         {
                           MCTSNode node = Manager.Context.Tree.GetNode(nodeRef.Index);
                           node.Annotate();
                           MCTSEventSource.TestMetric1++;
                           Console.WriteLine(node.Annotation.Pos.MiscInfo.RepetitionCount);
                           if (true || node.Annotation.Pos.MiscInfo.RepetitionCount == 0)
                           {
                               // Backout all but one draw
                               BackupRevertDrawVisits(ref nodeRef, node.N - 1);
                           }
                         }
                         return true;
                           //                           return nodeRef.DepthInTree < 8; // *** FIX
                         }, TreeTraversalType.Sequential);
    }



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
                       bool moveImmediateIfOnlyOneMove = false,
                       List<MGMove> searchMovesTablebaseRestricted = null)
    {
      if (searchLimit == null)
      {
        throw new ArgumentNullException(nameof(searchLimit));
      }

      if (searchLimit.SearchCanBeExpanded && !MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC)
      {
        throw new Exception("STORAGE_USE_INCREMENTAL_ALLOC must be true when SearchCanBeExpanded.");
      }

      if (!MCTSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC && !searchLimit.IsNodesLimit)
      {
        throw new Exception("SearchLimit must be NodesPerMove or NodesPerGame when STORAGE_USE_INCREMENTAL_ALLOC is false");
      }

      bool forceNoTablebaseTerminals = PosIsTablebaseWinWithNoDTZAvailable(paramsSearch, priorMoves);

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

      SearchLimit searchLimitToUse = SearchLimitPerMove(priorMoves.FinalPosition, searchLimit, 0, 0,
                                                          paramsSearch, limitManager,
                                                          gameMoveHistory, isFirstMoveOfGame);

      Manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, positionEvalCache, reuseNodeCache, null,
                                nnEvaluators, paramsSearch, paramsSelect, searchLimitToUse,
                                limitManager, startTime, gameMoveHistory, isFirstMoveOfGame, 
                                forceNoTablebaseTerminals, searchMovesTablebaseRestricted);

      MCTSIterator context = Manager.Context;

      reuseNodeCache?.ResetCache(false);
      reuseNodeCache?.SetContext(context);

      (BestMove, TimingInfo) = MCTSManager.Search(Manager, verbose, progressCallback,
                                                  possiblyUsePositionCache, moveImmediateIfOnlyOneMove);
    }

    internal static bool PosIsTablebaseWinWithNoDTZAvailable(ParamsSearch paramsSearch, PositionWithHistory priorMoves)
    {
      // Check if this is the unusual situation of a tablebase hit
      // but only WDL and not DTZ available.
      Position startPos = priorMoves.FinalPosition;
      bool forceNoTablebaseTerminals = false;
      if (startPos.PieceCount <= 7 && paramsSearch.EnableTablebases)
      {
        LeafEvaluatorSyzygy evaluatorTB = new LeafEvaluatorSyzygy(CeresUserSettingsManager.Settings.TablebaseDirectory, false);
        if (startPos.PieceCount <= evaluatorTB.MaxCardinality)
        {
          MGMove ret = evaluatorTB.Evaluator.CheckTablebaseBestNextMove(in startPos, out GameResult result,
            out List<MGMove> fullWinningMoveList, out bool winningMoveListOrderedByDTM);

          if (result == GameResult.Checkmate && !winningMoveListOrderedByDTM)
          {
            // No DTZ were available to guide search, must start a new tree
            // and perform actual NN search to find the win.
            forceNoTablebaseTerminals = true;
          }
        }
      }

      return forceNoTablebaseTerminals;
    }


    /// <summary>
    /// Returns a new SearchLimit, which is converted
    /// from a game limit to per-move limit if necessary.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="searchLimit"></param>
    /// <param name="searchRootN"></param>
    /// <param name="searchRootQ"></param>
    /// <param name="searchParams"></param>
    /// <param name="limitManager"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="isFirstMoveOfGame"></param>
    /// <returns></returns>
    SearchLimit SearchLimitPerMove(in Position position,
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
        SearchLimitType targetType = searchLimit.IsTimeLimit ? SearchLimitType.SecondsPerMove
                                                             : SearchLimitType.NodesPerMove;

        LastGameLimitInputs = new(in position,
                                  searchParams, gameMoveHistory,
                                  targetType, searchRootN, searchRootQ,
                                  searchLimit.Value, searchLimit.ValueIncrement,
                                  searchLimit.MaxTreeNodes, searchLimit.MaxTreeVisits,
                                  float.NaN, float.NaN,
                                  maxMovesToGo: searchLimit.MaxMovesToGo,
                                  isFirstMoveOfGame: isFirstMoveOfGame);

        LastGameLimitOutputs = limitManager.ComputeMoveAllocation(LastGameLimitInputs);
        return LastGameLimitOutputs.LimitTarget;
      }
    }


    public ManagerGameLimitInputs LastGameLimitInputs;
    public ManagerGameLimitOutputs LastGameLimitOutputs;

    public ManagerTreeReuse.ReuseDecision LastReuseDecision;


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
      Manager.TerminationManager.SearchMoves?.Clear();

      searchLimit = AdjustedSearchLimit(searchLimit, Manager.Context.ParamsSearch);

      MCTSIterator priorContext = Manager.Context;
      MCTSNodeStore store = priorContext.Tree.Store;
      int numNodesInitial = Manager == null ? 0 : Manager.Root.N;

      MCTSNodeStructIndex newRootIndex;
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

      // Compute search limit forced to per move type.
      SearchLimit searchLimitPerMove = SearchLimitPerMove(newPositionAndMoves.FinalPosition, searchLimit,
                                                          newRoot.IsNull ? 0 : newRoot.N,
                                                          newRoot.IsNull ? 0 : (float)newRoot.Q,
                                                          priorContext.ParamsSearch, Manager.LimitManager,
                                                          gameMoveHistory, isFirstMoveOfGame);
      Debug.Assert(!searchLimitPerMove.IsPerGameLimit);

      LastMakeNewRootTimingStats = default;

      bool forceNoTablebaseTerminals = PosIsTablebaseWinWithNoDTZAvailable(priorSearch.Manager.Context.ParamsSearch, newPositionAndMoves);

      ManagerTreeReuse.Method reuseMethod = ManagerTreeReuse.Method.NewStore;

      bool instamove;
      if (forceNoTablebaseTerminals
      && !priorContext.Manager.ForceNoTablebaseTerminals)
      {
        // Just exit with no store reuse.
        // The prior tree was bulit allowing tablebase terminals in node evaluations,
        // but now we need to do actual search to find the winning move sequence.
      }
      else
      {
        if (newRoot.IsNotNull)
        {
          const bool VERBOSE = false;
          LastReuseDecision = ManagerTreeReuse.ChooseMethod(priorContext.Tree.Root, newRoot, searchLimit.MaxTreeNodes);
          reuseMethod = LastReuseDecision.ChosenMethod;
        }

        // Check for possible instant move
        instamove = priorContext.Tree.Root == newRoot ? false
                                                      : CheckInstamove(Manager, searchLimitPerMove, newRoot, reuseMethod);
        if (instamove)
        {
          // Modify in place to point to the new root
          continuationSubroot = newRoot;
          BestMove = newRoot.BestMoveInfo(false).BestMove;
          Manager.StopStatus = MCTSManager.SearchStopStatus.Instamove;
          Manager.RootNWhenSearchStarted = continuationSubroot.N;
          TimingInfo = new TimingStats();
          return;
        }
        else
        {
          CountSearchContinuations = 0;
        }
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
                                 searchLimitPerMove, possiblyUsePositionCache, reuseMethod, moveImmediateIfOnlyOneMove,
                                 Manager.TerminationManager.SearchMovesTablebaseRestricted);
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
               isFirstMoveOfGame, moveImmediateIfOnlyOneMove, Manager.TerminationManager.SearchMovesTablebaseRestricted);
      }
    }

    void MarkUnreachable(MGPosition newRootPos, MCTSNode node, string desc)
    {
      int countReachable = 0;
      int countUnreachable = 0;
      foreach (var child1 in node.ChildrenSorted((MCTSNode v) => 0))
      {
        child1.Annotate();
        bool reachable1 = MGPositionReachability.IsProbablyReachable(newRootPos, child1.Annotation.PosMG);
        if (!reachable1)
        {
          Console.WriteLine(desc + " Invalidate " + child1.N);
          countUnreachable += child1.N;
        }
        else
        {
          Console.WriteLine(desc + " Keep " + child1.N);
          countReachable += child1.N;
        }
      }
      Console.WriteLine("reachable/unreachable " + countReachable + " " + countUnreachable);
    }

    public TimingStats LastMakeNewRootTimingStats;

    private void SearchContinueRetainTree(MCTSIterator reuseOtherContextForEvaluatedNodes, PositionWithHistory newPositionAndMoves,
                                          List<GameMoveStat> gameMoveHistory, bool verbose, DateTime startTime,
                                          MCTSManager.MCTSProgressCallback progressCallback, bool isFirstMoveOfGame,
                                          MCTSIterator priorContext, MCTSNodeStore store, int numNodesInitial, MCTSNode newRoot,
                                          SearchLimit searchLimitTargetAdjusted, bool possiblyUsePositionCache,
                                          ManagerTreeReuse.Method reuseMethod,
                                          bool moveImmediateIfOnlyOneMove, List<MGMove> searchMovesTablebaseRestricted)
    {
      IMCTSNodeCache reuseNodeCache = Manager.Context.Tree.NodeCache;

      // Now rewrite the tree nodes and children "in situ"
      PositionEvalCache reusePositionCache = null;
      if (Manager.Context.ParamsSearch.TreeReuseRetainedPositionCacheEnabled
       && reuseMethod != ManagerTreeReuse.Method.KeepStoreSwapRoot) // swap root already keeps all nodes accessible
      {
        reusePositionCache = new PositionEvalCache();
      }

      // We will either create a new transposition table (if tree rebuild) 
      // or modify existing one (if swap root).
      TranspositionRootsDict newTranspositionRoots = null;

      // TODO: Consider sometimes or always skip rebuild via MakeChildNewRoot,
      //       instead just set a new root (move it into place as first node).
      //       Perhaps rebuild only if the MCTSNodeStore would become excessively large.
      LastMakeNewRootTimingStats = new TimingStats();
      using (new TimingBlock(LastMakeNewRootTimingStats, TimingBlock.LoggingType.None))
      {
        //        const float THRESHOLD_RETAIN_TREE = 0.70f;

        //        float fracRetain = (float)newRoot.Ref.N / priorContext.Tree.Root.N;
        if (reuseMethod == ManagerTreeReuse.Method.UnchangedStore)
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
#if NOT
          ///////////////////////////////////////////////////////
          var parent = newRoot.Parent;
          MGPosition newRootPos = newRoot.Annotation.PosMG;

          Console.WriteLine("\r\nREACHABILITY ANALYSIS " + parent.Annotation.Pos.FEN + " " + newRoot.StructRef.PriorMove + " "  + newRootPos.ToPosition.FEN);
          MarkUnreachable(newRootPos, newRoot.Parent.Parent, "parent_parent");
          MarkUnreachable(newRootPos, newRoot.Parent, "parent");
          Console.WriteLine("----------------------------");
#endif

#if NOT
          foreach (var child in parent.ChildrenSorted((MCTSNode v)=>0))
          {
            //if (child != newRoot)
            {
              bool reachable = MGPositionReachability.IsProbablyReachable(newRootPos, child.Annotation.PosMG);
              Console.WriteLine(reachable + " " + child);
              if (!reachable)
              {
//                Console.WriteLine(newRootPos.ToPosition.FEN);
//                Console.WriteLine(child.Annotation.PosMG.ToPosition.FEN);
                System.Diagnostics.Debug.Assert(true);
              }
            }
          }
#endif
          ///////////////////////////////////////////////////////

          newTranspositionRoots = Manager.Context.Tree.TranspositionRoots;
          MCTSNodeStructStorage.DoMakeChildNewRootSwapRoot(Manager.Context.Tree, ref newRoot.StructRef, newPositionAndMoves,
                                                           reusePositionCache, newTranspositionRoots,
                                                           priorContext.ParamsSearch.Execution.TranspositionMaximizeRootN,
                                                           MCTSParamsFixed.NEW_ROOT_SWAP_RETAIN_NODE_CACHE);

#if NOT
          int fail = 0;
            int noFail = 0;
            using (new TimingBlock("Scan reachability " + Manager.Context.Tree.Store.RootNode.N + " " + newRoot.N))
            {
              using (new SearchContextExecutionBlock(Manager.Context))
              {
              MCTSNode rootNode = Manager.Context.Tree.GetNode(Manager.Context.Tree.Store.RootNode.Index);
                MGPosition newRootPos = rootNode.Annotation.PosMG;

                rootNode.StructRef.TraverseSequential(Manager.Context.Tree.Store, (ref MCTSNodeStruct nodeRef, MCTSNodeStructIndex index) =>
                {
                  Manager.Context.Tree.GetNode(index).Annotate();
                  MGPosition scanPos = Manager.Context.Tree.GetNode(index).Annotation.PosMG;
                  if (!MGPositionReachability.IsProbablyReachable(newRootPos, scanPos))
                  {
                    if (!nodeRef.IsOldGeneration)
                    {
                      Console.WriteLine(newRootPos.ToPosition.FEN);
                      Console.WriteLine(scanPos.ToPosition.FEN);
                      //throw new Exception("wrongtobe");
                    }
                    fail++;
                  }
                  else
                  {
                    noFail++;
                  }
                  return true;
                }
              );

              }
            }
            Console.WriteLine((100.0f * ((float)noFail / (noFail + fail))) + " Reachability: fail: " + fail + " ok:" + noFail);
#endif
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
          // Reload new root since now moved.
          newRoot = Manager.Context.Tree.Root;
        }
      }

      MCTSManager.TotalTimeSecondsInMakeNewRoot += (float)LastMakeNewRootTimingStats.ElapsedTimeSecs;
      
      CeresEnvironment.LogInfo("MCTS", "MakeChildNewRoot", $"Select {newRoot.N:N0} from {numNodesInitial:N0} "
                              + $"in {(int)(LastMakeNewRootTimingStats.ElapsedTimeSecs / 1000.0)}ms");

      // Construct a new search manager reusing this modified store and modified transposition roots.
      Manager = new MCTSManager(store, reuseOtherContextForEvaluatedNodes, reusePositionCache,
                                reuseNodeCache, newTranspositionRoots,
                                priorContext.NNEvaluators, priorContext.ParamsSearch, priorContext.ParamsSelect,
                                searchLimitTargetAdjusted, Manager.LimitManager,
                                startTime, gameMoveHistory, isFirstMoveOfGame: isFirstMoveOfGame, 
                                priorContext.Manager.ForceNoTablebaseTerminals, searchMovesTablebaseRestricted);
      Manager.Context.ContemptManager = priorContext.ContemptManager;

      Manager.Context.Tree.NodeCache.SetContext(Manager.Context);

      // NOTE: disabled, this happens in practive very rearely (and the revert code is not yet fully working)
      //FixupDrawsInvalidatedByTreeReuse();

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
