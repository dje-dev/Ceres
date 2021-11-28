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
using System.Diagnostics;
using System.Threading;

using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;

using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Search
{
  public interface InstalledThreadPool
  {
    bool SupportsWaitDone => false;
    void QueueUserWorkItem(WaitCallback callback);
    void WaitDone();
    void Shutdown();
  }

  /// <summary>
  /// MCTS leaf selector algorithm which descends tree, choosing highest scoring children
  /// according to PUCT algorithm, yielding a set of leafs each of which is either:
  ///   - a new leaf node, not yet evaluated, or
  ///   - a terminal node (draw/checkmate) being visited again.
  ///   
  /// As a performance optimization, this algorithm can be asked to return more than one (N) leafs.
  /// This works as follows:
  ///   - Initially as we descend each level in the tree simulate N consecutive visits,
  ///     assuming NInFlight is sequentially incremented upon each visit. 
  ///     This yields a set of child indices and number of visits for each child.
  ///   - Then we recursively descend each of the selected children, 
  ///     requsting the corresponding number of leafs from that child (between 1 and N).
  ///     At this time the NInFlight increments are first propagated up the tree.
  ///   - As a performance optimziation, some of these descents may be spun off to separate threads for processing
  /// </summary>
  public class LeafSelectorMulti : ILeafSelector
  {
    // This is believed probably safe and necessary...
    internal const bool LAX_CHILD_ORDER = true;
    // Especially with .NET 6.0 it seems that the custom thread pool
    // is no longer more efficient than the standard .NET one.
    const bool USE_CUSTOM_THREADPOOL = false;

    #region Constructor arguments

    /// <summary>
    /// The sequence of moves which preceeded search.
    /// </summary>
    public readonly PositionWithHistory PriorSequence;


    /// <summary>
    /// Index of this selector.
    /// </summary>
    public int SelectorID { get; private set; }

    #endregion

    #region Internal data

    /// <summary>
    /// Synchronization object used to signal when all leaves have finished being gathered.
    /// </summary>
    CountdownEvent countdownPendingNumLeafs;

    public readonly MCTSIterator Context;

    // Local reference to execution parameters for efficiency
    ParamsSearchExecution paramsExecution;

    /// <summary>
    /// List of MCTSNodes being accumulated for expansion.
    /// </summary>
    ListBounded<MCTSNode> leafs;

    /// <summary>
    /// Pool of thread pools used across all active searches.
    /// </summary>
    ThreadPoolManaged tpm;
    static Lazy<ObjectPool<ThreadPoolManaged>> tpmPool = new Lazy<ObjectPool<ThreadPoolManaged>>(MakeObjectPoolThreads);

    #endregion

    #region Constructor

    static int numThreadPoolsCreated = 0;

    public delegate InstalledThreadPool ThreadPoolFactory(int numThreads);

    public static ThreadPoolFactory InstalledThreadPoolFactory;

    private InstalledThreadPool installedThreadPool;

    bool TrackWaitCount
    {
      get
      {
        if (installedThreadPool != null)
        {
          return !installedThreadPool.SupportsWaitDone;
        }
        else
        {
          return !USE_CUSTOM_THREADPOOL;
        }
      }
    }


    /// <summary>
    /// Constructor for selector over specified MCTSIterator.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="selectorID"></param>
    /// <param name="priorSequence"></param>
    /// <param name="guessNumLeaves"></param>
    public LeafSelectorMulti(MCTSIterator context, int selectorID, PositionWithHistory priorSequence, int guessNumLeaves)
    {
      Debug.Assert(selectorID < ILeafSelector.MAX_SELECTORS);

      if (InstalledThreadPoolFactory != null)
      {
        installedThreadPool = InstalledThreadPoolFactory(HardwareManager.MaxAvailableProcessors);
      }
      else
      {
        if (USE_CUSTOM_THREADPOOL)
        {
          tpm = tpmPool.Value.GetFromPool();
        }
      }

      countdownPendingNumLeafs = TrackWaitCount ? new CountdownEvent(1) : null;

      SelectorID = selectorID;
      PriorSequence = priorSequence;
      paramsExecution = context.ParamsSearch.Execution;

      int extraLeafsDynamic = 0;
      if (context.ParamsSearch.PaddedBatchSizing)
      {
        extraLeafsDynamic = context.ParamsSearch.PaddedExtraNodesBase + (int)(context.ParamsSearch.PaddedExtraNodesMultiplier * guessNumLeaves);
      }

      leafs = new ListBounded<MCTSNode>(guessNumLeaves + extraLeafsDynamic);

      Context = context;
    }


    /// <summary>
    /// Initializes the pool of worker thread pools.
    /// </summary>
    /// <returns></returns>
    static ObjectPool<ThreadPoolManaged> MakeObjectPoolThreads()
    {
      // Make sure pool is intialized
      const int MAX_THREAD_POOLS = 128;
      return new ObjectPool<ThreadPoolManaged>(() => new ThreadPoolManaged(numThreadPoolsCreated++.ToString(), HardwareManager.MaxAvailableProcessors, 0),
                                               MAX_THREAD_POOLS);
    }


    /// <summary>
    /// Shuts down the selector, restoring the thread pool back to the static pool
    /// </summary>
    public void Shutdown()
    {
      if (tpm != null)
      {
        tpmPool.Value.RestoreToPool(tpm);
      }

      tpm = null;
    }

    #endregion

    #region Main public worker method

    int NInFlightThisSelector(MCTSNode node) => SelectorID == 0 ? node.NInFlight : node.NInFlight2;


    /// <summary>
    /// Returns a set of newly selected nodes given target number of leafs.
    /// </summary>
    /// <param name="root"></param>
    /// <param name="numTargetVisits"></param>
    /// <param name="vLossDynamicBoost"></param>
    /// <returns></returns>
    public ListBounded<MCTSNode> SelectNewLeafBatchlet(MCTSNode root, int numTargetVisits, float vLossDynamicBoost)
    {
      Reset();

#if FEATURE_SUPPLEMENTAL
      supplementalCandidates.Clear();
#endif
      DoSelectNewLeafBatchlet(root, numTargetVisits, vLossDynamicBoost);

      root.Context.BatchPostprocessAllEvaluators();

      return leafs;
    }


    /// <summary>
    /// Private worker method to calcaulte a new set of selected nodes given target number of leafs.
    /// </summary>
    /// <param name="root"></param>
    /// <param name="numTargetLeafs"></param>
    /// <param name="vLossDynamicBoost"></param>
    /// <returns></returns>
    void DoSelectNewLeafBatchlet(MCTSNode root, int numTargetLeafs, float vLossDynamicBoost)
    {
      InsureAnnotated(root);

      ListBounded<MCTSNode> gatheredNodes = new ListBounded<MCTSNode>(numTargetLeafs);
      DoGatherLeafBatchlet(root, numTargetLeafs, vLossDynamicBoost, gatheredNodes);

      if (paramsExecution.SelectParallelEnabled)
      {
        WaitDone();
      }


      leafs.Add(gatheredNodes);
    }

    #endregion

    #region Clear

    /// <summary>
    /// Resets state of selector to be prepared for 
    /// selecting a new set of nodes.
    /// </summary>
    public void Reset()
    {
      leafs.Clear(false);
    }

    #endregion

    #region Annotation

    /// <summary>
    /// Annotates specified node if not already annotated.
    /// </summary>
    /// <param name="node"></param>
    public void InsureAnnotated(MCTSNode node)
    {
      if (!node.IsAnnotated)
      {
        Context.Tree.Annotate(node, true);
      }
    }

    #endregion

    #region Threads

    /// <summary>
    /// Waits until all threads have completed their work
    /// doing leaf selection.
    /// </summary>
    void WaitDone()
    {
      if (TrackWaitCount)
      {
        countdownPendingNumLeafs.Signal(); // take out initialization value of 1
        countdownPendingNumLeafs.Wait();
        countdownPendingNumLeafs.Reset();
      }
      else if (installedThreadPool != null)
      {
        installedThreadPool.WaitDone();
      }
      else
      {
        tpm.WaitDone();
      }
    }

    #endregion

    #region Node visitation

    /// <summary>
    /// Takes any actions necessary upon visit to an inner note.
    /// </summary>
    /// <param name="node"></param>
    void DoVisitInnerNode(MCTSNode node)
    {
      InsureAnnotated(node);

      // ************************************************ TEST ONLY
      if (node.Context.EvaluatorDefSecondary != null)
      {
#if NOT
        if (node.StructRef.SecondaryNN)
        {
          MCTSEventSource.TestMetric1++;
        }
        else
        {
          MCTSEventSource.TestCounter1++;
        }
#endif
      }

        if (node.N > 10 && node.Context.EvaluatorDefSecondary != null)
      {
        // Add this node to pending list to be evaluated by secondary evaluator
        // if not aleady evaluated by secondary and visit fraction is sufficiently high.
        float nFractionOfRootN = (float)node.N / node.Context.Root.N; 
        if (!node.StructRef.SecondaryNN
          && node.Terminal == GameResult.Unknown
          && nFractionOfRootN > Context.ParamsSearch.ParamsSecondaryEvaluator.UpdateMinNFraction
          && Context.Root.N > 100 + Context.ParamsSearch.ParamsSecondaryEvaluator.InitialTreeNodesForceSecondary
          )
        {
          // Add it to the pending list and mark as processed so not added multiple times.
          Context.PendingSecondaryNodes.Add(node);
          node.StructRef.SecondaryNN = true;
        }
      }
    }


    /// <summary>
    /// Verifies the sum of the children inflight is equal to the parent in flight.
    /// </summary>
    /// <param name="root"></param>
    /// <returns></returns>
    bool VerifyNInFlightConsistent(MCTSNode root)
    {
      int countInFlight = 0;
      int countInFlight2 = 0;
      for (int i = 0; i < leafs.Count; i++)
      {
        countInFlight += leafs[i].NInFlight;
        countInFlight2 += leafs[i].NInFlight2;
      }

      if (countInFlight != root.NInFlight || countInFlight2 != root.NInFlight2)
      {
        Console.WriteLine($"root in flights are  {root.NInFlight} {root.NInFlight2} but sum of children are {countInFlight} {countInFlight2}");
        return false;
      }

      return true;
    }

    internal void DoVisitLeafNode(MCTSNode node, int numVisits, ListBounded<MCTSNode> gatheredNodes)
    {
      ref MCTSNodeStruct nodeRef = ref node.StructRef;

      int initialNInFlightValue;
      if (SelectorID == 0)
      {
        initialNInFlightValue = nodeRef.NInFlight;
        nodeRef.UpdateNInFlight(numVisits, 0);
      }
      else
      {
        initialNInFlightValue = nodeRef.NInFlight2;
        nodeRef.UpdateNInFlight(0, numVisits);
      }

      // Add as a leaf if this was not already visited
      if (initialNInFlightValue == 0)
      {
        // Annotate
        InsureAnnotated(node);

        ref MCTSNodeAnnotation annotationRef = ref node.Annotation;

        nodeRef.NumPieces = annotationRef.Pos.PieceCount;
        nodeRef.NumRank2Pawns = annotationRef.PosMG.NumPawnsRank2;

        // Verify this looks like a true non-leaf
        Debug.Assert(nodeRef.N == 0
                          || nodeRef.Terminal != Chess.GameResult.Unknown
                          || nodeRef.NumVisitsPendingTranspositionRootExtraction > 0);

        // Set default action 
        node.ActionType = MCTSNodeInfo.NodeActionType.MCTSApply;

        // Add to set of leafs
        gatheredNodes.Add(node);
      }

      // Possibly scan not-yet visited siblings and record information for
      // possible use later in value backup.
      if (node.Context.ParamsSearch.EnableUseSiblingEvaluations
       && !node.IsRoot)
      {
        Debug.Assert(node.Context.EvaluatorDef.NumCacheHashPositions == 1); // this hardcoded value  is assumed below
        MCTSNodeSiblingEval.TrySetPendingSiblingValue(node, 1);
      }

    }


#endregion

#region Selection algorithm

    static long applyCount = 0;
    static double applyCPUCTTot = 0;
    static double applyAbsUncertaintyTot = 0;

    /// <summary>
    /// To get numTargetLeafs we usually don't have to check all children. 
    /// We definitely need to search over all visited children, and then add on
    /// the "worst possible" case that all new visits come from unvisited children (in sequence).
    /// </summary>
    /// <param name="node"></param>
    /// <param name="numTargetLeafs"></param>
    /// <returns></returns>
    int NumChildrenNeededToBeChecked(MCTSNode node, int numTargetLeafs)
    {
      return MathUtils.MinOfPositivesFast(node.NumPolicyMoves, node.NumChildrenVisited + numTargetLeafs);
    }


    internal void GetChildSelectionCounts(int selectorID, MCTSNode node, int numTargetLeafs, int numChildrenToCheck,
                                          Span<short> visitChildCounts, float vLossDynamicBoost)
    {
#if NOT
      // TODO: clean up
      // NOTE: 
      //   - below limited, including to second half of search
      if (node.Depth == 0
        && node.Context.ParamsSelect.FirstMoveThompsonSamplingFactor > 0
        && node.Context.FirstMoveSampler.NumSamples > 500 // beware tree reuse which can reset our samples
        && numTargetLeafs > 10
//        && node.N > node.Context.SearchLimit.EstNumNodes(30_000) / 2   // *** CLEANUP ***
        && node.N % 2 != 0 // ** 3 out of 4 in second half are done with sampling *** CLEANUP ***
        )
      {
        float SD_SCALING_FACTOR = node.Context.ParamsSelect.FirstMoveThompsonSamplingFactor; // e.g. 0.15

        int[] indices = node.Context.FirstMoveSampler.GetIndicesOfBestSamples(false, node.NumChildrenExpanded, numTargetLeafs, SD_SCALING_FACTOR);

        for (int i = 0; i < visitChildCounts.Length; i++)
          visitChildCounts[i] = (short)indices[i];

        //CalcWeightedDistanceFromBestMoveVisitCounts(node, numTargetLeafs, visitChildCounts);
        return;
      }
#endif
#if EXPERIMENTAL
      if (node.Context.ParamsSearch.TEST_FLAG && node.N > 2)
      {
        const bool VERBOSE = false;
        short[] counts = LeafSelectorRegularizedPolicyOptimization.GetVisitCountsPOI(node, numTargetLeafs, numChildrenToCheck, node.NumChildrenVisited, VERBOSE);

        for (int i = 0; i < visitChildCounts.Length; i++)
          visitChildCounts[i] = counts[i];

        // Validated count (debugging only)
        int totalCount = 0;
        for (int i = 0; i < numChildrenToCheck; i++)
          totalCount += visitChildCounts[i];
        if (totalCount != numTargetLeafs)
          throw new Exception("wrong");
      }
      else
      {
#endif

#if FEATURE_UNCERTAINTY

      float cpuctMultiplier = 1;

      const float UNCERTAINTY_MIDPOINT = 0.05f;
      const float UNCERTAINTY_SCALE = 1.5f;
      if (node.Context.ParamsSearch.TestFlag && node.N > 10)
      {
        float uncertaintyDiff = node.Uncertainty - UNCERTAINTY_MIDPOINT;
        cpuctMultiplier = 0.90f + uncertaintyDiff * UNCERTAINTY_SCALE;
        cpuctMultiplier = StatUtils.Bounded(cpuctMultiplier, 0.80f, 1.25f);

        applyCount++;
        applyCPUCTTot += cpuctMultiplier;
        applyAbsUncertaintyTot += Math.Abs(uncertaintyDiff);

        if (node.Ref.ZobristHash % 1000 == 999)
        {
          MCTSEventSource.TestMetric1 = (float)(applyCPUCTTot / applyCount);
          MCTSEventSource.TestCounter1 = (int)Math.Round(100 * (applyAbsUncertaintyTot / applyCount), 0);
        }
      }
#else
      const float cpuctMultiplier = 1;
#endif

      Span<float> scores = default;
      node.InfoRef.ComputeTopChildScores(selectorID, node.Depth,
                                 vLossDynamicBoost, 0, numChildrenToCheck - 1, numTargetLeafs,
                                 scores, visitChildCounts, cpuctMultiplier);

      if (node.Depth == 0)
      {
        Context.RootMoveTracker?.UpdateVisitCounts(visitChildCounts, numChildrenToCheck, numTargetLeafs);
      }
    }


    float TopNFractionToTopQMove
    {
      get
      {
        MCTSNode topMove = Context.Root.BestMove(false);
        if (topMove.IsNull || topMove.NumPolicyMoves <= 1) return 1.0f;

        return (float)topMove.ChildWithLargestN.N / (float)topMove.N;
      }
    }


#if FEATURE_SUPPLEMENTAL
    internal ConcurrentBag<(MCTSNode ParentNode, int SelectorID, int ChildIndex)> supplementalCandidates = new ();
#endif

    /// <summary>
    /// Starts or continues a MCTS descent to ultimately select a set of leaf nodes
    /// (of specified size).
    /// 
    /// </summary>
    /// <param name="node"></param>
    /// <param name="parentNode"></param>
    /// <param name="numTargetLeafs"></param>
    private void DoGatherLeafBatchlet(MCTSNode node, int numTargetLeafs, float vLossDynamicBoost,
                                      ListBounded<MCTSNode> gatheredNodes)
    {
      ref MCTSNodeStruct nodeRef = ref node.StructRef;

      Debug.Assert(numTargetLeafs > 0);

      // Materialize if transposition linked and either:
      //   - the number of target leaves is more than number available (pending)
      //   - the number of target leaves is more than 1 (pending values are valid only for one use)
      if (paramsExecution.TranspositionMode == TranspositionMode.SingleNodeDeferredCopy
       && node.IsTranspositionLinked
       && (numTargetLeafs > 1  || numTargetLeafs > node.NumVisitsPendingTranspositionRootExtraction))
      {
        node.StructRef.MaterializeSubtreeFromTranspositionRoot(node.Tree);
      }

      if (paramsExecution.TranspositionMode == TranspositionMode.SingleNodeDeferredCopy
       && node.IsTranspositionLinked)
      {
        Debug.Assert(node.NumVisitsPendingTranspositionRootExtraction > 0);
        Debug.Assert(node.TranspositionRootIndex != 0);
        LeafEvaluatorTransposition.EnsurePendingTranspositionValuesSet(node, true);
        Debug.Assert(!FP16.IsNaN(node.PendingTranspositionV));

        // Treat this as if it were a leaf node (i.e. do not descend further yet).
        DoVisitLeafNode(node, numTargetLeafs, gatheredNodes);

        return;
      }
      else if (paramsExecution.TranspositionMode == TranspositionMode.SharedSubtree)
      {
        throw new NotImplementedException();
        //InitializeChildrenFromDeferredTransposition(node);
      }

      bool isUnvisited = node.N == 0;
      if (isUnvisited || nodeRef.Terminal.IsTerminal() || nodeRef.IsTranspositionLinked)
      {
        DoVisitLeafNode(node, numTargetLeafs, gatheredNodes);
        return;
      }

      if (paramsExecution.TranspositionMode == TranspositionMode.SharedSubtree)
      {
        ref MCTSNodeStruct biggestTranspositionNode = ref MCTSNodeTranspositionManager.GetNodeWithMaxNInCluster(node);
        if (!biggestTranspositionNode.IsNull)
        {
          if (biggestTranspositionNode.N < node.N)
          {
            // We are already bigger, so ignore the transposition root
          }
          else if (biggestTranspositionNode.N > node.N)
          {
            // Borrow from the other bigger subtree
            FP16 vToUse = (FP16)((float)(biggestTranspositionNode.W - node.W)
                                 / (float)(biggestTranspositionNode.N - node.N));

            throw new Exception("need to restore following lines, overrideV and m currently disabled since mSum is marked private");
            //node.OverrideVToApplyFromTransposition = vToUse;
            //FP16 mToUse = (FP16)((float)(biggestTranspositionNode.mSum - node.Ref.mSum)
            //                     / (float)(biggestTranspositionNode.N - node.N));
            //node.OverrideMPositionToApplyFromTransposition = mToUse;

            DoVisitLeafNode(node, numTargetLeafs, gatheredNodes);
            return;
          }
          else if (biggestTranspositionNode.N == node.N)
          {
            if (biggestTranspositionNode.Index.Index == node.Index)
            {
              // Biggest node is ourself! nothing to do
            }
            else if (biggestTranspositionNode.IsInFlight)
            {
              // Abandon search below this node
              // Also we have to undo the NInFlight updates alread made to parents
              if (!node.IsRoot)
              {
                node.Parent.StructRef.BackupDecrementInFlight(SelectorID == 0 ? numTargetLeafs : 0,
                                                  SelectorID == 1 ? numTargetLeafs : 0);
              }
              return;
            }
            else
            {
              // We assume the role of the new master 
              MCTSNodeStructIndex indexTranspositionRoot = biggestTranspositionNode.Index;
              MCTSNodeStructIndex indexThis = new MCTSNodeStructIndex(node.Index);

              MCTSNodeStructStorage.ModifyParentsChildRef(node.Store, indexTranspositionRoot, indexThis);
              MCTSNodeStructStorage.ModifyParentsChildRef(node.Store, indexThis, indexTranspositionRoot);

              // Swap parents
              MCTSNodeStructIndex saveIndexThisParent = node.StructRef.ParentIndex;
              nodeRef.ParentIndex = biggestTranspositionNode.ParentIndex;
              biggestTranspositionNode.ParentIndex = saveIndexThisParent;

              node = node.Tree.GetNode(biggestTranspositionNode.Index);
            }

          }

        }
      }

      // Mark node as visited, make sure we get associated annotation
      DoVisitInnerNode(node);

      // Prefetch not obviously helpful
      //node.Ref.PossiblyPrefetchChildArray(node.Store, new MCTSNodeStructIndex(node.Index));

#if EXPERIMENTAL
      if (node.Context.ParamsSearch.TestFlag && node.NumChildrenExpanded == 0 && node.NumPolicyMoves > 1)
      {
        throw new NotImplementedException();
        bool got = MCTSManager.ThreadSearchContext.Tree.TranspositionRoots.TryGetValue(node.StructRef.ZobristHash, out int siblingTanspositionNodeIndex);
        if (got)
        {
          //              MCTSNode transpositionRootNode = MCTSManager.ThreadSearchContext.Tree.GetNode(new MCTSNodeStructIndex(siblingTanspositionNodeIndex));
          ref readonly MCTSNodeStruct transpositionRootNode = ref MCTSManager.ThreadSearchContext.Tree.Store.Nodes.nodes[siblingTanspositionNodeIndex];
          if (transpositionRootNode.NumChildrenExpanded > 1)
          {
            if (transpositionRootNode.ChildAtIndexRef(0).N * 2 < transpositionRootNode.ChildAtIndexRef(1).N)
            {
              node.StructRef.SwapFirst();
              MCTSEventSource.TestMetric1++;
//              Console.WriteLine(got + " " + transpositionRootNode.N + " " + transpositionRootNode.ChildAtIndexRef(0).N + " " + transpositionRootNode.ChildAtIndexRef(1).N);
            }
          }

        }
      }
      else
      {
//        Console.WriteLine("nogo");
      }
#endif

      int numChildrenToCheck = NumChildrenNeededToBeChecked(node, numTargetLeafs);
      Span<short> childVisitCounts = stackalloc short[numChildrenToCheck];

      if (numChildrenToCheck == 1)
      {
        // No need to compute in this special case of first visit
        Debug.Assert((node.NumPolicyMoves == 1) || (numTargetLeafs == 1 && node.NumChildrenVisited == 0));
        childVisitCounts[0] = (short)numTargetLeafs;
      }
      else
      {
        GetChildSelectionCounts(SelectorID, node, numTargetLeafs, numChildrenToCheck, childVisitCounts, vLossDynamicBoost);
      }

      VerifyTargetLeafsCorrect(numChildrenToCheck, childVisitCounts, numTargetLeafs);

      Span<MCTSNodeStructChild> children = nodeRef.ChildrenFromStore(node.Store);

      int numVisitsProcessed = 0;
      if (paramsExecution.SelectParallelEnabled)
      {
        if (numTargetLeafs > paramsExecution.SelectParallelThreshold)
        {
          // Immediately make a first pass to immediately launch the children
          // that have enough visits to be processed in parallel  
          ProcessSelectedChildren(node, numTargetLeafs, vLossDynamicBoost, numChildrenToCheck,
                                paramsExecution.SelectParallelThreshold, true,
                                childVisitCounts, children, ref numVisitsProcessed, gatheredNodes);
        }

        // Make a second pass process any remaining chidren having visits (not parallel)
        if (numVisitsProcessed < numTargetLeafs)
        {
          ProcessSelectedChildren(node, numTargetLeafs, vLossDynamicBoost, numChildrenToCheck,
                                  1, false, childVisitCounts, children, ref numVisitsProcessed, gatheredNodes);
        }

      }
      else
      {
        // Process all children with nonzero counts (not parallel)
        ProcessSelectedChildren(node, numTargetLeafs, vLossDynamicBoost, numChildrenToCheck,
                                1, false, childVisitCounts, children, ref numVisitsProcessed, gatheredNodes);
      }
    }

    void ProcessSelectedChildren(MCTSNode node, int numTargetLeafs, float vLossDynamicBoost, int numChildrenToCheck,
                                 int minVisitsCountToProcess, bool launchParallel,
                                 Span<short> childVisitCounts, Span<MCTSNodeStructChild> children,
                                 ref int numVisitsProcessed, ListBounded<MCTSNode> gatheredNodes)
    {
      MCTSNodeStruct nodeRef = node.StructRef;

#if NOT
      if (node.Context.ParamsSearch.TestFlag)
      {
        // TODO: cleanup
        // In test mode we might not have policy order and therefore get chosen nodes off end
        // Fixup.
        bool haveMoved = false;
        for (int childIndex = 0; childIndex < numChildrenToCheck; childIndex++)
        {
          int numThisChild = childVisitCounts[childIndex];
          //            Console.WriteLine("Saw (and ignoring) out of order possibly due to tie in score");
          if (childIndex > node.NumChildrenExpanded)
          {
            if (!haveMoved && childVisitCounts[numChildrenToCheck - 1] == 0)
            {
              childVisitCounts[numChildrenToCheck - 1] = 1;
              childVisitCounts[childIndex] = 0;
              haveMoved = true;
            }
          }
        }

      }
#endif

      for (int childIndex = 0; childIndex < numChildrenToCheck; childIndex++)
      {
        int numThisChild = childVisitCounts[childIndex];
        if (numThisChild >= minVisitsCountToProcess)
        {
          if (childIndex > node.NumChildrenExpanded)
          {
            if (!LAX_CHILD_ORDER)
            {
              Console.WriteLine("Warning: saw (and ignoring) out of order possibly due to tie in score");
              continue;
            }
          }
          MCTSNode thisChild = default;

          MCTSNodeStructChild childInfo = children[childIndex];
          if (!childInfo.IsExpanded)
          {
            thisChild = node.CreateChild(childIndex, LAX_CHILD_ORDER);
          }
          else
          {
            thisChild = node.Child(childInfo);
          }

          node.UpdateRecordVisitsToChild(SelectorID, childIndex, numThisChild);

          // Prefetch not obviously helpful
          // nodeRef.PossiblyPrefetchChild(node.Store, new MCTSNodeStructIndex(node.Index), childIndex);

#if FEATURE_SUPPLEMENTAL
          // Warning: this slows down search by up to 10%
          if (node.NumPolicyMoves > childIndex + 1
           && node.ChildAtIndexInfo(childIndex).p > 0.18f
           && node.ChildAtIndexInfo(childIndex).p - node.ChildAtIndexInfo(childIndex + 1).p < 0.03)
          {
            supplementalCandidates.Add((node, SelectorID, childIndex + 1));
          }
#endif

          Debug.Assert(node.Depth < 255);

          int numVisitsLeftAfterThisChild = numTargetLeafs - (numVisitsProcessed + numThisChild);
          if (launchParallel && numVisitsLeftAfterThisChild > minVisitsCountToProcess / 2)
          {
            LaunchGatherLeafBatchletParallel(node, numThisChild, thisChild, vLossDynamicBoost);
          }
          else
          {
            DoGatherLeafBatchlet(thisChild, numThisChild, vLossDynamicBoost, gatheredNodes);
          }

          // mark this child as done
          childVisitCounts[childIndex] = 0;

          numVisitsProcessed += numThisChild;
          {
            if (numVisitsProcessed == numTargetLeafs)
            {
              break;
            }
          }
        }
      }
    }



    private void LaunchGatherLeafBatchletParallel(MCTSNode node, int numThisChild, MCTSNode thisChild, float vLossDynamicBoost)
    {
      if (TrackWaitCount)
      {
        countdownPendingNumLeafs.AddCount(numThisChild);
      }

      MCTSIterator thisContext = node.Context;

      WaitCallback action = (object obj) =>
      {
        try
        {
          // Gather the nodes.
          ListBounded<MCTSNode> gatheredNodes = new(numThisChild);
          using (new SearchContextExecutionBlock(thisContext))
          {
            DoGatherLeafBatchlet(thisChild, numThisChild, vLossDynamicBoost, gatheredNodes);
          }

          // Append nodes to node list.
          lock (leafs)
          {
            leafs.Add(gatheredNodes);
          }

          // Signal done.
          if (TrackWaitCount)
          {
            countdownPendingNumLeafs.Signal(numThisChild);
          }
        }
        catch (Exception exc)
        {
          Console.WriteLine("Internal error: worker failed " + exc + " " + exc.InnerException);
        }
      };

      if (installedThreadPool != null)
      {
        installedThreadPool.QueueUserWorkItem(action);
      }
      else if (USE_CUSTOM_THREADPOOL)
      {
        tpm.QueueUserWorkItem(action);
      }
      else
      {
        ThreadPool.QueueUserWorkItem(action);
      }
    }

#endregion

#region Internals

    /// <summary>
    /// Diagnostic method that verifies internal consistency of child visit counts.
    /// </summary>
    /// <param name="numChildrenToCheck"></param>
    /// <param name="childVisitCounts"></param>
    /// <param name="numTargetLeafs"></param>
    [Conditional("DEBUG")]
    static void VerifyTargetLeafsCorrect(int numChildrenToCheck, Span<short> childVisitCounts, int numTargetLeafs)
    {
      if (numChildrenToCheck > 0)
      {
        int numChildrenFound = 0;
        for (int i = 0; i < numChildrenToCheck; i++)
          numChildrenFound += childVisitCounts[i];
        if (numTargetLeafs != numChildrenFound)
          throw new Exception("Internal error");
      }
    }

#endregion

  }
}
