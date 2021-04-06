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
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base;
using Ceres.Base.DataType;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Base.Math.Random;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Search
{
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
    // Using the custom threadpool implementation in Ceres is generally slightly more efficient 
    // (especially from the persective of total CPU time rather than elapsed time)
    // than the shared System.Threading.ThreadPool, partly because it only
    // needs to support the narrow set of requirements here 
    // (e.g. no need to dynamically shirnk or grow the pool).
    const bool USE_CUSTOM_THREADPOOL = true;
    
    #region Constructor arguments

    public readonly PositionWithHistory PriorSequence;
    public int SelectorID { get; private set; }

    #endregion

    #region Statistics
    public int CountChooseChild { get; private set; }

    #endregion

    #region Internal data

    CountdownEvent countdownPendingNumLeafs = USE_CUSTOM_THREADPOOL ? null : new CountdownEvent(1);

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

      if (USE_CUSTOM_THREADPOOL) tpm = tpmPool.Value.GetFromPool();

      SelectorID = selectorID;
      PriorSequence = priorSequence;
      paramsExecution = context.ParamsSearch.Execution;

      int maxNodesPerBatchForRootPreload = context.ParamsSearch.Execution.RootPreloadDepth > 0 ? MCTSSearchFlow.MAX_PRELOAD_NODES_PER_BATCH : 0;
      int extraLeafsDynamic = 0;
      if (context.ParamsSearch.PaddedBatchSizing)
        extraLeafsDynamic = context.ParamsSearch.PaddedExtraNodesBase + (int)(context.ParamsSearch.PaddedExtraNodesMultiplier * guessNumLeaves);

      leafs = new ListBounded<MCTSNode>(guessNumLeaves + maxNodesPerBatchForRootPreload + extraLeafsDynamic);

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
      if (tpm != null) tpmPool.Value.RestoreToPool(tpm);
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
    ListBounded<MCTSNode> DoSelectNewLeafBatchlet(MCTSNode root, int numTargetLeafs, float vLossDynamicBoost)
    {
      InsureAnnotated(root);
      DoGatherLeafBatchlet(root, numTargetLeafs, vLossDynamicBoost);

      if (paramsExecution.SelectParallelEnabled)
        WaitDone();

      return leafs;
    }

#endregion

#region Clear

    /// <summary>
    /// Resets state of selector to be prepared for 
    /// selecting a new set of nodes.
    /// </summary>
    public void Reset()
    {
      CountChooseChild = 0;
    
      leafs.Clear();
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
        Context.Tree.Annotate(node);
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
      if (USE_CUSTOM_THREADPOOL)
        tpm.WaitDone();
      else
      {
        countdownPendingNumLeafs.Signal(); // take out initialization value of 1
        countdownPendingNumLeafs.Wait();
        countdownPendingNumLeafs.Reset();
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

    internal void DoVisitLeafNode(MCTSNode node, int numVisits)
    {
      ref MCTSNodeStruct nodeRef = ref node.Ref;

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

        // Verify this looks like a true non-leaf
        Debug.Assert(nodeRef.N == 0
                          || nodeRef.Terminal != Chess.GameResult.Unknown
                          || nodeRef.NumNodesTranspositionExtracted > 0
                          || !FP16.IsNaN(node.OverrideVToApplyFromTransposition));

        // Set default action 
        node.ActionType = MCTSNode.NodeActionType.MCTSApply;

        // if (node.Ref.NumNodesTranspositionExtracted > 0)
        //    Console.WriteLine($"{node.Ref.NumNodesTranspositionExtracted}  transposition multivisted {node}");

        // Add to set of leafs
        if (paramsExecution.SelectParallelEnabled)
        {
          lock (leafs) leafs.Add(node);
        }
        else
          leafs.Add(node);
      }

    }


#endregion

#region Selection algorithm


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
      CountChooseChild++;

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
      Span<float> scores = default;
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

      node.ComputeTopChildScores(selectorID, node.Depth,
                                 vLossDynamicBoost, 0, numChildrenToCheck - 1, numTargetLeafs,
                                 scores, visitChildCounts);

      if (node.Depth == 0)
      {
        Context.RootMoveTracker?.UpdateVisitCounts(visitChildCounts, numChildrenToCheck, numTargetLeafs);
      }
    }


    static int nextWorkerPoolToReceiveSelect = 0;
    static int abandonCount = 0;

    float TopNFractionToTopQMove
    {
      get
      {
        MCTSNode topMove = Context.Root.BestMove(false);
        if (topMove == null || topMove.NumPolicyMoves <= 1) return 1.0f;

        MCTSNode[] childrenSorted = topMove.ChildrenSorted(c => (float)-c.N);
        return (float)childrenSorted[0].N / (float)topMove.N;
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
    private void DoGatherLeafBatchlet(MCTSNode node, int numTargetLeafs, float vLossDynamicBoost)
    {
      ref MCTSNodeStruct nodeRef = ref node.Ref;

      //      Console.WriteLine($"target {numTargetLeafs} {node}");
      Debug.Assert(numTargetLeafs > 0);

      if (paramsExecution.TranspositionMode == TranspositionMode.SingleNodeDeferredCopy
       && node.NumNodesTranspositionExtracted > 0)
      {
        InitializeChildrenFromDeferredTransposition(node);
      }
      else if (paramsExecution.TranspositionMode == TranspositionMode.SharedSubtree
            && node.NumNodesTranspositionExtracted > 0)
      {
        InitializeChildrenFromDeferredTransposition(node);
      }

      bool isUnvisited = node.N == 0;
      if (isUnvisited || nodeRef.Terminal.IsTerminal() || nodeRef.IsTranspositionLinked)
      {
        DoVisitLeafNode(node, numTargetLeafs);
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
            node.OverrideVToApplyFromTransposition = vToUse;

            throw new Exception("need to restore following 3 lines, currently disabled since mSum is marked private");
            //FP16 mToUse = (FP16)((float)(biggestTranspositionNode.mSum - node.Ref.mSum)
            //                     / (float)(biggestTranspositionNode.N - node.N));
            //node.OverrideMPositionToApplyFromTransposition = mToUse;

            DoVisitLeafNode(node, numTargetLeafs);
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
                node.Parent.Ref.BackupDecrementInFlight(SelectorID == 0 ? numTargetLeafs : 0,
                                                  SelectorID == 1 ? numTargetLeafs : 0);
              }
              return;
            }
            else
            {
              // We assume the role of the new master 
              MCTSNodeStructIndex indexTranspositionRoot = biggestTranspositionNode.Index;
              MCTSNodeStructIndex indexThis = new MCTSNodeStructIndex(node.Index);

              MCTSNodeStructStorage.ModifyParentsChildRef(node.Context.Tree.Store, indexTranspositionRoot, indexThis);
              MCTSNodeStructStorage.ModifyParentsChildRef(node.Context.Tree.Store, indexThis, indexTranspositionRoot);

              // Swap parents
              MCTSNodeStructIndex saveIndexThisParent = node.Ref.ParentIndex;
              nodeRef.ParentIndex = biggestTranspositionNode.ParentIndex;
              biggestTranspositionNode.ParentIndex = saveIndexThisParent;

              node = node.Context.Tree.GetNode(biggestTranspositionNode.Index);
            }

          }

        }
      }

      // Mark node as visited, make sure we get associated annotation
      DoVisitInnerNode(node);

      node.Ref.PossiblyPrefetchChildArray(node.Context.Tree.Store, new MCTSNodeStructIndex(node.Index));

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

      Span<MCTSNodeStructChild> children = nodeRef.ChildrenFromStore(node.Context.Tree.Store);

      int numVisitsProcessed = 0;
      if (paramsExecution.SelectParallelEnabled)
      {
        // Immediately make a first pass to immediately launch the children
        // that have enough visits to be processed in parallel  
        ProcessSelectedChildren(node, numTargetLeafs, vLossDynamicBoost, numChildrenToCheck,
                                paramsExecution.SelectParallelThreshold, true,
                                childVisitCounts, children, ref numVisitsProcessed);

        // Make a second pass process any remaining chidren having visits (not parallel)
        if (numVisitsProcessed < numTargetLeafs)
        {
          ProcessSelectedChildren(node, numTargetLeafs, vLossDynamicBoost, numChildrenToCheck,
                                  1, false, childVisitCounts, children, ref numVisitsProcessed);
        }

      }
      else
      {
        // Process all children with nonzero counts (not parallel)
        ProcessSelectedChildren(node, numTargetLeafs, vLossDynamicBoost, numChildrenToCheck,
                                1, false, childVisitCounts, children, ref numVisitsProcessed);
      }
    }


    void ProcessSelectedChildren(MCTSNode node, int numTargetLeafs, float vLossDynamicBoost, int numChildrenToCheck,
                                 int minVisitsCountToProcess, bool launchParallel,
                                 Span<short> childVisitCounts, Span<MCTSNodeStructChild> children,
                                 ref int numVisitsProcessed)
    {
      MCTSNodeStruct nodeRef = node.Ref;

      for (int childIndex = 0; childIndex < numChildrenToCheck; childIndex++)
      {
        int numThisChild = childVisitCounts[childIndex];
        if (numThisChild >= minVisitsCountToProcess)
        {
          if (childIndex > node.NumChildrenExpanded)
          {
            Console.WriteLine("Saw (and ignoring) out of order possibly due to tie in score");
            continue;
          }
          MCTSNode thisChild = default;

          // NOTE: for unclear reasons, the lock here cannot be limited
          //       to the (potetial call to CreateChild)
          //       because that results in fairly rapid curruption when parallel and transposition both enabled
          //       (to see this, call store.Validate at the beginning of each batch and run few a few minutes).
          using (node.Tree.ChildCreateLocks.LockBlock(node.Index))
          {
            MCTSNodeStructChild childInfo = children[childIndex];
            if (!childInfo.IsExpanded)
            {
              //lock (node.Context.ChildCreateLockObjs[node.Index % SearchContext.NUM_LOCK_OBJ])
              {
                thisChild = node.CreateChild(childIndex);
                node.UpdateRecordVisitsToChild(SelectorID, childIndex, numThisChild);
              }
            }
            else
            {
              thisChild = node.Child(childInfo);
              node.UpdateRecordVisitsToChild(SelectorID, childIndex, numThisChild);
            }

            nodeRef.PossiblyPrefetchChild(node.Context.Tree.Store, new MCTSNodeStructIndex(node.Index), childIndex);
          }

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
            LaunchGatherLeafBatchletParallel(node, numThisChild, thisChild, vLossDynamicBoost);
          else
            DoGatherLeafBatchlet(thisChild, numThisChild, vLossDynamicBoost);

          // mark this child as done
          childVisitCounts[childIndex] = 0;

          numVisitsProcessed += numThisChild;
          if (numVisitsProcessed == numTargetLeafs)
            break;
        }
      }
    }


    private void InitializeChildrenFromDeferredTransposition(MCTSNode node)
    {
      // If this was a deferred single node copy,
      // the node was already visited once and the V was extracted, but
      // the extraction of the children from the tranposition root was deferred 
      // (because possibly it would never be required)
      // Now we will need the children to continue leaf selection, so copy them over now ("just in time")
      Debug.Assert(node.NumNodesTranspositionExtracted == 1);

      int transpositionNodeIndex = node.TranspositionRootIndex;

#if EXPERIMENTAL
      if (node.Annotation != null && !node.Context.TranspositionRoots.ContainsKey(node.Annotation.PositionHashForCaching))
      {
        Console.WriteLine(node.Annotation.PositionHashForCaching);
        node.Annotation = null;
        node.Context.Annotater.Annotate(node);
        Console.WriteLine(node.Annotation.PositionHashForCaching);
        var otherNode = new MCTSNode(node.Context, new MCTSNodeStructIndex(transpositionNodeIndex));
        var otherAnn = node.Context.Annotater.Annotate(otherNode);
        Console.WriteLine(node.Annotation.Pos.FEN);
        Console.WriteLine(otherAnn.Pos.FEN);
        Console.WriteLine(otherAnn.PositionHashForCaching);
        Console.WriteLine("num roots " + node.Context.TranspositionRoots.Count);
      }
#endif

      // Copy children
      node.Ref.CopyUnexpandedChildrenFromOtherNode(node.Tree, new MCTSNodeStructIndex(transpositionNodeIndex));
    }


    private void LaunchGatherLeafBatchletParallel(MCTSNode node, int numThisChild, MCTSNode thisChild, float vLossDynamicBoost)
    {
      if (!USE_CUSTOM_THREADPOOL) countdownPendingNumLeafs.AddCount(numThisChild);

      MCTSIterator thisContext = node.Context;

      WaitCallback action = (object obj) =>
      {
        try
        {
          using (new SearchContextExecutionBlock(thisContext))
          {
            DoGatherLeafBatchlet(thisChild, numThisChild, vLossDynamicBoost);
          }
          if (!USE_CUSTOM_THREADPOOL) countdownPendingNumLeafs.Signal(numThisChild);
        }
        catch (Exception exc)
        {
          Console.WriteLine("Internal error: worker failed " + exc + " " + exc.InnerException);
        }
      };

      if (USE_CUSTOM_THREADPOOL)
        tpm.QueueUserWorkItem(action);
      else
        ThreadPool.QueueUserWorkItem(action);
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
