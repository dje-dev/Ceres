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
using System.Threading.Tasks;
using System.Diagnostics;

using Ceres.Base.DataTypes;
using Ceres.Base.Environment;

using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Params;
using Ceres.MCTS.Utils;
using Ceres.MCTS.Environment;

#endregion

namespace Ceres.MCTS.Search
{
  /// <summary>
  /// Primary coordinator of an MCTS search, orchestrating the
  /// dual overlapped selectors.
  /// 
  /// See the comments below for a sketch of the batch gathering/submission algorithm.
  /// </summary>
  public partial class MCTSSearchFlow
  {
    public static float LastNNIdleTimeSecs = 0;
    public static float TotalNNIdleTimeSecs = 0;
    public static float TotalNNWaitTimeSecs = 0;

    const bool DUMP_WAITING = false;

    void WaitEvaluationDoneAndApply(Task<MCTSNodesSelectedSet> finalizingTask, int curCount = 0)
    {
      if (finalizingTask != null)
      {
        DateTime timeStartedWait = default;

        bool waited = false;
        if (finalizingTask.IsCompleted)
        {
          if (timeLastNNFinished != default)
          {
            float lastNNIdleTimeSecs = (float)(DateTime.Now - timeLastNNFinished).TotalSeconds;
            LastNNIdleTimeSecs = lastNNIdleTimeSecs;
            TotalNNIdleTimeSecs += lastNNIdleTimeSecs;
          }

          if (DUMP_WAITING) Console.WriteLine($"Wait {LastNNIdleTimeSecs * 1000.0f,6:F2}ms GC={GC.CollectionCount(0)} cur= {curCount} last= ");
          waited = true;
        }
        else
        {
          if (DUMP_WAITING) Console.WriteLine($"Nowait ms GC={GC.CollectionCount(0)} cur= {curCount} last= ");

          LastNNIdleTimeSecs = 0;

          // Since we are waiting here for at least some amount of time,
          // give hint to garbage collector that it might want to 
          // do a collection now if getting close to the need
          GC.Collect(1, GCCollectionMode.Optimized, true);

          if (!finalizingTask.IsCompleted)
          {
            if (CeresEnvironment.MONITORING_METRICS) timeStartedWait = DateTime.Now;
          }
        }

        finalizingTask.Wait();

        MCTSNodesSelectedSet resultNodes = finalizingTask.Result;

        if (timeStartedWait != default)
        {
          TotalNNWaitTimeSecs += (float)(DateTime.Now - timeStartedWait).TotalSeconds;
        }

        if (DUMP_WAITING && waited && resultNodes != null)
        {
          Console.WriteLine(resultNodes.NodesNN.Count);
        }

        if (resultNodes != null)
        {
          resultNodes.ApplyAll();
        }

      }
    }


    // Algorithm
    //   priorEvaluateTask <- null
    //   priorNodesNN <- null
    //   do
    //   { 
    //     // Select new nodes
    //     newNodes <- Select()
    //    
    //     // Remove any which may have been already selected by alternate selector
    //     newNodes <- Deduplicate(newNodes, priorNodesNN)
    //     
    //     // Check for those that can be immediately evaluated and split them out
    //     (newNodesNN, newNodesImm) <- TryEvalImmediateAndPartition(newNodes)
    //     if (OUT_OF_ORDER_ENABLED) BackupApply(newNodesImm) 
    //     
    //     // Launch evaluation of new nodes which need NN evaluation
    //     newEvaluateTask <- new Task(Evaluate(newNodesNN))
    //     
    //     // Wait for prior NN evaluation to finish and apply nodes
    //     if (priorEvaluateTask != null)
    //     {
    //       priorNodesNN <- Wait(priorEvaluateTask)
    //       BackupApply(priorNodesNN)
    //     }
    //     
    //     if (!OUT_OF_ORDER_ENABLED) BackupApply(newNodesImm)
    //
    //     // Prepare to cycle again       
    //     priorEvaluateTask <- newEvaluateTask
    //   } until (end of search)
    // 
    //   // Finalize last batch
    //   priorNodesNN <- Wait(priorEvaluateTask)
    //   BackupApply(priorNodesNN)


    public void ProcessDirectOverlapped(MCTSManager manager, int hardLimitNumNodesToCompute, 
                                        int startingBatchSequenceNum, int? forceBatchSize)
    {
      Debug.Assert(!manager.Root.IsInFlight);
      if (hardLimitNumNodesToCompute == 0)
      {
        hardLimitNumNodesToCompute = 1;
      }

      MCTSNode rootNode = Context.Root;

      bool overlappingAllowed = Context.ParamsSearch.Execution.FlowDirectOverlapped;
      int initialRootN = rootNode.N;
      int maxBatchSize = Math.Min(Context.NNEvaluators.MaxBatchSize, Context.ParamsSearch.Execution.MaxBatchSize);

      int guessMaxNumLeaves = maxBatchSize;

      ILeafSelector selector1;
      ILeafSelector selector2;

      selector1 = new LeafSelectorMulti(Context, 0, Context.StartPosAndPriorMoves, guessMaxNumLeaves);
      int secondSelectorID = Context.ParamsSearch.Execution.FlowDualSelectors ? 1 : 0;
      selector2 = overlappingAllowed ? new LeafSelectorMulti(Context, secondSelectorID, Context.StartPosAndPriorMoves, guessMaxNumLeaves) : null;

      MCTSNodesSelectedSet[] nodesSelectedSets = new MCTSNodesSelectedSet[overlappingAllowed ? 2 : 1];
      for (int i = 0; i < nodesSelectedSets.Length; i++) 
      {
        nodesSelectedSets[i] = new MCTSNodesSelectedSet(Context, 
                                                        i == 0 ? (LeafSelectorMulti)selector1 
                                                               : (LeafSelectorMulti)selector2,
                                                        guessMaxNumLeaves, guessMaxNumLeaves, BlockApply,
                                                        Context.ParamsSearch.Execution.InFlightThisBatchLinkageEnabled,
                                                        Context.ParamsSearch.Execution.InFlightOtherBatchLinkageEnabled);
      }
      
      int selectorID = 0;
      int batchSequenceNum = startingBatchSequenceNum;

      Task<MCTSNodesSelectedSet> overlappingTask = null;
      MCTSNodesSelectedSet pendingOverlappedNodes = null;
      int numOverlappedNodesImmediateApplied = 0;

      int iterationCount = 0;
      int numSelected = 0;
      int nodesLastSecondaryNetEvaluation = 0;
      while (true)
      {
        // If tree size is currently sufficiently small and requested in definition,
        // force the NN evaluators to use the secondary network.
        bool forceSecondaryEvaluator = false;
        if (Context.ParamsSearch.ParamsSecondaryEvaluator != null
          && rootNode.N < Context.ParamsSearch.ParamsSecondaryEvaluator.InitialTreeNodesForceSecondary)
          forceSecondaryEvaluator = true;
        BlockNNEval1.Evaluator.EvaluateUsingSecondaryEvaluator = forceSecondaryEvaluator;
        if (BlockNNEval2 != null)
        {
          BlockNNEval2.Evaluator.EvaluateUsingSecondaryEvaluator = forceSecondaryEvaluator;
        }

        // Apply search moves as soon as possible (need the root to have been evaluated).
        if (!manager.TerminationManager.HaveAppliedSearchMoves
         && rootNode.N > 0)
        {
          manager.TerminationManager.ApplySearchMoves();
        }

        // Only start overlapping past 1500 nodes because
        // CPU latency will be very small at small tree sizes,
        // obviating the overlapping beneifts of hiding this latency.
        bool overlapThisSet = overlappingAllowed && numSelected > 1500;

        iterationCount++;
        Context.Tree.NodeCache.NextBatchSequenceNumber++;

        ILeafSelector selector = selectorID == 0 ? selector1 : selector2;

        float thisBatchDynamicVLossBoost = batchingManagers[selectorID].VLossDynamicBoostForSelector();

        // Call progress callback and check if reached search limit
        Context.ProgressCallback?.Invoke(manager);
        Manager.UpdateSearchStopStatus();
        if (Manager.StopStatus != MCTSManager.SearchStopStatus.Continue)
        {
          break;
        }

        int numCurrentlyOverlapped = rootNode.NInFlight + rootNode.NInFlight2;

        int numApplied = rootNode.N - initialRootN;
        int hardLimitNumNodesThisBatch = int.MaxValue;
        if (hardLimitNumNodesToCompute > 0)
        {
          // Subtract out number already applied or in flight
          hardLimitNumNodesThisBatch = hardLimitNumNodesToCompute - (numApplied + numCurrentlyOverlapped);

          // Stop search if we have already exceeded search limit
          // or if remaining number is very small relative to full search
          // (this avoids incurring latency with a few small batches at end of a search).
          if (hardLimitNumNodesThisBatch <= numApplied / 1000)
          {
            int numNodesComputed = rootNode.N - initialRootN;
            Manager.StopStatus = MCTSManager.SearchStopStatus.SearchLimitExceeded;
            break;
          }
        }

        int targetThisBatch = OptimalBatchSizeCalculator.CalcOptimalBatchSize(Manager.EstimatedNumSearchNodes, rootNode.N,
                                                                              overlapThisSet,                                                                    
                                                                              Context.ParamsSearch.Execution.FlowDualSelectors,
                                                                              maxBatchSize,
                                                                              Context.ParamsSearch.BatchSizeMultiplier,
                                                                              Context.ParamsSearch);


        targetThisBatch = Math.Min(targetThisBatch, Manager.MaxBatchSizeDueToPossibleNearTimeExhaustion);
        if (forceBatchSize.HasValue) targetThisBatch = forceBatchSize.Value;
        if (targetThisBatch > hardLimitNumNodesThisBatch)
        {
          targetThisBatch = hardLimitNumNodesThisBatch;
        }

        int thisBatchTotalNumLeafsTargeted = 0;

        // Compute number of dynamic nodes to add (do not add any when tree is very small and impure child selection is particularly deleterious)
        int numNodesPadding = 0;
        if (rootNode.N > 50 && manager.Context.ParamsSearch.PaddedBatchSizing)
          numNodesPadding = manager.Context.ParamsSearch.PaddedExtraNodesBase
                          + (int)(targetThisBatch * manager.Context.ParamsSearch.PaddedExtraNodesMultiplier);
        int numVisitsTryThisBatch = targetThisBatch + numNodesPadding;

        numVisitsTryThisBatch = (int)(numVisitsTryThisBatch * batchingManagers[selectorID].BatchSizeDynamicScaleForSelector());
        numVisitsTryThisBatch = Math.Max(1, numVisitsTryThisBatch);

        // Select a batch using this selector
        // It will select a set of Leafs completely independent of what a possibly other selector already selected
        // It may find some unevaluated leafs in the tree (extant but N = 0) due to action of the other selector
        // These leafs will nevertheless be recorded but specifically ignored later
        MCTSNodesSelectedSet nodesSelectedSet = nodesSelectedSets[selectorID];
        nodesSelectedSet.Reset(pendingOverlappedNodes);

        // Select the batch of nodes  
        if (numVisitsTryThisBatch < 5 || !Context.ParamsSearch.Execution.FlowSplitSelects)
        {
          // Possibly compute a smart sized batch
          SetMaxNNNodesUsingSmartSizing(nodesSelectedSet, (int)(targetThisBatch * 0.8f));

          thisBatchTotalNumLeafsTargeted += numVisitsTryThisBatch;
          ListBounded<MCTSNode> selectedNodes = selector.SelectNewLeafBatchlet(rootNode, numVisitsTryThisBatch, thisBatchDynamicVLossBoost);
          nodesSelectedSet.AddSelectedNodes(selectedNodes, true);
        }
        else
        {
          // Set default assumed max batch size
          nodesSelectedSet.MaxNodesNN = numVisitsTryThisBatch;

          // In first attempt try to get 60% of target, second 40%
          int numTry1 = Math.Max(1, (int)(numVisitsTryThisBatch * 0.60f));
          int numTry2 = (int)(numVisitsTryThisBatch * 0.40f);
          thisBatchTotalNumLeafsTargeted += numTry1;

          // Possibly compute a smart sized batch
          SetMaxNNNodesUsingSmartSizing(nodesSelectedSet, (int)(numTry1 * 0.9f));

          ListBounded<MCTSNode> selectedNodes1 = selector.SelectNewLeafBatchlet(rootNode, numTry1, thisBatchDynamicVLossBoost);
          nodesSelectedSet.AddSelectedNodes(selectedNodes1, true);
          int numGot1 = nodesSelectedSet.NumNewLeafsAddedNonDuplicates;
          nodesSelectedSet.ApplyImmeditateNotYetApplied();

          // Only try to collect the second half of the batch if the first one yielded
          // a good fraction of desired nodes (otherwise too many collisions to profitably continue)
          const float THRESHOLD_SUCCESS_TRY1 = 0.667f;
          bool shouldProcessTry2 = numTry1 < 10 || ((float)numGot1 / (float)numTry1) >= THRESHOLD_SUCCESS_TRY1;
          if (shouldProcessTry2)
          {
            // Possibly compute a smart sized batch
            SetMaxNNNodesUsingSmartSizing(nodesSelectedSet, EstAdditionalNNNodesForTry2(nodesSelectedSet, numTry1, numTry2));

            thisBatchTotalNumLeafsTargeted += numTry2;
            ListBounded<MCTSNode> selectedNodes2 = selector.SelectNewLeafBatchlet(rootNode, numTry2, thisBatchDynamicVLossBoost);

            // TODO: clean this up
            //  - Note that ideally we might not apply immeidate nodes here (i.e. pass false instead of true in next line)
            //  - This is because once done selecting nodes for this batch, we want to get it launched as soon as possible,
            //    we could defer and call ApplyImmeditateNotYetApplied only later (below)
            // *** WARNING*** However, setting this to false causes NInFlight errors (seen when running test matches within 1 or 2 minutes)
            nodesSelectedSet.AddSelectedNodes(selectedNodes2, true); // MUST BE true; see above
          }
        }

        nodesSelectedSet.ApplyBatchSizeBreakHints();

#if FEATURE_SUPPLEMENTAL
        //if (Context.ParamsSearch.TestFlag)
        {
          TryAddSupplementalNodes(manager, MAX_PRELOAD_NODES_PER_BATCH, nodesSelectedSet, selector);
        }
#endif

        // TODO: make flow private belows   
        if (Context.EvaluatorDefSecondary != null)
        {
          ParamsSearchSecondaryEvaluator secondaryParams = Context.ParamsSearch.ParamsSecondaryEvaluator;
          int minBatchSize = secondaryParams.MinBatchSize(rootNode.N);
          if (Context.PendingSecondaryNodes.Count >= minBatchSize)
          {
            manager.RunSecondaryNetEvaluations(0, manager.flow.BlockNNEvalSecondaryNet);
            Context.PendingSecondaryNodes.Clear();
          }
#if NOT
          int numNodesElapsed = rootNode.N - nodesLastSecondaryNetEvaluation;
          bool satisfiesAbsoluteMinN = numNodesElapsed >= secondaryParams.UpdateFrequencyMinNodesAbsolute;
          bool satisfiesRelativeMinN = numNodesElapsed >= secondaryParams.UpdateFrequencyMinNodesRelative * rootNode.N;

          if (satisfiesAbsoluteMinN && satisfiesRelativeMinN)
          {
            int minN = (int)(secondaryParams.UpdateMinNFraction * rootNode.N);
            manager.RunSecondaryNetEvaluations(minN, manager.flow.BlockNNEvalSecondaryNet);
            nodesLastSecondaryNetEvaluation = rootNode.N;
          }
#endif
        }

        // Update statistics
        numSelected += nodesSelectedSet.NumNewLeafsAddedNonDuplicates;
        UpdateStatistics(selectorID, thisBatchTotalNumLeafsTargeted, nodesSelectedSet);

        // Convert any excess nodes to CacheOnly
        if (Context.ParamsSearch.PaddedBatchSizing)
        {
          throw new Exception("Needs remediation");
          // Mark nodes not eligible to be applied as "cache only"
          //for (int i = numApplyThisBatch; i < selectedNodes.Count; i++)
          //  selectedNodes[i].ActionType = MCTSNode.NodeActionType.CacheOnly;
        }

        CeresEnvironment.LogInfo("MCTS", "Batch", $"Batch Target={numVisitsTryThisBatch} "
                                 + $"yields NN={nodesSelectedSet.NodesNN.Count} Immediate= {nodesSelectedSet.NodesImmediateNotYetApplied.Count} "
                                 + $"[CacheOnly={nodesSelectedSet.NumCacheOnly} None={nodesSelectedSet.NumNotApply}]", manager.InstanceID);

        // Now launch NN evaluation on the non-immediate nodes
        bool isPrimary = selectorID == 0;
        if (overlapThisSet)
        {
          Task<MCTSNodesSelectedSet> priorOverlappingTask = overlappingTask;

          numOverlappedNodesImmediateApplied = nodesSelectedSet.NodesImmediateNotYetApplied.Count;

          // Launch a new task to preprocess and evaluate these nodes
          overlappingTask = Task.Run(() => LaunchEvaluate(manager, targetThisBatch, isPrimary, nodesSelectedSet));
          nodesSelectedSet.ApplyImmeditateNotYetApplied();
          pendingOverlappedNodes = nodesSelectedSet;

          WaitEvaluationDoneAndApply(priorOverlappingTask, nodesSelectedSet.NodesNN.Count);
        }
        else
        {
          LaunchEvaluate(manager, targetThisBatch, isPrimary, nodesSelectedSet);
          nodesSelectedSet.ApplyAll();
          //Console.WriteLine("applied " + selector.Leafs.Count + " " + rootNode);
        }

        RunPeriodicMaintenance(manager, batchSequenceNum, iterationCount);

        // Advance (rotate) selector
        if (overlappingAllowed) selectorID = (selectorID + 1) % 2;
        batchSequenceNum++;
      }

      WaitEvaluationDoneAndApply(overlappingTask);

      //      Debug.Assert(!manager.Root.IsInFlight);

      if ((rootNode.NInFlight != 0 || rootNode.NInFlight2 != 0) && !haveWarned)
      {
        Console.WriteLine($"Internal error: search ended with N={rootNode.N} NInFlight={rootNode.NInFlight} NInFlight2={rootNode.NInFlight2} " + rootNode);
        int count = 0;
        rootNode.StructRef.TraverseSequential(rootNode.Store, delegate (ref MCTSNodeStruct node, MCTSNodeStructIndex index)
        {
          if (node.IsInFlight && node.NumChildrenVisited == 0 && count++ < 20)
          {
            Console.WriteLine("  " + index.Index + " " + node.Terminal + " " + node.N + " " + node.IsTranspositionLinked + " " + node.NumVisitsPendingTranspositionRootExtraction);
          }
          return true;
        });
        haveWarned = true;
      }

      selector1.Shutdown();
      selector2?.Shutdown();
    }

    private void SetMaxNNNodesUsingSmartSizing(MCTSNodesSelectedSet nodesSelectedSet, int estimatedNNNodes)
    {
      // TODO: Handle this for multiple GPUs, at least the simple case when they are homogeneous.
      if (Context.ParamsSearch.Execution.SmartSizeBatches
       && Context.EvaluatorDef.NumDevices == 1
       && Context.NNEvaluators.PerfStatsPrimary != null) 
      {
        int[] optimalBatchSizeBreaks;
        if (Context.NNEvaluators.PerfStatsPrimary.Breaks != null)
        {
          optimalBatchSizeBreaks = Context.NNEvaluators.PerfStatsPrimary.Breaks;
        }
        else
        {
          optimalBatchSizeBreaks = Context.GetOptimalBatchSizeBreaks(Context.EvaluatorDef.DeviceIndices[0]);
        }

        // Fallback to any break value which is not more than 20% below current estimate
        const float NEARBY_BREAK_FRACTION = 0.20f;
        int? closeByBreak = NearbyBreak(optimalBatchSizeBreaks, estimatedNNNodes, NEARBY_BREAK_FRACTION);
        if (closeByBreak is not null)
        {
          nodesSelectedSet.MaxNodesNN = nodesSelectedSet.NodesNN.Count + closeByBreak.Value;
          //Console.WriteLine($"Selecting total {numTry2} est nn {estimatedAdditionalNNNodesTry2} using break {closeByBreak} total NN max {nodesSelectedSet.MaxNodesNN}");
        }

      }
    }

    private static int EstAdditionalNNNodesForTry2(MCTSNodesSelectedSet nodesSelectedSet, int numTry1, int numTry2)
    {
      // Make an educated guess about the total number of NN nodes that will found in try2
      // We base this on the fraction of nodes in try1 which actually are going to NN
      // then discounted by 0.8 because the yield on the second try is typically lower
      const float TRY2_SUCCESS_DISCOUNT_FACTOR = 0.8f;
      float fracNodesFirstTryGoingToNN = (float)nodesSelectedSet.NodesNN.Count / (float)numTry1;
      int estimatedAdditionalNNNodesTry2 = (int)(numTry2 * fracNodesFirstTryGoingToNN * TRY2_SUCCESS_DISCOUNT_FACTOR);
      return estimatedAdditionalNNNodesTry2;
    }


    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    static int? NearbyBreak(int[] breaks, int value, float maxDeviationFractionUpOrDown)
    {
      // Nothing to do if no breaks available
      if (breaks == null) return null;

      float min = value - value * maxDeviationFractionUpOrDown;
      float max = value + value * maxDeviationFractionUpOrDown;

      // Find break value which largest break less than targetTotalNumNodes
      for (int i = 0; i < breaks.Length; i++)
      {
        if (breaks[i] >= min && breaks[i] < max)
        {
          return breaks[i];
        }
      }

      return null;
    }


    private void UpdateStatistics(int selectorID, int numLeafsAttempted, MCTSNodesSelectedSet nodesSet)
    {
      if (nodesSet.NumNewLeafsAddedNonDuplicates > 0)
      {
        Context.NumNodeVisitsAttempted += numLeafsAttempted;
        Context.NumNodeVisitsSucceeded += nodesSet.NumNewLeafsAddedNonDuplicates;

        MCTSIterator.totalNumNodeVisitsAttempted += numLeafsAttempted;
        MCTSIterator.totalNumNodeVisitsSucceeded+= nodesSet.NumNewLeafsAddedNonDuplicates;

        float lastYield = (float)nodesSet.NumNewLeafsAddedNonDuplicates / (float)numLeafsAttempted;
        MCTSIterator.LastBatchYieldFrac = lastYield;
        batchingManagers[selectorID].UpdateVLossDynamicBoost(numLeafsAttempted, lastYield);
      }
    }

    bool rootNodeHasBeenInitialized = false;

    private void RunPeriodicMaintenance(MCTSManager manager, int batchSequenceNum, int iterationCount)
    {
      if (!rootNodeHasBeenInitialized && manager.Root.NumPolicyMoves > 0)
      {
        // We can only apply search noise after first node (so children initialized)
        manager.PossiblySetSearchNoise();
        rootNodeHasBeenInitialized = true;
      }

      // Use this time to perform housekeeping (tree is quiescent)
      manager.TerminationManager.UpdatePruningFlags();
      if (batchSequenceNum % 3 == 2)
      {
        manager.UpdateEstimatedNPS();
      }

      // Check if node cache needs pruning.
      Context.Tree?.PossiblyPruneCache();
    }



    bool haveWarned = false;


#if FEATURE_SUPPLEMENTAL
    private void TryAddSupplementalNodes(MCTSManager manager, int maxNodes,
                                         MCTSNodesSelectedSet selectedNodes, ILeafSelector selector)
    {
      foreach ((MCTSNode parentNode, int selectorID, int childIndex) in ((LeafSelectorMulti)selector).supplementalCandidates) // TODO: remove cast
      {
        if (childIndex <= parentNode.NumChildrenExpanded -1)
        {
          // This child was already selected as part of the normal leaf gathering process.
          continue;
        }
        else
        {
          MCTSEventSource.TestCounter1++;

          // Record visit to this child in the parent (also increments the child NInFlight counter)
          parentNode.UpdateRecordVisitsToChild(selectorID, childIndex, 1);

          MCTSNode node = parentNode.CreateChild(childIndex);

          ((LeafSelectorMulti)selector).DoVisitLeafNode(node, 1);// TODO: remove cast

          if (!parentNode.IsRoot)
          {
            if (selectorID == 0)
              parentNode.Parent.Ref.BackupIncrementInFlight(1, 0);
            else
              parentNode.Parent.Ref.BackupIncrementInFlight(0, 1);
          }

          // Try to process this node
          int nodesBefore = selectedNodes.NodesNN.Count;
          selector.InsureAnnotated(node);
          selectedNodes.ProcessNode(node);
          bool wasSentToNN = selectedNodes.NodesNN.Count != nodesBefore;
          //if (wasSentToNN) MCTSEventSource.TestCounter2++;

          // dje: add counter?
        }
      }

    }
#endif

    enum ApplyMode { ApplyNone, ApplyIfImmediate, ApplyAll };

    DateTime timeLastNNFinished;


    private MCTSNodesSelectedSet LaunchEvaluate(MCTSManager manager, int numNodesTargeted,
                                                    bool isPrimary, MCTSNodesSelectedSet nodes)
    {
      if (nodes.NodesNN.Count == 0) return null;

      using (new SearchContextExecutionBlock(manager.Context))
      {
        if (nodes.NodesNN.Count > numNodesTargeted)
        {
          // Mark the excess nodes as "CacheOnly"
          for (int i = numNodesTargeted; i < nodes.NodesNN.Count; i++)
          {
            nodes.NodesNN[i].ActionType = MCTSNodeInfo.NodeActionType.CacheOnly;
          }
        }

        if (isPrimary)
        {
          BlockNNEval1.Evaluate(manager.Context, nodes.NodesNN);
        }
        else
        {
          BlockNNEval2.Evaluate(manager.Context, nodes.NodesNN);
        }

        timeLastNNFinished = DateTime.Now;
        return nodes;
      }

    }
  }
}
