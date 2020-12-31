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

using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Search;
using System.Diagnostics.Tracing;
using Ceres.MCTS.MTCSNodes;
using Ceres.Base.OperatingSystem;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.NNEvaluators.Internals;
using Ceres.MCTS.Managers;
using Ceres.MCTS.Iteration;

#endregion

namespace Ceres.MCTS.Environment
{
  /// <summary>
  /// The .NET Event source class that exposues
  /// continuously updated statistics relating to the Ceres engine.
  /// </summary>
  [EventSource(Guid= "6501e085-68df-43e6-8371-714c437e0638", Name = "Ceres.MCTS.Environment.MCTSEventSource")]
  public sealed class MCTSEventSource : EventSource
  {
    private static MCTSEventSource? mctsEventSource;

    private PollingCounter storageVirtualAllocBytes;

    PollingCounter numBatchesGPU0, numBatchesGPU1, numBatchesGPU2, numBatchesGPU3;
    PollingCounter numPositionsGPU0, numPositionsGPU1, numPositionsGPU2, numPositionsGPU3;


    private PollingCounter tablebaseHitsTotal;
    private IncrementingPollingCounter tablebaseHits;
    private PollingCounter tablebasePly1HitsTotal;

    private IncrementingPollingCounter numRootPreloadNodes;

    private PollingCounter makeNewRootTotalTimeSeconds;

    private PollingCounter lastBatchYield;
    private PollingCounter batchYield;

    private PollingCounter lastUnutilizedNNEvaluationTimeMS;
    private PollingCounter totalUnutilizedNNEvaluationTimeMS;
    private PollingCounter totalNNWaitingTimeMS;

    private PollingCounter totalNumMovesNoiseOverridden;

    private IncrementingPollingCounter searchNumBatches;
    private IncrementingPollingCounter searchNumApply;
    private IncrementingPollingCounter searchNumCache;

    private IncrementingPollingCounter numNodesApplied;
    private IncrementingPollingCounter numNodesAppliedInc;
    private IncrementingPollingCounter numNodesDualSelectorDuplicate;

    private PollingCounter numNodesSelectedIntoTreeCache;
    private PollingCounter numNodesAppliedFromTreeCache;


    private IncrementingPollingCounter dualSelectorAverageNNEvalWaitMS;

    private PollingCounter totalNumInFlightTranspositions;
    private PollingCounter nodeAnnotateCacheHitRate;
    private PollingCounter nnCacheHitRate;
    private PollingCounter opponentTreeReuseHitRate;
    private PollingCounter opponentTreeReuseNumHits;
    private PollingCounter nnTranspositionsHitRate;


    private PollingCounter mlhMoveModificationFraction;

    private PollingCounter numAnnotations;

    static bool initialized = false;
    public static void Initialize()
    {
      if (!initialized)
      {
        initialized = true;
        mctsEventSource = new MCTSEventSource();
      }
    }
    private MCTSEventSource() : base("Ceres.MCTS.Environment.MCTSEventSource")
    {
    }

    protected override void OnEventCommand(EventCommandEventArgs command)
    {
      if (command.Command == EventCommand.Enable)
      {
        storageVirtualAllocBytes ??= new PollingCounter("storage-virtual-alloc-bytes", this, () => WindowsVirtualAllocManager.BytesCurrentlyAllocated);
        
//        storageNodesAllocated ??= new PollingCounter("storage-nodes-allocated", this, () => MCTSNodeStructStorage.NumAllocatedNodes);
//        storageNodeChildrenAllocated ??= new PollingCounter("storage-node-children-allocated", this, () => MCTSNodeStructChildStorage.NumAllocatedNodes);

        numBatchesGPU0 ??= new PollingCounter("gpu-0-batches", this, () => NNEvaluatorStats.TotalBatchesPerGPU[0]);
        numBatchesGPU1 ??= new PollingCounter("gpu-1-batches", this, () => NNEvaluatorStats.TotalBatchesPerGPU[1]);
        numBatchesGPU2 ??= new PollingCounter("gpu-2-batches", this, () => NNEvaluatorStats.TotalBatchesPerGPU[2]);
        numBatchesGPU3 ??= new PollingCounter("gpu-3-batches", this, () => NNEvaluatorStats.TotalBatchesPerGPU[3]);

        numPositionsGPU0 ??= new PollingCounter("gpu-0-positions", this, () => NNEvaluatorStats.TotalPosEvaluationsPerGPU[0]);
        numPositionsGPU1 ??= new PollingCounter("gpu-1-positions", this, () => NNEvaluatorStats.TotalPosEvaluationsPerGPU[1]);
        numPositionsGPU2 ??= new PollingCounter("gpu-2-positions", this, () => NNEvaluatorStats.TotalPosEvaluationsPerGPU[2]);
        numPositionsGPU3 ??= new PollingCounter("gpu-3-positions", this, () => NNEvaluatorStats.TotalPosEvaluationsPerGPU[3]);


        numNodesApplied ??= new IncrementingPollingCounter("applied", this, () => MCTSApply.TotalNumNodesApplied);
        numNodesDualSelectorDuplicate ??= new IncrementingPollingCounter("mcts-selected-dual-duplicate", this, () => MCTSNodesSelectedSet.TotalNumDualSelectorDuplicates);

        numNodesSelectedIntoTreeCache ??= new PollingCounter("mcts-nodes-selected-into-tree-cache", this, () => MCTSNodesSelectedSet.TotalNumNodesSelectedIntoTreeCache);
        numNodesAppliedFromTreeCache ??= new PollingCounter("mcts-nodes-appled-from-tree-cache", this, () => MCTSNodesSelectedSet.TotalNumNodesAppliedFromTreeCache);
        
        lastBatchYield ??= new PollingCounter("yield_pct_last_batch", this, () => 100.0f * MCTSIterator.LastBatchYieldFrac);
        batchYield ??= new PollingCounter("yield-pct-total", this, () => 100.0f * MCTSIterator.TotalYieldFrac);
      
        numAnnotations ??= new PollingCounter("num-annotations", this, () => MCTSTree.NumAnnotations);

        lastUnutilizedNNEvaluationTimeMS ??= new PollingCounter("nn-idle-time-ms-last", this, () => 1000.0f * MCTSSearchFlow.LastNNIdleTimeSecs);
        totalUnutilizedNNEvaluationTimeMS ??= new PollingCounter("nn-idle-time-sec-total", this, () => MCTSSearchFlow.TotalNNIdleTimeSecs);
        totalNNWaitingTimeMS ??= new PollingCounter("nn-wait-time-sec-total", this, () => MCTSSearchFlow.TotalNNWaitTimeSecs);

        makeNewRootTotalTimeSeconds ??= new PollingCounter("mcts-make-new-root-total-secs", this, () => MCTSManager.TotalTimeSecondsInMakeNewRoot); ;

        totalNumMovesNoiseOverridden ??= new PollingCounter("noise_best_move_overrides_total", this, () => MCTSIterator.TotalNumMovesNoiseOverridden);
        
        nnCacheHitRate ??= new PollingCounter("mcts-nncache-hit-rate_pct", this, () => LeafEvaluatorCache.HitRatePct);
        opponentTreeReuseHitRate ??= new PollingCounter("mcts-opponent-tree-reuse-hit-rate-pct", this, () => 100.0f * LeafEvaluatorReuseOtherTree.HitRate);
        opponentTreeReuseNumHits ??= new PollingCounter("mcts-opponent-tree-reuse-hit-count", this, () => LeafEvaluatorReuseOtherTree.NumHits);

        nnTranspositionsHitRate ??= new PollingCounter("mcts-transposition-hit-rate_pct", this, () => LeafEvaluatorTransposition.HitRatePct);
        totalNumInFlightTranspositions ??= new PollingCounter("transpositions-in-flight-total", this, () => MCTSNodesSelectedSet.TotalNumInFlightTranspositions);
        numRootPreloadNodes ??= new IncrementingPollingCounter("mcts-root-preload-nodes", this, () => MCTSRootPreloader.TotalCumulativeRootPreloadNodes);

        mlhMoveModificationFraction ??= new PollingCounter("mcts-mlh-move-modified-pct", this, () => 100.0f * ManagerChooseRootMove.MLHMoveModifiedFraction);

        tablebaseHitsTotal ??= new PollingCounter("tablebase-hits-total", this, () => LC0DLLSyzygyEvaluator.NumTablebaseHits);
        tablebaseHits ??= new IncrementingPollingCounter("tablebase-hits", this, () => LC0DLLSyzygyEvaluator.NumTablebaseHits);
        tablebasePly1HitsTotal ??= new PollingCounter("tablebase-ply1-hits-total", this, () => LeafEvaluatorSyzygyPly1.NumHits);

        nodeAnnotateCacheHitRate = new PollingCounter("node-cache-hit-rate", this, () => MCTSTree.HitRate)
        {
          DisplayName = "node-annotation-cache-hit-rate"
        };

      }
    }
  }
}

