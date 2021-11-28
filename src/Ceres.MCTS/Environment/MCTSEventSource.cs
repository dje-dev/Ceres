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
using Ceres.MCTS.MTCSNodes.Struct;

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
    /// <summary>
    /// Spare counter reserved for temporary ad-hoc use when debugging new features.
    /// </summary>
    public static long TestCounter1 = 0;
    
    /// <summary>
    /// Spare counter reserved for temporary ad-hoc use when debugging new features.
    /// </summary>
    public static double TestMetric1 = 0;


    public static float MaximumTimeAllotmentOvershoot = -9999f;

    private PollingCounter instamoveCounter;

    private static MCTSEventSource? mctsEventSource;

    private PollingCounter timeAllotmentOvershootMax;

    private PollingCounter testCounter1, testMetric1;
    private PollingCounter searchCount;

    private PollingCounter storageVirtualAllocCurrentBytes;
    private PollingCounter storageVirtualAllocMaxBytes;

    PollingCounter numBatchesGPU0, numBatchesGPU1, numBatchesGPU2, numBatchesGPU3;
    PollingCounter numPositionsGPU0, numPositionsGPU1, numPositionsGPU2, numPositionsGPU3;


    private PollingCounter tablebaseHitsTotal;
    private IncrementingPollingCounter tablebaseHits;
    private PollingCounter tablebasePly1HitsTotal;

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
    private PollingCounter numNodesAppliedTotal;
#if DEBUG
    private IncrementingPollingCounter numNodesDualSelectorDuplicate;
#endif

    private PollingCounter numSecondaryEvaluations;
    private PollingCounter numSecondaryBatches;


    private IncrementingPollingCounter dualSelectorAverageNNEvalWaitMS;

    private PollingCounter nnCacheHitRate;
    private PollingCounter opponentTreeReuseHitRate;
    private PollingCounter opponentTreeReuseNumHits;
    private PollingCounter nnTranspositionsHitRate;
    private PollingCounter nnTranspositionCacheHitRate;
    private PollingCounter nnTranspositionsHitRateOldGeneration;
    private PollingCounter countSiblingEvaluationsUsed;


    private PollingCounter mlhMoveModificationFraction;

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
        testCounter1 ??= new PollingCounter("test-counter1", this, () => TestCounter1);
        testMetric1 ??= new PollingCounter("test-metric", this, () => TestMetric1);
        searchCount ??= new PollingCounter("mcts-search-count", this, () => MCTSearch.SearchCount);

        instamoveCounter ??= new PollingCounter("instamove-count", this, () => MCTSearch.InstamoveCount);

        storageVirtualAllocCurrentBytes ??= new PollingCounter("storage-alloc-bytes-cur", this, 
          () => SoftwareManager.IsWindows ? WindowsVirtualAllocManager.BytesCurrentlyAllocated : RawMemoryManagerIncrementalLinuxStats.BytesCurrentlyAllocated);
        storageVirtualAllocMaxBytes ??= new PollingCounter("storage-alloc-bytes-max", this,
          () => SoftwareManager.IsWindows ? WindowsVirtualAllocManager.MaxBytesAllocated : RawMemoryManagerIncrementalLinuxStats.MaxBytesAllocated);

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


        numNodesAppliedTotal ??= new PollingCounter("applied-primary-tot", this, () => MCTSApply.TotalNumNodesApplied);
        numSecondaryEvaluations ??= new PollingCounter("applied-secondary-tot", this, () => MCTSManager.NumSecondaryEvaluations);
        numSecondaryBatches ??= new PollingCounter("applied-secondary-batches", this, () => MCTSManager.NumSecondaryBatches);
        numNodesApplied ??= new IncrementingPollingCounter("applied", this, () => MCTSApply.TotalNumNodesApplied);
#if DEBUG
        numNodesDualSelectorDuplicate ??= new IncrementingPollingCounter("selected-dual-duplicate", this, () => MCTSNodesSelectedSet.TotalNumDualSelectorDuplicates);
#endif

        lastBatchYield ??= new PollingCounter("yield_pct_last_batch", this, () => 100.0f * MCTSIterator.LastBatchYieldFrac);
        batchYield ??= new PollingCounter("yield-pct-total", this, () => 100.0f * MCTSIterator.TotalYieldFrac);
      
        lastUnutilizedNNEvaluationTimeMS ??= new PollingCounter("nn-idle-time-ms-last", this, () => 1000.0f * MCTSSearchFlow.LastNNIdleTimeSecs);
        totalUnutilizedNNEvaluationTimeMS ??= new PollingCounter("nn-idle-time-sec-total", this, () => MCTSSearchFlow.TotalNNIdleTimeSecs);
        totalNNWaitingTimeMS ??= new PollingCounter("nn-wait-time-sec-total", this, () => MCTSSearchFlow.TotalNNWaitTimeSecs);

        makeNewRootTotalTimeSeconds ??= new PollingCounter("make-new-root-total-secs", this, () => MCTSManager.TotalTimeSecondsInMakeNewRoot); ;

//        totalNumMovesNoiseOverridden ??= new PollingCounter("noise_best_move_overrides_total", this, () => MCTSIterator.TotalNumMovesNoiseOverridden);
        
//        nnCacheHitRate ??= new PollingCounter("nncache-hit-rate_pct", this, () => LeafEvaluatorCache.HitRatePct);
        opponentTreeReuseHitRate ??= new PollingCounter("opponent-tree-reuse-hit-rate-pct", this, () => 100.0f * LeafEvaluatorReuseOtherTree.HitRate);
        opponentTreeReuseNumHits ??= new PollingCounter("opponent-tree-reuse-hit-count", this, () => LeafEvaluatorReuseOtherTree.NumHits.Value);

        nnTranspositionsHitRate ??= new PollingCounter("transposition-store-hit-rate_pct", this, () => LeafEvaluatorTransposition.HitRatePct);
        nnTranspositionCacheHitRate ??= new PollingCounter("transposition-cache-hit-rate_pct", this, () => LeafEvaluatorCache.HitRatePct);

        // Compute the fraction of the tranpsosition hits (from store or tree) that were old generation
        static float PctOldGen()
        {
          long numHits = LeafEvaluatorCache.NumHits.Value + LeafEvaluatorTransposition.NumHits.Value;
          long numOldGen = LeafEvaluatorCache.NumHitsOldGeneration.Value + LeafEvaluatorTransposition.NumHitsOldGeneration.Value;
          float pctOldGen = 100.0f * ((float)numOldGen / numHits);
          return pctOldGen;
        }
        nnTranspositionsHitRateOldGeneration ??= new PollingCounter("transposition-old-gen-pct", this, () => PctOldGen());

        //        mlhMoveModificationFraction ??= new PollingCounter("mlh-move-modified-pct", this, () => 100.0f * ManagerChooseBestMove.MLHMoveModifiedFraction);

        tablebaseHitsTotal ??= new PollingCounter("tablebase-hits-total", this, () => LC0DLLSyzygyEvaluator.NumTablebaseHits);
        tablebaseHits ??= new IncrementingPollingCounter("tablebase-hits", this, () => LC0DLLSyzygyEvaluator.NumTablebaseHits);
        tablebasePly1HitsTotal ??= new PollingCounter("tablebase-ply1-hits-total", this, () => LeafEvaluatorSyzygyPly1.NumHits.Value);
        countSiblingEvaluationsUsed ??= new PollingCounter("sibling-evals-used", this, () => MCTSNodeSiblingEval.CountSiblingEvalsUsed.Value);
        
        timeAllotmentOvershootMax ??= new PollingCounter("max-search-time-overshoot", this, () => MaximumTimeAllotmentOvershoot);
       

        testCounter1 ??= new PollingCounter("test", this, () => TestCounter1);

      }
    }
  }
}

