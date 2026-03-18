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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.MCGS.Environment;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Search.Phases.Backup;
using Ceres.MCGS.Search.Phases.Evaluation;
using Ceres.MCGS.Search.Strategies;
using Microsoft.Extensions.Logging;
using static Ceres.MCGS.Search.Phases.MCGSSelect;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Coordinates the evaluation and selection process in a graph-based search algorithm. 
/// </summary>
public partial class MCGSEngine
{
  public readonly MCGSManager Manager;

  public static bool VERBOSE = false;

  public readonly Graph Graph;

  public readonly MCGSSelectBackupStrategyBase Strategy;

  public readonly MCGSSelect Select;
  public readonly MCGSBackup Backup;

  readonly SelectTerminatorPrefetched evaluatorPrecomputed;

  // Possible lock to restrict evaluator to single thread
  // (or null if each overlapping selector has its own evaluator).
  internal readonly LockTimed EvaluatorLock;

  // New PhaseCoordinator to manage ordered execution of select and backup phases.
  internal readonly PhaseCoordinator Coordinator = new();

  internal int numVisitsInFlight = 0;


  /// <summary>
  /// Information aboug nodes above the search root node up to (but not including) the graph root.
  /// </summary>
  public GraphRootToSearchRootNodeInfo[] SearchRootPathFromGraphRoot { get; internal set; }

  public GNode SearchRootNode { get; internal set; }
  public MGPosition SearchRootPosMG;

  /// <summary>
  /// Per-square ply-since-last-move values at the search root position (64 bytes).
  /// Computed once at search start for TPG neural network evaluation.
  /// </summary>
  internal byte[] SearchRootPlySinceLastMove;

  /// <summary>
  /// If true, PlySinceLastMove values are maintained incrementally on MCGSPath during selection.
  /// </summary>
  internal bool NeedsPlySinceLastMove;


  public PosHash96MultisetRunning SearchRootRunningHash { get; internal set; }


  internal bool startedOverlapping = false;

  MCGSIterator iterator0;
  MCGSIterator iterator1;

  public WorkerPool<ExtendPathsWorkerInfo>[] SelectWorkerPools;

  internal int nextBatchID;


  /// <summary>
  /// Optional logging object.
  /// </summary>
  internal readonly ILogger<MCGSEngine> Logger;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="manager">The MCGSManager that owns this engine</param>
  /// <param name="graph">The search graph</param>
  public MCGSEngine(MCGSManager manager,
                    WorkerPool<ExtendPathsWorkerInfo>[] selectWorkerPools,
                    Graph graph,
                    GraphRootToSearchRootNodeInfo[] searchRootPathFromGraphRoot)
  {
    Manager = manager;
    Graph = graph;
    Strategy = new MCGSStrategyPUCT(this);
    SelectWorkerPools = selectWorkerPools;

    const bool DEBUG_DUMP = false;
    if (DEBUG_DUMP)
    {
      Console.WriteLine("** HISTORY **");
      foreach (Position pos in graph.Store.PositionHistory.Positions)
      {
        Console.WriteLine(pos.FEN);
      }
      Console.WriteLine("** GRAPH ROOT --> SEARCH ROOT **");
      foreach (GraphRootToSearchRootNodeInfo pos in searchRootPathFromGraphRoot)
      {
        Console.WriteLine(pos.ChildPosMG.ToPosition.FEN);
      }
      Console.WriteLine("\r\nHISTORY HASHES");
      Graph.Store.HistoryHashes.Dump();
      Console.WriteLine("\r\nGraphRootToSearchRootNodeInfo");
      foreach (GraphRootToSearchRootNodeInfo nodeInfo in searchRootPathFromGraphRoot)
      {
        Console.WriteLine(nodeInfo);
      }
      Console.WriteLine();
    }

    SearchRootPathFromGraphRoot = searchRootPathFromGraphRoot;
    if (searchRootPathFromGraphRoot.Length > 0)
    {
      SearchRootNode = searchRootPathFromGraphRoot[^1].ChildNode;
      SearchRootPosMG = searchRootPathFromGraphRoot[^1].ChildPosMG;
      //Debug.Assert(SearchRootNode.CalcPosition() == SearchRootPosMG);
      // Start with graph root running hash and add on nodes
      PosHash96MultisetRunning runningHash = graph.Store.HistoryHashes.PriorPositionsHashesRunning[^1];
      foreach (GraphRootToSearchRootNodeInfo nodeInfo in searchRootPathFromGraphRoot)
      {
        if (nodeInfo.MoveToChildIrreversible)
        {
          runningHash = default;
        }
        runningHash.Add(nodeInfo.ChildHashStandalone96);
      }

      // In position equivalence mode, we need to remove any draw-by-repetition counts
      // which can be proven to no longer valid after graph truncation.
      if (MCGSParamsFixed.POSITION_MODE_DEPTH_BACKUP_INVALIDATED_REPETITION > 0
        && Manager.ParamsSearch.EnableGraph
        && Manager.ParamsSearch.PathTranspositionMode == PathMode.PositionEquivalence
        && Manager.ParamsSearch.TestFlag
        )
      {
        SearchRootNode.RemoveInvalidatedDrawByRepetitionsFromNodeEdges(MCGSParamsFixed.POSITION_MODE_DEPTH_BACKUP_INVALIDATED_REPETITION, SearchRootPathFromGraphRoot);
      }

      SearchRootRunningHash = runningHash;
    }
    else
    {
      SearchRootNode = graph.GraphRootNode;
      SearchRootRunningHash = graph.Store.HistoryHashes.PriorPositionsHashesRunning[^1];
      SearchRootPosMG = graph.Store.HistoryHashes.PriorPositionsMG[^1];
      //Debug.Assert(SearchRootNode.CalcPosition() == SearchRootPosMG);

    }

    if (MCGSParamsFixed.LOGGING_ENABLED)
    {
      Logger = MCGSEnvironment.CreateLogger<MCGSEngine>();
    }

    // If dual evaluators are not configured, we need a lock to serialize access
    if (Manager.ParamsSearch.Execution.DualOverlappedIterators && !Manager.ParamsSearch.Execution.DualEvaluators)
    {
      // Only one evaluator, so we need a lock to serialize access.
      EvaluatorLock = new LockTimed(false);
    }

    evaluatorPrecomputed = new SelectTerminatorPrefetched();

    ComputeSearchRootPlySinceLastMove();

    Select = new MCGSSelect(this);
    Backup = new MCGSBackup(this);
  }


  /// <summary>
  /// Computes the ply-since-last-move array at the search root by walking
  /// the full game history (prehistory moves + graph-root-to-search-root path).
  /// </summary>
  void ComputeSearchRootPlySinceLastMove()
  {
    const byte DEFAULT_PLIES_SINCE_LAST_MOVE_STARTPOS = 30;

    PositionWithHistory prehistory = Graph.Store.PositionHistory;

    // Validate: if there are history positions but no moves, this indicates the PositionWithHistory
    // was constructed from positions only (e.g., from training data) without move information.
    // The LastMovePlies feature requires actual moves to compute ply-since-last-move correctly.
    int numHistoryPositions = prehistory.GetPositions().Length;
    if (numHistoryPositions > 1 && prehistory.Moves.Count == 0)
    {
      throw new InvalidOperationException(
        $"PositionWithHistory has {numHistoryPositions} history positions but Moves.Count is 0. " +
        $"This indicates the history was constructed from positions without move information " +
        $"(e.g., from training data via ToPositionWithHistory). " +
        $"The LastMovePlies feature requires move history to compute ply-since-last-move values. " +
        $"Either populate the Moves list from the position history, or disable TestFlag/LastMovePlies for this use case.");
    }

    int totalPlies = prehistory.Moves.Count + SearchRootPathFromGraphRoot.Length;

    byte[] curr = new byte[64];
    byte initVal = (byte)Math.Max(0, DEFAULT_PLIES_SINCE_LAST_MOVE_STARTPOS - totalPlies);
    Array.Fill(curr, initVal);

    byte[] temp = new byte[64];

    // Process prehistory moves.
    foreach (MGMove move in prehistory.Moves)
    {
      PlySinceLastMoveArray.ApplyMoveWithSwap(ref curr, ref temp, in move);
    }

    // Process graph-root-to-search-root path moves.
    foreach (GraphRootToSearchRootNodeInfo nodeInfo in SearchRootPathFromGraphRoot)
    {
      MGMove move = nodeInfo.MoveToChild;
      PlySinceLastMoveArray.ApplyMoveWithSwap(ref curr, ref temp, in move);
    }

    // Verify no zero values exist (0 means "never moved" which is invalid after processing moves).
    Debug.Assert(!curr.Contains((byte)0), "ComputeSearchRootPlySinceLastMove produced a zero value, which is invalid.");

    SearchRootPlySinceLastMove = curr;
  }


  static readonly Lock logLock = new();


  [Conditional("DEBUG")]
  internal void DebugLogInfo(string message, params object[] args)
  {
    if (Logger != null)
    {
      lock (logLock)
      {
        Logger.LogInformation(message, args);
      }
    }
  }



  DateTime? lastCallbackTime;
  DateTime firstCallbackTime;

  // High-performance tick counter variables for fast bypass logic
  private long lastTickCheck = 0;
  private static readonly long TicksFor10Ms = Stopwatch.Frequency / 100; // 10ms in ticks

  internal void PossiblyInvokeCallback()
  {
    // Fast check to see if we should update timers and consider calling progress callback.
    long currentTicks = Stopwatch.GetTimestamp();
    if (currentTicks - lastTickCheck >= TicksFor10Ms)
    {
      lastTickCheck = currentTicks;

      // Now do the expensive DateTime logic (only ~every 10ms)
      DateTime now = DateTime.Now;
      if (lastCallbackTime == null)
      {
        firstCallbackTime = now;
        lastCallbackTime = now;
      }

      float INTERVAL_SECONDS_CALLBACK = (now - firstCallbackTime).TotalSeconds > 10 ? 0.25f : 0.5f;
      if ((now - lastCallbackTime.Value).TotalSeconds > INTERVAL_SECONDS_CALLBACK)
      {
        Manager.ProgressCallback?.Invoke(Manager);
        lastCallbackTime = now;
      }
    }
  }


  internal bool ShouldContinue()
  {
    //  Manager.UpdateSearchStopStatus();
    return Manager.StopStatus == MCGSManager.SearchStopStatus.Continue
       && !Manager.ExternalStopRequested;
  }

  internal WorkerPool<ExtendPathsWorkerInfo> GetWorkerPool(int iteratorID)
  {
    if (SelectWorkerPools[iteratorID] == null
     && Manager.ParamsSearch.Execution.SelectOperationParallelThresholdNumVisits < int.MaxValue)
    {
      // TODO: size this based on expected search length
      SelectWorkerPools[iteratorID] = new(MCGSParamsFixed.PARALLEL_SELECT_NUM_INITIAL_WORKERS,
                                          MCGSParamsFixed.PARALLEL_SELECT_NUM_WORKERS_GROWTH_INCREMENT,
                                          null, "MCGSPathSelect");
    }
    return SelectWorkerPools[iteratorID];
  }


  internal void RunLoop(int hardMaxRootN)
  {
    const bool DEBUG_MODE = false;

    Manager.RootNWhenSearchStarted = SearchRootNode.N;

    numVisitsInFlight = 0;
    startedOverlapping = false;

    int firstIteratorID = 0;
    int secondIteratorID = 1;

    iterator0 = new(this, firstIteratorID, Manager.EvaluatorNN0);

    if (!SearchRootNode.IsEvaluated)
    {
      EvaluateRootAndSetNodeValues(iterator0, false);
    }

    //    int numVisitsToTarget = hardLimitTreeSize - RootNode.N;
    int startOverlappingN = Manager.ParamsSearch.Execution.DualOverlappedIterators ? MCGSParamsFixed.MIN_N_START_OVERLAP : int.MaxValue;

    // Execute initial search phase always without overlap
    int numTriesNoProgress = 0;
    while (SearchRootNode.N < hardMaxRootN
        && SearchRootNode.N < startOverlappingN
        && ShouldContinue())
    {
      int startN = SearchRootNode.N;
      int batchSize = GetNextBatchSize(hardMaxRootN);
      iterator0.RunOnce(batchSize, hardMaxRootN);
      int numVisitsAdded = SearchRootNode.N - startN;

      if (numVisitsAdded == 0)
      {
        numTriesNoProgress++;
        if (numTriesNoProgress >= 3)
        {
          // No progress after several tries, so give up.
          break;
        }
      }
      else
      {
        numTriesNoProgress = 0;
      }
    }

    iterator1 = null;
    if (Manager.ParamsSearch.Execution.DualOverlappedIterators && ShouldContinue())
    {
      MCGSEvaluatorNeuralNet evaluator1ToUse = Manager.ParamsSearch.Execution.DualEvaluators ? Manager.EvaluatorNN1 : Manager.EvaluatorNN0;
      iterator1 = new MCGSIterator(this, secondIteratorID, evaluator1ToUse);

      startedOverlapping = true;

      // Pre-create synchronization barrier if requested
      int syncN = Manager.ParamsSearch.Execution.SyncEveryNBatches;
      if (syncN > 0 && IterationSyncBarrier == null)
      {
        // Exactly two iterators participate
        IterationSyncBarrier = new Barrier(2, b =>
        {
          if (false)
          {
            // Sample validation code.
            Console.WriteLine("Synchronized! (every {0} batches) phase={1}", syncN, b.CurrentPhaseNumber);
            Graph.Validate(true, true, true);
            ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "Validated graph " + SearchRootNode.N);
          }
        });
      }

      while (SearchRootNode.N < hardMaxRootN && ShouldContinue())
      {
        int startRootN = SearchRootNode.N;

        // Possibly continue search with dual overlapped iterators.
        Task iterator0LoopTask = Task.Run(() => iterator0.RunLoop(() => GetNextBatchSize(hardMaxRootN), hardMaxRootN));
        Task iterator1LoopTask = Task.Run(() => iterator1.RunLoop(() => GetNextBatchSize(hardMaxRootN), hardMaxRootN));

        Task.WaitAll(iterator0LoopTask, iterator1LoopTask);

        int numVisitsProcessed = SearchRootNode.N - startRootN;
        if (numVisitsProcessed == 0)
        {
          //Console.WriteLine(hardMaxRootN - SearchRootNode.N);
          break;
        }
      }
    }


    if (SearchRootNode.N < hardMaxRootN)
    {
      iterator0.RunLoop(() => GetNextBatchSize(hardMaxRootN), hardMaxRootN);
    }

    if (false)
    {
      Console.WriteLine("Iterator 0");
      iterator0.PathsSet.DumpDistribution();
      Console.WriteLine("\r\nIterator 1");
      iterator1?.PathsSet.DumpDistribution();
      Console.WriteLine("\r\nAll iterators");
      DumpDistributionPathLengths();
    }

    iterator0.Dispose();
    iterator1?.Dispose();

    IterationSyncBarrier?.Dispose();
  }


  public int GetNextBatchSize(int hardMaxRootN)
  {
    int numVisitsNeededRemaining = hardMaxRootN - SearchRootNode.N - numVisitsInFlight;

    ref readonly ParamsSearch paramsSearch = ref Manager.ParamsSearch;
    Debug.Assert(!paramsSearch.EnableEarlySmallBatchSizes);
    int targetBatchSize = OptimalBatchSizeCalculator.CalcOptimalBatchSize(numVisitsNeededRemaining, SearchRootNode.N,
                                                                          paramsSearch.Execution.DualOverlappedIterators,
                                                                          paramsSearch.Execution.MaxBatchSize, paramsSearch.BatchSizeMultiplier,
                                                                          paramsSearch.EnableEarlySmallBatchSizes);

    targetBatchSize = Math.Min(numVisitsNeededRemaining, targetBatchSize);
    targetBatchSize = Math.Min(targetBatchSize, Manager.MaxBatchSizeDueToPossibleNearTimeExhaustion);

    //    Debug.Assert(targetBatchSize > 0);
    return targetBatchSize;
  }


  internal Barrier IterationSyncBarrier;

  // Add helper (place near other internal helpers)
  internal void PossiblySynchronizeIterators(MCGSIterator iterator)
  {
    // Fast exits
    int n = Manager.ParamsSearch.Execution.SyncEveryNBatches;
    if (n <= 0 || iterator1 is null || !ShouldContinue())
    {
      return;
    }

    Barrier barrier = IterationSyncBarrier;
    if (barrier == null)
    {
      return;                      // not initialized (no overlap / disabled)
    }

    int batch = iterator.BatchSequenceNum;
    if (batch == 0 || (batch % n) != 0)
    {
      return; // not a sync boundary
    }

    // Timed wait loop to avoid permanent deadlock if peer stops.
    const int WAIT_MS = 1;
    while (ShouldContinue())
    {
      try
      {
        // Signal and wait for the other participant.
        // The return value indicates if this thread was the one that executed the post-phase action.
        // In either case (true or false), the barrier has been successfully passed by both threads.
        barrier.SignalAndWait(WAIT_MS);
        return; // Both threads should exit after successful synchronization.
      }
      catch (BarrierPostPhaseException)
      {
        // The post-phase action threw an exception. This is unexpected but we should continue.
        return;
      }
      catch (ObjectDisposedException)
      {
        // The barrier was disposed, likely during shutdown.
        return;
      }
      catch (TimeoutException)
      {
        // The wait timed out. The loop will continue as long as ShouldContinue() is true.
      }
    }

    // If we exit the loop because ShouldContinue() is false, try to unblock the other thread.
    if (!ShouldContinue())
    {
      try { barrier.RemoveParticipant(); } catch { }
    }
  }

  private void EvaluateRootAndSetNodeValues(MCGSIterator iterator, bool debugMode)
  {
    Debug.Assert(Manager.Engine.Graph.Store.NodesStore.NumUsedNodes == 1);

    MCGSEvaluatorNeuralNet terminatorNN = Manager.EvaluatorNN0;

    MCGSPath pathForRootNode = iterator.AllocatedPath(1);

    pathForRootNode.AddRoot(Manager.Engine.SearchRootNode.CalcPosition());

    if (NeedsPlySinceLastMove)
    {
      SearchRootPlySinceLastMove.AsSpan().CopyTo(pathForRootNode.PlySinceLastMove.SquarePlySince);
    }

    ref MCGSPathVisit refRootPathVisit = ref pathForRootNode.LeafVisitRef;

    ListBounded<MCGSPath> visitsList = new(1)
    {
      pathForRootNode
    };

    terminatorNN.BatchGenerate(this, visitsList);
    Graph.RegisterNNBatch(1);

    Debug.Assert(Manager.Engine.SearchRootNode.N == 0);
    Strategy.BackupToNode(Manager.Engine.SearchRootNode, 1,
                          pathForRootNode.TerminationInfo.V,
                          pathForRootNode.TerminationInfo.DrawP);

    Manager.NNEvaluator0.BuffersLock?.Release();

    if (debugMode)
    {
      Console.WriteLine("Evaluated root: " + Manager.Engine.SearchRootNode);
    }
  }


  internal int ProcessBatchHarvest(int selectorID, int numTargetVisits, bool debugMode)
  {
    evaluatorPrecomputed.InPrefetchMode = false;
    return DoSelectAndBackupBatch(new MCGSStrategyPUCT(this), selectorID, false, numTargetVisits, debugMode);
  }

  public static long TotalNumNodesPrefetched;

  internal int ProcessBatchPrefetch(int numVisits, int maxWidth, int maxDepth,
                                     Func<GNode, int, bool> moveAcceptPredicate, bool debugMode)
  {
    evaluatorPrecomputed.InPrefetchMode = true;
    MCGSStrategyPrefetch prefetcher = new(this, maxDepth, maxWidth, moveAcceptPredicate, debugMode);
    int priorNodesFromGraphRoot = Graph.GraphRootNode.IsLeaf ? 0 : Graph.Store.NodesStore.NumUsedNodes;
    DoSelectAndBackupBatch(prefetcher, 0, true, numVisits, debugMode);
    int afterNodes = this.Graph.Store.NodesStore.NumUsedNodes;

    int numPrefetched = afterNodes - priorNodesFromGraphRoot;

    Interlocked.Add(ref TotalNumNodesPrefetched, numPrefetched);
    return numPrefetched;
  }

  private int DoSelectAndBackupBatch(MCGSSelectBackupStrategyBase strategy, int selectorID,
                                     bool prefetchMode, int numVisits, bool debugMode)
  {
    throw new NotImplementedException();
  }


  public void ApplyNodeEvaluationValues(GNode node,
                                        in MGPosition position,
                                        MGMoveList moves,
                                        int policyActionIndex,
                                        Memory<CompressedPolicyVector> policies,
                                        Memory<CompressedActionVector> actions,
                                        in SelectTerminationInfo evalResult,
                                        float? overridePolicySoftmax = null)
  {
    Debug.Assert(!node.Terminal.IsTerminal()); // terminals always stored on the edge with no actual node
    Debug.Assert(node.Terminal != GameResult.Draw || (node.WinP == 0 && node.LossP == 0 && evalResult.V == 0));

    ref GNodeStruct nodeRef = ref node.NodeRef;

    nodeRef.Terminal = evalResult.GameResult;

    Debug.Assert(nodeRef.IsWhite == (evalResult.Side == SideType.White));
    Debug.Assert(!ParamsSelect.VIsForcedLoss(evalResult.V)); // network not expected to "prove" anything

    nodeRef.WinP = evalResult.WinP;
    nodeRef.LossP = evalResult.LossP;
    nodeRef.M = (byte)MathF.Round(evalResult.M, 0);
    nodeRef.UncertaintyValue = evalResult.UncertaintyV;
    nodeRef.UncertaintyPolicy = evalResult.UncertaintyP;
    nodeRef.FortressP = evalResult.FortressP.ToFloat;

    Debug.Assert(evalResult.GameResult.IsTerminal() || Math.Abs(nodeRef.V) <= 1);


    if (!nodeRef.Terminal.IsTerminal())
    {
      node.SetPolicy(overridePolicySoftmax ?? Manager.ParamsSelect.PolicySoftmax, ParamsSelect.MinPolicyProbability,
                     in position,
                     moves,
                     in policies.Span[policyActionIndex],
#if ACTION_ENABLED
                     NNEvaluator.HasAction,
                     NNEvaluator.HasAction ? in actions.Span[policyActionIndex] : default,
#endif
                     Manager.NNEvaluator0.PolicyReturnedSameOrderMoveList);

      const bool DEBUG = false;
      if (node.Graph.Store.HasState)
      {
#if ACTION_ENABLED

        if (DEBUG && evalResult.State != null)
        {
          Console.WriteLine(node.Index.Index + " STATE_SET " + evalResult.State[0]);
        }
        node.Graph.Store.AllStateVectors[node.Index.Index] = evalResult.State;
#endif
      }
    }
  }

  /// <summary>
  /// Depth of the deepest path seen so far.
  /// </summary>
  public int MaxPathDepth => iterator1 == null ? iterator0.MaxPathDepth
                                                   : Math.Max(iterator0.MaxPathDepth, iterator1.MaxPathDepth);

  /// <summary>
  /// Average depth of all paths seen so far.
  /// </summary>
  public float AvgPathDepth => iterator1 == null ? iterator0.AvgPathDepth
                                                     : StatUtils.Average(iterator0.AvgPathDepth, iterator1.AvgPathDepth);

  /// <summary>
  /// Fraction of node selection attempts that yielded a usable node.
  /// </summary>
  public float NodeSelectionYieldFrac => iterator1 == null ? iterator0.NodeSelectionYieldFrac
                                                           : StatUtils.Average(iterator0.NodeSelectionYieldFrac, iterator1.NodeSelectionYieldFrac);
}

