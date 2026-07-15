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

  /// <summary>
  /// Helper used to optionally perform an (experimental) full bottom-up recomputation
  /// of all node Q values after each batch is backed up (ParamsSearch.PostBackupQMode == FullRecompute).
  /// Shared across iterators; invoked only from within the (mutually exclusive) backup lock.
  /// </summary>
  public readonly BottomUpQRecalculator QRecalculator;

  /// <summary>
  /// Helper used to optionally perform selective, amortized upward Q propagation after each batch
  /// is backed up (enabled via ParamsSearch.PostBackupQMode == StaleDrain).
  /// Shared across iterators; invoked only from within the (mutually exclusive) backup lock.
  /// </summary>
  public readonly SelectiveQPropagator QPropagator;

  internal int numVisitsInFlight = 0;

  // Evaluator metadata (constant for a search) used to bound batch sizes by network size and device count.
  readonly int cachedNumDevicesInEvaluator;
  readonly long cachedNetFileSizeBytes;


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
  /// Optional hook invoked at the end of each iterator batch (MCGSIterator.RunOnce), after the
  /// batch's backup has fully completed and all coordinator gates have been exited. At that point
  /// the graph is quiescent in single-iterator harnesses (must not be used with
  /// DualOverlappedIterators). Used by external instrumentation such as the Q-probe
  /// training-data harvester (see MCGSIterator.RunProbeSpecs).
  /// </summary>
  internal Action<MCGSIterator> PostBatchHook;


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

    // Cache evaluator metadata (constant for the search) used to bound batch sizes.
    NNEvaluator evaluator0 = Manager.NNEvaluator0;
    cachedNumDevicesInEvaluator = evaluator0?.NumDevices ?? 1;
    cachedNetFileSizeBytes = evaluator0?.Info?.NetworkFileSizeBytes ?? -1;

    PostBatchHook = manager.ParamsSearch.PostBatchHook;

    Graph = graph;
    Graph.PTBMaxRepDrawFraction = manager.ParamsSearch.PseudoTranspositionBlendingMaxRepDrawFraction;
    Strategy = new MCGSStrategyPUCT(this);
    SelectWorkerPools = selectWorkerPools;
    QRecalculator = new BottomUpQRecalculator(this);
    QPropagator = new SelectiveQPropagator(this);

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

      // In PositionEquivalence mode a board-coalesced node is reused across game plies. When the current
      // position repeats a board reached earlier in the game, child edges can carry non-draw visits
      // accumulated when those moves were NOT yet repetitions, diluting the value of moves that NOW
      // complete a repetition so the engine walks into (or fails to claim) a draw with its eval frozen.
      // Reclassify such moves as full draws, from the search root down to the configured depth (1 = the
      // root's direct children, the dominant decision-determining case). See GNode.ReconcileDrawByRepetitions.
      int reconcileDepth = Manager.ParamsSearch.RepetitionDrawReconciliationDepth;
      if (reconcileDepth >= 1
        && Manager.ParamsSearch.EnableGraph
        && Manager.ParamsSearch.PathTranspositionMode == PathMode.PositionEquivalence)
      {
        // Cheap necessary-condition gate (O(rule50)): a reconcilable edge can exist only if some board
        // already recurs within the spine+prehistory reversible run. When it does not, the whole walk is
        // provably a no-op and is skipped (most positions - openings, anything soon after a pawn move or
        // capture, or any unrepeated middlegame - take this path).
        bool repPossible = MCGSPath.SpinePrehistoryHasRepetitionTarget(Graph, SearchRootPathFromGraphRoot);

        if (reconcileDepth <= 1)
        {
          if (repPossible)
          {
            SearchRootNode.ReconcileDrawByRepetitions(SearchRootPathFromGraphRoot, 1);
          }
        }
        else
        {
          // Deeper reconciliation: time it and emit a per-search stats line. Use the normal console color
          // unless the pass actually reclassified visit mass, in which case make it yellow to stand out.
          Stopwatch reconcileTimer = Stopwatch.StartNew();
          GNode.RepetitionReconcileStats stats = repPossible
            ? SearchRootNode.ReconcileDrawByRepetitions(SearchRootPathFromGraphRoot, reconcileDepth)
            : default;
          reconcileTimer.Stop();

          const bool DEBUG_DUMP_RECONCILE_INFO = false;
          if (DEBUG_DUMP_RECONCILE_INFO && repPossible)
          {
            string msg = $"[RepDrawReconcile] depth={reconcileDepth} rootN={SearchRootNode.N:N0} "
              + (repPossible
                  ? $"nodesWalked={stats.NodesWalked:N0} maxDepthReached={stats.MaxDepthReached} "
                    + $"edgesReconciled={stats.EdgesReconciled:N0} visitMassReconciled={stats.VisitMassReconciled:N0} "
                  : "skipped (no repetition target in history) ")
              + $"time={reconcileTimer.Elapsed.TotalMilliseconds:F1}ms"
              + (stats.HitNodeCap ? " (HIT NODE CAP - walk halted early)" : "");

            if (stats.VisitMassReconciled > 0)
            {
              ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, msg);
            }
            else
            {
              Console.WriteLine(msg);
            }
          }
        }
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

    // Configure in-order ("no crossing") backup enforcement from the search parameters.
    Coordinator.EnforceInOrderBackup = !Manager.ParamsSearch.Execution.AllowOutOfOrderBatches;

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
                                                                          paramsSearch.EnableEarlySmallBatchSizes,
                                                                          cachedNumDevicesInEvaluator, cachedNetFileSizeBytes);

    targetBatchSize = Math.Min(numVisitsNeededRemaining, targetBatchSize);
    targetBatchSize = Math.Min(targetBatchSize, Manager.MaxBatchSizeDueToPossibleNearTimeExhaustion);

    //    Debug.Assert(targetBatchSize > 0);
    return targetBatchSize;
  }


  /// <summary>
  /// Runs "deep rollout" visits that begin at the specified inner nodes (rather than the search
  /// root), growing the existing graph. Each round sends exactly one visit to each still-active
  /// node; rollouts are aggregated into shared NN batches and backed up (propagating each value up
  /// to the search root). The exploration term in selection is scaled by explorationMultiplier
  /// (CPUCT for PUCT, CBGPUCT_SelectLambdaC for CBGPUCT). When stopNodeVisitsIfTerminalReached is
  /// true, a node whose rollout reaches a terminal leaf is dropped from subsequent rounds.
  ///
  /// startNodes must already be filtered to evaluated, non-terminal, strict descendants of the
  /// search root (see MCGSManager.DoSearchInnerNodes).
  /// </summary>
  /// <param name="startNodes"></param>
  /// <param name="numVisitsEachNode"></param>
  /// <param name="explorationMultiplier"></param>
  /// <param name="stopNodeVisitsIfTerminalReached"></param>
  /// <param name="preBackupCallback">If not null, invoked with each round's constructed PathsSet just before backup.</param>
  /// <param name="postBackupCallback">If not null, invoked with each round's PathsSet just after backup.</param>
  /// <param name="deepRollout">
  /// If true, configures the rollout for greedy depth extension to a true frontier / terminal:
  /// bypasses the transposition-sufficiency stop and applies a pessimistic FPU (Absolute, 1.0) for
  /// the duration (the shared ParamsSelect FPU is snapshotted and restored afterwards).
  /// </param>
  /// <param name="deadline">Optional wall-clock deadline; remaining rounds are abandoned once passed.</param>
  /// <param name="dryUpRounds">
  /// If positive, a node is dropped from subsequent rounds once this many consecutive rounds
  /// produced no new distinct line below it (its descents have dried up).
  /// </param>
  /// <returns>Per-start-node accumulated rollout statistics, keyed by node index.</returns>
  internal Dictionary<NodeIndex, InnerNodeRolloutStats> RunFromInnerNodes(GNode[] startNodes, int numVisitsEachNode,
                                                                          float explorationMultiplier, bool stopNodeVisitsIfTerminalReached,
                                                                          bool deepRollout,
                                                                          Action<MCGSPathsSet> preBackupCallback = null,
                                                                          Action<MCGSPathsSet> postBackupCallback = null,
                                                                          DateTime? deadline = null,
                                                                          int dryUpRounds = 0)
  {
    Dictionary<NodeIndex, InnerNodeRolloutStats> stats = new();
    if (startNodes.Length == 0 || numVisitsEachNode <= 0)
    {
      return stats;
    }

    int maxBatchSize = Manager.ParamsSearch.Execution.MaxBatchSize;

    // For deep rollouts, force a pessimistic Absolute FPU so the greedy descent commits to the best
    // explored line instead of fanning out across unexpanded siblings. Snapshot and restore so the
    // shared ParamsSelect is unaffected after the call.
    ParamsSelect.FPUType savedFPUMode = Manager.ParamsSelect.FPUMode;
    float savedFPUValue = Manager.ParamsSelect.FPUValue;
    if (deepRollout)
    {
      Manager.ParamsSelect.FPUMode = ParamsSelect.FPUType.Absolute;
      Manager.ParamsSelect.FPUValue = 1.0f;
    }

    MCGSIterator iterator = null;
    try
    {
      // Fresh evaluator wrapper over the (still-alive) underlying NNEvaluator: a prior search's
      // RunLoop disposes its iterator (and thus Manager.EvaluatorNN0's batch / pooled buffers), but
      // the underlying NNEvaluator itself is left intact and reusable.
      MCGSEvaluatorNeuralNet evaluatorNN = new MCGSEvaluatorNeuralNet(
        Manager.EvaluatorsSet.EvaluatorDef, Manager.NNEvaluator0, null,
        Manager.ParamsSearch.HistoryFillIn,
        Math.Min(Manager.NNEvaluator0.MaxBatchSize, maxBatchSize),
        false, Manager.ParamsSearch.ValueTemperature, Manager.ParamsSearch.EnableState,
        null, null, Manager.EvaluatorNN0.EngineIsWhite);

      iterator = new MCGSIterator(this, 0, evaluatorNN);
      iterator.CPUCTMultiplier = explorationMultiplier;
      iterator.DisableTranspositionSufficiencyStop = deepRollout;

      List<GNode> activeNodes = new(startNodes.Length);
      activeNodes.AddRange(startNodes);

      List<GNode> chunk = new(Math.Min(maxBatchSize, startNodes.Length));
      Dictionary<NodeIndex, int> dryRoundCounts = dryUpRounds > 0 ? new Dictionary<NodeIndex, int>() : null;
      Dictionary<NodeIndex, int> noResultRoundCounts = new();
      HashSet<NodeIndex> producedResultThisRound = new();

      for (int round = 0; round < numVisitsEachNode && activeNodes.Count > 0; round++)
      {
        if (Manager.ExternalStopRequested || (deadline.HasValue && DateTime.Now >= deadline.Value))
        {
          break;
        }

        HashSet<NodeIndex> droppedThisRound = stopNodeVisitsIfTerminalReached || dryUpRounds > 0
                                                ? new HashSet<NodeIndex>() : null;
        producedResultThisRound.Clear();

        // Process the active nodes in chunks no larger than the NN max batch size, so each chunk's
        // leaves are aggregated into a single NN batch.
        for (int chunkStart = 0; chunkStart < activeNodes.Count && !Manager.ExternalStopRequested; chunkStart += maxBatchSize)
        {
          chunk.Clear();
          int chunkEnd = Math.Min(chunkStart + maxBatchSize, activeNodes.Count);
          for (int i = chunkStart; i < chunkEnd; i++)
          {
            chunk.Add(activeNodes[i]);
          }

          foreach ((NodeIndex node, int depthBelow, int terminalKind, double leafQFromStart, NodeIndex[] sequence)
                     in iterator.RunOnceFromNodes(chunk, preBackupCallback, postBackupCallback))
          {
            producedResultThisRound.Add(node);

            if (!stats.TryGetValue(node, out InnerNodeRolloutStats nodeStats))
            {
              nodeStats = new InnerNodeRolloutStats();
              stats[node] = nodeStats;
            }

            nodeStats.NumVisits++;
            nodeStats.SumLeafQAllPaths += leafQFromStart;
            if (depthBelow > nodeStats.MaxDepthBelowNode)
            {
              nodeStats.MaxDepthBelowNode = depthBelow;
            }

            // Accumulate the distinct rollout lines (drop fully-overlapping repeats). The maximal
            // (tip) subset is derived from these when the result tuple is built.
            bool isNewDistinctLine = !ContainsSequence(nodeStats.DistinctSequences, sequence);
            if (isNewDistinctLine)
            {
              nodeStats.DistinctSequences.Add(sequence);
              nodeStats.DistinctLeafQ.Add(leafQFromStart);
            }

            if (dryRoundCounts != null)
            {
              // Each node receives exactly one rollout per round, so this per-rollout update
              // is the per-round dry-up accounting for the node.
              if (isNewDistinctLine)
              {
                dryRoundCounts[node] = 0;
              }
              else
              {
                dryRoundCounts.TryGetValue(node, out int dryCount);
                dryRoundCounts[node] = ++dryCount;
                if (dryCount >= dryUpRounds)
                {
                  nodeStats.DroppedDryUp = true;
                  droppedThisRound.Add(node);
                }
              }
            }

            switch (terminalKind)
            {
              case 1:
                nodeStats.NumTerminalWin++;
                break;
              case 2:
                nodeStats.NumTerminalDraw++;
                break;
              case 3:
                nodeStats.NumTerminalLoss++;
                break;
            }

            if (terminalKind != 0 && stopNodeVisitsIfTerminalReached)
            {
              nodeStats.DroppedTerminal = true;
              droppedThisRound.Add(node);
            }
          }
        }

        // Drop nodes whose rollouts repeatedly produce no result (e.g. systematically aborted
        // by the path capacity / deep-rollout depth guards) - they would otherwise spin
        // unproductively for all remaining rounds.
        const int MAX_CONSECUTIVE_NO_RESULT_ROUNDS = 3;
        foreach (GNode activeNode in activeNodes)
        {
          if (producedResultThisRound.Contains(activeNode.Index))
          {
            noResultRoundCounts.Remove(activeNode.Index);
          }
          else
          {
            noResultRoundCounts.TryGetValue(activeNode.Index, out int noResultCount);
            noResultRoundCounts[activeNode.Index] = ++noResultCount;
            if (noResultCount >= MAX_CONSECUTIVE_NO_RESULT_ROUNDS)
            {
              droppedThisRound ??= new HashSet<NodeIndex>();
              droppedThisRound.Add(activeNode.Index);
            }
          }
        }

        if (droppedThisRound != null && droppedThisRound.Count > 0)
        {
          activeNodes.RemoveAll(node => droppedThisRound.Contains(node.Index));
        }
      }
    }
    finally
    {
      iterator?.Dispose();
      Manager.ParamsSelect.FPUMode = savedFPUMode;
      Manager.ParamsSelect.FPUValue = savedFPUValue;
    }

    return stats;
  }


  /// <summary>
  /// Returns whether a node-index sequence equal (same length and same elements) to the candidate
  /// is already present in the list. Used to keep maximalPathsNodes free of fully-overlapping
  /// duplicate lines.
  /// </summary>
  /// <param name="sequences"></param>
  /// <param name="candidate"></param>
  /// <returns></returns>
  private static bool ContainsSequence(List<NodeIndex[]> sequences, NodeIndex[] candidate)
  {
    foreach (NodeIndex[] existing in sequences)
    {
      if (existing.Length != candidate.Length)
      {
        continue;
      }

      bool equal = true;
      for (int i = 0; i < candidate.Length; i++)
      {
        if (existing[i].Index != candidate[i].Index)
        {
          equal = false;
          break;
        }
      }

      if (equal)
      {
        return true;
      }
    }

    return false;
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

    Debug.Assert(evalResult.GameResult.IsTerminal() || Math.Abs(nodeRef.V) <= 1);


    if (!nodeRef.Terminal.IsTerminal())
    {
      // Determine effective policy softmax, possibly adjusted by uncertainty policy.
      float effectivePolicySoftmax = overridePolicySoftmax ?? Manager.ParamsSelect.PolicySoftmax;

      // If EnablePolicyUncertaintyTemperatureBoosting is enabled and UncertaintyP head is populated,
      // apply supplemental temperature.
      if (Manager.ParamsSearch.EnablePolicyUncertaintyTemperatureBoosting 
       && !FP16.IsNaN(evalResult.UncertaintyP))
      {
        float up = evalResult.UncertaintyP.ToFloat;
        const bool AGGRESSIVE = false;
        float tempMultiplier = up switch
        {
          <= 0.031f => AGGRESSIVE ? 0.92f : 0.94f,
          <= 0.076f => AGGRESSIVE ? 0.94f : 0.97f,
          <= 0.168f => 1.00f,// no adjustment
          <= 0.321f => AGGRESSIVE ? 1.06f :1.05f,
          _         => AGGRESSIVE ? 1.13f : 1.09f
        };

        effectivePolicySoftmax *= tempMultiplier;
      }

      bool hasAction = actions.Span.Length > 0; //Manager.NNEvaluator0.HasAction ?
      ref readonly CompressedActionVector actionVectorRef = ref (hasAction
         ? ref actions.Span[policyActionIndex]
         : ref EMPTY_ACTION_VECTOR);
      node.SetPolicy(effectivePolicySoftmax, ParamsSelect.MinPolicyProbability,
                     in position,
                     moves,
                     in policies.Span[policyActionIndex],
#if ACTION_ENABLED
                     Manager.NNEvaluator0.HasAction,
                     in actionVectorRef, // TODO: pass structs using in for performance
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

  static CompressedActionVector EMPTY_ACTION_VECTOR;

  /// <summary>
  /// Depth of the deepest path seen so far.
  /// </summary>
  public int MaxPathDepth => iterator1 == null ? (iterator0 == null ? 0 : iterator0.MaxPathDepth)
                                                   : Math.Max(iterator0.MaxPathDepth, iterator1.MaxPathDepth);

  /// <summary>
  /// Average depth of all paths seen so far.
  /// </summary>
  public float AvgPathDepth => iterator1 == null ? (iterator0 == null ? 0 : iterator0.AvgPathDepth)
                                                     : StatUtils.Average(iterator0.AvgPathDepth, iterator1.AvgPathDepth);

  /// <summary>
  /// Fraction of node selection attempts that yielded a usable node.
  /// </summary>
  public float NodeSelectionYieldFrac => iterator1 == null ? (iterator0 == null ? 0 : iterator0.NodeSelectionYieldFrac)
                                                           : StatUtils.Average(iterator0.NodeSelectionYieldFrac, iterator1.NodeSelectionYieldFrac);
}

