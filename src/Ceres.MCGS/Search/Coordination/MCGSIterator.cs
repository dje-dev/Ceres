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
using System.Diagnostics;
using System.IO;
using System.Threading;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Base.Misc;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.Enumerators;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Search.Phases.Evaluation;
using Ceres.MCGS.Search.Strategies;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// An iterator  which repeatedly executes the select/evaluate/backup phases.
/// 
/// Typically two iterators are used to allow partly overlapped phases, 
/// subject to the rule that at most one of the iterators can be in either select or backup
/// phase at any one time.
/// </summary>
public class MCGSIterator : IDisposable
{
  /// <summary>
  /// Tags for metrics about why paths terminated (cached for efficiency).
  /// </summary>
  private static readonly TagList[] pathTerminationTags = MetricTagHelper.PrecomputeEnumTagLists<MCGSPathTerminationReason>("PathTerminationReason");

  /// <summary>
  /// Parent manager of this iterator.
  /// </summary>
  public MCGSManager Manager;

  /// <summary>
  /// Associated engine instance.
  /// </summary>
  public readonly MCGSEngine Engine;

  /// <summary>
  /// Identifier for this iterator (0 or 1).
  /// </summary>
  public readonly int IteratorID;

  /// <summary>
  /// Evaluator instance used by this iterator.
  /// </summary>
  public readonly MCGSEvaluatorNeuralNet EvaluatorNN;

  /// <summary>
  /// Backup strategy engine used for selecting and managing backup strategies.
  readonly MCGSSelectBackupStrategyBase backupStrategy;

  /// <summary>
  /// Set of MCGSPath objects created during the selection phase
  /// </summary>
  public readonly MCGSPathsSet PathsSet;

  /// <summary>
  /// Mode to be used for the backup phase of the current batch.
  /// </summary>
  public BackupMethodEnum BackupMode;

  /// <summary>
  /// Pool of slots to be used for MCGSPathVisit structs.
  /// </summary>
  internal readonly ArraySegmentPool<MCGSPathVisit> pathVisitPool;

  /// <summary>
  /// The sequence number of the current batch being processed.
  /// </summary>
  int batchSequenceNum = 0;

  /// <summary>
  /// Flag to indicate if Dispose has been called.
  /// </summary>
  private bool disposed;

  /// <summary>
  /// Use a cache of MCGSPath to reduce memory allocations
  /// and reused across batches (being cleared in between).
  /// </summary>
  readonly MCGSPath[] paths;

  /// <summary>
  /// Multiplier to apply to CPUCT value during selection phase.
  /// </summary>
  internal float CPUCTMultiplier = 1.0f;

  /// <summary>
  /// When true, the transposition-sufficiency stop (MCGSSelect.IsTranspositionSufficientN) is
  /// bypassed during selection, so descents continue through well-visited transposition nodes to a
  /// true frontier leaf or terminal. Used by inner-node "deep rollouts" (DoSearchInnerNodes).
  /// </summary>
  internal bool DisableTranspositionSufficiencyStop = false;

  internal int numAllocatedPaths = 0;

  /// <summary>
  /// Per-iterator pool of MGMoveList snapshots, reused across batches to avoid the per-new-node
  /// allocation of an MGMoveList (and its MGMove[]) when retaining a generated move list for a
  /// newly created node (see AllocatedMoveListSnapshot / MCGSSelect.DoNewlyCreatedNode). Sized
  /// to the path pool capacity (a snapshot is taken at most once per allocated path) and reset
  /// each batch alongside numAllocatedPaths. The snapshots are consumed during the same batch's
  /// NN evaluation and are no longer referenced after that batch's backup, so reusing them at the
  /// next batch's reset is safe.
  /// </summary>
  readonly MGMoveList[] moveListSnapshotPool;

  internal int numMoveListSnapshotsAllocated = 0;

  const bool ENABLE_LOGGING = false;
  private static readonly TextWriter iteratorLogWriter
    = ENABLE_LOGGING ? TextWriter.Synchronized(new StreamWriter(@"c:\temp\iterator_log.txt", append: true) { AutoFlush = false })
                     : null;

  private void LogWrite(Func<string> messageFunc)
    => iteratorLogWriter?.WriteLine($"{IteratorID} [{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] {messageFunc()}");
  private void LogFlush() => iteratorLogWriter?.Flush();


  /// <summary>
  /// Returns a initialized for use by the iterator
  /// </summary>
  /// <param name="initialNumSlots"></param>
  /// <returns></returns>
  internal MCGSPath AllocatedPath(int? initialNumSlots = null)
  {
    int pathIndex = Interlocked.Increment(ref numAllocatedPaths) - 1;
    
    MCGSPath thisPath;   
    if (paths[pathIndex] == null)
    {
      thisPath = paths[pathIndex] = new MCGSPath(this);
    }
    else
    {
      thisPath = paths[pathIndex];
      thisPath.Reinitialize();
    }

    thisPath.PathID = pathIndex;
    thisPath.slots = pathVisitPool.AllocateSegment(initialNumSlots);

    return thisPath;
    
  }

  /// <summary>
  /// Returns a pooled MGMoveList holding an exactly-usable copy of the given source moves,
  /// reused across batches to avoid allocating a fresh MGMoveList (and MGMove[]) per newly
  /// created node. Lock-free (Interlocked slot reservation), mirroring AllocatedPath; safe for
  /// the parallel select workers. The pool is reset each batch in ResetPaths.
  /// </summary>
  internal MGMoveList AllocatedMoveListSnapshot(MGMoveList source)
  {
    int index = Interlocked.Increment(ref numMoveListSnapshotsAllocated) - 1;

    MGMoveList list = moveListSnapshotPool[index];
    if (list == null)
    {
      list = moveListSnapshotPool[index] = new MGMoveList(source.NumMovesUsed);
    }

    // Copy reuses the backing array when large enough (else grows it once and keeps it pooled).
    list.Copy(source);
    return list;
  }

  internal void ResetPaths()
  {
    numAllocatedPaths = 0;
    numMoveListSnapshotsAllocated = 0;
  }


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="engine"></param>
  /// <param name="iteratorID"></param>
  /// <param name="evaluatorNN"></param>
  public MCGSIterator(MCGSEngine engine, int iteratorID, MCGSEvaluatorNeuralNet evaluatorNN)
  {
    Engine = engine;
    Manager = engine.Manager;

    IteratorID = iteratorID;
    EvaluatorNN = evaluatorNN;
    this.backupStrategy = new MCGSStrategyPUCT(engine);

    int maxBatchSize = Engine.Manager.ParamsSearch.Execution.MaxBatchSize;
    PathsSet = new(this, maxBatchSize);
    paths = new MCGSPath[maxBatchSize + 5];
    moveListSnapshotPool = new MGMoveList[maxBatchSize + 5];

    pathVisitPool = new ArraySegmentPool<MCGSPathVisit>();
  }


  /// <summary>
  /// Releases all resources used by this iterator.
  /// </summary>
  public void Dispose()
  {
    if (disposed)
    {
      return;
    }

    PathsSet.Dispose();
    EvaluatorNN?.Dispose();

    disposed = true;

    GC.SuppressFinalize(this);
  }


  /// <summary>
  /// Finalizer.
  /// </summary>
  ~MCGSIterator()
  {
    Dispose();
  }


  /// <summary>
  /// Logs a debug-level informational message if the application is running in debug mode.
  /// </summary>
  /// <param name="message"></param>
  /// <param name="args"></param>
  [Conditional("DEBUG")]
  internal void DebugLogInfo(string message, params object[] args)
  {
    if (IteratorID != MCGSParamsFixed.LOGGING_EXCLUDE_ITERATOR_NUM)
    {
      Engine.DebugLogInfo(message, args);
    }
  }


  /// <summary>
  /// Returns if the iterator is approaching the maximum path capacity.
  /// </summary>
  internal bool IsApproachingMaxPathCapacity => pathVisitPool.FractionInUse > 0.95;


  /// <summary>
  /// Depth of the deepest path seen so far.
  /// </summary>
  public int MaxPathDepth => PathsSet.MaxNonAbortedPathDepth;

  /// <summary>
  /// Average depth of all paths seen so far.
  /// </summary>
  public float AvgPathDepth => (float)PathsSet.SumNonAbortedPathVisits / PathsSet.CountNonAbortedPathVisits;

  /// <summary>
  /// Fraction of node selection attempts that yielded a usable node.
  /// </summary>
  public float NodeSelectionYieldFrac => PathsSet.CountTotalPathsAttempted == 0
                                       ? 0
                                       : (float)PathsSet.CountNonAbortedPathVisits / (float)PathsSet.CountTotalPathsAttempted;


  /// <summary>
  /// Runs the iteration loop.
  /// </summary>
  /// <param name="getBatchSizeFunc"></param>
  internal void RunLoop(Func<int> getBatchSizeFunc, int hardMaxRootN)
  {
    try
    {
      DoRunLoop(getBatchSizeFunc ,hardMaxRootN);
    }
    catch (Exception e)
    {
      // Make sure Exceptions are not silently swallowed.
      Console.WriteLine(e);
      System.Environment.Exit(3);    
    }
  }

  private void DoRunLoop(Func<int> getBatchSizeFunc, int hardMaxRootN)
  {
    int numRetries = 0;
    while (Engine.ShouldContinue())
    {
      int batchSize = getBatchSizeFunc();
      if (batchSize <= 0)
      {
        // TODO: verify this is ok, why does it happen?
        return;
      }

      LogWrite(()=> $"Starting batch {batchSequenceNum} with size {batchSize}, rootN={Engine.SearchRootNode.N}, inFlight={Engine.numVisitsInFlight}");

      Interlocked.Add(ref Engine.numVisitsInFlight, batchSize);

      int startN = Engine.SearchRootNode.N;

      RunOnce(batchSize, hardMaxRootN);
      int numVisitsAdded = Engine.SearchRootNode.N - startN;

      if (numVisitsAdded == 0)
      {
        numRetries += 1;
        if (numRetries > 3)
        {
          if (!haveWarnedTooManyRetries)
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Iterator {IteratorID} exiting after {numRetries} retries with no visits added");
          }
          haveWarnedTooManyRetries = true;
          return;
        }
      }
      else
      {
        numRetries = 0;
      }      
      
      Interlocked.Add(ref Engine.numVisitsInFlight, -batchSize);
    }
  }


  static bool haveWarnedTooManyRetries = false;
  static bool haveWarned = false;

  // Per-phase wall-clock instrumentation now lives on the per-search PhaseCoordinator instance
  // (see PhaseCoordinator.Record{Select,Eval,Backup}Phase / ResetPhaseTiming / PhaseTimingSummary).
  // It was previously held in process-global static fields here, which were cross-incremented and
  // mutually reset by other searches running concurrently in the same process.


  internal void RunOnce(int batchSize, int hardMaxRootN)
  {
    // Apply search moves as soon as possible (need the root to have been evaluated).
    if (Engine.SearchRootNode.N > 0)
    {
      Engine.Manager.TerminationManager.ApplySearchMovesIfNeeded();
    }

    int thisBatchID = Interlocked.Add(ref Engine.nextBatchID, 1) - 1;

    PathsSet.Reset();
    ResetPaths();

    // N.B. No need to clear underlying memory here
    //      because we take care that all fields are initialized 
    //      before a MCGSPathVisit is actually used.
    pathVisitPool.Clear(false);

   // Determine the backup mode to actually be used for this batch
   // (based on ParamsSearch setting and also graph state).
   // The select and backup phases will both adjust their behavior based on this.
   BackupMode = Engine.Backup.BackupModeToUse();

    // STEP1 : Descend graph selecting children to build paths.
    long tsPhase = Stopwatch.GetTimestamp();
    Engine.Coordinator.EnterSelect(IteratorID, thisBatchID);
    LogWrite(() => $"Start select batch {batchSequenceNum}");

    RunSelectionPhase(batchSize);

    UpdateNNYieldEstimate(batchSize);

    PossiblyRunSecondSelectionForNNBatchSizePadding(batchSize, hardMaxRootN);

    LogWrite(() => $"End select batch {batchSequenceNum} with {PathsSet.Paths.Count} paths");
    Engine.Coordinator.ExitSelect(IteratorID, thisBatchID);
    long tsAfterSelect = Stopwatch.GetTimestamp();
    long selectTicks = tsAfterSelect - tsPhase;
    Engine.Coordinator.RecordSelectPhase(selectTicks);


    if (PathsSet.Paths.Count == 0)
    {
      // No paths to evaluate/backup, but any visits dropped during select still need
      // their deferred ledger backout applied (otherwise their edge in-flight counts
      // and ancestor visit counters would leak). This batch performs no backup, but it must
      // still pass through the in-order-backup gate so the turn counter stays contiguous (and
      // the ledger backout is ordered consistently with the surrounding backups).
      Engine.Coordinator.EnterBackupOrder(thisBatchID);
      Engine.Coordinator.RecordBackupOrder(thisBatchID, didBackup: false);
      ApplyPendingDroppedVisits();
      Engine.Coordinator.ExitBackupOrder(thisBatchID);
      return;
    }

    // Don't dump to console while overlapping to avoid jumbled Console output
    // (over validate due to concurrent updates).
    if (MCGSParamsFixed.DEBUG_MODE && !Engine.startedOverlapping)
    {
      Console.WriteLine("\r\nPATHS FOR BATCH on iterator " + IteratorID);
      Engine.Graph.DumpNodesStructure();
      for (int i = 0; i < PathsSet.Paths.Count; i++)
      {
        PathsSet.Paths[i].DumpAllVisits();
      }

      Engine.Graph.Validate(false);
    }

    // STEP2: Evaluate any nodes needing neural network.
    Engine.Coordinator.EnterEvaluate(IteratorID, thisBatchID);
    LogWrite(() => $"Start evaluate batch {batchSequenceNum} with {PathsSet.NNPaths.Count} NN paths");

    // Retrieve deferred NN results outside of the locked region (if two distinct evaluators exist).
    bool deferRetrieveResults = Manager.ParamsSearch.Execution.DualEvaluators 
                             && Manager.ParamsSearch.Execution.DualOverlappedIterators;
    RunNNEvaluationPhase(deferRetrieveResults);
    LogWrite(() => $"End evaluate batch {batchSequenceNum} with {PathsSet.NNPaths.Count} NN paths");
    Engine.Coordinator.ExitEvaluate(IteratorID, thisBatchID);

    if (deferRetrieveResults && PathsSet.NNPaths.Count > 0)
    {
      EvaluatorNN.RetrieveDeferredResults();
    }
    long tsAfterEval = Stopwatch.GetTimestamp();
    Engine.Coordinator.RecordEvalPhase(tsAfterEval - tsAfterSelect);

    // STEP3: Backup the selected visits.
    Engine.Coordinator.EnterBackup(IteratorID, thisBatchID);
    LogWrite(() => $"Start backup with mode {BackupMode}");

    const bool VERIFY_MULTISET_CORRECTNESS = false;
    if (VERIFY_MULTISET_CORRECTNESS)
    {
      foreach (MCGSPath path in PathsSet.Paths)
      {
        DoVerifySiblingsAreNotMultisetEquivalent(Manager, path, false);
      }
    }

    RunBackupPhase();

#if DEBUG
    // Validate that all of the path visits were backed up successfully
    // (resulting in their NumVisitsAttemptedPendingBackup ending in 0).
    bool foundError = false;
    foreach (MCGSPath path in PathsSet.Paths)
    {
      if (path.NumVisitsInPath > 0)
      {
        foreach (MCGSPathVisitMember visit in path.PathVisitsLeafToRoot)
        {
          if (visit.PathVisitRef.NumVisitsAttemptedPendingBackup != 0) // negative would indicate over-decrement
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Found pending backup visits = {visit.PathVisitRef.NumVisitsAttemptedPendingBackup} after backup phase on iterator {IteratorID} for path {path.PathID} visit {visit.PathVisitRef} (root N: {Engine.SearchRootNode.N})");
            foundError = true;
          }
        }
      }
    }    
#endif

    if (!haveWarned && MCGSParamsFixed.ENABLE_EXTENDED_RELEASE_ASSERTIONS)
    {
      // Do a quick test to verify no edges start out in flight on this iterator.
      foreach (GEdge rootEdge in Engine.SearchRootNode.ChildEdgesExpanded)
      {
        if (rootEdge.NInFlightForIterator(IteratorID) != 0)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Expected no in-flight visits at root at start of batch " + rootEdge + " " + Engine.SearchRootNode);
          haveWarned = true;
        }
      }
    }

    const bool CHECK_AFTER_BACKUP = false;
    if (CHECK_AFTER_BACKUP)
    {
      //      Console.WriteLine("iterator: " + IteratorID);
      Console.WriteLine();
      Engine.Graph.DumpNodesStructure();
      Engine.Graph.Validate(true);
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "Validated graph " + Engine.SearchRootNode.N);
    }

    // Note that this validation appears inside the lock to avoid concurrent updates.
    if (MCGSParamsFixed.VALIDATE_GRAPH_EACH_BATCH)
    {
      if (Manager.ParamsSearch.Execution.DualOverlappedIterators && Engine.SearchRootNode.N >= MCGSParamsFixed.MIN_N_START_OVERLAP)
      {
        throw new Exception("Probably not possible to validate graph while another overlapping executor is possibly active");
      }
      Engine.Graph.Validate(false);
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Validated graph " + Engine.SearchRootNode.N);
    }

    // Optionally perform a full bottom-up recomputation of all node Q values
    // (experimental, enabled via ParamsSearch.TestFlag). This is done here while we still
    // hold the backup lock and the graph is quiescent (no other iterator is concurrently
    // in its select or backup phase).
    PostBackupRecomputeQIfEnabled();

    // Possibly invoke the callback
    // At this point we hold the select/backup lock
    // and graph is quiescent.
    Engine.PossiblyInvokeCallback();

    Engine.Manager.TerminationManager.ApplySearchMovesIfNeeded();

    Manager.RunPeriodicMaintenance(batchSequenceNum);
    batchSequenceNum++;

    LogWrite(() => $"End backup with mode {BackupMode}");
    Engine.Coordinator.ExitBackup(IteratorID, thisBatchID);
    long backupTicks = Stopwatch.GetTimestamp() - tsAfterEval;
    Engine.Coordinator.RecordBackupPhase(backupTicks);

    LogFlush();

    Engine.PossiblySynchronizeIterators(this);
  }


  /// <summary>
  /// Variant of RunOnce used by MCGSManager.DoSearchInnerNodes. Instead of selecting a batch of
  /// visits beginning at the search root, it sends exactly one visit to each specified inner node:
  /// each rollout is a full path searchRoot -> node -> ... -> leaf (built by
  /// MCGSSelect.ExtendPathFromInnerNode), all leaves are evaluated in a single aggregated NN batch,
  /// and all rollouts are backed up (propagating each value up to the search root).
  ///
  /// Returns one record per non-aborted rollout this round: the originating start node, the depth
  /// descended below it, a terminal classification (0 = not terminal, 1 = win, 2 = draw, 3 = loss),
  /// the leaf Q, and the sequence of node indices from the start node (index 0) down to the leaf.
  /// The terminal classification and leaf Q are both from the start node's side-to-move perspective.
  /// The exploration multiplier applied during the descent below each start node is the current
  /// CPUCTMultiplier (set by the caller).
  ///
  /// startNodes.Count must not exceed the configured max batch size (the caller chunks if needed).
  /// </summary>
  /// <param name="startNodes"></param>
  /// <param name="preBackupCallback">If not null, invoked with the constructed PathsSet just before it is backed up.</param>
  /// <param name="postBackupCallback">If not null, invoked with the PathsSet just after it is backed up.</param>
  /// <returns></returns>
  internal List<(NodeIndex node, int depthBelow, int terminalKind, double leafQFromStart, NodeIndex[] sequence)> RunOnceFromNodes(
                                                                                      IReadOnlyList<GNode> startNodes,
                                                                                      Action<MCGSPathsSet> preBackupCallback,
                                                                                      Action<MCGSPathsSet> postBackupCallback)
  {
    Debug.Assert(startNodes.Count <= Manager.ParamsSearch.Execution.MaxBatchSize);

    int thisBatchID = Interlocked.Add(ref Engine.nextBatchID, 1) - 1;

    PathsSet.Reset();
    ResetPaths();
    pathVisitPool.Clear(false);

    BackupMode = Engine.Backup.BackupModeToUse();

    // STEP 1: Build one single-visit rollout path per start node.
    Engine.Coordinator.EnterSelect(IteratorID, thisBatchID);
    for (int i = 0; i < startNodes.Count; i++)
    {
      Engine.Select.ExtendPathFromInnerNode(this, startNodes[i], 1);
    }
    Engine.SelectWorkerPools[IteratorID]?.WaitAll();
    Engine.Coordinator.ExitSelect(IteratorID, thisBatchID);

    if (PathsSet.Paths.Count == 0)
    {
      // Apply any deferred ledger backouts for visits dropped during select. Still pass through
      // the in-order-backup gate so the turn counter stays contiguous (no stall).
      Engine.Coordinator.EnterBackupOrder(thisBatchID);
      Engine.Coordinator.RecordBackupOrder(thisBatchID, didBackup: false);
      ApplyPendingDroppedVisits();
      Engine.Coordinator.ExitBackupOrder(thisBatchID);
      return new List<(NodeIndex node, int depthBelow, int terminalKind, double leafQFromStart, NodeIndex[] sequence)>(0);
    }

    // STEP 2: Evaluate any leaves needing the neural network (single aggregated batch).
    Engine.Coordinator.EnterEvaluate(IteratorID, thisBatchID);
    RunNNEvaluationPhase(deferRetrieveResults: false);
    Engine.Coordinator.ExitEvaluate(IteratorID, thisBatchID);

    // STEP 3: Backup the rollouts (propagates each leaf value up to the search root).
    Engine.Coordinator.EnterBackup(IteratorID, thisBatchID);

    // The PathsSet is fully constructed (and its leaves evaluated) but not yet backed up.
    preBackupCallback?.Invoke(PathsSet);

    // Snapshot the completed paths for building the per-rollout result records below (and for the
    // post-backup callback). Backup does not modify PathsSet.Paths (the parallel reduction copies
    // into its own buffer), so this is a stable handle rather than a drain guard.
    MCGSPath[] completedPaths = PathsSet.Paths.ToArray();

    RunBackupPhase();

    postBackupCallback?.Invoke(PathsSet);

    Engine.PossiblyInvokeCallback();
    Engine.Coordinator.ExitBackup(IteratorID, thisBatchID);

    batchSequenceNum++;

    // Build one result record per non-aborted rollout: depth below the start node, terminal
    // classification, leaf Q (from the start node's perspective), and the node-index sequence
    // from the start node (index 0) down to the leaf.
    List<(NodeIndex node, int depthBelow, int terminalKind, double leafQFromStart, NodeIndex[] sequence)> results = new();
    foreach (MCGSPath path in completedPaths)
    {
      if (path.InnerSearchStartNode.IsNull
       || path.TerminationReason == MCGSPathTerminationReason.Abort)
      {
        continue;
      }

      int depthBelow = path.NumVisitsInPath - path.InnerSearchStartDepth;
      (double leafQFromStart, int terminalKind) = LeafResultFromStart(path, depthBelow);
      NodeIndex[] sequence = BuildNodeSequenceBelowStart(path);
      results.Add((path.InnerSearchStartNode.Index, depthBelow, terminalKind, leafQFromStart, sequence));
    }

    return results;
  }


  /// <summary>
  /// Returns the leaf value and terminal classification of an inner-node rollout, both expressed
  /// from the perspective of the side to move at the start node (the leaf's own value is converted
  /// using the parity of the depth descended below the start node). terminalKind: 0 = not terminal,
  /// 1 = win, 2 = draw, 3 = loss. Non-terminal leaves return the leaf node's Q with terminalKind 0.
  /// </summary>
  /// <param name="path"></param>
  /// <param name="depthBelow"></param>
  /// <returns></returns>
  private static (double leafQFromStart, int terminalKind) LeafResultFromStart(MCGSPath path, int depthBelow)
  {
    GEdge leafEdge = path.LeafVisitRef.ParentChildEdge;
    bool isTerminal = path.TerminationReason == MCGSPathTerminationReason.Terminal
                   || path.TerminationReason == MCGSPathTerminationReason.TerminalEdge;

    // Leaf value in the leaf's own side-to-move perspective (0 for draws / repetition stops).
    double leafVLeaf;
    bool isDraw = false;
    if (path.TerminationReason == MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode
     || leafEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn)
    {
      leafVLeaf = 0.0;
      isDraw = leafEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn;
    }
    else if (leafEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
    {
      double q = leafEdge.ChildNode.Q;
      leafVLeaf = double.IsNaN(q) ? 0.0 : q;
      isDraw = isTerminal && leafEdge.ChildNode.Terminal == GameResult.Draw;
    }
    else
    {
      // Decisive terminal edge (checkmate / tablebase win or loss).
      leafVLeaf = leafEdge.Q;
    }

    double leafQFromStart = (depthBelow % 2 == 0) ? leafVLeaf : -leafVLeaf;

    int terminalKind = 0;
    if (isTerminal)
    {
      const double EPS = 1e-6;
      terminalKind = isDraw ? 2
                            : (leafQFromStart > EPS ? 1 : (leafQFromStart < -EPS ? 3 : 2));
    }

    return (leafQFromStart, terminalKind);
  }


  /// <summary>
  /// Builds the sequence of node indices for an inner-node rollout, from the start node (index 0)
  /// down to the leaf. The root-to-start-node prefix is excluded. A terminal-edge leaf contributes
  /// no node (a terminal edge has no child node), so such a sequence ends at the deepest real node.
  /// </summary>
  /// <param name="path"></param>
  /// <returns></returns>
  private static NodeIndex[] BuildNodeSequenceBelowStart(MCGSPath path)
  {
    int startDepth = path.InnerSearchStartDepth;
    List<NodeIndex> seq = new(Math.Max(1, path.numSlotsUsed - startDepth + 1));
    seq.Add(path.InnerSearchStartNode.Index);   // index 0 is the start node itself
    for (int i = startDepth; i < path.numSlotsUsed; i++)
    {
      GEdge edge = path.slots[i].ParentChildEdge;
      if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        seq.Add(edge.ChildNode.Index);
      }
    }
    return seq.ToArray();
  }


  /// <summary>
  /// Invoked after each batch has been backed up, while this iterator still holds the
  /// backup lock (so the graph is quiescent). When ParamsSearch.TestFlag is enabled,
  /// performs a full bottom-up recomputation of every node's Q value and aggregates
  /// statistics about the magnitude of the changes (see BottomUpQRecalculator).
  /// </summary>
  private void PostBackupRecomputeQIfEnabled()
  {
    switch (Manager.ParamsSearch.PostBackupQMode)
    {
      case ParamsSearch.PostBackupQModeType.FullRecompute:
        Engine.QRecalculator.RunPass();
        break;

      case ParamsSearch.PostBackupQModeType.StaleDrain:
        Engine.QPropagator.RunPass(PathsSet);
        break;
    }
  }


  /// <summary>
  /// Possibly runs another selection pass thru the graph to select more leafs
  /// if first-pass count of paths requiring NN evaluation is far from optimal alignment boundary
  /// (and certain other conditions are satisfied).
  /// </summary>
  /// <param name="batchSize"></param>
  /// <param name="hardMaxRootN"></param>
  private void PossiblyRunSecondSelectionForNNBatchSizePadding(int batchSize, int hardMaxRootN)
  {
    if (Manager.ParamsSearch.Execution.NNBatchSizeFillToEvaluatorCapacity
     && PossiblyRunSecondSelectionToFillEvaluatorCapacity(batchSize, hardMaxRootN))
    {
      return;
    }

    int nnBatchSizeAlignmentTarget = Manager.ParamsSearch.Execution.NNBatchSizeAlignmentTarget;

    if (nnBatchSizeAlignmentTarget > 0)
    {
      int rootN = Engine.SearchRootNode.N;
      int positionsBeforeHardBatchLimit = hardMaxRootN - (rootN + batchSize + Engine.numVisitsInFlight);

      // Scale down for small N
      if (rootN < 50)
      {
        nnBatchSizeAlignmentTarget /= 4;
      }
      else if (rootN < 100)
      {
        nnBatchSizeAlignmentTarget /= 2;
      }

      if (rootN < nnBatchSizeAlignmentTarget * 8 // graph not large relative to possible alignment
       || nnBatchSizeAlignmentTarget == 0        // feature not enabled
       || IsApproachingMaxPathCapacity)          // avoid overflowing batch
      {
        return;
      }

      if (nnBatchSizeAlignmentTarget > 0)
      {
        int numNNPaths = PathsSet.NNPaths.Count;
        int numNNPathsBeyondPriorAlignmentPoint = numNNPaths % nnBatchSizeAlignmentTarget;

        if (numNNPaths < 196 // already large batches won't benefit much from padding (and expensive to RunSelectionPhase)
         && numNNPathsBeyondPriorAlignmentPoint != 0 // not already aligned
         && numNNPathsBeyondPriorAlignmentPoint <= (nnBatchSizeAlignmentTarget / 2)// not already half the way to next alignment point
         && rootN > nnBatchSizeAlignmentTarget * 5 // graph size large relative to possible increment in batch size 
         && numAllocatedPaths < Engine.Manager.ParamsSearch.Execution.MaxBatchSize - nnBatchSizeAlignmentTarget * 2) // not close to max batch size
        {
          int numFiller = (int)MathUtils.RoundedUp(numNNPaths, nnBatchSizeAlignmentTarget) - numNNPaths;
          numFiller = numFiller + numFiller / 3; // take a chance of over-requesting because typically some significant fraction of selected will be non-NN paths
          int newBatchSizeTarget = numAllocatedPaths + numFiller;
          RunSelectionPhase(newBatchSizeTarget);
        }
      }
    }
  }


  /// <summary>
  /// Exponential moving average of the fraction of requested visits which result in a
  /// path requiring NN evaluation (the remainder end in transpositions, terminals,
  /// collisions etc.). Used to oversize the fill-to-capacity selection pass.
  /// </summary>
  float emaNNYieldPerVisit = 1.0f;

  /// <summary>
  /// Updates the running estimate of NN evaluations yielded per requested visit
  /// (based on the outcome of the just-completed primary selection pass).
  /// </summary>
  /// <param name="numVisitsRequested"></param>
  private void UpdateNNYieldEstimate(int numVisitsRequested)
  {
    const int MIN_BATCH_SIZE_FOR_ESTIMATE = 8; // tiny batches are too noisy
    if (numVisitsRequested >= MIN_BATCH_SIZE_FOR_ESTIMATE)
    {
      const float EMA_ALPHA = 0.25f; // average over roughly the last few batches
      float yield = Math.Clamp((float)PathsSet.NNPaths.Count / numVisitsRequested, 0.05f, 1.0f);
      emaNNYieldPerVisit = (1.0f - EMA_ALPHA) * emaNNYieldPerVisit + EMA_ALPHA * yield;
    }
  }


  /// <summary>
  /// Possibly runs another selection pass thru the graph to top up the batch so the
  /// number of positions sent to the NN evaluator fills the evaluator's padded batch
  /// capacity (see NNEvaluator.PaddedBatchCapacity). The padding slots are computed by
  /// the device regardless, so filling them with real positions is (nearly) free.
  ///
  /// Because some of the requested visits will not yield an NN evaluation (transpositions,
  /// terminals, collisions), the request is oversized by the tracked NN yield ratio
  /// (slightly conservatively). Overshoot beyond the evaluator capacity is prevented by
  /// arming PathsSet.NNEvalSlotLimit: once the budget is reached, further descents
  /// are aborted (in MCGSSelect.CapacityAbortNeeded) before any new node is expanded.
  ///
  /// Returns true if this method handled the batch (fill performed or not needed),
  /// or false if the evaluator reported no padding (so the caller may fall back to
  /// the divisor-based alignment logic).
  /// </summary>
  /// <param name="batchSize"></param>
  /// <param name="hardMaxRootN"></param>
  private bool PossiblyRunSecondSelectionToFillEvaluatorCapacity(int batchSize, int hardMaxRootN)
  {
    int numNNPaths = PathsSet.NNPaths.Count;
    if (numNNPaths == 0 || IsApproachingMaxPathCapacity)
    {
      return true;
    }

    int capacity = EvaluatorNN.Evaluator.PaddedBatchCapacity(numNNPaths);
    int numFiller = capacity - numNNPaths;
    if (numFiller <= 0)
    {
      // Either already exactly at capacity or evaluator does not report padding.
      return false;
    }

    int rootN = Engine.SearchRootNode.N;

    // Quality guard: only fill when the tree is large relative to the filled batch
    // (otherwise the extra visits would degrade selection quality via collisions).
    if (rootN < 100 * capacity)
    {
      return true;
    }

    // Don't bother if the absolute number of filler positions is small
    const int MIN_FILLER_FOR_EXTRA_SELECTION_PASS = 8;
    if (numFiller < MIN_FILLER_FOR_EXTRA_SELECTION_PASS)
    {
      return true;
    } 

    // Oversize the request to compensate for visits which will not yield an NN evaluation,
    // slightly conservatively (favor a small undershoot over overshoot).
    const float OVERSIZE_CONSERVATISM = 0.75f;
    const float MAX_OVERSIZE_MULTIPLIER = 2.5f;
    float multiplier = Math.Clamp(OVERSIZE_CONSERVATISM / emaNNYieldPerVisit, 1.0f, MAX_OVERSIZE_MULTIPLIER);
    int numVisitsToRequest = (int)(numFiller * multiplier);

    // Respect the remaining node budget of the search.
    int positionsBeforeHardBatchLimit = hardMaxRootN - (rootN + batchSize + Engine.numVisitsInFlight);
    numVisitsToRequest = Math.Min(numVisitsToRequest, positionsBeforeHardBatchLimit);

    // Respect maximum batch size (in terms of allocated paths).
    numVisitsToRequest = Math.Min(numVisitsToRequest, Engine.Manager.ParamsSearch.Execution.MaxBatchSize - numAllocatedPaths);

    if (numVisitsToRequest <= 0)
    {
      return true;
    }

    //Console.WriteLine(rootN + " " + batchSize + " " + numVisitsToRequest + " " + emaNNYieldPerVisit);

    // Arm the NN slot budget so the oversized request cannot overshoot the capacity
    // (descents are aborted once NNPaths reaches the limit), then run the fill pass.
    PathsSet.NNEvalSlotLimit = capacity;
    try
    {
      RunSelectionPhase(numAllocatedPaths + numVisitsToRequest);
    }
    finally
    {
      PathsSet.NNEvalSlotLimit = int.MaxValue;
    }

    return true;
  }


  internal int BatchSequenceNum => batchSequenceNum;


  static void DumpOverlapInPositions(MCGSPath path, GNode node2, bool verbose)
  {
    void DoWriteLine(string str) { if (verbose) Console.WriteLine(str); };

    DoWriteLine("\r\nANALYZE OVERLAP");
    if (!node2.CalcPosition().EqualPiecePositionsIncludingEnPassant(path.LeafNode.CalcPosition()))
    {
      throw new Exception("Expected same ending positions");
    }

    DoWriteLine("------> " + node2.CalcPosition());

    Dictionary<MGPosition, MCGSPathVisitMember> priorPositionsPath = [];
    DoWriteLine("FROM PATH ");
    int j = 0;
    foreach (MCGSPathVisitMember pathVisit in path.PathVisitsLeafToRoot)
    {
      DoWriteLine(j++ + " #" + pathVisit.PathVisitRef.ChildNode.Index + " " + pathVisit.PathVisitRef.ChildPosition.ToPosition.FEN + " " + pathVisit.PathVisitRef);

      priorPositionsPath[pathVisit.PathVisitRef.ChildPosition] = pathVisit;

      if (pathVisit.PathVisitRef.MoveIrreverisible)
      {
        break;
      }
    }

    Dictionary<MGPosition, GNode> priorPositionsNode2 = new()
    {
      [node2.CalcPosition()] = node2
    };
    GNode currentNode = node2;
    bool haveSeenFalse = false;
    DoWriteLine("\r\nFROM SIBLING");
    int i = 0;
    while (true)
    {
      bool found = priorPositionsPath.ContainsKey(currentNode.CalcPosition());
      haveSeenFalse |= !found;

      DoWriteLine(i++ + " " + found + " #" + currentNode.Index + " " + currentNode.CalcPosition().ToPosition.FEN);
      priorPositionsNode2[currentNode.CalcPosition()] = currentNode;

      ParentEdgesEnumerator enumer = currentNode.ParentEdges.GetEnumerator();
      enumer.MoveNext();
      GEdge parentEdge = enumer.Current;
      
      GNode nextNode = parentEdge.ParentNode;
      if (nextNode.IsSearchRoot)
      {
        break;
      }

      MGPosition nextPosition = nextNode.CalcPosition();
      MGMove move = MCGSEvaluatorNeuralNet.MoveBetweenPositions(nextPosition.ToPosition, currentNode.CalcPosition().ToPosition);
      if (currentNode.CalcPosition().IsIrreversibleMove(move, nextPosition))
      {
        break; 
      }
      currentNode = nextNode;
    }

    if (!haveSeenFalse)
    {
      throw new Exception("Internal error, sibling should have mapped to same internal sequence node; multiset same.");
    }
  }


  /// <summary>
  /// Checks all siblings of leaf node and makes sure they are not
  /// multiset equivalents of this node (in which case they should have been merged).
  /// Throws Exception if any found.
  /// </summary>
  /// <param name="manager"></param>
  /// <param name="path"></param>
  /// <param name="verbose"></param>
  static void DoVerifySiblingsAreNotMultisetEquivalent(MCGSManager manager, MCGSPath path, bool verbose)
  {
    void DoWriteLine(string str) { if (verbose) Console.WriteLine(str); };
    
    MCGSPathVisit leafVisit = path.LeafVisitRef;
    if (leafVisit.ParentChildEdge.Type.IsTerminal())
    {
      return;
    }

    GNode leafNode = leafVisit.ChildNode;

    int numDumped = 0;
    PosHash64WithMove50AndReps hash64WithMove50AndReps
    = MGPositionHashing.Hash64WithMove50AndRepsAdded(leafVisit.ChildNode.HashStandalone,
                                                     leafVisit.ChildNode.HasRepetitions ? 1 : 0,
                                                     leafVisit.ChildNode.NodeRef.Move50Category);
    if (leafNode.Graph.transpositionsPosStandalone.TryGetValue(hash64WithMove50AndReps, out GNodeIndexSetIndex setIndex)
         && !setIndex.IsNull)
    {
      // Get the NodeIndexSet from the store
      NodeIndexSet siblingsSet = leafNode.Graph.NodeIndexSetStore.sets[setIndex.NodeSetIndex];
      if (!siblingsSet.IsSingleton(leafNode.Index))
      {
        for (int i = 0; i < siblingsSet.Count; i++)
        {
          NodeIndex info = siblingsSet[i];
          GNode siblingNode = manager.Engine.Graph[info];
          if (siblingNode.Index != leafNode.Index)
          {
            if (verbose && numDumped == 0)
            {
              Console.WriteLine("\r\n\r\n-----------------------------------");
              Console.WriteLine("DumpOverlapInSiblingPositions");
              path.DumpAllVisits();
            }
            DumpOverlapInPositions(path, siblingNode, verbose);
            numDumped++;
          }
        }
      }
    }
  }


  void RunSelectionPhase(int batchSize, bool debugMode = false)
  {
    int startNumPaths = PathsSet.Paths.Count;
    int visitsRemaining = batchSize - PathsSet.Paths.Count;
    int thisSelectSize = visitsRemaining;

    Engine.Select.ExtendPathsRecursively(this, null, thisSelectSize);

    // Wait for any spawned selection tasks to complete and then clear tasks list.
    Engine.SelectWorkerPools[IteratorID]?.WaitAll();

    if (MCGSParamsFixed.LOGGING_ENABLED)
    {
      foreach (MCGSPath path in PathsSet.Paths)
      {
        // Update TerminationReason stats
        MCGSMetrics.PathTerminationResultHits.Add(1, pathTerminationTags[(int)path.TerminationReason]);
      }
    }

#if DEBUG
    HashSet<GEdge> seenEdges = new();
    foreach (MCGSPath path in PathsSet.Paths)
    {
      path.DebugValidateState(true);

      bool cycleExists = path.DebugCheckCycleExists;
      if (cycleExists)
      {
        path.DumpAllVisits();
        throw new Exception("Cycle detected in path: " + path.ToString());
      }

#if NOT_UNTIL_FIXED_BECAUSE_ALIGNMENT_ALLOWS_THIS
      // Verify no GEdge appears in more than slot of any path or across paths in the batch.
      for (int i = 0; i < path.numSlotsUsed; i++)
      {
        if (!seenEdges.Add(path.slots[i].ParentChildEdge))
        {
          throw new Exception("Edge visited multiple times in batch: " + path.slots[i].ParentChildEdge);
        }
      }
#endif
    }
    //    int crossing = PathsSet.CheckCrossingPathVisits(true);
    //    int crossing = PathsSet.CheckCrossingPathVisitsByEdge(true);
#endif
  }


  /// <summary>
  /// Executes the neural network evaluation phase for a batch of paths. This phase identifies paths requiring
  /// evaluation, processes them in batches through the neural network, and applies the evaluation results to the
  /// corresponding nodes.
  /// </summary>
  void RunNNEvaluationPhase(bool deferRetrieveResults)
  {
    // Perform the batched evaluation. EvaluatorLock is non-null only when a single evaluator is
    // shared by the two overlapped iterators (DualEvaluators=false); with distinct evaluators the
    // two GPU evaluations are allowed to run concurrently (the lock is null).
    if (PathsSet.NNPaths.Count > 0)
    {
      using (Engine.EvaluatorLock?.Acquire())
      {
        EvaluatorNN.BatchGenerate(Engine, PathsSet.NNPaths, deferRetrieveResults);
      }

      Engine.Graph.RegisterNNBatch(PathsSet.NNPaths.Count);
      Interlocked.Add(ref Manager.NumEvalsThisSearch, PathsSet.NNPaths.Count);
    }
  }


  /// <summary>
  /// Applies the deferred ledger backouts for visits dropped during the select phase
  /// (see MCGSPathsSet.PendingDroppedVisits). Must run before any path backup so the
  /// merge counters at shared ancestor visits reflect only the visits actually carried.
  /// </summary>
  internal void ApplyPendingDroppedVisits()
  {
    while (PathsSet.PendingDroppedVisits.TryDequeue(out (MCGSPath path, int numSlotsUsed, int numVisits) drop))
    {
      MCGSSelect.ApplyDroppedVisits(drop.path, drop.numSlotsUsed, drop.numVisits);
    }
  }


  /// <summary>
  /// Executes the backup phase for the current set of paths using the specified backup mode.
  /// </summary>
  void RunBackupPhase()
  {
    ApplyPendingDroppedVisits();

#if DEBUG
    foreach (MCGSPath path in PathsSet.Paths)
    {
      if (path.TerminationReason == MCGSPathTerminationReason.NotYetTerminated)
      {
        throw new Exception("Found NotYetTerminated in paths remaining to backup"); ;
      }
    }

    VerifyAllPathVisitNodesPrepared(!PhaseCoordinator.ENABLE_CONCURRENT_EVALUATE_WITH_BACKUP_SELECT);
#endif

    switch (BackupMode)
      {
        case BackupMethodEnum.ReductionSingleThread:
          Engine.Backup.BackupAllSingleReduction(PathsSet.Paths, backupStrategy, this);
          break;

        case BackupMethodEnum.ReductionMultiThread:
          Engine.Backup.BackupAllParallelReduction(PathsSet.Paths, backupStrategy, this);
          break;

        default:
          throw new NotSupportedException("Unknown backup mode: " + Engine.Manager.ParamsSearch.Execution.BackupMode);
      }

    if (PathsSet.NNPaths.Count > 0)
    {
      EvaluatorNN.Evaluator.BuffersLock?.Release();
    }
  }


  /// <summary>
  /// Verifies that all nodes referenced by path visits are unlocked.
  /// </summary>
  private void VerifyAllPathVisitNodesPrepared(bool expectAllUnlocked)
  {
    foreach (MCGSPath path in PathsSet.Paths)
    {      
      if (expectAllUnlocked)
      {
        foreach (MCGSPathVisitMember visitiRef in path.PathVisitsLeafToRoot)
        {
          ref readonly GEdge pathEdge = ref visitiRef.PathVisitRef.ParentChildEdge;
          if (pathEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
          {
            byte lockValue = pathEdge.ChildNode.NodeRef.LockRef.StateRaw;

            if (pathEdge.ChildNode.IsLocked)
            {
              Console.WriteLine("active: " + Engine.Coordinator.NumActive);
              throw new Exception(pathEdge.ChildNode + " Found locked node before Backup phase begins " + lockValue);
            }
          }
        }
      }
    }
  }


  /// <summary>
  /// Returns a string representation.
  /// </summary>
  /// <returns></returns>
  public override string ToString() => $"<MCGSIterator #{IteratorID}>";
}
