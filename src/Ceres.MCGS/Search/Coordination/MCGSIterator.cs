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

  internal int numAllocatedPaths = 0;

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

  internal void ResetPaths() => numAllocatedPaths = 0;


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
    paths = new MCGSPath[maxBatchSize];

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
    Engine.Coordinator.EnterSelect(IteratorID, thisBatchID);
    LogWrite(() => $"Start select batch {batchSequenceNum}");

    RunSelectionPhase(batchSize);

    PossiblyRunSecondSelectionForNNBatchSizePadding(batchSize, hardMaxRootN);

    LogWrite(() => $"End select batch {batchSequenceNum} with {PathsSet.Paths.Count} paths");
    Engine.Coordinator.ExitSelect(IteratorID, thisBatchID);


    if (PathsSet.Paths.Count == 0)
    {
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
        PathsSet.Paths.ToArray()[i].DumpAllVisits();
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
          if (visit.PathVisitRef.NumVisitsAttemptedPendingBackup > 0)
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

    // Possibly invoke the callback
    // At this point we hold the select/backup lock
    // and graph is quiescent.
    Engine.PossiblyInvokeCallback();

    LogWrite(() => $"End backup with mode {BackupMode}");
    Engine.Coordinator.ExitBackup(IteratorID, thisBatchID);

    LogFlush();

    // Apply search moves as soon as possible (need the root to have been evaluated).
    Engine.Manager.TerminationManager.ApplySearchMovesIfNeeded();

    Manager.RunPeriodicMaintenance(batchSequenceNum);
    batchSequenceNum++;

    Engine.PossiblySynchronizeIterators(this);
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
    // Perform the batched evaluation.
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
  /// Executes the backup phase for the current set of paths using the specified backup mode. 
  /// </summary>
  void RunBackupPhase()
  {
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
