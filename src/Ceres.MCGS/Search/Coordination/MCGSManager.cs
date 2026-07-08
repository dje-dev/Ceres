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
using System.Linq;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.Positions;

using Ceres.MCGS.Environment;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Phases.Evaluation;

#endregion

namespace Ceres.MCGS.Search.Coordination;

public partial class MCGSManager : IDisposable
{
  public static bool ENABLE_SHARED_NN_ALLOCATOR = true;


  /// <summary>
  /// Callback called periodically during search to facilitate
  /// tracking of search progress.
  /// </summary>
  /// <param name="manager"></param>
  public delegate void MCGSProgressCallback(MCGSManager manager);

  public MCGSEngine Engine;
  public MCGSProgressCallback ProgressCallback;
  public Dictionary<ulong, int> TranspositionDict = null;

  public readonly ParamsSearch ParamsSearch;
  public readonly ParamsSelect ParamsSelect;

  /// <summary>
  /// Definition of neural network evaluator
  /// </summary>
  public NNEvaluatorDef EvaluatorDef => EvaluatorsSet.EvaluatorDef;

  /// <summary>
  /// Set of underlying neural network evaluators.
  /// </summary>
  public NNEvaluatorSet EvaluatorsSet;

  /// <summary>
  /// Primary neural network evaluator
  /// </summary>
  public NNEvaluator NNEvaluator0 => EvaluatorsSet.Evaluator0;

  /// <summary>
  /// Secondary neural network evaluator (optional)
  /// </summary>
  public NNEvaluator NNEvaluator1 => EvaluatorsSet.Evaluator1;

  /// <summary>
  /// Primary neural network evaluator wrapper
  /// </summary>
  public readonly MCGSEvaluatorNeuralNet EvaluatorNN0;

  /// <summary>
  /// Secondary neural network evaluator wrapper (optional)
  /// </summary>
  public readonly MCGSEvaluatorNeuralNet EvaluatorNN1;


  /// <summary>
  /// Position and prior moves from the start of the search.
  /// </summary>
  public PositionWithHistory StartPosAndPriorMoves => Engine?.Graph.Store.NodesStore.PositionHistory;

  /// <summary>
  /// Position at the root of the search
  /// </summary>
  public MGPosition RootMGPos => Engine.SearchRootNode.CalcPosition();


  /// <summary>
  /// If tablebase evaluations should not be marked as terminal
  /// (needed when root position is a win but no DTZ files available).
  /// </summary>
  public readonly bool ForceNoTablebaseTerminals;


  /// <summary>
  /// Count of tablebase hits since beginning of search.
  /// </summary>
  public long CountTablebaseHits => evaluatorTB == null ? 0 : evaluatorTB.NumHits.Value;

  internal EvaluatorSyzygy evaluatorTB;
  internal bool TablebaseDTZAvailable;

  bool searchWasStarted = false;


  /// <summary>
  /// Current status of search.
  /// </summary>
  public SearchStopStatus StopStatus = SearchStopStatus.Continue;

  private bool disposed;
  private MCGSIterator iterator0;
  private MCGSIterator iterator1;

  public float FractionExtendedSoFar = 0;


  public float AvgDepth => Engine.AvgPathDepth;
  public int MaxDepth => Engine.MaxPathDepth;

  SearchLimit LastSearchLimit;


  public CheckTablebaseBestNextMoveDelegate CheckTablebaseBestNextMove;


  /// <summary>
  /// Futility pruning manager associated with this search
  /// (for determining if and when top-level moves should be not further searched).
  /// </summary>
  public MCGSFutilityPruning TerminationManager;

  /// <summary>
  /// Time manager associated with this search
  /// (for allocating time or nodes searched to each node).
  /// </summary>
  public readonly IManagerGameLimit LimitManager;

  /// <summary>
  /// Time when search method was first invoked.
  /// </summary>
  public DateTime StartTimeThisSearch;

  /// <summary>
  /// Time when visits were started 
  /// (after any preparatory steps such as graph reuse preparation).
  /// </summary>
  public DateTime StartTimeFirstVisit;

  /// <summary>
  /// Search limit initially allocated.
  /// </summary>
  public SearchLimit SearchLimitInitial;

  /// <summary>
  /// Search limit used as of last set of iterations 
  /// (possibly multiple of search was extended).
  /// </summary>
  public SearchLimit SearchLimit;

  /// <summary>
  /// Optional fixed search limit known at engine creation time.
  /// When specified, allows optimizations for small searches.
  /// </summary>
  public readonly SearchLimit FixedSearchLimit;


  public bool ExternalStopRequested;

  public readonly bool IsFirstMoveOfGame;

  public readonly List<GameMoveStat> PriorMoveStats;


  /// <summary>
  /// Pruning status of root moves which may flag certain 
  /// moves as not being eligible for additional search visits.
  /// 
  /// These status be progressively toggled as search progresses and it becomes
  /// clear that certain children can never "catch up" to 
  /// the current highest N node and thus would never be chosen
  /// (instead we can allocate remaining visits over children still having a chance)
  /// </summary>
  public Managers.MCGSFutilityPruningStatus[] RootMovesPruningStatus;

  /// <summary>
  /// The N of the root node when search started
  /// (possibly nonzero due to graph reuse)
  /// </summary>
  public int RootNWhenSearchStarted;

  /// <summary>
  /// Number of MCGS visits actually taken so far in this search (not including initial graph).
  /// </summary>
  public int NumNodesVisitedThisSearch => Engine.SearchRootNode.N - RootNWhenSearchStarted;

  /// <summary>
  /// Number of nerual network evaluations made for this search.
  /// </summary>
  public int NumEvalsThisSearch;

  /// <summary>
  /// True wall-clock seconds of the last search: from when the search method was first
  /// invoked (StartTimeThisSearch) until the move was actually decided (end of DoSearch).
  /// This is captured at search completion (NOT recomputed live), so it is not inflated by
  /// any idle time before a later dump command. NaN until a search has completed.
  /// </summary>
  public double TimeElapsedTotalSeconds = double.NaN;

  /// <summary>
  /// Cumulative wall-clock seconds during the last search in which at least one of
  /// the evaluators (Evaluator0/Evaluator1) was executing inside the backend
  /// (C++ interop boundary). The ratio TimeDeviceBackendWaitSeconds/TimeElapsedTotalSeconds
  /// approaches 1.0 as the whole move becomes GPU-bound (i.e. C# overhead approaches 0).
  /// Reported as NaN when the backend does not support this instrumentation
  /// (currently NNEvaluatorTensorRT and NNEvaluatorCUDA).
  /// </summary>
  public double TimeDeviceBackendWaitSeconds = double.NaN;

  /// <summary>
  /// Complement of TimeDeviceBackendWaitSeconds: cumulative seconds during the last
  /// search in which NEITHER evaluator was inside the backend (GPU idle / C# overhead).
  /// NaN when unsupported.
  /// </summary>
  public double TimeDeviceBackendIdleSeconds = double.NaN;


  public ManagerGameLimitInputs LastGameLimitInputs;
  public ManagerGameLimitOutputs LastGameLimitOutputs;


  /// <summary>
  /// Replaces this move's search limit after the manager was constructed.
  /// Used when a reused graph is abandoned after the initial (warm) time/node allocation:
  /// the search will actually run from a cold (empty) graph, so the budget is recomputed
  /// with a cold-start input and installed here.
  /// </summary>
  internal void OverrideSearchLimit(SearchLimit newLimit,
                                    ManagerGameLimitInputs gameLimitInputs,
                                    ManagerGameLimitOutputs gameLimitOutputs)
  {
    SearchLimit = newLimit;
    SearchLimitInitial = newLimit;
    LastSearchLimit = newLimit;
    LastGameLimitInputs = gameLimitInputs;
    LastGameLimitOutputs = gameLimitOutputs;
  }


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="evaluatorSet"></param>
  /// <param name="paramsSearch"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="forceNoTablebaseTerminals"></param>
  /// <param name="searchMovesTablebaseRestricted"></param>
  /// <param name="fixedSearchLimit"></param>
  public MCGSManager(NNEvaluatorSet evaluatorSet,
                     ParamsSearch paramsSearch,
                     ParamsSelect paramsSelect,
                     SearchLimit searchLimit,
                     IManagerGameLimit limitManager,
                     DateTime startTime,
                     List<GameMoveStat> gameMoveHistory,
                     bool isFirstMoveOfGame,
                     bool forceNoTablebaseTerminals,
                     List<MGMove> searchMovesTablebaseRestricted,
                     bool engineIsWhite,
                     SearchLimit fixedSearchLimit = null)
  {
    // Ensure engine initialization is performed (thread-safe, only runs once)
    MCGSEngineInitialization.BaseInitialize();

    if (searchLimit.IsPerGameLimit)
    {
      throw new Exception("Per game search limits not supported");
    }

    ParamsSearch = paramsSearch;
    ParamsSelect = paramsSelect;
    EvaluatorsSet = evaluatorSet;
    ForceNoTablebaseTerminals = forceNoTablebaseTerminals;

    StartTimeThisSearch = startTime;

    SearchLimit = searchLimit;
    SearchLimitInitial = searchLimit;
    FixedSearchLimit = fixedSearchLimit;
    if (searchLimit.IsPerGameLimit)
    {
      throw new Exception("Per game search limits not supported");
    }

    IsFirstMoveOfGame = isFirstMoveOfGame;

    // Make our own copy of move history.
    PriorMoveStats = [];
    if (gameMoveHistory != null)
    {
      PriorMoveStats.AddRange(gameMoveHistory);
    }

    LastSearchLimit = searchLimit;

#if NOT
// Can't do this here, don't have the position at hand
    if (searchMovesTablebaseRestricted == null && searchLimit.SearchMoves != null)
    {
      Position startPos = pos.FinalPosition;
      foreach (Move move in searchLimit.SearchMoves)
      {
        searchMovesTablebaseRestricted.Add(MGMoveConverter.MGMoveFromPosAndMove(startPos, move));
      }
    }
#endif
#if NOT
    List<MGMove> searchMovesTablebaseRestricted = null;
    if (searchLimit.SearchMoves != null)
    {
      Position startPos = pos.FinalPosition;
      foreach (Move move in searchLimit.SearchMoves)
      {
        searchMovesTablebaseRestricted.Add(MGMoveConverter.MGMoveFromPosAndMove(startPos, move));
      }
    }
#endif
    TerminationManager = new MCGSFutilityPruning(this, searchLimit.SearchMoves, searchMovesTablebaseRestricted);
    LimitManager = limitManager;


    // Create a buffer synchronization object so we can prevent
    // overlapping executors from concurrently using the output buffers.
    // Skip this for a pooled evaluator: it returns freshly allocated result buffers to each caller
    // (so there is no shared output buffer to protect) and the evaluator instance is shared across
    // many engines, so a single lock on it would be incorrectly shared (and its count corrupted).
    if (!ParamsSearch.Execution.DualEvaluators && NNEvaluator0 is not NNEvaluatorPooled)
    {
      NNEvaluator0.BuffersLock = new System.Threading.SemaphoreSlim(1, 1);
    }

    // Possibly allocate a device allocator to be shared across evaluators.
    // This enables possible concurrent non-overlapping placement across devices.
    ItemsInBucketsAllocator nnDeviceAllocator = null;
    if (ENABLE_SHARED_NN_ALLOCATOR
     && paramsSearch.Execution.DualOverlappedIterators
     && NNEvaluator0 is NNEvaluatorSplit splitEvaluator)
    {
      nnDeviceAllocator = new(splitEvaluator.PreferredFractions);

      // Use a larger min split size if there are many devices
      // because in this situation using fewer devices per batch
      // may keep sufficient devices free for unencumbered use by another concurrent batch.
      const int THRESHOLD_NUM_DEVICES_LARGER_MIN_SPLIT_SIZE = 4;
      bool manyDevices = splitEvaluator.Evaluators.Length >= THRESHOLD_NUM_DEVICES_LARGER_MIN_SPLIT_SIZE;
      splitEvaluator.MinSplitSize = manyDevices ? 32 : 16;
    }

    const bool LOW_PRIORITY = false;
    EvaluatorNN0 = new MCGSEvaluatorNeuralNet(EvaluatorsSet.EvaluatorDef, NNEvaluator0, nnDeviceAllocator,
                                              paramsSearch.HistoryFillIn,
                                              Math.Min(NNEvaluator0.MaxBatchSize, ParamsSearch.Execution.MaxBatchSize),
                                              LOW_PRIORITY, paramsSearch.ValueTemperature,
                                              paramsSearch.EnableState,
                                              null,  // GFIX: context.Tree.PositionCache,
                                              null, engineIsWhite); // batch index dynamic selector

    if (paramsSearch.Execution.DualOverlappedIterators)
    {
      NNEvaluator evaluatorToUse = ParamsSearch.Execution.DualEvaluators ? NNEvaluator1 : NNEvaluator0;
      EvaluatorNN1 = new MCGSEvaluatorNeuralNet(EvaluatorsSet.EvaluatorDef,
                                                evaluatorToUse, nnDeviceAllocator,
                                                paramsSearch.HistoryFillIn,
                                                Math.Min(evaluatorToUse.MaxBatchSize, ParamsSearch.Execution.MaxBatchSize),
                                                LOW_PRIORITY, paramsSearch.ValueTemperature,
                                                paramsSearch.EnableState,
                                                null, // GFIX: context.Tree.PositionCache,
                                                null, engineIsWhite); // batch index dynamic selector
    }

    // TODO: cleanup? see notes relating to forceNoTablebaseTerminals in MCGSSearch.cs
    if (paramsSearch.EnableTablebases && paramsSearch.TablebasePaths != null && !forceNoTablebaseTerminals)
    {
      evaluatorTB = new EvaluatorSyzygy(paramsSearch.TablebasePaths, ForceNoTablebaseTerminals);
      CheckTablebaseBestNextMove = RootTablebaseMoveCheck;
      TablebaseDTZAvailable = evaluatorTB.Evaluator.DTZAvailable;

      // TODO: Also add a 1-ply lookahead evaluator (for captures yielding tablebase terminal)
      //evaluatorTBPly1 = new LeafEvaluatorSyzygyPly1(evaluatorTB, Manager.ForceNoTablebaseTerminals);
    }

    //    LeafEvaluatorSyzygyPly1 evaluatorTBPly1;

    MGMove RootTablebaseMoveCheck(in Position currentPos, out WDLResult result, out List<(MGMove, short)> fullWinningMoveList, out bool winningMoveListOrderedByDTM)
    {
      MGMove ret = evaluatorTB.Evaluator.CheckTablebaseBestNextMove(in currentPos, out result, out fullWinningMoveList, out winningMoveListOrderedByDTM);
      return ret;
    }
  }


  public void DumpParams()
  {
    Console.WriteLine(ObjUtils.FieldValuesDumpString<SearchLimit>(SearchLimit, SearchLimit.NodesPerMove(1), false));
    //      writer.Write(ObjUtils.FieldValuesDumpString<NNEvaluatorDef>(Def.NNEvaluators1.EvaluatorDef, new ParamsNN(), differentOnly));
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSelect>(ParamsSelect, new ParamsSelect(), false));
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearch>(ParamsSearch, new ParamsSearch(), false));
    //DumpTimeManagerDifference(differentOnly, null, timeManager1);
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(ParamsSearch.Execution, new ParamsSearchExecution(), false));
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsRootMinimaxBlend>(ParamsSearch.RootMinimaxBlend, new ParamsRootMinimaxBlend(), false));
  }


  bool haveWarnedRunPeriodicMaintenanceException = false;

  internal void RunPeriodicMaintenance(int batchSequenceNum)
  {
    // Use this time to perform housekeeping (graph is quiescent)
    using (new NodeLockBlock(Engine.SearchRootNode))
    {
      try
      {
        // UpdatePruningFlags is comparatively expensive (and only does real work in the latter
        // part of the search), so throttle it to every 3rd batch rather than running it every
        // batch on the critical path while the select/backup exclusion lock is held. Use a
        // different residue than UpdateEstimatedNPS (% 3 == 2) so the two do not stack on the
        // same batch. UpdateSearchStopStatus stays every-batch so stop detection is not delayed.
        if (batchSequenceNum % 3 == 0)
        {
          TerminationManager.UpdatePruningFlags();
        }
        UpdateSearchStopStatus();
      }
      catch (Exception exc)
      {
        if (!haveWarnedRunPeriodicMaintenanceException)
        {
          Console.WriteLine("RunPeriodicMaintenance Exception" + exc);
          haveWarnedRunPeriodicMaintenanceException = true;
        }
      }
    }

    if (batchSequenceNum % 3 == 2) // TODO: update batchSequenceNum somewhere
    {
      UpdateEstimatedNPS();
    }
  }



  public static (MGMove, BestMoveInfoMCGS) DoSearch(MCGSManager manager,
                                                    bool verbose, MCGSProgressCallback progressCallback = null,
                                                    bool moveImmediateIfOnlyOneMove = false,
                                                    MGMove forceMove = default)
  {
    if (manager.searchWasStarted)
    {
      throw new Exception("Cannot reuse MCGSManager for multiple searches.");
    }
    manager.searchWasStarted = true;

    manager.StartTimeThisSearch = DateTime.Now;
    manager.RootNWhenSearchStarted = manager.Engine.SearchRootNode.N;
    manager.NumEvalsThisSearch = 0;

    // Reset the backend-time tracker once at the true start of the search so backend-busy time
    // accumulates across all passes (including any extension passes) over the whole move.
    manager.EvaluatorsSet?.BackendTimeTracker?.Reset();
    manager.Engine.Coordinator.ResetPhaseTiming();

    // Reset the per-batch ID sequence and the in-order-backup turn so batch indices remain
    // contiguous from 0 within each search (required by EnforceInOrderBackup).
    manager.Engine.nextBatchID = 0;
    manager.Engine.Coordinator.ResetBackupOrder();
    manager.Engine.Coordinator.ResetBackupOrderStats();

    PositionWithHistory priorMoves = manager.Engine.Graph.Store.NodesStore.PositionHistory;

    // Make sure not already checkmate/stalemate
    GameResult terminalStatus = priorMoves.FinalPosition.CalcTerminalStatus();
    if (terminalStatus != GameResult.Unknown)
    {
      throw new Exception($"The initial position is terminal: {terminalStatus} {priorMoves.FinalPosition.FEN}");
    }

    // If only one legal move then force the search to run only one node
    // (to get root populated) and then stop.
    MGPosition startPos = priorMoves.FinalPosMG;
    MGMoveList moves = new();
    MGMoveGen.GenerateMoves(in startPos, moves);

    // Possibly initialize cache
    if (manager.ImmediateTablebaseBestMoveIsAvailable(manager.Engine.SearchRootNode))
    {
      // Force root position to be evaluated using neural net to initialize some fields.
      manager.Engine.RunLoop(1);

      // The overwrite some fields with results from tablebase hit.
      manager.TrySetImmediateBestMove(manager.Engine.SearchRootNode);

      manager.StopStatus = SearchStopStatus.TablebaseImmediateMove;
      return (manager.TablebaseImmediateBestMove, null);
    }

    // Check if playing using only action head.
    if (manager.SearchLimit.Type == SearchLimitType.BestActionMove)
    {
      return (ChooseActionHeadMove(manager, priorMoves), null);
    }


    // Check if playing using only value head (TopV).
    if (manager.SearchLimit.Type == SearchLimitType.BestValueMove)
    {
      return (ChooseValueHeadMove(manager, priorMoves, ref moves), null);
    }

    bool shouldStopAfterTwoNodesDueToOnlyOneLegalMove = false;
    if (moveImmediateIfOnlyOneMove && moves.NumMovesUsed == 1)
    {
      // Need to get at least two nodes (i.e. one edge from root)
      // otherwise the SearchPrincipalVariationMCGS will fail
      // (for example see position R6Q/6pk/1q6/7p/8/5PP1/6K1/8 b - - 5 1).
      shouldStopAfterTwoNodesDueToOnlyOneLegalMove = true;
      manager.SearchLimit = SearchLimit.NodesPerMove(2);
    }
    else
    {
      PossiblyPrefetch(manager);
    }

    GNode root = manager.Engine.SearchRootNode;
    BestMoveInfoMCGS bestMoveInfo;

    //root.MarkImmediateDrawsByRepetition();

    // Do the search
    SearchLimit thisSearchLimit = manager.SearchLimit with { };
    int numSearches = 0;
    SearchLimit startingSearchLimit = manager.SearchLimit with { };

    bool shouldExtendSearch;
    do
    {
      shouldExtendSearch = false;

      TimingStats timingStats = manager.DoSearchInner(thisSearchLimit, progressCallback);

      // Get best child 
      // TODO: technically the "updateStats" should only be true if we end up accepting this move
      if (forceMove != default)
      {
        if (!moves.Contains(forceMove))
        {
          throw new Exception($"Specified forced move {forceMove} is not legal in position {priorMoves.FinalPosition.FEN}");
        }
      }

      ManagerChooseBestMoveMCGS chooseMCGS = new(manager, manager.Engine.SearchRootNode, true, default, true);
      bestMoveInfo = chooseMCGS.BestMoveCalc;

      if (manager.ParamsSearch.EnableSearchExtension)
      {
        throw new NotImplementedException("EnableSearchExtension disabled below");
      }
#if NOT
        // If the chosen move is far away from the best Q node, 
        // try to extend the search unless the position is obviously won/lost.
        const float Q_THRESHOLD = 0.01f;
        const int MAX_RETRIES = 3;
        const float INCREMENT_FRACTION = 0.20f;
        bool possiblyExtend = bestMoveInfo.BestMoveQSuboptimality > Q_THRESHOLD; // other move has much better Q already
        if (manager.Context.ParamsSearch.EnableSearchExtension
         && !shouldStopAfterOneNodeDueToOnlyOneLegalMove
         && possiblyExtend
         && root.Q < 0.75f                                     // don't retry if position is already won
         && numSearches < MAX_RETRIES                          // don't retry many times to avoid using too much extra time
         && manager.NumNodesVisitedThisSearch > 100            // don't retry for very small searches to because batch sizing make this imprecise
         && manager.FractionExtendedSoFar <
            startingSearchLimit.FractionExtensibleIfNeeded)    // only extend if we haven't already extended too much
        {
          thisSearchLimit = manager.SearchLimitInitial * INCREMENT_FRACTION;

          // Make sure top N and Q are not futilty pruned because we now have more search budget.
          manager.Context.SetNodeNotFutilityPruned(bestMoveInfo.BestNNode);
          manager.Context.SetNodeNotFutilityPruned(bestMoveInfo.BestQNode);

          // Reset starting counters
          // TODO: clean this up.
          // TODO: Inefficient to restart search because of repeated initialization (e.g. create selected sets, leaf evalutors, etc.)
          manager.StartTimeThisSearch = DateTime.Now;
          manager.RootNWhenSearchStarted = manager.Root.N;

          manager.FractionExtendedSoFar += INCREMENT_FRACTION;

          if (false)
          {
            Console.WriteLine(" extend try " + numSearches + " Was " + bestMoveInfo.QOfBest + " " + bestMoveInfo.BestMove
                              + "  Extending to " + thisSearchLimit + " because QSuboptimality "
                              + bestMoveInfo.BestMoveQSuboptimality + " original limit " + manager.SearchLimit
                              + " N was " + manager.Root.N);
          }
          shouldExtendSearch = true;
          manager.StopStatus = SearchStopStatus.Continue;
        }
        else
        {
          //if (numSearches > 0) Console.WriteLine("Extended finished " + manager.Root.N);
          shouldExtendSearch = false;
          manager.UpdateTopNodeInfo();
        }

        numSearches++;
      } while (shouldExtendSearch);
#endif
      //if (numSearches > 0) Console.WriteLine("Extended finished " + manager.Root.N);
      shouldExtendSearch = false;
      using (new NodeLockBlock(manager.Engine.SearchRootNode))
      {
        manager.UpdateTopNodeInfo();
      }

      numSearches++;
    } while (shouldExtendSearch);

    // Capture the true search duration and device-backend utilization at the actual end of the
    // search (after best-move selection across all passes). Measured here rather than live so the
    // values reflect begin->end of the real search and are not inflated by idle wall-clock before
    // a later dump command. The denominator is this whole-search time, so the busy fraction
    // accounts for any C# overhead outside the GPU-bound inner loop as well.
    manager.TimeElapsedTotalSeconds = (DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;
    var backendTracker = manager.EvaluatorsSet?.BackendTimeTracker;
    manager.TimeDeviceBackendWaitSeconds = (backendTracker?.EverUsed ?? false) ? backendTracker.BusySeconds : double.NaN;
    manager.TimeDeviceBackendIdleSeconds = double.IsNaN(manager.TimeDeviceBackendWaitSeconds)
                                         ? double.NaN
                                         : Math.Max(0, manager.TimeElapsedTotalSeconds - manager.TimeDeviceBackendWaitSeconds);

    if (shouldStopAfterTwoNodesDueToOnlyOneLegalMove)
    {
      manager.StopStatus = SearchStopStatus.OnlyOneLegalMove;
      return (moves.MovesArray[0], null);
    }
    else
    {
      return (bestMoveInfo.BestMove, bestMoveInfo);
    }
  }


  bool haveWarnedInFlight;

  /// <summary>
  /// Launches the search with specified limit.
  /// </summary>
  /// <param name="searchLimit"></param>
  /// <param name="progressCallback"></param>
  /// <param name="nnRemoteEvaluatorExtraSuffix"></param>
  /// <returns></returns>
  internal TimingStats DoSearchInner(SearchLimit searchLimit, MCGSProgressCallback progressCallback)
  {
    SearchLimit = searchLimit;

    ProgressCallback = progressCallback;

    TimingStats stats = new();

    using (new TimingBlock($"MCGS SEARCH {searchLimit}", stats, TimingBlock.LoggingType.None))
    {
      int batchNum = 0;

      int? hardLimitNumNodesToCompute = null;
      bool shouldProcess = true;
      if (searchLimit.Type == SearchLimitType.NodesPerMove)
      {
        hardLimitNumNodesToCompute = (int)searchLimit.Value;
      }
      else if (searchLimit.Type == SearchLimitType.NodesPerTree)
      {
        if (Engine.SearchRootNode.N >= searchLimit.Value)
        {
          shouldProcess = false;
          StopStatus = SearchStopStatus.NodeLimitReached;
        }
        else
        {
          hardLimitNumNodesToCompute = (int)searchLimit.Value - Engine.SearchRootNode.N;
        }
      }

      StartTimeFirstVisit = DateTime.Now;
      if (shouldProcess)
      {
        int hardMaxRootN = hardLimitNumNodesToCompute == null ? int.MaxValue
                                                              : Engine.SearchRootNode.N + hardLimitNumNodesToCompute.Value;
        if (Engine.SearchRootNode.N < hardMaxRootN)
        {
          Engine.RunLoop(hardMaxRootN);
        }
      }
    }

    // Make sure nothing was left in flight after the search
    foreach (GEdge edge in Engine.SearchRootNode.ChildEdgesExpanded)
    {
      if ((edge.NumInFlight0 > 0 || edge.NumInFlight1 > 0) && !haveWarnedInFlight)
      {
        Console.WriteLine($"Internal error: search ended with N={Engine.SearchRootNode.N}  {Engine.SearchRootNode} {edge}");
        haveWarnedInFlight = true;
        break;
      }
    }

    // Possibly validate graph integrity.
    if (Engine.Manager.ParamsSearch.ValidateAfterSearch)
    {
      Engine.Graph.Validate(fastMode: true);
    }

    return stats;
  }



  private static void PossiblyPrefetch(MCGSManager manager)
  {
    if (manager.ParamsSearch.PrefetchParams != null)
    {
      GraphPrefetcher prefetcher = new(manager);

      prefetcher.DoPrefetch(manager.ParamsSearch.PrefetchParams,
                            Math.Min(manager.NNEvaluator0.MaxBatchSize, manager.ParamsSearch.Execution.MaxBatchSize),
                            manager.ParamsSearch.DebugDumpVerifyMode);

      // Possibly use the newly gathered value head info to
      // rearrange children based on attractiveness of V.
      if (manager.ParamsSearch.PrefetchParams.PrefetchResortChildrenUsingV)
      {
        prefetcher.PossiblyResortChildrenUsingV();
      }
    }
  }

  private static MGMove ChooseValueHeadMove(MCGSManager manager, PositionWithHistory priorMoves, ref MGMoveList moves)
  {
    if (manager.SearchLimit.Value != 1)
    {
      throw new Exception("BestValueMove only supported for NodesPerMove == 1.");
    }

    NNEvaluatorResult[] valueEvalResults = null;
    (MGMove bestMove, float bestV, float bestD) = BestValueMove(manager.EvaluatorNN0.Evaluator, priorMoves,
                                                                ref moves, ref valueEvalResults,
                                                                manager.ParamsSearch.HistoryFillIn, false,
                                                                manager.Engine.SearchRootPlySinceLastMove);
    GNode rootNode = manager.Engine.SearchRootNode;
    rootNode.NodeRef.Q = bestV;
    rootNode.NodeRef.D = bestD;
    rootNode.NodeRef.N = 1;
    return manager.TopVForcedMove = bestMove;
  }

  private static MGMove ChooseActionHeadMove(MCGSManager manager, PositionWithHistory priorMoves)
  {
    if (manager.SearchLimit.Value != 1)
    {
      throw new Exception("BestActionMove only supported for NodesPerMove == 1.");
    }

    NNEvaluatorResult rootResult = manager.Engine.Manager.EvaluatorNN0.Evaluator.Evaluate(priorMoves, manager.ParamsSearch.HistoryFillIn, false);

    MGPosition thisPos = priorMoves.FinalPosMG;

    MGMove bestMove = default;
    float bestActionV = float.MinValue;
    foreach ((EncodedMove Move, float Probability) policyMove in rootResult.Policy.ProbabilitySummary())
    {
      (float w, float d, float l) zz = rootResult.ActionWDLForMove(policyMove.Move);
      float actionV = -(zz.w - zz.l);
      if (actionV > bestActionV)
      {
        bestMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(policyMove.Move, thisPos);
        bestActionV = actionV;
      }
    }

    Debug.Assert(bestActionV != float.MinValue); // expected to find at least one move

    GNode rootNode = manager.Engine.SearchRootNode;
    rootNode.NodeRef.Q = bestActionV;
    rootNode.NodeRef.D = 0; // Action head doesn't provide WDL info
    rootNode.NodeRef.N = 1;
    return manager.TopVForcedMove = bestMove;
  }



  public void SetIterators(MCGSIterator iterator0, MCGSIterator iterator1 = null)
  {
    this.iterator0 = iterator0;
    this.iterator1 = iterator1;
  }


  /// <summary>
  /// 
  /// </summary>
  public void Dispose()
  {
    if (disposed)
    {
      return;
    }

    disposed = true;
    iterator0?.Dispose();
    iterator1?.Dispose();

    // Dispose the per-manager NN evaluator wrapper(s) owned by this manager. Each only
    // shuts down its own batch buffers (Batch.Shutdown()); the underlying shared
    // NNEvaluator / EvaluatorsSet is passed in and intentionally NOT released here.
    EvaluatorNN0?.Dispose();
    EvaluatorNN1?.Dispose();
    //    EvaluatorsSet?.Dispose(); // do not release, shared (passed in)
  }


  // No finalizer by design. This type holds only managed state.
}
