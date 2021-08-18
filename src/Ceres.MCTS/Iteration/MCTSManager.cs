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

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.Environment;
using Ceres.Base.Math;
using Ceres.Base.Misc;

using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.NNFiles;
using Ceres.Chess.GameEngines;
using Ceres.Chess.EncodedPositions;

using Ceres.MCTS.Evaluators;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Search;
using Ceres.MCTS.Search.IteratedMCTS;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Iteration
{
  public partial class MCTSManager : ObjectWithInstanceID, IDisposable
  {
    /// <summary>
    /// Callback called periodically during search to facilitate
    /// tracking of search progress.
    /// </summary>
    /// <param name="manager"></param>
    public delegate void MCTSProgressCallback(MCTSManager manager);

    /// <summary>
    /// Current status of search.
    /// </summary>
    public SearchStopStatus StopStatus = SearchStopStatus.Continue;

    /// <summary>
    /// Root of search tree.
    /// </summary>
    public MCTSNode Root => Context.Root;

    /// <summary>
    /// Ambient context of the current MCTS worker thread
    /// </summary>
    [ThreadStatic]
    public static MCTSIterator ThreadSearchContext;

    /// <summary>
    /// Statistic tracking total number of seconds spent
    /// in the operation of making a tree node the new root
    /// (used for tree reuse).
    /// </summary>
    public static float TotalTimeSecondsInMakeNewRoot = 0;

    /// <summary>
    /// Number of positions evaluated using a secondary evaluator.
    /// </summary>
    public static int NumSecondaryEvaluations = 0;

    /// <summary>
    /// Associated searh context.
    /// </summary>
    public MCTSIterator Context { get; private set; }


    /// <summary>
    /// Futility pruning manager associated with this search
    /// (for determing if and when top-level moves should be not further searched).
    /// </summary>
    public readonly MCTSFutilityPruning TerminationManager;

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
    /// (after any preparatory steps such as tree reuse preparation).
    /// </summary>
    public DateTime StartTimeFirstVisit;

    public SearchLimit SearchLimit;
    public bool ExternalStopRequested;

    public readonly List<GameMoveStat> PriorMoveStats;

    /// <summary>
    /// The N of the root node when search started
    /// (possibly nonzero due to tree reuse)
    /// </summary>
    public int RootNWhenSearchStarted;

    float estimatedNPS = float.NaN;


    /// <summary>
    /// The Node currently having the largest N value.
    /// </summary>
    public MCTSNode TopNNode;


    /// <summary>
    /// Number of MCTS steps actually taken so far in this search (not including initial tree).
    /// </summary>
    public int NumStepsTakenThisSearch => Root.N - RootNWhenSearchStarted;


    /// <summary>
    /// The N value when the current best node came to be best.
    /// </summary>
    public int NumNodesWhenChoseTopNNode;
    
    public float FractionNumNodesWhenChoseTopNNode => (float)NumNodesWhenChoseTopNNode / Root.N;

    internal MCTSSearchFlow flow; // TODO: make private

    public readonly bool IsFirstMoveOfGame;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="reuseOtherContextForEvaluatedNodes"></param>
    /// <param name="reusePositionCache"></param>
    /// <param name="reuseTranspositionRoots"></param>
    /// <param name="nnEvaluators"></param>
    /// <param name="searchParams"></param>
    /// <param name="childSelectParams"></param>
    /// <param name="searchLimit"></param>
    /// <param name="paramsSearchExecutionPostprocessor"></param>
    /// <param name="limitManager"></param>
    /// <param name="startTime"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="isFirstMoveOfGame"></param>
    public MCTSManager(MCTSNodeStore store,
                       MCTSIterator reuseOtherContextForEvaluatedNodes,
                       PositionEvalCache reusePositionCache,
                       TranspositionRootsDict reuseTranspositionRoots,
                       NNEvaluatorSet nnEvaluators,
                       ParamsSearch searchParams,
                       ParamsSelect childSelectParams,
                       SearchLimit searchLimit,
                       IManagerGameLimit limitManager,
                       DateTime startTime,
                       List<GameMoveStat> gameMoveHistory,
                       bool isFirstMoveOfGame)
    {
      if (searchLimit.IsPerGameLimit)
      {
        throw new Exception("Per game search limits not supported");
      }

      StartTimeThisSearch = startTime;
      RootNWhenSearchStarted = store.Nodes.nodes[store.RootIndex.Index].N;

      IsFirstMoveOfGame = isFirstMoveOfGame;
      SearchLimit = searchLimit;

      // Make our own copy of move history.
      PriorMoveStats = new List<GameMoveStat>();
      if (gameMoveHistory != null)
      {
        PriorMoveStats.AddRange(gameMoveHistory);
      }

      // Possibly autoselect new optimal parameters
      ParamsSearchExecutionChooser paramsChooser = new ParamsSearchExecutionChooser(nnEvaluators.EvaluatorDef,
                                                                                    searchParams, childSelectParams, searchLimit);

      // TODO: technically this is overwriting the params belonging to the prior search, that's ugly (but won't actually cause a problem)
      paramsChooser.ChooseOptimal(searchLimit.EstNumNodes(50_000, false)); // TODO: make 50_000 smarter


      int estNumNodes = EstimatedNumSearchNodesForEvaluator(searchLimit, nnEvaluators);

      // Adjust the nodes estimate if we are continuing an existing search
      if (searchLimit.Type == SearchLimitType.NodesPerMove && RootNWhenSearchStarted > 0)
      {
        estNumNodes = Math.Max(0, estNumNodes - RootNWhenSearchStarted);
      }
      Context = new MCTSIterator(store, reuseOtherContextForEvaluatedNodes, reusePositionCache, reuseTranspositionRoots, 
                                  nnEvaluators, searchParams, childSelectParams, searchLimit, estNumNodes);
      ThreadSearchContext = Context;

      TerminationManager = new MCTSFutilityPruning(this, searchLimit.SearchMoves);
      LimitManager = limitManager;

      CeresEnvironment.LogInfo("MCTS", "Init", $"SearchManager created for store {store}", InstanceID);
    }



    /// <summary>
    /// Saves cache file of full search state to disk 
    /// in file with specified name.
    /// </summary>
    /// <param name="cacheFileName"></param>
    public void SaveCache(string cacheFileName)
    {
      Context.Tree.PositionCache.SaveToDisk(cacheFileName);

    }


    /// <summary>
    /// Launches the search with specified limit.
    /// </summary>
    /// <param name="searchLimit"></param>
    /// <param name="progressCallback"></param>
    /// <param name="nnRemoteEvaluatorExtraSuffix"></param>
    /// <returns></returns>
    internal (TimingStats, MCTSNode) DoSearch(SearchLimit searchLimit, MCTSProgressCallback progressCallback)
    {
      SearchLimit = searchLimit;

      CheckMemoryExhaustion();

      ThreadSearchContext = this.Context;

      Context.ProgressCallback = progressCallback;

      TimingStats stats = new TimingStats();

      using (new TimingBlock($"MCTS SEARCH {searchLimit}", stats, TimingBlock.LoggingType.None))
      {
        flow = new MCTSSearchFlow(this, Context);

        int batchNum = 0;

        int hardLimitNumNodes = -1;
        bool shouldProcess = true;
        if (searchLimit.Type == SearchLimitType.NodesPerMove)
        {
          if (Root.N >= searchLimit.Value)
          {
            shouldProcess = false;
            StopStatus = SearchStopStatus.SearchLimitExceeded;
          }
          else
          {
            hardLimitNumNodes = (int)searchLimit.Value - Root.N;
          }
        }

        StartTimeFirstVisit = DateTime.Now;
        if (shouldProcess)
        {
          flow.ProcessDirectOverlapped(this, hardLimitNumNodes, batchNum, null);
        }

        batchNum++;
      }

      // Make sure nothing was left in flight after the search
      if ((Root.NInFlight != 0 || Root.NInFlight2 != 0) && !haveWarned)
      {
        Console.WriteLine($"Test={Root.Context.ParamsSearch.TestFlag} Internal error: search ended with N={Root.N} NInFlight={Root.NInFlight} NInFlight2={Root.NInFlight2} {Root}");
        haveWarned = true;
      }

      // Possibly validate tree integrity.
      if (MCTSDiagnostics.VerifyTreeIntegrityAtSearchEnd)
      {
        using (new SearchContextExecutionBlock(Context))
        {
          Context.Tree.Store.Validate();
        }
      }

      return (stats, Root);
    }

  
    bool haveWarned = false;

    /// <summary>
    /// Runs the secondary network evaluator for all nodes in the tree
    /// having a specified minimum number of visits
    /// (and stores result in EvalReslutSecondaryResult).
    /// </summary>
    /// <param name="minN"></param>
    /// <param name="evaluatorSecondary"></param>
    public void RunSecondaryNetEvaluations(int minN, MCTSNNEvaluator evaluatorSecondary)
    {
      List<MCTSNode> nodes = new List<MCTSNode>();

      float accAbsDiff = 0;
      int numNodes = 0;

      const int BATCH_SIZE = 256;

      Root.Ref.TraverseSequential(Context.Tree.Store, (ref MCTSNodeStruct nodeRef, MCTSNodeStructIndex index) =>
      {
        if (nodeRef.N >= minN 
         && nodeRef.Terminal == GameResult.Unknown
         && !nodeRef.IsTranspositionLinked
         && nodeRef.Uncertainty == 0 // ** TODO: fix this
         /*&& FP16.IsNaN(nodeRef.VSecondary)*/)// )
        {
          MCTSNode node = Context.Tree.GetNode(index);

          node.EvalResultSecondary = default;
          nodes.Add(node);

          nodeRef.Uncertainty = 1; // TODO: FIX THIS

          numNodes++;

          if (nodes.Count == BATCH_SIZE)
          {
            EvaluateSecondaryNodes(evaluatorSecondary, nodes, ref accAbsDiff);
            nodes.Clear();
          }
        }

        return true;
      });

      // Process any final nodes
      EvaluateSecondaryNodes(evaluatorSecondary, nodes, ref accAbsDiff);

      //Console.WriteLine($"V difference {numNodes} {accAbsDiff / numNodes:F2} from {Root.N}");
    }


    private void EvaluateSecondaryNodes(MCTSNNEvaluator evaluatorSecondary, List<MCTSNode> nodes, ref float accAbsDiff)
    {
      if (nodes.Count > 0)
      {
        ListBounded<MCTSNode> thisBatch = new (nodes.ToArray(), ListBounded<MCTSNode>.CopyMode.ReferencePassedMembers);
        evaluatorSecondary.Evaluate(Context, thisBatch);

        ParamsSearchSecondaryEvaluator secondaryParams = Context.ParamsSearch.ParamsSecondaryEvaluator;

        foreach (MCTSNode node in nodes)
        {
          ref MCTSNodeStruct nodeRef = ref node.Ref;

//          nodeRef.VSecondary = FP16.NaN;
          //          nodeRef.VSecondary = (FP16)node.EvalResultSecondary.V;

          if (node.Terminal == GameResult.Unknown)
          {
            ref readonly CompressedPolicyVector otherPolicy = ref node.EvalResultSecondary.PolicyRef;

            node.BlendPolicy(in otherPolicy, secondaryParams.UpdatePolicyFraction);

            if (secondaryParams.UpdateValueFraction > 0)
            {
              float diff = node.EvalResultSecondary.V - nodeRef.V;
              nodeRef.BackupApplyWDeltaOnly(secondaryParams.UpdateValueFraction * diff); // TODO: MAYBE USE 2.0 multiplier (?)

              accAbsDiff += Math.Abs(diff);
            }
          }
        }

        NumSecondaryEvaluations += thisBatch.Count;
      }
    }


    /// <summary>
    /// Resets the state of the tree back to all nodes
    /// having been univisited (but still present in tree).
    /// </summary>
    /// <param name="materializeTranspositions"></param>
    internal void ResetTreeState(bool materializeTranspositions)
    {
      int numBackedOut = 0;
      if (materializeTranspositions) Root.MaterializeAllTranspositionLinks(); // TODO: can we avoid having to do this?          

      Root.Ref.TraverseSequential(Context.Tree.Store, (ref MCTSNodeStruct node, MCTSNodeStructIndex index) =>
      {
        // TODO: could we improve eficiency by only doing this for the current generation (ReuseGenerationNum)?

        node.ResetExpandedState();
        numBackedOut++;
        return true;
      });
    }


    static DateTime lastMemoryExhaustionCheck = DateTime.Now;

    private static void CheckMemoryExhaustion()
    {
      // For efficiency reasons only check periodically (PagedMemorySize call is expensive)
      if ((DateTime.Now - lastMemoryExhaustionCheck).TotalSeconds > 30)
      {
#if NOT_USED
        // TODO: make this dependent on RAM available
        const long MAX_MEMORY_USAGE = 55_000_000_000;
        if (Process.GetCurrentProcess().PagedMemorySize > MAX_MEMORY_USAGE)
        {
          Console.WriteLine("Stopping execution until <CR> due to memory consumption of {Process.GetCurrentProcess().PagedMemorySize}");
          Console.ReadLine();
        }
#endif

        lastMemoryExhaustionCheck = DateTime.Now;
      }
    }

    public MGMove TablebaseImmediateBestMove;

    /// <summary>
    /// Consults the tablebase to determine if immediate
    /// best move can be directly determined.
    /// </summary>
    void TrySetImmediateBestMove()
    {
      // If using tablebases, lookup to see if we have immediate win (choosing the shortest one)
      if (Context.CheckTablebaseBestNextMove != null)
      {
        GameResult result;
        Position pos = Context.Tree.Store.Nodes.PriorMoves.FinalPosition;
        TablebaseImmediateBestMove = Context.CheckTablebaseBestNextMove(pos, out result);

        if (result == GameResult.Checkmate)
        {
          Debug.Assert(pos.ToMGPosition.IsLegalMove(TablebaseImmediateBestMove));

          // Set the evaluation of the position to be a win
          // TODO: possibly use distance to mate to set the distance more accurately than fixed at 1
          const int DISTANCE_TO_MATE = 1;

          float winP = ParamsSelect.WinPForProvenWin(DISTANCE_TO_MATE);

          Context.Root.Ref.W = winP;
          Context.Root.Ref.N = 1;
          Context.Root.Ref.WinP = (FP16)winP;
          Context.Root.Ref.LossP = 0;
          Context.Root.Ref.MPosition = DISTANCE_TO_MATE;
          Context.Root.EvalResult = new LeafEvaluationResult(GameResult.Checkmate, (FP16)winP, 0, DISTANCE_TO_MATE);
          Context.Root.Ref.Terminal = GameResult.Checkmate;
        }
        else if (result == GameResult.Draw)
        {
          // Set the evaluation of the position to be a draw
          // TODO: possibly use distance to end of game to set the distance more accurately than fixed at 1
          // TODO: do we have to adjust for possible contempt?
          const int DISTANCE_TO_END_OF_GAME = 1;

          Context.Root.Ref.W = 0;
          Context.Root.Ref.N = 1;
          Context.Root.Ref.WinP = 0;
          Context.Root.Ref.LossP = 0;
          Context.Root.Ref.MPosition = DISTANCE_TO_END_OF_GAME;
          Context.Root.Ref.Terminal = GameResult.Draw;
          Context.Root.EvalResult = new LeafEvaluationResult(GameResult.Draw, 0, 0, 1);
        }
        else if (result == GameResult.Unknown && TablebaseImmediateBestMove != default)
        {
          // N.B. Unknown as a result actually means "Lost"
          //      base currently the GameResult enum has no way to represent that.
          //      TODO: Clean this up, create a new Enum to represent this more cleanly.
          // Set the evaluation of the position to be a loss.
          // TODO: possibly use distance to mate to set the distance more accurately than fixed at 1
          const int DISTANCE_TO_MATE = 1;

          float lossP = ParamsSelect.LossPForProvenLoss(DISTANCE_TO_MATE);

          Context.Root.Ref.W = -lossP;
          Context.Root.Ref.N = 1;
          Context.Root.Ref.WinP = 0;
          Context.Root.Ref.LossP = (FP16)lossP;
          Context.Root.Ref.MPosition = DISTANCE_TO_MATE;
          Context.Root.EvalResult = new LeafEvaluationResult(GameResult.Checkmate, 0, (FP16)lossP, DISTANCE_TO_MATE);
          Context.Root.Ref.Terminal = GameResult.Checkmate;
        }

      }
    }

    readonly static object dumpToConsoleLock = new ();


    public static (MGMove, TimingStats)
    Search(MCTSManager manager, bool verbose,
           MCTSProgressCallback progressCallback = null,
           bool possiblyUsePositionCache = false)
    {
      MCTSearch.SearchCount++;
      manager.StartTimeThisSearch = DateTime.Now;
      manager.RootNWhenSearchStarted = manager.Root.N;

      MCTSIterator context = manager.Context;
      PositionWithHistory priorMoves = context.Tree.Store.Nodes.PriorMoves;

      // Make sure not already checkmate/stalemate
      GameResult terminalStatus = priorMoves.FinalPosition.CalcTerminalStatus();
      if (terminalStatus != GameResult.Unknown)
      {
        throw new Exception($"The initial position is terminal: {terminalStatus} {priorMoves.FinalPosition.FEN}");
      }

      // Possibly initialize cache
      if (possiblyUsePositionCache && context.EvaluatorDef.CacheMode == PositionEvalCache.CacheMode.MemoryAndDisk)
      {
        if (context.EvaluatorDef.PreloadedCache != null)
        {
          if (context.EvaluatorDef.CacheFileName != null) throw new Exception("Cannot specify both CacheFileName and PreloadedCache");
          manager.Context.Tree.PositionCache = context.EvaluatorDef.PreloadedCache;
        }
        else
          manager.Context.Tree.PositionCache.LoadFromDisk(context.EvaluatorDef.CacheFileName);
      }

      manager.TrySetImmediateBestMove();
      if (manager.TablebaseImmediateBestMove != default(MGMove))
      {
        manager.StopStatus = SearchStopStatus.TablebaseImmediateMove;
        return (manager.TablebaseImmediateBestMove, new TimingStats());
      }

      MCTSNode root = manager.Root;

      // Do the search
      IteratedMCTSDef schedule = manager.Context.ParamsSearch.IMCTSSchedule;
      bool useIMCTS = schedule != null & manager.SearchLimit.EstNumNodes(30_000, false) > 100;

      TimingStats stats;
      MCTSNode selectedMove;
      BestMoveInfo bestMoveInfo;

      SearchLimit thisSearchLimit = manager.SearchLimit;
      int numSearches = 0;
      BestMoveInfo firstTryBestMoveInfo = null;
      SearchLimit startingSearchLimit = manager.SearchLimit;
      bool shouldExtendSearch;
      do
      {
        shouldExtendSearch = false;

        (stats, selectedMove) = useIMCTS ? new IteratedMCTSSearchManager().IteratedSearch(manager, progressCallback, schedule)
                                         : manager.DoSearch(thisSearchLimit, progressCallback);

        // Get best child 
        // TODO: technically the "updateStats" should only be true if we end up acccepting this move
        bestMoveInfo = root.BestMoveInfo(true);

        if (numSearches == 0)
        {
          firstTryBestMoveInfo = bestMoveInfo;
        }
        else
        {
          //Console.WriteLine("after retry move " + bestMoveInfo.BestMove + " N now " + root.N + " Retry, Q now " + bestMoveInfo.QOfBest + " " + bestMoveInfo.BestMove + " on search" + thisSearchLimit);
          if (firstTryBestMoveInfo.BestMove != bestMoveInfo.BestMove)
          {
            //Environment.MCTSEventSource.TestMetric1++;
            //Console.WriteLine("************* Changed");
          }
        }

        // If the chosen move is far away from the best Q node, 
        // try to extend the search unless the position is obviously won/lost.
        const int MAX_RETRIES = 4;
        const float Q_THRESHOLD = 0.01f;
        const float INCREMENT_FRACTION = 0.20f;

        float fractionExtendedSoFar = 0;
        if (bestMoveInfo.BestMoveQSuboptimality > Q_THRESHOLD  // don't retry if Q is already best or nearly so
         && Math.Abs(root.Q) < 0.75f                           // don't retry if position is already won/lost
         && numSearches < MAX_RETRIES                          // don't retry many times to avoid using too much extra time
         && manager.NumStepsTakenThisSearch > 100              // don't retry for very small searches to because batch sizing make this imprecise
         && fractionExtendedSoFar <
            startingSearchLimit.FractionExtensibleIfNeeded)    // only extend if we are limit constrained only at game level, not strictly each move
        {

          int incrementalNodes = (int)(manager.NumStepsTakenThisSearch * INCREMENT_FRACTION);
          thisSearchLimit = SearchLimit.NodesPerMove(root.N + incrementalNodes);
          fractionExtendedSoFar += INCREMENT_FRACTION;

          //Console.WriteLine(" try " + numSearches + " Was " + bestMoveInfo.QOfBest + " " + bestMoveInfo.BestMove + "  Extending to " + thisSearchLimit + " because QSuboptimality " 
          //                + bestMoveInfo.BestMoveQSuboptimality  + " original limit " + manager.SearchLimit);
          shouldExtendSearch = true;
          manager.StopStatus = SearchStopStatus.Continue;
        }
        else
        {
          shouldExtendSearch = false;
          manager.UpdateTopNodeInfo();
        }

        numSearches++;
      } while (shouldExtendSearch);

      if (verbose)
      {
        DumpMoveStatistics(root);
      }

      if (possiblyUsePositionCache && manager.Context.EvaluatorDef.CacheMode == PositionEvalCache.CacheMode.MemoryAndDisk)
      {
        manager.SaveCache(manager.Context.EvaluatorDef.CacheFileName);
      }

      return (bestMoveInfo.BestMove, stats);
    }


    /// <summary>
    /// Dumps move statistics from the root node.
    /// </summary>
    /// <param name="maxMoves"></param>
    public void DumpRootMoveStatistics(int maxMoves = int.MaxValue)
    {
      DumpMoveStatistics(Root, maxMoves);
    }


    /// <summary>
    /// Dumps summary of search statistics associated with moves possible from a specified node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="maxMoves"></param>
    private static void DumpMoveStatistics(MCTSNode node, int maxMoves = int.MaxValue)
    {
      lock (dumpToConsoleLock)
      {
        // First quick dump at level 1
        Console.WriteLine();
        Console.WriteLine("VERBOSE ROOT MOVE STATISTICS");
        node.Dump(1, 1, maxMoves: maxMoves);
        Console.WriteLine("-----------------------------------------------------------------------------------------------------------------\r\n");
      }
    }


    /// <summary>
    /// Sets the level of Dirichlet noise if this feature is enabled.
    /// </summary>
    internal void PossiblySetSearchNoise()
    {
      if (Context.ParamsSearch.SearchNoisePolicy != null)
      {
        SearchNoisePolicyDef noiseDef = Context.ParamsSearch.SearchNoisePolicy;
        if (!float.IsNaN(noiseDef.DirichletAlpha))
        {
          Context.SetDirichletExplorationNoise(noiseDef.DirichletAlpha, noiseDef.DirichletFraction);
        }
      }
    }

#region IDisposable Support

    public MGMove BestMoveMG
    {
      get
      {
        if (TablebaseImmediateBestMove != default(MGMove))
          return TablebaseImmediateBestMove;

        using (new SearchContextExecutionBlock(Context))
        {
          return Context.Root.BestMove(false).Annotation.PriorMoveMG;
        }
      }
    }


    bool disposed = false;

    public void Dispose()
    {
      if (!disposed)
      {
        //      MCTSPosTreeNodeDumper.DumpAllNodes(Context, ref Context.Store.RootNode);
        Context.Tree.Store.Dispose();

        // Release references to objects
        ThreadSearchContext = null;
        Context = null;

        // If the search was sufficeintly large, trigger an aynchronous full garbage collection
        // We do this now for two reasons:
        //   - we have just released references to potentially large objects, and
        //   - possibly this search will result in a move being played followed by waiting time
        //     during which we can do the GC "for free"
        //int finalN = Context.Root.N;      
        //    const int THRESHOLD_N_TRIGGER_GC = 2_000;
        //    if (finalN >= THRESHOLD_N_TRIGGER_GC)
        //      ThreadPool.QueueUserWorkItem((obj) => System.GC.Collect(1, GCCollectionMode.Optimized));
        disposed = true;
      }
    }

#endregion

#region Time management

    // TODO: make this smarter (aware of hardware and NN)
    public int EstimatedNumSearchNodes => EstimatedNumSearchNodesForEvaluator(SearchLimit, Context.NNEvaluators);

    public static int EstimatedNumSearchNodesForEvaluator(SearchLimit searchLimit, NNEvaluatorSet nnEvaluators)
    {
      if (searchLimit.Type == SearchLimitType.NodesPerMove)
        return (int)searchLimit.Value;
      else if (searchLimit.Type == SearchLimitType.SecondsPerMove)
      {
        // TODO: someday look at the particular network and hardware to make this smarter.
        const int EST_AVG_NPS = 20_000;
        return (int)(searchLimit.Value * EST_AVG_NPS);
      }
      else if (searchLimit.Type == SearchLimitType.SecondsForAllMoves)
      {
        // As a crude approximation, assume 1/20 of time spent on each move
        float estimatedSecondsPerMove = searchLimit.Value / 20.0f;
        return EstimatedNumSearchNodesForEvaluator(new SearchLimit(SearchLimitType.SecondsPerMove, estimatedSecondsPerMove), nnEvaluators);
      }
      else if (searchLimit.Type == SearchLimitType.NodesForAllMoves)
      {
        // As a crude approximation, assume 1/20 of nodes spent on each move
        return (int)(searchLimit.Value / 20.0f);
      }
      else
        throw new NotImplementedException();
    }

    public int MaxBatchSizeDueToPossibleNearTimeExhaustion
    {
      get
      {
        if (SearchLimit.Type != SearchLimitType.SecondsPerMove) return int.MaxValue;

        float elapsedTime = (float)(DateTime.Now - StartTimeThisSearch).TotalSeconds;
        float remainingTime = SearchLimit.Value - elapsedTime;

        // TODO: tune these based on hardware and network (EstimatedNPS)
        if (remainingTime < 0.03)
          return 192;
        else if (remainingTime < 0.05)
          return 384;
        else
          return int.MaxValue;
      }
    }

    public float FractionSearchCompleted => 1.0f - FractionSearchRemaining;

    public float FractionSearchRemaining
    {
      get
      {
        return SearchLimit.Type switch
        {
          SearchLimitType.SecondsPerMove => MathHelpers.Bounded(RemainingTime / SearchLimit.Value, 0, 1),
          SearchLimitType.NodesPerMove => MathHelpers.Bounded((SearchLimit.Value - Root.N) / SearchLimit.Value, 0, 1),
          _ => throw new NotImplementedException()
        };
      }
    }


    public float RemainingTime
    {
      get
      {
        if (SearchLimit.Type != SearchLimitType.SecondsPerMove) return float.MaxValue;

        float elapsedTime = (float)(DateTime.Now - StartTimeThisSearch).TotalSeconds;
        float remainingTime = SearchLimit.Value - elapsedTime;
        return remainingTime;
      }
    }


    public void UpdateSearchStopStatus()
    {
      StopStatus = CalcSearchStopStatus();
    }

    SearchStopStatus CalcSearchStopStatus()
    {
      if (Root.N < 2)
      {
        return SearchStopStatus.Continue; // Have to search at least two nodes to successfully get a move
      }
      else if (Root.NumPolicyMoves <= 1)
      {
        return SearchStopStatus.OnlyOneLegalMove;
      }

      UpdateTopNodeInfo();

      if (ExternalStopRequested)
      {
        return SearchStopStatus.ExternalStopRequested;
      }

      if (RemainingTime <= 0.01)
      {
        return SearchStopStatus.TimeExpired;
      }

      if (SearchLimit.MaxTreeNodes != null 
       && Root.N >= SearchLimit.MaxTreeNodes
       && NumStepsTakenThisSearch > 0 // always allow a little search to insure state fully initialized
         )
      {
        return SearchStopStatus.MaxTreeNodesExceeded;
      }

      int numNotShutdowChildren = TerminationManager.NumberOfNotShutdownChildren();

      // Exit if only one possible move, and smart pruning is turned on
      if (Context.ParamsSearch.FutilityPruningStopSearchEnabled)
      {
        if (Root.N > 0 && Root.NumPolicyMoves == 1)
        {
          return SearchStopStatus.FutilityPrunedAllMoves;
        }
        else if (numNotShutdowChildren == 1)
        {
          return SearchStopStatus.FutilityPrunedAllMoves;
        }
      }

      return SearchStopStatus.Continue;
    }


    internal float EstimatedNPS => estimatedNPS;

    internal void UpdateEstimatedNPS()
    {
      const float MIN_TIME = 0.02f;
      const float MIN_VISITS = 10;

      float elapsedSecs = (float)(DateTime.Now -  StartTimeFirstVisit).TotalSeconds;
      bool insufficientData = elapsedSecs < MIN_TIME || NumStepsTakenThisSearch < MIN_VISITS;
      estimatedNPS = insufficientData ? float.NaN : NumStepsTakenThisSearch / elapsedSecs;
    }

    public int? EstimatedNumVisitsRemaining()
    {
      if (SearchLimit.Type == SearchLimitType.NodesPerMove)
      {
        return (int)(SearchLimit.Value - Root.N);
      }
      else if (SearchLimit.Type == SearchLimitType.SecondsPerMove)
      {
        float estNPS = EstimatedNPS;
        if (float.IsNaN(estNPS))
        {
          return null; // unkown
        }

        float elapsedTime = (float)(DateTime.Now - StartTimeThisSearch).TotalSeconds;
        float remainingTime = SearchLimit.Value - elapsedTime;

        return (int)MathF.Max(0, remainingTime * estNPS);
      }
      else
        throw new NotImplementedException();
    }


    public void UpdateTopNodeInfo()
    {
      MCTSNode newBestNode = Root.ChildWithLargestN;
      if (TopNNode == null || (newBestNode != TopNNode))
      {
        TopNNode = newBestNode;
        NumNodesWhenChoseTopNNode = Root.N;
      }
    }


    public int CountTablebaseHits
    {
      get
      {
        LeafEvaluatorSyzygyLC0 syzygyEvaluator = (LeafEvaluatorSyzygyLC0)Context.LeafEvaluators.Find(eval => eval.GetType() == typeof(LeafEvaluatorSyzygyLC0));
        return syzygyEvaluator == null ? 0 : syzygyEvaluator.NumHits;
      }
    }

    public void DumpParams()
    {
      Console.WriteLine(ObjUtils.FieldValuesDumpString<SearchLimit>(SearchLimit, SearchLimit.NodesPerMove(1), false));
      //      writer.Write(ObjUtils.FieldValuesDumpString<NNEvaluatorDef>(Def.NNEvaluators1.EvaluatorDef, new ParamsNN(), differentOnly));
      Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSelect>(Context.ParamsSelect, new ParamsSelect(), false));
      Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearch>(Context.ParamsSearch, new ParamsSearch(), false));
      //DumpTimeManagerDifference(differentOnly, null, timeManager1);
      Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(Context.ParamsSearch.Execution, new ParamsSearchExecution(), false));

    }

#endregion


    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<MCTSManager start {Context.StartPosAndPriorMoves} Root {Root}>";
    }

  }
}
