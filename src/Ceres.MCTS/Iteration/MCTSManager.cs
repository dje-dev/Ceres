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

using Ceres.Base;
using Ceres.Base.DataTypes;
using Ceres.Base.Environment;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.LC0;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Search;
using Ceres.MCTS.Search.IteratedMCTS;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.NNFiles;
using Ceres.Chess.GameEngines;
using Ceres.Chess.EncodedPositions;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.Chess.Positions;
using System.Threading;
using Ceres.MCTS.Params;
using Ceres.Base.Benchmarking;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.Utils;
using System.IO;

#endregion

namespace Ceres.MCTS.Iteration
{
  public partial class MCTSManager : ObjectWithInstanceID, IDisposable
  {
    /// <summary>
    /// Reason a search was stopped.
    /// </summary>
    public enum SearchStopStatus
    {
      Continue,
      OnlyOneLegalMove,
      ExternalStopRequested,
      TimeExpired,
      FutilityPrunedAllMoves
    }


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

    public MCTSNode TopNNode;

    public int NumStepsTakenThisSearch => Root.N - RootNWhenSearchStarted;

    // Information relating to move with highest N (at what point it was selected)
    public int NumNodesWhenChoseTopNNode;
    public float FractionNumNodesWhenChoseTopNNode => (float)NumNodesWhenChoseTopNNode / Root.N;

    public ParamsSearchExecutionModifier ParamsSearchExecutionPostprocessor;

    internal MCTSSearchFlow flow; // TODO: make private

    public readonly bool IsFirstMoveOfGame;


    ManagerGameLimitInputs limitsManagerInputs;


    /// <summary>
    /// 
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
    /// <param name="priorManager"></param>
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
                       ParamsSearchExecutionModifier paramsSearchExecutionPostprocessor,
                       IManagerGameLimit limitManager,
                       DateTime startTime,
                       MCTSManager priorManager,
                       List<GameMoveStat> gameMoveHistory,
                       bool isFirstMoveOfGame)
    {
      StartTimeThisSearch = startTime;
      RootNWhenSearchStarted = store.Nodes.nodes[store.RootIndex.Index].N;
      ParamsSearchExecutionPostprocessor = paramsSearchExecutionPostprocessor;
      IsFirstMoveOfGame = isFirstMoveOfGame;

      PriorMoveStats = new List<GameMoveStat>();

      // Make our own copy of move history.
      if (gameMoveHistory != null)
      {
        PriorMoveStats.AddRange(gameMoveHistory);
      }

      // Possibly convert time limit per game into time for this move.
      if (searchLimit.IsPerGameLimit)
      {
        SearchLimitType type = searchLimit.Type == SearchLimitType.SecondsForAllMoves
                                                       ? SearchLimitType.SecondsPerMove
                                                       : SearchLimitType.NodesPerMove;
        float rootQ = priorManager == null ? float.NaN : (float)store.RootNode.Q;


        limitsManagerInputs = new(store.Nodes.PriorMoves.FinalPosition, 
                                searchParams, PriorMoveStats,
                                type, store.RootNode.N, rootQ, 
                                searchLimit.Value, searchLimit.ValueIncrement, 
                                float.NaN, float.NaN, 
                                maxMovesToGo:searchLimit.MaxMovesToGo,                                                  
                                isFirstMoveOfGame: isFirstMoveOfGame);

        ManagerGameLimitOutputs timeManagerOutputs = limitManager.ComputeMoveAllocation(limitsManagerInputs);
        SearchLimit = timeManagerOutputs.LimitTarget;
      }
      else
      {
        SearchLimit = searchLimit;
      }

      // Possibly autoselect new optimal parameters
      ParamsSearchExecutionChooser paramsChooser = new ParamsSearchExecutionChooser(nnEvaluators.EvaluatorDef,
                                                                                    searchParams, childSelectParams, searchLimit);

      // TODO: technically this is overwriting the params belonging to the prior search, that's ugly (but won't actually cause a problem)
      paramsChooser.ChooseOptimal(searchLimit.EstNumNodes(50_000, false), paramsSearchExecutionPostprocessor); // TODO: make 50_000 smarter


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
    internal (TimingStats, MCTSNode) DoSearch(SearchLimit searchLimit,
                                              MCTSProgressCallback progressCallback)
    {
      CheckMemoryExhaustion();

      ThreadSearchContext = this.Context;

      Context.ProgressCallback = progressCallback;

      TimingStats stats = new TimingStats();

      int numMCTSNodesProcessedTotal = 0;
      using (new TimingBlock($"MCTS SEARCH {searchLimit}", stats, TimingBlock.LoggingType.None))
      {
        flow = new MCTSSearchFlow(this, Context);

        int batchNum = 0;

        int hardLimitNumNodes = -1;
        bool shouldProcess = true;
        if (searchLimit.Type == SearchLimitType.NodesPerMove)
        {
          if (Root.N >= searchLimit.Value)
            shouldProcess = false;
          else
            hardLimitNumNodes = (int)searchLimit.Value - Root.N;
        }

        StartTimeFirstVisit = DateTime.Now;
        if (shouldProcess) flow.ProcessDirectOverlapped(this, hardLimitNumNodes, batchNum, null);

        batchNum++;
      }

      // Make sure nothing was left in flight after the search
      if ((Root.NInFlight != 0 || Root.NInFlight2 != 0) && !haveWarned)
      {
        Console.WriteLine($"Internal error: search ended with N={Root.N} NInFlight={Root.NInFlight} NInFlight2={Root.NInFlight2} {Root}");
        haveWarned = true;
      }
       
      return (stats, Root);
    }

  
    bool haveWarned = false;
#if NOTUSED
    private bool PossiblyEarlyStopMoveSecondary(int childIndex, in MCTSNodeStruct child, int numRemainingSteps, 
                                                MCTSNode bestMove, float qAdjustment)
    {
      if (Root.N > 500)
      {
        if (Context.ParamsSelect.CPUCT2 != 0) throw new NotImplementedException(); // need to add to NumVisitsToEqualize method

        int neededSteps = VisitsToEqualizeCalculator.NumVisitsToEqualize(
                 Context.ParamsSelect.UCTNonRootNumeratorExponent,
                 Context.ParamsSelect.UCTRootDenominatorExponent, 
                 Root.N, 
                 bestMove.P, (float)-bestMove.Q, bestMove.N,
                 child.P, (float)-child.Q, child.N,
                 StatUtils.Bounded(-(float)bestMove.Q - qAdjustment, -1, 1),
                 StatUtils.Bounded(-(float)child.Q + qAdjustment, -1, 1));

        return neededSteps > numRemainingSteps;
      }
      else
        return false;
    }

#endif


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
        if (nodeRef.N >= minN && FP16.IsNaN(nodeRef.VSecondary))// && !nodeRef.IsTranspositionLinked)
        {
          MCTSNode node = Context.Tree.GetNode(index);
          node.EvalResultSecondary = default;
          nodes.Add(node);

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

      Console.WriteLine($"V difference {numNodes} {accAbsDiff / numNodes:F2} from {Root.N}");
    }

    float[] policyBuffer = new float[EncodedPolicyVector.POLICY_VECTOR_LENGTH];

    private void EvaluateSecondaryNodes(MCTSNNEvaluator evaluatorSecondary, List<MCTSNode> nodes, ref float accAbsDiff)
    {
      if (nodes.Count > 0)
      {
        ListBounded<MCTSNode> thisBatch = new ListBounded<MCTSNode>(nodes.ToArray(), ListBounded<MCTSNode>.CopyMode.ReferencePassedMembers);
        evaluatorSecondary.Evaluate(Context, thisBatch);
        foreach (MCTSNode node in nodes)
        {
          ref MCTSNodeStruct nodeRef = ref node.Ref;
          nodeRef.VSecondary = (FP16)node.EvalResultSecondary.V;
          float diff = nodeRef.VSecondary - nodeRef.V;
          accAbsDiff += Math.Abs(diff);

          // TODO: tune this threshold
          if (node.Terminal == GameResult.Unknown)// && Math.Abs(diff) > 0.01)
          {
            nodeRef.BackupApplyWDeltaOnly(diff); // MAYBE USE 2.0 multiplier
#if NOT
            // Update policy (replace)
            // TODO: Remove this. Not working. 
            //       Very difficult/impossible to change policy when already partly expanded,
            //       particularly since chlidren are assumed sorted descending by policy.
            ref readonly ChessPolicyVectorCompressed policy = ref node.EvalResultSecondary.PolicyRef;
            policy.DoDecoded(false, policyBuffer);
            foreach (MCTSNodeStructChild child in nodeRef.Children)
            {
              float policyValue = policyBuffer[child.Move.IndexNeuralNet];
              if (policyValue != 0)
              {
                if (child.IsExpanded)
                  child.ChildRef.P = (FP16)policyValue;
                else
                  child.SetUnexpandedPolicyValues(child.Move, (FP16)policyValue);                
              }
            }

            if (node.NumChildrenVisited < node.NumPolicyMoves)
            {
              Span<MCTSNodeStructChild> children = node.Ref.Children;
              int numSwapped;
              do
              {
                numSwapped = 0;
                for (int i = node.NumChildrenVisited + 1; i < node.NumPolicyMoves; i++)
                {
                  if (!children[i].IsExpanded && !children[i-1].IsExpanded)
                  {
                    if (children[i].p < children[i - 1].p)
                    {
                      FP16 temp = children[i].p;
                      children[i].p = children[i - 1].p;
                      children[i - 1].p = temp;
                      numSwapped++;
                    }
                  }
                }
              }
              while (numSwapped > 0);

            }
#endif
          }
        }

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
        TablebaseImmediateBestMove = Context.CheckTablebaseBestNextMove(Context.Tree.Store.Nodes.PriorMoves.FinalPosition, out result);

        if (result == GameResult.Checkmate)
        {
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

      }
    }

    readonly static object dumpToConsoleLock = new ();


    public static (MGMove, TimingStats)
    Search(MCTSManager manager, bool verbose,
           MCTSProgressCallback progressCallback = null,
           bool possiblyUsePositionCache = false)
    {
      MCTSIterator context = manager.Context;
      PositionWithHistory priorMoves = context.Tree.Store.Nodes.PriorMoves;

      // Make sure not already checkmate/stalemate
      GameResult terminalStatus = priorMoves.FinalPosition.CalcTerminalStatus();
      if (terminalStatus != GameResult.Unknown)
        throw new Exception($"The initial position is terminal: {terminalStatus} {priorMoves.FinalPosition.FEN}");

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
        manager.StartTimeThisSearch = DateTime.Now;
        return (manager.TablebaseImmediateBestMove, new TimingStats());
      }
     
      // Do the search
      IteratedMCTSDef schedule = manager.Context.ParamsSearch.IMCTSSchedule;
      bool useIMCTS = schedule != null & manager.SearchLimit.EstNumNodes(30_000, false) > 100;
      (TimingStats stats, MCTSNode selectedMove) =
        useIMCTS ? new IteratedMCTSSearchManager().IteratedSearch(manager, progressCallback, schedule)
                 : manager.DoSearch(manager.SearchLimit, progressCallback);

      manager.UpdateTopNodeInfo();

      MCTSNode root = manager.Root;

      // Get best child 
      MCTSNode bestMoveNode = root.BestMove(true);
      if (bestMoveNode == null)
        throw new NotImplementedException("Cannot return best child, only zero or one nodes evaluated");

      if (verbose)
      {
        DumpMoveStatistics(root);
      }

      if (possiblyUsePositionCache && manager.Context.EvaluatorDef.CacheMode == PositionEvalCache.CacheMode.MemoryAndDisk) 
        manager.SaveCache(manager.Context.EvaluatorDef.CacheFileName);

      bestMoveNode.Annotate();
      return (bestMoveNode.Annotation.PriorMoveMG, stats);
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
        INNWeightsFileInfo netDef = NNWeightsFiles.LookupNetworkFile(nnEvaluators.EvaluatorDef.Nets[0].Net.NetworkID);
        bool isSmallNet = netDef.NumBlocks <= 20;

        if (searchLimit.Value <= 0.5)
          return isSmallNet ? 10_000 : 5_000;
        else
          return isSmallNet ? 40_000 : 20_000;
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
//      if (StopStatus != SearchStopStatus.Continue)
//        Console.WriteLine("STOP " + StopStatus);
    }

    SearchStopStatus CalcSearchStopStatus()
    {
      //if (Root == null) return true; // This seemed to happen?   8/8/1p3p1p/1P5k/5P1P/6K1/1P6/8 w - - 1 113 
      if (Root.N < 2) return SearchStopStatus.Continue; // Have to search at least two nodes to successfully get a move
      if (Root.NumPolicyMoves <= 1) return SearchStopStatus.OnlyOneLegalMove;

      UpdateTopNodeInfo();

      if (ExternalStopRequested) return SearchStopStatus.ExternalStopRequested;

//      if (SearchLimit == null) return false;

      if (RemainingTime <= 0.01) return SearchStopStatus.TimeExpired;

      int numNotShutdowChildren = TerminationManager.NumberOfNotShutdownChildren();

      // Exit if only one possible move, and smart pruning is turned on
      if (Context.ParamsSearch.FutilityPruningStopSearchEnabled)
      {
        if (Root.N > 0 && Root.NumPolicyMoves == 1) return SearchStopStatus.FutilityPrunedAllMoves;
        if (numNotShutdowChildren == 1) return SearchStopStatus.FutilityPrunedAllMoves;
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
        return (int)(SearchLimit.Value - Root.N);
      else if (SearchLimit.Type == SearchLimitType.SecondsPerMove)
      {
        float estNPS = EstimatedNPS;
        if (float.IsNaN(estNPS)) return null; // unkown

        float elapsedTime = (float)(DateTime.Now - StartTimeThisSearch).TotalSeconds;
        float remainingTime = SearchLimit.Value - elapsedTime;

        return (int)MathF.Max(0, remainingTime * estNPS);
      }
      else
        throw new NotImplementedException();
    }


    public void UpdateTopNodeInfo()
    {
      MCTSNode newBestNode = Root.NumChildrenExpanded == 0 ? null : Root.ChildrenSorted(n => -n.N)[0];
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
        LeafEvaluatorBase syzygyEvaluator = Context.LeafEvaluators.Find(eval => eval.GetType() == typeof(LeafEvaluatorSyzygyLC0));
        return syzygyEvaluator == null ? 0 : syzygyEvaluator.CountHits;
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
