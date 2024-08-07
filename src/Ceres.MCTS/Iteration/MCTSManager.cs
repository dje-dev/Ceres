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
using Ceres.MCTS.NodeCache;
using Ceres.MCTS.Environment;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using System.Linq;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.EncodedPositions.Basic;

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
    /// Parent search object.
    /// </summary>
    public MCTSearch Search;

    /// <summary>
    /// Current status of search.
    /// </summary>
    public SearchStopStatus StopStatus = SearchStopStatus.Continue;

    /// <summary>
    /// Root of search tree.
    /// </summary>
    public MCTSNode Root => Context.Root;

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
    /// Number of batches evaluated using a secondary evaluator.
    /// </summary>
    public static int NumSecondaryBatches = 0;

    /// <summary>
    /// Associated searh context.
    /// </summary>
    public MCTSIterator Context { get; private set; }


    /// <summary>
    /// Futility pruning manager associated with this search
    /// (for determing if and when top-level moves should be not further searched).
    /// </summary>
    public MCTSFutilityPruning TerminationManager;

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

    /// <summary>
    /// Search limit initially allocated.
    /// </summary>
    public SearchLimit SearchLimitInitial;

    /// <summary>
    /// Search limit used as of last set of iterations 
    /// (possibly multiple of search was extended).
    /// </summary>
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
    /// The index of the child node currently having the largest N value.
    /// </summary>
    public int? TopNChildIndex;
    public int TopNChildN;

    /// <summary>
    /// Number of MCTS steps actually taken so far in this search (not including initial tree).
    /// </summary>
    public int NumNodesVisitedThisSearch => Root.N - RootNWhenSearchStarted;


    /// <summary>
    /// The N value when the current best node came to be best.
    /// </summary>
    public int NumNodesWhenChoseTopNNode;

    public float FractionNumNodesWhenChoseTopNNode => (float)NumNodesWhenChoseTopNNode / Root.N;

    internal MCTSSearchFlow flow; // TODO: make private

    public readonly bool IsFirstMoveOfGame;

    /// <summary>
    /// If tablebase evaluations should not be marked as terminal
    /// (needed when root position is a win but no DTZ files available).
    /// </summary>
    public readonly bool ForceNoTablebaseTerminals;

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
    /// <param name="isFirstMoveOfGame"></param>s
    public MCTSManager(MCTSearch search,
                       MCTSNodeStore store,
                       MCTSIterator reuseOtherContextForEvaluatedNodes,
                       PositionEvalCache reusePositionCache,
                       IMCTSNodeCache reuseNodeCache,
                       TranspositionRootsDict reuseTranspositionRoots,
                       NNEvaluatorSet nnEvaluators,
                       ParamsSearch searchParams,
                       ParamsSelect childSelectParams,
                       SearchLimit searchLimit,
                       IManagerGameLimit limitManager,
                       DateTime startTime,
                       List<GameMoveStat> gameMoveHistory,
                       bool isFirstMoveOfGame,
                       bool forceNoTablebaseTerminals,
                       List<MGMove> searchMovesTablebaseRestricted)
    {
      if (searchLimit.IsPerGameLimit)
      {
        throw new Exception("Per game search limits not supported");
      }

      Search = search;
      StartTimeThisSearch = startTime;
      RootNWhenSearchStarted = store.RootNode.N;

      IsFirstMoveOfGame = isFirstMoveOfGame;
      ForceNoTablebaseTerminals = forceNoTablebaseTerminals;
      SearchLimit = searchLimit;
      SearchLimitInitial = searchLimit;

      // Make our own copy of move history.
      PriorMoveStats = new List<GameMoveStat>();
      if (gameMoveHistory != null)
      {
        PriorMoveStats.AddRange(gameMoveHistory);
      }

      // Possibly autoselect new optimal parameters
      ParamsSearchExecutionChooser paramsChooser = new ParamsSearchExecutionChooser(nnEvaluators.EvaluatorDef,
                                                                                    searchParams, childSelectParams, searchLimit);

     
      float estNPS = ManagerGameLimitInputs.TrailingAvg(PriorMoveStats, 5, store.Nodes.PriorMoves.FinalPosition.MiscInfo.SideToMove, 
                                                        m => m.NumNodesComputed > 10, m => m.NodesPerSecond);
      if (float.IsNaN(estNPS))
      {
        estNPS = SearchLimit.DEFAULT_NPS; 
      }

      int estNumFinalTreeNodes = searchLimit.EstNumFinalNodes(RootNWhenSearchStarted, (int)estNPS, false);
      int? hardMaxNumFinalTreeNodes = searchLimit.HardMaxNumFinalNodes(RootNWhenSearchStarted, (int)estNPS, false);
      int estNumSearchNodes = searchLimit.EstNumSearchNodes(RootNWhenSearchStarted, (int)estNPS, false);

      // TODO: technically this is overwriting the params belonging to the prior search, that's ugly (but won't actually cause a problem)
      paramsChooser.ChooseOptimal(estNumFinalTreeNodes);

      // Adjust the nodes estimate if we are continuing an existing search
      if (searchLimit.Type == SearchLimitType.NodesPerMove && RootNWhenSearchStarted > 0)
      {
        estNumSearchNodes = Math.Max(0, estNumSearchNodes - RootNWhenSearchStarted);
      }

      // Possibly the node cache was created for a prior search with a smaller fixed max nodes.
      // Do not reuse it if undersized.
      if (reuseNodeCache != null && !reuseNodeCache.IsLargeEnough(hardMaxNumFinalTreeNodes))
      {
        reusePositionCache = null;
      }

      Context = new MCTSIterator(this, store, reuseOtherContextForEvaluatedNodes, reusePositionCache, reuseNodeCache, reuseTranspositionRoots,
                                 nnEvaluators, searchParams, childSelectParams, hardMaxNumFinalTreeNodes, estNumSearchNodes);

      TerminationManager = new MCTSFutilityPruning(this, searchLimit.SearchMoves, searchMovesTablebaseRestricted);
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

      Context.ProgressCallback = progressCallback;

      TimingStats stats = new TimingStats();

      using (new TimingBlock($"MCTS SEARCH {searchLimit}", stats, TimingBlock.LoggingType.None))
      {
        flow = new MCTSSearchFlow(this, Context);

        int batchNum = 0;

        int hardLimitNumNodesToCompute = -1;
        bool shouldProcess = true;
        if (searchLimit.Type == SearchLimitType.NodesPerMove)
        {
          hardLimitNumNodesToCompute = (int) searchLimit.Value;
        }
        else  if (searchLimit.Type == SearchLimitType.NodesPerTree)
        {
          if (Root.N >= searchLimit.Value)
          {
            shouldProcess = false;
            StopStatus = SearchStopStatus.NodeLimitReached;
          }
          else
          {
            hardLimitNumNodesToCompute = (int)searchLimit.Value - Root.N;
          }
        }

        StartTimeFirstVisit = DateTime.Now;
        if (shouldProcess)
        {
          flow.ProcessDirectOverlapped(this, hardLimitNumNodesToCompute, batchNum, null);
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
        Context.Tree.Store.Validate(Context.Tree.TranspositionRoots);
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
      float accAbsDiff = 0;
      EvaluateSecondaryNodes(evaluatorSecondary, Context.PendingSecondaryNodes.ToArray(), ref accAbsDiff);
      return;

#if NOT
      const int BATCH_SIZE = 384;
      List<MCTSNode> nodes = new List<MCTSNode>(BATCH_SIZE);

      
      int numNodes = 0;

//      using (new TimingBlock("Traverse " + Root.N))
      Root.StructRef.TraverseSequential(Context.Tree.Store, (ref MCTSNodeStruct nodeRef, MCTSNodeStructIndex index) =>
      {
        if (nodeRef.N >= minN
         && nodeRef.Terminal == GameResult.Unknown
         && !nodeRef.IsTranspositionLinked
         && !nodeRef.IsOldGeneration
         && !nodeRef.SecondaryNN
         /*&& FP16.IsNaN(nodeRef.VSecondary)*/)// )
        {
          MCTSNode node = Context.Tree.GetNode(index);

          node.EvalResult = default;
          nodes.Add(node);

          nodeRef.SecondaryNN = true;

          numNodes++;

          if (nodes.Count == BATCH_SIZE)
          {
            EvaluateSecondaryNodes(evaluatorSecondary, nodes.ToArray(), ref accAbsDiff);
            nodes.Clear();
          }
        }

        return true;
      });

      // Process any final nodes
      EvaluateSecondaryNodes(evaluatorSecondary, nodes.ToArray(), ref accAbsDiff);

      //Console.WriteLine($"V difference {numNodes} {accAbsDiff / numNodes:F2} from {Root.N}");
#endif

    }


    private void EvaluateSecondaryNodes(MCTSNNEvaluator evaluatorSecondary, MCTSNode[] nodes, ref float accAbsDiff)
    {
      if (nodes.Length > 0)
      {
        // Run the neural network evaluations.
        ListBounded<MCTSNode> thisBatch = new(nodes, ListBounded<MCTSNode>.CopyMode.ReferencePassedMembers);
        evaluatorSecondary.Evaluate(Context, thisBatch);

        ParamsSearchSecondaryEvaluator secondaryParams = Context.ParamsSearch.ParamsSecondaryEvaluator;
        //Console.WriteLine(this.Root.N + " middle batch " + nodes.Length);

        // Process each node, blending in policy and/or value.
        foreach (MCTSNode node in nodes)
        {
          if (node.Terminal == GameResult.Unknown)
          {
            ref MCTSNodeStruct nodeRef = ref node.StructRef;
            if (secondaryParams.UpdatePolicyFraction > 0)
            {
              ref readonly CompressedPolicyVector otherPolicy = ref node.EvalResult.PolicyRef;
              node.BlendPolicy(in otherPolicy, secondaryParams.UpdatePolicyFraction);
            }

            if (secondaryParams.UpdateValueFraction > 0)
            {
              float diff = node.EvalResult.V - nodeRef.V;
              nodeRef.BackupApplyWDeltaOnly(secondaryParams.UpdateValueFraction * diff); // TODO: MAYBE USE 2.0 multiplier (?)

              accAbsDiff += Math.Abs(diff);
            }
          }
        }

        NumSecondaryBatches++;
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

      Root.StructRef.TraverseSequential(Context.Tree.Store, (ref MCTSNodeStruct node, MCTSNodeStructIndex index) =>
      {
        // TODO: could we improve eficiency by only doing this for the current generation (ReuseGenerationNum)?

        node.ResetExpandedState(Context.Tree.Store.StoreID);
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


    /// <summary>
    /// Probes the tablebases (if any) for the best move at root,
    /// subject to verification that the possible move will not 
    /// trigger a draw by repetition given the present move history.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    internal (WDLResult result, MGMove immediateMove) TryGetTablebaseImmediateMove(MCTSNode node)
    {
      // Not possible to find if tablebase method is not installed (not available).
      if (Context.CheckTablebaseBestNextMove == null)
      {
        return (WDLResult.Unknown, default);
      }

      node.Annotate();
      Position pos = node.Annotation.Pos;
      MGMove immediateMove = Context.CheckTablebaseBestNextMove(in pos, out WDLResult result, out List<(MGMove, short)> fullWinningMoveList, out bool winningMoveListOrderedByDTM);
      if (result == WDLResult.Win && !winningMoveListOrderedByDTM)
      {
        // Not safe to rely upon the tablebase probe because unable (e.g. because of missing DTZ files)
        // to indicate which moves bring us closer to checkmate.
        // Instead return failure in lookup thus engine will do normal search.
        // However filter search moves to only include the winning moves.
        if (fullWinningMoveList != null && fullWinningMoveList.Count > 0)
        {
          List<MGMove> moveList = new List<MGMove>(fullWinningMoveList.Count);
          foreach ((MGMove, short) move in fullWinningMoveList)
          {
            moveList.Add(move.Item1);
          }
          TerminationManager.SearchMovesTablebaseRestricted = moveList;
        }

        return (WDLResult.Unknown, default);
      }

      if (result == WDLResult.Win)
      {
        Debug.Assert(pos.ToMGPosition.IsLegalMove(immediateMove));
        Span<Position> historyPositions = node.Context.Tree.HistoryPositionsForNode(node);

        // Try to avoid making a move which would allow opponent to claim draw.
        bool wouldBeDrawByRepetition = PositionRepetitionCalc.DrawByRepetitionWouldBeClaimable(in pos, immediateMove, historyPositions.ToArray());
        if (wouldBeDrawByRepetition)
        {
          if (fullWinningMoveList == null)
          {
            // Perhaps DTZ tablebase files not available.
            // Do not blindly play into the draw, return no tablbase hit
            // and allow engine to search normally as fallback.
            return (WDLResult.Unknown, default);
          }
          else
          {
            // Check other moves to see if any of them avoids falling into the draw by repetition trap.
            foreach ((MGMove, short) move in fullWinningMoveList)
            {
              if (!PositionRepetitionCalc.DrawByRepetitionWouldBeClaimable(in pos, move.Item1, historyPositions.ToArray()))
              {
                immediateMove = move.Item1;
                break;
              }
            }
          }
        }

      }

      return (result, immediateMove);
    }

    public MGMove TablebaseImmediateBestMove;

    /// <summary>
    /// Consults the tablebase to determine if immediate
    /// best move can be directly determined.
    /// </summary>
    void TrySetImmediateBestMove(MCTSNode node)
    {
      TablebaseImmediateBestMove = default;

      // If using tablebases, lookup to see if we have immediate win (choosing the shortest one)
      if (Context.CheckTablebaseBestNextMove != null)
      {
        (WDLResult result, MGMove immediateMove) = TryGetTablebaseImmediateMove(node);
        TablebaseImmediateBestMove = immediateMove;

        if (result == WDLResult.Win)
        {
          SetRootAsWin();
        }
        else if (result == WDLResult.Draw)
        {
          // Set the evaluation of the position to be a draw
          // TODO: possibly use distance to end of game to set the distance more accurately than fixed at 1
          // TODO: do we have to adjust for possible contempt?
          const int DISTANCE_TO_END_OF_GAME = 1;

          Context.Root.StructRef.W = 0;
          Context.Root.StructRef.N = 1;
          Context.Root.StructRef.WinP = 0;
          Context.Root.StructRef.LossP = 0;
          Context.Root.StructRef.MPosition = DISTANCE_TO_END_OF_GAME;
          Context.Root.StructRef.Terminal = GameResult.Draw;
          Context.Root.EvalResult = new LeafEvaluationResult(GameResult.Draw, 0, 0, 1, 0);
        }
        else if (result == WDLResult.Loss && TablebaseImmediateBestMove != default)
        {
          // Set the evaluation of the position to be a loss.
          // TODO: possibly use distance to mate to set the distance more accurately than fixed at 1
          const int DISTANCE_TO_MATE = 1;

          float lossP = ParamsSelect.LossPForProvenLoss(DISTANCE_TO_MATE, false);

          Context.Root.StructRef.W = -lossP;
          Context.Root.StructRef.N = 1;
          Context.Root.StructRef.WinP = 0;
          Context.Root.StructRef.LossP = (FP16)lossP;
          Context.Root.StructRef.MPosition = DISTANCE_TO_MATE;
          Context.Root.EvalResult = new LeafEvaluationResult(GameResult.Checkmate, 0, (FP16)lossP, DISTANCE_TO_MATE, 0);
          Context.Root.StructRef.Terminal = GameResult.Checkmate;
        }
      }
    }

    private void SetRootAsWin()
    {
      // Set the evaluation of the position to be a win
      // TODO: possibly use distance to mate to set the distance more accurately than fixed at 1
      const int DISTANCE_TO_MATE = 1;

      float winP = ParamsSelect.WinPForProvenWin(DISTANCE_TO_MATE, false);

      Context.Root.StructRef.W = winP;
      Context.Root.StructRef.N = 1;
      Context.Root.StructRef.WinP = (FP16)winP;
      Context.Root.StructRef.LossP = 0;
      Context.Root.StructRef.MPosition = DISTANCE_TO_MATE;
      Context.Root.EvalResult = new LeafEvaluationResult(GameResult.Checkmate, (FP16)winP, 0, DISTANCE_TO_MATE, 0);
      Context.Root.StructRef.Terminal = GameResult.Checkmate;
    }

    readonly static object dumpToConsoleLock = new();

    public float FractionExtendedSoFar = 0;

    static bool DUMP = false;

    public static (MGMove, float) BestValueMove(NNEvaluator nnEvaluator, 
                                                PositionWithHistory priorMoves, 
                                                ref MGMoveList moves,
                                                ref NNEvaluatorResult[] evalResults,
                                                bool fillInHistory,
                                                bool dumpInfo)
    {
      // Compute move list if not already provided.
      if (moves == null)
      {
        moves = new MGMoveList();
        MGMoveGen.GenerateMoves(priorMoves.FinalPosMG, moves);
      }

      // Prepare a batch builder in which to enqueue the positions to be evaluated.
      EncodedPositionBatchBuilder batchBuilder = new (128, nnEvaluator.InputsRequired | NNEvaluator.InputTypes.Positions);

      // Prepare array of prior positions initialized from prior positions,
      // with extra last slot to be for new position after each move to be evaluated.
      Position[] positions = new Position[priorMoves.Count + 1];
      Array.Copy(priorMoves.Positions.ToArray(), positions, priorMoves.Positions.Length);

      Span<short> repetitionCounts = stackalloc short[moves.NumMovesUsed];

      // Loop over all moves, find new resulting position, and add to batch.
      int indexOfDrawByRepetition = -1;
      for (int i=0;i<moves.NumMovesUsed;i++)
      {
        MGPosition pos = priorMoves.FinalPosition.ToMGPosition;
        pos.MakeMove(moves.MovesArray[i]);
        positions[^1] = pos.ToPosition;

        // Need to calc repetition count for position of this move in the context of all prior moves
        int finalRepetitionCount = PositionRepetitionCalc.SetFinalPositionRepetitionCount(positions);
        repetitionCounts[i] = (short)finalRepetitionCount;
        if (finalRepetitionCount >= 2)
        {
          indexOfDrawByRepetition = i;
        }

        EncodedPositionWithHistory eph = new EncodedPositionWithHistory();
        eph.SetFromSequentialPositions(positions, fillInHistory);

        batchBuilder.Add(in eph, false);
      }

      // Build the batch and evaluate it.
      EncodedPositionBatchFlat thisBatch = batchBuilder.GetBatch();
      evalResults = nnEvaluator.EvaluateBatch(thisBatch);

      float bestVRaw = evalResults.Min(v => -v.V);

      NNEvaluatorResult parentResult = dumpInfo ? nnEvaluator.Evaluate(priorMoves.FinalPosition) : default;

      // Determine which move had position yielding best value evaluation.
      int moveIndex = 0;
      int bestMoveIndex = 0;
      float bestV = bestVRaw;
      foreach (NNEvaluatorResult v in evalResults)
      {
        float vOurPerspective = -v.V;

        if (nnEvaluator.UseBestValueMoveUseRepetitionHeuristic)
        {
          if (repetitionCounts[moveIndex] == 1) // on the way to draw by repetition
          {
            if (bestVRaw >= 0) // probably winning
            {
              // Disfavor positions where repetition count is nonzero.
              vOurPerspective = vOurPerspective - 0.01f;
            }
            else if (bestVRaw <= 0) // probably losing
            {
              // Favor positions where repetition count is nonzero.
              vOurPerspective = vOurPerspective + 0.01f;
            }
          }
        }

        if (vOurPerspective > bestV)
        {
          bestV = vOurPerspective;
          bestMoveIndex = moveIndex;
        }
        moveIndex++;
      }

      if (nnEvaluator.UseBestValueMoveUseRepetitionHeuristic && indexOfDrawByRepetition != -1 && bestV < 0)
      {
        // If we appear worse, choose the draw by repetition at hand.
        bestV = 0;
        bestMoveIndex = indexOfDrawByRepetition;
      }

      if (dumpInfo)
      {
        Console.WriteLine($"BestValueMove detail using {nnEvaluator} from {priorMoves}");
        for (int i = 0; i < moves.Count(); i++)
        {
          EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(moves.MovesArray[i]);
          float policyPct = 0;
          if (parentResult.Policy.IndexOfMove(encodedMove) != -1)
          {
            policyPct = 100 * parentResult.Policy.PolicyInfoAtIndex(parentResult.Policy.IndexOfMove(encodedMove)).Probability;
          }
          bool isBest = i == bestMoveIndex;
          string warnStr = isBest && policyPct < 10 ? "?" : " ";
          Console.Write(isBest ? "** " : "  ");
          Console.WriteLine(warnStr + "  " +  moves.MovesArray[i] + " " + -evalResults[i].V
                      + "   " + policyPct + "%");
        }
      }

      // Return the best move.
      MGMove bestMove = moves.MovesArray[bestMoveIndex];
      return (bestMove, bestV);
    }


    public static MGMove DoSearch(MCTSManager manager, bool verbose,
                                  MCTSProgressCallback progressCallback = null,
                                  bool possiblyUsePositionCache = false,
                                  bool moveImmediateIfOnlyOneMove = false,
                                  MGMove forceMove = default)
    {
      #if NOT

      if (manager.Root.N > 0)
      {
        for(int i=0;i<manager.Root.NumChildrenExpanded;i++)
        {
          var child = manager.Root.ChildAtIndex(i);
          child.Annotate();
          if (child.Annotation.Pos.PieceCount > 7)
          {
            if (child.Terminal.IsTerminal())
            {
              var pmm = manager.Root.Tree.Store.Nodes.PriorMoves;

              int equalCount = 0;
              Console.WriteLine("ROOT " + child.Annotation.Pos.FEN);
              int index = 0;
              foreach (var move in pmm.PositionsWithMoves)
              {
                if (child.Annotation.Pos.EqualAsRepetition(move.Position))
                {
                  Console.WriteLine(index + " " + move.Position.FEN);
                  equalCount++;
                }
                index++;
//                Console.WriteLine(move.Position + "  " + move.Position.MiscInfo.RepetitionCount);
              }

              Console.WriteLine($"Num equal {equalCount}");
              if (equalCount == 0)
              {
                Console.WriteLine("found top level terminal " + child.NumChildrenExpanded + " " + child.Annotation.Pos.CalcTerminalStatus()
                                 + " " + child.Annotation.Pos.FEN);
                DumpMoveStatistics(manager.Root);
              }
            }
          }
        }
      }
#endif

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

      manager.TrySetImmediateBestMove(manager.Root);
      if (manager.TablebaseImmediateBestMove != default(MGMove))
      {
        manager.StopStatus = SearchStopStatus.TablebaseImmediateMove;
        return manager.TablebaseImmediateBestMove;
      }

      // If only one legal move then force the search to run only one node
      // (to get root populated) and then stop.
      MGPosition startPos = priorMoves.FinalPosMG;
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in startPos, moves);

      // Check if playing using only action head.
      if (manager.SearchLimit.Type == SearchLimitType.BestActionMove)
      {
        if (manager.SearchLimit.Value != 1)
        {
          throw new Exception("BestActionMove only supported for NodesPerMove == 1.");
        }

        NNEvaluatorResult rootResult = manager.Context.NNEvaluators.Evaluator1.Evaluate(priorMoves, manager.Context.ParamsSearch.HistoryFillIn, false);

        MGPosition thisPos = priorMoves.FinalPosMG;

        MGMove bestMove = default;
        float bestActionV = float.MinValue;
        foreach ((EncodedMove Move, float Probability) policyMove in rootResult.Policy.ProbabilitySummary())
        {
          (float w, float d, float l) zz = rootResult.ActionWDLForMove(policyMove.Move);
          float actionV = zz.w - zz.l;
          if (actionV > bestActionV)
          {
            bestMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(policyMove.Move, thisPos);
            bestActionV = actionV;
          }
        }

        Debug.Assert(bestActionV != float.MinValue); // expected to find at least one move

        context.Root.StructRef.W = bestActionV;
        context.Root.StructRef.N = 1;
        return context.TopVForcedMove = bestMove;
      }

      // Check if playing using only value head (TopV).
      if (manager.SearchLimit.Type == SearchLimitType.BestValueMove)
      {
        if (manager.SearchLimit.Value != 1)
        {
          throw new Exception("BestValueMove only supported for NodesPerMove == 1.");
        }
        
        NNEvaluatorResult[] valueEvalResults = null;
        (MGMove bestMove, float bestV)  = BestValueMove(manager.Context.NNEvaluators.Evaluator1, priorMoves, 
                                                        ref moves, ref valueEvalResults,
                                                        manager.Context.ParamsSearch.HistoryFillIn, false);

        context.Root.StructRef.W = bestV;
        context.Root.StructRef.N = 1;
        return context.TopVForcedMove = bestMove;        
      }

      bool shouldStopAfterOneNodeDueToOnlyOneLegalMove = false;
      if (moveImmediateIfOnlyOneMove && moves.NumMovesUsed == 1)
      {
        shouldStopAfterOneNodeDueToOnlyOneLegalMove = true;
        manager.SearchLimit = SearchLimit.NodesPerMove(1);
      }
      
      MCTSNode root = manager.Root;

      //root.MarkImmediateDrawsByRepetition();

      // Do the search
      IteratedMCTSDef schedule = manager.Context.ParamsSearch.IMCTSSchedule;
      bool useIMCTS = schedule != null & manager.SearchLimit.EstNumSearchNodes(root.N, 30_000, false) > 100;

      MCTSNode selectedMove;
      BestMoveInfo bestMoveInfo;

      SearchLimit thisSearchLimit = manager.SearchLimit with { };
      int numSearches = 0;
      BestMoveInfo firstTryBestMoveInfo = null;
      SearchLimit startingSearchLimit = manager.SearchLimit with { };
      bool shouldExtendSearch;
      do
      {
        shouldExtendSearch = false;

        selectedMove = useIMCTS ? new IteratedMCTSSearchManager().IteratedSearch(manager, progressCallback, schedule).Item2
                                : manager.DoSearch(thisSearchLimit, progressCallback).Item2;

        // Get best child 
        // TODO: technically the "updateStats" should only be true if we end up accepting this move
        if (forceMove != default)
        {
          if (!moves.Contains(forceMove))
          {
            throw new Exception($"Specified forced move {forceMove} is not legal in position {priorMoves.FinalPosition.FEN}");
          }
        }

        bestMoveInfo = root.BestMoveInfo(true, forceMove);

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

#if FEATURE_RAPIDLY_IMPROVING
        bool foundRapidlyImprovingQ = false;
        if (false && root.Context.ParamsSearch.TestFlag2)
        {
          for (int i = 0; i < root.NumChildrenExpanded; i++)
          {
            if (root.Context.RootMoveTracker.RunningVValues != null)
            {
              MCTSNode child = root.ChildAtIndex(i);
              float fracNOfParent = (float)child.N / root.N;
              float runningQ = root.Context.RootMoveTracker.RunningVValues[i];
              float runningQBetterThanBestQ = -runningQ - bestMoveInfo.QOfBest;

              if (child.N > 200
                && root.Context.RootMoveTracker.LastRootN[i] > 0
                && ((fracNOfParent > 0.10f && runningQBetterThanBestQ > 0.10f)
                  || (fracNOfParent > 0.20f && runningQBetterThanBestQ > 0.05f))
                && (i != bestMoveInfo.BestMoveNode.IndexInParentsChildren))
              {
                foundRapidlyImprovingQ = true;
                throw new Exception("Need to pass this true flag down to if statement below");
                MCTSEventSource.TestCounter1++;
                if (bestMoveInfo.BestMoveQSuboptimality <= 0.01) /// TEMPORARY redundant below
                  Console.WriteLine(root.N + "  " +  child.N + " found extra extension " + (float)child.N / root.N + "  " + runningQBetterThanBestQ);
//root.Context.Manager.DumpRootMoveStatistics();
              }
            }
          }
        }
#endif
        // If the chosen move is far away from the best Q node, 
        // try to extend the search unless the position is obviously won/lost.
        const float Q_THRESHOLD = 0.01f;
        const int MAX_RETRIES = 3;
        const float INCREMENT_FRACTION = 0.20f;
        bool possiblyExtend = bestMoveInfo.BestMoveQSuboptimality > Q_THRESHOLD; // other move has much better Q already
        if (manager.Context.ParamsSearch.EnableSearchExtension
         &&!shouldStopAfterOneNodeDueToOnlyOneLegalMove
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

      if (verbose)
      {
        DumpMoveStatistics(root);
      }

      if (possiblyUsePositionCache && manager.Context.EvaluatorDef.CacheMode == PositionEvalCache.CacheMode.MemoryAndDisk)
      {
        manager.SaveCache(manager.Context.EvaluatorDef.CacheFileName);
      }

      if (shouldStopAfterOneNodeDueToOnlyOneLegalMove)
      {
        manager.StopStatus = SearchStopStatus.OnlyOneLegalMove;
        return moves.MovesArray[0];
      }
      else
      {
        return bestMoveInfo.BestMove;
      }
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
        {
          return TablebaseImmediateBestMove;
        }

        return Context.Root.BestMove(false).Annotation.PriorMoveMG;
      }
    }


    bool disposed = false;

    public void Dispose()
    {
      if (!disposed)
      {
        Context.Tree.Store.Dispose();

        // Release references to objects
        Context = null;

        disposed = true;
      }
    }

#endregion

#region Time management

    public int MaxBatchSizeDueToPossibleNearTimeExhaustion
    {
      get
      {
        if (SearchLimit.Type != SearchLimitType.SecondsPerMove)
        {
          return int.MaxValue;
        }

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
          SearchLimitType.NodesPerMove => MathHelpers.Bounded((SearchLimit.Value - NumNodesVisitedThisSearch) / SearchLimit.Value, 0, 1),
          SearchLimitType.NodesPerTree => MathHelpers.Bounded(1.0f - (NumNodesVisitedThisSearch / (SearchLimit.Value - RootNWhenSearchStarted)), 0, 1),
          SearchLimitType.BestValueMove => MathHelpers.Bounded(1.0f - (NumNodesVisitedThisSearch / (SearchLimit.Value - RootNWhenSearchStarted)), 0, 1),
          SearchLimitType.BestActionMove => MathHelpers.Bounded(1.0f - (NumNodesVisitedThisSearch / (SearchLimit.Value - RootNWhenSearchStarted)), 0, 1),
          _ => throw new NotImplementedException()
        };
      }
    }


    public float RemainingTime
    {
      get
      {
        if (SearchLimit.Type != SearchLimitType.SecondsPerMove)
        {
          return float.MaxValue;
        }

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

      UpdateTopNodeInfo();

      if (Search.LastGameLimitInputs != null && LimitManager.CheckStopSearch(Search, Search.LastGameLimitInputs))
      {
        return SearchStopStatus.LimitsManagerRequestedStop;
      }

        if (ExternalStopRequested)
      {
        return SearchStopStatus.ExternalStopRequested;
      }

      if (RemainingTime <= 0.01)
      {
        return SearchStopStatus.TimeLimitReached;
      }

      if (SearchLimit.MaxTreeVisits != null
       && Root.N >= SearchLimit.MaxTreeVisits
       && NumNodesVisitedThisSearch > 0) // always allow a little search to insure state fully initialized
      {
        return SearchStopStatus.MaxTreeVisitsExceeded;
      }

      if (SearchLimit.MaxTreeNodes != null
       && Root.Tree.Store.Nodes.NumTotalNodes >= (SearchLimit.MaxTreeNodes - 2048)
       && NumNodesVisitedThisSearch > 0) // always allow a little search to insure state fully initialized
      {
        return SearchStopStatus.MaxTreeAllocatedNodesExceeded;
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

      float elapsedSecs = (float)(DateTime.Now - StartTimeFirstVisit).TotalSeconds;
      bool insufficientData = elapsedSecs < MIN_TIME || NumNodesVisitedThisSearch < MIN_VISITS;
      estimatedNPS = insufficientData ? float.NaN : NumNodesVisitedThisSearch / elapsedSecs;
    }

    public int? EstimatedNumVisitsRemaining()
    {
      if (SearchLimit.Type == SearchLimitType.NodesPerMove)
      {
        int nodesProcessedAlready = Root.N - RootNWhenSearchStarted;
        return (int)MathF.Max(0, SearchLimit.Value - nodesProcessedAlready);
      }
      else if (SearchLimit.Type == SearchLimitType.NodesPerTree)
      {
        return (int)MathF.Max(0, SearchLimit.Value - Root.N);
      }
      else if (SearchLimit.Type == SearchLimitType.BestValueMove)
      {
        return (int)MathF.Max(0, SearchLimit.Value - Root.N);
      }
      else if (SearchLimit.Type == SearchLimitType.BestActionMove)
      {
        return (int)MathF.Max(0, SearchLimit.Value - Root.N);
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
      if (!newBestNode.IsNull)
      {
        if (TopNChildIndex is null || (newBestNode.IndexInParentsChildren != TopNChildIndex))
        {
          TopNChildIndex = newBestNode.IndexInParentsChildren;
          TopNChildN = newBestNode.N;

          NumNodesWhenChoseTopNNode = Root.N;
        }
      }
    }

    public GameMoveStat FirstMoveBySide(SideType side)
    {
      for (int i = 0; i < PriorMoveStats.Count; i++)
      {
        if (PriorMoveStats[i].Side == side)
        {
          return PriorMoveStats[i];
        }
      }
      return null;
    }


    public int CountTablebaseHits
    {
      get
      {
        LeafEvaluatorSyzygy syzygyEvaluator = (LeafEvaluatorSyzygy)Context.LeafEvaluators.Find(eval => eval.GetType() == typeof(LeafEvaluatorSyzygy));
        return syzygyEvaluator == null ? 0 : (int)LeafEvaluatorSyzygy.NumHits.Value;
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
