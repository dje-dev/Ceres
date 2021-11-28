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

using Ceres.Base.Math;
using Ceres.Base.Math.Probability;
using Ceres.Base.Math.Random;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.NNFiles;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;

using Ceres.MCTS.Evaluators;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.NodeCache;
using Ceres.MCTS.Params;
using Ceres.MCTS.Search;

#endregion

namespace Ceres.MCTS.Iteration
{
  [Serializable]
  public class MCTSIterator
  {
    public readonly ParamsSearch ParamsSearch;
    public readonly ParamsSelect ParamsSelect;
    public NNEvaluatorDef EvaluatorDef => NNEvaluators.EvaluatorDef;
    public NNEvaluatorDef EvaluatorDefSecondary => NNEvaluators.EvaluatorDefSecondary;

    public ContemptManager ContemptManager;

    public MCTSNodeChildrenStatsTracker RootMoveTracker = new MCTSNodeChildrenStatsTracker();

    /// <summary>
    /// Set of nodes which have been identified as ready
    /// for batch evaluation by secondary evaluator.
    /// </summary>
    public ConcurrentBag<MCTSNode> PendingSecondaryNodes = new();

    internal struct TranspositionCluster
    {
      internal int ParentIndex;
      internal List<int> LinkedIndices;

      internal TranspositionCluster(int parentIndex)
      { 
        ParentIndex = parentIndex;
        LinkedIndices = null;
      }

      internal void AddLinkedNode(int nodeIndex)
      {
        if (LinkedIndices == null) LinkedIndices = new List<int>(4);
        LinkedIndices.Add(nodeIndex);
      }
    }

   
    public readonly List<LeafEvaluatorBase> LeafEvaluators;

    // Basic state
    public readonly PositionWithHistory StartPosAndPriorMoves;
    public SideType SideToMove => StartPosAndPriorMoves.FinalPosition.MiscInfo.SideToMove;

    public readonly NNEvaluatorSet NNEvaluators;

    public MCTSTree Tree;

    public MCTSManager.MCTSProgressCallback  ProgressCallback;

    /// <summary>
    /// Pruning status of root moves which may flag certain 
    /// moves as not being eligible for additional search visits.
    /// 
    /// These status be progressively toggled as search progresses and it becomes
    /// clear that certain children can never "catch up" to 
    /// the current highest N node and thus would never be chosen
    /// (instead we can allocate remaining visits over children still having a chance)
    /// </summary>
    public MCTSFutilityPruningStatus[] RootMovesPruningStatus;

    // Incremented by the depth of every node selected
    // Used to track "average depth" of search tree
    internal AccumulatorMultithreaded CumulativeSelectedLeafDepths;

    /// <summary>
    /// Total number of nodes that were requested for select
    /// </summary>
    public int NumNodeVisitsAttempted = 0;

    /// <summary>
    /// Total number of visits that succeeded. 
    /// This is the number of leafs which were either new children expanded,
    /// or repeat visits to terminal nodes.
    /// It is typically less than NumNodeVisitsAttempted due to either
    /// collisions from the same selector, or cross-selector duplication.
    /// </summary>
    public int NumNodeVisitsSucceeded = 0;

    internal static long totalNumNodeVisitsSucceeded;
    internal static long totalNumNodeVisitsAttempted;

    public static float TotalYieldFrac => (float)totalNumNodeVisitsSucceeded / (float) totalNumNodeVisitsAttempted;

    public MultinomialBayesianThompsonSampler FirstMoveSampler;

    MCTSIterator reuseOtherContextForEvaluatedNodes;
    public MCTSIterator ReuseOtherContextForEvaluatedNodes { get { return reuseOtherContextForEvaluatedNodes; } }
    public void ClearSharedContext() => reuseOtherContextForEvaluatedNodes = null;

    public static float LastBatchYieldFrac;

    public MCTSNode Root => Tree.Root;

    public int NumNNBatches = 0;

    public int NumNNNodes = 0;

    public CheckTablebaseBestNextMoveDelegate CheckTablebaseBestNextMove;

    public bool TablebasesInUse => CheckTablebaseBestNextMove != null;

    public float AvgDepth => (float)CumulativeSelectedLeafDepths.Value / (float)Root.N;
    public float NodeSelectionYieldFrac => NumNodeVisitsAttempted == 0 
                                         ? 0
                                         : (float)NumNodeVisitsSucceeded / (float)NumNodeVisitsAttempted;



    void VerifyParamsValid()
    {
      // TODO: generalize this to validation method called on each of the parameter objects
      if (ParamsSearch.ApplyTrendBonus && !MCTSParamsFixed.TRACK_NODE_TREND)
      {
        throw new Exception("ApplyTrendBonus requires MCTSParamsFixed.TRACK_NODE_TREND");
      }

      if (ParamsSearch.Execution.TranspositionMode == TranspositionMode.None &&
        (ParamsSearch.Execution.InFlightThisBatchLinkageEnabled || ParamsSearch.Execution.InFlightOtherBatchLinkageEnabled))
      {
        throw new Exception("Requesting IN_FLIGHT_LINKAGE_ENABLED is incompatible with TranspositionMode.None");
      }
    }


    public void RecordVisitToTopLevelMove(MCTSNode leafNode, MCTSNodeStructIndex indexOfChildDescendentFromRoot, LeafEvaluationResult evalResult)
    {
#if NOT
      if (indexOfChildDescendentFromRoot == -1) return; // if the root

      int childIndex = -1;
      Span<MCTSNodeStructChild> children = Root.Ref.Children;
      for (int i = 0; i < Root.NumChildrenExpanded; i++)
      {
        if (children[i].ChildIndex.Index == indexOfChildDescendentFromRoot)
        {
          childIndex = i;
          break;
        }
      }
      if (childIndex == -1) throw new Exception("Internal error");

      if (rootChildrenMovingAverageValues == null)
      {
        rootChildrenMovingAverageN = new float[Root.NumPolicyMoves];
        rootChildrenMovingAverageValues = new float[Root.NumPolicyMoves];
        rootChildrenMovingAverageSquaredValues = new float[Root.NumPolicyMoves];
      }

      float v = evalResult.V * (leafNode.Depth % 2 == 0 ? -1 : 1);
      float diff =  v - (float)Root.Ref.Children[childIndex].ChildRef.Q;

      const float C1 = 0.99f;
      const float C2 = 1.0f - C1;
      static void UpdateStat(ref float statistic, float newValue) =>
        statistic = statistic * C1 + newValue * C2;
      static void UpdateStat2(ref float statistic, float newValue) =>
        statistic = C1 * (statistic + (C2 * newValue));



      UpdateStat(ref rootChildrenMovingAverageValues[childIndex], diff);// -evalResult.V);
      UpdateStat2(ref rootChildrenMovingAverageSquaredValues[childIndex], diff * diff);

      if (false && COUNT++ % 1000 == 999)
      {
        Console.WriteLine();
        for (int i = 0; i < Root.NumChildrenExpanded; i++)
        {
          if (rootChildrenMovingAverageValues[i] != 0)
          {
            Console.WriteLine(i + " " + Root.Ref.Children[i].ChildRef.PriorMove
                                + " " + Root.Ref.Children[i].ChildRef.P
                                + " " + Root.Ref.Children[i].ChildRef.N + " " + Root.Ref.Children[i].ChildRef.Q 
                                + " " + rootChildrenMovingAverageValues[i] + " " + rootChildrenMovingAverageSquaredValues[i]
                                + " " + GetU2(i));
          }
        }

      }
#endif
    }

    public MCTSIterator(MCTSNodeStore store)
    {
      throw new Exception("remediate next line");
     // Tree.Store = store; // TODO: ** TEMPORARY, remove this method
    }

    public readonly MCTSManager Manager;

    public MCTSIterator(MCTSManager manager, MCTSNodeStore store,
                         MCTSIterator reuseOtherContextForEvaluatedNodes,
                         PositionEvalCache reusePositionCache,
                         IMCTSNodeCache reuseNodeCache,
                         TranspositionRootsDict reuseTranspositionRoots,
                         NNEvaluatorSet nnEvaluators,
                         ParamsSearch paramsSearch,
                         ParamsSelect paramsSelect,
                         int estimatedNumSearchNodes)
    {
      Manager = manager;

      // Make sure params arguments look initialized
      if (nnEvaluators == null) throw new ArgumentNullException(nameof(nnEvaluators));

      this.reuseOtherContextForEvaluatedNodes = reuseOtherContextForEvaluatedNodes;
      
      // Release the leaf evaluators on the prior context to break chain to all prior Contexts
      // TODO: clean up the LeafEvaluatorReuseOtherThree to be more minimal what it attaches to
      reuseOtherContextForEvaluatedNodes?.LeafEvaluators.Clear();

      NNEvaluators = nnEvaluators;
      ParamsSearch = paramsSearch;
      ParamsSelect = paramsSelect;

      VerifyParamsValid();

      if (paramsSelect.FirstMoveThompsonSamplingFactor != 0) FirstMoveSampler = new MultinomialBayesianThompsonSampler(MCTSScoreCalcVector.MAX_CHILDREN);                                                                                       

      ContemptManager = new ContemptManager(paramsSearch.Contempt, paramsSearch.ContemptAutoScaleWeight);

      // The store might subsequently be modified and reused (tree reuse)
      // thus we need to save our own copy of the PriorMoves which is won't change
      StartPosAndPriorMoves = new PositionWithHistory(store.Nodes.PriorMoves);

      PositionEvalCache positionCache = null;
      if (reusePositionCache != null)
      {
        positionCache = reusePositionCache;
      }
      else if (nnEvaluators.EvaluatorDef.CacheMode != PositionEvalCache.CacheMode.None)
      {
        positionCache = new PositionEvalCache();
        positionCache.InitializeWithSize(false, estimatedNumSearchNodes / 2);
      }
      
      int estimatedNodesBound = store.Nodes.NumUsedNodes + estimatedNumSearchNodes;

      Tree = new MCTSTree(store, this, estimatedNodesBound, positionCache, reuseNodeCache);

      if (ParamsSearch.Execution.TranspositionMode != TranspositionMode.None)
      {
        Tree.TranspositionRoots = reuseTranspositionRoots ?? new TranspositionRootsDict(estimatedNodesBound);
      }

      LeafEvaluators = BuildPreprocessors();
      Tree.ImmediateEvaluators = LeafEvaluators;
      CumulativeSelectedLeafDepths.Initialize();
    }

    public float CurrentContempt => ContemptManager.TargetContempt;


   internal void BatchPostprocessAllEvaluators()
   {
     LeafEvaluators.ForEach((evaluator) => evaluator.BatchPostprocess());
   }

   List<LeafEvaluatorBase> BuildPreprocessors()
    {
      // Note that the preprocessors will be called in same order as this list,
      // so we put the ones which are more definitive (e.g. terminal) 
      // or easier to compute (e.g. cache lookup) first
      List<LeafEvaluatorBase> evaluators = new List<LeafEvaluatorBase>();

      // Put the inexpensive (no movegen needed) draw checks first.
      // It is possibly important to detect draw by repetitions before
      // transpositions which are less sensitive to move history.
      evaluators.Add(new LeafEvaluatorTerminalDrawn());

      // First check transposition table (if enabled)
      // since this is very inexpensive (just a dictionary lookup)
      // and does not need to generate and store moves for the node.
      if (ParamsSearch.Execution.TranspositionMode != TranspositionMode.None)
      {
//        if (ParamsNN.CACHE_MODE != Chess.PositionEvalCaching.PositionEvalCache.CacheMode.None)
//          throw new Exception("USE_TRANSPOSITIONS requires CACHE_MODE be None, probably incompatable.");
        evaluators.Add(new LeafEvaluatorTransposition(Tree, Tree.TranspositionRoots));
      }

      // Next check for terminal (which will need to generate moves).
      evaluators.Add(new LeafEvaluatorTerminalCheckmateStalemate());

      // Possibly try to levarage NN evaluations stored inside the tree from the separate search context
      if (ReuseOtherContextForEvaluatedNodes != null)
      {
        evaluators.Add(new LeafEvaluatorReuseOtherTree(ReuseOtherContextForEvaluatedNodes));
      }

      if (Tree.PositionCache != null)
      {
        // We are inheriting a position cache
        if (NNEvaluators.EvaluatorDef.CacheMode == PositionEvalCache.CacheMode.None)
        {
          // Cache passed in for reference purposes but not to be updated
          // (such as when using cache from nodes of a reused tree that were not retained)
          Tree.PositionCache.ReadOnly = true;
        }

        evaluators.Add(new LeafEvaluatorCache(Tree.PositionCache));
      }
      else if (NNEvaluators.EvaluatorDef.CacheMode != PositionEvalCache.CacheMode.None)
      {
        // Create a position cache for this search.
        evaluators.Add(new LeafEvaluatorCache(Tree.PositionCache));
      }

      if (ParamsSearch.EnableTablebases)
      {
        evaluatorTB  = new LeafEvaluatorSyzygyLC0(CeresUserSettingsManager.Settings.TablebaseDirectory, Manager.ForceNoTablebaseTerminals);
        evaluators.Add(evaluatorTB);
        CheckTablebaseBestNextMove = (in Position currentPos, out GameResult result, out List<MGMove> otherWinningMoves, out bool winningMoveListOrderedByDTM)
          => RootTablebaseMoveCheck(in currentPos, out result, out otherWinningMoves, out winningMoveListOrderedByDTM);
        TablebaseDTZAvailable = evaluatorTB.Evaluator.DTZAvailable;

        // Also add a 1-ply lookahead evaluator (for captures yielding tablebase terminal)
        evaluatorTBPly1 = new LeafEvaluatorSyzygyPly1(evaluatorTB, Manager.ForceNoTablebaseTerminals);
        evaluators.Add(evaluatorTBPly1);
      }

      return evaluators;
    }

    LeafEvaluatorSyzygyLC0 evaluatorTB;
    LeafEvaluatorSyzygyPly1 evaluatorTBPly1;

    MGMove RootTablebaseMoveCheck(in Position currentPos, out GameResult result, out List<MGMove> fullWinningMoveList, out bool winningMoveListOrderedByDTM)
    {
      MGMove ret = evaluatorTB.Evaluator.CheckTablebaseBestNextMove(in currentPos, out result, out fullWinningMoveList, out winningMoveListOrderedByDTM);
      return ret;
    }

    public void SetNodeNotFutilityPruned(MCTSNode node)
    {
      int nodeIndex = node.IndexInParentsChildren;
      if (RootMovesPruningStatus[nodeIndex] == MCTSFutilityPruningStatus.PrunedDueToFutility)
      {
        RootMovesPruningStatus[nodeIndex] = MCTSFutilityPruningStatus.NotPruned;
      }
    }


    internal bool TablebaseDTZAvailable;

    public int[] OptimalBatchSizesForNetDims(int gpuNum, int numFilters, int numLayers)
    {
      if (gpuNum == 3)
      {
        if (numFilters == 128 && numLayers == 10)
          return new[] { 56, 96, 144, 192 };
        else if (numFilters == 192 && numLayers == 16)
          return new[] { 72, 144, 288, 386 };
        else if (numFilters == 256 && numLayers == 20)
          return new[] { 72, 144, 216, 288, 336 };
        else if (numFilters == 320 && numLayers == 24)
          return new[] { 144, 288, 432 };
        else if (numFilters == 384 && numLayers == 30)
          return new[] { 48, 96, 144, 192, 240, 288, 336 };
        else
          return null;
      }
      else
      {
        if (numFilters == 128 && numLayers == 10)
          return new[] { 160, 320 };
        else if (numFilters == 192 && numLayers == 16)
          return new[] { 96, 160, 320 };
        else if (numFilters == 256 && numLayers == 20)
          return new[] { 160 };
        else if (numFilters == 320 && numLayers == 24)
          return new[] { 80, 160, 240 };
        else if (numFilters == 384 && numLayers == 30)
          return new[] { 80, 160, 192, 240 };
        else
          return null;
      }
    }

    public int[] GetOptimalBatchSizeBreaks(int gpuNum)
    {
      int numFilters = -1;
      int numLayers = -1;

      for (int i = 0; i < NNEvaluators.EvaluatorDef.Devices.Length; i++)
      {
        INNWeightsFileInfo def = NNWeightsFiles.LookupNetworkFile(NNEvaluators.EvaluatorDef.Nets[0].Net.NetworkID);

        // Give up if dissimilar
        if (def.NumFilters != numFilters || def.NumBlocks != numLayers)
        {
          if (i == 0)
          {
            numFilters = def.NumFilters;
            numLayers = def.NumBlocks;
          }
          else
          {
            return null; // give up if not all same. TODO: improve this
          }
        }
      }

      return OptimalBatchSizesForNetDims(gpuNum, numFilters, numLayers);
    }

    public int NumMovesNoiseOverridden { get; internal set; }

    public static int TotalNumMovesNoiseOverridden = 0;

    public void SetDirichletExplorationNoise(float dirichletAlpha, float fractionNoise)
    {
      Random rand = new Random((int)DateTime.Now.Ticks);
      float[] overrideRootMovePriors = new float[Root.NumPolicyMoves];
      float sumP = 0;
      for (int i = 0; i < overrideRootMovePriors.Length; i++)
      {
        float p = Root.ChildAtIndexInfo(i).p;
        float gamma = (float)GammaDistribution.RandomDraw(rand, dirichletAlpha, 1.0f);
        float newP = p * (1.0f - fractionNoise) + (gamma * fractionNoise);
        overrideRootMovePriors[i] = newP;
        sumP += newP;
      }

      // Normalize to sum to 1.0
      float adj = 1.0f / sumP;
      for (int i = 0; i < overrideRootMovePriors.Length; i++)
      {
        overrideRootMovePriors[i] *= adj;
      }

      throw new Exception("Dirichlet noise not yet fully working (next need to inject override P into node store and resort children)");
    }



    public float MBonusForNode(ref MCTSNodeStruct nodeRef, bool isOurMove)
    {
      // NOTE: always returns a value even if context.ParamsSearch.ApplyMBonus is false
      if (float.IsNaN(nodeRef.MPosition) || nodeRef.IsRoot) return 0;

      const float M_SCALE = 0.02f;
      const float MAX_BONUS = M_SCALE;

      float mChange = nodeRef.MAvg - nodeRef.ParentRef.MAvg;
      float contempt = CurrentContempt;

      float bonus = 0;
      if (isOurMove)
      {
        bonus = ((float)Root.Q - contempt) * -mChange * M_SCALE; // more positive as we are winning, or MLH decreases
      }
      else
      {
        bonus = (float)Root.Q * mChange * M_SCALE; // more positive as we are losing by more or MLH increases
      }
//        bonus = (contempt - (float)Root.Q) * mChange * M_SCALE; // becomes more negative the higher our Q or the nubmer of moves left increases

      //      if (isOurMove) bonus *= -1;

      bonus = MathHelpers.Bounded(bonus, -MAX_BONUS, MAX_BONUS);

//      if(nodeRef.N % 723==3) Console.WriteLine(bonus + " MBONUS parentM="  + nodeRef.ParentRef.MAvg  + " M=" + + nodeRef.MAvg + " Q=" + Root.Q);

      return -bonus; // lower numbers are better
    }
  }
}
