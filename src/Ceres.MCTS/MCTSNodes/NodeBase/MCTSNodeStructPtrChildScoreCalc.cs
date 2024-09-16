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
using System.Diagnostics;
using Ceres.Base.Math;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Managers.Uncertainty;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  public unsafe partial struct MCTSNodeInfo
  {
    /// <summary>
    /// Internal class that holds the spans in which the child statistcs are gathered.
    /// </summary>
    [ThreadStatic] static GatheredChildStats gatherStats;


    /// <summary>
    /// Returns the thread static variables, intializaing if first time accessed by this thread.
    /// </summary>
    /// <returns></returns>
    internal static GatheredChildStats CheckInitThreadStatics()
    {
      GatheredChildStats stats = gatherStats;
      if (stats == null)
      {
        return gatherStats = new GatheredChildStats();
      }
      else
      {
        return stats;
      }
    }


    /// <summary>
    /// Applies CPUCT selection to determine for each child
    /// their U scores and the number of visits each should receive
    /// if a speciifed number of total visits will be made to this node.
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="depth"></param>
    /// <param name="dynamicVLossBoost"></param>
    /// <param name="minChildIndex"></param>
    /// <param name="maxChildIndex"></param>
    /// <param name="numVisitsToCompute">number of child visits to select, or 0 to merely calculate scores</param>
    /// <param name="scores"></param>
    /// <param name="childVisitCounts"></param>
    /// <param name="cpuctMultiplier"></param>
    public void ComputeTopChildScores(int selectorID, int depth, float dynamicVLossBoost,
                                      int minChildIndex, int maxChildIndex, int numVisitsToCompute,
                                      Span<float> scores, Span<short> childVisitCounts, float cpuctMultiplier,
                                      float[] empiricalDistrib, float empiricalWeight, float actionHeadSelectionWeight)
    {
      GatheredChildStats stats = CheckInitThreadStatics();

      Debug.Assert(numVisitsToCompute >= 0);
      Debug.Assert(minChildIndex == 0); // implementation restriction
      Debug.Assert(maxChildIndex <= MCTSScoreCalcVector.MAX_CHILDREN);

      ref MCTSNodeStruct nodeRef = ref Ref;

      int numToProcess = Math.Min(Math.Min(maxChildIndex + 1, (int)nodeRef.NumPolicyMoves), MCTSScoreCalcVector.MAX_CHILDREN);

      if (numToProcess == 0) return;

      Span<float> gatherStatsNSpan = stats.N.Span;
      Span<float> gatherStatsInFlightSpan = stats.InFlight.Span;
      Span<float> gatherStatsPSpan = stats.P.Span;
      Span<float> gatherStatsWSpan = stats.W.Span;
      Span<float> gatherStatsUSpan = stats.U.Span;

      // TODO: avoid inefficient allocation/initialization of CompressedActionVector even when needed
      CompressedActionVector actionVector = default;
      bool hasAction = false;
      if (actionHeadSelectionWeight != 0 && nodeRef.Context.Store.AllActionVectors != null)
      {
        hasAction = true;
        actionVector = nodeRef.Context.Store.AllActionVectors[index.Index];
      } 

      // Gather necessary fields
      // TODO: often NInFlight of parent is null (thus also children) and we could
      //       have special version of Gather which didn't bother with that
      nodeRef.GatherChildInfo(Context, new MCTSNodeStructIndex(Index), selectorID, depth, numToProcess - 1,
                              in actionVector, actionHeadSelectionWeight != 0,
                              gatherStats);
      
      if (nodeRef.IsRoot 
        && nodeRef.N > 500 // sufficient number of samples
        && nodeRef.Context.Info.Context.ParamsSelect.FracWeightUseRunningQ > 0)
      {
        float frac = nodeRef.Context.Info.Context.ParamsSelect.FracWeightUseRunningQ;
        for (int i = 0; i <= maxChildIndex; i++)
        {
          float runningQ = nodeRef.Context.Info.Context.RootMoveTracker.RunningVValues[i];
          gatherStatsWSpan[i] = (1.0f - frac) * gatherStatsWSpan[i] 
                              + frac          * (float)runningQ * gatherStatsNSpan[i];
        }
      }

      if (empiricalWeight > 0)
      {
        for (int i=0; i<numToProcess; i++)
        {
          gatherStatsPSpan[i] = gatherStatsPSpan[i] * (1.0f - empiricalWeight)
                              + empiricalDistrib[i] * empiricalWeight;
        }
      }

      if (Context.ParamsSelect.PolicyDecayFactor > 0)
      {
        ApplyPolicyDecay(numToProcess, gatherStatsPSpan);
      }


#if NOT
      // Katago CPUCT scaling technique (TestFlag2)
      if (nodeRef.N > MIN_N_USE_UNCERTAINTY 
      && Context.ParamsSearch.TestFlag2)
      {
        float var = (nodeRef.VarianceAccumulator - (float)(nodeRef.Q*nodeRef.Q))
                  / (nodeRef.N - MCTSNodeStruct.VARIANCE_START_ACCUMULATE_N);
        if (var > 0)
        {
          float sd = MathF.Sqrt(var);

          const float MULT = 1.5f;
          float diffFromAvg = sd - 0.12f;
          cpuctMultiplier = 1 + diffFromAvg * MULT;
          cpuctMultiplier = StatUtils.Bounded(cpuctMultiplier - 0.03f, 0.88f, 1.12f);
//          cpuctMultiplier = 1 + -(cpuctMultiplier - 1);
          accScale += cpuctMultiplier;
          countScale++;

          MCTSEventSource.TestCounter1++;

          if (Ref.ZobristHash % 2_000 == 0)
          {
            MCTSEventSource.TestMetric1 = (accScale / countScale);
          }
        }
      }
#endif

#if ENABLE_REFUTATION_SEARCH_MORE_EXPORATION_FEATURE
      bool isOpponentMove = depth % 2 == 1;
      MCTSEventSource.TestCounter1++;

      // Search for refutations only on opponent's move.
      if (/*Ref.Q < -0.20f && */isOpponentMove && Context.ParamsSearch.TestFlag2 && Context.Root.N > 500)
      { 
        float fractionOfTree = (float)N / Context.Root.N;
        bool isProbablyOnPV = fractionOfTree > 0.2f;
        bool hasSiblingSupport = false;
        if (isProbablyOnPV)
        {
          if (depth != 1) // Since first move is irrevocable, a sibling move cannot serve as support since we have already committed
          {
            ref readonly MCTSNodeStruct parent = ref Ref.ParentRef;
            foreach (MCTSNodeStructChild sibling in parent.Children)
            {
              const float SIBLING_THRESHOLD = 0.10f;
              if (sibling.IsExpanded && Math.Abs(sibling.ChildRef(parent.Context.Store).Q - Ref.Q) < SIBLING_THRESHOLD)
              {
                hasSiblingSupport = true;
                break;
              }
            }
          }
        }

        const float CPUCT_MULTIPLIER_IF_REFUTATION_SEARCH= 1.35f;
        bool boostCPUCT = isOpponentMove && isProbablyOnPV && !hasSiblingSupport;
        if (boostCPUCT)
        {
          cpuctMultiplier *= CPUCT_MULTIPLIER_IF_REFUTATION_SEARCH;
          MCTSEventSource.TestMetric1++;
        }
      }

#endif
      if (Context.ParamsSearch.TestFlag2)
      {
        for (int i = 0; i < NumChildrenVisited; i++)
        {
          if (N > 3 /*&& gatherStatsNSpan[i] < 2*/ && gatherStatsNSpan[i] > 0)
          {
            // *0.02, max 0.05 +6 Elo
            float boost =  (gatherStatsUSpan[i] - 15) * 0.01f;
            const float MAX = 0.05f;
//            boost = gatherStatsNSpan[i] * StatUtils.Bounded(boost, 0, MAX);
            boost = StatUtils.Bounded(boost, -MAX, MAX); // N.B. not multiplied by NSpan so influence wanes as N grows since W is an accumulator
            gatherStatsWSpan[i] -= boost; // smaller becomes more attractive
//            Console.WriteLine(gatherStatsUSpan[i] +  " " + boost);
          }
        }
      }

      if (MCTSParamsFixed.UNCERTAINTY_TESTS_ENABLED && Context.ParamsSearch.TestFlag2)
      {
        float uTotal = 0;
        int uCount = 0;
        for (int i = 0; i < NumChildrenVisited; i++)
        {
          uTotal += gatherStatsUSpan[i];
          uCount++;
        }

        float uAvg = uTotal / uCount;
        for (int i = 0; i < NumChildrenVisited; i++)
        {
          if (/*gatherStatsNSpan[i] > 0 && */gatherStatsInFlightSpan[i] == 0)
          {
            //            int iToUse = (i + 1) % NumChildrenExpanded; // ******** RANDOMIZE TEST ONLY ********8
            float boostBase = (gatherStatsUSpan[i] - uAvg) / MathF.Sqrt(gatherStatsNSpan[i]);
            const float MAX = 0.08f;
            boostBase = StatUtils.Bounded(2 * 0.01f * boostBase, -MAX, MAX);
            //Console.WriteLine(boostBase);
            gatherStatsWSpan[i] -= boostBase * gatherStatsNSpan[i]; // smaller becomes more attractive
            gatherStatsUSpan[i] = boostBase; // store back as boost for later use
//              Console.WriteLine("set to " + boostBase);
          }
          else
          {
            gatherStatsUSpan[i] = 0;
          }
        }

      }
      
      if (false && Context.ParamsSearch.TestFlag2)
      {
        bool beRisky = Context.Root.Q < -0.2f && nodeRef.DepthInTree % 2 == 0;
//        if (beRisky)
        {
          float sumU = 0;
          int sumCount = 0;
          for (int i = 0; i < numToProcess && i < NumChildrenExpanded; i++)
          {
            sumU += gatherStatsUSpan[i];
            sumCount++;
          }
          float avgU = sumU / sumCount;
          for (int i = 0; i < numToProcess && i < NumChildrenExpanded; i++)
          {
            float diffFromAvg = Math.Min(7, gatherStatsUSpan[i] - avgU);

            if (diffFromAvg > 0)
            {
              float boostBase = (diffFromAvg * 0.01f) * 1f;
              float boostDiv = MathF.Pow(gatherStatsNSpan[i],  0.2f);
              float boost = boostBase / boostDiv;
              //gatherStatsWSpan[i] -= boost * gatherStatsNSpan[i]; // smaller becomes more attractive
//              gatherStatsPSpan[i] += 0.01f * diffFromAvg;// good (only -3 Elo)
              
              float add = 1 *  0.2f * 0.01f * diffFromAvg;
              gatherStatsPSpan[i] *= Math.Min(gatherStatsPSpan[i] + 0.10f, 1.0f + (0.02f * diffFromAvg));
              gatherStatsPSpan[i] += add;
            }
          }
        }
      }
#if LEGACY_UNCERTAINTY
      // We've gathered all the uncertainty statistics into U.
      // Now if they are applicable (feature turned on, enough samples)
      // compute the appropriate scaling factor
      // and update gatherStatsPSpan via multiplication to have this applied.
      if (N >= MCTSNodeUncertaintyAccumulator.MIN_N_ESTIMATE
       && Context.ParamsSearch.EnableUncertaintyBoosting)
      {
        // Note that to be precise there should be an additional term subtraced off
        // (for the mean squared) in variance calculation below, but it is omitted because:
        //   - in practice the magnitude is too small to be worth the computational effort.
        //   - the mean is computed over all visits but the variance accumulated not over first VARIANCE_START_ACCUMULATE_N
        //     therefore this subtraction could produce non-consistent results (negative variance)
        float parentMAD = nodeRef.Uncertainty.Uncertainty;

        // Possibly apply scaling to each child.
        float accUncertaintyMultiplier = 0;
        float nUsed = 0;
        for (int i = 0; i < numToProcess && i < NumChildrenExpanded; i++)
        {
          if (gatherStatsNSpan[i] >= MCTSNodeUncertaintyAccumulator.MIN_N_ESTIMATE)
          {
            float childMAD = gatherStatsUSpan[i];
            float explorationScaling = UncertaintyScalingCalculator.ExplorationMultiplier(childMAD, parentMAD);

            accUncertaintyMultiplier += explorationScaling * gatherStatsNSpan[i];
            nUsed += gatherStatsNSpan[i];

            // Scale P by this uncertainty scaling factor
            // which is equivalent to having separate multiplicand
            // but more convenient than creating separately.
            gatherStatsPSpan[i] *= explorationScaling;
          }
        }

        // Adjust all the modified multipliers 
        // such that the weighted (by N) average adjustment is 1.0
        // to avoid unintended change in average CPUCT.
        float avgScale = accUncertaintyMultiplier /= nUsed;
        float adj = 1.0f / avgScale;
        for (int i = 0; i < numToProcess
                     && i < NumChildrenExpanded; i++)
        {
          if (gatherStatsNSpan[i] >= MCTSNodeUncertaintyAccumulator.MIN_N_ESTIMATE)
          {
            gatherStatsPSpan[i] *= adj;
          }
        }
      }
#endif

        // Possibly disqualify pruned moves from selection.
        if ((IsRoot && Context.RootMovesPruningStatus != null)
       && (numVisitsToCompute != 0) // do not skip any if only querying all scores 
         )
      {
        for (int i = 0; i < numToProcess; i++)
        {
          // Note that moves are never pruned if the do not yet have any visits
          // because otherwise the subsequent leaf selection will never 
          // be able to proceed beyond this unvisited child.
          if (Context.RootMovesPruningStatus[i] != MCTSFutilityPruningStatus.NotPruned
           && gatherStatsNSpan[i] > 0)
          {
            // At root the search wants best Q values 
            // but because of minimax prefers moves with worse Q and W for the children
            // Therefore we set W of the child very high to make it discourage visits to it.
            gatherStatsWSpan[i] = float.MaxValue;
          }
        }
      }

      // If any child is a checkmate then exploration is not appropriate,
      // set cpuctMultiplier to 0 as an elegant means of effecting certainty propagation
      // (no changes to algorithm are needed, all subsequent visits will go to this terminal node).
      if (ParamsSearch.CheckmateCertaintyPropagationEnabled
       && nodeRef.CheckmateKnownToExistAmongChildren)
      {
        const bool ALLOW_MINIMAL_EXPORATION = true;
        if (ALLOW_MINIMAL_EXPORATION)
        {
          // Minimal exploration may allow "better mates" to be eventually found
          // (e.g. a tablebase mate in 3 instead of mate in 30).
          cpuctMultiplier = 0.1f;
        }
        else
        {
          cpuctMultiplier = 0f;
          numToProcess = Math.Min(numToProcess, nodeRef.NumChildrenExpanded);
        }
      }

      // Compute scores of top children
      MCTSScoreCalcVector.ScoreCalcMulti(Context.ParamsSearch.Execution.FlowDualSelectors, Context.ParamsSelect, selectorID, dynamicVLossBoost,
                                         nodeRef.IsRoot, nodeRef.N, selectorID == 0 ? nodeRef.NInFlight : nodeRef.NInFlight2,
                                         (float)nodeRef.Q, nodeRef.SumPVisited,
                                         stats,
                                         numToProcess, numVisitsToCompute,
                                         scores, childVisitCounts, cpuctMultiplier, actionHeadSelectionWeight);
      if (numVisitsToCompute > 0)
      {
        FillInSequentialVisitHoles(childVisitCounts, ref nodeRef, numToProcess);
      }
    }


    /// <summary>
    /// Ceres algorithms require children to be visited strictly sequentially,
    /// so no child is visited before all of its siblings with smaller indices have already been visited.
    /// 
    /// This method insures this condition is always satisfied by shfting leftward
    /// any children which otherwise be to the right of some unexpanded node.
    /// </summary>
    /// <param name="childVisitCounts"></param>
    /// <param name="nodeRef"></param>
    /// <param name="numToProcess"></param>
    private static void FillInSequentialVisitHoles(Span<short> childVisitCounts, ref MCTSNodeStruct nodeRef, int numToProcess)
    {
      // Fixup any holes
      int numExpanded = nodeRef.NumChildrenExpanded;
      for (int i = numExpanded; i < numToProcess; i++)
      {
        if (childVisitCounts[i] == 0)
        {
          for (int j = numToProcess - 1; j > i; j--)
          {
            if (childVisitCounts[j] > 0)
            {
              childVisitCounts[i] = 1;
              childVisitCounts[j]--;
              break;
            }
          }
        }
      }
    }


    /// <summary>
    /// Possibly applies supplemental decay to policies priors
    /// (if PolicyDecayFactor is not zero).
    /// </summary>
    /// <param name="numToProcess"></param>
    /// <param name="gatherStatsPSpan"></param>
    private void ApplyPolicyDecay(int numToProcess, Span<float> gatherStatsPSpan)
    {
      const int MIN_N = 100; // too little impact to bother spending time on this if if few nodes

      // Policy decay is only applied at root node where
      // the distortion created from pure MCTS will not be problematic
      // because the extra visits are not propagated up and
      // the problem at the root is best arm identification.
      if (N > MIN_N && Depth == 0)
      {
        float policyDecayFactor = Context.ParamsSelect.PolicyDecayFactor;

        if (policyDecayFactor > 0)
        {
          float policyDecayExponent = Context.ParamsSelect.PolicyDecayExponent;

          // Determine how much probability is included in this set of moves
          // (when we are done we must leave this undisturbed)
          float startingProb = 0;
          for (int i = 0; i < numToProcess; i++) startingProb += gatherStatsPSpan[i];

          // determine what softmax exponent to use
          float softmax = 1 + MathF.Log(1 + policyDecayFactor * 0.0002f * MathF.Pow(N, policyDecayExponent));

          float acc = 0;
          float power = 1.0f / softmax;
          for (int i = 0; i < numToProcess; i++)
          {
            float value = MathF.Pow(gatherStatsPSpan[i], power);
            gatherStatsPSpan[i] = value;
            acc += value;
          }

          // Renormalize so that the final sum of probability is still startingProb
          float multiplier = startingProb / acc;
          for (int i = 0; i < numToProcess; i++)
          {
            gatherStatsPSpan[i] *= multiplier;
          }
        }
      }
    }


    /// <summary>
    /// Returns the UCT score (used to select best child) for specified child
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="depth"></param>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    public float ChildScore(int selectorID, int depth, int childIndex) => CalcChildScores(selectorID, depth)[childIndex];


    /// <summary>
    /// Computes the UCT scores (used to select best child) for all children
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="depth"></param>
    /// <param name="dynamicVLossBoost"></param>
    /// <returns></returns>
    public float[] CalcChildScores(int selectorID, int depth, float dynamicVLossBoost = 0, float cpuctMultiplier = 1)
    {
      Span<float> scores = new float[NumPolicyMoves];
      Span<short> childVisitCounts = new short[NumPolicyMoves];

      ComputeTopChildScores(selectorID, depth, dynamicVLossBoost, 0, NumPolicyMoves - 1, 0, scores, childVisitCounts, 
                            cpuctMultiplier, default, 0, Context.ParamsSearch.ActionHeadSelectionWeight);
      return scores.ToArray();
    }

  }
}
