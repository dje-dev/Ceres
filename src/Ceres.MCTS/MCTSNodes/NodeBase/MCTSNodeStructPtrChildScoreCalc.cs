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
using Ceres.Base.DataType;
using Ceres.Base.Math;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
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
    private class GatheredChildStats
    {
      internal SpanAligned<float> N;
      internal SpanAligned<float> InFlight;
      internal SpanAligned<float> P;
      internal SpanAligned<float> W;
      internal SpanAligned<float> U;

      internal GatheredChildStats()
      {
        const int ALIGNMENT = 64; // For AVX efficiency

        N = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
        InFlight = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
        P = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
        W = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
        U = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
      }
    }

    [ThreadStatic] static GatheredChildStats gatherStats;


    /// <summary>
    /// Returns the thread static variables, intializaing if first time accessed by this thread.
    /// </summary>
    /// <returns></returns>
    static GatheredChildStats CheckInitThreadStatics()
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



    [ThreadStatic] static double accScale;
    [ThreadStatic] static int countScale;

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
                                      Span<float> scores, Span<short> childVisitCounts, float cpuctMultiplier)
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

      // Gather necessary fields
      // TODO: often NInFlight of parent is null (thus also children) and we could
      //       have special version of Gather which didn't bother with that
      nodeRef.GatherChildInfo(Context, new MCTSNodeStructIndex(Index), selectorID, depth, numToProcess - 1,
                              gatherStatsNSpan, gatherStatsInFlightSpan,
                              gatherStatsPSpan, gatherStatsWSpan, gatherStatsUSpan);


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

      int MIN_N_USE_UNCERTAINTY = 30; // only use once sufficient data to be reliable
      if (N > MIN_N_USE_UNCERTAINTY && Context.ParamsSearch.EnableUncertaintyBoosting)
      {
        // Note that to be precise there should be an additional term subtraced off
        // (for the mean squared) in variance calculation below, but it is omitted because:
        //   - in practice the magnitude is too small to be worth the computational effort.
        //   - the mean is computed over all visits but the variance accumulated not over first VARIANCE_START_ACCUMULATE_N
        //     therefore this subtraction could produce non-consistent results (negative variance)
        float parentMAD = nodeRef.VarianceAccumulator / (nodeRef.N - MCTSNodeStruct.VARIANCE_START_ACCUMULATE_N);

        // Possibly apply scaling to each child.
        for (int i = 0; i < numToProcess
                     && i < NumChildrenExpanded; i++)
        {
          if (gatherStatsNSpan[i] > MIN_N_USE_UNCERTAINTY)
          {
            float explorationScaling = 1.0f;
            float childMAD = gatherStatsUSpan[i] / (gatherStatsNSpan[i] - MCTSNodeStruct.VARIANCE_START_ACCUMULATE_N);

#if NOT
            if (nonlinear)
            {
                // The uncertainty scaling is a number centered at 1 which is
                // higher for children with more emprical volatility than the parent and
                // lower for children with less volatility.
                float p = gatherStatsPSpan[i];
                float adj = 5 * (childMAD - parentMAD);
                explorationScaling = (1 + 15 * p + adj) / (1 + 15 * p);
                explorationScaling += 0.04f;
                explorationScaling = StatUtils.Bounded(explorationScaling, 0.666f, 1.5f);
              //              Console.WriteLine(explorationScaling + " adj=" + adj + " p=" + p);
            }
            else
            {
#endif
            float UNCERTAINTY_DIFF_MULTIPLIER = 2.0f;
            float UNCERTAINTY_MAX_DEVIATION = 0.15f;
            float BIAS_ADJUST = 0.01f; // adjustment to make average value turn out to be very close to 1.0

            // The uncertainty scaling is a number centered at 1 which is
            // higher for children with more emprical volatility than the parent and
            // lower for children with less volatility.
            explorationScaling = 1 + BIAS_ADJUST + UNCERTAINTY_DIFF_MULTIPLIER * (childMAD - parentMAD);
            explorationScaling = StatUtils.Bounded(explorationScaling, 1.0f - UNCERTAINTY_MAX_DEVIATION, 1.0f + UNCERTAINTY_MAX_DEVIATION);

            const bool SHOW_UNCERTAINTY_STATS = false; // perforamnce degrading
            if (SHOW_UNCERTAINTY_STATS)
            {
              // Update statistics.
              if (MathF.Abs(explorationScaling - 1) > 0.05f)// && Ref.ZobristHash % 1_000 == 0)
              {
                accScale += explorationScaling;
                countScale++;
                MCTSEventSource.TestCounter1++;
                MCTSEventSource.TestMetric1 = (accScale / countScale);
              }
            }

            // Scale P by this uncertainty scaling factor
            // which is equivalent to having separate multiplicand
            // but more convenient than creating separately.
            gatherStatsPSpan[i] *= explorationScaling;
          }

        }
      }

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
                                         gatherStatsPSpan, gatherStatsWSpan,
                                         gatherStatsNSpan, gatherStatsInFlightSpan,
                                         numToProcess, numVisitsToCompute,
                                         scores, childVisitCounts, cpuctMultiplier);

      FillInSequentialVisitHoles(childVisitCounts, ref nodeRef, numToProcess);
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
    public float ChildScore(int selectorID, int depth, int childIndex) => CalcChildScores(selectorID, depth, 0, 0)[childIndex];


    /// <summary>
    /// Computes the UCT scores (used to select best child) for all children
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="depth"></param>
    /// <param name="dynamicVLossBoost"></param>
    /// <returns></returns>
    public float[] CalcChildScores(int selectorID, int depth, float dynamicVLossBoost, float cpuctMultiplier)
    {
      Span<float> scores = new float[NumPolicyMoves];
      Span<short> childVisitCounts = new short[NumPolicyMoves];

      ComputeTopChildScores(selectorID, depth, dynamicVLossBoost, 0, NumPolicyMoves - 1, 0, scores, childVisitCounts, cpuctMultiplier);
      return scores.ToArray();
    }


  }
}
