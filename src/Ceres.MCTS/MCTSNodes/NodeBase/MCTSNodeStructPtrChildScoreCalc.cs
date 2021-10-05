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

using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Struct;

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

      internal GatheredChildStats()
      {
        const int ALIGNMENT = 64; // For AVX efficiency

        N = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
        InFlight = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
        P = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
        W = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
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

      // Gather necessary fields
      // TODO: often NInFlight of parent is null (thus also children) and we could
      //       have special version of Gather which didn't bother with that
      nodeRef.GatherChildInfo(Context, new MCTSNodeStructIndex(Index), selectorID, depth, numToProcess - 1,
                              gatherStatsNSpan, gatherStatsInFlightSpan,
                              gatherStatsPSpan, gatherStatsWSpan);

      if (Context.ParamsSelect.PolicyDecayFactor > 0)
      {
        ApplyPolicyDecay(numToProcess, gatherStatsPSpan);
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
          if (Context.RootMovesPruningStatus[i] != Iteration.MCTSFutilityPruningStatus.NotPruned
           && gatherStatsNSpan[i] > 0)
          {
            // At root the search wants best Q values 
            // but because of minimax prefers moves with worse Q and W for the children
            // Therefore we set W of the child very high to make it discourage visits to it.
            gatherStatsWSpan[i] = float.MaxValue;
          }
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
