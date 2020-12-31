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

using Ceres.Base.DataType;
using Ceres.Base.Math;

using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  public unsafe sealed partial class MCTSNode
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

    [ThreadStatic] static Random qNoiseRandom;


    /// <summary>
    /// Returns the thread static variables, intializaing if first time accessed by this thread.
    /// </summary>
    /// <returns></returns>
    static (GatheredChildStats, Random) CheckInitThreadStatics()
    {
      GatheredChildStats stats = gatherStats;
      if (stats == null)
      {
        stats = gatherStats = new GatheredChildStats();
        Random noiseRandom = qNoiseRandom = new Random();
        return (gatherStats, noiseRandom);
      }
      else
        return (stats, qNoiseRandom);
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
    /// <param name="numVisitsToCompute">target number of visits across all children</param>
    /// <param name="scores">the output U values</param>
    /// <param name="childVisitCounts">the output child visit counts</param>
    public void ComputeTopChildScores(int selectorID, int depth, float dynamicVLossBoost,
                                      int minChildIndex, int maxChildIndex, int numVisitsToCompute,
                                      Span<float> scores, Span<short> childVisitCounts)
    {
      (GatheredChildStats stats, Random qNoise) = CheckInitThreadStatics();

      if (numVisitsToCompute <= 0) throw new ArgumentOutOfRangeException(nameof(numVisitsToCompute), "must be positive");
      if (minChildIndex != 0) throw new ArgumentOutOfRangeException(nameof(minChildIndex), "must be zero (current implementation restriction)");
      if (maxChildIndex > MCTSScoreCalcVector.MAX_CHILDREN) throw new ArgumentOutOfRangeException(nameof(maxChildIndex), "must be less than MCTSScoreCalcVector.MAX_CHILDREN");

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

      // Possibly add an additional exploration bonus term with two features:
      //   - does not go to zero as quickly as n --> N
      //   - influence of the prior probabiities diminish as N gets larger
      // TODO: inefficient
      float CPUCT2 = Context.ParamsSelect.CPUCT2 / 1000.0f;
      const int CPUCT2_THRESHOLD = 10;
      if (CPUCT2 > 0)
      {
        Span<float> w = gatherStatsWSpan;
        Span<float> n = gatherStatsNSpan;

        // float newV2 = MathF.Sqrt(P) * (MathF.Log(N / n) * CPUCT2);
        for (int i = 0; i < numToProcess; i++)
        {
          if (n[i] > CPUCT2_THRESHOLD)
            w[i] -= CPUCT2 * FastLog.Ln(N) / n[i];
        }
      }

      float cPolicyFade = Context.ParamsSelect.CPolicyFade;
      const int MIN_N = 1000; // too little impact to bother spending time on this if if few nodes
      if (depth == 0 && N > MIN_N && cPolicyFade > 0) 
      {
        // Determine how much probability is included in this set of moves
        // (when we are done we must leave this undisturbed)
        float startingProb = 0;
        for (int i = 0; i < gatherStatsPSpan.Length; i++) startingProb += gatherStatsPSpan[i];

        // determine what softmax exponent to use
        float softmax = 1.0f + (cPolicyFade * (MathF.Log(N, 2) - 8) / 100);

        float acc = 0;
        for (int i = 0; i < gatherStatsPSpan.Length; i++)
        {
          float value = MathF.Pow(gatherStatsPSpan[i], 1.0f / softmax);
          gatherStatsPSpan[i] = value;
          acc += value;
        }

        // Renormalize so that the final sum of probability is still startingProb
        for (int i = 0; i < gatherStatsPSpan.Length; i++) gatherStatsPSpan[i] *= startingProb / acc;
      }

      // Possibly apply Q noise     
      if (Context.ParamsSearch.QNoiseFactorNonRoot > 0 
       || Context.ParamsSearch.QNoiseFactorRoot > 0)
      {
        float qRandomVariabilityRoot = Context.ParamsSearch.QNoiseFactorRoot;
        float qRandomVariabilityNonRoot = Context.ParamsSearch.QNoiseFactorNonRoot;
        for (int i = 0; i < numToProcess; i++)
        {
          // Apply noise only to nodes with many visits
          // since exploration will already play a major role
          if (gatherStatsNSpan[i] > 1000)
          {
            float rand = (float)qNoise.NextDouble();

            // Adjust W by:
            //   - shift to center noise around 0
            //   - scale by specified coefficient
            //   - also scale by N since W is N times Q
            float multiplier = IsRoot ? qRandomVariabilityRoot : qRandomVariabilityNonRoot;
            gatherStatsWSpan[i] += gatherStatsNSpan[i] * (rand - 0.5f) * multiplier;
          }
        }
      }

      // Strongly disfavor node if smaller than specified minimum
      if (IsRoot && Context.RootMovesArePruned != null)
      {
        for (int i = 0; i < numToProcess; i++)
        {
          if (Context.RootMovesArePruned[i])
          {
            // At root the search wants best Q values 
            // but because of minimax prefers moves with worse Q and W for the children
            // Therefore we set W of the child very high to make it discourage visits to it
            gatherStatsWSpan[i] = gatherStatsNSpan[i];
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
                                         scores, childVisitCounts);
    }


    /// <summary>
    /// Returns the UCT score (used to select best child) for specified child
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="depth"></param>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    public float ChildScore(int selectorID, int depth, int childIndex) => CalcChildScores(selectorID, depth, 0)[childIndex];


    /// <summary>
    /// Computes the UCT scores (used to select best child) for all children
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="depth"></param>
    /// <param name="dynamicVLossBoost"></param>
    /// <returns></returns>
    public float[] CalcChildScores(int selectorID, int depth, float dynamicVLossBoost)
    {
      Span<float> scores = new Span<float>(new float[NumPolicyMoves]);
      Span<short> childVisitCounts = new Span<short>(new short[NumPolicyMoves]);

      ComputeTopChildScores(selectorID, depth, dynamicVLossBoost, 0, NumPolicyMoves - 1, 1, scores, childVisitCounts);
      return scores.ToArray();
    }


  }
}
