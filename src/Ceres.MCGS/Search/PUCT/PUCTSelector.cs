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
using System.Threading;
using Ceres.Base.Math;
using Ceres.Chess;

using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.QProbeSelect;
using Ceres.MCGS.Search.RPO;
using Ceres.MCGS.Search.Strategies;

#endregion

namespace Ceres.MCGS.Search.PUCT;

public static partial class PUCTSelector
{
  /// <summary>
  /// Internal class that holds the spans in which the child statistics are gathered.
  /// </summary>
  [ThreadStatic] static GatheredChildStats gatherStats;

  /// <summary>
  /// Thread-local buffer for qWhenNoChildrenComposite to avoid per-call allocations.
  /// </summary>
  [ThreadStatic] static float[] qWhenNoChildrenBuffer;

  /// <summary>
  /// Thread-local float copies of the (double) Q-uncertainty adjustment arrays, needed because
  /// the score kernel loads them as Vector&lt;float&gt;. Only populated when QUnc is active.
  /// </summary>
  [ThreadStatic] static float[] quncScoreBonusFloat;
  [ThreadStatic] static float[] quncUMultiplierFloat;

  /// <summary>
  /// Thread-local score-bonus array for the attention directives (Graph.AttentionEntries),
  /// used only when the QUnc bonus array is not already materialized for this call.
  /// </summary>
  [ThreadStatic] static float[] attentionBonusFloat;


  /// <summary>
  /// Narrows the first numToProcess entries of a (double) per-child adjustment array into a
  /// thread-local float array, padding the remainder with a neutral value so the SIMD block
  /// loads in the score kernel read defined lanes.
  /// </summary>
  static float[] NarrowAdjustments(double[] source, ref float[] dest, int numToProcess, float neutralFill)
  {
    if (source == null)
    {
      return null;
    }

    dest ??= new float[PUCTScoreCalcVector.MAX_CHILDREN];
    for (int i = 0; i < numToProcess; i++)
    {
      dest[i] = (float)source[i];
    }
    for (int i = numToProcess; i < dest.Length; i++)
    {
      dest[i] = neutralFill;
    }

    return dest;
  }


  /// <summary>
  /// Returns the thread static variables, initializing if first time accessed by this thread.
  /// </summary>
  /// <returns></returns>
  internal static GatheredChildStats CheckInitThreadStatics()
  {
    GatheredChildStats stats = gatherStats;
    return stats ?? (gatherStats = new GatheredChildStats());
  }


  /// <summary>
  /// Applies CPUCT selection to determine for each child
  /// their U scores and the number of visits each should receive
  /// if a specified number of total visits will be made to this node.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="node"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="selectorID"></param>
  /// <param name="rootMovePruningStatus"></param>
  /// <param name="dualCollisionFraction"></param>
  /// <param name="minChildIndex"></param>
  /// <param name="maxChildIndex"></param>
  /// <param name="numTargetVisits"></param>
  /// <param name="scores"></param>
  /// <param name="childVisitCounts"></param>
  /// <param name="cpuctMultiplier"></param>
  /// <param name="temperatureMultiplier"></param>
  /// <param name="quncContext">Optional Q-uncertainty select context (null = exact stock
  /// behavior). Callers must pass null on probe-suppressed descents (refreshStaleEdges
  /// false) so data harvesting stays model-free.</param>
  public static NodeSelectAccumulator ComputeTopChildScores(Graph graph, GNode node,
                                                            ParamsSearch paramsSearch, ParamsSelect paramsSelect,
                                                            int selectorID, bool refreshStaleEdges,
                                                            MCGSFutilityPruningStatus[] rootMovePruningStatus,
                                                            float dualCollisionFraction,
                                                            int minChildIndex, int maxChildIndex, int numTargetVisits,
                                                            Span<double> scores, Span<short> childVisitCounts,
                                                            float cpuctMultiplier,
                                                            float temperatureMultiplier,
                                                            QUncSelectContext quncContext = null,
                                                            int quncPathDepth = 0)
  {
    Debug.Assert(cpuctMultiplier >= 0);

    GatheredChildStats stats = CheckInitThreadStatics();

    Debug.Assert(numTargetVisits >= 0);
    Debug.Assert(minChildIndex == 0); // implementation restriction
    Debug.Assert(maxChildIndex <= PUCTScoreCalcVector.MAX_CHILDREN);
    Debug.Assert(node.IsLocked);

    ref readonly GNodeStruct nodeRef = ref node.NodeRef;

    int numToProcess = Math.Min(Math.Min(maxChildIndex + 1, nodeRef.NumPolicyMoves),
                                PUCTScoreCalcVector.MAX_CHILDREN);

    if (numToProcess == 0)
    {
      return new NodeSelectAccumulator(int.MinValue, double.NaN, double.NaN, 0);
    }

    // Gather necessary fields
    // TODO: often NInFlight of parent is null (thus also children) and we could
    //       have special version of Gather which didn't bother with that


    graph.GatherChildInfoViaChildren(node, selectorID, maxChildIndex, dualCollisionFraction, stats, refreshStaleEdges);

    // Possibly use action head directly
    float[] qWhenNoChildrenComposite = null;
    if (numToProcess > 1 && paramsSelect.GetFPUMode(node.IsSearchRoot) == ParamsSelect.FPUType.ActionHead)
    {
      // Use per-move value from the neural network action head as FPU for unvisited children.
      float fallbackFPU = (float)paramsSelect.CalcQWhenNoChildren(node.IsSearchRoot, node.Q, stats.SumPVisited);
      ReadOnlySpan<float> actionSpan = stats.A.Span;
      qWhenNoChildrenComposite = qWhenNoChildrenBuffer ??= new float[PUCTScoreCalcVector.MAX_CHILDREN];
      for (int i = 0; i < numToProcess; i++)
      {
        float actionV = actionSpan[i] + MCGSStrategyPUCT.ACTION_HEAD_FPU_VALUE;

        // Negate because child Q values are stored from opponent's perspective
        qWhenNoChildrenComposite[i] = float.IsNaN(actionV) ? fallbackFPU : -actionV;
      }
    }
    else if (numToProcess > 1
          && node.NumEdgesExpanded > 0
          && paramsSelect.GetFPUMode(node.IsSearchRoot) is ParamsSelect.FPUType.PolicyImputedRPO)
    {
      qWhenNoChildrenComposite = ApplyRPOImputedFPU(paramsSelect, node, stats, numToProcess);
    }

    if (FPURunningStats.DEBUG_DUMP_FPU_CORRELATION_STATS)
    {
      FPURunningStats.Record(node, paramsSearch, paramsSelect, qWhenNoChildrenComposite, stats.P.Span, numToProcess);
    }

    // Possibly apply supplemental temperature scaling.
    if (temperatureMultiplier != 1 && numToProcess > 1)
    {
      TemperatureScaler.ApplyTemperature(node.NumPolicyMoves, stats.P.Span[..numToProcess],
                                         stats.SumPVisited, temperatureMultiplier);
    }


    if (GatheredChildStats.GATHER_UNCERTAINTY && paramsSearch.TestFlag)
    {
      Span<float> uncertaintyPolicySpan = stats.UP.Span;
      Span<float> uncertaintyValueSpan = stats.UV.Span;
      Span<float> nSpan = stats.N.Span;
      Span<float> wSpan = stats.W.Span;
      for (int i = 0; i < Math.Min(node.NumEdgesExpanded, numToProcess); i++)
      {
        if (!float.IsNaN(uncertaintyValueSpan[i]))
        {
          const float VALUE_UNCERTAINTY_WEIGHT = 0.3f;
          float n = MathF.Max(1, nSpan[i]);
          if (n == 1)
          {
            float adjust = (VALUE_UNCERTAINTY_WEIGHT * uncertaintyValueSpan[i]) / MathF.Sqrt(n);
            //adjust *= -1;
            wSpan[i] -= adjust * nSpan[i];
          }
        }
      }
    }


    // In old class MCTSNodeSTructScoreCalc see implementations of:
    //      if (Context.ParamsSelect.PolicyDecayFactor > 0)
    // Katago CPUCT scaling technique (TestFlag2)


    // Possibly disqualify pruned moves from selection.
    if (node.IsSearchRoot && rootMovePruningStatus != null
   && numTargetVisits != 0) // do not skip any if only querying all scores          
    {
      Span<float> gatherStatsNSpan = stats.N.Span;
      Span<float> gatherStatsWSpan = stats.W.Span;
      for (int i = 0; i < numToProcess; i++)
      {
        // Note that moves are never pruned if the do not yet have any visits
        // because otherwise the subsequent leaf selection will never 
        // be able to proceed beyond this unvisited child.
        if (rootMovePruningStatus[i] != Managers.MCGSFutilityPruningStatus.NotPruned
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
    // set cpuctMultiplier to low value as an elegant means of effecting certainty propagation
    // (no changes to algorithm are needed, all subsequent visits will go to this terminal node).
    if (ParamsSearch.CheckmateCertaintyPropagationEnabled && nodeRef.CheckmateKnownToExistAmongChildren)
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
        numToProcess = Math.Min(numToProcess, node.NumEdgesExpanded);
      }
    }

    double sumPVisited = stats.SumPVisited;

#if DEBUG
    double sumPVisitedRecalc = 0;
    for (int i=0;i<node.NumEdgesExpanded;i++)
    {
      GEdge childEdge = node.ChildEdgeAtIndex(i);
      if (childEdge.N > 0)
      {
        // Debug.Assert(childEdge.N > 0); not true if parallel enabled
        sumPVisitedRecalc += childEdge.P;
      }
    }
    // Non-agreement here due to NumChildrenVisited not yet counting nodes with N=0, but NInFlight>0
    //Debug.Assert(Math.Abs(sumPVisited - sumPVisitedRecalc) < 1e-6);
    // Therefore do this weaker test:
    Debug.Assert(sumPVisited >= sumPVisitedRecalc);
#endif

    int numVisitsAccepted = 0;
    if (numToProcess == 1 && scores.IsEmpty)
    {
      // No need to compute in this special case of only child to consider and scores not requested.
      childVisitCounts[0] = (short)numTargetVisits;
      numVisitsAccepted = numTargetVisits;
    }
    else
    {
      // previously: int parentNumInFlightX = selectorID == 0 ? nodeRef.NInFlight : nodeRef.NInFlight1;      
      // TODO: Tests at 50 and 500 nodes/move suggest setting this always zero is better?
      //       Probably this is not correct, reflects only a poor tuning of CPUCT,
      //       and this is just having effect of backdoor CPUCT change.
      double parentNumInFlight = stats.SumNumInFlightAll;


      // Compute scores of top children
      float thresholdPUCTSuboptimalityReject = float.MaxValue;
      if (paramsSearch.VisitSuboptimalityRejectThreshold != null)
      {
        thresholdPUCTSuboptimalityReject = paramsSearch.VisitSuboptimalityRejectThreshold.Value;
      }

      // Q-uncertainty methods: per-child score bonuses / U multipliers (M1-M4) applied
      // inside the score kernels. The driver also runs for M5-only configurations
      // (returning no adjustments) because select-time gathers are what populate the
      // per-child forecast cache that the TPS backup (M5a/M5b) reads.
      double[] quncScoreBonus = null;
      double[] quncUMultiplier = null;
      if (quncContext != null && numTargetVisits > 0 && numToProcess > 1)
      {
        QUncEvalDriver.PrepareAdjustments(quncContext, node, in nodeRef, paramsSelect, numToProcess,
                                          quncPathDepth, out quncScoreBonus, out quncUMultiplier);
      }

      // The score kernel consumes these as Vector<float>; narrow (neutral fill past numToProcess).
      float[] quncScoreBonusF = NarrowAdjustments(quncScoreBonus, ref quncScoreBonusFloat, numToProcess, 0f);
      float[] quncUMultiplierF = NarrowAdjustments(quncUMultiplier, ref quncUMultiplierFloat, numToProcess, 1f);

      // Attention directives (ADVOCATE): fading additive selection bonuses on specific
      // (parent, child) edges, merged through the same per-child score-bonus channel the QUnc
      // methods use (a lone non-null bonus array routes through the QUnc kernel with a neutral
      // U multiplier, i.e. score += bonus). Inert unless AttentionBonusEpsilon > 0 and this node
      // is a flagged parent; gated on refreshStaleEdges (like quncContext) so probe-harvest
      // descents stay stock.
      bool attentionApplied = false;
      float attentionEps = paramsSelect.AttentionBonusEpsilon;
      if (attentionEps > 0 && refreshStaleEdges && numTargetVisits > 0
          && !nodeRef.CheckmateKnownToExistAmongChildren)
      {
        ProbeAttentionEntry[] attention = graph.AttentionEntries;
        if (attention != null)
        {
          int parentIndex = node.Index.Index;
          for (int e = 0; e < attention.Length; e++)
          {
            if (attention[e].ParentNodeIndex != parentIndex)
            {
              continue;
            }
            int slot = IndexOfExpandedChild(node, attention[e].ChildNodeIndex,
                                            Math.Min(numToProcess, nodeRef.NumEdgesExpanded));
            if (slot < 0)
            {
              continue;
            }
            if (quncScoreBonusF == null)
            {
              quncScoreBonusF = attentionBonusFloat ??= new float[PUCTScoreCalcVector.MAX_CHILDREN];
              Array.Clear(quncScoreBonusF);
            }
            float k = attention[e].K;
            quncScoreBonusF[slot] += attentionEps * attention[e].W * k / (stats.N.Span[slot] + k);
            attentionApplied = true;
          }
        }
      }

      numVisitsAccepted = PUCTScoreCalcVector.ScoreCalcMulti(paramsSelect,
                                                              node.IsSearchRoot, nodeRef.N,
                                                              parentNumInFlight,
                                                              nodeRef.Q, sumPVisited,
                                                              stats,
                                                              qWhenNoChildrenComposite,
                                                              numToProcess, numTargetVisits,
                                                              scores, childVisitCounts, cpuctMultiplier,
                                                              thresholdPUCTSuboptimalityReject,
                                                              parentNode: node,
                                                              quncScoreBonus: quncScoreBonusF,
                                                              quncUMultiplier: quncUMultiplierF);

      // Some scoring paths can produce allocations that violate the sequential-expansion
      // invariant ("no child gets a visit before all of its left siblings have at least
      // one visit"):
      //   - Per-child FPU (ActionHead or PolicyImputedRPO): the imputed per-child q is not
      //     monotonic in policy, so a later unvisited child can outscore an earlier one.
      //     qWhenNoChildrenComposite is non-null exactly when per-child FPU was built above.
      // Leaving a hole (an unexpanded slot before another that got a visit) corrupts memory
      // in Graph.InitializeNewEdge, so the fixup must run for every per-child path.  It is
      // cheap and only relocates visits when an actual hole is present.  The attention bonus
      // is also a per-child adjustment, so it runs the same fixup (insurance: its targets are
      // always expanded edges, but a bonus can still reorder visits among unexpanded slots
      // indirectly).
      if (numTargetVisits > 0
          && (qWhenNoChildrenComposite != null || attentionApplied))
      {
        FillInSequentialVisitHoles(childVisitCounts, ref node.NodeRef, numToProcess);
      }
    }

    // Return accumulated value across all children and also contribution from the node itself.
    double nToUse = node.Terminal.IsTerminal() ? node.N : 1;
    return new NodeSelectAccumulator(nToUse + stats.SumNVisited,
                                     (nToUse * (double)nodeRef.V) + -stats.SumWVisited,
                                     (nToUse * (double)nodeRef.DrawP) + stats.SumDVisited,
                                     numVisitsAccepted);
  }


  /// <summary>
  /// Returns the slot of the expanded child edge whose child node has the given index,
  /// or -1 if none. Used by the attention directives, which identify the child by node
  /// index because edge slots can be permuted between searches (move ordering phases).
  /// </summary>
  static int IndexOfExpandedChild(GNode node, int childNodeIndex, int numSlots)
  {
    for (int i = 0; i < numSlots; i++)
    {
      if (node.ChildEdgeAtIndex(i).ChildNodeIndex.Index == childNodeIndex)
      {
        return i;
      }
    }
    return -1;
  }


  /// <summary>
  /// Ceres algorithms require children to be visited strictly sequentially,
  /// so no child is visited before all of its siblings with smaller indices have already been visited.
  /// 
  /// This method insures this condition is always satisfied by shifting leftward
  /// any children which otherwise be to the right of some unexpanded node.
  /// </summary>
  /// <param name="childVisitCounts"></param>
  /// <param name="nodeRef"></param>
  /// <param name="numToProcess"></param>
  private static void FillInSequentialVisitHoles(Span<short> childVisitCounts,
                                                 ref readonly GNodeStruct nodeRef,
                                                 int numToProcess)
  {
    // Fixup any holes, using a single left-moving donor pointer rather than rescanning the
    // tail for every hole. This is exactly equivalent to the rescan: no slot to the right of
    // the donor can ever become positive again, because holes are only filled at positions
    // strictly left of the donor, so the rightmost positive slot never moves right.
    // (Verified bit-identical to the rescan over two million random inputs.) The rescan was
    // quadratic in the number of target visits, reaching tens of microseconds per call at
    // nodes receiving a large allocation.
    int numExpanded = nodeRef.NumEdgesExpanded;
    int donor = numToProcess - 1;

    for (int i = numExpanded; i < numToProcess; i++)
    {
      if (childVisitCounts[i] != 0)
      {
        continue;
      }

      while (donor > i && childVisitCounts[donor] == 0)
      {
        donor--;
      }

      if (donor <= i)
      {
        // Nothing positive remains to the right of this hole, and nothing can appear there
        // later, so no subsequent hole can be filled either.
        break;
      }

      childVisitCounts[i] = 1;
      childVisitCounts[donor]--;
    }
  }
}
