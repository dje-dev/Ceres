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
using Ceres.MCGS.Search.RPO;
using Ceres.MCGS.Search.Strategies;

#endregion

namespace Ceres.MCGS.Search.PUCT;

public static class PUCTSelector
{
  /// <summary>
  /// Internal class that holds the spans in which the child statistics are gathered.
  /// </summary>
  [ThreadStatic] static GatheredChildStats gatherStats;

  /// <summary>
  /// Thread-local buffer for qWhenNoChildrenComposite to avoid per-call allocations.
  /// </summary>
  [ThreadStatic] static double[] qWhenNoChildrenBuffer;


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
  public static NodeSelectAccumulator ComputeTopChildScores(Graph graph, GNode node,
                                                            ParamsSearch paramsSearch, ParamsSelect paramsSelect, 
                                                            int selectorID, bool refreshStaleEdges,
                                                            MCGSFutilityPruningStatus[] rootMovePruningStatus,
                                                            float dualCollisionFraction,
                                                            int minChildIndex, int maxChildIndex, int numTargetVisits,
                                                            Span<double> scores, Span<short> childVisitCounts,
                                                            float cpuctMultiplier,
                                                            float temperatureMultiplier)
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


    graph.GatherChildInfoViaChildren(node, selectorID, maxChildIndex, dualCollisionFraction, stats, refreshStaleEdges,
                                     crossParentNActive: paramsSelect.CBGPUCT_SelectCrossParentNEnabled);

    // Possibly use action head directly
    double[] qWhenNoChildrenComposite = null;
    if (numToProcess > 1 && paramsSelect.GetFPUMode(node.IsSearchRoot) == ParamsSelect.FPUType.ActionHead)
    {
      // Use per-move value from the neural network action head as FPU for unvisited children.
      double fallbackFPU = paramsSelect.CalcQWhenNoChildren(node.IsSearchRoot, node.Q, stats.SumPVisited);
      ReadOnlySpan<double> actionSpan = stats.A.Span;
      qWhenNoChildrenComposite = qWhenNoChildrenBuffer ??= new double[PUCTScoreCalcVector.MAX_CHILDREN];
      for (int i = 0; i < numToProcess; i++)
      {
        double actionV = actionSpan[i] + MCGSStrategyPUCT.ACTION_HEAD_FPU_VALUE;

        // Negate because child Q values are stored from opponent's perspective
        qWhenNoChildrenComposite[i] = double.IsNaN(actionV) ? fallbackFPU : -actionV;
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


    if (false && paramsSearch.TestFlag)
    {
      Span<double> uncertaintyPolicySpan = stats.UP.Span;
      Span<double> uncertaintyValueSpan = stats.UV.Span;
      Span<double> nSpan = stats.N.Span;
      Span<double> wSpan = stats.W.Span;
      for (int i = 0; i < Math.Min(node.NumEdgesExpanded, numToProcess); i++)
      {
        if (!double.IsNaN(uncertaintyValueSpan[i]))
        {
          const double VALUE_UNCERTAINTY_WEIGHT = 0.3f;
          double n = Math.Max(1, nSpan[i]);
          if (n == 1)
          {
            double adjust = (VALUE_UNCERTAINTY_WEIGHT * uncertaintyValueSpan[i]) / Math.Sqrt(n);
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
      Span<double> gatherStatsNSpan = stats.N.Span;
      Span<double> gatherStatsWSpan = stats.W.Span;
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
          gatherStatsWSpan[i] = double.MaxValue;
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
      
      numVisitsAccepted = PUCTScoreCalcVector.ScoreCalcMulti(paramsSelect,
                                                              node.IsSearchRoot, nodeRef.N,
                                                              parentNumInFlight,
                                                              nodeRef.Q, sumPVisited,
                                                              stats,
                                                              qWhenNoChildrenComposite,
                                                              numToProcess, numTargetVisits,
                                                              scores, childVisitCounts, cpuctMultiplier,
                                                              thresholdPUCTSuboptimalityReject,
                                                              parentNode: node);

      // Some scoring paths can produce allocations that violate the sequential-expansion
      // invariant ("no child gets a visit before all of its left siblings have at least
      // one visit"):
      //   - ActionHead FPU: per-child q from the network is not monotonic in policy.
      //   - CB-GPUCT: the visit-target pi_bar reflects (mu, q) jointly, so cross-parent
      //     N skew, per-child FPU, and (rarely) fixed-point iteration with clamp-induced
      //     ties can produce a non-monotonic deficit ordering among unvisited children.
      // The fixup is cheap and only relocates visits when an actual hole is present.
      if (numTargetVisits > 0
          && (paramsSelect.FPUMode == ParamsSelect.FPUType.ActionHead
              || paramsSelect.CBGPUCTSelectActive))
      {
        FillInSequentialVisitHoles(childVisitCounts, ref node.NodeRef, numToProcess);
      }
    }

    // Return accumulated value across all children and also contribution from the node itself.
    double nToUse = node.Terminal.IsTerminal() ? node.N : 1;
    return new NodeSelectAccumulator(nToUse + gatherStats.SumNVisited,
                                     (nToUse * (double)nodeRef.V) + -gatherStats.SumWVisited,
                                     (nToUse * (double)nodeRef.DrawP) + gatherStats.SumDVisited,
                                     numVisitsAccepted);
  }


  /// <summary>
  /// Per-child FPU computed via the unified RegularizedPolicyOptimum primitive.
  /// Imputes parent-perspective Q for every child from the policy prior and any
  /// observed q's, using the KL direction selected by ParamsSelect.RPOFPURegularization
  /// (default ForwardKLSoftmax, matching legacy Boltzmann behavior).
  ///
  /// Anchor selection (forward-KL only; reverse-KL ignores the anchor mode):
  ///   - If the top-policy child (index 0) is visited:
  ///       MatchChild anchor with index 0, value = node.Q.
  ///       Note: this preserves a legacy quirk where the anchor index is taken to
  ///       be 0 (the top-policy child) regardless of which visited child has the
  ///       most informative Q.  See earlier dead code computing 'bestIndex' for
  ///       context.  Bug-for-bug preserved by request.
  ///   - Otherwise (top-policy child unvisited):
  ///       MatchValue anchor with value = node.Q, so E_mu[q_fill] = node.Q.
  /// </summary>
  private static double[] ApplyRPOImputedFPU(ParamsSelect paramsSelect, GNode node, GatheredChildStats stats, int numToProcess)
  {
    ReadOnlySpan<double> pSpan = stats.P.Span;
    ReadOnlySpan<double> nSpan = stats.N.Span;
    ReadOnlySpan<double> wSpan = stats.W.Span;

    int numExpanded = node.NumEdgesExpanded;

    // Build mu (normalization happens inside Solve), and q with NaN for unvisited children.
    Span<double> mu = stackalloc double[numToProcess];
    Span<double> qIn = stackalloc double[numToProcess];
    for (int i = 0; i < numToProcess; i++)
    {
      mu[i] = pSpan[i];
      qIn[i] = (i < numExpanded && nSpan[i] > 0) ? -wSpan[i] / nSpan[i] : double.NaN;
    }

    // Anchor choice mirrors legacy ApplyRPOImputedFPU.  The reverse-KL path ignores
    // the anchor (it must be None there); construct the appropriate anchor for forward KL.
    RPORegularization regularization = paramsSelect.RPOFPURegularization;
    RPOAnchor anchor = regularization == RPORegularization.ReverseKL
      ? RPOAnchor.None
      : (nSpan[0] > 0
          ? new RPOAnchor(RPOAnchorMode.MatchChild, 0, node.Q)
          : new RPOAnchor(RPOAnchorMode.MatchValue, -1, node.Q));

    double lambda = paramsSelect.PolicyImputationTau;
    RPOOptions opts = new(bisectionIterations: 12,
                          bisectionResidualTol: 1e-6,
                          clampQToUnitInterval: true,
                          minPriorProbability: 0.0);

    double[] result = qWhenNoChildrenBuffer ??= new double[PUCTScoreCalcVector.MAX_CHILDREN];
    Span<double> resultSpan = result.AsSpan(0, numToProcess);

    RegularizedPolicyOptimum.Solve(mu, qIn, lambda, anchor, regularization,
                                   yOut: default,
                                   qFillOut: resultSpan,
                                   out double _,
                                   options: opts,
                                   nanFallbackQ: node.Q);

    // Final clamp to [-1, 1] to match legacy effective output range.  The legacy
    // Boltzmann code internally clipped to [-1.2, 1.2] to leave headroom for a
    // descending-epsilon hack that is now obsolete (sequential expansion ordering
    // is preserved through P-ordering of children).
    for (int i = 0; i < numToProcess; i++)
    {
      double v = result[i];
      if (v < -1.0) result[i] = -1.0;
      else if (v > 1.0) result[i] = 1.0;
    }

    // Cap Q values for unexpanded children to not exceed defaultFPU + 0.30.
    double defaultFPU = paramsSelect.CalcQWhenNoChildren(node.IsSearchRoot, node.Q, stats.SumPVisited);
    double maxQ = 0.30 + defaultFPU;
    for (int i = numExpanded; i < numToProcess; i++)
    {
      if (result[i] > maxQ)
      {
        result[i] = maxQ;
      }
    }

    if (CBGPUCTDumpDiagnostics.DEBUG_DUMP_FPU_CALCS)
    {
      CBGPUCTDumpDiagnostics.DumpFPURPO(node, pSpan, nSpan, wSpan, resultSpan,
                                        numToProcess, numExpanded,
                                        lambda, regularization, anchor, defaultFPU);
    }

    return result;
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
    // Fixup any holes
    int numExpanded = nodeRef.NumEdgesExpanded;
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
}
