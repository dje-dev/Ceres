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
using System.Text;

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.RPO;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// Implements the optional CB-GPUCT selection rule using a visit-target deficit
/// against the Grill et al. reverse-KL regularized policy pi_bar.
///
/// Selection: pick the child with the largest deficit
///   deficit(a) = pi_bar(a) * (sum N_a + 1) - N_a
/// where pi_bar is the closed-form regularized policy:
///   pi_bar(a) = lambda_N * P(a) / (alpha - q(a))
/// with alpha solved by bisection so sum_a pi_bar(a) = 1.
///
/// For unvisited children, q(a) is imputed by ReverseKlPosteriorPolicy
/// (parent Q is the fallback). This guarantees pi_bar(a) > 0 for all P(a) > 0,
/// so every legal child is eventually selected (UCB1-style guarantee).
///
/// When CBGPUCT_Mode includes BackupOnly or SelectAndBackup, V_bar is propagated upward via BackupToNode
/// (overwrites node.Q; the existing edge.QChild refresh path handles propagation).
/// </summary>
internal static class CBGPUCTScoreCalc
{


  /// <summary>
  /// Computes lambda_N for the SELECTION phase (visit-target deficit pi_bar).
  /// </summary>
  /// <param name="paramsSelect"></param>
  /// <param name="sumN"></param>
  /// <param name="numChildren"></param>
  /// <returns></returns>
  internal static double ComputeLambdaNForSelection(ParamsSelect paramsSelect, double sumN, int numChildren)
    => ComputeLambdaNCore(paramsSelect.CBGPUCT_SelectLambdaSchedule,
                          paramsSelect.CBGPUCT_SelectLambdaC,
                          paramsSelect.CBGPUCT_SelectLambdaExp,
                          sumN, numChildren);


  /// <summary>
  /// Computes lambda_N for the BACKUP phase (V_bar regularized value).
  /// </summary>
  /// <param name="paramsSelect"></param>
  /// <param name="sumN"></param>
  /// <param name="numChildren"></param>
  /// <returns></returns>
  internal static double ComputeLambdaNForBackup(ParamsSelect paramsSelect, double sumN, int numChildren)
    => ComputeLambdaNCore(paramsSelect.CBGPUCT_BackupLambdaSchedule,
                          paramsSelect.CBGPUCT_BackupLambdaC,
                          paramsSelect.CBGPUCT_BackupLambdaExp,
                          sumN, numChildren);


  /// <summary>
  /// Core lambda_N formula. Selects between Pow and UCT schedules.
  /// </summary>
  /// <param name="schedule"></param>
  /// <param name="lambdaC"></param>
  /// <param name="lambdaExp"></param>
  /// <param name="sumN"></param>
  /// <param name="numChildren"></param>
  /// <returns></returns>
  private static double ComputeLambdaNCore(ParamsSelect.CBGPUCTLambdaScheduleType schedule,
                                           double lambdaC, double lambdaExp,
                                           double sumN, int numChildren)
  {
    const int DENOMINATOR_N_BASE = ParamsSelect.CGPUCT_BaseNumVisits;
    switch (schedule)
    {
      case ParamsSelect.CBGPUCTLambdaScheduleType.UCT:
        return lambdaC * Math.Sqrt(Math.Log(sumN + Math.E) / (DENOMINATOR_N_BASE + sumN));

      case ParamsSelect.CBGPUCTLambdaScheduleType.Pow:
      default:
        return sumN > 0
          ? lambdaC * Math.Pow(sumN, lambdaExp) / (DENOMINATOR_N_BASE + sumN)
          : 0;
    }
  }



  /// <summary>
  /// Visit-target child selection using the regularized policy pi_bar.
  /// Virtual loss is handled implicitly: in-flight visits count toward the
  /// effective N in the deficit formula, so collisions naturally rebalance.
  /// </summary>
  /// <param name="paramsSelect"></param>
  /// <param name="parentNode"></param>
  /// <param name="stats"></param>
  /// <param name="qParent"></param>
  /// <param name="parentSumPVisited"></param>
  /// <param name="numChildren"></param>
  /// <param name="numVisitsToCompute"></param>
  /// <param name="outputScores"></param>
  /// <param name="outputChildVisitCounts"></param>
  /// <returns>Number of visits actually accepted.</returns>
  internal static int ScoreCalc(ParamsSelect paramsSelect,
                                GNode parentNode,
                                GatheredChildStats stats,
                                double qParent, double parentSumPVisited,
                                int numChildren, int numVisitsToCompute,
                                Span<double> outputScores, Span<short> outputChildVisitCounts,
                                double[] qWhenNoChildrenPerChild = null)
  {
    Debug.Assert(numChildren > 0);
    Debug.Assert(numChildren <= PUCTScoreCalcVector.MAX_CHILDREN);

    Span<double> nEdge = stats.N.Span;
    Span<double> p = stats.P.Span;
    Span<double> w = stats.W.Span;
    Span<double> nInFlight = stats.NInFlightAdjusted.Span;

    // In CB-GPUCT graph-aware mode, edge.N is itself a (slightly stale) snapshot of the
    // destination node's total N (maintained by BackupToEdge / edge-init at expansion time).
    // So nEdge already carries the cross-parent visit count - no extra deref needed in gather.
    // In non-graph-aware mode, nEdge is the per-edge visit count as usual.
    bool graphAware = paramsSelect.CBGPUCT_GraphAwareDeficit;

    // FPU values for unvisited children: use the same machinery as vanilla PUCT
    // (Reduction / Same / Absolute / ACPI / ActionHead - whatever FPUMode is set to).
    // qWhenNoChildrenPerChild is non-null for ACPI/ActionHead (per-child values);
    // otherwise the scalar qWhenNoChildren applies to all unvisited children.
    double qWhenNoChildren = paramsSelect.CalcQWhenNoChildren(parentNode.IsSearchRoot,
                                                              qParent, parentSumPVisited);

    // Build (Q, P, U) triples for the regularized posterior bisection.
    // Q is in this node's perspective: -edge.Q for visited (since edge.Q is child perspective);
    // FPU-imputed for unvisited (already in parent perspective).
    Span<(double Q, double PriorP, int N, double U)> actions = stackalloc (double Q, double PriorP, int N, double U)[numChildren];
    Span<double> piBar = stackalloc double[numChildren];
    Span<double> qImputed = stackalloc double[numChildren];

    double sumNStart = 0;
    for (int i = 0; i < numChildren; i++)
    {
      double q;
      if (nEdge[i] == 0)
      {
        q = qWhenNoChildrenPerChild != null ? qWhenNoChildrenPerChild[i] : qWhenNoChildren;
      }
      else
      {
        q = -w[i] / nEdge[i];
      }
      actions[i] = (q, p[i], i < parentNode.NumEdgesExpanded ? parentNode.ChildEdgeAtIndex(i).N : 0, double.NaN);
      sumNStart += nEdge[i] + nInFlight[i];
    }

    // lambda_N: regularization-to-prior strength. Pow schedule peaks around sumN = |A|
    // at exp=0.5; UCT schedule decays as sqrt(log N / N).
    double lambdaN = ComputeLambdaNForSelection(paramsSelect, sumNStart, numChildren);

    ReverseKlPosteriorPolicy.Options options = new(ReverseKlPosteriorPolicy.UncertaintyMode.None,
                                                   bisectionIterations: 20);

    ReverseKlPosteriorPolicy.ComputePosterior(actions, lambdaN, lambdaQ: 0.0,
                                              rootQ: qParent, rootU: 0.0,
                                              piBar, qImputed, options);

    Span<double> scores = stackalloc double[numChildren];
    bool scoresOnly = numVisitsToCompute == 0;

    // Track effective sum incrementally: each accepted visit adds exactly 1.
    double sumEffN = sumNStart;
    int numVisits = 0;

    while (numVisits < numVisitsToCompute || (scoresOnly && numVisits == 0))
    {
      double targetTotal = sumEffN + 1;

      // Score = deficit relative to regularized target distribution.
      // In graph-aware mode, nEdge[i] is already the (slightly stale) child.N snapshot,
      // so the same expression covers both modes.
      for (int i = 0; i < numChildren; i++)
      {
        scores[i] = piBar[i] * targetTotal - (nEdge[i] + nInFlight[i]);
      }

      if (!outputScores.IsEmpty)
      {
        Debug.Assert(numVisits == 0);
        scores[..numChildren].CopyTo(outputScores);
      }

      if (scoresOnly)
      {
        return 0;
      }

      if (MCGSParamsFixed.DEBUG_CBGPUCT
          && numVisits == 0
          && numVisitsToCompute > 1
          && parentNode.IsSearchRoot)
      {
        // Find top-3 by deficit (most-under-target). Picked child = top[0].
        int t0 = -1, t1 = -1, t2 = -1;
        for (int i = 0; i < numChildren; i++)
        {
          double s = scores[i];
          if (t0 < 0 || s > scores[t0]) { t2 = t1; t1 = t0; t0 = i; }
          else if (t1 < 0 || s > scores[t1]) { t2 = t1; t1 = i; }
          else if (t2 < 0 || s > scores[t2]) { t2 = i; }
        }

        int nVisited = 0;
        for (int i = 0; i < numChildren; i++)
        {
          if (nEdge[i] > 0)
          {
            nVisited++;
          }
        }

        StringBuilder sb = new();
        sb.Append($"[CBGPUCT] root sel req={numVisitsToCompute} lambda_N={lambdaN:F4} ");
        sb.Append($"visited={nVisited}/{numChildren} ");
        sb.Append($"deficitN={(graphAware ? "child" : "edge")} top3:");

        Span<int> tops = stackalloc int[3] { t0, t1, t2 };
        for (int rank = 0; rank < 3; rank++)
        {
          int i = tops[rank];
          if (i < 0)
          {
            break;
          }
          // q display: "fpu:VALUE" for unvisited (Q from FPU imputation - same logic
          // vanilla PUCT uses for FPU), "prn" for pruned root moves (W set to
          // double.MaxValue by PUCTSelector root-pruning logic; clamped to -1
          // internally by ComputePosterior), otherwise the raw parent-perspective Q.
          string qStr;
          if (nEdge[i] == 0)
          {
            double fpuQ = qWhenNoChildrenPerChild != null
              ? qWhenNoChildrenPerChild[i]
              : qWhenNoChildren;
            qStr = $"fpu:{fpuQ:+0.000;-0.000}";
          }
          else
          {
            double rawQ = -w[i] / nEdge[i];
            qStr = Math.Abs(rawQ) > 10
              ? "prn"
              : rawQ.ToString("+0.000;-0.000");
          }
          sb.Append(rank == 0 ? " *#" : " | #");
          sb.Append(i);
          sb.Append($" P={p[i]:F4} q={qStr} pi={piBar[i]:F4} ");
          sb.Append($"tgt={piBar[i] * targetTotal:F1} ");
          // In graph-aware mode, nEdge[i] is the (stale) child.N snapshot.
          // In non-graph-aware mode, nEdge[i] is the per-edge visit count.
          sb.Append($"act={(nEdge[i] + nInFlight[i]):F1} ");
          sb.Append($"d={scores[i]:+0.000;-0.000}");
        }
        Console.WriteLine(sb.ToString());
      }

      // argmax of deficit. Always positive for at least one child in pure visit-target
      // mode (sum of targets = targetTotal > sumEffN). In graph-aware mode it can
      // become negative for all children when actual N (cross-parent total) exceeds
      // the per-parent target distribution (a transposition child has been visited
      // heavily via other parents) - in that case we stop allocating and let the
      // caller absorb the remaining visits at the parent.
      int maxIndex = 0;
      double maxScore = scores[0];
      for (int i = 1; i < numChildren; i++)
      {
        if (scores[i] > maxScore)
        {
          maxIndex = i;
          maxScore = scores[i];
        }
      }

      if (graphAware && maxScore < 0)
      {
        // No child is under its pi_bar target - stop here; remaining visits
        // (numVisitsToCompute - numVisits) will be absorbed at the parent.
        break;
      }

      nInFlight[maxIndex] += 1;
      outputChildVisitCounts[maxIndex] += 1;
      sumEffN += 1;
      numVisits += 1;
    }

    return numVisits;
  }


  /// <summary>
  /// Computes V_bar(node) via Grill reverse-KL regularized policy improvement
  /// over this node's children, then blends in the node's own network value V
  /// using the standard Ceres convention (self-V counts as 1 visit out of N total).
  /// Reuses ReverseKlPosteriorPolicy.ComputePosterior for the bisection on alpha.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="paramsSelect"></param>
  /// <returns>V_bar in this node's perspective.</returns>
  internal static double ComputeVBar(GNode node, ParamsSelect paramsSelect)
  {
    int numChildren = node.NumEdgesExpanded;
    if (numChildren == 0)
    {
      // No expanded children yet: V_bar is just the network value.
      return node.NodeRef.V;
    }

    Span<(double Q, double PriorP, int N, double U)> actions = stackalloc (double Q, double PriorP, int N, double U)[numChildren];
    Span<double> piBar = stackalloc double[numChildren];
    Span<double> qImputed = stackalloc double[numChildren];

    double sumN = 0;
    for (int i = 0; i < numChildren; i++)
    {
      GEdge edge = node.ChildEdgeAtIndex(i);
      // Edge.Q is in child perspective; negate to get this node's perspective.
      double q = edge.N == 0 ? double.NaN : -edge.Q;
      actions[i] = (q, (double)edge.P, edge.N, double.NaN);
      sumN += edge.N;
    }

    double lambdaN = ComputeLambdaNForBackup(paramsSelect, sumN, numChildren);

    ReverseKlPosteriorPolicy.Options options = new(ReverseKlPosteriorPolicy.UncertaintyMode.None,
                                                   numUncertaintyPseudovisits: paramsSelect.CGPUCT_NumUncertaintyPseudovisits,
                                                   bisectionIterations: 20);

    // Note that uncertainy mode is used (withLowNUncertainty)
    ReverseKlPosteriorPolicy.ComputePosterior(actions,
                                              lambdaN,
                                              lambdaQ: 0.0,
                                              rootQ: node.Q,
                                              rootU: 0.0,
                                              piBar,
                                              qImputed,
                                              options);

    double childContribution = 0;
    for (int i = 0; i < numChildren; i++)
    {
      childContribution += piBar[i] * qImputed[i];
    }

    // Blend self-V (counts as 1 visit) with regularized child contribution (counts as sumN visits).
    // This mirrors the Ceres convention used in standard backup (vanilla Q includes self-V via
    // the first backup) and the existing RPO reference at RPOTestsNEW.CalcRPOQ.
    int totalN = node.NodeRef.N;
    if (totalN <= 0)
    {
      return node.NodeRef.V;
    }
    return (childContribution * (totalN - 1) + node.NodeRef.V) / totalN;
  }
}
