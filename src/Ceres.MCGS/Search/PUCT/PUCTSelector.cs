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
using System.Buffers.Text;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Strategies;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Search.PUCT;

public static class PUCTSelector
{
  /// <summary>
  /// Internal class that holds the spans in which the child statistics are gathered.
  /// </summary>
  [ThreadStatic] static GatheredChildStats gatherStats;


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


    graph.GatherChildInfoViaChildren(node, selectorID, maxChildIndex, dualCollisionFraction, stats, refreshStaleEdges);

    // Possibly use ACPI (Action Conditional Policy Imputation)
    // to impute Q values for unvisited children based on policy and Q values of visited children.
    double[] qWhenNoChildrenComposite = null;
    if (numToProcess > 1
     && node.N > 1
     && node.NumEdgesExpanded > 0
     && paramsSelect.FPUMode == ParamsSelect.FPUType.ACPI
     )
    {
      int numExpanded = node.NumEdgesExpanded;
      string FormatRow(string label, Func<int, string> valueFunc)
      {
        IEnumerable<string> values = Enumerable.Range(0, numToProcess)
          .Select(i => valueFunc(i) + (i == numExpanded - 1 ? " |" : ""));
        return $"{label,-20} " + string.Join(" ", values);
      }

      const bool VERBOSE = false;
      if (VERBOSE)
      {
        // Compute what the default FPU value would be without imputation
        double defaultFPU = paramsSelect.CalcQWhenNoChildren(node.IsSearchRoot, node.Q, stats.SumPVisited);

        Console.WriteLine();
        Console.WriteLine($"NumEdgesExpanded = {numExpanded}, Q = {node.Q:0.00}, DefaultFPU = {defaultFPU:0.00}");
        Console.WriteLine(FormatRow("Before imputation:", i =>
          i < numExpanded && stats.N.Span[i] > 0
            ? (-stats.W.Span[i] / stats.N.Span[i]).ToString("0.00").PadLeft(5)
            : " N/A "));
      }

      qWhenNoChildrenComposite = ImputeQForUnvisitedChildren(node, stats, numToProcess);
      Debug.Assert(!double.IsNaN(qWhenNoChildrenComposite[0]));

#if DEBUG
        // Validate no NaNs in the array
        for (int i = 0; i < numToProcess; i++)
        {
          Debug.Assert(!double.IsNaN(qWhenNoChildrenComposite[i]), 
                       $"qWhenNoChildrenComposite[{i}] is NaN");
        }

        // Validate values to the right of numExpanded are strictly descending
        for (int i = numExpanded + 1; i < numToProcess; i++)
        {
          Debug.Assert(qWhenNoChildrenComposite[i] < qWhenNoChildrenComposite[i - 1],
                       $"qWhenNoChildrenComposite not strictly descending at index {i}: " +
                       $"{qWhenNoChildrenComposite[i - 1]:F4} >= {qWhenNoChildrenComposite[i]:F4}");
        }
#endif

      if (VERBOSE)
      {
        Console.WriteLine(FormatRow("After imputation:", i => qWhenNoChildrenComposite[i].ToString("0.00").PadLeft(5)));
        Console.WriteLine(FormatRow("Policy:", i => stats.P.Span[i].ToString("0.00").PadLeft(5)));
        Console.WriteLine();
      }
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
        const double VALUE_UNCERTAINTY_WEIGHT = 1.0f;
        double n = Math.Max(1, nSpan[i]);
        double adjust = (VALUE_UNCERTAINTY_WEIGHT * uncertaintyValueSpan[i]) / Math.Sqrt(n + 1);
        wSpan[i] -= adjust * nSpan[i];
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
                                                              paramsSearch.ActionHeadSelectionWeight,
                                                              thresholdPUCTSuboptimalityReject);      

      if (numTargetVisits > 0 && MCGSParamsFixed.OUT_OF_ORDER_CHILDREN_ALLOWED)
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
  ///	Returns imputed Q - values for use as FPU(First Play Urgency) estimates.
  ///
  /// Implements a Boltzmann - based Q-value imputation for unvisited child nodes in MCTS.The goal is to estimate Q - values for children that haven't been explored yet, using the policy distribution and known Q-values from visited children.
  /// Algorithm Steps:
  ///  	(1) Extract & Normalize Policy: Copies the first numToProcess policy values from p into a local array pi[] and normalizes them to sum to 1.
  ///	  (2) Find Best Child: Iterates through children to find the one with the highest Q-value(bestQ = W[i] / N[i]).
  /// 	(3) Impute Q via Boltzmann Calibration: Uses the formula:
  /// 	   Q_i = Q_anchor + tau * (log(pi_i) - log(pi_anchor))
  /// where:
  ///   -	tau = 0.10 — temperature parameter fitted from empirical data
  ///   -  anchor is the best child's negated Q (note: -q is passed, flipping perspective for minimax)
  /// </summary>
  /// 
  private static double[] ImputeQForUnvisitedChildren(GNode node, GatheredChildStats stats, int numToProcess)
  {
    double[] qWhenNoChildrenComposite;
    float[] pi = new float[numToProcess];
    float sumPi = 0;

    // Copy policy values into local array and compute sum for normalization.
    for (int i = 0; i < pi.Length; i++)
    {
      float thisPi = (float)stats.P.Span[i];
      sumPi += thisPi;
      pi[i] = (float) thisPi;
    }

    // Normalize policy values to sum to 1.
    // Tests suggested this is beneficial (approximately +6 Elo +/-10).
    for (int i = 0; i < pi.Length; i++)
    {
      pi[i] /= sumPi;
    }

    // Determine best child to use as anchor for imputation.
    int bestIndex = 0;
    double bestQ = float.MinValue;
    for (int i = 0; i < node.NumEdgesExpanded; i++)
    {
      if (stats.N.Span[i] > 0)
      {
        double thisQ = stats.W.Span[i] / stats.N.Span[i];
        if (thisQ > bestQ)
        {
          bestIndex = i;
          bestQ = thisQ;
        }
      }
    }

    // when fitted from sample data from small/medium-sized graphs we see 0.07 (low N) to 0.13 (higher N)
    const float TAU = 0.30f;

    void SetQFromChild(int childIndex, Span<float> ret)
    {
      float q = (float)(stats.W.Span[childIndex] / stats.N.Span[childIndex]);
      BoltzmannCalibration.ComputeQFromPolicy_AnchorChild(pi, childIndex, -q, TAU, ret,
                                                              renormalizeIfNeeded: false, clipToRange: true, clipMin: -1.2f, clipMax: 1.2f);
      if (float.IsNaN(ret[0]))
      {
        throw new Exception("NaN in SetQFromChild");
      }
    }

    // Using parent as anchor didn't seem to work as well
    //      double[] qWhenNoChildrenPerChildParent = new double[stats.P.Length];
    //      BoltzmannValueCalibrator.ComputeQFromPolicy_MatchParentValue(pi, node.V, TAU, qWhenNoChildrenPerChildParent);

    Span<float> qWhenNoChildrenPerBestChild = stackalloc float[stats.P.Length];
    SetQFromChild(bestIndex, qWhenNoChildrenPerBestChild);

#if NOT
    const bool ALSO_USE_CLOSEST_CHILD_AS_ANCHOR = false;
    if (ALSO_USE_CLOSEST_CHILD_AS_ANCHOR)
    {
      static float[] Avg(ReadOnlySpan<float> p0, ReadOnlySpan<float> p1)
      {
        float[] ret = new float[p1.Length];
        for (int i = 0; i < p1.Length; i++)
        {
          ret[i] = 0.5f * p0[i] + 0.5f * p1[i];
        }
        return ret;
      }

      Span<float> qWhenNoChildrenPerClosestChild = stackalloc float[stats.P.Length];
      SetQFromChild(node.NumEdgesExpanded - 1, qWhenNoChildrenPerClosestChild);
      qWhenNoChildrenComposite = Avg(qWhenNoChildrenPerBestChild, qWhenNoChildrenPerClosestChild);
    }
#endif
    double[] result = new double[qWhenNoChildrenPerBestChild.Length];
    for (int i = 0; i < qWhenNoChildrenPerBestChild.Length; i++)
    {
      result[i] = qWhenNoChildrenPerBestChild[i];
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


  /// <summary>
  /// Computes the UCT scores (used to select best child) for all children
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="node"></param>
  /// <param name="paramsSearch"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="selectorID"></param>
  /// <param name="dualCollisionFraction"></param>
  /// <param name="cpuctMultiplier"></param>
  /// <param name="temperatureMultiplier"></param>
  /// <returns></returns>
  public static double[] CalcChildScores(Graph graph, 
                                         GNode node,
                                         ParamsSearch paramsSearch, 
                                         ParamsSelect paramsSelect,
                                         int selectorID, 
                                         bool refreshStaleEdges,
                                         float dualCollisionFraction = 0.25f, 
                                         float cpuctMultiplier = 1, 
                                         float temperatureMultiplier = 1)
  {
    double[] scores = new double[node.NodeRef.NumPolicyMoves];

    ComputeTopChildScores(graph, node, paramsSearch, paramsSelect, selectorID, refreshStaleEdges, null,
                          dualCollisionFraction, 0, node.NodeRef.NumPolicyMoves - 1,
                          1, scores, default, cpuctMultiplier, temperatureMultiplier);
    return scores;
  }

}
