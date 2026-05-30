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

using Ceres.Chess.EncodedPositions;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.RPO;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// Thin pass-through that adapts CB-GPUCT call sites to the unified RPO primitives:
///   - ScoreCalc dispatches to RegularizedPolicyOptimum.Solve plus RPOVisitAllocator.Allocate.
///   - ComputeVBar dispatches to RegularizedPolicyOptimum.Solve.
/// The lambda_N schedule helpers stay here since they are CB-GPUCT-specific.
///
/// See "Monte-Carlo Tree Search as Regularized Policy Optimization" (Grill et al. 2020)
/// for the underlying regularized policy improvement framework.
/// </summary>
internal static class CBGPUCTScoreCalc
{
  /// <summary>
  /// Computes lambda_N for the SELECTION phase (visit-target deficit pi_bar).  The
  /// effective coefficient that scales the base schedule grows with sumN via the
  /// CBGPUCT_SelectLambdaCLogFactor knob - directly paralleling Ceres-PUCT's CPUCT
  /// log term.  Pass mu to enable per-node resolution of the
  /// NumMovesWithPolicyOver5Pct denominator option; an empty span falls back to
  /// using numChildren in its place.
  /// </summary>
  internal static double ComputeLambdaNForSelection(ParamsSelect paramsSelect, double sumN, int numChildren,
                                                    ReadOnlySpan<double> mu = default)
    => ComputeLambdaNCore(paramsSelect.CBGPUCT_SelectLambdaSchedule,
                          paramsSelect.CBGPUCT_SelectLambdaC,
                          paramsSelect.CBGPUCT_SelectLambdaExp,
                          ResolveDenomBase(paramsSelect.CBGPUCT_SelectLambdaDenominatorBase, mu, numChildren),
                          paramsSelect.CBGPUCT_SelectLambdaCLogBase,
                          paramsSelect.CBGPUCT_SelectLambdaCLogFactor,
                          sumN, numChildren);


  /// <summary>
  /// Computes lambda_N for the BACKUP phase (V_bar regularized value).  No log growth
  /// is applied here - backup computes a regularized value rather than an exploration
  /// policy, so the PUCT-style coefficient scaling does not apply.  Pass mu to enable
  /// per-node resolution of the NumMovesWithPolicyOver5Pct denominator option; an empty
  /// span falls back to using numChildren in its place.
  /// </summary>
  internal static double ComputeLambdaNForBackup(ParamsSelect paramsSelect, double sumN, int numChildren,
                                                 ReadOnlySpan<double> mu = default)
    => ComputeLambdaNCore(paramsSelect.CBGPUCT_BackupLambdaSchedule,
                          paramsSelect.CBGPUCT_BackupLambdaC,
                          paramsSelect.CBGPUCT_BackupLambdaExp,
                          ResolveDenomBase(paramsSelect.CBGPUCT_BackupLambdaDenominatorBase, mu, numChildren),
                          cLogBase: 0.0, cLogFactor: 0.0,
                          sumN, numChildren);


  /// <summary>
  /// Resolves the per-phase enum denom-base choice to a numeric value used in the
  /// lambda_N schedule denominator.  The NumMovesWithPolicyOver5Pct option counts
  /// children with policy P > 5%, producing an effective branching factor that ignores
  /// low-policy tail moves; an empty mu span (e.g. from diagnostic call sites that
  /// don't have policy in scope) falls back to numChildren.  The returned value is
  /// always at least 1.0.
  /// </summary>
  private static double ResolveDenomBase(ParamsSelect.CBGPUCTLambdaDenominatorBaseType type,
                                         ReadOnlySpan<double> mu, int numChildren)
  {
    switch (type)
    {
      case ParamsSelect.CBGPUCTLambdaDenominatorBaseType.One: return 1.0;
      case ParamsSelect.CBGPUCTLambdaDenominatorBaseType.Three: return 3.0;
      case ParamsSelect.CBGPUCTLambdaDenominatorBaseType.Five: return 5.0;
      case ParamsSelect.CBGPUCTLambdaDenominatorBaseType.Ten: return 10.0;
      case ParamsSelect.CBGPUCTLambdaDenominatorBaseType.Thirty: return 30.0;
      case ParamsSelect.CBGPUCTLambdaDenominatorBaseType.NumMovesWithPolicyOver5Pct:
        if (mu.IsEmpty)
        {
          return Math.Max(1, numChildren);
        }
        int count = 0;
        int limit = Math.Min(mu.Length, numChildren);
        for (int i = 0; i < limit; i++)
        {
          if (mu[i] > 0.05)
          {
            count++;
          }
        }
        return Math.Max(1, count);
      default: return 1.0;
    }
  }


  /// <summary>
  /// Core lambda_N formula.  Dispatches across the schedule types and uses an effective
  /// coefficient that may grow with sumN according to
  ///   c(sumN) = lambdaC + cLogFactor * log((sumN + cLogBase + 1) / cLogBase)
  /// which exactly mirrors Ceres-PUCT's CPUCT log term (CPUCT + CPUCTFactor * log((N+Base+1)/Base)).
  /// Pass cLogFactor = 0 (or cLogBase = 0) to disable the log growth.  denominatorBase and
  /// lambdaExp are interpreted differently per schedule; see the enum docs.
  /// </summary>
  private static double ComputeLambdaNCore(ParamsSelect.CBGPUCTLambdaScheduleType schedule,
                                           double lambdaC, double lambdaExp,
                                           double denominatorBase,
                                           double cLogBase, double cLogFactor,
                                           double sumN, int numChildren)
  {
    double cEffective = lambdaC;
    if (cLogFactor != 0.0 && cLogBase > 0.0)
    {
      cEffective = lambdaC + cLogFactor * Math.Log((sumN + cLogBase + 1.0) / cLogBase);
    }

    switch (schedule)
    {
      case ParamsSelect.CBGPUCTLambdaScheduleType.UCT:
        return cEffective * Math.Sqrt(Math.Log(sumN + Math.E) / (denominatorBase + sumN));

      case ParamsSelect.CBGPUCTLambdaScheduleType.Pow:
        // Plain inverse-power: cEff / pow(sumN, exp).  Diverges at sumN == 0; return 0
        // there (no children visited yet, nothing to regularize).
        return sumN > 0
          ? cEffective / Math.Pow(sumN, lambdaExp)
          : 0;

      case ParamsSelect.CBGPUCTLambdaScheduleType.Log:
        // Slow 1/log decay.  log(sumN + e) >= 1 for all sumN >= 0, so always safe.
        return cEffective / Math.Log(sumN + Math.E);

      case ParamsSelect.CBGPUCTLambdaScheduleType.AlphaZero:
      default:
        return sumN > 0
          ? cEffective * Math.Pow(sumN, lambdaExp) / (denominatorBase + sumN)
          : 0;
    }
  }


  /// <summary>
  /// Lower bound on pi_bar entries.  Matches the engine-wide minimum policy probability
  /// guaranteed for any legal move (DEFAULT_MIN_PROBABILITY_LEGAL_MOVE = 0.0005, i.e.
  /// 0.05%).  Selecting the same value means post-floor pi_bar is never below what the
  /// engine already considers the floor for a legal move's raw policy.
  /// </summary>
  private const double PI_BAR_FLOOR = CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE;


  /// <summary>
  /// Computes the base anchor value for per-child Q imputation, dispatched by anchor
  /// type.  Both PUCTSelector.ApplyRPOImputedFPU (FPU) and
  /// CBGPUCTScoreCalc.ComputeVBar (backup) call this helper so the anchor semantics
  /// stay aligned across the two imputation sites.
  ///
  /// qRaw is the per-child observed Q in PARENT perspective (i.e. -edge.Q for visited
  /// children, NaN for unvisited).  Callers that have qRaw in a different form should
  /// build it once and pass the same span here.
  /// </summary>
  internal static double ComputeImputationAnchor(ParamsSelect.CBGPUCTQAnchorType type,
                                                 GNode node,
                                                 ReadOnlySpan<double> qRaw, int numChildren)
  {
    switch (type)
    {
      case ParamsSelect.CBGPUCTQAnchorType.ParentV:
        return node.NodeRef.V;

      case ParamsSelect.CBGPUCTQAnchorType.ParentQ:
        return node.Q;

      case ParamsSelect.CBGPUCTQAnchorType.FirstChildElseParentQ:
        return (numChildren > 0 && !double.IsNaN(qRaw[0])) ? qRaw[0] : node.Q;

      case ParamsSelect.CBGPUCTQAnchorType.BestChildElseParentQ:
      case ParamsSelect.CBGPUCTQAnchorType.BlendBestChildParentQ:
        {
          double best = double.NegativeInfinity;
          for (int i = 0; i < numChildren; i++)
          {
            double q = qRaw[i];
            if (!double.IsNaN(q) && q > best)
            {
              best = q;
            }
          }
          if (double.IsNegativeInfinity(best))
          {
            return node.Q;
          }
          return type == ParamsSelect.CBGPUCTQAnchorType.BestChildElseParentQ
            ? best
            : 0.5 * (best + node.Q);
        }

      default:
        return node.Q;
    }
  }

  /// <summary>
  /// Cumulative excess mass (from flooring up) above which we rescale the above-floor
  /// values to preserve sum-to-1.  Below this threshold we leave the distribution
  /// slightly above 1 - the cost in the deficit allocator is negligible (IterativeLargestDeficit
  /// is invariant under uniform pi_bar scaling for argmax; HamiltonClosedForm has a
  /// sub-visit-fraction skew that's within tie-breaking noise).
  /// </summary>
  private const double PI_BAR_RENORMALIZE_THRESHOLD = 0.01;


  /// <summary>
  /// Applies a minimum floor to pi_bar so that no child is functionally starved when its
  /// imputed Q and policy are jointly low.  Two passes: first counts how many entries are
  /// below the floor and the mass they hold; second applies the floor (and conditionally
  /// rescales above-floor entries) based on the excess introduced.  Fast path when no entries
  /// fall below the floor returns after a single comparison pass with no writes.
  /// </summary>
  private static void ApplyPiBarFloor(Span<double> piBar, int n)
  {
    if (n <= 0 || n * PI_BAR_FLOOR >= 1.0)
    {
      // Floor too aggressive to fit n children; bail out (caller's pi_bar unchanged).
      return;
    }

    double belowSum = 0.0;
    int belowCount = 0;
    for (int i = 0; i < n; i++)
    {
      double v = piBar[i];
      if (v < PI_BAR_FLOOR)
      {
        belowSum += v;
        belowCount++;
      }
    }
    if (belowCount == 0)
    {
      return;   // fast path: nothing below floor
    }

    double excessMass = belowCount * PI_BAR_FLOOR - belowSum;
    bool renormalize = excessMass > PI_BAR_RENORMALIZE_THRESHOLD;
    if (renormalize)
    {
      double targetAbove = 1.0 - belowCount * PI_BAR_FLOOR;
      double currentAbove = 1.0 - belowSum;
      double scaleAbove = currentAbove > 0.0 ? targetAbove / currentAbove : 0.0;
      for (int i = 0; i < n; i++)
      {
        if (piBar[i] < PI_BAR_FLOOR)
        {
          piBar[i] = PI_BAR_FLOOR;
        }
        else
        {
          piBar[i] *= scaleAbove;
        }
      }
    }
    else
    {
      // Cheap path: raise entries below the floor without rescaling the others.  The
      // sum slips to slightly above 1, but neither allocator is materially affected at
      // this excess level (see PI_BAR_RENORMALIZE_THRESHOLD docstring).
      for (int i = 0; i < n; i++)
      {
        if (piBar[i] < PI_BAR_FLOOR)
        {
          piBar[i] = PI_BAR_FLOOR;
        }
      }
    }
  }


  /// <summary>
  /// Computes the per-child "policy-implied Q" by running the FPU-style RPO imputation
  /// with all q inputs treated as unknown.  Uses the same machinery as PolicyImputedRPO
  /// FPU (RPOFPURegularization, PolicyImputationTau, MatchValue anchor at referenceQ)
  /// but forces every entry to NaN so the result reflects only the policy prior, not
  /// any observed q.  Used as the per-child shrinkage target for Q-shrinkage: for a
  /// 0.2%-policy move, the implied Q is low (the network thinks it's bad), so noisy
  /// observed q's get pulled toward that pessimistic prior rather than toward the
  /// parent's average Q.
  /// </summary>
  private static void ComputePolicyImpliedQ(ParamsSelect paramsSelect,
                                            ReadOnlySpan<double> mu, double referenceQ,
                                            Span<double> output, int n)
  {
    Span<double> qNaN = stackalloc double[n];
    for (int i = 0; i < n; i++)
    {
      qNaN[i] = double.NaN;
    }

    RPORegularization regularization = paramsSelect.RPOFPURegularization;
    // ReverseKL uses no anchor (level set via nanFallbackQ); forward-KL family uses
    // MatchValue anchor so E_y[q_fill] = referenceQ holds.
    RPOAnchor anchor = regularization == RPORegularization.ReverseKL
      ? RPOAnchor.None
      : new RPOAnchor(RPOAnchorMode.MatchValue, -1, referenceQ);

    double lambda = paramsSelect.PolicyImputationTau;
    RPOOptions opts = new(bisectionIterations: 12,
                          bisectionResidualTol: 1e-6,
                          clampQToUnitInterval: true,
                          minPriorProbability: 0.0);

    RegularizedPolicyOptimum.Solve(mu, qNaN, lambda, anchor, regularization,
                                   yOut: default,
                                   qFillOut: output,
                                   out double _,
                                   options: opts,
                                   nanFallbackQ: referenceQ);
  }


  /// <summary>
  /// CB-GPUCT visit-target child selection.  Builds the (mu, q) inputs from gathered
  /// child stats (q pre-filled with FPU values for unvisited children, matching legacy
  /// behavior), solves for pi_bar via RegularizedPolicyOptimum.Solve, then apportions
  /// the visit budget across children via RPOVisitAllocator.Allocate.
  ///
  /// Virtual loss is handled implicitly: in-flight visits count toward currentN so
  /// collisions naturally rebalance.
  /// </summary>
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
    Span<double> pSpan = stats.P.Span;
    Span<double> wSpan = stats.W.Span;
    Span<double> nInFlight = stats.NInFlightAdjusted.Span;

    // FPU for unvisited children: either per-child (from ActionHead / PolicyImputed)
    // or scalar fallback from CalcQWhenNoChildren.
    double qWhenNoChildren = paramsSelect.CalcQWhenNoChildren(parentNode.IsSearchRoot,
                                                              qParent, parentSumPVisited);

    // Build (mu, q, currentN).  nEdge[i] is always the per-edge visit count (visits
    // taken along this parent's edge to the child).  When CBGPUCT_CrossParentNFraction
    // is positive, we additionally fold in a fraction of the cross-parent surplus
    // (child.N minus per-edge.N) so that transposition children visited heavily via
    // other parents are partially counted toward this parent's visit target.
    float crossParentFraction = paramsSelect.CBGPUCT_SelectCrossParentNFraction;
    Span<double> mu = stackalloc double[numChildren];
    Span<double> qIn = stackalloc double[numChildren];
    Span<double> currentN = stackalloc double[numChildren];
    for (int i = 0; i < numChildren; i++)
    {
      mu[i] = pSpan[i];
      if (nEdge[i] == 0)
      {
        qIn[i] = qWhenNoChildrenPerChild != null ? qWhenNoChildrenPerChild[i] : qWhenNoChildren;
      }
      else
      {
        qIn[i] = -wSpan[i] / nEdge[i];
      }
      double effectiveN = nEdge[i];
      // Apply the cross-parent surplus blend only to edges that have at least one direct
      // visit.  If nEdge[i] == 0 while childN is large (transposition heavily visited via
      // OTHER parents), inflating effectiveN by crossParentFraction * childN would push the
      // deficit negative indefinitely whenever childN grows in step with sumN; combined with
      // StopWhenAllOverQuota that starves the edge forever.  Keeping effectiveN = 0 for
      // unvisited edges lets the PI_BAR_FLOOR * (sumN+1) target produce a positive deficit
      // so the first visit arrives promptly.
      if (crossParentFraction > 0.0f && nEdge[i] > 0 && i < parentNode.NumEdgesExpanded)
      {
        GEdge edge = parentNode.ChildEdgeAtIndex(i);
        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
        {
          double childN = edge.ChildNode.NodeRef.N;
          double surplus = childN - nEdge[i];
          if (surplus > 0.0)
          {
            effectiveN += crossParentFraction * surplus;
          }
        }
      }
      currentN[i] = effectiveN + nInFlight[i];
    }

    // Snapshot qIn before any modification (shrinkage / fixed-point) so the optional
    // select diagnostic can show the raw -W/N (for visited) and original FPU (for
    // unvisited) values alongside the final qIn that goes into Solve.  Cost is negligible
    // (at most 64 doubles stack-copied); when the diagnostic gate is false the snapshot is
    // simply unused.
    Span<double> qInOriginal = stackalloc double[numChildren];
    qIn.CopyTo(qInOriginal);

    // Support-shrinkage of visited children's q toward the policy-shaped empirical-Bayes
    // consensus prior - the select-phase analogue of the backup ComputeVBar mechanism.
    // The select q is the per-edge average (-W/N), so its statistical support is edge.N;
    // visited children are shrunk toward the consensus by edge.N, while unvisited children
    // keep their FPU value (the exploration signal) untouched.  Disabled when K == 0.
    float selectShrinkK = paramsSelect.CBGPUCT_SelectSupportShrinkageK;
    if (selectShrinkK > 0.0f)
    {
      // Consensus q_bar over visited children, weighted per CBGPUCT_ConsensusWeight.  In
      // select the q is the per-edge average whose support is edge.N (already transposition-
      // free), so the saturating gate is NOT applied here: the policy-family modes (Policy,
      // PolicyChildNSaturating) use mu, and the ChildN-family modes (ChildN, EdgeN,
      // ChildNSaturating) all resolve to edge.N.  Falls back to qParent if none visited.
      var consensusMode = paramsSelect.CBGPUCT_ConsensusWeight;
      bool policyWeight = consensusMode == ParamsSelect.CBGPUCTConsensusWeightType.Policy
                       || consensusMode == ParamsSelect.CBGPUCTConsensusWeightType.PolicyChildNSaturating;
      double sumW = 0.0;
      double sumWQ = 0.0;
      for (int i = 0; i < numChildren; i++)
      {
        if (nEdge[i] > 0.0)
        {
          double w = policyWeight ? mu[i] : nEdge[i];
          sumW += w;
          sumWQ += w * qIn[i];
        }
      }
      double consensusQ = sumW > 0.0 ? sumWQ / sumW : qParent;

      // Policy-shaped prior anchored at the consensus (reuses the FPU forward-KL imputation).
      Span<double> priorSelect = stackalloc double[numChildren];
      ComputePolicyImpliedQ(paramsSelect, mu, consensusQ, priorSelect, numChildren);

      double decayExpSelect = paramsSelect.CBGPUCT_SelectSupportShrinkageDecayExponent;
      for (int i = 0; i < numChildren; i++)
      {
        double nI = nEdge[i];
        if (nI > 0.0)
        {
          double nPow = decayExpSelect == 1.0 ? nI : Math.Pow(nI, decayExpSelect);
          double precision = nPow / (nPow + selectShrinkK);
          qIn[i] = qIn[i] * precision + priorSelect[i] * (1.0 - precision);
        }
      }
    }

    // sumN at the start of the batch drives the lambda_N schedule.
    double sumNStart = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      sumNStart += currentN[i];
    }
    double lambdaN = ComputeLambdaNForSelection(paramsSelect, sumNStart, numChildren, mu);

    // Solve for pi_bar.  q is fully filled (no NaN); anchor is ignored for reverse KL.
    Span<double> piBar = stackalloc double[numChildren];
    RPOOptions opts = new(bisectionIterations: 20,
                          bisectionResidualTol: 1e-6,
                          clampQToUnitInterval: true,
                          minPriorProbability: 0.0);
    RPOAnchor anchor = paramsSelect.RPOSelectRegularization == RPORegularization.ReverseKL
      ? RPOAnchor.None
      : new RPOAnchor(RPOAnchorMode.MatchValue, -1, qParent);

    RegularizedPolicyOptimum.Solve(mu, qIn, lambdaN, anchor, paramsSelect.RPOSelectRegularization,
                                   yOut: piBar,
                                   qFillOut: default,
                                   out double vStar,
                                   options: opts,
                                   nanFallbackQ: qParent);

    // Robustness floor on pi_bar.  Combining low policy (mu) with a low imputed Q
    // (e.g., PolicyImputed FPU pulling unvisited Q way down for low-mu children)
    // effectively double-counts the policy signal and can produce pi_bar values low
    // enough to functionally starve those children of visits.  Floor pi_bar to the
    // same minimum the engine guarantees for any legal move's raw policy probability.
    ApplyPiBarFloor(piBar, numChildren);

    // Apportion the visit budget across children.
    // Early-break when all children are over their visit target is only meaningful
    // when cross-parent N is being folded in (otherwise the deficit cannot saturate).
    RPOAllocationOptions allocOpts = new(paramsSelect.RPOSelectAllocator,
                                          stopWhenAllOverQuota: paramsSelect.CBGPUCT_SelectCrossParentNEnabled);

    // Score-only mode: write deficits and return without touching nInFlight.
    bool scoresOnly = numVisitsToCompute == 0;
    if (scoresOnly)
    {
      Span<short> dummyVisits = stackalloc short[numChildren];
      RPOVisitAllocator.Allocate(piBar, currentN, budget: 0,
                                 visitsAddedOut: dummyVisits,
                                 firstStepDeficitsOut: outputScores,
                                 allocOpts);
      return 0;
    }

    // Optional diagnostic: pull the first-step deficits into outputScores for the
    // caller, then drive allocation.  Visit-and-allocate path requires outputScores
    // to be empty when numVisitsToCompute > 1 (asserted upstream), so use a private
    // scratch when caller did not ask for scores.
    Span<double> firstStepDeficits = !outputScores.IsEmpty ? outputScores : stackalloc double[numChildren];
    int placed = RPOVisitAllocator.Allocate(piBar, currentN, numVisitsToCompute,
                                            visitsAddedOut: outputChildVisitCounts,
                                            firstStepDeficitsOut: firstStepDeficits,
                                            allocOpts);

    // Diagnostic: log top-3 deficits at the root.
    if (MCGSParamsFixed.DEBUG_CBGPUCT
        && numVisitsToCompute > 1
        && parentNode.IsSearchRoot)
    {
      LogTop3Deficits(parentNode, paramsSelect, nEdge, nInFlight, pSpan, wSpan,
                      qWhenNoChildrenPerChild, qWhenNoChildren, piBar, firstStepDeficits,
                      numChildren, numVisitsToCompute, lambdaN);
    }

    if (CBGPUCTDumpDiagnostics.DEBUG_DUMP_CBGPUCT_SELECT_CALCS)
    {
      // Gather each child's child.N (the child node's total visit count) for the diagnostic.
      // In graph search this can exceed the per-edge nEdge when the child is reached via OTHER
      // parents (transpositions); it is the statistical support behind the child's Q and the
      // quantity the cross-parent blend (CBGPUCT_SelectCrossParentNFraction) folds into currentN.
      // Terminal edges have no child node, so their support is edge.N; unexpanded slots are 0.
      Span<double> childN = stackalloc double[numChildren];
      int numExpandedForDump = parentNode.NumEdgesExpanded;
      for (int i = 0; i < numChildren; i++)
      {
        if (i < numExpandedForDump)
        {
          GEdge edge = parentNode.ChildEdgeAtIndex(i);
          childN[i] = edge.Type == GEdgeStruct.EdgeType.ChildEdge
            ? edge.ChildNode.NodeRef.N
            : nEdge[i];
        }
        else
        {
          childN[i] = 0;
        }
      }

      CBGPUCTDumpDiagnostics.DumpCBGPUCTSelect(paramsSelect, parentNode, pSpan, qInOriginal, qIn, piBar,
                                               firstStepDeficits, currentN, nEdge, childN, nInFlight,
                                               outputChildVisitCounts,
                                               numChildren, numVisitsToCompute, lambdaN, sumNStart,
                                               vStar, paramsSelect.RPOSelectRegularization);
    }

    // Fold allocator's visits back into nInFlight (allocator is pure; caller does this).
    for (int i = 0; i < numChildren; i++)
    {
      nInFlight[i] += outputChildVisitCounts[i];
    }

    return placed;
  }


  /// <summary>
  /// Minimum raw network policy P for an unvisited child (either expanded with edge.N==0
  /// or unexpanded entirely) to be included in the V_bar dot-product.  
  /// Expanded children at indices [0, NumEdgesExpanded) are always included regardless of P; 
  /// unexpanded children at indices [NumEdgesExpanded, NumPolicyMoves) 
  /// are included only when header.P meets this threshold.  
  /// Set to float.MaxValue to compute V_bar is computed over the expanded children only.
  /// </summary>
  private const float MIN_P_FOR_Q_IF_UNVISITED = 0.01f;


  /// <summary>
  /// Computes V_bar(node): the support-shrinkage regularized backup value.  Forms ONE
  /// posterior-mean Q estimate per child,
  ///   q_hat(a) = (N_a^p * q_obs(a) + K * m_a) / (N_a^p + K)   (visited),  m_a (unvisited),
  /// and uses it consistently for both the pi_bar weights and the V_bar dot product, then
  /// blends in the node's own network value V (self-V counts as 1 visit out of N total).
  /// N_a = child.N (statistical support; cross-parent in graph mode), m_a = policy-shaped
  /// empirical-Bayes prior anchored at the support-weighted child consensus q_bar (node.V
  /// when none visited), K = CBGPUCT_BackupSupportShrinkageK, p =
  /// CBGPUCT_BackupSupportShrinkageDecayExponent.  K == 0 disables shrinkage (plain Grill
  /// V_bar with policy-imputed unvisited children).
  ///
  /// Action-set coverage: always includes every expanded child (i &lt; NumEdgesExpanded),
  /// plus any unexpanded child whose header.P is at least MIN_P_FOR_Q_IF_UNVISITED.
  /// Unvisited slots take the prior m_a directly (the N_a = 0 limit).
  /// </summary>
  internal static double ComputeVBar(GNode node, ParamsSelect paramsSelect)
  {
    int numExpanded = node.NumEdgesExpanded;
    if (numExpanded == 0)
    {
      // No expanded children yet: V_bar is just the network value (same as legacy).
      return node.NodeRef.V;
    }

    // Considered action set: all expanded children plus any unexpanded child whose raw
    // policy P meets MIN_P_FOR_Q_IF_UNVISITED (identical coverage rule to the legacy path).
    int numChildren = numExpanded;
    int numPolicyMoves = node.NumPolicyMoves;
    if (MIN_P_FOR_Q_IF_UNVISITED < 1.0f && numPolicyMoves > numExpanded)
    {
      Span<GEdgeHeaderStruct> coverageHeaders = node.EdgeHeadersSpan;
      int maxKeptIndex = numExpanded - 1;
      for (int i = numExpanded; i < numPolicyMoves; i++)
      {
        if ((float)coverageHeaders[i].P >= MIN_P_FOR_Q_IF_UNVISITED)
        {
          maxKeptIndex = i;
        }
      }
      numChildren = maxKeptIndex + 1;
    }

    Span<double> mu = stackalloc double[numChildren];        // prior policy
    Span<double> qRaw = stackalloc double[numChildren];      // observed Q (parent persp), NaN if unvisited
    Span<double> edgeNSpan = stackalloc double[numChildren]; // per-edge visits (lambda schedule + diagnostics)
    Span<double> nSupport = stackalloc double[numChildren];  // child.N: statistical support of qRaw

    // First pass: gather mu / qRaw / edge.N / child.N (mirrors the legacy first pass).
    // sumEdgeN feeds the lambda_N schedule (kept on per-edge N for comparability); the
    // shrinkage and the consensus use child.N (the true support of each Q estimate).
    double sumEdgeN = 0.0;
    Span<GEdgeHeaderStruct> headersSpan = node.EdgeHeadersSpan;
    for (int i = 0; i < numChildren; i++)
    {
      if (i < numExpanded)
      {
        GEdge edge = node.ChildEdgeAtIndex(i);
        mu[i] = edge.P;
        int edgeN = edge.N;
        edgeNSpan[i] = edgeN;
        qRaw[i] = edgeN == 0 ? double.NaN : -edge.Q;   // child perspective -> parent perspective
        sumEdgeN += edgeN;
        nSupport[i] = edge.Type == GEdgeStruct.EdgeType.ChildEdge
          ? edge.ChildNode.NodeRef.N
          : edgeN;                                     // terminal edges: Q exact, use edge.N
      }
      else
      {
        mu[i] = (double)headersSpan[i].P;
        edgeNSpan[i] = 0;
        qRaw[i] = double.NaN;
        nSupport[i] = 0;
      }
    }

    // Empirical-Bayes anchor: the weighted consensus of the visited children,
    // q_bar = sum_a w_a q_a / sum_a w_a, falling back to the node's own network value
    // when nothing is visited.  Anchoring the shrinkage target on what search has
    // actually found - rather than on the possibly stale/optimistic parent value - is
    // what lets a single global K stay unbiased when the parent and the child evidence
    // disagree (no anchorK blend needed).  The weight basis w_a is selectable via
    // CBGPUCT_ConsensusWeight: child.N (precision pooling), edge.N (local,
    // transposition-free), mu (policy), or the saturating-gate forms child.N/(child.N+Kc)
    // [+ mu], which cap the influence of any single high-N child (the principled
    // finite-between-move-variance estimator; see the CBGPUCT_ConsensusWeight docs).
    // NOTE: only the consensus WEIGHT varies; the shrinkage PRECISION below always uses
    // child.N (nSupport) - the true reliability of each q, which transpositions sharpen
    // rather than bias.
    var consensusMode = paramsSelect.CBGPUCT_ConsensusWeight;
    double consensusKc = paramsSelect.CBGPUCT_ConsensusReliabilityK;
    double sumSupport = 0.0;
    double sumSupportQ = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      if (!double.IsNaN(qRaw[i]) && nSupport[i] > 0.0)
      {
        // Saturating reliability gate g = N/(N+Kc) in [0,1): -> N/Kc (proportional to
        // precision) at low N, -> 1 (saturated) at high N.  Kc <= 0 disables it (g = 1).
        double gate = consensusKc > 0.0 ? nSupport[i] / (nSupport[i] + consensusKc) : 1.0;
        double w = consensusMode switch
        {
          ParamsSelect.CBGPUCTConsensusWeightType.EdgeN => edgeNSpan[i],
          ParamsSelect.CBGPUCTConsensusWeightType.Policy => mu[i],
          ParamsSelect.CBGPUCTConsensusWeightType.ChildNSaturating => gate,
          ParamsSelect.CBGPUCTConsensusWeightType.PolicyChildNSaturating => mu[i] * gate,
          _ => nSupport[i],   // ChildN (default)
        };
        sumSupport += w;
        sumSupportQ += w * qRaw[i];
      }
    }
    double consensusQ = sumSupport > 0.0 ? sumSupportQ / sumSupport : node.NodeRef.V;

    // Policy-shaped prior m_a anchored at the consensus.  Reuses the exact forward-KL
    // imputation used by the FPU path (tau = PolicyImputationTau): low-policy children
    // get a lower prior, high-policy children a higher one, with E_mu[m] = consensusQ.
    Span<double> prior = stackalloc double[numChildren];
    ComputePolicyImpliedQ(paramsSelect, mu, consensusQ, prior, numChildren);

    // Posterior-mean Q estimate q_hat, used CONSISTENTLY below for both pi_bar and V_bar.
    // Visited:   q_hat = (N^p q_obs + K m) / (N^p + K)   (conjugate / James-Stein shrinkage).
    // Unvisited: q_hat = m                               (the N = 0 limit; no special case).
    float kShrink = paramsSelect.CBGPUCT_BackupSupportShrinkageK;
    double decayExp = paramsSelect.CBGPUCT_BackupSupportShrinkageDecayExponent;
    Span<double> qHat = stackalloc double[numChildren];
    for (int i = 0; i < numChildren; i++)
    {
      if (double.IsNaN(qRaw[i]))
      {
        qHat[i] = prior[i];
      }
      else
      {
        double nPow = decayExp == 1.0 ? nSupport[i] : Math.Pow(nSupport[i], decayExp);
        double precision = nPow / (nPow + kShrink);
        qHat[i] = qRaw[i] * precision + prior[i] * (1.0 - precision);
      }
    }

    // pi_bar from the denoised q_hat.  q_hat has no NaN entries, so Solve performs no
    // imputation; qFill echoes q_hat (clamped to [-1, 1]) and is what the weights were
    // actually computed against, so we use it in the dot product for exact consistency.
    double lambdaN = ComputeLambdaNForBackup(paramsSelect, sumEdgeN, numChildren, mu);
    RPOOptions opts = new(bisectionIterations: 20,
                          bisectionResidualTol: 1e-6,
                          clampQToUnitInterval: true,
                          minPriorProbability: 0.0);
    RPOAnchor anchor = paramsSelect.RPOBackupRegularization == RPORegularization.ReverseKL
      ? RPOAnchor.None
      : new RPOAnchor(RPOAnchorMode.MatchValue, -1, node.Q);

    Span<double> piBar = stackalloc double[numChildren];
    Span<double> qFill = stackalloc double[numChildren];
    RegularizedPolicyOptimum.Solve(mu, qHat, lambdaN, anchor, paramsSelect.RPOBackupRegularization,
                                   yOut: piBar,
                                   qFillOut: qFill,
                                   out double _,
                                   options: opts,
                                   nanFallbackQ: consensusQ);

    double childContribution = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      childContribution += piBar[i] * qFill[i];
    }

    // Blend self-V (counts as 1 visit) with the regularized child contribution.
    int totalN = node.NodeRef.N;
    double vBar = totalN <= 0
      ? node.NodeRef.V
      : (childContribution * (totalN - 1) + node.NodeRef.V) / totalN;

    // Breadth bonus: additive max-entropy credit for nodes with MULTIPLE good moves,
    // measured on a fixed-temperature value-softmax (decoupled from lambda_N so the signal
    // persists at high N, where pi_bar collapses to one-hot and its own entropy -> 0).
    // No-op when beta <= 0.  Added in the node's OWN (side-to-move) perspective, so through
    // the negamax edge negation an ancestor reads it with alternating sign: it rewards the
    // mover's own optionality and penalizes the opponent's reply breadth (prophylaxis).
    double breadthBonus = ComputeBreadthBonus(paramsSelect, mu, qHat, numChildren, vBar,
                                              out double breadthFrac);
    if (breadthBonus != 0.0)
    {
      vBar = Math.Clamp(vBar + breadthBonus, -1.0, 1.0);
    }

    // Orthogonal late-search greedy override (no-op when CBGPUCT_BackupGreedyMaxAboveN == 0).
    vBar = ApplyBackupGreedyOverride(node, paramsSelect, qRaw, numChildren, numExpanded, totalN, vBar);

    if (CBGPUCTDumpDiagnostics.DEBUG_DUMP_CBGPUCT_BACKUP_CALCS)
    {
      // q_hat appears in both the Q_shrunk and Q_fill rows; there is no separate pi_bar
      // shrink stage so pre/post pi_bar are identical.  childContribution is passed
      // explicitly (correct), so the V_bar / bias line is faithful.  breadthFrac/breadthBonus
      // expose the breadth-bonus contribution (both 0 when the bonus is disabled).
      CBGPUCTDumpDiagnostics.DumpCBGPUCTBackup(node, paramsSelect, mu, qRaw, qHat, qFill, piBar, piBar, edgeNSpan,
                                               nSupport,
                                               numChildren, numExpanded, sumEdgeN, lambdaN,
                                               childContribution, consensusQ, vBar,
                                               breadthFrac, breadthBonus,
                                               paramsSelect.RPOBackupRegularization);
    }

    return vBar;
  }


  /// <summary>
  /// Computes the additive BACKUP BREADTH BONUS (see CBGPUCT_BackupBreadthBonusBeta).
  /// Rewards nodes with multiple good moves by measuring how much "good-move mass" is
  /// spread across the children, using a value-softmax at the FIXED temperature
  /// CBGPUCT_BackupBreadthTemperature:
  ///   w(a) proportional to mu(a) * exp((q_hat(a) - max_a q_hat) / tau_b),
  ///   breadthFrac = H(w) / ln(#contributing)  in [0, 1],
  ///   bonus = clamp( beta * breadthFrac * (1 - |vBar|), 0, CBGPUCT_BackupBreadthBonusMax ).
  /// The fixed temperature (NOT lambda_N) is what keeps the signal alive at high N; the
  /// (1 - |vBar|) gate fades it out at decided/terminal nodes; the cap keeps it a tie-
  /// breaker that can never override a real value gap.  q_hat / mu are the SAME per-child
  /// vectors the V_bar dot product consumes, so the bonus is consistent with the value it
  /// augments.  Returns 0 (breadthFrac 0) when disabled or fewer than two children carry
  /// positive weight.
  /// </summary>
  private static double ComputeBreadthBonus(ParamsSelect paramsSelect,
                                            ReadOnlySpan<double> mu, ReadOnlySpan<double> qHat,
                                            int numChildren, double vBar, out double breadthFrac)
  {
    breadthFrac = 0.0;
    double beta = paramsSelect.CBGPUCT_BackupBreadthBonusBeta;
    double tau = paramsSelect.CBGPUCT_BackupBreadthTemperature;
    if (beta <= 0.0 || tau <= 0.0 || numChildren < 2)
    {
      return 0.0;
    }

    // Decided-ness gate first (cheap): no bonus at won/lost/terminal nodes.
    double gate = 1.0 - Math.Abs(vBar);
    if (gate <= 0.0)
    {
      return 0.0;
    }

    // max q_hat for numerical stability of the exp.
    double maxQ = double.NegativeInfinity;
    for (int i = 0; i < numChildren; i++)
    {
      if (qHat[i] > maxQ)
      {
        maxQ = qHat[i];
      }
    }
    if (double.IsInfinity(maxQ))
    {
      return 0.0;
    }

    // Unnormalized value-softmax weights w(a) = mu(a) * exp((q_hat(a) - maxQ) / tau).
    double z = 0.0;
    int contributing = 0;
    Span<double> wUn = stackalloc double[numChildren];
    for (int i = 0; i < numChildren; i++)
    {
      double m = mu[i];
      if (m <= 0.0)
      {
        wUn[i] = 0.0;
        continue;
      }
      double wi = m * Math.Exp((qHat[i] - maxQ) / tau);
      wUn[i] = wi;
      z += wi;
      if (wi > 0.0)
      {
        contributing++;
      }
    }
    if (z <= 0.0 || contributing < 2)
    {
      return 0.0;
    }

    // Normalized Shannon entropy H(w) / ln(#contributing) in [0, 1].
    double h = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      double p = wUn[i] / z;
      if (p > 0.0)
      {
        h -= p * Math.Log(p);
      }
    }
    double hMax = Math.Log(contributing);
    breadthFrac = hMax > 0.0 ? h / hMax : 0.0;
    if (breadthFrac <= 0.0)
    {
      return 0.0;
    }

    double bonus = beta * breadthFrac * gate;
    double cap = paramsSelect.CBGPUCT_BackupBreadthBonusMax;
    if (cap > 0.0 && bonus > cap)
    {
      bonus = cap;
    }
    return bonus;
  }


  /// <summary>
  /// Orthogonal late-search greedy override: linearly blends the regularized V_bar with
  /// the minimax-best visited-child Q so that at high effectiveN the returned value is
  /// the best-child Q (mirroring AlphaZero's late-search single-move conviction).
  /// effectiveN is normally totalN, but gets bumped up to the min child.N over the
  /// high-P expanded children when transposition visits have given even the weakest
  /// "interesting" child more support than the parent's own visit count would imply.
  /// No-op (returns vBar unchanged) when CBGPUCT_BackupGreedyMaxAboveN == 0.
  /// See CBGPUCT_BackupGreedyMaxAboveN docs.
  /// </summary>
  private static double ApplyBackupGreedyOverride(GNode node, ParamsSelect paramsSelect,
                                                  ReadOnlySpan<double> qRaw,
                                                  int numChildren, int numExpanded,
                                                  int totalN, double vBar)
  {
    // Best visited-child Q in PARENT perspective.  Unvisited slots (qRaw NaN)
    // skipped: only observed q values qualify for the minimax-best candidate.
    double bestChildQ = double.NegativeInfinity;
    for (int i = 0; i < numChildren; i++)
    {
      if (!double.IsNaN(qRaw[i]) && qRaw[i] > bestChildQ)
      {
        bestChildQ = qRaw[i];
      }
    }
    if (double.IsNegativeInfinity(bestChildQ))
    {
      return vBar;
    }

    // Minimum child.N over expanded children with raw P > 0.02.  Terminals
    // (no ChildNode) skipped: their values are exact, so they should not
    // gate the "weakest support" tally.  Unexpanded slots also skipped (no
    // child node yet).  If no children qualify, the min stays +inf and the
    // fallback to totalN below applies.
    const float GREEDY_HIGH_P_THRESHOLD = 0.02f;
    double minChildNHighP = double.PositiveInfinity;
    for (int i = 0; i < numExpanded; i++)
    {
      GEdge edge = node.ChildEdgeAtIndex(i);
      if ((float)edge.P > GREEDY_HIGH_P_THRESHOLD
          && edge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        double childN = edge.ChildNode.NodeRef.N;
        if (childN < minChildNHighP)
        {
          minChildNHighP = childN;
        }
      }
    }

    return vBar;
  }


  /// <summary>
  /// Diagnostic dump of the top-3 visit-target deficits at the search root.
  /// </summary>
  private static void LogTop3Deficits(GNode parentNode, ParamsSelect paramsSelect,
                                      ReadOnlySpan<double> nEdge, ReadOnlySpan<double> nInFlight,
                                      ReadOnlySpan<double> pSpan, ReadOnlySpan<double> wSpan,
                                      double[] qWhenNoChildrenPerChild, double qWhenNoChildren,
                                      ReadOnlySpan<double> piBar, ReadOnlySpan<double> deficits,
                                      int numChildren, int numVisitsToCompute, double lambdaN)
  {
    int t0 = -1, t1 = -1, t2 = -1;
    for (int i = 0; i < numChildren; i++)
    {
      double s = deficits[i];
      if (t0 < 0 || s > deficits[t0]) { t2 = t1; t1 = t0; t0 = i; }
      else if (t1 < 0 || s > deficits[t1]) { t2 = t1; t1 = i; }
      else if (t2 < 0 || s > deficits[t2]) { t2 = i; }
    }

    int nVisited = 0;
    for (int i = 0; i < numChildren; i++)
    {
      if (nEdge[i] > 0) nVisited++;
    }

    float crossParentFraction = paramsSelect.CBGPUCT_SelectCrossParentNFraction;
    double sumEffN = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      sumEffN += nEdge[i] + nInFlight[i];
    }
    double targetTotal = sumEffN + 1.0;

    StringBuilder sb = new();
    sb.Append($"[CBGPUCT] root sel req={numVisitsToCompute} lambda_N={lambdaN:F4} ");
    sb.Append($"visited={nVisited}/{numChildren} ");
    sb.Append($"crossParentN={crossParentFraction:F2} top3:");

    Span<int> tops = stackalloc int[3] { t0, t1, t2 };
    for (int rank = 0; rank < 3; rank++)
    {
      int i = tops[rank];
      if (i < 0) break;

      string qStr;
      if (nEdge[i] == 0)
      {
        double fpuQ = qWhenNoChildrenPerChild != null ? qWhenNoChildrenPerChild[i] : qWhenNoChildren;
        qStr = $"fpu:{fpuQ:+0.000;-0.000}";
      }
      else
      {
        double rawQ = -wSpan[i] / nEdge[i];
        qStr = Math.Abs(rawQ) > 10 ? "prn" : rawQ.ToString("+0.000;-0.000");
      }
      sb.Append(rank == 0 ? " *#" : " | #");
      sb.Append(i);
      sb.Append($" P={pSpan[i]:F4} q={qStr} pi={piBar[i]:F4} ");
      sb.Append($"tgt={piBar[i] * targetTotal:F1} ");
      sb.Append($"act={(nEdge[i] + nInFlight[i]):F1} ");
      sb.Append($"d={deficits[i]:+0.000;-0.000}");
    }
    Console.WriteLine(sb.ToString());
  }
}
