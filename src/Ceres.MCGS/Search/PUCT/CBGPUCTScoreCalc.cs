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
  /// Per-child Bayesian-style shrinkage of pi_bar toward the normalized prior mu, by
  /// visit count N.  For each child:
  ///   pi_bar[i] := pi_bar[i] * (N_i/(N_i+K)) + mu_norm[i] * (K/(N_i+K))
  /// then renormalize so the result sums to 1.  Children with N >= K are barely shrunk;
  /// children with N == 0 collapse fully to mu_norm.  Pass k = 0 to disable.
  ///
  /// Motivation: a single noisy high-Q rollout to a low-policy child can drive its pi_bar
  /// to a value near 1, even without statistical support.  This shrinkage caps the
  /// concentration on low-N children by pulling them back toward the prior.
  /// </summary>
  private static void ApplyPiBarShrinkage(Span<double> piBar, ReadOnlySpan<double> mu,
                                          ReadOnlySpan<double> nValues, double k, int n)
  {
    if (n <= 0 || !(k > 0.0))
    {
      return;
    }

    // Normalize mu over the considered subset so the shrinkage target is a proper
    // distribution.  Skip the work entirely if mu is degenerate.
    double muSum = 0.0;
    for (int i = 0; i < n; i++)
    {
      double m = mu[i];
      if (m > 0.0)
      {
        muSum += m;
      }
    }
    if (!(muSum > 0.0))
    {
      return;
    }
    double invMuSum = 1.0 / muSum;

    // Apply shrinkage in place and accumulate the new sum for renormalization.
    double piSum = 0.0;
    for (int i = 0; i < n; i++)
    {
      double nI = nValues[i];
      double precision = nI / (nI + k);
      double muNorm = (mu[i] > 0.0 ? mu[i] : 0.0) * invMuSum;
      double v = piBar[i] * precision + muNorm * (1.0 - precision);
      piBar[i] = v;
      piSum += v;
    }
    if (piSum > 0.0)
    {
      double inv = 1.0 / piSum;
      for (int i = 0; i < n; i++)
      {
        piBar[i] *= inv;
      }
    }
  }


  /// <summary>
  /// BOUNDED RELATIVE pi_bar shrinkage: caps the influence of low-N (relative to
  /// peers) visited children whose pi_bar exceeds what their statistical support
  /// would justify, then redistributes the freed mass to the UNCAPPED visited
  /// siblings.  For each visited child (qRaw[i] not NaN):
  ///   alpha_i = min(1, (N_i / N_max_visited)^p)
  ///   cap_i   = alpha_i * pi_bar[i] + (1 - alpha_i) * mu_norm_visited[i]
  ///   pi_bar[i] := min(pi_bar[i], cap_i)
  /// where mu_norm_visited is mu normalized over the visited subset.  The freed
  /// mass (sum over capped slots of pi_bar - cap) is distributed proportionally
  /// across the uncapped visited slots so they get a uniform multiplicative
  /// boost; capped slots STAY at their cap (the influence bound is preserved
  /// strictly, not partially undone by a global rescale).  Unvisited slots are
  /// never touched.
  ///
  /// WHY ONE-SIDED CAP, NOT SYMMETRIC SHRINKAGE: the principle is bounded
  /// RELATIVE INFLUENCE, not minimum-MSE estimation.  Symmetric shrinkage
  /// (mixing pi_bar toward muNorm in both directions) has two failure modes:
  ///   1. For slots whose pi_bar is BELOW muNorm because Q legitimately said
  ///      the move is unattractive, the symmetric mixture spuriously BOOSTS
  ///      them - rewarding low-Q slots is the opposite of what we want.
  ///   2. With visited's mu_norm summing to 1.0 but visited's pi_bar summing
  ///      to visitedSumPre, any rescale that preserves visitedSumPre also
  ///      scales the alpha=1 "leader" slot, violating the documented invariant
  ///      that fully-supported slots are untouched.  Compounded across backups
  ///      this becomes a systematic V_bar pessimism bias proportional to
  ///      (pi_bar_leader - muNorm_leader).
  /// The one-sided cap is invariant for any slot whose pi_bar is at or below
  /// cap (which always includes the leader), so the only V_bar effect comes
  /// from actually-overconcentrated slots being pulled back.
  ///
  /// MASS PRESERVATION: total visited mass is preserved exactly because the
  /// excess from capped slots is added back to uncapped visited slots only.
  /// Unvisited slots are neither capped nor scaled, so the visited-vs-unvisited
  /// split also matches what Solve produced.
  ///
  /// Set exponent = 0 to disable (no-op return).
  /// </summary>
  private static void ApplyPiBarShrinkageBoundedRelative(Span<double> piBar, ReadOnlySpan<double> mu,
                                                         ReadOnlySpan<double> nValues,
                                                         ReadOnlySpan<double> qRaw,
                                                         double exponent, int n)
  {
    if (n <= 0 || !(exponent > 0.0))
    {
      return;
    }

    // Find N_max over VISITED slots only.  Unvisited slots have qRaw == NaN and
    // their nValues entries (typically 0) must not skew the ratio denominator.
    double nMax = 0.0;
    for (int i = 0; i < n; i++)
    {
      if (!double.IsNaN(qRaw[i]) && nValues[i] > nMax)
      {
        nMax = nValues[i];
      }
    }
    if (!(nMax > 0.0))
    {
      return;
    }

    // Normalize mu over the VISITED subset only.  Sum-to-1 over visited makes
    // mu_norm[i] a proper per-slot share that can be directly compared to
    // pi_bar[i] when deciding whether a slot is overconcentrated.
    double muSumVisited = 0.0;
    for (int i = 0; i < n; i++)
    {
      if (!double.IsNaN(qRaw[i]) && mu[i] > 0.0)
      {
        muSumVisited += mu[i];
      }
    }
    if (!(muSumVisited > 0.0))
    {
      return;
    }
    double invMuSumVisited = 1.0 / muSumVisited;

    // Single pass: compute cap per visited slot, apply min(piBar, cap), and
    // track which slots ended up capped vs uncapped.  Capped slots contribute
    // their excess (piBar - cap) to cappedFreedMass; uncapped slots' current
    // mass goes into uncappedMass so we can compute the redistribute factor.
    Span<bool> isCapped = stackalloc bool[n];
    double cappedFreedMass = 0.0;
    double uncappedMass = 0.0;
    for (int i = 0; i < n; i++)
    {
      isCapped[i] = false;
      if (double.IsNaN(qRaw[i]))
      {
        continue;   // unvisited slot: untouched throughout
      }

      double ratio = nValues[i] / nMax;
      double alphaI;
      if (ratio >= 1.0)
      {
        alphaI = 1.0;
      }
      else if (exponent == 1.0)
      {
        alphaI = ratio;
      }
      else if (exponent == 0.5)
      {
        alphaI = Math.Sqrt(ratio);
      }
      else
      {
        alphaI = Math.Pow(ratio, exponent);
      }

      double muNorm = (mu[i] > 0.0 ? mu[i] : 0.0) * invMuSumVisited;
      double cap = alphaI * piBar[i] + (1.0 - alphaI) * muNorm;
      if (piBar[i] > cap)
      {
        cappedFreedMass += piBar[i] - cap;
        piBar[i] = cap;
        isCapped[i] = true;
      }
      else
      {
        uncappedMass += piBar[i];
      }
    }

    // Redistribute freed mass uniformly across the uncapped visited slots ONLY.
    // Capped slots remain strictly at their cap value (the influence bound is
    // preserved), unvisited slots remain untouched.  When all visited slots
    // happen to be uncapped (cappedFreedMass == 0), nothing to do; same when
    // nothing is uncapped (degenerate; would only happen if every visited slot
    // were over-cap, which can't happen because the leader's cap equals its own
    // pi_bar and so the leader is always uncapped).
    if (cappedFreedMass > 0.0 && uncappedMass > 0.0)
    {
      double factor = (uncappedMass + cappedFreedMass) / uncappedMass;
      for (int i = 0; i < n; i++)
      {
        if (!double.IsNaN(qRaw[i]) && !isCapped[i])
        {
          piBar[i] *= factor;
        }
      }
    }
  }


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

    // Bayesian Q-shrinkage toward the per-child POLICY-IMPLIED Q: pulls low-N (noisy)
    // Q estimates toward the value implied by the policy alone, not toward the parent's
    // average Q.  This is policy-aware: a 0.2%-policy child is shrunk toward a low
    // imputed Q (the network thinks the move is bad), while a high-policy child is
    // shrunk toward a higher value.
    //
    // Schedule: shrink(N) = K / (N^p + K) where K = f/(1-f) and p is the decay exponent.
    // p = 1 is the standard Bayesian (1/N) decay; p > 1 gives faster decay past N=1
    // while preserving shrink(1) = f.  Unvisited children (N == 0) pass through unchanged
    // so they remain selectable.
    float fractionAtN1Select = paramsSelect.CBGPUCT_SelectQShrinkageFractionAtN1;
    if (fractionAtN1Select > 0.0f)
    {
      double kPseudo = fractionAtN1Select / (1.0 - fractionAtN1Select);
      double decayExp = paramsSelect.CBGPUCT_SelectQShrinkageDecayExponent;
      Span<double> impliedQ = stackalloc double[numChildren];
      ComputePolicyImpliedQ(paramsSelect, mu, qParent, impliedQ, numChildren);
      for (int i = 0; i < numChildren; i++)
      {
        double nI = nEdge[i];
        if (nI > 0.0)
        {
          double nPow = decayExp == 1.0 ? nI : Math.Pow(nI, decayExp);
          double precision = nPow / (nPow + kPseudo);
          qIn[i] = qIn[i] * precision + impliedQ[i] * (1.0 - precision);
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

    // Optional Sinkhorn-style fixed-point refinement: for unvisited children, refresh
    // the imputed q via
    //     q(a) = v_parent + lambda * (1 - mu_norm(a) / pi_bar(a))
    // and re-solve.  At convergence v* (= E_y[q]) equals v_parent, enforcing the
    // soft-Bellman expectation that the parent's value is consistent with the
    // regularized aggregate of its children.  q and pi_bar are then mutually
    // self-consistent under this external constraint.  This is the reverse-KL
    // analogue of the MatchValue forward-KL anchor.
    //
    // Skipped when CBGPUCT_SelectFixedPointIterations == 0 - zero overhead.  Default
    // is 2 iterations.  Only valid for reverse KL; the forward-KL MatchValue anchor
    // handles this case directly without iteration.
    int fpIters = paramsSelect.CBGPUCT_SelectFixedPointIterations;
    if (fpIters > 0 && paramsSelect.RPOSelectRegularization == RPORegularization.ReverseKL)
    {
      // Normalize mu once for the identity (Solve internally normalizes, but we
      // need the normalized values here for the q-update step).
      Span<double> muNorm = stackalloc double[numChildren];
      double sumMu = 0.0;
      for (int i = 0; i < numChildren; i++)
      {
        sumMu += mu[i];
      }
      double invSumMu = sumMu > 0.0 ? 1.0 / sumMu : 1.0 / numChildren;
      for (int i = 0; i < numChildren; i++)
      {
        muNorm[i] = sumMu > 0.0 ? mu[i] * invSumMu : 1.0 / numChildren;
      }

      double convergenceTol = paramsSelect.RPOSelectFixedPointTol;
      for (int iter = 0; iter < fpIters; iter++)
      {
        double maxDelta = 0.0;
        for (int i = 0; i < numChildren; i++)
        {
          if (nEdge[i] != 0)
          {
            continue;
          }
          double piBarI = piBar[i];
          if (!(piBarI > 1e-12))
          {
            continue;
          }
          double qNew = qParent + lambdaN * (1.0 - muNorm[i] / piBarI);
          // Clamp for numerical stability.
          if (qNew < -1.0) qNew = -1.0;
          else if (qNew > 1.0) qNew = 1.0;
          double delta = Math.Abs(qNew - qIn[i]);
          if (delta > maxDelta) maxDelta = delta;
          qIn[i] = qNew;
        }
        if (maxDelta < convergenceTol)
        {
          break;
        }
        RegularizedPolicyOptimum.Solve(mu, qIn, lambdaN, anchor, paramsSelect.RPOSelectRegularization,
                                       yOut: piBar,
                                       qFillOut: default,
                                       out vStar,
                                       options: opts,
                                       nanFallbackQ: qParent);
      }
    }

    // Pi_bar shrinkage toward the normalized prior, by per-child N.  Caps the
    // concentration on low-N children whose pi_bar can run away on a single noisy
    // high-Q rollout.  Disabled when CBGPUCT_PiBarShrinkageSelectPseudoVisits == 0.
    if (paramsSelect.CBGPUCT_SelectPiBarShrinkagePseudoVisits > 0.0f)
    {
      ApplyPiBarShrinkage(piBar, mu, nEdge,
                          paramsSelect.CBGPUCT_SelectPiBarShrinkagePseudoVisits, numChildren);
    }

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
      CBGPUCTDumpDiagnostics.DumpCBGPUCTSelect(paramsSelect, parentNode, pSpan, qInOriginal, qIn, piBar,
                                               firstStepDeficits, currentN, nEdge, nInFlight,
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
  /// or unexpanded entirely) to be included in the V_bar dot-product.  Expanded children
  /// at indices [0, NumEdgesExpanded) are always included regardless of P; unexpanded
  /// children at indices [NumEdgesExpanded, NumPolicyMoves) are included only when
  /// header.P meets this threshold.  Set to float.MaxValue to recover the legacy
  /// behavior where V_bar is computed over the expanded children only.
  /// </summary>
  private const float MIN_P_FOR_Q_IF_UNVISITED = 0.01f;


  /// <summary>
  /// Computes the V_bar pre-blend child contribution:
  ///   sum_i pi_bar[i] * q_used[i]
  /// where q_used is qRaw for visited slots and qFill for unvisited slots in
  /// the default (full-coverage) mode.  In observedOnly mode the sum is restricted
  /// to visited slots and pi_bar is renormalized over that subset, so the result
  /// depends only on observed q values (selfV is still blended in by the caller).
  /// Returns nodeQFallback when observedOnly is set but no slot is visited.
  /// </summary>
  private static double ComputeVBarChildContribution(ReadOnlySpan<double> piBar,
                                                     ReadOnlySpan<double> qRaw,
                                                     ReadOnlySpan<double> qFill,
                                                     int numChildren,
                                                     double nodeQFallback,
                                                     bool observedOnly)
  {
    if (!observedOnly)
    {
      double sumAll = 0.0;
      for (int i = 0; i < numChildren; i++)
      {
        sumAll += piBar[i] * (double.IsNaN(qRaw[i]) ? qFill[i] : qRaw[i]);
      }
      return sumAll;
    }

    double piVisited = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      if (!double.IsNaN(qRaw[i]))
      {
        piVisited += piBar[i];
      }
    }
    if (!(piVisited > 0.0))
    {
      return nodeQFallback;
    }
    double inv = 1.0 / piVisited;
    double sumObs = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      if (!double.IsNaN(qRaw[i]))
      {
        sumObs += piBar[i] * inv * qRaw[i];
      }
    }
    return sumObs;
  }


  /// <summary>
  /// Computes V_bar(node) via the regularized policy improvement closed form over the
  /// children's q values, then blends in the node's own network value V using the
  /// standard Ceres convention (self-V counts as 1 visit out of N total).
  ///
  /// Action-set coverage: always includes every expanded child (i &lt; NumEdgesExpanded),
  /// plus any unexpanded child whose header.P is at least MIN_P_FOR_Q_IF_UNVISITED.
  /// Unvisited slots in the considered range have qRaw = NaN and are imputed by Solve
  /// via the standard RPO inverse closed form (parameterized by RPOBackupRegularization);
  /// the imputed value is captured in qFill and used in the V_bar dot product, so the
  /// imputation source is consistent between pi_bar and V_bar.
  /// </summary>
  internal static double ComputeVBar(GNode node, ParamsSelect paramsSelect)
  {
    int numExpanded = node.NumEdgesExpanded;
    if (numExpanded == 0)
    {
      // No expanded children yet: V_bar is just the network value.  We do not extend
      // coverage to unexpanded children here because every imputed Q would equal nodeQ
      // (which equals selfV at N==1), making the result equivalent to selfV anyway.
      return node.NodeRef.V;
    }

    // Determine the considered action set.  Always include all expanded children; then
    // additionally include unexpanded children with raw policy P at or above
    // MIN_P_FOR_Q_IF_UNVISITED.  Unexpanded headers are sorted by P descending so the
    // contiguous run of high-P unexpanded slots starts immediately after NumEdgesExpanded;
    // a single linear pass is sufficient.  We do not early-break because move-ordering
    // rearrangement (GNode.CheckMoveOrderRearrangeAtIndex) can leave the unexpanded
    // segment slightly non-monotonic.
    int numChildren = numExpanded;
    int numPolicyMoves = node.NumPolicyMoves;
    if (MIN_P_FOR_Q_IF_UNVISITED < 1.0f && numPolicyMoves > numExpanded)
    {
      Span<GEdgeHeaderStruct> headers = node.EdgeHeadersSpan;
      int maxKeptIndex = numExpanded - 1;
      for (int i = numExpanded; i < numPolicyMoves; i++)
      {
        if ((float)headers[i].P >= MIN_P_FOR_Q_IF_UNVISITED)
        {
          maxKeptIndex = i;
        }
      }
      numChildren = maxKeptIndex + 1;
    }

    // Build the q vectors.  Two separate vectors are needed when Bayesian shrinkage is
    // active so that pi_bar uses shrunk Q (for noise robustness) while the final dot
    // product uses raw Q for visited children (the unbiased visit-weighted child Q):
    //   pi_bar : weights computed from q_shrunk, so low-N (noisy) children get
    //            downweighted.
    //   V_bar  : visited contributions use raw qRaw[i] (avoids double-counting the
    //            shrinkage prior); unvisited contributions use qFill[i] from Solve
    //            (the same value Solve used internally for pi_bar so the imputation
    //            is internally consistent).
    Span<double> mu = stackalloc double[numChildren];
    Span<double> qShrunk = stackalloc double[numChildren];
    Span<double> qRaw = stackalloc double[numChildren];
    Span<double> edgeNSpan = stackalloc double[numChildren];
    // Per-child statistical support: in graph mode with transpositions, child.N
    // accumulates across all parents while edge.N counts only visits via THIS edge.
    // The Q estimate's statistical support is the child-side count.  Used by the
    // bounded-relative pi_bar shrinkage; for terminal edges (no child node) we fall
    // back to edge.N since the Q is exact anyway.
    Span<double> childNSpan = stackalloc double[numChildren];

    double sumN = 0.0;
    double nodeQ = node.Q;
    float fractionAtN1Backup = paramsSelect.CBGPUCT_BackupQShrinkageFractionAtN1;
    // Convert fraction at N=1 to pseudo-visit count K (the actual coefficient in the
    // standard shrinkage formula).  Computed once; reused per child.  decayExpBackup
    // controls the rate of decay past N=1: p=1 is Bayesian 1/N, p>1 decays faster.
    double kPseudoBackup = fractionAtN1Backup > 0.0f ? fractionAtN1Backup / (1.0 - fractionAtN1Backup) : 0.0;
    double decayExpBackup = paramsSelect.CBGPUCT_BackupQShrinkageDecayExponent;

    // First pass: build mu, qRaw (raw observed Q in this node's perspective), edgeNSpan,
    // and childNSpan (child-node N for bounded-relative pi_bar shrinkage).
    // Expanded slots read from the edge (P / N / Q); unexpanded slots read header.P only
    // and contribute qRaw = NaN, edgeN = 0 to the Solve.  sumN counts visited edges only
    // (unvisited contribute 0), matching the lambda_N input domain.
    Span<GEdgeHeaderStruct> headersSpan = node.EdgeHeadersSpan;
    for (int i = 0; i < numChildren; i++)
    {
      if (i < numExpanded)
      {
        GEdge edge = node.ChildEdgeAtIndex(i);
        mu[i] = edge.P;
        int edgeN = edge.N;
        edgeNSpan[i] = edgeN;
        if (edgeN == 0)
        {
          // Expanded but never backed up: Solve imputes via the qFill closed form.
          qRaw[i] = double.NaN;
        }
        else
        {
          // Edge.Q is in child perspective; negate for this node's perspective.
          qRaw[i] = -edge.Q;
        }
        sumN += edgeN;
        // Statistical support = child node's total visit count (across all parents).
        // Terminal edges have no child node; use edge.N (their Q is exact).
        childNSpan[i] = edge.Type == GEdgeStruct.EdgeType.ChildEdge
          ? edge.ChildNode.NodeRef.N
          : edgeN;
      }
      else
      {
        // Unexpanded slot (no edge object): use header policy, no observed Q.
        mu[i] = (double)headersSpan[i].P;
        edgeNSpan[i] = 0;
        childNSpan[i] = 0;
        qRaw[i] = double.NaN;
      }
    }

    // Stage 1: pick the base anchor according to CBGPUCT_QAnchorTypeBackup.
    // Stage 2: blend that base anchor with the visit-weighted observed-child Q
    // (parent perspective) using CBGPUCT_BackupImputationAnchorK as the pseudo-visit
    // count.  Together these two knobs control where the imputation is calibrated.
    //
    // nodeQ under CBGPUCT backup is the previous V_bar (selfV-blended and possibly
    // stale w.r.t. recent search observations); pinning imputed q's to it can keep
    // anchoring the imputation at a value the children's observed q's already
    // disagree with.  The N-dependent blend lets evidence override the prior as N
    // grows.  See CBGPUCT_BackupImputationAnchorK docs for the calibration rationale.
    double anchorForImputation = ComputeImputationAnchor(paramsSelect.CBGPUCT_QAnchorTypeBackup,
                                                         node, qRaw, numChildren);
    float anchorK = paramsSelect.CBGPUCT_BackupImputationAnchorK;
    if (anchorK < float.MaxValue && sumN > 0)
    {
      double sumWQ = 0.0;
      for (int i = 0; i < numChildren; i++)
      {
        if (!double.IsNaN(qRaw[i]))
        {
          sumWQ += edgeNSpan[i] * qRaw[i];
        }
      }
      double observedAvg = sumWQ / sumN;
      double w = sumN / (sumN + anchorK);
      anchorForImputation = (1.0 - w) * anchorForImputation + w * observedAvg;
    }

    // Per-child policy-implied Q targets - used as the shrinkage target for visited
    // q's (when fractionAtN1Backup > 0) and as the imputation source for unvisited
    // slots (unconditionally).  Shrinking visited q's toward these policy-aware
    // values (rather than the scalar nodeQ) makes the shrinkage prior depend on each
    // child's policy mass: low-mu children get pulled toward a low target, high-mu
    // children toward a higher target.  Unvisited slots receive the per-child value
    // directly, mirroring the FPU imputation already used during selection.
    Span<double> impliedQ = stackalloc double[numChildren];
    ComputePolicyImpliedQ(paramsSelect, mu, anchorForImputation, impliedQ, numChildren);

    // Second pass: build qShrunk (used as input to Solve for pi_bar).
    //   - Visited (qRaw not NaN) with absolute shrinkage: Bayesian shrinkage toward impliedQ.
    //   - Visited without absolute shrinkage: qShrunk = qRaw passthrough.
    //   - Unvisited: pre-fill with the per-child policy-implied Q (= what the SELECT
    //     phase FPU uses).  Solve then sees a finite q and uses it directly, bypassing
    //     the scalar nanFallback path - so pi_bar AND the V_bar dot product see per-
    //     child policy-correlated q for unvisited children, instead of a single
    //     uniform value.
    //
    // Bounded-relative robustness is applied at the pi_bar (not q) layer downstream;
    // see CBGPUCT_BackupPiBarShrinkageBoundedRelativeExponent / ApplyPiBarShrinkageBoundedRelative.
    for (int i = 0; i < numChildren; i++)
    {
      if (double.IsNaN(qRaw[i]))
      {
        qShrunk[i] = impliedQ[i];
      }
      else if (fractionAtN1Backup > 0.0f)
      {
        double edgeN = edgeNSpan[i];
        double nPow = decayExpBackup == 1.0 ? edgeN : Math.Pow(edgeN, decayExpBackup);
        double precision = nPow / (nPow + kPseudoBackup);
        qShrunk[i] = qRaw[i] * precision + impliedQ[i] * (1.0 - precision);
      }
      else
      {
        qShrunk[i] = qRaw[i];
      }
    }

    double lambdaN = ComputeLambdaNForBackup(paramsSelect, sumN, numChildren, mu);

    RPOOptions opts = new(bisectionIterations: 20,
                          bisectionResidualTol: 1e-6,
                          clampQToUnitInterval: true,
                          minPriorProbability: 0.0);
    RPOAnchor anchor = paramsSelect.RPOBackupRegularization == RPORegularization.ReverseKL
      ? RPOAnchor.None
      : new RPOAnchor(RPOAnchorMode.MatchValue, -1, node.Q);

    // Per-child lambda vector, built only when the exponent knob is positive AND we're
    // in reverse-KL mode (forward-KL Solve does not support per-child lambda).  Formula:
    //   lambda_a = lambdaN * (max(N_a, N_FLOOR) / N_max)^p
    // The floor prevents N_a=0 (unvisited slots, including extended-coverage admittees)
    // from collapsing lambda_a to zero; we don't want them at exactly zero either since
    // the bisection then ignores them entirely.  When lambdaPerChild stays empty Solve
    // uses scalar lambdaN (existing behavior).
    float perChildExp = paramsSelect.CBGPUCT_BackupLambdaPerChildExponent;
    bool usePerChildLambda = perChildExp > 0.0f
                          && paramsSelect.RPOBackupRegularization == RPORegularization.ReverseKL;
    Span<double> lambdaPerChild = stackalloc double[numChildren];
    if (usePerChildLambda)
    {
      double nMax = 0.0;
      for (int i = 0; i < numChildren; i++)
      {
        if (edgeNSpan[i] > nMax)
        {
          nMax = edgeNSpan[i];
        }
      }
      if (nMax > 0.0)
      {
        const double N_FLOOR = 0.5;
        for (int i = 0; i < numChildren; i++)
        {
          double effN = edgeNSpan[i] >= N_FLOOR ? edgeNSpan[i] : N_FLOOR;
          double ratio = effN / nMax;
          double scale = perChildExp == 1.0f ? ratio : Math.Pow(ratio, perChildExp);
          lambdaPerChild[i] = lambdaN * scale;
        }
      }
      else
      {
        // No visited children at all - per-child can't discriminate; fall back to uniform.
        for (int i = 0; i < numChildren; i++)
        {
          lambdaPerChild[i] = lambdaN;
        }
      }
    }

    Span<double> piBar = stackalloc double[numChildren];
    // qFill captures Solve's NaN-imputed q vector so the V_bar dot product can use the
    // SAME imputed values Solve internally used for pi_bar (avoiding the prior silent
    // mismatch in ForwardKL where Solve would impute via lambda*log(mu)+C(s) but V_bar
    // would substitute scalar nodeQ).  For visited slots qFill[i] equals the input
    // (subject to clamping); we still use qRaw[i] for those in the dot product to keep
    // the existing anti-double-count behavior for shrinkage.
    Span<double> qFill = stackalloc double[numChildren];
    RegularizedPolicyOptimum.Solve(mu, qShrunk, lambdaN, anchor, paramsSelect.RPOBackupRegularization,
                                   yOut: piBar,
                                   qFillOut: qFill,
                                   out double _,
                                   options: opts,
                                   nanFallbackQ: node.Q,
                                   lambdaPerChild: usePerChildLambda ? lambdaPerChild : default);

    bool vBarObservedOnly = paramsSelect.CBGPUCT_BackupVBarObservedOnly;

    // Optional Sinkhorn-style fixed-point refinement for BACKUP.  Same algebraic
    // identity as the select-side iteration:
    //     q(a) = vRef + lambda * (1 - mu_norm(a) / pi_bar(a))
    // but with vRef set to the current V_bar pre-blend rather than the constant
    // qParent used in selection.  At convergence E_y[q] equals V_bar, making V_bar
    // the self-consistent regularized value (the soft-Bellman fixed point), rather
    // than a one-shot computation anchored to the previous backup's node.Q via the
    // imputation.
    //
    // When CBGPUCT_BackupVBarObservedOnly is set, vRef is computed observed-only
    // (renormalized pi_bar over visited slots, summed against qRaw only).  The
    // iteration then converges to a DIFFERENT fixed point: vRef equals the
    // observed-only V_bar, not the Grill-style E_y[q] over all actions.  Imputed
    // q's for unvisited slots still get refreshed (and pi_bar still shifts), but
    // those imputed q's no longer enter the value target.  See
    // CBGPUCT_BackupVBarObservedOnly docs for the rationale and recommended usage.
    //
    // Skipped when CBGPUCT_BackupFixedPointIterations == 0 (default) - zero overhead,
    // behavior identical to legacy.  Only valid for reverse KL (the forward-KL
    // MatchValue anchor enforces the equivalent constraint in closed form).
    int backupFpIters = paramsSelect.CBGPUCT_BackupFixedPointIterations;
    if (backupFpIters > 0 && paramsSelect.RPOBackupRegularization == RPORegularization.ReverseKL)
    {
      // Normalize mu once for the identity (Solve internally normalizes, but we need
      // the normalized values here for the q-update step).
      Span<double> muNorm = stackalloc double[numChildren];
      double sumMu = 0.0;
      for (int i = 0; i < numChildren; i++)
      {
        sumMu += mu[i];
      }
      double invSumMu = sumMu > 0.0 ? 1.0 / sumMu : 1.0 / numChildren;
      for (int i = 0; i < numChildren; i++)
      {
        muNorm[i] = sumMu > 0.0 ? mu[i] * invSumMu : 1.0 / numChildren;
      }

      // Initial reference value (full-coverage OR observed-only per knob).
      double vRef = ComputeVBarChildContribution(piBar, qRaw, qFill, numChildren,
                                                 nodeQ, vBarObservedOnly);

      double convergenceTol = paramsSelect.RPOBackupFixedPointTol;
      for (int iter = 0; iter < backupFpIters; iter++)
      {
        double maxDelta = 0.0;
        for (int i = 0; i < numChildren; i++)
        {
          if (!double.IsNaN(qRaw[i]))
          {
            // Visited slot: keep its shrunk q (the iteration only refines imputed q's).
            continue;
          }
          double piBarI = piBar[i];
          if (!(piBarI > 1e-12))
          {
            continue;
          }
          double qNew = vRef + lambdaN * (1.0 - muNorm[i] / piBarI);
          if (qNew < -1.0) qNew = -1.0;
          else if (qNew > 1.0) qNew = 1.0;
          double delta = Math.Abs(qNew - qShrunk[i]);
          if (delta > maxDelta) maxDelta = delta;
          qShrunk[i] = qNew;
        }
        if (maxDelta < convergenceTol)
        {
          break;
        }
        // Re-solve with the refreshed imputed q's; vRef rebuilt from updated pi_bar/qFill.
        // Same lambdaPerChild as the initial solve - per-child lambda stays fixed
        // across fixed-point iterations (it depends only on edge.N, not on q).
        RegularizedPolicyOptimum.Solve(mu, qShrunk, lambdaN, anchor, paramsSelect.RPOBackupRegularization,
                                       yOut: piBar,
                                       qFillOut: qFill,
                                       out double _,
                                       options: opts,
                                       nanFallbackQ: vRef,
                                       lambdaPerChild: usePerChildLambda ? lambdaPerChild : default);
        vRef = ComputeVBarChildContribution(piBar, qRaw, qFill, numChildren,
                                            nodeQ, vBarObservedOnly);
      }
    }

    // Snapshot pi_bar from Solve BEFORE any shrinkage, so the dump can show both the
    // raw Solve output and the shrunk version that actually drives V_bar.  Tiny cost
    // (numChildren doubles); always taken so diagnostics stay consistent whether the
    // dump is enabled or not.
    Span<double> piBarPreShrink = stackalloc double[numChildren];
    piBar[..numChildren].CopyTo(piBarPreShrink);

    // Pi_bar shrinkage toward the normalized prior, by per-child N.  Highly recommended
    // for backup: prevents a single high-Q low-N rollout from dominating V_bar.  Disabled
    // when CBGPUCT_PiBarShrinkageBackupPseudoVisits == 0.  Uses edge.N (legacy semantics).
    if (paramsSelect.CBGPUCT_BackupPiBarShrinkagePseudoVisits > 0.0f)
    {
      ApplyPiBarShrinkage(piBar, mu, edgeNSpan,
                          paramsSelect.CBGPUCT_BackupPiBarShrinkagePseudoVisits, numChildren);
    }

    // BOUNDED RELATIVE pi_bar shrinkage: pulls low-N (relative to peers) children's
    // pi_bar toward the normalized prior mu.  Uses child.N (statistical support).
    // See CBGPUCT_BackupPiBarShrinkageBoundedRelativeExponent docs for the principle.
    // Layers on top of the absolute shrinkage above when both are active (mu target
    // is the same in both, so order does not affect the algebraic fixed point modulo
    // the per-step renormalizations).
    if (paramsSelect.CBGPUCT_BackupPiBarShrinkageBoundedRelativeExponent > 0.0f)
    {
      ApplyPiBarShrinkageBoundedRelative(piBar, mu, childNSpan, qRaw,
                                         paramsSelect.CBGPUCT_BackupPiBarShrinkageBoundedRelativeExponent,
                                         numChildren);
    }

    // V_bar dot product (helper applies observed-only renormalization when knob is set).
    // Default mode: visited slots contribute qRaw, unvisited contribute qFill (Solve's
    // imputation).  Observed-only mode: only visited slots contribute (qRaw only), with
    // pi_bar renormalized over the visited subset so weights sum to 1.  Uses the post-
    // shrinkage piBar (the one we actually committed to).
    double childContribution = ComputeVBarChildContribution(piBar, qRaw, qFill, numChildren,
                                                            nodeQ, vBarObservedOnly);

    // Blend self-V (counts as 1 visit) with regularized child contribution (counts as totalN - 1).
    int totalN = node.NodeRef.N;
    double vBar = totalN <= 0
      ? node.NodeRef.V
      : (childContribution * (totalN - 1) + node.NodeRef.V) / totalN;

    // Optional greedy override: linearly blend the regularized V_bar with the
    // minimax-best child Q so that at high effectiveN the returned value is the
    // best-child Q (mirroring AlphaZero's late-search single-move conviction).
    // effectiveN is normally totalN, but gets bumped up to min child.N over the
    // high-P expanded children when transposition visits have given even the
    // weakest "interesting" child more support than the parent's own visit
    // count would imply.  See CBGPUCT_BackupGreedyMaxAboveN docs for the design.
    int greedyMaxAboveN = paramsSelect.CBGPUCT_BackupGreedyMaxAboveN;
    if (greedyMaxAboveN > 0)
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
      if (!double.IsNegativeInfinity(bestChildQ))
      {
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
        double effectiveN = totalN;
        if (!double.IsPositiveInfinity(minChildNHighP) && minChildNHighP > effectiveN)
        {
          effectiveN = minChildNHighP;
        }
        double greedyWeight = effectiveN / greedyMaxAboveN;
        if (greedyWeight > 1.0)
        {
          greedyWeight = 1.0;
        }
        else if (greedyWeight < 0.0)
        {
          greedyWeight = 0.0;
        }
        vBar = greedyWeight * bestChildQ + (1.0 - greedyWeight) * vBar;
      }
    }

    if (CBGPUCTDumpDiagnostics.DEBUG_DUMP_CBGPUCT_BACKUP_CALCS)
    {
      CBGPUCTDumpDiagnostics.DumpCBGPUCTBackup(node, paramsSelect, mu, qRaw, qShrunk, qFill, piBarPreShrink, piBar, edgeNSpan,
                                               numChildren, numExpanded, sumN, lambdaN,
                                               childContribution, vBar,
                                               paramsSelect.RPOBackupRegularization);
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
