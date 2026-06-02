#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;
using System.Diagnostics;
using System.Numerics.Tensors;
using Ceres.Base.Math;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Utils;

namespace Ceres.MCGS.Search.RPO;

/// <summary>
/// Uses regularized policy optimization to determine optimal posterior policy
/// given prior policy and current Q values (and possibly visit counts for use with weighting of prior influence).
/// 
/// Based closely on:
/// "Monte-Carlo tree search as regularized policy optimization" by Grill et al (2020)
/// </summary>
public static class RPOUtils
{
  public static RPOResult ChooseBestMove(float nodeV, 
                                         Span<float> priorP, 
                                         Span<float> q, 
                                         Span<int> visitCounts,
//                                           float fillInQ,
                                         int n, float lambda, float lambdaPower, 
                                         float shrinkFactor = 0, bool weighted = false)
  {
    // Compute Q in expected range [0...1]
    float[] qAdj = new float[q.Length];
    for (int i =0; i<qAdj.Length;i++)
    {
      qAdj[i] = (float)(q[i] + 1f) *0.5f;
    }

    RPOResult ret = DoChooseBestMove(priorP, qAdj, visitCounts, /*fillInQ, */ n, lambda, lambdaPower, shrinkFactor, weighted);
    return ret with { NewQ = ret.NewQ * 2 - 1 };
  }


  public static RPOResult DoChooseBestMove(Span<float> priorP, Span<float> q, Span<int> visitCounts,
//                                           float fillInQ,
                                         int n, float lambda, float lambdaPower, float shrinkFactor = 0, bool weighted = false)
  {
#if NOT
    // ** LOWER BOUND LOGIC BASED ON N (per action)
    const bool SHRINK_USING_N = false;
    for (int i = 0; i < q.Length; i++)
    {
      float visitCount = Math.Max(0, MathF.Max(1, visitCounts[i]));
      const float UNCERTAINTY_MULTIPLIER = 0.10f;
      q[i] -= UNCERTAINTY_MULTIPLIER * (1.0f / MathF.Sqrt(visitCount));
    }
    if (shrinkFactor > 0)
    {
      // TODO: Shrinkage idea abandoned for now
      //       It's not clear what to shrink toward;
      //       e.g. for low-policy moves we expect possibly much worse than current Q.
      throw new NotImplementedException();
      //PolicyUtils.ShrinkTowardQ(q, priorQ, counts, shrinkFactor);
    }
#endif

    //int maxVisitCount = (int)TensorPrimitives.Max(visitCounts);
    int maxVisitCount = 0;
    for (int ix=0;ix<visitCounts.Length;ix++)
    {
      if (visitCounts[ix] > maxVisitCount)
      {
        maxVisitCount = visitCounts[ix];
      }
    }

    // We can't do optimization which includes any moves that have no (or possibly little) data.
    // Create possibly shortened arrays with only the points that are usable.
    int MIN_VISITS = MCGSSelect.RPO_CHOOSES_NEW_CHILDREN ? 0 : 1;// (weighted ? 1 : 2);
    // TODO: eliminate expensive array resizing if always keeping all (i.e. RPO_CHOOSES_NEW_CHILDREN is true)

    (int[] visitCountsEligible, float[] priorPEligible, float[] qEligible, int[] indicesEligible)
       = RPOArrayUtils.SubsetGreaterThan(visitCounts, priorP, q, MIN_VISITS);

    // No useful computation possible if there is only a single point left.
    if (visitCountsEligible.Length < 2)
    {
      return new RPOResult(null, null, null, q.Length > 0 ? q[0] : double.NaN, 0);
    }

    StatUtils.Normalize(priorPEligible);

    const float EPSILON = 1e-3f;// 1e-3f; // seems relatively (or totally?) insensitive to this value

    // Possibly compute shrinkage factors to be applied elementwise to
    // adjust the magnitude of the influence of the prior (elementwise)
    // so that low-visits elements (high statisitcal uncertainty) can be regularized more.
    Span<float> weights = weighted ? new float[qEligible.Length] : Span<float>.Empty;
    if (weighted)
    {
      for (int i = 0; i < qEligible.Length; i++)
      {
        static float Scale(float ratioAllToThis)
        {
          // ratio = N / n
          return MathF.Pow(30.0f / MathF.Min(30.0f / ratioAllToThis, 30.0f),
                           MCGSParamsFixed.RPO_VISIT_COUNT_SHRINK_POWER);
        }

        // Compute scaling factors for how much additional shrinkage to apply to each data point
        float ratio = (float)visitCounts[i] / (n - 1); // subtract out parent visit
        if (ratio > 0.5) // at 50% of visits to one more (or more), there is no additional scaling toward prior
        {
          ratio = 1;
        }
        else
        {
          ratio *= 2;
        }
        weights[i] = Scale(1.0f/ratio);
      }
    }

    //      TensorPrimitives.Pow(weights, 2, weights);
    // NOTES:
    //   - we disable weights by filling in (1)
    //   - additionally we call PolicyOptimizerAVXDouble but this 
    //     DOES NOT SUPPORT WEIGHTS (only the reverse version does)
    //weights.Fill(1);  // **************************************************** DISABLED (see above) ****************
    Debug.Assert(weights.Length == 0); // not supported in optimizer below

    float lambdaNReverse;
    if (lambdaPower > 0)
    {
      lambdaNReverse = lambda / MathF.Pow(n, lambdaPower); // (try n+2)?
    }
    else
    { 
      const float BASE = 2;
      lambdaNReverse = lambda / (BASE + 10*MathF.Log(n));
    }

    float[] wtsP = new float[priorPEligible.Length];

// ORG:      Span<float> piStar2 = PolicyOptimizerReverse.Optimize(EPSILON, qEligible, lambdaNReverse, priorPEligible, weights);

    Span<float> piStar2 = PolicyOptimizerAVXDouble.Optimize(EPSILON, qEligible, lambdaNReverse, priorPEligible);//, Span<float>.Empty);
    //      Console.WriteLine("post P : " + String.Join(", ", piStar2.ToArray()));

    double newQ = 0;// TensorPrimitives.Dot(piStar2, qEligible);

    // Add in influence of node itself
    //      double newW = newQ * (n - 1) + nodeV;
    //      newQ = newW / n;

    // Best move according to RPO
    int indexEligible = StatUtils.IndexOfMax(piStar2);
    int indexAbsolute = indicesEligible[indexEligible];

    // Fallback move (using N)
    (int maxIndex, int maxN) = IndexMoveMaxN(visitCounts);


    // Some visit counts may have been filtered out, so need to get count used
    int nSeen = 0;
    for (int i = 0; i < piStar2.Length; i++)
    {
      nSeen += visitCounts[i];
    }

    double[] empiricalP = new double[piStar2.Length];
    for (int i=0; i<empiricalP.Length;i++)
    {
      empiricalP[i] = visitCounts[i] / nSeen;
    }

    const float FRACTION_MIN_N = 0.2f; // values from 0.20 to 0.40 seem to yield similar results (@2k or @5k nodes)
    if (visitCounts[indexAbsolute] < maxN * FRACTION_MIN_N)
    {
      // Insufficient visits to the chose move, fall back to move with max N
      return new RPOResult(visitCountsEligible, empiricalP, piStar2.ToArray(), (float)newQ, maxIndex);
    }
    else
    {
      return new RPOResult(visitCountsEligible, empiricalP, piStar2.ToArray(), (float)newQ, indexAbsolute);
    }
  }


  static (int maxIndex, int maxN) IndexMoveMaxN(Span<int> visitCounts)
  {
    int maxIndex = -1;
    int maxN = 0;
    for (int i = 0; i < visitCounts.Length; i++)
    {
      if (visitCounts[i] > maxN)
      {
        maxIndex = i;
        maxN = visitCounts[i];
      }
    }
    return (maxIndex, maxN);
  }

#if NOT
  public static int NextMoveToVisit(GNode node, float lambda, float lambdaPower)
  {
    RPOResult rpoResult = BestMoveInfo(node, lambda, lambdaPower);
    if (rpoResult.empiricalP == null)
    {
      return -1;
    }
    int maxBelowIndex = -1;
    float maxBelow = float.MinValue;
    for (int i = 0; i < rpoResult.optimalP.Length; i++)
    {
      float amtBelow = rpoResult.optimalP[i] - rpoResult.empiricalP[i];
      if (amtBelow > maxBelow)
      {
        maxBelow = amtBelow;
        maxBelowIndex = i;
      }
    }
    return maxBelowIndex;
  }
#endif

  public static GEdge BestMove(GNode node, float qWhenNoChildren, int numChildrenToConsider, 
                               float lambda, float lambdaPower, bool weighted = false)
  {
    RPOResult rpoResult = BestMoveInfo(node, qWhenNoChildren, numChildrenToConsider, lambda, lambdaPower, weighted);
    int bestMoveIndex = rpoResult.indexTopPMove;

    return node.ChildEdgeAtIndex(bestMoveIndex);
  }

  public static RPOResult BestMoveInfo(GNode node, float qWhenNoChildren, int numChildrenToConsider, 
                                       float lambda, float lambdaPower, 
                                       bool weighted = false, bool testMode = false)
  {
#if NOT
    if (testMode)
    {
      if (float.IsNaN(node.UncertaintyPolicy))
      {
        throw new Exception("UncertaintyPolicy enabled in RPO but value was NaN");
      }
      lambda*=  (1.0f - node.UncertaintyPolicy);
    }
#endif
    Span<float> qN = stackalloc float[numChildrenToConsider];
    Span<float> pN = stackalloc float[numChildrenToConsider];
    Span<int> vN = stackalloc int[numChildrenToConsider];
    int numExpanded = node.NumEdgesExpanded;
    for (int i = 0; i < numChildrenToConsider; i++)
    {
      pN[i] = node.ChildEdgeAtIndex(i).P;

      if (i < numExpanded)
      {
        qN[i] = (float)-node.ChildEdgeAtIndex(i).Q;
        vN[i] = node.ChildEdgeAtIndex(i).N;
        if (weighted)
        {

          // It's not clear what to shrink toward;
          // e.g. for low-policy moves we expect possibly much worse than current Q.
          float qToShrinkTo = (float)node.Q - 0.15f; // bias downward slightly

          // Apply weighting to blend in parent Q value based on visit count
          // (shrink parent contribution based on sqrt(child_n)).
          float fractionParent = 0.2f * (1.0f / MathF.Sqrt(vN[i]));
          qN[i] = (fractionParent * qToShrinkTo) + (1.0f - fractionParent) * qN[i];
        }
      }
      else
      {
        qN[i] = qWhenNoChildren; // + MCGSSelect.RPO_NEW_CHILDREN_FPU_ADJUSTMENT;
        vN[i] = 0;
      }
    }
weighted=false; // <---------------------- HACK. WEIGHTING ALREADY HANDLED JUST ABOVE.

    RPOResult rpo = ChooseBestMove(node.V, pN, qN, vN, node.N, 
                                   lambda, lambdaPower, shrinkFactor: 0, weighted);
    return rpo;
  }


  public static void Test()
  {

    Span<float> q = stackalloc float[] { 0.50f, 0.2f, 1f };
    Span<float> priorP = stackalloc float[] { 0.3f, 0.65f, 0.05f };
    Span<int> visitCounts = stackalloc int[] { 10, 10, 1 };
    const int N = 21;



    Span<float> piStar1 = PolicyOptimizerNonSIMD.Optimize(1e-6f, q, lambdaN: 0.05f, priorP);
    Console.WriteLine(String.Join(", ", piStar1.ToArray()));

    Span<float> piStar2 = PolicyOptimizerAVXDouble.Optimize(1e-6f, q, lambdaN: 0.05f, priorP);
    Console.WriteLine(String.Join(", ", piStar2.ToArray()));

    Span<float> piStar3 = PolicyOptimizerAVXFloat.Optimize(1e-6f, q, lambdaN: 0.05f, priorP);
    Console.WriteLine(String.Join(", ", piStar3.ToArray()));

    //Console.WriteLine();
    //	Console.WriteLine(String.Join(", ", q.ToArray()));
    //	PolicyUtils.ShrinkTowardQ(q, 0.4f, [17, 17, 3], 1f);
    //	Console.WriteLine(String.Join(", ", q.ToArray()));

    Console.WriteLine();
    Console.WriteLine("pre Q  : " + String.Join(", ", q.ToArray()));
    Console.WriteLine("P      : " + String.Join(", ", priorP.ToArray()));
    float value = Test1(0.1f, priorP, q, [17, 17, 1], 1f);
    Console.WriteLine("post Q : " + String.Join(", ", q.ToArray()));
    Console.WriteLine(value);
    //BenchmarkRunner.Run<PolicyOptimizerBenchmarks>();
    return;
  }


  static float Test1(float priorQ, Span<float> priorP, Span<float> q, Span<int> counts, float shrinkFactor)
  {
    if (shrinkFactor > 0)
    {
      PolicyUtils.ShrinkTowardQ(q, priorQ, counts, shrinkFactor);
    }
    Span<float> piStar2 = PolicyOptimizerAVXDouble.Optimize(1e-6f, q, lambdaN: 0.05f, priorP);
    Console.WriteLine("post P : " + String.Join(", ", piStar2.ToArray()));

    return TensorPrimitives.Dot(piStar2, q);
  }


  /// <summary>
  /// Golden-value smoke tests for the unified RPO primitive
  /// (RegularizedPolicyOptimum + RPOVisitAllocator) against the legacy
  /// implementations they replace.  Prints results to Console; does not assert
  /// (this is invoked manually as a sanity check).
  /// </summary>
  public static void TestRPOUnified()
  {
    Console.WriteLine();
    Console.WriteLine("=== TestRPOUnified ===");

    TestReverseKLParity();
    TestForwardKLMatchValueParity();
    TestForwardKLMatchChildParity();
    TestVBarRegression();
    TestAllocatorParity();
    TestFixedPointSelfConsistency();
    TestLambdaScheduleCurve();

    Console.WriteLine("=== TestRPOUnified complete ===");
    Console.WriteLine();
  }


  /// <summary>
  /// Prints the lambda_N selection-phase schedule across a range of sumN values both with
  /// the coefficient log-growth term off (legacy / backup-phase) and on (selection-phase,
  /// using the current defaults).  Values are computed by direct evaluation of the formula
  /// rather than by calling the internal helper.
  /// </summary>
  private static void TestLambdaScheduleCurve()
  {
    // Mirror Ceres select-phase defaults.
    double lambdaC = 1.75;
    double lambdaExp = 0.5;
    double denominatorBase = 1.0;
    double cLogBase = Ceres.MCGS.Search.Params.ParamsSelect.DEFAULT_LOG_GROWTH_BASE;
    double cLogFactor = Ceres.MCGS.Search.Params.ParamsSelect.DEFAULT_LOG_GROWTH_FACTOR;

    int[] sumNs = { 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000 };

    Console.WriteLine("[LambdaSchedule]   sumN   | no-cLog  | with-cLog | c(N)   | boost");
    foreach (int N in sumNs)
    {
      double basePart = N > 0 ? lambdaC * Math.Pow(N, lambdaExp) / (denominatorBase + N) : 0;
      double cEff = lambdaC + cLogFactor * Math.Log((N + cLogBase + 1.0) / cLogBase);
      double withCLog = N > 0 ? cEff * Math.Pow(N, lambdaExp) / (denominatorBase + N) : 0;
      double boostPct = basePart > 0 ? 100.0 * (withCLog / basePart - 1.0) : 0;
      Console.WriteLine($"                {N,7:N0}  | {basePart,7:F5}  | {withCLog,7:F5}   | {cEff,5:F3}  | +{boostPct,5:F1}%");
    }
  }


  /// <summary>
  /// Compares RegularizedPolicyOptimum.Solve (ReverseKL) to legacy
  /// ReverseKlPosteriorPolicy.ComputePosterior on a small fixture.
  /// </summary>
  private static void TestReverseKLParity()
  {
    // Existing self-test fixture from BoltzmannCalibration.TestBoltzmann.
    double[] mu = { 0.769, 0.231 };
    double[] q = { 0.664, 0.600 };
    double lambda = 0.064;

    double[] yNew = new double[mu.Length];
    RegularizedPolicyOptimum.Solve(mu, q, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                   yNew, default, out double vStarNew);

    (double Q, double PriorP, int N, double U)[] actions = new (double, double, int, double)[mu.Length];
    for (int i = 0; i < mu.Length; i++)
    {
      actions[i] = (q[i], mu[i], 1, double.NaN);
    }
    double[] yLegacy = new double[mu.Length];
    double[] qFillLegacy = new double[mu.Length];
    ReverseKlPosteriorPolicy.ComputePosterior(actions, lambda, lambdaQ: 0.0,
                                              rootQ: 0.6, rootU: 0.0,
                                              yLegacy, qFillLegacy,
                                              new ReverseKlPosteriorPolicy.Options(bisectionIterations: 60));

    double maxDelta = 0.0;
    for (int i = 0; i < mu.Length; i++)
    {
      double d = Math.Abs(yNew[i] - yLegacy[i]);
      if (d > maxDelta) maxDelta = d;
    }
    Console.WriteLine($"[ReverseKL parity]   y_new=[{yNew[0]:F4},{yNew[1]:F4}] y_legacy=[{yLegacy[0]:F4},{yLegacy[1]:F4}] maxDelta={maxDelta:E2} vStar={vStarNew:F4}");
  }


  /// <summary>
  /// Compares RegularizedPolicyOptimum.Solve (ForwardKL, MatchValue anchor) to
  /// legacy BoltzmannCalibration.ComputeQFromPolicy_MatchParentValue on a small fixture.
  /// </summary>
  private static void TestForwardKLMatchValueParity()
  {
    double[] mu = { 0.769, 0.231 };
    double[] qIn = { double.NaN, double.NaN };
    double parentValue = 0.20;
    double tau = 0.05;

    double[] qFillNew = new double[mu.Length];
    RegularizedPolicyOptimum.Solve(mu, qIn, tau,
                                   new RPOAnchor(RPOAnchorMode.MatchValue, -1, parentValue),
                                   RPORegularization.ForwardKLSoftmax,
                                   default, qFillNew, out double _,
                                   new RPOOptions(bisectionIterations: 12, bisectionResidualTol: 1e-6,
                                                  clampQ: false, minPriorProbability: 0.0));

    float[] muF = { 0.769f, 0.231f };
    float[] qLegacy = new float[mu.Length];
    BoltzmannCalibration.ComputeQFromPolicy_MatchParentValue(muF, (float)parentValue, (float)tau, qLegacy,
                                                             renormalizeIfNeeded: false,
                                                             clipToRange: false);

    double maxDelta = 0.0;
    for (int i = 0; i < mu.Length; i++)
    {
      double d = Math.Abs(qFillNew[i] - qLegacy[i]);
      if (d > maxDelta) maxDelta = d;
    }
    Console.WriteLine($"[Fwd MatchValue]     qFill_new=[{qFillNew[0]:F4},{qFillNew[1]:F4}] qFill_legacy=[{qLegacy[0]:F4},{qLegacy[1]:F4}] maxDelta={maxDelta:E2}");
  }


  /// <summary>
  /// Compares RegularizedPolicyOptimum.Solve (ForwardKL, MatchChild anchor) to
  /// legacy BoltzmannCalibration.ComputeQFromPolicy_AnchorChild on a small fixture.
  /// </summary>
  private static void TestForwardKLMatchChildParity()
  {
    double[] mu = { 0.769, 0.231 };
    double[] qIn = { double.NaN, double.NaN };
    double anchorQ = 0.664;
    double tau = 0.05;
    int anchorIdx = 0;

    double[] qFillNew = new double[mu.Length];
    RegularizedPolicyOptimum.Solve(mu, qIn, tau,
                                   new RPOAnchor(RPOAnchorMode.MatchChild, anchorIdx, anchorQ),
                                   RPORegularization.ForwardKLSoftmax,
                                   default, qFillNew, out double _,
                                   new RPOOptions(bisectionIterations: 12, bisectionResidualTol: 1e-6,
                                                  clampQ: false, minPriorProbability: 0.0));

    float[] muF = { 0.769f, 0.231f };
    float[] qLegacy = new float[mu.Length];
    BoltzmannCalibration.ComputeQFromPolicy_AnchorChild(muF, anchorIdx, (float)anchorQ, (float)tau, qLegacy,
                                                         renormalizeIfNeeded: false,
                                                         clipToRange: false);

    double maxDelta = 0.0;
    for (int i = 0; i < mu.Length; i++)
    {
      double d = Math.Abs(qFillNew[i] - qLegacy[i]);
      if (d > maxDelta) maxDelta = d;
    }
    Console.WriteLine($"[Fwd MatchChild]     qFill_new=[{qFillNew[0]:F4},{qFillNew[1]:F4}] qFill_legacy=[{qLegacy[0]:F4},{qLegacy[1]:F4}] maxDelta={maxDelta:E2}");
  }


  /// <summary>
  /// V_bar regression: builds a small action set with some NaN q's, runs the new
  /// Solve under ReverseKL, and compares its vStar output to sum_i y_i * qFill_i
  /// computed by the legacy ComputePosterior on the same inputs.
  /// </summary>
  private static void TestVBarRegression()
  {
    double[] mu = { 0.50, 0.30, 0.20 };
    double[] qIn = { -0.20, double.NaN, 0.10 };
    double lambda = 0.05;
    double rootQ = 0.0;

    double[] yNew = new double[mu.Length];
    double[] qFillNew = new double[mu.Length];
    RegularizedPolicyOptimum.Solve(mu, qIn, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                   yNew, qFillNew, out double vStarNew,
                                   options: new RPOOptions(bisectionIterations: 60, bisectionResidualTol: 1e-9,
                                                           clampQ: true, minPriorProbability: 0.0),
                                   nanFallbackQ: rootQ);

    (double Q, double PriorP, int N, double U)[] actions = new (double, double, int, double)[mu.Length];
    for (int i = 0; i < mu.Length; i++)
    {
      actions[i] = (qIn[i], mu[i], 1, double.NaN);
    }
    double[] yLegacy = new double[mu.Length];
    double[] qFillLegacy = new double[mu.Length];
    ReverseKlPosteriorPolicy.ComputePosterior(actions, lambda, lambdaQ: 0.0,
                                              rootQ: rootQ, rootU: 0.0,
                                              yLegacy, qFillLegacy,
                                              new ReverseKlPosteriorPolicy.Options(bisectionIterations: 60));

    double vStarLegacy = 0.0;
    for (int i = 0; i < mu.Length; i++)
    {
      vStarLegacy += yLegacy[i] * qFillLegacy[i];
    }
    double vStarDelta = Math.Abs(vStarNew - vStarLegacy);
    Console.WriteLine($"[V_bar regression]   vStar_new={vStarNew:F6} vStar_legacy={vStarLegacy:F6} delta={vStarDelta:E2}");
  }


  /// <summary>
  /// Sanity check that IterativeLargestDeficit and HamiltonClosedForm allocate
  /// the same total visits on a representative fixture (per-child counts may
  /// differ by 1 at tie boundaries).
  /// </summary>
  private static void TestAllocatorParity()
  {
    double[] piBar = { 0.40, 0.35, 0.25 };
    double[] currentN = { 10.0, 12.0, 8.0 };
    int budget = 20;

    short[] visitsIter = new short[piBar.Length];
    int placedIter = RPOVisitAllocator.Allocate(piBar, currentN, budget,
                                                visitsIter, default,
                                                new RPOAllocationOptions(RPOAllocationAlgorithm.IterativeLargestDeficit, true));

    short[] visitsHam = new short[piBar.Length];
    int placedHam = RPOVisitAllocator.Allocate(piBar, currentN, budget,
                                               visitsHam, default,
                                               new RPOAllocationOptions(RPOAllocationAlgorithm.HamiltonClosedForm, true));

    Console.WriteLine($"[Allocator parity]   iter={placedIter}:[{visitsIter[0]},{visitsIter[1]},{visitsIter[2]}] "
                    + $"hamilton={placedHam}:[{visitsHam[0]},{visitsHam[1]},{visitsHam[2]}]");
  }


  /// <summary>
  /// Verifies that the Sinkhorn-style fixed-point iteration converges and produces
  /// a state where the closed-form identity  q(a) = v* + lambda * (1 - mu(a) / pi_bar(a))
  /// holds for unvisited children (within tolerance).
  /// </summary>
  private static void TestFixedPointSelfConsistency()
  {
    // Mixed visited/unvisited case.
    double[] mu = { 0.40, 0.30, 0.20, 0.10 };
    double[] qIn = { 0.30, double.NaN, double.NaN, double.NaN };
    double parentQ = 0.20;
    double lambda = 0.10;

    // Sum mu (already 1.0, but track for safety).
    double sumMu = 0.0;
    for (int i = 0; i < mu.Length; i++) sumMu += mu[i];
    double[] muNorm = new double[mu.Length];
    for (int i = 0; i < mu.Length; i++) muNorm[i] = mu[i] / sumMu;

    // Seed: for unvisited, start at parentQ.
    double[] qWorking = new double[mu.Length];
    for (int i = 0; i < mu.Length; i++) qWorking[i] = double.IsNaN(qIn[i]) ? parentQ : qIn[i];

    double[] piBar = new double[mu.Length];
    RPOOptions opts = new(bisectionIterations: 60, bisectionResidualTol: 1e-9,
                          clampQ: true, minPriorProbability: 0.0);

    // Initial solve (iteration 0 in CBGPUCT.ScoreCalc terms).
    RegularizedPolicyOptimum.Solve(mu, qWorking, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                   piBar, default, out double vStar, opts, parentQ);

    // Run 5 fixed-point iterations.
    int iterations = 0;
    double maxDelta = double.PositiveInfinity;
    for (int iter = 0; iter < 30 && maxDelta > 1e-8; iter++)
    {
      maxDelta = 0.0;
      for (int i = 0; i < mu.Length; i++)
      {
        if (!double.IsNaN(qIn[i])) continue;
        if (!(piBar[i] > 1e-12)) continue;
        double qNew = parentQ + lambda * (1.0 - muNorm[i] / piBar[i]);
        if (qNew < -1.0) qNew = -1.0;
        else if (qNew > 1.0) qNew = 1.0;
        double d = Math.Abs(qNew - qWorking[i]);
        if (d > maxDelta) maxDelta = d;
        qWorking[i] = qNew;
      }
      RegularizedPolicyOptimum.Solve(mu, qWorking, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                     piBar, default, out vStar, opts, parentQ);
      iterations = iter + 1;
    }

    // Verify self-consistency: at the fixed point, v* should equal parentQ.
    double maxResidual = Math.Abs(vStar - parentQ);

    Console.WriteLine($"[FixedPoint A]       iters={iterations} |vStar-parentQ|={maxResidual:E2} "
                    + $"q_final=[{qWorking[0]:F3},{qWorking[1]:F3},{qWorking[2]:F3},{qWorking[3]:F3}] "
                    + $"piBar=[{piBar[0]:F3},{piBar[1]:F3},{piBar[2]:F3},{piBar[3]:F3}] vStar={vStar:F4} parentQ={parentQ:F4}");

    // Second case: deliberately bad initial FPU (highly optimistic for unvisited).
    // Should converge to a value much lower than the initial guess.
    double[] qWorking2 = new double[mu.Length];
    qWorking2[0] = 0.30;
    for (int i = 1; i < mu.Length; i++) qWorking2[i] = 0.95;
    double initialFpu = qWorking2[1];

    RegularizedPolicyOptimum.Solve(mu, qWorking2, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                   piBar, default, out vStar, opts, parentQ);

    iterations = 0;
    maxDelta = double.PositiveInfinity;
    for (int iter = 0; iter < 30 && maxDelta > 1e-8; iter++)
    {
      maxDelta = 0.0;
      for (int i = 1; i < mu.Length; i++)
      {
        if (!(piBar[i] > 1e-12)) continue;
        double qNew = parentQ + lambda * (1.0 - muNorm[i] / piBar[i]);
        if (qNew < -1.0) qNew = -1.0;
        else if (qNew > 1.0) qNew = 1.0;
        double d = Math.Abs(qNew - qWorking2[i]);
        if (d > maxDelta) maxDelta = d;
        qWorking2[i] = qNew;
      }
      RegularizedPolicyOptimum.Solve(mu, qWorking2, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                     piBar, default, out vStar, opts, parentQ);
      iterations = iter + 1;
    }
    Console.WriteLine($"[FixedPoint B]       iters={iterations} initialFPU={initialFpu:F3} -> "
                    + $"q_final=[{qWorking2[0]:F3},{qWorking2[1]:F3},{qWorking2[2]:F3},{qWorking2[3]:F3}] "
                    + $"piBar=[{piBar[0]:F3},{piBar[1]:F3},{piBar[2]:F3},{piBar[3]:F3}] vStar={vStar:F4}");

    // Third case: per-child FPU varied across unvisited (simulates Boltzmann or action-head FPU).
    // Initial q's are NOT self-consistent so the fixed point actually moves them.
    double[] qWorking3 = new double[mu.Length];
    qWorking3[0] = 0.30;     // visited
    qWorking3[1] = 0.10;     // optimistic-ish
    qWorking3[2] = -0.30;    // pessimistic
    qWorking3[3] = 0.50;     // very optimistic
    double[] qInitial = (double[])qWorking3.Clone();

    RegularizedPolicyOptimum.Solve(mu, qWorking3, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                   piBar, default, out vStar, opts, parentQ);

    iterations = 0;
    maxDelta = double.PositiveInfinity;
    for (int iter = 0; iter < 30 && maxDelta > 1e-8; iter++)
    {
      maxDelta = 0.0;
      for (int i = 1; i < mu.Length; i++)
      {
        if (!(piBar[i] > 1e-12)) continue;
        double qNew = parentQ + lambda * (1.0 - muNorm[i] / piBar[i]);
        if (qNew < -1.0) qNew = -1.0;
        else if (qNew > 1.0) qNew = 1.0;
        double d = Math.Abs(qNew - qWorking3[i]);
        if (d > maxDelta) maxDelta = d;
        qWorking3[i] = qNew;
      }
      RegularizedPolicyOptimum.Solve(mu, qWorking3, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                     piBar, default, out vStar, opts, parentQ);
      iterations = iter + 1;
    }
    Console.WriteLine($"[FixedPoint C]       iters={iterations} initial=[{qInitial[1]:F2},{qInitial[2]:F2},{qInitial[3]:F2}] -> "
                    + $"final=[{qWorking3[1]:F3},{qWorking3[2]:F3},{qWorking3[3]:F3}] "
                    + $"piBar=[{piBar[0]:F3},{piBar[1]:F3},{piBar[2]:F3},{piBar[3]:F3}] vStar={vStar:F4}");

    // Case D: production-realistic setting (2 iterations) starting from parentQ FPU.
    // Shows the mild-correction regime: partial movement toward the constraint v* = parentQ.
    double[] qWorking4 = new double[mu.Length];
    qWorking4[0] = 0.30;
    for (int i = 1; i < mu.Length; i++) qWorking4[i] = parentQ;

    RegularizedPolicyOptimum.Solve(mu, qWorking4, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                   piBar, default, out double vStar0, opts, parentQ);
    double piBar0Vis = piBar[0];
    double vStar0Save = vStar0;

    for (int iter = 0; iter < 2; iter++)
    {
      for (int i = 1; i < mu.Length; i++)
      {
        if (!(piBar[i] > 1e-12)) continue;
        double qNew = parentQ + lambda * (1.0 - muNorm[i] / piBar[i]);
        if (qNew < -1.0) qNew = -1.0;
        else if (qNew > 1.0) qNew = 1.0;
        qWorking4[i] = qNew;
      }
      RegularizedPolicyOptimum.Solve(mu, qWorking4, lambda, RPOAnchor.None, RPORegularization.ReverseKL,
                                     piBar, default, out vStar0, opts, parentQ);
    }
    Console.WriteLine($"[FixedPoint D iter=2] before piBar[0]={piBar0Vis:F3} vStar={vStar0Save:F4} | "
                    + $"after piBar[0]={piBar[0]:F3} vStar={vStar0:F4} "
                    + $"q_unvisited=[{qWorking4[1]:F3},{qWorking4[2]:F3},{qWorking4[3]:F3}]");
  }
}
