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
}
