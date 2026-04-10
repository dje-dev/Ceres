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
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

using Ceres.Base.DataType;
using Ceres.Base.Math;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// SIMD (AVX) code for selecting one or (typically) more children
/// to be followed next according to PUCT in the tree descent,
/// using a formula to balance exploration vs. exploitation.
/// 
/// Note that AVX512 is not currently automatically used by System.Numerics.Vector even in .NET 10:
/// "We want to make the default 512 bit in the future..."
/// though an environment variable may enable this:
///   set DOTNET_MaxVectorTBitWidth = 512
/// </summary>
public unsafe static class PUCTScoreCalcVector
{
  public const int MAX_CHILDREN = 64;

  // ThreadStatic buffer for SIMD-aligned per-child Q values (avoids stackalloc per call)
  [ThreadStatic]
  private static double[] t_simdQChildBuffer;

  /// <summary>
  /// Static variable available for debugging purposes to
  /// control of SIMD versions of PUCT logic are used
  /// instead of C# fallback.
  /// </summary>
  public static bool ENABLE_SIMD_CALCS = true; 


  /// <summary>
  /// Entry point for the score calculation.
  /// 
  /// Given a set of information about the current node 
  /// the number of visits to be made to the subtree, returns:
  ///   - a Span of visit counts indicating the number of visits warranted for each child
  ///   - optionally a Span of the computed child PUCT scores
  /// </summary>
  /// <param name="paramsSelect"></param>
  /// <param name="parentIsRoot"></param>
  /// <param name="parentN"></param>
  /// <param name="parentNInFlight"></param>
  /// <param name="qParent"></param>
  /// <param name="parentSumPVisited"></param>
  /// <param name="childStats"></param>
  /// <param name="qWhenNoChildrenPerChild"></param>
  /// <param name="numChildren"></param>
  /// <param name="numVisitsToCompute"></param>
  /// <param name="outputScores"></param>
  /// <param name="outputChildVisitCounts"></param>
  /// <param name="cpuctMultiplier"></param>
  /// <param name="thresholdPUCTSuboptimalityReject"></param>
  /// <returns></returns>
  internal static int ScoreCalcMulti(ParamsSelect paramsSelect,
                                     bool parentIsRoot, int parentN, double parentNInFlight,
                                     double qParent, double parentSumPVisited,
                                     GatheredChildStats childStats,
                                     double[] qWhenNoChildrenPerChild,
                                     int numChildren, int numVisitsToCompute,
                                     Span<double> outputScores, Span<short> outputChildVisitCounts,
                                     double cpuctMultiplier,
                                     float thresholdPUCTSuboptimalityReject)
  {
    Debug.Assert(!double.IsNaN(qParent));

    // Saving output scores only makes sense when a single visit being computed
    Debug.Assert(!(!outputScores.IsEmpty && numVisitsToCompute > 1));
    Debug.Assert(numChildren <= MAX_CHILDREN);

    Debug.Assert(outputScores.IsEmpty || outputScores.Length >= numChildren);
    Debug.Assert(numVisitsToCompute == 0 || outputChildVisitCounts.Length >= numChildren);

    float virtualLossMultiplier;
    if (ParamsSelect.VLossRelative)
    {
      virtualLossMultiplier = (float)qParent + paramsSelect.VirtualLossDefaultRelative;
    }
    else
    {
      virtualLossMultiplier = paramsSelect.VirtualLossDefaultAbsolute;
    }

    double cpuctValue = cpuctMultiplier * paramsSelect.CalcCPUCT(parentIsRoot, parentN);

    // Compute qWhenNoChildren based on FPU mode
    // TODO: to be more precise, parentSumPVisited should possibly be updated as we visit children
    double qWhenNoChildren = paramsSelect.CalcQWhenNoChildren(parentIsRoot, qParent, parentSumPVisited);

    const bool DUMP_Q_WHEN_NO_CHILDREN = false;
    if (DUMP_Q_WHEN_NO_CHILDREN && qWhenNoChildrenPerChild != null)
    {
      Console.WriteLine($"\r\nN= {parentN} Q={qParent}");
      for (int i = 0; i < numChildren; i++)
      {
        double delta = qWhenNoChildrenPerChild[i] - qWhenNoChildren;
        Console.WriteLine($"{100 * childStats.P.Span[i],6:F0}%  {(childStats.W.Span[i] / childStats.N.Span[i]),6:F2}  DELTA= {delta,6:F2}  {qWhenNoChildrenPerChild[i],6:F2}  was: {qWhenNoChildren,6:F2}");
      }
    }

    if (parentIsRoot
      && parentN > paramsSelect.RootCPUCTExtraMultiplierDivisor
      && paramsSelect.RootCPUCTExtraMultiplierExponent != 0)
    {
      cpuctValue *= Math.Pow(parentN / paramsSelect.RootCPUCTExtraMultiplierDivisor,
                              paramsSelect.RootCPUCTExtraMultiplierExponent);
    }

    int numVisitsAccepted = Compute(parentN, qParent, parentNInFlight, childStats, numChildren, numVisitsToCompute, outputScores,
                                    outputChildVisitCounts, virtualLossMultiplier,
                                    parentIsRoot ? paramsSelect.UCTRootNumeratorExponent : paramsSelect.UCTNonRootNumeratorExponent,
                                    cpuctValue, qWhenNoChildren, qWhenNoChildrenPerChild,
                                    parentIsRoot ? paramsSelect.UCTRootDenominatorExponent : paramsSelect.UCTNonRootDenominatorExponent,
                                    thresholdPUCTSuboptimalityReject);
    return numVisitsAccepted;
  }



  /// <summary>
  /// Worker method that coordinates looping over all the requested visits,
  /// including a performance optimization that attempts to 
  /// detect the condition where many consecutive visits will be made to the same child.
  /// </summary>
  /// <param name="parentN"></param>
  /// <param name="parentNInFlight"></param>
  /// <param name="p"></param>
  /// <param name="w"></param>
  /// <param name="n"></param>
  /// <param name="nInFlight"></param>
  /// <param name="numChildren"></param>
  /// <param name="numVisitsToCompute"></param>
  /// <param name="outputScores"></param>
  /// <param name="outputChildVisitCounts"></param>
  /// <param name="numBlocks"></param>
  /// <param name="virtualLossMultiplier"></param>
  /// <param name="uctParentPower"></param>
  /// <param name="cpuctValue"></param>
  /// <param name="qWhenNoChildren"></param>
  /// <param name="uctDenominatorPower"></param>
  [SkipLocalsInit]
  private static int Compute(int parentN, double qParent, double parentNInFlight,
                             GatheredChildStats childStats,
                             int numChildren, int numVisitsToCompute,
                             Span<double> outputScores, Span<short> outputChildVisitCounts,
                             float virtualLossMultiplier, float uctParentPower,
                             double cpuctValue,
                             double qWhenNoChildren, double[] qWhenNoChildrenPerChild,
                             double uctDenominatorPower,
                             float thresholdPUCTSuboptimalityReject)
  {
    // Load the vectors that do not change
    Span<double> nInFlight = childStats.NInFlightAdjusted.Span;


    int maxScratchChildren = (int)MathUtils.RoundedUp(Math.Min(MAX_CHILDREN, numChildren + numVisitsToCompute), Vector<double>.Count);
    Span<double> childScores = stackalloc double[maxScratchChildren];

    int numVisits = 0;
    int numTooSuboptimal = 0;

    while (numVisits < numVisitsToCompute
        || numVisits == 0 && numVisitsToCompute == 0) // just querying scores, no children to select
    {
      // Get constant term handy
      double numVisitsByParentToChildren = parentNInFlight + (parentN < 2 ? 1 : parentN - 1);
      double cpuctSqrtParentN = cpuctValue * ParamsSelect.UCTParentMultiplier(numVisitsByParentToChildren, uctParentPower);
      ComputeChildScores(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild, virtualLossMultiplier,
                         childScores, cpuctSqrtParentN, uctDenominatorPower);

      // Assert that none of the scores were NaN.
#if DEBUG
      // TODO: remove conditional compilation once TensorPrimitives versioning issue resolved.
      Debug.Assert(!double.IsNaN(TensorPrimitives.Max(childScores[..numChildren])));
#endif
      // Save back to output scores (if these were requested)
      if (!outputScores.IsEmpty)
      {
        Debug.Assert(numVisits <= 1);

        Span<double> scoresSpan = childScores[..numChildren];
        scoresSpan.CopyTo(outputScores);
      }

      if (numVisitsToCompute == 0)
      {
        return numVisits;
      }

      // Find the best child and record this visit
      int maxIndex = ArrayUtils.IndexOfElementWithMaxValue(childScores, numChildren);

      if (thresholdPUCTSuboptimalityReject < float.MaxValue)
      {
        // Determine suboptimality of this child wrt. parent Q.
        double thisQ = childStats.W.Span[maxIndex] / childStats.N.Span[maxIndex];
        double qSuboptimality = thisQ + qParent;

        // Allow at least 1 and up to 10% of requested visits to exceed the limit.
        int maxVisitsAllowedOverSuboptimalityLimit = 1 + parentN / 10;
        if (qSuboptimality > thresholdPUCTSuboptimalityReject)
        {
          numTooSuboptimal++;

          if (numTooSuboptimal > maxVisitsAllowedOverSuboptimalityLimit)
          {
//            Console.WriteLine($"Reducing visit count from {numVisitsToCompute} to {numVisits} at parentN= {parentN}");
            numTooSuboptimal++;
            return numVisits;
          }
        }
      }

      // Update items to reflect this visit
      parentNInFlight += 1;
      nInFlight[maxIndex] += 1;
      numVisits += 1;

      outputChildVisitCounts[maxIndex] += 1;

      int numRemainingVisits = numVisitsToCompute - numVisits;

      // If we just found our first child we repeatedly try to  
      // "jump ahead" by 10 visits at a time 
      // as long as the first top child child remains the best child.
      // This optimizes for the common case that one child is dominant,
      // and empirically reduces the number of calls to ComputeChildScores by more than 30%. 
      const int REPEATED_VISITS_DIVISOR = 10;
      int numAdditionalTryVisitsPerIteration = Math.Max(10, numRemainingVisits / REPEATED_VISITS_DIVISOR);
      if (numVisits == 1 && numRemainingVisits > numAdditionalTryVisitsPerIteration + 5)
      {
        int numSuccessfulVisitsAllIterations = 0;

        do
        {
          // Modify state to simulate additional visits to this top child
          double newNInFlight = nInFlight[maxIndex] += numAdditionalTryVisitsPerIteration;

          // Compute new child scores
          numVisitsByParentToChildren = newNInFlight + parentNInFlight + (parentN < 2 ? 1 : parentN - 1);
          cpuctSqrtParentN = cpuctValue * ParamsSelect.UCTParentMultiplier(numVisitsByParentToChildren, uctParentPower);
          ComputeChildScores(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild, virtualLossMultiplier,
                             childScores, cpuctSqrtParentN, uctDenominatorPower);

          // Check if the best child was still the same
          if (maxIndex == ArrayUtils.IndexOfElementWithMaxValue(childScores, numChildren))
          {
            // Child remained same, increment successful count
            numSuccessfulVisitsAllIterations += numAdditionalTryVisitsPerIteration;
          }
          else
          {
            // Failed, back out the last update to nInFlight and stop iterating
            nInFlight[maxIndex] -= numAdditionalTryVisitsPerIteration;

            break;
          }

          numAdditionalTryVisitsPerIteration = Math.Max(10, numRemainingVisits / REPEATED_VISITS_DIVISOR);
        } while (numRemainingVisits - numSuccessfulVisitsAllIterations > numAdditionalTryVisitsPerIteration);

        if (numSuccessfulVisitsAllIterations > 0)
        {
          // The nInFlight have already been kept continuously up to date
          // but need to update the other items to reflect these visits
          parentNInFlight += numSuccessfulVisitsAllIterations;
          numVisits += numSuccessfulVisitsAllIterations;
          if (!outputChildVisitCounts.IsEmpty)
          {
            outputChildVisitCounts[maxIndex] += (short)numSuccessfulVisitsAllIterations;
          }
        }
      }
    }

    return numVisits;
  }


  /// <summary>
  /// Computes the PUCT child scores for this node into computedChildScores.
  /// </summary>
  /// <param name="p"></param>
  /// <param name="w"></param>
  /// <param name="n"></param>
  /// <param name="nInFlight"></param>
  /// <param name="numChildren"></param>
  /// <param name="qWhenNoChildren"></param>
  /// <param name="virtualLossMultiplier"></param>
  /// <param name="computedChildScores"></param>
  /// <param name="cpuctSqrtParentN"></param>
  /// <param name="uctDenominatorPower"></param>
  private static void ComputeChildScores(GatheredChildStats childStats,
                                         int numChildren, 
                                         double qWhenNoChildren, double [] qWhenNoChildrenPerChild,
                                         double virtualLossMultiplier, Span<double> computedChildScores,
                                         double cpuctSqrtParentN, double uctDenominatorPower)
  {
    // Note: SIMD path blends action into Q globally via weight (Q = (1-w)*Q + w*A).
    //       The new FPUType.ActionHead mode uses action values as per-child FPU instead,
    //       which flows through qWhenNoChildrenPerChild and does not require the blending logic.

#if OLD_ACTION_COMMENT
    Need to review / harmonize logic between these two methods(pick which one is intended).
Findings comparing ComputeChildScoresSIMD vs ComputeChildScoresNonSIMD(ignoring commented -out code):
	Action - head logic not equivalent(High)
	SIMD path: Only blends action head into Q if ACTION_ENABLED is defined, 
  using Q = (1 - w)*Q + w*A for all items when weight != 0.
	Non - SIMD path: Always compiled and applies a different rule
  only for unvisited moves (i > 0 && N[i] == 0 && weight != 0): 
    Q = max(Q, A[i] + 0.10).No global blending.
This changes selection behavior even when ACTION_ENABLED is not defined and adds a fixed +0.10 offset floor.
#endif

    if (ENABLE_SIMD_CALCS && Vector.IsHardwareAccelerated)
    {
      ComputeChildScoresSIMD(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild,
                             virtualLossMultiplier, computedChildScores,
                             cpuctSqrtParentN, uctDenominatorPower);
    }
    else
    {
      ComputeChildScoresNonSIMD(childStats, numChildren, qWhenNoChildren, qWhenNoChildrenPerChild,
                                virtualLossMultiplier, computedChildScores, 
                                cpuctSqrtParentN, uctDenominatorPower);
    }
  }


  private static void ComputeChildScoresSIMD(GatheredChildStats childStats,
                                             int numChildren, 
                                             double qWhenNoChildren, double[] qWhenNoChildrenPerChild,
                                             double virtualLossMultiplier, Span<double> computedChildScores,
                                             double cpuctSqrtParentN, double uctDenominatorPower)
  {
    int simdWidth = Vector<double>.Count;
    int numBlocks = numChildren / simdWidth + (numChildren % simdWidth == 0 ? 0 : 1);

    Span<double> p = childStats.P.Span;
    Span<double> w = childStats.W.Span;
    Span<double> n = childStats.N.Span;
#if ACTION_ENABLED
    Span<double> a = childStats.A.Span;
#endif
    Span<double> nInFlight = childStats.NInFlightAdjusted.Span;

    int blockCount = 0;
    while (blockCount < numBlocks)
    {
      int startOffset = blockCount * simdWidth;

      // Load vectors from spans (caller guarantees adequate padding; no tail handling required)
      Vector<double> vW = new(w[startOffset..]);
      Vector<double> vN = new(n[startOffset..]);
      Vector<double> vP = new(p[startOffset..]);
#if ACTION_ENABLED
      Vector<double> vA = new Vector<double>(a[startOffset..]);
#endif
      Vector<double> vQWhenNoChildren;
      if (qWhenNoChildrenPerChild != null)
      {
        double[] qPerChildPadded = t_simdQChildBuffer ??= new double[Vector<double>.Count];
        int remaining = qWhenNoChildrenPerChild.Length - startOffset;
        for (int i = 0; i < simdWidth; i++)
        {
          qPerChildPadded[i] = i < remaining ? qWhenNoChildrenPerChild[startOffset + i] : qWhenNoChildren;
        }
        vQWhenNoChildren = new Vector<double>(qPerChildPadded);
      }
      else
      {
        vQWhenNoChildren = new Vector<double>(qWhenNoChildren);
      }
      Vector<double> vNInFlight = new(nInFlight[startOffset..]);

      Vector<double> vScore = ComputeScoresSIMD(vW, vN, vP,
#if ACTION_ENABLED
                                                vA,
#endif
                                                virtualLossMultiplier, cpuctSqrtParentN, uctDenominatorPower,
                                                vQWhenNoChildren, vNInFlight);

      vScore.CopyTo(computedChildScores[startOffset..]);

      blockCount++;
    }
  }
  


#if FEATURE_UNCERTAINTY_SCALING
      const double AVG_UV = 10;
      const double POW = 0.5; // if not 0.5, need to use ToPowerAVX below
      const double MULTIPLIER = 0.15;

      // Take sqrt(uv) and approximately center in a range approximately [-3, +3]
      Vector256<double> vUAdj = Avx.Sqrt(vUV); // ** NOTE: Must use POW=0.5 above!
      //Vector256<double> vUAdj = ToPowerAVX(vUV,POW);
      vUAdj = Avx.Subtract(vUAdj, Vector256.Create(MathF.Pow(AVG_UV, POW)));

      // Replace elements in vuADJ with 0 if the corresponding element in nPlusNInFlightPlus1 is identically zero
      Vector256<double> mask = Avx.Compare(vNPlusNInFlight, Vector256.Create(0f), FloatComparisonMode.OrderedEqualNonSignaling);
      vUAdj = Avx.AndNot(mask, vUAdj);

      if (false) 
      {
        // Replace uncertainty adjustment with 0 for nodes which are better than parent
        Vector256<double> v = Avx.Divide(vW, vN);
        Vector256<double> mask1 = Avx.Compare(v, Vector256.Create(-qParent), FloatComparisonMode.OrderedLessThanNonSignaling);
        vUAdj = Avx.AndNot(mask1, vUAdj);
      }

      // Scale down adjustment by some factor
      vUAdj = Avx.Multiply(vUAdj, Vector256.Create(MULTIPLIER));

      // Divide adjustment by sqrt(N+NInFlight+1) to reduce influence as more visits are made
      Vector256<double> nPlusNInFlightPlus1 = Avx.Add(vNPlusNInFlight, Vector256.Create(1f));

      // Reduce magnitude of adjustment as more visits are made
      vUAdj = Avx.Divide(vUAdj, nPlusNInFlightPlus1); // divide by number of visits already made

      // Finally, center around 1 instead of 0
      vUAdj = Avx.Add(vUAdj, Vector256.Create(1f));
      denominator = Avx.Divide(denominator, vUAdj);
    }
#endif

  #region Non-SIMD versions

  /// <summary>
  /// Direct C# version of ComputeChildScores without use of SIMD.
  /// About 60% as fast as the AVX version.
  /// </summary>
  private unsafe static void ComputeChildScoresNonSIMD(GatheredChildStats childStats,
                                                       int numChildren, 
                                                       double qWhenNoChildren, double[] qWhenNoChildrenPerChild,
                                                       double virtualLossMultiplier, Span<double> computedChildScores,
                                                       double cpuctSqrtParentN, double uctDenominatorPower)
  {
    ComputeScoresNonSIMD(numChildren, childStats.W.Span, childStats.N.Span, 
                         childStats.P.Span, childStats.A.Span,
                         virtualLossMultiplier, cpuctSqrtParentN, uctDenominatorPower,
                         qWhenNoChildren, qWhenNoChildrenPerChild,
                         childStats.NInFlightAdjusted.Span, computedChildScores);
  }


  private static void ComputeScoresNonSIMD(int numScores, Span<double> vW, Span<double> vN, Span<double> vP, Span<double> vA,
                                           double virtualLossMultiplier,
                                           double cpuctSqrtParentN, 
                                           double uctDenominatorPower,
                                           double qWhenNoChildren, double[] qWhenNoChildrenPerChild, 
                                           Span<double> vNInFlight,
                                           Span<double> outputVScore)
  {
    for (int i = 0; i < numScores; i++)
    {
      double nPlusNInFlight = vN[i] + vNInFlight[i];
      double _denominator;
      if (uctDenominatorPower == 1.0f)
      {
        _denominator = nPlusNInFlight;
      }
      else if (uctDenominatorPower == 0.5f)
      {
        _denominator = Math.Sqrt(nPlusNInFlight);
      }
      else
      {
        _denominator = Math.Pow(nPlusNInFlight, uctDenominatorPower);
      }

      double _vQ;
      double _vLossContrib = vNInFlight[i] * virtualLossMultiplier;
      if (nPlusNInFlight > 0)
      {
        _vQ = (_vLossContrib - vW[i]) / nPlusNInFlight;
      }
      else
      {
        double thisQWhenNoChildren = qWhenNoChildrenPerChild != null ? qWhenNoChildrenPerChild[i] : qWhenNoChildren;
        _vQ = thisQWhenNoChildren + _vLossContrib;
      }

      // [Experimental action blending, superseded by FPUType.ActionHead mode]
      // if (i > 0 && vN[i] == 0 && actionHeadSelectionWeight != 0)
      // {
      //   double aDiff = vA[i] - vA[0];
      //   _vQ = Math.Max(_vQ, vA[i] + 0.10f);
      // }

      // U
      double _vUNumerator = vP[i] * cpuctSqrtParentN;
      double _vDenominator = 1 + _denominator;
      double _vU = _vUNumerator / _vDenominator;

      outputVScore[i] = _vU + _vQ;
    }
  }

  #endregion


  #region Platform-agnostic

  /// <summary>
  /// Platform-agnostic vectorized worker that implements the CPUCT math using System.Numerics.Vector.
  /// The JIT maps this to AVX/AVX2 on x86 and AdvSimd on ARM64. It does not currently auto-use AVX-512 for Vector<T>.
  /// </summary>
  /// <remarks>
  /// Caller guarantees input spans are sized in multiples of Vector<double>.Count, so no tail handling here.
  /// </remarks>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static Vector<double> ComputeScoresSIMD(Vector<double> vW, Vector<double> vN, Vector<double> vP,
#if ACTION_ENABLED
                                                 Vector<double> vA,
#endif
                                                 double virtualLossMultiplier,
                                                 double cpuctSqrtParentN, double uctDenominatorPower,
                                                 Vector<double> vQWhenNoChildren, Vector<double> vNInFlight)
  {
    Vector<double> vNPlusNInFlight = vN + vNInFlight;
    Vector<double> vVirtualLossMultiplier = new Vector<double>(virtualLossMultiplier);

    Vector<double> denominator;
    if (uctDenominatorPower == 1.0)
    {
      denominator = vNPlusNInFlight;
    }
    else if (uctDenominatorPower == 0.5)
    {
      denominator = Vector.SquareRoot(vNPlusNInFlight);
    }
    else
    {
      denominator = ToPowerVector(vNPlusNInFlight, uctDenominatorPower);
    }

    Vector<double> vLossContrib = vNInFlight * vVirtualLossMultiplier;

    Vector<double> vCPUCTSqrtParentN = new(cpuctSqrtParentN);
    Vector<double> vUNumerator = vP * vCPUCTSqrtParentN;
    Vector<double> vDenominator = Vector<double>.One + denominator;
    Vector<double> vU = vUNumerator / vDenominator;

    Vector<double> vQWithChildren = (vLossContrib - vW) / vNPlusNInFlight;
    Vector<double> vQWithoutChildren = vQWhenNoChildren + vLossContrib;
    Vector<long> maskNoChildren = Vector.GreaterThan(vNPlusNInFlight, Vector<double>.Zero);
    Vector<double> vQ = Vector.ConditionalSelect(maskNoChildren, vQWithChildren, vQWithoutChildren);

    Vector<double> vScore = vU + vQ;
    return vScore;
  }


  // Platform-agnostic power for System.Numerics.Vector<double>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  static Vector<double> ToPowerVector(Vector<double> values, double power)
  {
    Span<double> buf = stackalloc double[Vector<double>.Count];
    values.CopyTo(buf);
    for (int i = 0; i < buf.Length; i++)
    {
      buf[i] = Math.Pow(buf[i], power);
    }
    return new Vector<double>(buf);
  }

  #endregion
}
