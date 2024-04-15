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
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

using Ceres.Base.DataType;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.LeafExpansion
{
  /// <summary>
  /// SIMD (AVX) code for selecting one or (typically) more children
  /// to be followed next according to PUCT in the tree descent.
  /// 
  /// Note that an attempt was made to pipelining/unroll the AVX code
  /// to improve performance, but it does not (easily) work 
  /// seemingly due to register pressure.
  /// </summary>
  public unsafe static class MCTSScoreCalcVector
  {
    public const int MAX_CHILDREN = 64;

    [ThreadStatic]
    static float[] childScoresTempBuffer;


    /// <summary>
    /// Entry point for the score calculation.
    /// 
    /// Given a set of information about the current node 
    /// the number of visits to be made to the subtree,
    /// an array of visit counts is returned 
    /// indicating the number of visits that should be made to each child.
    /// </summary>
    /// <param name="dualSelectorMode"></param>
    /// <param name="paramsSelect"></param>
    /// <param name="selectorID"></param>
    /// <param name="dynamicVLossBoost"></param>
    /// <param name="parentIsRoot"></param>
    /// <param name="parentN"></param>
    /// <param name="parentNInFlight"></param>
    /// <param name="qParent"></param>
    /// <param name="parentSumPVisited"></param>
    /// <param name="p"></param>
    /// <param name="w"></param>
    /// <param name="n"></param>
    /// <param name="nInFlight"></param>
    /// <param name="numChildren"></param>
    /// <param name="numVisitsToCompute"></param>
    /// <param name="outputScores"></param>
    /// <param name="outputChildVisitCounts"></param>
    internal static void ScoreCalcMulti(bool dualSelectorMode, ParamsSelect paramsSelect,
                                      int selectorID, float dynamicVLossBoost,
                                      bool parentIsRoot, float parentN, float parentNInFlight,
                                      float qParent, float parentSumPVisited,
                                      GatheredChildStats childStats,
                                      int numChildren, int numVisitsToCompute,
                                      Span<float> outputScores, Span<short> outputChildVisitCounts,
                                      float cpuctMultiplier,
                                      float actionHeadSelectionWeight)
    {
#if NOT
Note: Possible optimization/inefficiency: 
       the profiler shows about 5% of runtime here, 
       and the disassembly shows an unexplained zeroing 
       of seemingly about 500 bytes, possibly related to passing the Spans
       on to the Compute method at the end of the method.
  00007FFD0D4BAA0C  vxorps      xmm4,xmm4,xmm4  
  00007FFD0D4BAA10  mov         rax,0FFFFFFFFFFFFFF10h  
  00007FFD0D4BAA1A  vmovdqa     xmmword ptr [rax+rbp],xmm4  
  00007FFD0D4BAA1F  vmovdqa     xmmword ptr [rbp+rax+10h],xmm4  
  00007FFD0D4BAA25  vmovdqa     xmmword ptr [rbp+rax+20h],xmm4  
  00007FFD0D4BAA2B  add         rax,30h  
#endif

      // Saving output scores only makes sense when a single visit being computed
      Debug.Assert(!(outputScores != default && numVisitsToCompute > 1));
      Debug.Assert(numChildren <= MAX_CHILDREN);

      Debug.Assert(outputScores == default || outputScores.Length >= numChildren);
      Debug.Assert(numVisitsToCompute == 0 || outputChildVisitCounts.Length >= numChildren);

      float virtualLossMultiplier;
      if (ParamsSelect.VLossRelative)
      {
        virtualLossMultiplier = (qParent + paramsSelect.VirtualLossDefaultRelative + dynamicVLossBoost);
      }
      else
      {
        virtualLossMultiplier = paramsSelect.VirtualLossDefaultAbsolute;
      }

      float cpuctValue = cpuctMultiplier * paramsSelect.CalcCPUCT(parentIsRoot, dualSelectorMode, selectorID, parentN);

      // Compute qWhenNoChildren
      float fpuValue = -paramsSelect.CalcFPUValue(parentIsRoot);

      // TODO: to be more precise, parentSumPVisited should possibly be updated as we visit children
      bool useFPUReduction = paramsSelect.GetFPUMode(parentIsRoot) == ParamsSelect.FPUType.Reduction;
      float qWhenNoChildren = useFPUReduction ? (+qParent + fpuValue * MathF.Sqrt(parentSumPVisited)) : fpuValue;

      if (parentIsRoot
        && parentN > paramsSelect.RootCPUCTExtraMultiplierDivisor
        && paramsSelect.RootCPUCTExtraMultiplierExponent != 0)
      {
        cpuctValue *= MathF.Pow(parentN / paramsSelect.RootCPUCTExtraMultiplierDivisor,
                                paramsSelect.RootCPUCTExtraMultiplierExponent);
      }

      Compute(parentN, parentNInFlight, childStats, numChildren, numVisitsToCompute, outputScores,
              outputChildVisitCounts, virtualLossMultiplier,
              parentIsRoot ? paramsSelect.UCTRootNumeratorExponent : paramsSelect.UCTNonRootNumeratorExponent,
              cpuctValue, qWhenNoChildren,
              parentIsRoot ? paramsSelect.UCTRootDenominatorExponent : paramsSelect.UCTNonRootDenominatorExponent,
              actionHeadSelectionWeight);
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
    private static void Compute(float parentN, float parentNInFlight,
                                GatheredChildStats childStats,
                                int numChildren, int numVisitsToCompute,
                                Span<float> outputScores, Span<short> outputChildVisitCounts,
                                float virtualLossMultiplier, float uctParentPower,
                                float cpuctValue, float qWhenNoChildren, float uctDenominatorPower,
                                float actionHeadSelectionWeight)
    {
      // Load the vectors that do not change
      Span<float> nInFlight = childStats.InFlight.Span;

      // Make sure ThreadStatics are initialized, and get local copies for efficient access
      float[] localResultAVXScratch = childScoresTempBuffer;
      if (localResultAVXScratch == null)
      {
        InitializedForThread();
        localResultAVXScratch = childScoresTempBuffer;
      }

      int numVisits = 0;
      while ((numVisits < numVisitsToCompute)
          || (numVisits == 0 && numVisitsToCompute == 0) // just quering scores, no children to select
            )
      {
        // Get constant term handy
        float numVisitsByParentToChildren = parentNInFlight + ((parentN < 2) ? 1 : parentN - 1);
        float cpuctSqrtParentN = cpuctValue * ParamsSelect.UCTParentMultiplier(numVisitsByParentToChildren, uctParentPower);
        ComputeChildScores(childStats, numChildren, qWhenNoChildren, virtualLossMultiplier,
                           localResultAVXScratch, cpuctSqrtParentN, uctDenominatorPower, actionHeadSelectionWeight);

        // Save back to output scores (if these were requested)
        if (outputScores != default)
        {
          Debug.Assert(numVisits <= 1);

          Span<float> scoresSpan = new Span<float>(localResultAVXScratch).Slice(0, numChildren);
          scoresSpan.CopyTo(outputScores);
        }

        // Find the best child and record this visit
        int maxIndex = ArrayUtils.IndexOfElementWithMaxValue(localResultAVXScratch, numChildren);

        // Update either 3 or 4 items items to reflect this visit
        parentNInFlight += 1;
        nInFlight[maxIndex] += 1;
        numVisits += 1;
        if (outputChildVisitCounts != default) outputChildVisitCounts[maxIndex] += 1;

        int numRemainingVisits = numVisitsToCompute - numVisits;

        // If we just found our first child we repeatedly try to  
        // "jump ahead" by 10 visits at a time 
        // as long as the first top child child remains the best child.
        // This optimizes for the common case that one child is dominant,
        // and empriically reduces the number of calls to ComputeChildScores by more than 30%.
        // 
        // Note that would be possible to try this "jump ahead" technique
        // after not only the first visit, but in practice this did not improve performance.
        const int REPEATED_VISITS_DIVISOR = 10;
        int numAdditionalTryVisitsPerIteration = Math.Max(10, numRemainingVisits / REPEATED_VISITS_DIVISOR);
        if (numVisits == 1 && numRemainingVisits > numAdditionalTryVisitsPerIteration + 5)
        {
          int numSuccessfulVisitsAllIterations = 0;

          do
          {
            // Modify state to simulate additional visits to this top child
            float newNInFlight = nInFlight[maxIndex] += numAdditionalTryVisitsPerIteration;

            // Compute new child scores
            numVisitsByParentToChildren = newNInFlight + parentNInFlight + ((parentN < 2) ? 1 : parentN - 1);
            cpuctSqrtParentN = cpuctValue * ParamsSelect.UCTParentMultiplier(numVisitsByParentToChildren, uctParentPower);
            ComputeChildScores(childStats, numChildren, qWhenNoChildren, virtualLossMultiplier,
                                 localResultAVXScratch, cpuctSqrtParentN, uctDenominatorPower, actionHeadSelectionWeight);

            // Check if the best child was still the same
            if (maxIndex == ArrayUtils.IndexOfElementWithMaxValue(localResultAVXScratch, numChildren))
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
            if (outputChildVisitCounts != default)
              outputChildVisitCounts[maxIndex] += (short)numSuccessfulVisitsAllIterations;
          }
        }
      }
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
                                       int numChildren, float qWhenNoChildren,
                                       float virtualLossMultiplier, float[] computedChildScores,
                                       float cpuctSqrtParentN, float uctDenominatorPower,
                                       float actionHeadSelectionWeight)
    {
      if (actionHeadSelectionWeight == 0 && Avx.IsSupported)
      {
        ComputeChildScoresAVX(childStats, numChildren, qWhenNoChildren, virtualLossMultiplier, computedChildScores, cpuctSqrtParentN, uctDenominatorPower, actionHeadSelectionWeight);
      }
      else if (actionHeadSelectionWeight == 0 && AdvSimd.IsSupported)
      {
        // The SIMD version is about 3x as fast as non-SIMD.
        ComputeChildScoresARM(childStats, numChildren, qWhenNoChildren, virtualLossMultiplier, computedChildScores, cpuctSqrtParentN, uctDenominatorPower, actionHeadSelectionWeight);
      }
      else
      {
        ComputeChildScoresNonSIMD(childStats, numChildren, qWhenNoChildren, virtualLossMultiplier, computedChildScores, cpuctSqrtParentN, uctDenominatorPower, actionHeadSelectionWeight);
      }
    }


    private static void ComputeChildScoresAVX(GatheredChildStats childStats,
                                              int numChildren, float qWhenNoChildren,
                                              float virtualLossMultiplier, float[] computedChildScores,
                                              float cpuctSqrtParentN, float uctDenominatorPower,
                                              float actionHeadSelectionWeight)
    {
      int numBlocks = (numChildren / 8) + ((numChildren % 8 == 0) ? 0 : 1);

      Span<float> p = childStats.P.Span;
      Span<float> w = childStats.W.Span;
      Span<float> n = childStats.N.Span;
      Span<float> nInFlight = childStats.InFlight.Span;

      // Process in AVX blocks of 8 at a time
      int blockCount = 0;
      while (blockCount < numBlocks)
      {
        int startOffset = blockCount * 8;

        fixed (float* pNInFlight = &nInFlight[startOffset],
                      pP = &p[startOffset],
                      pW = &w[startOffset],
                      pN = &n[startOffset],
                      pComputedChildScores = &computedChildScores[startOffset])
        {
          // Load vector registers
          Vector256<float> vW = Avx.LoadAlignedVector256(pW);
          Vector256<float> vN = Avx.LoadAlignedVector256(pN);
          Vector256<float> vP = Avx.LoadAlignedVector256(pP);
          Vector256<float> vQWhenNoChildren = Vector256.Create(qWhenNoChildren);

          // Do computation
          Vector256<float> vNInFlight = Avx.LoadAlignedVector256(pNInFlight);
          Vector256<float> vScore = ComputeScoresAVX(vW, vN, vP, virtualLossMultiplier, cpuctSqrtParentN, uctDenominatorPower,
                                                      actionHeadSelectionWeight, vQWhenNoChildren, vNInFlight);
          Avx.Store(pComputedChildScores, vScore);
        }

        blockCount++;
      }
    }

    private static void ComputeChildScoresARM(GatheredChildStats childStats,
                                          int numChildren, float qWhenNoChildren,
                                          float virtualLossMultiplier, float[] computedChildScores,
                                          float cpuctSqrtParentN, float uctDenominatorPower,
                                          float actionHeadSelectionWeight)
    {
      int numBlocks = (numChildren / 4) + ((numChildren % 4 == 0) ? 0 : 1);

      Span<float> p = childStats.P.Span;
      Span<float> w = childStats.W.Span;
      Span<float> n = childStats.N.Span;
      Span<float> nInFlight = childStats.InFlight.Span;

      // Process in blocks of 4 at a time
      int blockCount = 0;
      while (blockCount < numBlocks)
      {
        int startOffset = blockCount * 4;

        fixed (float* pNInFlight = &nInFlight[startOffset],
                      pP = &p[startOffset],
                      pW = &w[startOffset],
                      pN = &n[startOffset],
                      pComputedChildScores = &computedChildScores[startOffset])
        {
          // Load vector registers
          Vector128<float> vW = AdvSimd.LoadVector128(pW);
          Vector128<float> vN = AdvSimd.LoadVector128(pN);
          Vector128<float> vP = AdvSimd.LoadVector128(pP);
          Vector128<float> vQWhenNoChildren = Vector128.Create(qWhenNoChildren);

          // Do computation
          Vector128<float> vNInFlight = AdvSimd.LoadVector128(pNInFlight);
          Vector128<float> vScore = ComputeScoresARM(vW, vN, vP, virtualLossMultiplier, cpuctSqrtParentN, uctDenominatorPower, actionHeadSelectionWeight,
                                                     vQWhenNoChildren, vNInFlight);
          AdvSimd.Store(pComputedChildScores, vScore);
        }

        blockCount++;
      }
    }

    /// <summary>
    /// Returns the value obtained by raising every element of a vector to a speciifed power.
    /// 
    /// Note that this implementation is not efficient, but is only used for experimental variants.
    /// </summary>
    /// <param name="values"></param>
    /// <param name="power"></param>
    /// <returns></returns>
    static Vector256<float> ToPowerAVX(Vector256<float> values, float power)
    {
      Span<float> valuesSource = stackalloc float[8];
      Span<float> valuesDest = stackalloc float[8];
      fixed (float* pValuesSource = &valuesSource[0])
      {
        Avx.Store(pValuesSource, values);
      }

      for (int i = 0; i < valuesDest.Length; i++)
      {
        valuesDest[i] = MathF.Pow(valuesSource[i], power);
      }

      fixed (float* pValues = &valuesDest[0])
      {
        return Avx.LoadVector256(pValues);
      }
    }


    /// <summary>
    /// Low-level AVX worker method that implements the CPUCT math.
    /// </summary>
    /// <param name="vW"></param>
    /// <param name="vN"></param>
    /// <param name="vP"></param>
    /// <param name="virtualLossMultiplier"></param>
    /// <param name="cpuctSqrtParentN"></param>
    /// <param name="uctDenominatorPower"></param>
    /// <param name="vQWhenNoChildren">fill-in value for Q for children having N=0 (larger values are better)</param>
    /// <param name="vNInFlight"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ComputeScoresAVX(Vector256<float> vW, Vector256<float> vN, Vector256<float> vP,
                                                     float virtualLossMultiplier,
                                                     float cpuctSqrtParentN, float uctDenominatorPower,
                                                     float actionHeadSelectionWeight,
                                                     Vector256<float> vQWhenNoChildren, Vector256<float> vNInFlight)
    {
      if (actionHeadSelectionWeight != 0)
      {
        throw new NotImplementedException();
      }

      Vector256<float> vNPlusNInFlight = Avx.Add(vN, vNInFlight);
      Vector256<float> vVirtualLossMultiplier = Vector256.Create(virtualLossMultiplier);

      Vector256<float> denominator = uctDenominatorPower switch
      {
        1.0f => vNPlusNInFlight,
        0.5f => Avx.Sqrt(vNPlusNInFlight),
        _ => ToPowerAVX(vNPlusNInFlight, uctDenominatorPower)
      };

      Vector256<float> vLossContrib = Avx.Multiply(vNInFlight, vVirtualLossMultiplier);

      // Compute U = ((p)(cpuct)(sqrt_parentN)) / (n + n_in_flight + 1)
      // Note that Vector.Create() is used below for constants rather than referencing statics
      // because the JIT has specific knowledge of inline Vector*.Create methods.
      Vector256<float> vCPUCTSqrtParentN = Vector256.Create(cpuctSqrtParentN);
      Vector256<float> vUNumerator = Avx.Multiply(vP, vCPUCTSqrtParentN);
      Vector256<float> vDenominator = Avx.Add(Vector256.Create(1f), denominator);
      Vector256<float> vU = Avx.Divide(vUNumerator, vDenominator);

      // Compute Q where the number of visits is not zero.
      // Note that subtraction here has effect of inverting, such that bigger Q are now "better".
      Vector256<float> vQWithChildren = Avx.Divide(Avx.Subtract(vLossContrib, vW), vNPlusNInFlight);

      Vector256<float> vQWithoutChildren = Avx.Add(vQWhenNoChildren, vLossContrib);
      Vector256<float> maskNoChildren = Avx.Compare(vNPlusNInFlight, Vector256.Create(0f), FloatComparisonMode.OrderedGreaterThanSignaling);

      Vector256<float> vQ = Avx.BlendVariable(vQWithoutChildren, vQWithChildren, maskNoChildren);

      Vector256<float> vScore = Avx.Add(vU, vQ);
      return vScore;
    }


    /// <summary>
    /// Low-level ARM SIMD worker method that implements the CPUCT math.
    /// </summary>
    /// <param name="vW"></param>
    /// <param name="vN"></param>
    /// <param name="vP"></param>
    /// <param name="virtualLossMultiplier"></param>
    /// <param name="cpuctSqrtParentN"></param>
    /// <param name="uctDenominatorPower"></param>
    /// <param name="vQWhenNoChildren">fill-in value for Q for children having N=0 (larger values are better)</param>
    /// <param name="vNInFlight"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ComputeScoresARM(Vector128<float> vW, Vector128<float> vN, Vector128<float> vP,
                                                     float virtualLossMultiplier,
                                                     float cpuctSqrtParentN, float uctDenominatorPower, float actionHeadSelectionWeight,
                                                     Vector128<float> vQWhenNoChildren, Vector128<float> vNInFlight)
    {
      if (actionHeadSelectionWeight != 0)
      {
        throw new NotImplementedException();
      }

      Vector128<float> vNPlusNInFlight = AdvSimd.Add(vN, vNInFlight);
      Vector128<float> vVirtualLossMultiplier = Vector128.Create(virtualLossMultiplier);
      Vector128<float> denominator = uctDenominatorPower switch
      {
        1.0f => vNPlusNInFlight,
        0.5f => AdvSimd.Arm64.Sqrt(vNPlusNInFlight),
        _ => throw new NotImplementedException()// ToPower(vNPlusNInFlight, uctDenominatorPower)
      };

      Vector128<float> vLossContrib = AdvSimd.Multiply(vNInFlight, vVirtualLossMultiplier);

      // Compute U = ((p)(cpuct)(sqrt_parentN)) / (n + n_in_flight + 1)
      // Note that Vector.Create() is used below for constants rather than referencing statics
      // because the JIT has specific knowledge of inline Vector*.Create methods.
      Vector128<float> vCPUCTSqrtParentN = Vector128.Create(cpuctSqrtParentN);
      Vector128<float> vUNumerator = AdvSimd.Multiply(vP, vCPUCTSqrtParentN);
      Vector128<float> vDenominator = AdvSimd.Add(Vector128.Create(1f), denominator);
      Vector128<float> vU = AdvSimd.Arm64.Divide(vUNumerator, vDenominator);

      // Compute Q where the number of visits is not zero.
      // Note that subtraction here has effect of inverting, such that bigger Q are now "better".
      Vector128<float> vQWithChildren = AdvSimd.Arm64.Divide(AdvSimd.Subtract(vLossContrib, vW), vNPlusNInFlight);

      Vector128<float> vQWithoutChildren = AdvSimd.Add(vQWhenNoChildren, vLossContrib);
      Vector128<float> maskNoChildren = AdvSimd.CompareGreaterThan(vNPlusNInFlight, Vector128.Create(0f));
      //Vector128<float> maskNoChildren = Avx.Compare(vNPlusNInFlight, Vector128.Create(0f), FloatComparisonMode.OrderedGreaterThanSignaling);

      Vector128<float> vQ = AdvSimd.BitwiseSelect(maskNoChildren, vQWithChildren, vQWithoutChildren);

      Vector128<float> vScore = AdvSimd.Add(vU, vQ);

      return vScore;
    }


    #region Non-SIMD versions

    /// <summary>
    /// Direct C# version of ComputeChildScores without use of SIMD.
    /// About 60% as fast as the AVX version.
    /// </summary>
    private unsafe static void ComputeChildScoresNonSIMD(GatheredChildStats childStats,
                                                         int numChildren, float qWhenNoChildren,
                                                         float virtualLossMultiplier, float[] computedChildScores,
                                                         float cpuctSqrtParentN, float uctDenominatorPower, float actionHeadSelectionWeight)
    {
      ComputeScoresNonSIMD(numChildren, childStats.W.Span, childStats.N.Span, childStats.P.Span, childStats.A.Span,
                              virtualLossMultiplier, cpuctSqrtParentN, uctDenominatorPower,
                           qWhenNoChildren, childStats.InFlight.Span, computedChildScores, actionHeadSelectionWeight);
    }


    private static void ComputeScoresNonSIMD(int numScores, Span<float> vW, Span<float> vN, Span<float> vP, Span<float> vA,
                                             float virtualLossMultiplier,
                                             float cpuctSqrtParentN, float uctDenominatorPower,
                                             float qWhenNoChildren, Span<float> vNInFlight,
                                             Span<float> outputVScore, float actionHeadSelectionWeight)
    {
      for (int i = 0; i < numScores; i++)
      {
        float nPlusNInFlight = vN[i] + vNInFlight[i];
        float _denominator;
        if (uctDenominatorPower == 1.0f)
        {
          _denominator = nPlusNInFlight;
        }
        else if (uctDenominatorPower == 0.5f)
        {
          _denominator = MathF.Sqrt(nPlusNInFlight);
        }
        else
        {
          _denominator = MathF.Pow(nPlusNInFlight, uctDenominatorPower);
        }

        float _vQ;
        float _vLossContrib = vNInFlight[i] * virtualLossMultiplier;
        if (nPlusNInFlight > 0)
        {
          _vQ = (_vLossContrib - vW[i]) / nPlusNInFlight;
        }
        else if (actionHeadSelectionWeight != 0)
        {
          float defaultQ = qWhenNoChildren + _vLossContrib; 
          float weight = actionHeadSelectionWeight;
//                                if (vP[i] < 0.02) weight *= 0.333f; // <--------- TEST ----------
          _vQ = weight * vA[i] + (1 - weight) * defaultQ;
        }
        else        
        {
          _vQ = qWhenNoChildren + _vLossContrib;
        }

        // U
        float _vUNumerator = vP[i] * cpuctSqrtParentN;
        float _vDenominator = 1 + _denominator;
        float _vU = _vUNumerator / _vDenominator;

        outputVScore[i] = _vU + _vQ;
      }
    }

    #endregion


    /// <summary>
    /// Initialization code that should be called once for each tread 
    /// executing score calculations.
    /// </summary>
    private static void InitializedForThread()
    {
      if (childScoresTempBuffer == null)
      {
        childScoresTempBuffer = new float[MAX_CHILDREN];
      }
    }

  }

}
