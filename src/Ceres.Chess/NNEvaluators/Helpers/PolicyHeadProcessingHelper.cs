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
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Chess.NNEvaluators.Helpers;

/// <summary>
/// Static helper class providing shared policy processing functionality
/// for neural network evaluators that support dual policy heads with
/// optional blending and per-head temperature adjustments.
/// </summary>
public static class PolicyHeadProcessingHelper
{
  /// <summary>
  /// Validates that the specified policy options are supported by the evaluator.
  /// Throws NotSupportedException if advanced policy features are requested but not supported.
  /// </summary>
  /// <param name="options">The evaluator options to validate</param>
  /// <param name="hasPolicySecondary">Whether the evaluator has a secondary policy head</param>
  /// <param name="evaluatorTypeName">Name of the evaluator type for error messages</param>
  /// <exception cref="NotSupportedException">Thrown when unsupported options are requested</exception>
  public static void ValidatePolicyOptionsSupported(NNEvaluatorOptions options,
                                                    bool hasPolicySecondary,
                                                    string evaluatorTypeName)
  {
    if (options == null)
    {
      return;
    }

    bool requiresPolicy2 = options.FractionPolicyHead2 != 0;
    bool requiresPerHeadTemp = options.Policy1Temperature != NNEvaluatorOptions.DEFAULT_POLICY1_TEMPERATURE
                            || options.Policy2Temperature != NNEvaluatorOptions.DEFAULT_POLICY2_TEMPERATURE;

    if (requiresPolicy2 && !hasPolicySecondary)
    {
      throw new NotSupportedException(
        $"{evaluatorTypeName} does not have a secondary policy head but FractionPolicyHead2={options.FractionPolicyHead2} was specified. " +
        "Use a network with a policy2 output or set FractionPolicyHead2=0.");
    }

    if (requiresPerHeadTemp && !hasPolicySecondary)
    {
      throw new NotSupportedException(
        $"{evaluatorTypeName} does not have a secondary policy head but per-head policy temperatures were specified " +
        $"(P1TEMP={options.Policy1Temperature}, P2TEMP={options.Policy2Temperature}). " +
        "Use a network with a policy2 output or use the default temperature values (1.0).");
    }
  }


  /// <summary>
  /// Throws NotSupportedException for evaluators that don't support advanced policy features.
  /// Call this from evaluators that only support basic policy processing.
  /// </summary>
  /// <param name="options">The evaluator options to check</param>
  /// <param name="evaluatorTypeName">Name of the evaluator type for error messages</param>
  /// <exception cref="NotSupportedException">Thrown when advanced policy options are requested</exception>
  public static void ThrowIfAdvancedPolicyFeaturesRequested(NNEvaluatorOptions options, string evaluatorTypeName)
  {
    if (options == null)
    {
      return;
    }

    if (options.FractionPolicyHead2 != 0)
    {
      throw new NotSupportedException(
        $"{evaluatorTypeName} does not support Policy2 blending (FractionPolicyHead2={options.FractionPolicyHead2}). " +
        "Use NNEvaluatorTensorRT or NNEvaluatorONNX for dual-head policy support.");
    }

    if (options.Policy1Temperature != NNEvaluatorOptions.DEFAULT_POLICY1_TEMPERATURE
     || options.Policy2Temperature != NNEvaluatorOptions.DEFAULT_POLICY2_TEMPERATURE)
    {
      throw new NotSupportedException(
        $"{evaluatorTypeName} does not support per-head policy temperatures " +
        $"(P1TEMP={options.Policy1Temperature}, P2TEMP={options.Policy2Temperature}). " +
        "Use NNEvaluatorTensorRT or NNEvaluatorONNX for per-head temperature support.");
    }
  }


  /// <summary>
  /// Processes dual policy heads with optional blending and per-head temperatures.
  /// Produces both a blended policy (for use in search) and the unblended policy2 (for analysis).
  /// </summary>
  /// <param name="policy1Logits">Logits from the primary policy head for legal moves</param>
  /// <param name="policy2Logits">Logits from the secondary policy head for legal moves</param>
  /// <param name="numMoves">Number of legal moves</param>
  /// <param name="fractionPolicyHead2">Fraction of policy2 to blend (0 = policy1 only, 1 = policy2 only)</param>
  /// <param name="blendInLogitSpace">If true, blend logits before softmax; if false, blend probabilities after softmax</param>
  /// <param name="policy1Temperature">Temperature for policy1 head</param>
  /// <param name="policy2Temperature">Temperature for policy2 head</param>
  /// <param name="baseTemperature">Base policy temperature applied to final result</param>
  /// <param name="policyUncertaintyScalingFactor">Scaling factor for policy uncertainty adjustment</param>
  /// <param name="policyUncertainty">Policy uncertainty value for this position</param>
  /// <param name="blendedPolicyProbs">Output: blended policy probabilities</param>
  /// <param name="unblendedPolicy2Probs">Output: unblended policy2 probabilities (for secondary policy storage)</param>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static void ProcessDualPolicyHeads(
    ReadOnlySpan<float> policy1Logits,
    ReadOnlySpan<float> policy2Logits,
    int numMoves,
    float fractionPolicyHead2,
    bool blendInLogitSpace,
    float policy1Temperature,
    float policy2Temperature,
    float baseTemperature,
    float policyUncertaintyScalingFactor,
    float policyUncertainty,
    Span<float> blendedPolicyProbs,
    Span<float> unblendedPolicy2Probs)
  {
    if (numMoves == 0)
    {
      return;
    }

    ReadOnlySpan<float> p1 = policy1Logits.Slice(0, numMoves);
    ReadOnlySpan<float> p2 = policy2Logits.Slice(0, numMoves);
    Span<float> blended = blendedPolicyProbs.Slice(0, numMoves);
    Span<float> unblendedP2 = unblendedPolicy2Probs.Slice(0, numMoves);

    // Compute unblended policy2 probabilities (always needed for Policies2 output)
    float maxLogit2 = TensorPrimitives.Max(p2);
    TensorPrimitives.Subtract(p2, maxLogit2, unblendedP2);
    if (baseTemperature != 1.0f)
    {
      TensorPrimitives.Multiply(unblendedP2, 1.0f / baseTemperature, unblendedP2);
    }
    TensorPrimitives.Exp(unblendedP2, unblendedP2);
    float sum2 = TensorPrimitives.Sum(unblendedP2);
    if (sum2 > 0)
    {
      TensorPrimitives.Multiply(unblendedP2, 1.0f / sum2, unblendedP2);
    }

    // Compute effective temperature with uncertainty adjustment
    float effectiveTemperature = ComputeEffectiveTemperature(baseTemperature, policyUncertaintyScalingFactor, policyUncertainty);

    // If no blending requested, just compute policy1 probabilities
    if (fractionPolicyHead2 == 0)
    {
      ComputeSoftmaxProbabilitiesWithMax(p1, effectiveTemperature, blended, TensorPrimitives.Max(p1));
      return;
    }

    if (blendInLogitSpace)
    {
      // Logit-space blending: apply per-head temperatures, blend logits, then softmax
      BlendLogits(p1, p2, numMoves, fractionPolicyHead2, policy1Temperature, policy2Temperature, blended);

      // Apply softmax to blended logits
      float maxBlended = TensorPrimitives.Max(blended);
      ComputeSoftmaxProbabilitiesWithMax(blended, effectiveTemperature, blended, maxBlended);
    }
    else
    {
      // Probability-space blending: softmax each head with per-head temperature, then blend
      float p1CombinedTemp = effectiveTemperature * policy1Temperature;
      float maxLogit1 = TensorPrimitives.Max(p1);
      ComputeSoftmaxProbabilitiesWithMax(p1, p1CombinedTemp, blended, maxLogit1);

      if (policy2Temperature != 1.0f)
      {
        // Recompute p2 probs with per-head temperature for blending
        Span<float> temperedP2 = stackalloc float[numMoves];
        ComputeSoftmaxProbabilitiesWithMax(p2, baseTemperature * policy2Temperature, temperedP2, TensorPrimitives.Max(p2));
        BlendProbabilities(blended, temperedP2, numMoves, fractionPolicyHead2);
      }
      else
      {
        BlendProbabilities(blended, unblendedP2, numMoves, fractionPolicyHead2);
      }
    }
  }


  /// <summary>
  /// Blends logits from two policy heads with optional per-head temperature scaling.
  /// Uses the softmax trick (subtract max before temperature scaling) to prevent numeric overflow/underflow.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static void BlendLogits(
    ReadOnlySpan<float> policy1Logits,
    ReadOnlySpan<float> policy2Logits,
    int numMoves,
    float fractionPolicyHead2,
    float policy1Temperature,
    float policy2Temperature,
    Span<float> output)
  {
    float frac1 = 1.0f - fractionPolicyHead2;
    float invT1 = policy1Temperature != 1.0f ? 1.0f / policy1Temperature : 1.0f;
    float invT2 = policy2Temperature != 1.0f ? 1.0f / policy2Temperature : 1.0f;

    if (policy1Temperature != 1.0f || policy2Temperature != 1.0f)
    {
      // Find max logits for each head to apply softmax trick before temperature scaling.
      // This prevents numeric overflow/underflow when temperature < 1 (which amplifies logits).
      float maxLogit1 = TensorPrimitives.Max(policy1Logits.Slice(0, numMoves));
      float maxLogit2 = TensorPrimitives.Max(policy2Logits.Slice(0, numMoves));

      // Apply temperature with max-subtraction: (logit - max) / temperature
      // This keeps values in a numerically stable range.
      for (int i = 0; i < numMoves; i++)
      {
        float centered1 = (policy1Logits[i] - maxLogit1) * invT1;
        float centered2 = (policy2Logits[i] - maxLogit2) * invT2;
        output[i] = frac1 * centered1 + fractionPolicyHead2 * centered2;
      }
    }
    else
    {
      // No temperature scaling needed
      for (int i = 0; i < numMoves; i++)
      {
        output[i] = frac1 * policy1Logits[i] + fractionPolicyHead2 * policy2Logits[i];
      }
    }
  }


  /// <summary>
  /// Blends two probability distributions.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static void BlendProbabilities(
    Span<float> policy1Probs,
    ReadOnlySpan<float> policy2Probs,
    int numMoves,
    float fractionPolicyHead2)
  {
    float frac1 = 1.0f - fractionPolicyHead2;
    for (int i = 0; i < numMoves; i++)
    {
      policy1Probs[i] = frac1 * policy1Probs[i] + fractionPolicyHead2 * policy2Probs[i];
    }
  }


  /// <summary>
  /// Computes the effective temperature with optional uncertainty scaling.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static float ComputeEffectiveTemperature(float baseTemperature, float uncertaintyScalingFactor, float uncertainty)
  {
    if (uncertaintyScalingFactor == 0.0f)
    {
      return baseTemperature;
    }

    float bump = uncertaintyScalingFactor * uncertainty;
    if (bump > 0.35f)
    {
      bump = 0.35f;
    }
    return baseTemperature + bump;
  }


  /// <summary>
  /// Computes softmax probabilities from logits with optional temperature scaling.
  /// </summary>
  /// <param name="logits">Input logits</param>
  /// <param name="numMoves">Number of moves to process</param>
  /// <param name="temperature">Temperature for softmax (1.0 = no scaling)</param>
  /// <param name="probabilities">Output probabilities</param>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static void ComputeSoftmaxProbabilities(
    ReadOnlySpan<float> logits,
    int numMoves,
    float temperature,
    Span<float> probabilities)
  {
    if (numMoves == 0)
    {
      return;
    }

    ReadOnlySpan<float> input = logits.Slice(0, numMoves);
    float maxLogit = TensorPrimitives.Max(input);
    ComputeSoftmaxProbabilitiesWithMax(input, temperature, probabilities.Slice(0, numMoves), maxLogit);
  }


  /// <summary>
  /// Computes softmax probabilities from logits with a pre-computed max value.
  /// This is more efficient when the max is already known or computed elsewhere.
  /// </summary>
  /// <param name="logits">Input logits (already sliced to numMoves)</param>
  /// <param name="temperature">Temperature for softmax (1.0 = no scaling)</param>
  /// <param name="probabilities">Output probabilities (already sliced to numMoves)</param>
  /// <param name="maxLogit">Pre-computed maximum logit value</param>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static void ComputeSoftmaxProbabilitiesWithMax(
    ReadOnlySpan<float> logits,
    float temperature,
    Span<float> probabilities,
    float maxLogit)
  {
    int numMoves = logits.Length;
    if (numMoves == 0)
    {
      return;
    }

    // Subtract max
    TensorPrimitives.Subtract(logits, maxLogit, probabilities);

    // Apply temperature
    if (temperature != 1.0f)
    {
      float invTemp = 1.0f / temperature;
      TensorPrimitives.Multiply(probabilities, invTemp, probabilities);
    }

    // Exp
    TensorPrimitives.Exp(probabilities, probabilities);

    // Normalize
    float sum = TensorPrimitives.Sum(probabilities);

    // Detect numeric underflow: if sum is 0 or NaN, softmax failed (likely due to extreme temperature scaling)
    Debug.Assert(sum > 0 && !float.IsNaN(sum),
      $"Softmax underflow detected: sum={sum}. This may indicate numeric instability from temperature scaling. " +
      $"numMoves={numMoves}, temperature={temperature}, maxLogit={maxLogit}");

    if (sum > 0)
    {
      TensorPrimitives.Multiply(probabilities, 1.0f / sum, probabilities);
    }
  }


  /// <summary>
  /// Computes softmax probabilities in-place with uncertainty-adjusted temperature.
  /// This method is designed to be called from ExtractPoliciesBufferFlat.
  /// </summary>
  /// <param name="logits">Input logits (modified in place to output probabilities)</param>
  /// <param name="numMoves">Number of moves to process</param>
  /// <param name="policyTemperature">Base policy temperature</param>
  /// <param name="policyUncertaintyScalingFactor">Scaling factor for uncertainty</param>
  /// <param name="policyUncertainty">Policy uncertainty value</param>
  /// <param name="maxLogit">Pre-computed maximum logit value</param>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static void SoftmaxInPlaceWithUncertainty(
    Span<float> logits,
    int numMoves,
    float policyTemperature,
    float policyUncertaintyScalingFactor,
    float policyUncertainty,
    float maxLogit)
  {
    if (numMoves == 0)
    {
      return;
    }

    Span<float> slice = logits.Slice(0, numMoves);
    float effectiveTemperature = ComputeEffectiveTemperature(policyTemperature, policyUncertaintyScalingFactor, policyUncertainty);

    // Subtract max
    TensorPrimitives.Subtract(slice, maxLogit, slice);

    // Apply temperature
    if (effectiveTemperature != 1.0f)
    {
      TensorPrimitives.Multiply(slice, 1.0f / effectiveTemperature, slice);
    }

    // Exp
    TensorPrimitives.Exp(slice, slice);

    // Normalize
    float sum = TensorPrimitives.Sum(slice);

    // Detect numeric underflow: if sum is 0 or NaN, softmax failed (likely due to extreme temperature scaling)
    Debug.Assert(sum > 0 && !float.IsNaN(sum),
      $"Softmax underflow detected in SoftmaxInPlaceWithUncertainty: sum={sum}. " +
      $"numMoves={numMoves}, effectiveTemperature={effectiveTemperature}, maxLogit={maxLogit}");

    if (sum > 0)
    {
      TensorPrimitives.Multiply(slice, 1.0f / sum, slice);
    }
  }
}
