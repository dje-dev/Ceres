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
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

using Microsoft.ML.OnnxRuntime;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// Represents the results of evaluation of a batch of positions 
  /// by a NN (value estimates, policies, possible ancillary outputs).
  /// </summary>
  public class PositionEvaluationBatch : IPositionEvaluationBatch
  {
    public enum PolicyType { Probabilities, LogProbabilities };
    public bool IsWDL;
    public bool HasM;
    public bool HasUncertaintyV;
    public bool HasUncertaintyP;
    public bool HasAction;
    public bool HasValueSecondary;
    public bool HasState;
    public int NumPos;

    #region Output raw data

    public Memory<CompressedPolicyVector> Policies;

    /// <summary>
    /// Vector of win probabilities.
    /// </summary>
    public Memory<FP16> W;

    /// <summary>
    /// Vector of loss probabilities.
    /// </summary>
    public Memory<FP16> L;

    /// <summary>
    /// Vector of win probabilities (primary value head).
    /// </summary>
    public Memory<FP16> W1;

    /// <summary>
    /// Vector of loss probabilities (primary value head).
    /// </summary>
    public Memory<FP16> L1;

    /// <summary>
    /// Vector of win probabilities (secondary value head).
    /// </summary>
    public Memory<FP16> W2;

    /// <summary>
    /// Vector of loss probabilities (secondary value head).
    /// </summary>
    public Memory<FP16> L2;

    /// <summary>
    /// Vector of moves left estimates.
    /// </summary>
    public Memory<FP16> M;

    /// <summary>
    /// Vector of uncertainty of V estimates.
    /// </summary>
    public Memory<FP16> UncertaintyV;

    /// <summary>
    /// Vector of uncertainty of policy estimates.
    /// </summary>
    public Memory<FP16> UncertaintyP;

    /// <summary>
    /// Vector of uncertainty of action head estimates.
    /// </summary>
    public Memory<CompressedActionVector> Actions;

    /// <summary>
    /// Optional additional state information.
    /// </summary>
    public Memory<Half[]> States;

    /// <summary>
    /// Activations of inner layers.
    /// </summary>
    public Memory<NNEvaluatorResultActivations> Activations;

    /// <summary>
    /// Optional additional statistic 0.
    /// </summary>
    public Memory<FP16> ExtraStat0;
    
    /// <summary>
    /// Optional additional statistic 1.
    /// </summary>
    public Memory<FP16> ExtraStat1;

    /// <summary>
    /// Optional dictionary of all raw network outputs.
    /// </summary>
    public Memory<FP16[][]> RawNetworkOutputs;

    #endregion


    /// <summary>
    /// Returns the net win probability (V) from the value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetV(int index) => IsWDL ? W.Span[index] - L.Span[index] : W.Span[index];

    /// <summary>
    /// Returns the win probability from the value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetWinP(int index) => W.Span[index];

    /// <summary>
    /// Returns the draw probability from the value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetDrawP(int index) => IsWDL ? (FP16)(1.0f - (W.Span[index] + L.Span[index])) : 0;

    /// <summary>
    /// Returns the loss probability from the value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetLossP(int index) => IsWDL ? L.Span[index] : 0;


    #region Primary value head

    /// <summary>
    /// Returns the net win probability (V) from the primary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetV1(int index) => W1.IsEmpty ? FP16.NaN : (IsWDL ? W1.Span[index] - L1.Span[index] : W1.Span[index]);

    /// <summary>
    /// Returns the win probability from the primary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetWin1P(int index) => W1.IsEmpty ? GetWinP(index) : W1.Span[index];

    /// <summary>
    /// Returns the draw probability from the primary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetDraw1P(int index) => IsWDL ? GetWin1P(index) - GetLoss1P(index) : 0;

    /// <summary>
    /// Returns the loss probability from the primary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetLoss1P(int index) => IsWDL ? (L1.IsEmpty ? GetLossP(index) : L1.Span[index]) : 0;

    #endregion


    #region Secondary value head

    /// <summary>
    /// Returns the net win probability (V) from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetV2(int index) => IsWDL ? GetWin2P(index) - GetLoss2P(index) : GetWin2P(index);

    /// <summary>
    /// Returns the win probability from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetWin2P(int index) => W2.Span.IsEmpty ? FP16.NaN : W2.Span[index];

    /// <summary>
    /// Returns the draw probability from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetDraw2P(int index) => IsWDL ? (FP16)(1.0f - (GetWin2P(index) + GetLoss2P(index))) : 0;

    /// <summary>
    /// Returns the loss probability from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetLoss2P(int index) => IsWDL ? (L2.Span.IsEmpty ? FP16.NaN : L2.Span[index]) : 0;

    #endregion


    /// <summary>
    /// Returns the value of the MLH head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetM(int index) => HasM ? (M.IsEmpty ? FP16.NaN : M.Span[index]) : FP16.NaN;

    /// <summary>
    /// Returns the value of the uncertainty of V head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetUncertaintyV(int index) => UncertaintyV.IsEmpty ? FP16.NaN : UncertaintyV.Span[index];

    /// <summary>
    /// Returns the value of the uncertainty of policy head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetUncertaintyP(int index) => UncertaintyP.IsEmpty ? FP16.NaN : UncertaintyP.Span[index];

    /// <summary>
    /// Returns optional state information for specified position.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public Half[] GetState(int index) => HasState && !States.IsEmpty ? States.Span[index] : null;

    /// <summary>
    /// Returns inner layer activations.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    NNEvaluatorResultActivations GetActivations(int index) => Activations.IsEmpty ? null : Activations.Span[index];

    /// <summary>
    /// Returns extra statistic 0 for specified position.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="statIndex"></param>
    /// <returns></returns>
    public FP16 GetExtraStat0(int index) => ExtraStat0.Span[index];


    /// <summary>
    /// Returns extra statistic 1 index for specified position.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="statIndex"></param>
    /// <returns></returns>
    public FP16 GetExtraStat1(int index) => ExtraStat1.Span[index];

    /// <summary>
    /// Returns string representation of the WDL values for position at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public string WDLInfoStr(int index)
    {
      if (IsWDL)
      {
        return $" { GetV(index) } ({GetWinP(index) }/{ GetDrawP(index) }/{ GetLossP(index) })";
      }
      else
        return $"{W.Span[index]}";
    }


    public TimingStats Stats;

    /// <summary>
    /// Returns if the batch was evaluated with a network that has an WDL head.
    /// </summary>
    bool IPositionEvaluationBatch.IsWDL => IsWDL;
    
    /// <summary>
    /// Returns if the batch was evaluated with a network that has an MLH head.
    /// </summary>
    bool IPositionEvaluationBatch.HasM => HasM;

    /// <summary>
    /// Returns if the batch was evaluated with a network that has an uncertainty of V head.
    /// </summary>
    bool IPositionEvaluationBatch.HasUncertaintyV => HasUncertaintyV;

    /// <summary>
    /// Returns if the batch was evaluated with a network that has an uncertainty of policy head.
    /// </summary>
    bool IPositionEvaluationBatch.HasUncertaintyP => HasUncertaintyP;

    /// <summary>
    /// Returns if the batch was evaluated with a network that has an action head.
    /// </summary>
    bool IPositionEvaluationBatch.HasAction => HasAction;

    /// <summary>
    /// Returns if the batch has a secondary value head.
    /// </summary>
    bool IPositionEvaluationBatch.HasValueSecondary => HasValueSecondary;

    /// <summary>
    /// Returns the number of positions in the batch.
    /// </summary>
    int IPositionEvaluationBatch.NumPos => NumPos;

    /// <summary>
    /// If the batch has state information.
    /// </summary>
    bool IPositionEvaluationBatch.HasState => HasState;

    /// <summary>
    /// Gets the value from the value head at a specified index indicating the win probability.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    FP16 IPositionEvaluationBatch.GetWinP(int index) => GetWinP(index);

    /// <summary>
    /// Gets the value from the value head at a specified index indicating the loss probability.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    FP16 IPositionEvaluationBatch.GetLossP(int index) => GetLossP(index);

    /// <summary>
    /// Returns the policy distribution for the position at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    (Memory<CompressedPolicyVector> policies, int index)
      IPositionEvaluationBatch.GetPolicy(int index) => (Policies, index);


    /// <summary>
    /// Returns the action distribution for the position at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    (Memory<CompressedActionVector> actions, int index) 
       IPositionEvaluationBatch.GetAction(int index) => (Actions, index);


    /// <summary>
    /// Gets the value from the MLH head at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    FP16 IPositionEvaluationBatch.GetM(int index) => GetM(index);

    /// <summary>
    /// Gets the value from the uncertainty of V head at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    FP16 IPositionEvaluationBatch.GetUncertaintyV(int index) => GetUncertaintyV(index);

    /// <summary>
    /// Returns optional extra statistic for specified position.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    FP16 IPositionEvaluationBatch.GetExtraStat0(int index) => ExtraStat0.IsEmpty ? FP16.NaN : GetExtraStat0(index);

    /// <summary>
    /// Returns optional extra statistic for specified position.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    FP16 IPositionEvaluationBatch.GetExtraStat1(int index) => ExtraStat1.IsEmpty ? FP16.NaN : GetExtraStat1(index);


    Half[] IPositionEvaluationBatch.GetState(int index) => GetState(index);

    NNEvaluatorResultActivations IPositionEvaluationBatch.GetActivations(int index) => GetActivations(index);

    /// <summary>
    /// Array of names of raw neural network outputs (if RetainRawOutputs is true).
    /// </summary>
    public string[] RawNetworkOutputNames;



    /// <summary>
    /// Constructs from directly provided set of evaluation results
    /// (which have already been fully decoded).
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="numPos"></param>
    /// <param name="policies"></param>
    /// <param name="actionProbabilties"></param>
    /// <param name="w"></param>
    /// <param name="d"></param>
    /// <param name="l"></param>
    /// <param name="m"></param>
    /// <param name="activations"></param>
    /// <param name="stats"></param>
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasUncertaintyP, bool hasAction, 
                                   bool hasValueSecondary, bool hasState, int numPos, 
                                   Memory<CompressedPolicyVector> policies,
                                   Memory<CompressedActionVector> actionProbabilties,
                                   Memory<FP16> w, Memory<FP16> l, 
                                   Memory<FP16> w2, Memory<FP16> l2,
                                   Memory<FP16> m, Memory<FP16> uncertaintyV, Memory<FP16> uncertaintyP,
                                   Memory<Half[]> states,
                                   Memory<NNEvaluatorResultActivations> activations,
                                   TimingStats stats, Memory<FP16> extraStat0 = default, Memory<FP16> extraStat1 = default,
                                   bool makeCopy = false,
                                   Memory<FP16[][]> rawNetworkOutputs = default,
                                   string[] rawNetworkOutputNames = null)
    {
      IsWDL = isWDL;
      HasM = hasM;
      NumPos = numPos;
      HasUncertaintyV = hasUncertaintyV;
      HasUncertaintyP = hasUncertaintyP;
      HasAction = hasAction;
      HasValueSecondary = hasValueSecondary;
      HasState = hasState;

      W = makeCopy ? w.Slice(0, numPos).ToArray() : w;
      Policies = makeCopy ? policies.Slice(0, numPos).ToArray() : policies;
      Actions = (hasAction && makeCopy) ? actionProbabilties.Slice(0, numPos).ToArray() : actionProbabilties;
      Activations = (activations.Length != 0 && makeCopy) ? activations.Slice(0, numPos).ToArray() : activations;
      States = (states.Length != 0 && makeCopy) ? states.Slice(0, numPos).ToArray() : states;
      ExtraStat0 = (extraStat0.Length != 0 && makeCopy) ? extraStat0.Slice(0, numPos).ToArray() : extraStat0;
      ExtraStat1 = (extraStat1.Length != 0 && makeCopy) ? extraStat1.Slice(0, numPos).ToArray() : extraStat0;
      UncertaintyV = (HasUncertaintyV && makeCopy) ? uncertaintyV.Slice(0, numPos).ToArray() : uncertaintyV;
      UncertaintyP = (HasUncertaintyP && makeCopy) ? uncertaintyP.Slice(0, numPos).ToArray() : uncertaintyP;
      Actions = (HasAction && makeCopy) ? actionProbabilties.Slice(0, numPos).ToArray() : actionProbabilties;
      States = (HasState && makeCopy) ? states.Slice(0, numPos).ToArray() : states;
      M = (hasM && makeCopy) ? m.Slice(0, numPos).ToArray() : m;
      L = (isWDL && makeCopy) ? l.Slice(0, numPos).ToArray() : l;

      if (!w2.IsEmpty)
      {
        W2 = makeCopy ? w2.Slice(0, numPos).ToArray() : w2;
        L2 = (isWDL && makeCopy) ? l2.Slice(0, numPos).ToArray() : l2;
      }

      Stats = stats;
      RawNetworkOutputs = rawNetworkOutputs;
      RawNetworkOutputNames = rawNetworkOutputNames;
    }


    /// <summary>
    /// Constructor (from specified value and policy values).
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="hasM"></param>
    /// <param name="numPos"></param>
    /// <param name="valueEvals"></param>
    /// <param name="valueEvals2"></param>
    /// <param name="valsAreLogistic"></param>
    /// <param name="stats"></param>
    private PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasUncertaintyP, bool hasAction, 
                                    bool hasValueSecondary, bool hasState, int numPos,
                                    ReadOnlySpan<FP16> valueEvals, ReadOnlySpan<FP16> valueEvals2,
                                    ReadOnlySpan<FP16> m, 
                                    ReadOnlySpan<FP16> uncertaintyV, ReadOnlyMemory<FP16> uncertaintyP,
                                    ReadOnlySpan<FP16> extraStats0, ReadOnlySpan<FP16> extraStats1,
                                    Span<Half[]> states,
                                    Span<NNEvaluatorResultActivations> activations,
                                    float fractionValueFromValue2,
                                    float temperatureValue1, float temperatureValue2,
                                    float value1TemperatureUncertaintyScalingFactor, float value2TemperatureUncertaintyScalingFactor,
                                    bool valsAreLogistic, 
                                    TimingStats stats,
                                    Memory<FP16[][]> rawNetworkOutputs = default,
                                    string[] rawNetworkOutputNames = null)
    {
      IsWDL = isWDL;
      HasM = hasM;
      HasUncertaintyV = hasUncertaintyV;
      HasUncertaintyP = hasUncertaintyP;
      HasValueSecondary = hasValueSecondary;
      HasAction = hasAction;
      HasState = hasState;
      NumPos = numPos;

      Activations = activations.ToArray();
      States = states.ToArray();
      RawNetworkOutputs = rawNetworkOutputs;
      RawNetworkOutputNames = rawNetworkOutputNames;

      InitializeValueEvals(valueEvals, valsAreLogistic, temperatureValue1, uncertaintyV, value1TemperatureUncertaintyScalingFactor, false);
      if (!valueEvals2.IsEmpty)
      {
        InitializeValueEvals(valueEvals2, valsAreLogistic, temperatureValue2, uncertaintyV, value2TemperatureUncertaintyScalingFactor, true);
      }

      if (fractionValueFromValue2 != 0)
      {
        Debug.Assert(!W2.IsEmpty);

        Span<FP16> w = W.Span;
        Span<FP16> w2 = W2.Span;
        Span<FP16> l = L.Span;
        Span<FP16> l2 = L2.Span;
        float fractionValueFromValue1 = 1 - fractionValueFromValue2;

        for (int i=0; i<w.Length; i++)
        { 
          w[i] = (FP16)(w[i] * fractionValueFromValue1 + w2[i] * fractionValueFromValue2);
          l[i] = (FP16)(l[i] * fractionValueFromValue1 + l2[i] * fractionValueFromValue2);
        }  
      }

      if (hasM)
      {
        M = m.ToArray();
      }

      if (HasUncertaintyV)
      {
        UncertaintyV = uncertaintyV.ToArray();
      }

      if (HasUncertaintyP)
      {
        UncertaintyP = uncertaintyP.ToArray();
      }

      if (!extraStats0.IsEmpty)
      {
        ExtraStat0 = extraStats0.ToArray();
      }

      if (!extraStats1.IsEmpty)
      {
        ExtraStat1 = extraStats1.ToArray();
      }

      if (valueEvals != null && valueEvals.Length < numPos * (IsWDL ? 3 : 1))
      {
        throw new ArgumentException("Wrong value size");
      }

      Stats = stats;      
    }

    
    /// <summary>
    /// Constructor (when the policies are full arrays of 1858 each)
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="numPos"></param>
    /// <param name="valueEvals"></param>
    /// <param name="policyProbs"></param>
    /// <param name="activations"></param>
    /// <param name="probType"></param>
    /// <param name="stats"></param>
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasUnertaintyP, bool hasAction, 
                                   bool hasValueSecondary, bool hasState, int numPos, 
                                   Span<FP16> valueEvals, Span<FP16> valueEvals2, 
                                   ReadOnlyMemory<Float16> policyProbs, ReadOnlyMemory<Float16> actionLogits,
                                   ReadOnlySpan<FP16> m, 
                                   ReadOnlySpan<FP16> uncertaintyV, ReadOnlyMemory<FP16> uncertaintyP,
                                   ReadOnlySpan<FP16> extraStats0, ReadOnlySpan<FP16> extraStats1,
                                   ReadOnlyMemory<Half[]> states, NNEvaluatorResultActivations[] activations,
                                   float fractionValueFromValue2, 
                                   float temperatureValue1, float temperatureValue2,
                                   float value1TemperatureUncertaintyScalingFactor, float value2TemperatureUncertaintyScalingFactor,
                                   bool valsAreLogistic, PolicyType probType, bool policyAlreadySorted,
                                   IEncodedPositionBatchFlat sourceBatchWithValidMoves,
                                   float policyTemperature, float policyUncertaintyScalingFactor,
                                   TimingStats stats,
                                   Memory<FP16[][]> rawNetworkOutputs = default,                                   
                                   string[] rawNetworkOutputNames = null)
   : this(isWDL, hasM, hasUncertaintyV, hasUnertaintyP, hasAction, hasValueSecondary, hasState, numPos, valueEvals, valueEvals2, 
          m, uncertaintyV,  uncertaintyP, extraStats0, extraStats1, states.ToArray(), activations,
          fractionValueFromValue2, 
          temperatureValue1, temperatureValue2,
          value1TemperatureUncertaintyScalingFactor, value2TemperatureUncertaintyScalingFactor,
          valsAreLogistic, stats, rawNetworkOutputs, rawNetworkOutputNames)
    {     
      (Policies, Actions) = ExtractPoliciesBufferFlat(numPos, policyProbs, uncertaintyP, 
                                                      hasAction, actionLogits,
                                                      policyTemperature, policyUncertaintyScalingFactor,
                                                      probType, policyAlreadySorted, sourceBatchWithValidMoves);
    }


    /// <summary>
    /// Constructor (when the policies are returned as TOP_K arrays)
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="numPos"></param>
    /// <param name="valueEvals"></param>
    /// <param name="policyIndices"></param>
    /// <param name="policyProbabilties"></param>
    /// <param name="activations"></param>
    /// <param name="probType"></param>
    /// <param name="stats"></param>
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasUncertaintyP,
                                   bool hasAction, bool hasValueSecondary,
                                   bool hasState, int numPos,
                                   ReadOnlySpan<FP16> valueEvals, ReadOnlySpan<FP16> valueEvals2,
                                   int topK, ReadOnlySpan<int> policyIndices, ReadOnlySpan<float> policyProbabilties,
                                   ReadOnlySpan<FP16> m, 
                                   ReadOnlySpan<FP16> uncertaintyV, ReadOnlySpan<FP16> uncertaintyP,
                                   ReadOnlySpan<FP16> extraStats0, ReadOnlySpan<FP16> extraStats1,
                                   ReadOnlySpan<Half[]> states,
                                   ReadOnlySpan<NNEvaluatorResultActivations> activations, bool valsAreLogistic,
                                   PolicyType probType, TimingStats stats, bool alreadySorted,
                                   float temperatureValue1= 1, float temperatureValue2 = 1, float fractionValue1FromValue2 = 0,
                                   Memory<FP16[][]> rawNetworkOutputs = default,                                   
                                   string[] rawNetworkOutputNames = null)
//      : this(isWDL, hasM, hasUncertaintyV, hasUncertaintyP, hasAction, hasValueSecondary, hasState, 
//             numPos, valueEvals, valueEvals2, m, uncertaintyV, uncertaintyP, extraStats0, extraStats1, states, activations, 
//             temperatureValue1, temperatureValue2, fractionValue1FromValue2, valsAreLogistic, stats)
    {
      throw new NotImplementedException();
      if (HasAction)
      {
        throw new NotImplementedException();
      }
//      Policies = ExtractPoliciesTopK(numPos, topK, policyIndices, policyProbabilties, probType, alreadySorted);
    }


    /// <summary>
    /// Constructor (for an empty batch, to be filled in later with CopyFrom method).
    /// </summary>
    /// <param name="maxPos"></param>
    /// <param name="retrieveSupplementalResults"></param>
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasValueSecondary, 
                                   int maxPos, bool retrieveSupplementalResults)
    {
      // Likely need to remediate this (and possibly also CopyFrom method, for additional heads).
      throw new NotImplementedException();
#if NOT
      Policies = new CompressedPolicyVector[maxPos];
      HasM = hasM;
      IsWDL = isWDL;
      HasUncertaintyV = hasUncertaintyV;
      HasValueSecondary = hasValueSecondary;

      if (retrieveSupplementalResults)
      {
        Activations = new NNEvaluatorResultActivations[maxPos];
      }
      
      W = new FP16[maxPos];
      W2 = hasValueSecondary ? new FP16[maxPos] : default;
      
      if (isWDL)
      {
        L = new FP16[maxPos];
        L2 = hasValueSecondary ? new FP16[maxPos] : default;
      }

      if (hasM)
      {
        M = new FP16[maxPos];
      }
#endif
    }

    /// <summary>
    /// Resets values of this batch to be
    /// copies of values taken from another specified batch.
    /// </summary>
    /// <param name="other"></param>
    /// <exception cref="ArgumentException"></exception>
    public void CopyFrom(PositionEvaluationBatch other)
    {
      // Likely need to remediate for additional heads.
      throw new NotImplementedException();
#if NOT
      if (other.NumPos > W.Length)
      {
        throw new ArgumentException($"Other batch number of positions {other.NumPos} exceeds maximum of {W.Length}");
      }

      IsWDL = other.IsWDL;
      HasM = other.HasM;
      NumPos = other.NumPos;
      HasUncertaintyV = other.HasUncertaintyV;
      HasValueSecondary = other.HasValueSecondary;
      HasState = other.HasState;
      HasAction = other.HasAction;

      int numPos = other.NumPos;

      other.Policies.Slice(0, numPos).CopyTo(Policies);

      if (other.Activations.Length == 0)
      {
        Activations = other.Activations;
      }
      else
      {
        other.Activations.Slice(0, numPos).CopyTo(Activations);
      }

      other.W.Slice(0, numPos).CopyTo(W);

      if (!W2.IsEmpty)
      {
        other.W2.Slice(0, numPos).CopyTo(W2);
      }

      if (HasUncertaintyV)
      {
        other.UncertaintyV.Slice(0, numPos).CopyTo(UncertaintyV);
      }

      if (!ExtraStat0.IsEmpty)
      {
        other.ExtraStat0.Slice(0, numPos).CopyTo(ExtraStat0);
      }

      if (!ExtraStat1.IsEmpty)
      {
        other.ExtraStat1.Slice(0, numPos).CopyTo(ExtraStat1);
      }

      if (!Actions.IsEmpty)
      {
        other.Actions.Slice(0, numPos).CopyTo(Actions);
      }

      if (IsWDL)
      {
        other.L.Slice(0, numPos).CopyTo(L);

        if (!L2.IsEmpty)
        {
          other.L2.Slice(0, numPos).CopyTo(L2);
        }
      }

      if (HasState)
      {
        other.States.Slice(0, numPos).CopyTo(States);
      }

      if (HasM)
      {
        other.M.Slice(0, numPos).CopyTo(M);
      }

      Stats = other.Stats;
#endif
    }


    /// <summary>
    /// Returns a string representation of the batch (summary).
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string timeStr = Stats.ElapsedTimeTicks == 0 ? "" : $" time: {Stats.ElapsedTimeSecs:F2} secs";
      return $"<PositionEvaluationBatch of {NumPos:n0}{timeStr} with first V: {GetV(0):F2}>";
    }



    /// <summary>
    /// Returns a CompressedPolicyVector array with the policy vectors
    /// extracted from all the positions in this batch.
    /// </summary>
    /// <param name="numPos"></param>
    /// <param name="topK"></param>
    /// <param name="indices"></param>
    /// <param name="probabilities"></param>
    /// <param name="probType"></param>
    /// <returns></returns>
    static CompressedPolicyVector[] ExtractPoliciesTopK(int numPos, int topK, Span<int> indices, Span<float> probabilities, PolicyType probType, bool alreadySorted)
    {
      if (probType == PolicyType.LogProbabilities)
      {
        throw new NotImplementedException();
        for (int i = 0; i < indices.Length; i++)
        {
          probabilities[i] = MathF.Exp(probabilities[i]);
        }
      }

      if (indices == null && probabilities == null)
      {
        return null;
      }
      if (probabilities.Length != indices.Length)
      {
        throw new ArgumentException("Indices and probabilties expected to be same length");
      }

      CompressedPolicyVector[] retPolicies = new CompressedPolicyVector[numPos];
    
      int offset = 0;
      for (int i = 0; i < numPos; i++)
      {
        CompressedPolicyVector.Initialize(ref retPolicies[i], indices.Slice(offset, topK), probabilities.Slice(offset, topK));
        offset += topK;
      }

      return retPolicies;
    }


    [ThreadStatic]
    static float[] policyTempBuffer;

    [ThreadStatic]
    static float[] actionTempBuffer;

    static (CompressedPolicyVector[], Memory<CompressedActionVector>)
      ExtractPoliciesBufferFlat(int numPos,
                                ReadOnlyMemory<Float16> policyProbs,
                                ReadOnlyMemory<FP16> policyUncertainties,
                                bool hasActions,
                                ReadOnlyMemory<Float16> actionLogits,
                                float policyTemperature,
                                float policyUncertaintyScalingFactor,
                                PolicyType probType, bool alreadySorted,
                                IEncodedPositionBatchFlat sourceBatchWithValidMoves)
    {
      Debug.Assert(!alreadySorted); // TODO: it seems nonsensical that these claim to be sorted
      // TODO: possibly needs work.
      // Do we handle WDL correctly? Do we flip the moves if we are black (using positions) ?

      if (policyProbs.IsEmpty)
      {
        return (null, default);
      }

      if (policyProbs.Length != EncodedPolicyVector.POLICY_VECTOR_LENGTH * numPos)
      {
        throw new ArgumentException("Wrong policy size");
      }

      CompressedActionVector[] actions = hasActions ? new CompressedActionVector[numPos] : default;

      CompressedPolicyVector[] retPolicies = new CompressedPolicyVector[numPos];
      Memory<MGMoveList> moves = sourceBatchWithValidMoves == null ? default : sourceBatchWithValidMoves.Moves;
      Parallel.For(0, numPos, i =>
      {
        ReadOnlySpan<Float16> policyProbsSpan = policyProbs.Span;
        ReadOnlySpan<FP16> policiesUncertaintiesSpan = policyUncertainties.Span;
        ReadOnlySpan<Float16> actionLogitsSpan = hasActions ? actionLogits.Span : default;
        hasActions = hasActions && !actionLogits.IsEmpty;

        int startIndexPolicy = EncodedPolicyVector.POLICY_VECTOR_LENGTH * i;

        if (probType == PolicyType.Probabilities)
        {
          throw new NotImplementedException();
        }
        else if (sourceBatchWithValidMoves == default)
        {
          throw new NotImplementedException(); // the below might be working, but seems not needed
#if NOT
          if (policyTempBuffer == null)
          {
            policyTempBuffer = new float[EncodedPolicyVector.POLICY_VECTOR_LENGTH];
            actionTempBuffer = new float[EncodedPolicyVector.POLICY_VECTOR_LENGTH * 3];
          }
          else
          {
            Array.Clear(policyTempBuffer, 0, EncodedPolicyVector.POLICY_VECTOR_LENGTH);
            Array.Clear(actionTempBuffer, 0, EncodedPolicyVector.POLICY_VECTOR_LENGTH * 3);
          }

          // N.B. It is not possible to apply move masking here, 
          //      so it is assumed this is already done by the caller.
          //Array.Copy(policyProbsSpan, startIndex, policyTempBuffer, 0, EncodedPolicyVector.POLICY_VECTOR_LENGTH);
          ReadOnlySpan<Float16> thesePolicyProbsFloat16 = policyProbsSpan.Slice(startIndex, EncodedPolicyVector.POLICY_VECTOR_LENGTH);
          ReadOnlySpan<Half> thesePolicyProbsHalf = MemoryMarshal.Cast<Float16, Half>(thesePolicyProbsFloat16);
          
          const float MIN_PROB = -100;
          CompressedPolicyVector policyVector = default;
          CompressedActionVector actionsVector = default;
          PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(ref policyVector, ref actionsVector, hasActions,
                                                                              true, EncodedPolicyVector.POLICY_VECTOR_LENGTH, 96,
                                                                              thesePolicyProbsHalf, MIN_PROB);

          if (hasActions)
          {
            throw new NotImplementedException("probably need remediation in next line");
          }

          retPolicies[i] = policyVector;
#endif
        }
        else
        {
//          Debug.Assert(actions.IsEmpty);

          // Collect together all move indices and policy logit values.
          MGMoveList movesThis = moves.Span[i];
          int numLegalMoves = movesThis.NumMovesUsed;

          Span<int> legalMoveIndices = stackalloc int[numLegalMoves];
          Span<float> legalMovePolicies = stackalloc float[numLegalMoves];
//          Span<Half> legalMovePoliciesHalf = stackalloc Half[numLegalMoves];
          ReadOnlySpan<Half> policyProbsSpanAsHalf =  MemoryMarshal.Cast<Float16, Half>(policyProbsSpan.Slice(startIndexPolicy, EncodedPolicyVector.POLICY_VECTOR_LENGTH));
          ReadOnlySpan<Half> actionLogitsSpanAsHalf = hasActions ? MemoryMarshal.Cast<Float16, Half>(actionLogitsSpan.Slice(i * 3 * EncodedPolicyVector.POLICY_VECTOR_LENGTH, 3 * EncodedPolicyVector.POLICY_VECTOR_LENGTH)) : default;

          float policyLogitMax = 0;
          for (int im = 0; im < numLegalMoves; im++)
          {
            EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesThis.MovesArray[im]);
            legalMoveIndices[im] = (ushort) encodedMove.IndexNeuralNet;

            // Extract action WDL (if any).
            if (hasActions)
            {
              double v1, v2, v3;
              int baseActionIndex = 3 * legalMoveIndices[im];
              float actionLogitMax = Math.Max(Math.Max((float)actionLogitsSpanAsHalf[baseActionIndex + 0], (float)actionLogitsSpanAsHalf[baseActionIndex + 1]), 
                                              (float)actionLogitsSpanAsHalf[baseActionIndex + 2]);

              v1 = Math.Exp((float)actionLogitsSpanAsHalf[baseActionIndex + 0] - actionLogitMax);
              v2 = Math.Exp((float)actionLogitsSpanAsHalf[baseActionIndex + 1] - actionLogitMax);
              v3 = Math.Exp((float)actionLogitsSpanAsHalf[baseActionIndex + 2] - actionLogitMax);

              double actionTotalProbs = v1 + v2 + v3;
              Debug.Assert(!double.IsNaN(actionTotalProbs));

              // Normalize action probabilities to sum to exactly 1.
              actions[i][im] = ((Half)(v1 / actionTotalProbs), (Half)(v3 / actionTotalProbs));
            }

            float policyValue =  (float)policyProbsSpanAsHalf[encodedMove.IndexNeuralNet];
            if (policyValue > policyLogitMax)
            {
              policyLogitMax = policyValue;
            } 
            legalMovePolicies[im] = policyValue;
          }

          float sumPolicyProbs = 0;
          for (int im = 0; im < numLegalMoves; im++)
          {
            float expVal;

            if (policyTemperature != 1 || policyUncertaintyScalingFactor != 0)
            {
              float thisTemperature = policyTemperature;
              if (policyUncertaintyScalingFactor != 0)
              {
                thisTemperature += policyUncertaintyScalingFactor * (float)policiesUncertaintiesSpan[i];
              }
              expVal = (float)Math.Exp((legalMovePolicies[im] - policyLogitMax) / thisTemperature);
            }
            else
            {
              // Fast path for temperature 1, no uncertainty.
              expVal = (float)Math.Exp((legalMovePolicies[im] - policyLogitMax));
            }

            legalMovePolicies[im] = expVal;
            sumPolicyProbs+= expVal;
          }

          for (int im = 0; im < numLegalMoves; im++)
          {
            legalMovePolicies[im] /= sumPolicyProbs;
          }

          if (hasActions)
          {
            CompressedPolicyVector.Initialize(ref retPolicies[i], legalMoveIndices, legalMovePolicies, false, true, ref actions[i]);
          }
          else
          {
            CompressedPolicyVector.Initialize(ref retPolicies[i], legalMoveIndices, legalMovePolicies, false);
          }
        }
      });

      return (retPolicies, actions);
    }


    /// <summary>
    /// Initializes the win/loss array with speicifed values from the value head.
    /// </summary>
    /// <param name="valueEvals"></param>
    /// <param name="valsAreLogistic"></param>
    void InitializeValueEvals(ReadOnlySpan<FP16> valueEvals, bool valsAreLogistic, float temperature,
                              ReadOnlySpan<FP16> uncertaintyV, float uncertaintyTemperatureScalingFactor,
                              bool secondaryValue)
    {
      Debug.Assert(!(secondaryValue && !HasValueSecondary));

      FP16[] w = null;
      FP16[] l = null;

      if (IsWDL)
      {
        w = new FP16[NumPos];
        l = new FP16[NumPos];

        for (int i = 0; i < NumPos; i++)
        {
          if (!valsAreLogistic)
          {
            if (temperature != 1 || uncertaintyTemperatureScalingFactor != 0)
            {
              throw new NotImplementedException("temperature/uncertainty with non-logistic values");
            }

            w[i] = valueEvals[i * 3 + 0];
            l[i] = valueEvals[i * 3 + 2];
            Debug.Assert(Math.Abs(1 - (valueEvals[i * 3 + 0] 
                                     + valueEvals[i * 3 + 1] 
                                     + valueEvals[i * 3 + 2])) <= 0.01);
          }
          else
          {
            CalcWLWithTemperature(valueEvals, temperature, w, l, uncertaintyV, uncertaintyTemperatureScalingFactor, i);
          }
        }
      }
      else
      {
        Debug.Assert(!valsAreLogistic);
        w = valueEvals.ToArray();
      }


      if (secondaryValue)
      {
        W2 = w;
        L2 = l;
      }
      else
      {
        W = w;
        L = l;
      }
    }

    private static void CalcWLWithTemperature(ReadOnlySpan<FP16> valueEvals, float temperature, 
                                              FP16[] w, FP16[] l,
                                              ReadOnlySpan<FP16> uncertaintyV, float uncertaintyTemperatureScalingFactor,
                                              int i)
    {

#if NOT
// FEATURE FOR ASYMMETRIC TEMPERATURE
            if (false && temperature > 1.25)
            {
              float wl = w[i] - l[i];
              if (Math.Abs(wl) > 0.50f)
              {
                const float POSITIVE_EVAL_TEMP_MULTIPLIER = 1.25f;
                CalcWLWithTemperature(valueEvals, temperature * POSITIVE_EVAL_TEMP_MULTIPLIER, w, l, i);
              }
            }
#endif

      double v1;
      double v2;
      double v3;

      float max = Math.Max(Math.Max(valueEvals[i * 3 + 0],
                                     valueEvals[i * 3 + 1]),
                            valueEvals[i * 3 + 2]);

      if (temperature == 1 && uncertaintyTemperatureScalingFactor == 0)
      {
        v1 = Math.Exp(valueEvals[i * 3 + 0] - max);
        v2 = Math.Exp(valueEvals[i * 3 + 1] - max);
        v3 = Math.Exp(valueEvals[i * 3 + 2] - max);
      }
      else
      {
        float thisTemp = uncertaintyTemperatureScalingFactor == 0
                        ? temperature :
                          temperature + uncertaintyTemperatureScalingFactor * uncertaintyV[i];

#if ASYMMETRIC_VALUE_TEMP        
        bool isWinning = valueEvals[i * 3 + 0] > valueEvals[i * 3 + 2];
if (uncertaintyTemperatureScalingFactor > 0)
{ 
  thisTemp += isWinning ? 0.50f :  -0.20f;
}
#endif

        v1 = Math.Exp((valueEvals[i * 3 + 0] - max) / thisTemp);
        v2 = Math.Exp((valueEvals[i * 3 + 1] - max) / thisTemp);
        v3 = Math.Exp((valueEvals[i * 3 + 2] - max) / thisTemp);
      }

      double totl = v1 + v2 + v3;
      Debug.Assert(!double.IsNaN(totl));
      
      // Set win/loss such that the sum of the three is 1.
      w[i] = (FP16)(v1 / totl);
      l[i] = (FP16)(v3 / totl);
    }


    public IEnumerator<NNPositionEvaluationBatchMember> GetEnumerator()
    {
      for (int i = 0; i < NumPos; i++)
      {
        yield return new NNPositionEvaluationBatchMember(this, i);
      }
    }


    IEnumerator IEnumerable.GetEnumerator()
    {
      throw new NotImplementedException();
    }


  }
}
