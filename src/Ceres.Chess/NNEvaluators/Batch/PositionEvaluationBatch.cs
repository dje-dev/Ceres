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
using System.Reflection;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

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
    public bool HasValueSecondary;
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

    #region Secondary value head

    /// <summary>
    /// Returns the net win probability (V) from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetV2(int index) => IsWDL ? W2.Span[index] - L2.Span[index] : W2.Span[index];

    /// <summary>
    /// Returns the win probability from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetWin2P(int index) => W2.Span[index];

    /// <summary>
    /// Returns the draw probability from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetDraw2P(int index) => IsWDL ? (FP16)(1.0f - (W2.Span[index] + L2.Span[index])) : 0;

    /// <summary>
    /// Returns the loss probability from the secondary value head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetLoss2P(int index) => IsWDL ? L2.Span[index] : 0;

    #endregion


    /// <summary>
    /// Returns the value of the MLH head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetM(int index) => HasM ? M.Span[index] : FP16.NaN;

    /// <summary>
    /// Returns the value of the uncertainty of V head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetUncertaintyV(int index) => HasUncertaintyV ? UncertaintyV.Span[index] : FP16.NaN;

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
    /// Returns if the batch has a secondary value head.
    /// </summary>
    bool IPositionEvaluationBatch.HasValueSecondary => HasValueSecondary;

    /// <summary>
    /// Returns the number of positions in the batch.
    /// </summary>
    int IPositionEvaluationBatch.NumPos => NumPos;

    /// <summary>
    /// Gets the value from the value head at a specified index indicating the win probabilty.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    FP16 IPositionEvaluationBatch.GetWinP(int index) => GetWinP(index);

    /// <summary>
    /// Gets the value from the value head at a specified index indicating the loss probabilty.
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


    NNEvaluatorResultActivations IPositionEvaluationBatch.GetActivations(int index) => GetActivations(index);



    /// <summary>
    /// Constructs from directly provided set of evaluation results
    /// (which have already been fully decoded).
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="numPos"></param>
    /// <param name="policies"></param>
    /// <param name="w"></param>
    /// <param name="d"></param>
    /// <param name="l"></param>
    /// <param name="m"></param>
    /// <param name="activations"></param>
    /// <param name="stats"></param>
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasValueSecondary, int numPos, 
                                   Memory<CompressedPolicyVector> policies,
                                   Memory<FP16> w, Memory<FP16> l, 
                                   Memory<FP16> w2, Memory<FP16> l2,
                                   Memory<FP16> m, Memory<FP16> uncertaintyV,
                                   Memory<NNEvaluatorResultActivations> activations,
                                   TimingStats stats, Memory<FP16> extraStat0 = default, Memory<FP16> extraStat1 = default, bool makeCopy = false)
    {
      IsWDL = isWDL;
      HasM = hasM;
      NumPos = numPos;
      HasUncertaintyV = hasUncertaintyV;
      HasValueSecondary = hasValueSecondary;

      Policies = makeCopy ? policies.Slice(0, numPos).ToArray() : policies;
      Activations = (activations.Length != 0 && makeCopy) ? activations.Slice(0, numPos).ToArray() : activations;

      W = makeCopy ? w.Slice(0, numPos).ToArray() : w;
      L = (isWDL && makeCopy) ? l.Slice(0, numPos).ToArray() : l;

      if (!w2.IsEmpty)
      {
        W2 = makeCopy ? w2.Slice(0, numPos).ToArray() : w2;
        L2 = (isWDL && makeCopy) ? l2.Slice(0, numPos).ToArray() : l2;
      }

      M = (hasM && makeCopy) ? m.Slice(0, numPos).ToArray() : m;
      UncertaintyV = (HasUncertaintyV && makeCopy) ? uncertaintyV.Slice(0, numPos).ToArray() : uncertaintyV;

      if (!extraStat0.IsEmpty)
      {
        ExtraStat0 = makeCopy ? extraStat0.Slice(0, numPos).ToArray() : extraStat0;
      }

      if (!extraStat1.IsEmpty)
      {
        ExtraStat1 = makeCopy ? extraStat1.Slice(0, numPos).ToArray() : extraStat1;
      }

      Stats = stats;
    }


    /// <summary>
    /// Constructor (from specified value and policy valeus).
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="hasM"></param>
    /// <param name="numPos"></param>
    /// <param name="valueEvals"></param>
    /// <param name="valueEvals2"></param>
    /// <param name="valsAreLogistic"></param>
    /// <param name="stats"></param>
    private PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasValueSecondary, int numPos,
                                    Span<FP16> valueEvals, Span<FP16> valueEvals2, 
                                    Span<FP16> m, Span<FP16> uncertaintyV,
                                    Span<NNEvaluatorResultActivations> activations,
                                    bool valsAreLogistic, TimingStats stats)
    {
      IsWDL = isWDL;
      HasM = hasM;
      HasUncertaintyV = hasUncertaintyV;
      HasValueSecondary = hasValueSecondary;

      NumPos = numPos;
      Activations = activations.ToArray();

      InitializeValueEvals(valueEvals, valsAreLogistic, false);
      if (!valueEvals2.IsEmpty)
      {
        InitializeValueEvals(valueEvals2, valsAreLogistic, true);
      }

      if (hasM)
      {
        M = m.ToArray();
      }

      if (HasUncertaintyV)
      {
        UncertaintyV = uncertaintyV.ToArray();
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
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasValueSecondary, int numPos, 
                                   Span<FP16> valueEvals, Span<FP16> valueEvals2, Memory<float> policyProbs,
                                   FP16[] m, FP16[] uncertaintyV, NNEvaluatorResultActivations[] activations,
                                   bool valsAreLogistic, PolicyType probType, bool policyAlreadySorted,
                                   IEncodedPositionBatchFlat sourceBatchWithValidMoves,
                                   TimingStats stats)
      : this(isWDL, hasM, hasUncertaintyV, hasValueSecondary, numPos, valueEvals, valueEvals2, m, uncertaintyV, activations, valsAreLogistic, stats)
    {
      Policies = ExtractPoliciesBufferFlat(numPos, policyProbs, probType, policyAlreadySorted, sourceBatchWithValidMoves);
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
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasValueSecondary, int numPos, 
                                   Span<FP16> valueEvals, Span<FP16> valueEvals2,
                                   int topK, Span<int> policyIndices, Span<float> policyProbabilties,
                                   Span<FP16> m, Span<FP16> uncertaintyV,
                                   Span<NNEvaluatorResultActivations> activations, bool valsAreLogistic,
                                   PolicyType probType, TimingStats stats, bool alreadySorted)
      : this(isWDL, hasM, hasUncertaintyV, hasValueSecondary, numPos, valueEvals, valueEvals2, m, uncertaintyV, activations, valsAreLogistic, stats)
    {
      Policies = ExtractPoliciesTopK(numPos, topK, policyIndices, policyProbabilties, probType, alreadySorted);
    }

    /// <summary>
    /// Constructor (for an empty batch, to be filled in later with CopyFrom method).
    /// </summary>
    /// <param name="maxPos"></param>
    /// <param name="retrieveSupplementalResults"></param>
    public PositionEvaluationBatch(bool isWDL, bool hasM, bool hasUncertaintyV, bool hasValueSecondary, int maxPos, bool retrieveSupplementalResults)
    {
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

    }

    /// <summary>
    /// Resets values of this batch to be
    /// copies of values taken from another specified batch.
    /// </summary>
    /// <param name="other"></param>
    /// <exception cref="ArgumentException"></exception>
    public void CopyFrom(PositionEvaluationBatch other)
    {
      if (other.NumPos > W.Length)
      {
        throw new ArgumentException($"Other batch number of positions {other.NumPos} exceeds maximum of {W.Length}");
      }

      IsWDL = other.IsWDL;
      HasM = other.HasM;
      NumPos = other.NumPos;
      HasUncertaintyV = other.HasUncertaintyV;
      HasValueSecondary = other.HasValueSecondary;

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

      if (IsWDL)
      {
        other.L.Slice(0, numPos).CopyTo(L);

        if (!L2.IsEmpty)
        {
          other.L2.Slice(0, numPos).CopyTo(L2);
        }
      }

      if (HasM)
      {
        other.M.Slice(0, numPos).CopyTo(M);
      }

      Stats = other.Stats;
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

    static CompressedPolicyVector[] ExtractPoliciesBufferFlat(int numPos, Memory<float> policyProbs, PolicyType probType, bool alreadySorted, 
                                                              IEncodedPositionBatchFlat sourceBatchWithValidMoves)
    {
      // TODO: possibly needs work.
      // Do we handle WDL correctly? Do we flip the moves if we are black (using positions) ?

      if (policyProbs.IsEmpty)
      {
        return null;
      }

      if (policyProbs.Length != EncodedPolicyVector.POLICY_VECTOR_LENGTH * numPos)
      {
        throw new ArgumentException("Wrong policy size");
      }

      CompressedPolicyVector[] retPolicies = new CompressedPolicyVector[numPos];
      Memory<MGMoveList> moves = sourceBatchWithValidMoves == null ? default : sourceBatchWithValidMoves.Moves;
      Parallel.For(0, numPos, i =>
      {
        Span<float> policyProbsSpan = policyProbs.Span;

        if (policyTempBuffer == null)
        {
          policyTempBuffer = new float[EncodedPolicyVector.POLICY_VECTOR_LENGTH];
        }
        else
        {
          Array.Clear(policyTempBuffer, 0, EncodedPolicyVector.POLICY_VECTOR_LENGTH);
        }

        int startIndex = EncodedPolicyVector.POLICY_VECTOR_LENGTH * i;

        if (probType == PolicyType.Probabilities)
        {
          throw new NotImplementedException();
        }
        else if (sourceBatchWithValidMoves == default)
        {
          // N.B. It is not possible to apply move masking here, 
          //      so it is assumed this is already done by the caller.
          //Array.Copy(policyProbsSpan, startIndex, policyTempBuffer, 0, EncodedPolicyVector.POLICY_VECTOR_LENGTH);
          policyProbsSpan.Slice(startIndex, EncodedPolicyVector.POLICY_VECTOR_LENGTH).CopyTo(policyTempBuffer.AsSpan().Slice(0, EncodedPolicyVector.POLICY_VECTOR_LENGTH));

          const float MIN_PROB = -100;
          CompressedPolicyVector ret = default;
          PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(ref ret, true, EncodedPolicyVector.POLICY_VECTOR_LENGTH, 96,
                                                                              policyTempBuffer, MIN_PROB);
          retPolicies[i] = ret;
        }
        else
        {
          // Compute an array if indices of valid moves.
          Span<int> legalMoveIndices = stackalloc int[128]; // TODO: make this short not int?
          MGMoveList movesThis = moves.Span[i];
          int numLegalMoves = movesThis.NumMovesUsed;
          for (int im = 0; im < numLegalMoves; im++)
          {
            EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesThis.MovesArray[im]);
            legalMoveIndices[im] = encodedMove.IndexNeuralNet;
          }

          // Avoid overflow by subtracting off max
          float max = float.MinValue;
          for (int j = 0; j < numLegalMoves; j++)
          {
            float val = policyProbsSpan[startIndex + legalMoveIndices[j]];
            if (val > max) max = val;
          }

          double acc = 0;
          for (int j = 0; j < numLegalMoves; j++)
          {
            float prob = policyProbsSpan[startIndex + legalMoveIndices[j]];
            if (prob > -1E10)
            {
              float value = (float)Math.Exp(prob - max);
              policyTempBuffer[legalMoveIndices[j]] = value;
              acc += value;
            }
            else
            {
              policyTempBuffer[legalMoveIndices[j]] = 0;
            }
          }

          if (numLegalMoves > 0)
          {
            if (acc == 0.0)
            {
              throw new Exception("Sum of unnormalized probabilities was zero.");
            }

            // As performance optimization, only adjust if significantly different from 1.0
            const float MAX_DEVIATION = 0.002f;
            if (acc < 1.0f - MAX_DEVIATION || acc > 1.0f + MAX_DEVIATION)
            {
              for (int j = 0; j < numLegalMoves; j++)
              {
                int targetIndex = legalMoveIndices[j];
                policyTempBuffer[targetIndex] = (float)(policyTempBuffer[targetIndex] / acc);
              }
            }

            CompressedPolicyVector.Initialize(ref retPolicies[i], policyTempBuffer, alreadySorted);
          }
        }
      });

      return retPolicies;
    }


    /// <summary>
    /// Initializes the win/loss array with speicifed values from the value head.
    /// </summary>
    /// <param name="valueEvals"></param>
    /// <param name="valsAreLogistic"></param>
    void InitializeValueEvals(Span<FP16> valueEvals, bool valsAreLogistic, bool secondaryValue)
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
            w[i] = valueEvals[i * 3 + 0];
            l[i] = valueEvals[i * 3 + 2];
            Debug.Assert(Math.Abs(1 - (valueEvals[i * 3 + 0] + valueEvals[i * 3 + 1] + valueEvals[i * 3 + 2])) <= 0.001);
          }
          else
          {
            // NOTE: Use min with 20 to deal with excessively large values (that would go to infinity)
            double v1 = Math.Exp(Math.Min(20, valueEvals[i * 3 + 0]));
            double v2 = Math.Exp(Math.Min(20, valueEvals[i * 3 + 1]));
            double v3 = Math.Exp(Math.Min(20, valueEvals[i * 3 + 2]));

            double totl = v1 + v2 + v3;
            Debug.Assert(!double.IsNaN(totl));

            w[i] = (FP16)(v1 / totl);
            l[i] = (FP16)(v3 / totl);
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
