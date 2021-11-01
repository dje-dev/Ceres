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
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
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
    public readonly bool IsWDL;
    public readonly bool HasM;
    public readonly int NumPos;

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
    /// Vector of moves left estimates.
    /// </summary>
    public Memory<FP16> M;

    /// <summary>
    /// Activations of inner layers.
    /// </summary>
    public Memory<NNEvaluatorResultActivations> Activations;  

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

    /// <summary>
    /// Returns the value of the MLH head for the position at a specified index in the batch.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public FP16 GetM(int index) => HasM ? M.Span[index] : FP16.NaN;

    /// <summary>
    /// Returns inner layer activations.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    NNEvaluatorResultActivations GetActivations(int index) => Activations.IsEmpty ? null : Activations.Span[index];

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
    /// Returns the number of positions in the batch.
    /// </summary>
    int IPositionEvaluationBatch.NumPos => NumPos;


    /// <summary>
    /// Returns a string representation of the batch (summary).
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string timeStr = Stats != null ? $" time: {Stats.ElapsedTimeSecs:F2} secs" : "";
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
    static CompressedPolicyVector[] ExtractPoliciesTopK(int numPos, int topK, Span<int> indices, Span<float> probabilities, PolicyType probType)
    {
      if (probType == PolicyType.LogProbabilities)
      {
        throw new NotImplementedException();
        for (int i = 0; i < indices.Length; i++)
          probabilities[i] = MathF.Exp(probabilities[i]);
      }

      if (indices == null && probabilities == null) return null;
      if (probabilities.Length != indices.Length) throw new ArgumentException("Indices and probabilties expected to be same length");

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

    static CompressedPolicyVector[] ExtractPoliciesBufferFlat(int numPos, float[] policyProbs, PolicyType probType, bool alreadySorted, 
                                                              IEncodedPositionBatchFlat sourceBatchWithValidMoves)
    {
      // TODO: possibly needs work.
      // Do we handle WDL correctly? Do we flip the moves if we are black (using positions) ?

      if (policyProbs == null)
      {
        return null;
      }

      if (policyProbs.Length != EncodedPolicyVector.POLICY_VECTOR_LENGTH * numPos)
      {
        throw new ArgumentException("Wrong policy size");
      }

      CompressedPolicyVector[] retPolicies = new CompressedPolicyVector[numPos];
      Span<float> policyProbsSpan = policyProbs.AsSpan();

      Parallel.For(0, numPos, i =>
      {
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
          //Array.Copy(policyProbs, startIndex, policyTempBuffer, 0, EncodedPolicyVector.POLICY_VECTOR_LENGTH);
        }
        else
        {
          // Compute an array if indices of valid moves.
          Span<int> legalMoveIndices = stackalloc int[128]; // TODO: make this short not int?
          int numLegalMoves = sourceBatchWithValidMoves.Moves[i].NumMovesUsed;

          for (int im = 0; im < numLegalMoves; im++)
          {
            EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(sourceBatchWithValidMoves.Moves[i].MovesArray[im]);
            legalMoveIndices[im] = encodedMove.IndexNeuralNet;
          }

          // Avoid overflow by subtracting off max
          float max = 0.0f;
          for (int j = 0; j < numLegalMoves; j++)
          {
            float val = policyProbs[startIndex + legalMoveIndices[j]];
            if (val > max) max = val;
          }

          double acc = 0;
          for (int j = 0; j < numLegalMoves; j++)
          {
            float prob = policyProbs[startIndex + legalMoveIndices[j]];
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
      });

      return retPolicies;
    }


    /// <summary>
    /// Initializes the win/loss array with speicifed values from the value head.
    /// </summary>
    /// <param name="valueEvals"></param>
    /// <param name="valsAreLogistic"></param>
    void InitializeValueEvals(Span<FP16> valueEvals, bool valsAreLogistic)
    {
      if (IsWDL)
      {
        FP16[] w = new FP16[NumPos];
        FP16[] l = new FP16[NumPos];

        W = w;
        L = l;

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
        W = valueEvals.ToArray();
      }

    }


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
    public PositionEvaluationBatch(bool isWDL, bool hasM, int numPos, Memory<CompressedPolicyVector> policies,
                             Memory<FP16> w, Memory<FP16> l, Memory<FP16> m,
                             Memory<NNEvaluatorResultActivations> activations,
                             TimingStats stats, bool makeCopy = false)
    {
      IsWDL = isWDL;
      HasM = hasM;
      NumPos = numPos;

      Policies = makeCopy ? policies.ToArray() : policies;
      Activations = activations;

      W = makeCopy ? w.ToArray() : w;
      L = (isWDL && makeCopy) ? l.ToArray() : l;
      M = (hasM && makeCopy) ? m.ToArray() : m;

      Stats = stats;
    }


    /// <summary>
    /// Constructor (from specified value and policy valeus).
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="hasM"></param>
    /// <param name="numPos"></param>
    /// <param name="valueEvals"></param>
    /// <param name="valueHeadConvFlat"></param>
    /// <param name="valsAreLogistic"></param>
    /// <param name="stats"></param>
    private PositionEvaluationBatch(bool isWDL, bool hasM, int numPos, 
                                    Span<FP16> valueEvals, Span<FP16> m,
                                    Span<NNEvaluatorResultActivations> activations, 
                                    bool valsAreLogistic, TimingStats stats)
    {
      IsWDL = isWDL;
      HasM = hasM;

      NumPos = numPos;
      Activations = activations.ToArray();

      InitializeValueEvals(valueEvals, valsAreLogistic);
      if (hasM)
      {
        this.M = m.ToArray();
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
    public PositionEvaluationBatch(bool isWDL, bool hasM, int numPos, Span<FP16> valueEvals, float[] policyProbs, 
                                   FP16[] m, NNEvaluatorResultActivations[] activations,
                                   bool valsAreLogistic, PolicyType probType, bool policyAlreadySorted,
                                   IEncodedPositionBatchFlat sourceBatchWithValidMoves,
                                   TimingStats stats) 
      : this(isWDL, hasM, numPos, valueEvals, m, activations, valsAreLogistic, stats)
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
    public PositionEvaluationBatch(bool isWDL, bool hasM, int numPos, Span<FP16> valueEvals, 
                             int topK, Span<int> policyIndices, Span<float> policyProbabilties,
                             Span<FP16> m,
                             Span<NNEvaluatorResultActivations> activations, bool valsAreLogistic,
                             PolicyType probType, TimingStats stats)
      : this(isWDL, hasM, numPos, valueEvals, m, activations, valsAreLogistic, stats)
    {
      Policies = ExtractPoliciesTopK(numPos, topK, policyIndices, policyProbabilties, probType);
    }


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


    NNEvaluatorResultActivations IPositionEvaluationBatch.GetActivations(int index) => GetActivations(index);

    public IEnumerator<NNPositionEvaluationBatchMember> GetEnumerator()
    {
      for (int i = 0; i < NumPos; i++)
        yield return new NNPositionEvaluationBatchMember(this, i);
    }


    IEnumerator IEnumerable.GetEnumerator()
    {
      throw new NotImplementedException();
    }


  }
}
