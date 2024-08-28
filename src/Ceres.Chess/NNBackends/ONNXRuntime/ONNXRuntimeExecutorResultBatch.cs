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
using System.Collections.Generic;
using System.Diagnostics;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess.LC0.Batches;
using Microsoft.ML.OnnxRuntime;

#endregion

namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  /// <summary>
  /// Represents results coming back from NN for a batch of positions.
  /// </summary>
  public class ONNXRuntimeExecutorResultBatch
  {
    public readonly bool IsWDL;

    /// <summary>
    /// Policy head.
    /// </summary>
    public readonly Memory<Float16> PolicyVectors;

    /// <summary>
    /// Action head.
    /// </summary>
    public readonly Memory<Float16> ActionLogits;

    /// <summary>
    /// Moves left head.
    /// </summary>
    public readonly Memory<Float16> MLH;

    /// <summary>
    /// Uncertainty of value head.
    /// </summary>
    public readonly Memory<Float16> UncertaintyV;

    /// <summary>
    /// Uncertainty of policy head.
    /// </summary>
    public readonly Memory<Float16> UncertaintyP;

    public readonly Memory<Float16> PriorState;

    /// <summary>
    /// Activation values for last FC layer before value output (possibly null)
    /// </summary>
    public readonly float[][] ValueFCActivations;

    public Memory<Float16> ValuesRaw;
    public Memory<Float16> Values2Raw;

    public readonly Memory<Float16> ExtraStats0;
    public readonly Memory<Float16> ExtraStats1;

    /// <summary>
    /// Optional dictionary of the raw neural network outputs.
    /// </summary>
    public Dictionary<string, Float16[]> RawNetworkOutputs;



    /// <summary>
    /// Number of positions with actual data
    /// If the batch was padded, with actual number of positions less than a full batch,
    /// then NumPositionsUsed will be less than BatchSize.
    /// </summary>
    public readonly int NumPositionsUsed;


    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="values"></param>
    /// <param name="policyLogisticVectors"></param>
    /// <param name="draws"></param>
    public ONNXRuntimeExecutorResultBatch(bool isWDL, Memory<Float16> values, Memory<Float16> values2, Memory<Float16> policyLogisticVectors,
                                          Memory<Float16> mlh, Memory<Float16> uncertaintyV, Memory<Float16> uncertaintyP,
                                          Memory<Float16> extraStats0, Memory<Float16> extraStats1,
                                          float[][] valueFCActiviations,
                                          Memory<Float16> actionLogisticVectors, Memory<Float16> priorState, 
                                          int numPositionsUsed,
                                          List<(string, Memory<Float16>)> rawNetworkOutputs = null)
    {
      ValuesRaw = values;
      Values2Raw = values2;

      Debug.Assert(!float.IsNaN((float)values.Span[0]));
      if (isWDL)
      {
        Debug.Assert(!float.IsNaN((float)values.Span[1]) && !float.IsNaN((float)values.Span[2]));
      }

      PolicyVectors = policyLogisticVectors; // still in logistic form
      ActionLogits = actionLogisticVectors; // still in logistic form

      MLH = mlh;
      ExtraStats0 = extraStats0;
      ExtraStats1 = extraStats1;

      UncertaintyV = uncertaintyV;
      UncertaintyP = uncertaintyP;
      PriorState = priorState;
      ValueFCActivations = valueFCActiviations;
      NumPositionsUsed = numPositionsUsed;
      IsWDL = isWDL;

      if (rawNetworkOutputs != null)
      {
        RawNetworkOutputs = new Dictionary<string, Float16[]>(rawNetworkOutputs.Count);
        foreach ((string name, Memory<Float16> data) in rawNetworkOutputs)
        {
          RawNetworkOutputs[name] = data.ToArray();
        }
      } 
    }


    #region Helpers



    /// <summary>
    ///
    /// We have to transform the inputs to look identical to what LZ0 expects.
    /// This involves reordering some planes, dividing the Rule50 by 99, and filling the last plane with 1's
    /// TO DO: possibly adopt this for our inputs
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    public static Half[] RebuildInputsForLC0Network(Memory<Half> inputs, int batchSize)
    {
      Half[] expandedResults = new Half[batchSize * 64 * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES];
      for (int i = 0; i < batchSize; i++)
      {
        int baseSrcThisBatchItem = i * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES * 64;
        int baseDstThisBatchItem = i * 112 * 64;

        void CopyPlanes(int numPlanesCopy, int planeFirstIndexInSource, int fillValue = -1, float divideValue = 1.0f)
        {
          Half fillValueHalf = (Half)fillValue;
          int newSrce = baseSrcThisBatchItem + planeFirstIndexInSource * 64;
          if (fillValue != -1)
          {
            for (int j = 0; j < numPlanesCopy * 64; j++)
              expandedResults[baseDstThisBatchItem + j] = fillValueHalf;
          }
          else if (divideValue != 1)
          {
            Span<Half> inputsS = inputs.Span;
            for (int j = 0; j < numPlanesCopy * 64; j++)
            {
              expandedResults[baseDstThisBatchItem + j] = (Half)((float)inputsS[newSrce + j] / divideValue);
            }
          }
          else
          {
            int size = numPlanesCopy * 64;
            inputs.Slice(newSrce, size).CopyTo(expandedResults.AsMemory().Slice(baseDstThisBatchItem, size));
            //            Array.Copy(floats, newSrce, expandedFloats, baseDstThisBatchItem, numPlanesCopy * 64);
          }
          baseDstThisBatchItem += numPlanesCopy * 64;
        }

        CopyPlanes(13, 13 * 0);
        CopyPlanes(13, 13 * 1);
        CopyPlanes(13, 13 * 2);
        CopyPlanes(13, 13 * 3);

        CopyPlanes(13, 13 * 3); // starting here we just repeat plane 3
        CopyPlanes(13, 13 * 3);
        CopyPlanes(13, 13 * 3);
        CopyPlanes(13, 13 * 3);
        CopyPlanes(5, 52);
        CopyPlanes(1, 57, divideValue: 99.0f); // LZ0 implementation handles this by shifting weights instead of inputs TODO: this probably needs remediation
        CopyPlanes(1, 58);
        CopyPlanes(1, 59, fillValue: 1); // last one must be all 1's
      }

      return expandedResults;
    }

    #endregion
  }
}
