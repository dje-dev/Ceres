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
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess.LC0.Batches;

#endregion

namespace Ceres.Chess.LC0NetInference
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
    public readonly float[] PolicyVectors;

    /// <summary>
    /// Action head.
    /// </summary>
    public readonly FP16[] ActionLogisticVectors;

    /// <summary>
    /// Moves left head.
    /// </summary>
    public readonly float[] MLH;

    /// <summary>
    /// Uncertainty of V head.
    /// </summary>
    public readonly float[] UncertaintyV;

    /// <summary>
    /// Activation values for last FC layer before value output (possibly null)
    /// </summary>
    public readonly float[][] ValueFCActivations;

    public FP16[] ValuesRaw;
    public FP16[] Values2Raw;

    public int BatchSize => PolicyVectors.GetLength(0);

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
    public ONNXRuntimeExecutorResultBatch(bool isWDL, FP16[] values, FP16[] values2, float[] policyLogisticVectors,
                                          float[] mlh, float[] uncertaintyV, float[][] valueFCActiviations,
                                          FP16[] actionLogisticVectors, int numPositionsUsed)
    {
      ValuesRaw = values;
      Values2Raw = values2;

      Debug.Assert(!FP16.IsNaN(values[0]));
      if (isWDL)
      { 
        Debug.Assert(!FP16.IsNaN(values[1]) && !FP16.IsNaN(values[2]));
      }

      PolicyVectors = policyLogisticVectors; // still in logistic form
      ActionLogisticVectors = actionLogisticVectors; // still in logistic form

      MLH = mlh;
      UncertaintyV = uncertaintyV;
      ValueFCActivations = valueFCActiviations;
      NumPositionsUsed = numPositionsUsed;
      IsWDL = isWDL;
    }

    #region Helpers



    /// <summary>
    ///
    /// We have to transform the inputs to look identical to what LZ0 expects.
    /// This involves reordering some planes, dividing the Rule50 by 99, and filling the last plane with 1's
    /// TO DO: possibly adopt this for our inputs
    /// </summary>
    /// <param name="floats"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    public static float[] RebuildInputsForLC0Network(Memory<float> floats, int batchSize)
    {
      float[] expandedFloats = new float[batchSize * 64 * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES];
      for (int i = 0; i < batchSize; i++)
      {
        int baseSrcThisBatchItem = i * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES * 64;
        int baseDstThisBatchItem = i * 112 * 64;

        void CopyPlanes(int numPlanesCopy, int planeFirstIndexInSource, int fillValue = -1, float divideValue = 1.0f)
        {
          int newSrce = baseSrcThisBatchItem + (planeFirstIndexInSource * 64);
          if (fillValue != -1)
          {
            for (int j = 0; j < numPlanesCopy * 64; j++)
              expandedFloats[baseDstThisBatchItem + j] = fillValue;
          }
          else if (divideValue != 1)
          {
            Span<float> floatsS = floats.Span;
            for (int j = 0; j < numPlanesCopy * 64; j++)
              expandedFloats[baseDstThisBatchItem + j] = floatsS[newSrce + j] / divideValue;
          }
          else
          {
            int size = numPlanesCopy * 64;
            floats.Slice(newSrce, size).CopyTo(expandedFloats.AsMemory<float>().Slice(baseDstThisBatchItem, size));
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

      return expandedFloats;
    }

    #endregion
  }
}
