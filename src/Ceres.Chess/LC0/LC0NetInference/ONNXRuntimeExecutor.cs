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
using System.Text;
using Ceres.Base.DataTypes;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators.Defs;
using Chess.Ceres.NNEvaluators;
using Microsoft.ML.OnnxRuntime.Tensors;

#endregion

namespace Ceres.Chess.LC0NetInference
{
  /// <summary>
  /// A neural network executor using the ONNX runtime.
  /// See: https://github.com/microsoft/onnxruntime.
  /// </summary>
  public class ONNXRuntimeExecutor : IDisposable
  {
    /// <summary>
    /// File name of source ONNX file
    /// </summary>
    public readonly string ONNXFileName;

    /// <summary>
    /// Precision of network.
    /// </summary>
    public readonly NNEvaluatorPrecision Precision;

    /// <summary>
    /// 
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Type of neural network (Leela Chess Zero or Ceres).
    /// </summary>
    public enum NetTypeEnum { Ceres, LC0, TPG };

    public readonly NetTypeEnum NetType;

    public readonly NNDeviceType DeviceType;

    /// <summary>
    /// Underlying ONNX executor object.
    /// </summary>
    NetExecutorONNXRuntime executor;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="onnxFileName"></param>
    /// <param name="batchSize"></param>
    /// <param name="netType"></param>
    /// <param name="deviceType"></param>
    /// <param name="gpuNum"></param>
    public ONNXRuntimeExecutor(string onnxFileName, int batchSize, NetTypeEnum netType, 
                               NNEvaluatorPrecision precision, NNDeviceType deviceType, int gpuNum, bool useTRT)
    {
      if (!onnxFileName.ToUpper().EndsWith(".ONNX"))
      {
        throw new Exception("Expected file with ONNX extension.");
      }

      if (precision != NNEvaluatorPrecision.FP32 && precision != NNEvaluatorPrecision.FP16)
      {
        throw new ArgumentException($"Only supported ONNX precisions are FP32 and FP16, not {precision}");
      }

      ONNXFileName = onnxFileName;
      NetType = netType;
      DeviceType = deviceType;
      BatchSize = batchSize;
      Precision = precision;

      int deviceIndex;
      if (deviceType == NNDeviceType.GPU)
      {
        deviceIndex = gpuNum;
      }
      else if (deviceType == NNDeviceType.CPU)
      {
        deviceIndex = -1; // by convention this indicates CPU
      }
      else
      {
        throw new NotImplementedException("Unsupported ONNX type " + deviceType);
      }

      executor = new NetExecutorONNXRuntime(onnxFileName, precision, deviceIndex, useTRT);
    }


    /// <summary>
    /// Evaluates a batch.
    /// </summary>
    /// <param name="isWDL"></param>
    /// <param name="positionEncoding"></param>
    /// <param name="numPositionsUsed"></param>
    /// <param name="debuggingDump"></param>
    /// <param name="alreadyConvertedToLZ0"></param>
    /// <returns></returns>
    public ONNXRuntimeExecutorResultBatch Execute(bool isWDL, 
                                                  float[] flatValuesPrimary, float[] flatValuesSecondary, int numPositionsUsed, 
                                                  bool debuggingDump = false, bool alreadyConvertedToLZ0 = false)
    {
      if (!alreadyConvertedToLZ0)
      {
        if (flatValuesPrimary.Length / BatchSize != (64 * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES))
        {
          throw new Exception("Internal error: incorrect number of planes.");
        }

        if (NetType == NetTypeEnum.LC0)
        {
          flatValuesPrimary = ONNXRuntimeExecutorResultBatch.RebuildInputsForLC0Network(flatValuesPrimary, BatchSize); // Centralize this
        }
        else if (NetType == NetTypeEnum.TPG)
        {
        }
        else
        {
          throw new NotImplementedException();
        }
      }


      // ** NICE DEBUGGING!
      if (debuggingDump && NetType != NetTypeEnum.TPG)
      {
        EncodedPositionBatchFlat.DumpDecoded(flatValuesPrimary, 112);
      }

      float[][] eval;

      if (NetType == NetTypeEnum.TPG)
      {
        var inputs = new (Memory<float> input, int[] shape)[flatValuesSecondary == null ? 1 : 2];
        inputs[0] = (flatValuesPrimary, new int[] { numPositionsUsed, 96, 38 });
        if (flatValuesSecondary != null)
        {
          inputs[1] = (flatValuesSecondary, new int[] { numPositionsUsed, 782 });
        }

        eval = executor.Run(inputs, Precision == NNEvaluatorPrecision.FP16);
      }
      else
      {
        var input = (flatValuesPrimary, new int[] { numPositionsUsed, 112, 8, 8 });
        eval = executor.Run(new (Memory<float> input, int[] shape)[] { input }, Precision == NNEvaluatorPrecision.FP16);
      }

      const int VALUE_FC_SIZE = 32 * 64;

      int numPlanes = NetType == NetTypeEnum.Ceres ? EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES : 112;

      if (NetType == NetTypeEnum.Ceres)
      {
        throw new NotImplementedException();
        //nRunner = session.GetRunner().AddInput("input_1", inputTensor).Fetch("value_out/Tanh").Fetch("policy_out/Softmax").Fetch("draw_out/Sigmoid");
      }
      else
      {
        bool hasMLH = eval.Length >= 3;

#if NOT
        Session.InputMetadata
        [0] input_1

        Session.OutputMetadata
        [0] apply_attention_policy_map [apply_policy_map IF NOT ATTENTION]
        [1] moves_left/dense2
        [2] tf.math.truediv [OMITTED IF NOT ATTENTION]
        [3] value/dense2
#endif
        int FindIndex(int expectedPerPosition)
        {
          int expectedLength = numPositionsUsed * expectedPerPosition;
          for (int i=0;i<eval.Length;i++)
          {
            if (eval[i].Length == expectedLength)
            {
              return i;
            }
          }
          throw new Exception("No output found with expected length " + expectedPerPosition);
        }

        // Rather than rely upon names, just look at the dimensions
        // of the outputs to infer the positions of value, policy and MLH heads.
        // TODO: currently limited to assuming MLH and WDL true, can this be improved?
        if (!isWDL || !hasMLH)
        {
          throw new Exception("Implmentation restriction, ONNX runtime nets expected to have  both WDL and MLH heads.");
        }

        int INDEX_POLICIES = FindIndex(1858);// FIX NetType == NetTypeEnum.Ceres ? 1858 : 96);
        int INDEX_MLH = FindIndex(1);
        int INDEX_WDL = FindIndex(3);

        float[] mlh = hasMLH ? eval[INDEX_MLH] : null;
        float[] policiesLogistics = eval[INDEX_POLICIES];

        FP16[] values = FP16.ToFP16(eval[INDEX_WDL]);
        Debug.Assert(values.Length == (isWDL ? 3 : 1) * numPositionsUsed);

        float[] value_fc_activations = null;// eval.Length < 3 ? null : eval[2];
        int? overridePolicyLen = NetType == NetTypeEnum.TPG ? 96 : null;
        ONNXRuntimeExecutorResultBatch result = new ONNXRuntimeExecutorResultBatch(isWDL, values, policiesLogistics, mlh, value_fc_activations, 
                                                                                   numPositionsUsed, overridePolicyLen);
        return result;

      }
    }

    public void Dispose()
    {
      executor.Dispose();
    }
  }

}

