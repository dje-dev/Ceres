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
    /// 
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Type of neural network (Leela Chess Zero or Ceres).
    /// </summary>
    public enum NetTypeEnum { Ceres, LC0 };

    public readonly NetTypeEnum NetType;

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
    /// <param name="gpuNum"></param>
    public ONNXRuntimeExecutor(string onnxFileName, int batchSize, NetTypeEnum netType, int gpuNum = -1)
    {
      if (!onnxFileName.ToUpper().EndsWith(".ONNX")) throw new Exception("Expected file with ONNX extension.");

      ONNXFileName = onnxFileName;
      NetType = netType;
      BatchSize = batchSize;

      executor = new NetExecutorONNXRuntime(onnxFileName, gpuNum);
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
    public ONNXRuntimeExecutorResultBatch Execute(bool isWDL, float[] positionEncoding, int numPositionsUsed, 
                                                 bool debuggingDump = false, bool alreadyConvertedToLZ0 = false)
    {
      if (!alreadyConvertedToLZ0)
      {
        if (positionEncoding.Length / BatchSize != (64 * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES)) 
          throw new Exception();

        if (NetType == NetTypeEnum.LC0)
        {
          positionEncoding = ONNXRuntimeExecutorResultBatch.RebuildInputsForLC0Network(positionEncoding, BatchSize); // Centralize this
        }
        else
          throw new NotImplementedException();
      }

      // ** NICE DEBUGGING!
      if (debuggingDump) EncodedPositionBatchFlat.DumpDecoded(positionEncoding, 112);

      float[][] eval = executor.Run(positionEncoding, new int[] { numPositionsUsed, 112, 64 });

      const int VALUE_FC_SIZE = 32 * 64;

      int numPlanes = NetType == NetTypeEnum.Ceres ? EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES : 112;

      if (NetType == NetTypeEnum.Ceres)
      {
        throw new NotImplementedException();
        //nRunner = session.GetRunner().AddInput("input_1", inputTensor).Fetch("value_out/Tanh").Fetch("policy_out/Softmax").Fetch("draw_out/Sigmoid");
      }
      else
      {
        float[] mlh = eval[0];

        float[] policiesLogistics = eval[1];

        FP16[] values = FP16.ToFP16(eval[2]);
        Debug.Assert(values.Length == (isWDL ? 3 : 1) * numPositionsUsed);

        float[] value_fc_activations = null;// eval.Length < 3 ? null : eval[2];

        ONNXRuntimeExecutorResultBatch result = new ONNXRuntimeExecutorResultBatch(isWDL, values, policiesLogistics, mlh, value_fc_activations, numPositionsUsed);
        return result;

      }
    }

    public void Dispose()
    {
      executor.Dispose();
    }
  }

}

