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
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators.Defs;
using Chess.Ceres.NNEvaluators;
using Microsoft.ML.OnnxRuntime;

#endregion

namespace Ceres.Chess.LC0NetInference
{
  /// <summary>
  /// A neural network executor using the ONNX runtime.
  /// See: https://github.com/microsoft/onnxruntime.
  /// </summary>
  public class ONNXRuntimeExecutor : IDisposable
  {
    public const int TPG_BYTES_PER_SQUARE_RECORD = 137; // TODO: should be referenced from TPGRecord
    public const int TPG_MAX_MOVES = 92; //  // TODO: should be referenced from TPGRecord


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

    /// <summary>
    /// Type of neural network.
    /// </summary>
    public readonly NetTypeEnum NetType;

    /// <summary>
    /// Device type (CPU or GPU).
    /// </summary>
    public readonly NNDeviceType DeviceType;

    /// <summary>
    /// Underlying ONNX executor object.
    /// </summary>
    NetExecutorONNXRuntime executor;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="shortID"></param>
    /// <param name="onnxFileName"></param>
    /// <param name="onnxModelBytes"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="netType"></param>
    /// <param name="deviceType"></param>
    /// <param name="gpuNum"></param>
    /// <param name="enableProfiling"></param>
    public ONNXRuntimeExecutor(string shortID,
                               string onnxFileName, byte[] onnxModelBytes, 
                               string[] inputNames,
                               int maxBatchSize, 
                               NetTypeEnum netType, 
                               NNEvaluatorPrecision precision, 
                               NNDeviceType deviceType, int gpuNum, 
                               bool useTRT, 
                               bool enableProfiling)
    {
      if (onnxFileName != null && !onnxFileName.ToUpper().EndsWith(".ONNX"))
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
      BatchSize = maxBatchSize;
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

      executor = new NetExecutorONNXRuntime(shortID, onnxFileName, onnxModelBytes, inputNames,
                                            precision, deviceIndex, useTRT, enableProfiling);
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
    public ONNXRuntimeExecutorResultBatch Execute(bool isWDL, bool hasState,
                                                  Memory<float> flatValuesPrimary, Memory<float> flatValuesSecondary, int numPositionsUsed, 
                                                  bool debuggingDump = false, bool alreadyConvertedToLZ0 = false,
                                                  float tpgDivisor = 1)
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

      List<(string, Memory<Float16>)> eval;

      if (NetType == NetTypeEnum.TPG)
      {
        int NUM_INPUTS = hasState ? 2 : 1; // assume state present
        (Memory<float> input, int[] shape)[] inputs = new (Memory<float> input, int[] shape)[NUM_INPUTS];
        if (tpgDivisor != 1.0f)
        {
          Span<float> flatValuesPrimarySpan = flatValuesPrimary.Span;
          for (int i=0;i<flatValuesPrimarySpan.Length;i++)
          {
            flatValuesPrimarySpan[i] /= tpgDivisor;
          }
        }

        const int STATE_NUM_PER_SQUARE = 4; // TODO: Fix this, currently hardcoded the only value actually currently used by Ceres nets.

        inputs[0] = (flatValuesPrimary, new int[] { numPositionsUsed, 64, TPG_BYTES_PER_SQUARE_RECORD });
        if (inputs.Length > 1)
        {
          inputs[1] = (flatValuesSecondary, new int[] { numPositionsUsed, 64, STATE_NUM_PER_SQUARE }); 
        } 

        if (hasState)
        {
          // fake it with zeros
          inputs[1] = (new float[numPositionsUsed*64* STATE_NUM_PER_SQUARE], new int[] { numPositionsUsed, 64, STATE_NUM_PER_SQUARE });

        }
#if NOT
        if (flatValuesSecondary.Length > 0)
        {
          Span<float> flatValuesSecondaryS = flatValuesSecondary.Span;
          for (int i = 0; i < flatValuesSecondary.Length; i++) flatValuesSecondaryS[i] /= DIVISOR;
          inputs[1] = (new Memory<float>(flatValuesSecondaryS.ToArray()), new int[] { numPositionsUsed, TPG_MAX_MOVES, TPG_BYTES_PER_MOVE_RECORD });
        }
#endif

        eval = executor.Run(inputs, numPositionsUsed, Precision == NNEvaluatorPrecision.FP16);
      }
      else
      {
        (Memory<float> flatValuesPrimary, int[]) input = default;
        input.Item1 = flatValuesPrimary.Slice(0, numPositionsUsed * 112 * 8 * 8);
        input.Item2 = [numPositionsUsed, 112, 8, 8 ];
        eval = executor.Run([input], numPositionsUsed, Precision == NNEvaluatorPrecision.FP16);
      }

      int numPlanes = NetType == NetTypeEnum.Ceres ? EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES : 112;

      if (NetType == NetTypeEnum.Ceres)
      {
        throw new NotImplementedException();
        //nRunner = session.GetRunner().AddInput("input_1", inputTensor).Fetch("value_out/Tanh").Fetch("policy_out/Softmax").Fetch("draw_out/Sigmoid");
      }
      else
      {
        bool hasMLH = eval.Count >= 3;
        bool hasUNC = eval.Count >= 4;
#if NOT
        Session.InputMetadata
        [0] input_1

        Session.OutputMetadata
        [0] apply_attention_policy_map [apply_policy_map IF NOT ATTENTION]
        [1] moves_left/dense2
        [2] tf.math.truediv [OMITTED IF NOT ATTENTION]
        [3] value/dense2
#endif
        int FindIndex(int expectedPerPosition, int indexToIgnore = -1, string mustContainString = null, bool optional = false)
        {
          int expectedLength = numPositionsUsed * expectedPerPosition;
          for (int i = 0; i < eval.Count; i++)
          {
            if (eval[i].Item2.Length == expectedLength 
              && i != indexToIgnore
              && (mustContainString == null || eval[i].Item1.Contains(mustContainString)))
            {
              // TODO: find a better way to do this
              // HACK: If the filename contains "optimistic",
              //       then accept policy vector only if contains string "optimistic"
              if (expectedPerPosition == 1858)
              {
                if (
                   (ONNXFileName.ToLower().Contains("policy_optimistic") && !eval[i].Item1.ToLower().Contains("optimistic"))
                || (ONNXFileName.ToLower().Contains("policy_vanilla") && !eval[i].Item1.ToLower().Contains("vanilla"))
                   )
                {
                  continue;
                }
              }
                if (expectedPerPosition == 3)
                {
                  if (
                     (ONNXFileName.ToLower().Contains("value_winner") && !eval[i].Item1.ToLower().Contains("value/winner"))
                  || (ONNXFileName.ToLower().Contains("value_q") && !eval[i].Item1.ToLower().Contains("value/q"))
                     )
                  {
                    continue;
                  }
                }
                  //+		[25]	("value/q/dense2", {float[3]})	(string, float[])
                  //                +[26]("value/q/dense_error", { float[1]})	(string, float[])
                  //               + [29]("value/winner/dense2", { float[3]})	(string, float[])
               

              return i;
            }
          }

          return optional ? -1 : throw new Exception("No output found with expected length " + expectedPerPosition);
        }


        // Rather than rely upon names, just look at the dimensions
        // of the outputs to infer the positions of value, policy and MLH heads.
        // TODO: currently limited to assuming MLH and WDL true, can this be improved?
        if (!isWDL || !hasMLH)
        {
          throw new Exception("Implementation restriction, ONNX runtime nets expected to have both WDL and MLH heads.");
        }

        // TODO: remove hack fo rBT3
        bool looksLikeBT3 = eval.Count > 10;
        string uncMustContainString = looksLikeBT3 ? "value/q/dense_error" : null;

        int INDEX_POLICIES = FindIndex(1858);// FIX NetType == NetTypeEnum.Ceres ? 1858 : 96);
        int INDEX_WDL = FindIndex(3);
        int INDEX_WDL2 = FindIndex(3, INDEX_WDL, "value2", true);
        int INDEX_MLH = FindIndex(1);
        int INDEX_UNC = hasUNC ? FindIndex(1, INDEX_MLH) : -1;
        int INDEX_ACTION = FindIndex(1858 * 3, -1, "action", true); // TODO: cleanup the output names to be better named

        if (looksLikeBT3)
        {
          Console.WriteLine("ONNX head mappings for " + ONNXFileName);
          Console.WriteLine("value       --> " + eval[INDEX_WDL].Item1);
          Console.WriteLine("policy      --> " + eval[INDEX_POLICIES].Item1);
          Console.WriteLine("mlh         --> " + eval[INDEX_MLH].Item1);
          Console.WriteLine("uncertainty --> " + eval[INDEX_UNC].Item1);
        }

        Memory<Float16> mlh = hasMLH ? eval[INDEX_MLH].Item2 : null;
        Memory<Float16> uncertantiesV = hasUNC ? eval[INDEX_UNC].Item2 : null;
        Memory<Float16> policiesLogistics = eval[INDEX_POLICIES].Item2;

        Memory<Float16> actionLogistics = INDEX_ACTION != -1 ? eval[INDEX_ACTION].Item2 : default;
        Memory<Float16> values = eval[INDEX_WDL].Item2;
        Debug.Assert(values.Length == (isWDL ? 3 : 1) * numPositionsUsed);

        Memory<Float16> values2 = INDEX_WDL2 == -1 ? default : eval[INDEX_WDL2].Item2;
        float[][] value_fc_activations = null;// eval.Length < 3 ? null : eval[2];

        // SWAP VALUE1, VALUE2: (values, values2) = (values2, values); // swap *** TEMPORARY

        Memory<Float16> extraStats0 = eval.Count > 5 ? eval[5].Item2 : default;
        Memory<Float16> extraStats1 = eval.Count > 6 ? eval[6].Item2 : default;

        // TODO: This is just a fake, fill it in someday
        Memory<Float16> priorState = hasState ? new Float16[numPositionsUsed * 64 * 4] : default;
        ONNXRuntimeExecutorResultBatch result = new ONNXRuntimeExecutorResultBatch(isWDL, values, values2, policiesLogistics, mlh, 
                                                                                   uncertantiesV, extraStats0, extraStats1, value_fc_activations,
                                                                                   actionLogistics, priorState,
                                                                                   numPositionsUsed);
        return result;

      }
    }


    public void EndProfiling() => executor.EndProfiling();


    public void Dispose()
    {
      executor.Dispose();
    }
  }

}

