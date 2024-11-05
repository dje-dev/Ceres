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

using Microsoft.ML.OnnxRuntime;

using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  /// <summary>
  /// Manages execution of ONNX networks for inference of chess neural networks
  /// (Leela Chess Zero and Ceres).
  /// </summary>
  public class ONNXNetExecutor : IDisposable
  {
    /// In August 2024 with TensorRT 10.2 it was discovered that (very) incorrect results
    /// were returned when batch size 1 was used with certain Ceres networks, presumably a bug.
    /// It was also observed that batch size 4 is no slower (maybe faster) than 1.
    /// Therefore as a workaround we use a minimum batch size of 4 for TensorRT execution provider
    /// for Ceres networks.
    /// </summary>
    internal const int MIN_BATCH_SIZE_TENSOR_RT_CERES = 4;

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
    /// Type of neural network (Leela Chess Zero or Ceres TPG format).
    /// </summary>
    public enum NetTypeEnum { LC0, TPG };

    /// <summary>
    /// Type of neural network.
    /// </summary>
    public readonly NetTypeEnum NetType;

    /// <summary>
    /// Device type (CPU or GPU).
    /// </summary>
    public readonly NNDeviceType DeviceType;

    /// <summary>
    /// If TensorRT execution provider should be used if available.
    /// </summary>
    public readonly bool UseTensorRT;

    /// <summary>
    /// Minimum batch size to be used.
    /// </summary>
    internal int MinBatchSize;

    /// <summary>
    /// Underlying ONNX executor object.
    /// </summary>
    internal ONNXExecutor executor;

    bool retainRawOutputs;

    /// <summary>
    /// If raw outputs should be retained.
    /// </summary>
    public bool RetainRawOutputs
    {
      get => retainRawOutputs;
      set
      {
        retainRawOutputs = value;
        if (executor != null)
        {
          executor.RetainRawInputs = value;
        }
      }
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="shortID"></param>
    /// <param name="onnxFileName"></param>
    /// <param name="onnxModelBytes"></param>
    /// <param name="inputNames"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="netType"></param>
    /// <param name="precision"></param>
    /// <param name="deviceType"></param>
    /// <param name="gpuNum"></param>
    /// <param name="useTensorRT"></param>
    /// <param name="enableProfiling"></param>
    /// <exception cref="Exception"></exception>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="NotImplementedException"></exception>
    public ONNXNetExecutor(string shortID,
                               string onnxFileName, byte[] onnxModelBytes,
                               string[] inputNames,
                               int maxBatchSize,
                               NetTypeEnum netType,
                               NNEvaluatorPrecision precision,
                               NNDeviceType deviceType, int gpuNum,
                               bool useTensorRT,
                               bool enableProfiling,
                               bool retainRawOutputs)
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
      UseTensorRT = deviceType == NNDeviceType.GPU && useTensorRT;
      RetainRawOutputs = retainRawOutputs;

      MinBatchSize = NetType == NetTypeEnum.TPG
                  && UseTensorRT
                  ? MIN_BATCH_SIZE_TENSOR_RT_CERES : 1;

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

      int precisionNumBits = precision switch
      {
        NNEvaluatorPrecision.FP32 => 32,
        NNEvaluatorPrecision.FP16 => 16,
        _ => throw new NotImplementedException($"Unsupported ONNX precision {precision}")
      };

      string nonBatchDimensions = netType switch
      {
        NetTypeEnum.TPG => $"64x{TPG_BYTES_PER_SQUARE_RECORD}",
        NetTypeEnum.LC0 => $"{EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES}x8x8",
        _ => throw new NotImplementedException($"The enum type '{netType}' is not handled."),
      };

      executor = new ONNXExecutor(shortID, onnxFileName, onnxModelBytes, inputNames, nonBatchDimensions,
                                  precisionNumBits, deviceIndex, useTensorRT, MinBatchSize, maxBatchSize, 
                                  enableProfiling, retainRawOutputs);
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
                                                  Memory<Half> flatValuesPrimary, Memory<Half[]> flatValuesState,
                                                  int numPositionsUsed,
                                                  bool debuggingDump = false, bool alreadyConvertedToLZ0 = false,
                                                  float tpgDivisor = 1,
                                                  Predicate<int> shouldUseStateForPos = null)
    {
      if (NetType == NetTypeEnum.TPG && !alreadyConvertedToLZ0)
      {
        throw new NotImplementedException();
      }

      int numPositionsInBatchSentToExecutor = numPositionsUsed < MinBatchSize ? MinBatchSize : numPositionsUsed;

      if (!alreadyConvertedToLZ0)
      {
        if (flatValuesPrimary.Length / BatchSize != 64 * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES)
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
        throw new NotImplementedException("Type switched to Half below");
        //EncodedPositionBatchFlat.DumpDecoded(flatValuesPrimary, 112);
      }

      List<(string, Memory<Float16>)> eval;

      if (NetType == NetTypeEnum.TPG)
      {
        int NUM_INPUTS = executor.NumInputs;
        (Memory<Half> input, int[] shape)[] inputs = new (Memory<Half> input, int[] shape)[NUM_INPUTS];
        if (tpgDivisor != 1.0f)
        {
          Span<Half> flatValuesPrimarySpan = flatValuesPrimary.Span;
          for (int i = 0; i < flatValuesPrimarySpan.Length; i++)
          {
            throw new Exception("don't we already divide in CopyAndDivide?");
            //flatValuesPrimarySpan[i] /= tpgDivisor;
          }
        }

        inputs[0] = (flatValuesPrimary, new int[] { numPositionsUsed, 64, TPG_BYTES_PER_SQUARE_RECORD });
        if (inputs.Length > 1)
        {
          // TODO: improve efficiency here
          if (flatValuesState.IsEmpty)
          {
            // No state available, pass all zeroes.
            inputs[1] = (new Half[numPositionsInBatchSentToExecutor * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE],
                       new int[] { numPositionsInBatchSentToExecutor, 64, NNEvaluator.SIZE_STATE_PER_SQUARE });
          }
          else
          {
            Span<Half[]> flatValuesStateSpan = flatValuesState.Span;

            // Reformat the state data into a 1D array
            // TODO: improve efficiency
            Half[] states1D = new Half[numPositionsInBatchSentToExecutor * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE];
            for (int i = 0; i < numPositionsUsed; i++)
            {
              if (flatValuesStateSpan[i] == null
               || (shouldUseStateForPos != null && !shouldUseStateForPos(i)))
              {
                // No state available, pass all zeroes.
                // Noting to do since array was created just above and is already zeroed.
              }
              else if (flatValuesStateSpan[i].Length != 64 * NNEvaluator.SIZE_STATE_PER_SQUARE)
              {
                throw new Exception("State input size mismatch.");
              }
              else
              {
                Array.Copy(flatValuesStateSpan[i], 0, states1D, i * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE, 64 * NNEvaluator.SIZE_STATE_PER_SQUARE);
              }
            }
            inputs[1] = (states1D, new int[] { states1D.Length });
          }
        }

#if NOT
        if (flatValuesSecondary.Length > 0)
        {
          Span<float> flatValuesSecondaryS = flatValuesSecondary.Span;
          for (int i = 0; i < flatValuesSecondary.Length; i++) flatValuesSecondaryS[i] /= DIVISOR;
          inputs[1] = (new Memory<float>(flatValuesSecondaryS.ToArray()), new int[] { numPositionsUsed, TPG_MAX_MOVES, TPG_BYTES_PER_MOVE_RECORD });
        }
#endif


        eval = executor.Run(inputs, numPositionsInBatchSentToExecutor, Precision == NNEvaluatorPrecision.FP16);

        if (numPositionsInBatchSentToExecutor != numPositionsUsed)
        {
          // Resize all results to match the number of positions actually used.
          for (int i = 0; i < eval.Count; i++)
          {
            eval[i] = (eval[i].Item1, eval[i].Item2.Slice(0, numPositionsUsed * eval[i].Item2.Length / numPositionsInBatchSentToExecutor));
          }
        }
      }
      else
      {
        (Memory<Half> flatValuesPrimary, int[]) input = default;
        input.Item1 = flatValuesPrimary.Slice(0, numPositionsUsed * 112 * 8 * 8);
        input.Item2 = [numPositionsUsed, 112, 8, 8];
        eval = executor.Run([input], numPositionsUsed, Precision == NNEvaluatorPrecision.FP16);
      }

      bool hasMLH = FindIndexExact("mlh") != -1;
      bool hasUNC = FindIndexExact("unc") != -1;
      bool hasUNC_POLICY = FindIndexExact("uncertainty_policy") != -1;

      int FindIndexExact(string name)
      {
        for (int i = 0; i < eval.Count; i++)
        {
          if (eval[i].Item1 == name)
          {
            return i;
          }
        }
        return -1;
      }

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
                 ONNXFileName.ToLower().Contains("policy_optimistic") && !eval[i].Item1.ToLower().Contains("optimistic")
              || ONNXFileName.ToLower().Contains("policy_vanilla") && !eval[i].Item1.ToLower().Contains("vanilla")
                 )
              {
                continue;
              }
            }
            if (expectedPerPosition == 3)
            {
              if (
                 ONNXFileName.ToLower().Contains("value_winner") && !eval[i].Item1.ToLower().Contains("value/winner")
              || ONNXFileName.ToLower().Contains("value_q") && !eval[i].Item1.ToLower().Contains("value/q")
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


      int INDEX_POLICIES = FindIndex(1858);// FIX NetType == NetTypeEnum.Ceres ? 1858 : 96);
      int INDEX_WDL = FindIndex(3);
      int INDEX_WDL2 = FindIndex(3, INDEX_WDL, "value2", true);
      int INDEX_MLH = FindIndex(1, optional: true);
      int INDEX_UNC = hasUNC ? FindIndex(1, INDEX_MLH, optional: true) : -1;
      int INDEX_UNC_POLICY = hasUNC_POLICY ? FindIndex(1, INDEX_MLH, "uncertainty_policy", true) : -1;
      int INDEX_ACTION = FindIndex(1858 * 3, -1, "action", true); // TODO: cleanup the output names to be better named
      int INDEX_STATE = FindIndex(256, -1, "state", true);

      Memory<Float16> mlh = hasMLH ? eval[INDEX_MLH].Item2 : null;
      Memory<Float16> uncertantiesV = hasUNC ? eval[INDEX_UNC].Item2 : null;
      Memory<Float16> uncertantiesP = hasUNC_POLICY ? eval[INDEX_UNC_POLICY].Item2 : null;
      Memory<Float16> policiesLogistics = eval[INDEX_POLICIES].Item2;

      Memory<Float16> actionLogits = INDEX_ACTION != -1 ? eval[INDEX_ACTION].Item2 : default;
      Memory<Float16> values = eval[INDEX_WDL].Item2;
      Debug.Assert(values.Length == (isWDL ? 3 : 1) * numPositionsUsed);

      Memory<Float16> values2 = INDEX_WDL2 == -1 ? default : eval[INDEX_WDL2].Item2;
      float[][] value_fc_activations = null;// eval.Length < 3 ? null : eval[2];

      Memory<Float16> extraStats0 = eval.Count > 5 ? eval[5].Item2 : default;
      Memory<Float16> extraStats1 = eval.Count > 6 ? eval[6].Item2 : default;

      // TODO: This is just a fake, fill it in someday
      Memory<Float16> priorState = hasState ? eval[INDEX_STATE].Item2 : default; //  new Float16[numPositionsUsed * 64 * 4]
      ONNXRuntimeExecutorResultBatch result = new(isWDL, values, values2, policiesLogistics, mlh,
                                                   uncertantiesV, uncertantiesP,
                                                   extraStats0, extraStats1, value_fc_activations,
                                                   actionLogits, priorState,
                                                   numPositionsUsed, RetainRawOutputs ? eval : null);
      return result;
    }


    public void EndProfiling() => executor.EndProfiling();


    public void Dispose()
    {
      executor.Dispose();
    }
  }
}
