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
using System.Threading;

#endregion

namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  /// <summary>
  /// Manages execution of ONNX networks for inference of chess neural networks
  /// (Leela Chess Zero and Ceres).
  /// </summary>
  public class ONNXNetExecutor : IDisposable
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

    /// <summary>
    /// If an input with the name "squares_byte" exists
    /// indicating the network can accept TPG style data in pure byte format.
    /// </summary>
    public bool HasSquaresByteInput = false;


    /// <summary>
    /// Name of the LoRA adapter file (if any).
    /// </summary>
    public readonly string LoRAAdapterFileName;

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
                               bool retainRawOutputs,
                               string loraAdapterFileName = null)
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

      MinBatchSize = 1;
      LoRAAdapterFileName = loraAdapterFileName;

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

      // TODO: Clean up, this is a hack.
      // Look for the input with name with -I8 indicating
      // the network can directly accept byte inputs.
      HasSquaresByteInput = netType == NetTypeEnum.TPG && onnxFileName.ToUpper().Contains("-I8");

      int inputsNumBits = HasSquaresByteInput ? 8 : (Precision == NNEvaluatorPrecision.FP16 ? 16 : 32);
      executor = new ONNXExecutor(shortID, onnxFileName, onnxModelBytes, inputNames, nonBatchDimensions,
                                  inputsNumBits, precisionNumBits, true, deviceIndex, useTensorRT, MinBatchSize, maxBatchSize,
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
    public ONNXRuntimeExecutorResultBatch[] ExecuteTPGByteInputs(bool isWDL, bool hasState,
                                                                 Memory<byte> flatValuesPrimary,
                                                                 Memory<Half[]> flatValuesState,
                                                                 int numPositionsUsed,
                                                                 Predicate<int> shouldUseStateForPos = null)
    {
      Debug.Assert(NetType == NetTypeEnum.TPG);

      List<(string, Memory<Float16>)> eval;

      int NUM_INPUTS = executor.NumInputs;
      (Memory<byte> input, int[] shape)[] inputs = new (Memory<byte> input, int[] shape)[NUM_INPUTS];

      int TOTAL_LEN = numPositionsUsed * 64 * TPG_BYTES_PER_SQUARE_RECORD;
      Memory<byte> input0 = flatValuesPrimary.Slice(0, TOTAL_LEN);

      inputs[0] = (input0, [numPositionsUsed, 64, TPG_BYTES_PER_SQUARE_RECORD]);

      if (inputs.Length > 1)
      {
        // TODO: improve efficiency here
        if (flatValuesState.IsEmpty)
        {
          // No state available, pass all zeroes.
          inputs[1] = (new byte[numPositionsUsed * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE],
                     new int[] { numPositionsUsed, 64, NNEvaluator.SIZE_STATE_PER_SQUARE });
        }
        else
        {
          Span<Half[]> flatValuesStateSpan = flatValuesState.Span;

          // Reformat the state data into a 1D array
          // TODO: improve efficiency
          //            Half[] states1D = new Half[numPositionsInBatchSentToExecutor * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE];
          throw new Exception("Needs remeditation; the inputs are assumed all byte but for this case the second one should be Half");
          byte[] states1D = new byte[numPositionsUsed * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE];
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

      eval = executor.Run(HasSquaresByteInput ? ONNXExecutor.ONNXInputTypeEnum.Byte
                                              : ONNXExecutor.ONNXInputTypeEnum.Float16,
                          inputs, numPositionsUsed);

      if (executor.MultiNetNames != null)
      {
        if (executor.MultiNetWeights == null || executor.MultiNetWeights.Length != 2)
        {
          throw new Exception("Expected to find metadata for Ceres_multinet_weights with exactly two constituents.");
        }
        ONNXRuntimeExecutorResultBatch model1Outputs = ExtractNetOutputs("model1_", isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval);
        ONNXRuntimeExecutorResultBatch model2Outputs = ExtractNetOutputs("model2_", isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval);
        return [model1Outputs, model2Outputs];
      }

      return [ExtractNetOutputs(null, isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval)];
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
    public ONNXRuntimeExecutorResultBatch[] ExecuteForTPGBytesDirect(bool isWDL, bool hasState,
                                                                     Memory<byte> flatValuesPrimary,
                                                                     Memory<Half[]> flatValuesState,
                                                                     int numPositionsUsed,
                                                                     Predicate<int> shouldUseStateForPos = null)
    {
      List<(string, Memory<Float16>)> eval;
      Span<byte> flatValuesPrimarySpan = flatValuesPrimary.Span;

      Debug.Assert(NetType == NetTypeEnum.TPG);

      int NUM_INPUTS = executor.NumInputs;
      (Memory<byte> input, int[] shape)[] inputs = new (Memory<byte> input, int[] shape)[NUM_INPUTS];

      // TODO: this code duplicated below, refactor/clean up.
      int TOTAL_LEN = numPositionsUsed * 64 * TPG_BYTES_PER_SQUARE_RECORD;
      Memory<byte> input0 = flatValuesPrimary.Slice(0, TOTAL_LEN);

      inputs[0] = (input0, [numPositionsUsed, 64, TPG_BYTES_PER_SQUARE_RECORD]);

      if (inputs.Length > 1)
      {
        // TODO: improve efficiency here
        if (flatValuesState.IsEmpty)
        {
          // No state available, pass all zeroes.
          inputs[1] = (new byte[numPositionsUsed * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE],
                     new int[] { numPositionsUsed, 64, NNEvaluator.SIZE_STATE_PER_SQUARE });
        }
        else
        {
          throw new NotImplementedException("Remediation for switch to bytes");
#if NOT
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
#endif        
        }

#if NOT
        if (flatValuesSecondary.Length > 0)
        {
          Span<float> flatValuesSecondaryS = flatValuesSecondary.Span;
          for (int i = 0; i < flatValuesSecondary.Length; i++) flatValuesSecondaryS[i] /= DIVISOR;
          inputs[1] = (new Memory<float>(flatValuesSecondaryS.ToArray()), new int[] { numPositionsUsed, TPG_MAX_MOVES, TPG_BYTES_PER_MOVE_RECORD });
        }
#endif
      }

      (Memory<byte> input, int[] shape)[] inputsByte = new (Memory<byte> input, int[] shape)[NUM_INPUTS];
      inputsByte[0].input = new Memory<byte>(new byte[flatValuesPrimary.Length]);

      Span<byte> flatValuesPrimaryTarget = inputsByte[0].input.Span;
      for (int i = 0; i < flatValuesPrimarySpan.Length; i++)
      {
        //            flatValuesPrimaryTarget[i] = (byte)flatValuesPrimarySpan[i];
      }

      //          CopyFloatToByteAvx2(flatValuesPrimarySpan, flatValuesPrimaryTarget);  
      //          TensorPrimitives.ConvertSaturating<Half, byte>(flatValuesPrimarySpan, flatValuesPrimaryTarget);
      //          TensorPrimitives.ConvertChecked<Half, byte>(flatValuesPrimarySpan, flatValuesPrimaryTarget);  
      //          for (int i = 0; i < flatValuesPrimarySpan.Length; i++)
      //          {
      //            // TODO: improve efficiency
      //            flatValuesPrimaryTarget[i] = (byte)flatValuesPrimarySpan[i];
      //          }

      inputsByte[0].shape = inputs[0].shape;
      eval = executor.Run(ONNXExecutor.ONNXInputTypeEnum.Byte, inputsByte, numPositionsUsed);

      if (executor.MultiNetNames != null)
      {
        if (executor.MultiNetWeights == null || executor.MultiNetWeights.Length != 2)
        {
          throw new Exception("Expected to find metadata for Ceres_multinet_weights with exactly two constituents.");
        }
        ONNXRuntimeExecutorResultBatch model1Outputs = ExtractNetOutputs("model1_", isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval);
        ONNXRuntimeExecutorResultBatch model2Outputs = ExtractNetOutputs("model2_", isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval);
        return [model1Outputs, model2Outputs];
      }

      return [ExtractNetOutputs(null, isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval)];
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
    public ONNXRuntimeExecutorResultBatch[] Execute(bool isWDL, bool hasState,
                                                    Memory<Half> flatValuesPrimary,
                                                    Memory<Half[]> flatValuesState,
                                                    int numPositionsUsed,
                                                    bool debuggingDump = false, bool alreadyConvertedToLZ0 = false,
                                                    float tpgDivisor = 1,
                                                    Predicate<int> shouldUseStateForPos = null)
    {
      if (NetType == NetTypeEnum.TPG && !alreadyConvertedToLZ0)
      {
        throw new NotImplementedException();
      }

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
      Span<Half> flatValuesPrimarySpan = flatValuesPrimary.Span;

      if (NetType == NetTypeEnum.TPG)
      {
        int NUM_INPUTS = executor.NumInputs;
        (Memory<Half> input, int[] shape)[] inputs = new (Memory<Half> input, int[] shape)[NUM_INPUTS];
        if (tpgDivisor != 1.0f)
        {
          for (int i = 0; i < flatValuesPrimarySpan.Length; i++)
          {
            // TODO: improve efficiency (slow conversion/ non-SIMD division here)
            //       possibly use a modified version of TPGConvertersToFlat.CopyAndDivideSIMD
            flatValuesPrimarySpan[i] = (Half)((float)flatValuesPrimarySpan[i] / tpgDivisor);
          }
        }

        int TOTAL_LEN = numPositionsUsed * 64 * TPG_BYTES_PER_SQUARE_RECORD;
        Memory<Half> input0 = flatValuesPrimary.Slice(0, TOTAL_LEN);
        inputs[0] = (input0, [numPositionsUsed, 64, TPG_BYTES_PER_SQUARE_RECORD]);

        if (inputs.Length > 1)
        {
          // TODO: improve efficiency here
          if (flatValuesState.IsEmpty)
          {
            // No state available, pass all zeroes.
            inputs[1] = (new Half[numPositionsUsed * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE],
                       new int[] { numPositionsUsed, 64, NNEvaluator.SIZE_STATE_PER_SQUARE });
          }
          else
          {
            throw new Exception("Need to access predefined buffer in ONNXExecutor, not allocate here");
            Span<Half[]> flatValuesStateSpan = flatValuesState.Span;

            // Reformat the state data into a 1D array
            // TODO: improve efficiency
            Half[] states1D = new Half[numPositionsUsed * 64 * NNEvaluator.SIZE_STATE_PER_SQUARE];
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

        if (HasSquaresByteInput)
        {
          (Memory<byte> input, int[] shape)[] inputsByte = new (Memory<byte> input, int[] shape)[NUM_INPUTS];
          inputsByte[0].input = new Memory<byte>(new byte[flatValuesPrimary.Length]);

          Span<byte> flatValuesPrimaryTarget = inputsByte[0].input.Span;
          for (int i = 0; i < flatValuesPrimarySpan.Length; i++)
          {
            //            flatValuesPrimaryTarget[i] = (byte)flatValuesPrimarySpan[i];
          }

          //          CopyFloatToByteAvx2(flatValuesPrimarySpan, flatValuesPrimaryTarget);  
          //          TensorPrimitives.ConvertSaturating<Half, byte>(flatValuesPrimarySpan, flatValuesPrimaryTarget);
          //          TensorPrimitives.ConvertChecked<Half, byte>(flatValuesPrimarySpan, flatValuesPrimaryTarget);  
          //          for (int i = 0; i < flatValuesPrimarySpan.Length; i++)
          //          {
          //            // TODO: improve efficiency
          //            flatValuesPrimaryTarget[i] = (byte)flatValuesPrimarySpan[i];
          //          }

          inputsByte[0].shape = inputs[0].shape;
          eval = executor.Run(ONNXExecutor.ONNXInputTypeEnum.Byte, inputsByte, numPositionsUsed);
        }
        else
        {
          eval = executor.Run(Precision == NNEvaluatorPrecision.FP16 ? ONNXExecutor.ONNXInputTypeEnum.Float16
                                                                     : ONNXExecutor.ONNXInputTypeEnum.Float32,
                              inputs, numPositionsUsed);
        }
      }
      else
      {
        (Memory<Half> flatValuesPrimary, int[]) input = default;
        input.Item1 = flatValuesPrimary.Slice(0, numPositionsUsed * 112 * 8 * 8);
        input.Item2 = [numPositionsUsed, 112, 8, 8];
        eval = executor.Run(ONNXExecutor.ONNXInputTypeEnum.Float16, [input], numPositionsUsed);
      }


      if (executor.MultiNetNames != null)
      {
        if (executor.MultiNetWeights == null || executor.MultiNetWeights.Length != 2)
        {
          throw new Exception("Expected to find metadata for Ceres_multinet_weights with exactly two constituents.");
        }
        ONNXRuntimeExecutorResultBatch model1Outputs = ExtractNetOutputs("model1_", isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval);
        ONNXRuntimeExecutorResultBatch model2Outputs = ExtractNetOutputs("model2_", isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval);
        return [model1Outputs, model2Outputs];
      }


      return [ExtractNetOutputs(null, isWDL, hasState, RetainRawOutputs, numPositionsUsed, eval)];
    }


    private static ONNXRuntimeExecutorResultBatch ExtractNetOutputs(string namePrefixFilter,
                                                                    bool isWDL, bool hasState,
                                                                    bool retainRawOutputs, int numPositionsUsed,
                                                                    List<(string, Memory<Float16>)> eval)
    {
      bool hasMLH = FindIndexExact("mlh") != -1;
      bool hasUNC = FindIndexExact("unc") != -1;
      bool hasUNC_POLICY = FindIndexExact("uncertainty_policy") != -1;

      bool MatchesNamePrefix(int index) => namePrefixFilter == null || eval[index].Item1.StartsWith(namePrefixFilter);

      int FindIndexExact(string name)
      {
        for (int i = 0; i < eval.Count; i++)
        {
          string outputNameWithoutPrefix = namePrefixFilter == null
                                           ? eval[i].Item1
                                           : eval[i].Item1.Replace(namePrefixFilter, "");
          if (MatchesNamePrefix(i) && outputNameWithoutPrefix == name)
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
          if (MatchesNamePrefix(i) &&
            eval[i].Item2.Length == expectedLength
            && i != indexToIgnore
            && (mustContainString == null || eval[i].Item1.Contains(mustContainString)))
          {
            return i;
          }
        }

        return optional ? -1 : throw new Exception("No output found with expected length " + expectedPerPosition);
      }


      int INDEX_POLICIES = FindIndex(1858);
      int INDEX_WDL = FindIndex(3);
      int INDEX_WDL2 = FindIndex(3, INDEX_WDL, "value2", true);
      int INDEX_WDL3 = FindIndex(3, INDEX_WDL, "value3", true);

      const bool SUBSTITUTE_VALUE3_INTO_VALUE2_IF_FOUND = true;
      if (SUBSTITUTE_VALUE3_INTO_VALUE2_IF_FOUND && INDEX_WDL3 != -1)
      {
        // TODO: Consider adding a slot for value 3 instead?
        // Store value3 in 2
        INDEX_WDL = INDEX_WDL3;
      }

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
                                                   numPositionsUsed, retainRawOutputs ? eval : null);
      return result;
    }

    public void EndProfiling() => executor.EndProfiling();


    /// <summary>
    /// Performs any initialization to prepare evaluator for delay-free execution.
    /// </summary>
    public void Warmup()
    {
      this.executor.Warmup();
    }


    public void Dispose()
    {
      executor.Dispose();
    }
  }
}
