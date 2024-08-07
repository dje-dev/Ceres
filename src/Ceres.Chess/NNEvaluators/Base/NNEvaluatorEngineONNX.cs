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
using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;

using Microsoft.ML.OnnxRuntime;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.DataType;
using Ceres.Base.Misc.ONNX;

using Ceres.Chess.LC0NetInference;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using System.Threading;

#endregion

namespace Chess.Ceres.NNEvaluators
{
  /// <summary>
  /// NNEvaluator subclass which reads network definitions from ONNX file
  /// via the ONNX Runtime (using ONNXRuntimeExecutor).
  /// </summary>
  public class NNEvaluatorEngineONNX : NNEvaluator
  {
    // TODO: When TPGRecord class is moved to Ceres project, instead reference TPGRecord.MAX_MOVES
    public const int MAX_MOVES = 92; 

    /// <summary>
    /// Name of file containing ONNX network definition.
    /// </summary>
    public readonly string ONNXFileName;

    /// <summary>
    /// Batch size to be used with this evaluator.
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Type of ONNX network.
    /// </summary>
    public readonly ONNXRuntimeExecutor.NetTypeEnum Type;


    /// <summary>
    /// Precision of network.
    /// </summary>
    public readonly NNEvaluatorPrecision Precision;

    /// <summary>
    /// Type of hardware device.
    /// </summary>
    public readonly NNDeviceType DeviceType;

    /// <summary>
    /// Executor object to run ONNX network evaluation.
    /// </summary>
    public readonly ONNXRuntimeExecutor Executor;

    /// <summary>
    /// Returns number of inputs specified by the ONNX file metadata.
    /// </summary>
    public int NumInputs => Executor.executor.NumInputs;

    /// <summary>
    /// Types of input(s) required by the evaluator.
    /// </summary>
    public override InputTypes InputsRequired => InputTypes.Positions | InputTypes.Boards | InputTypes.Moves | (HasState ? InputTypes.State : 0);


    /// <summary>
    /// If the network contains a WDL (win/draw/loss) style value head.
    /// </summary>
    public override bool IsWDL => isWDL;

    /// <summary>
    /// If the network contains a MLH (moves left head).
    /// </summary>
    public override bool HasM => hasM;

    /// <summary>
    /// If the network contains an uncertanity of V head.
    /// </summary>
    public override bool HasUncertaintyV => hasUncertaintyV;

    /// <summary>
    /// If the network contains an uncertanity of policy head.
    /// </summary>
    public override bool HasUncertaintyP => hasUncertaintyP;

    /// <summary>
    /// If action head is present in the network.
    /// </summary>
    public override bool HasAction => hasAction;

    /// <summary>
    /// If the evaluator has an secondary value head.
    /// </summary>
    public override bool HasValueSecondary => hasValueSecondary;


    readonly bool isWDL;
    readonly bool hasValueSecondary;
    readonly bool hasM;
    readonly bool hasUncertaintyV;
    readonly bool hasUncertaintyP;
    readonly bool hasAction;

    /// <summary>
    /// Name of policy output slot.
    /// </summary>
    public readonly string OutputPolicy;

    /// <summary>
    /// Name of value output slot (if non-WDL).
    /// </summary>
    public readonly string OutputValue;

    /// <summary>
    /// Name of WDL output slot (if WDL).
    /// </summary>
    public readonly string OutputWDL;

    /// <summary>
    /// Name of MLH output slot.
    /// </summary>
    public readonly string OutputMLH;

    /// <summary>
    /// If the output of the value head are logistic values (otherwise straight probabilities).
    /// </summary>
    public readonly bool ValueHeadLogistic;

    /// <summary>
    /// If the 50 move plane should be scaled down by 99.
    /// </summary>
    public readonly bool Scale50MoveCounter;

    /// <summary>
    /// If move history should be sent to ONNX network.
    /// </summary>
    public readonly bool UseHistory;
   
    /// <summary>
    /// Miscellaneous information about the evaluator.
    /// </summary>
    public override EvaluatorInfo Info => ONNXFileName == null ? null : new EvaluatorInfo(new ONNXNet(ONNXFileName).NumParams);



    public const int TPG_MODE_TOTAL_BYTES_ASSUMED  = 4060 + 782; // see DoEvaluateIntoBuffers


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="engineID"></param>
    /// <param name="onnxModelFileName"></param>
    /// <param name="onnxModelBytes"></param>
    /// <param name="deviceType"></param>
    /// <param name="gpuID"></param>
    /// <param name="useTRT"></param>
    /// <param name="type"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="precision"></param>
    /// <param name="isWDL"></param>
    /// <param name="hasM"></param>
    /// <param name="hasUncertaintyV"></param>
    /// <param name="hasUncertaintyP"></param>
    /// <param name="hasAction"></param>
    /// <param name="outputValue"></param>
    /// <param name="outputWDL"></param>
    /// <param name="outputPolicy"></param>
    /// <param name="outputMLH"></param>
    /// <param name="valueHeadLogistic"></param>
    /// <param name="scale50MoveCounter"></param>
    /// <param name="enableProfiling"></param>
    /// <param name="useHistory"></param>
    /// <param name="options"></param>
    /// <param name="hasValueSecondary"></param>
    /// <param name="hasState"></param>
    public NNEvaluatorEngineONNX(string engineID, string onnxModelFileName, byte[] onnxModelBytes,
                                 NNDeviceType deviceType, int gpuID, bool useTRT,
                                 ONNXRuntimeExecutor.NetTypeEnum type, int maxBatchSize,
                                 NNEvaluatorPrecision precision,
                                 bool isWDL, bool hasM, bool hasUncertaintyV, bool hasUncertaintyP, bool hasAction,
                                 string outputValue, string outputWDL, string outputPolicy, string outputMLH,
                                 bool valueHeadLogistic, bool scale50MoveCounter,
                                 bool enableProfiling = false,
                                 bool useHistory = true, NNEvaluatorOptions options = null,
                                 bool hasValueSecondary = false,
                                 bool hasState = false)
    {
      EngineNetworkID = engineID;
      ONNXFileName = onnxModelFileName;
      BatchSize = maxBatchSize;
      Precision = precision;
      this.isWDL = isWDL;
      this.hasValueSecondary = hasValueSecondary;
      this.hasM = hasM;
      this.hasUncertaintyV = hasUncertaintyV;
      this.hasUncertaintyP = hasUncertaintyP;
      this.hasAction = hasAction;
      this.HasState = hasState;
      DeviceType = deviceType;
      OutputValue = outputValue;
      OutputWDL = outputWDL;
      OutputPolicy = outputPolicy;
      OutputMLH = outputMLH;
      ValueHeadLogistic = valueHeadLogistic;
      Scale50MoveCounter = scale50MoveCounter;
      UseHistory = useHistory;
      Options = options ?? new NNEvaluatorOptions();

      Console.WriteLine("Starting ONNX runtime against " + onnxModelFileName + " from " + onnxModelFileName + " with " + deviceType + " " + gpuID);

      string[] inputNames = type == ONNXRuntimeExecutor.NetTypeEnum.TPG
        ? ["squares", "prior_state.1"]
        : ["/input/planes"];

      Executor = new ONNXRuntimeExecutor(engineID, onnxModelFileName, onnxModelBytes, inputNames,
                                         maxBatchSize, type, precision, deviceType, gpuID, useTRT, enableProfiling);
    }



    public Action<NNEvaluatorOptions, IEncodedPositionBatchFlat, bool, Half[], short[]> ConverterToFlat = null;
    public Func<NNEvaluatorOptions, object, byte[], int> ConverterToFlatFromTPG = null;

    [ThreadStatic] static byte[] inputsPrimaryNative;
    [ThreadStatic] static byte[] inputsSecondaryNative;
    [ThreadStatic] static Half[] inputsPrimaryNativeF;
    [ThreadStatic] static Half[] inputsSecondaryNativeF;


    /// <summary>
    /// Optional worker method which evaluates batch of positions which are already converted into native format needed by evaluator.
    /// </summary>
    /// <param name="positionsNativeInput"></param>
    /// <param name="usesSecondaryInputs"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public override IPositionEvaluationBatch DoEvaluateNativeIntoBuffers(object positionsNativeInput, bool usesSecondaryInputs, 
                                                                         int numPositions, Func<int, int, bool> posMoveIsLegal,
                                                                         bool retrieveSupplementalResults = false)
    {
      if (HasState)
      {
        throw new NotImplementedException("State not supported");
      }
      if (usesSecondaryInputs)
      {
        throw new NotImplementedException("Secondary inputs not supported");
      }
      Debug.Assert(!retrieveSupplementalResults);

      if (Executor.NetType != ONNXRuntimeExecutor.NetTypeEnum.TPG)
      {
        throw new Exception("DoEvaluateNativeIntoBuffers only supported for TPG net type.");
      }

      if (ConverterToFlatFromTPG == null)
      {
        throw new Exception("ConverterToFlatFromTPG must be provided");
      }

      if (inputsPrimaryNative == null)
      {
        // TO DO: Make smarter, check numPositions argument to smart size if possible (instead we make some guesses here on size, assume batch size 
        const int INPUT_SIZE_FLOATS = 2048 * 100 * 128; 
        inputsPrimaryNative = new byte[INPUT_SIZE_FLOATS];
        inputsSecondaryNative = new byte[INPUT_SIZE_FLOATS];
        inputsPrimaryNativeF = new Half[INPUT_SIZE_FLOATS];
      }

      int numConverted = ConverterToFlatFromTPG(Options, positionsNativeInput, inputsPrimaryNative);

      // Convert bytes to float      
      for (int i=0;i<numConverted;i++)
      {
        // TODO: Consider improving performance (though this may not be an often used mode of executing TPG files).
        //       To speed up, consider vectorize this, or call a built-in function, or partly unroll loop.
        inputsPrimaryNativeF[i] = inputsPrimaryNative[i];
        inputsSecondaryNativeF[i] = inputsSecondaryNative[i];
      }
      const float TPG_DIVISOR = ByteScaled.SCALING_FACTOR; // TODO: receive this in constructor instead. Should refer to TPGSquareRecord.SQUARE_BYTES_DIVISOR.
      PositionEvaluationBatch ret = DoEvaluateBatch(default, inputsPrimaryNativeF, null, //usesSecondaryInputs ? inputsSecondaryNativeF : null, 
                                                    numPositions, retrieveSupplementalResults, posMoveIsLegal,
                                                    TPG_DIVISOR);
      return ret;
    }


    /// <summary>
    /// Overrides worker method to evaluate a specified batch into internal buffers.
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat batch, bool retrieveSupplementalResults = false)
    {
      if (Executor.NetType == ONNXRuntimeExecutor.NetTypeEnum.TPG)
      {
        if (ConverterToFlat == null)
        {
          throw new Exception("ConverterToFlat must be provided");
        }

        int inputSizeAttention = batch.NumPos * 64 * ONNXRuntimeExecutor.TPG_BYTES_PER_SQUARE_RECORD;
        Half[] flatValuesAttention = ArrayPool<Half>.Shared.Rent(inputSizeAttention);
        Memory<Half> flatValuesAttentionM = flatValuesAttention.AsMemory().Slice(0, inputSizeAttention);

        short[] legalMoveIndices = null; // not needed, batch already contains moves
        ConverterToFlat(Options, batch, UseHistory, flatValuesAttention, legalMoveIndices);

        PositionEvaluationBatch ret = DoEvaluateBatch(batch, flatValuesAttentionM, batch.States, batch.NumPos, 
                                                      retrieveSupplementalResults, null, 1);
        Debug.Assert(!retrieveSupplementalResults);
        return ret;
      }
      else
      {
        int bufferLength = EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES * batch.NumPos * 64;
        Half[] flatValues = ArrayPool<Half>.Shared.Rent(bufferLength);

        batch.ValuesFlatFromPlanes(flatValues, false, Scale50MoveCounter);
        PositionEvaluationBatch ret = DoEvaluateBatch(batch, flatValues, null, batch.NumPos, retrieveSupplementalResults, null, 1);

        ArrayPool<Half>.Shared.Return(flatValues);
        return ret;
      }
    }


    /// <summary>
    /// If this evaluator produces the same output as another specified evaluator.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <returns></returns>
    public override bool IsEquivalentTo(NNEvaluator evaluator)
    {      
      return evaluator is NNEvaluatorEngineONNX
          && ((NNEvaluatorEngineONNX)evaluator).EngineNetworkID == EngineNetworkID;
    }


    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => BatchSize;


    #region Internals
    static bool haveWarned = false;
    static bool warned = false;

    /// <summary>
    /// Internal worker method to 
    /// </summary>
    /// <param name="flatValuesPrimary"></param>
    /// <param name="numPos"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    PositionEvaluationBatch DoEvaluateBatch(IEncodedPositionBatchFlat batch, 
                                            Memory<Half> flatValuesPrimary, Memory<Half[]> flatValuesState,
                                            int numPos, bool retrieveSupplementalResults, 
                                            Func<int,int, bool> posMoveIsLegal,
                                            float tpgDivisor)
    {
      if (retrieveSupplementalResults)
      {
        throw new Exception("retrieveSupplementalResults not supported");
      }

      ONNXRuntimeExecutorResultBatch result;
      TimingStats stats = new TimingStats();
//      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        lock (Executor)
        {
          result = Executor.Execute(IsWDL, HasState, flatValuesPrimary, flatValuesState, numPos,
                                    alreadyConvertedToLZ0: true, tpgDivisor:tpgDivisor);

          // Apply move masking
          if (posMoveIsLegal != null) 
          {
            throw new NotImplementedException(); // currently this is handled by the PositionEvaluationBatch constructor below intead
          }
        }
      }

      Half[][] states2D = null;
      if (HasState || Executor.executor.NumInputs > 1)
      {
        states2D = new Half[numPos][];
        if (result.PriorState.IsEmpty || !HasState)
        {
          // dummy values
          for (int i = 0; i < numPos; i++)
          {
            states2D[i] = new Half[64 * 4];
          }
        }
        else
        {
          Span<Half> statesSpan = MemoryMarshal.Cast<Float16, Half>(result.PriorState.Span);
          const int SIZE_STATE_PER_SQUARE = 4;
          // TODO: improve efficiency
          for (int i = 0; i < numPos; i++)
          {
            states2D[i] = new Half[64 * SIZE_STATE_PER_SQUARE];

            statesSpan.Slice(i * 64 * SIZE_STATE_PER_SQUARE, 64 * SIZE_STATE_PER_SQUARE)
                      .CopyTo(states2D[i]);
          }
        }
      }

      // NOTE: inefficient, above we convert from [] (flat) to [][] and here we convert back to []
      PositionEvaluationBatch ret =  new (IsWDL, HasM, HasUncertaintyV, HasUncertaintyP, 
                                         HasAction, HasValueSecondary, HasState, numPos,
                                         MemoryMarshal.Cast<Float16, FP16>(result.ValuesRaw.Span),
                                         MemoryMarshal.Cast<Float16, FP16>(result.Values2Raw.Span),
                                         result.PolicyVectors,//*/result.PolicyFlat, 
                                         result.ActionLogits,
                                         MemoryMarshal.Cast<Float16, FP16>(result.MLH.Span),
                                         MemoryMarshal.Cast<Float16, FP16>(result.UncertaintyV.Span),
                                         MemoryMarshal.Cast <Float16, FP16 >(result.UncertaintyP.Span).ToArray(), // TODO: eliminate array conversion
                                         MemoryMarshal.Cast<Float16, FP16>(result.ExtraStats0.Span),
                                         MemoryMarshal.Cast<Float16, FP16>(result.ExtraStats1.Span),
                                         states2D, // new Memory<Half[]>(states),
                                         default,
                                         Options.FractionValueHead2,
                                         Options.ValueHead1Temperature, Options.ValueHead2Temperature, 
                                         Options.Value1UncertaintyTemperatureScalingFactor, Options.Value2UncertaintyTemperatureScalingFactor,
                                         ValueHeadLogistic, PositionEvaluationBatch.PolicyType.LogProbabilities, false, 
                                         batch,
                                         Options.PolicyTemperature, Options.PolicyUncertaintyTemperatureScalingFactor,
                                         stats);

//#if NOT
// ** Experimental test code, triggered by having FractionValueFromValue2 >  1
      bool ADJUST = false && Options.FractionValueHead2 > 1 && !result.ExtraStats0.IsEmpty; // ***** TEMPORARY ******
      // flat: 0.8, 0.5
      const float THRESHOLD = 0.75f;
      const float BASE_COEFF = 0.4f;
      const float DRAW_THRESHOLD = 0.20f;
      if (ADJUST)
      {
        throw new NotImplementedException();
        Span<Float16> spanExtraStats0 = result.ExtraStats0.Span;
        Span<Float16> spanExtraStats1 = result.ExtraStats1.Span;

        for (int i = 0; i < numPos; i++)
        {
          (float w, float l) = ((float)ret.W2.Span[i], (float)ret.L2.Span[i]);

          float v = w - l;
          if (v > THRESHOLD)
          {
            float qDn = (float)spanExtraStats1[i] - (float)spanExtraStats0[i];
            float adjustedV = v  - BASE_COEFF * qDn;
            float movedV = v - adjustedV;
            w -= movedV / 2;
            // possibly we could adjust d, but doesn't really matter
            l += movedV / 2;
          }
          else if (v < -THRESHOLD) 
          {
            float qUp = (float)spanExtraStats0[i] - (float)spanExtraStats1[i];
            float adjustedV = v + BASE_COEFF * qUp;
            float movedV = adjustedV - v;
            w += movedV / 2;
            // possibly we could adjust d, but doesn't really matter
            l -= movedV / 2;
          }
          else if (true && Math.Abs(v) < DRAW_THRESHOLD)
          {
            const float COEFF_DRAW = BASE_COEFF * 3;
            float coeffAdj = COEFF_DRAW * (DRAW_THRESHOLD - MathF.Abs(v));
            float qVol = ((float)spanExtraStats0[i] + (float)spanExtraStats1[i]) / 2;
            float adjustedV = MathF.Sign(v) * (MathF.Abs(v) * (1 + COEFF_DRAW * qVol));
            float movedV = adjustedV - v;
            w += movedV / 2;
            l -= movedV / 2;
            //Console.WriteLine(v + " --> " + (w-l) + "  " + qVol);
          }

          (ret.W2.Span[i], ret.L2.Span[i]) = ((FP16)w, (FP16)l);
          (ret.W.Span[i], ret.L.Span[i]) = ((FP16)w, (FP16)l);
        }

      }
//#endif
      return ret;
    }

#endregion


    void ConvertTPGPolicyToExpanded(IEncodedPositionBatchFlat batch, ONNXRuntimeExecutorResultBatch result)
    {
      throw new NotImplementedException();
#if NOT
      Span<MGMoveList> allMoves = batch.Moves.Span;
      for (int i=0; i<batch.NumPos;i++)
      {
        // TODO: Very inefficient - create many arrays
        float[] policyVectorSource = result.PolicyVectors[i];
        float[] policyVectorTarget = new float[1858];

        MGMoveList moves = allMoves[i];
        for (int m=0; m<moves.NumMovesUsed;m++)
        {
          EncodedMove move = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(moves.MovesArray[m]);
         
          int index = move.IndexNeuralNet;
          policyVectorTarget[index] = policyVectorSource[m];
        }

        // Rewrite with expanded policy vector just created
        result.PolicyVectors[i] = policyVectorTarget;
      }
#endif
    }


    public void EndProfiling()
    {
      Executor.EndProfiling();
    }


    /// <summary>
    /// Returns string description of this evaluator.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return "<NNEvaluatorEngineONNX " + ONNXFileName + ">";
    }


    protected override void DoShutdown()
    {
      Executor.Dispose();
    }

  }
}
