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

using Ceres.Chess.LC0NetInference;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.LC0.Batches;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Base.DataType;
using Ceres.Base.Misc.ONNX;

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
    /// Types of input(s) required by the evaluator.
    /// </summary>
    public override InputTypes InputsRequired => InputTypes.Positions | InputTypes.Boards | InputTypes.Moves;


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
    /// If MoveRecord should be initialized.
    /// </summary>
    public readonly bool MovesEnabled;

    /// <summary>
    /// If move history should be sent to ONNX network.
    /// </summary>
    public readonly bool UseHistory;

    /// <summary>
    /// Optional evaluator-specific options object.
    /// </summary>
    public readonly object Options;

    /// <summary>
    /// Temperature value to use for value head.
    /// </summary>
    public readonly float TemperatureValue1;

    /// <summary>
    /// Temperature value to use for value secondary head.
    /// </summary>
    public readonly float TemperatureValue2;

    /// <summary>
    /// Fraction of value from value 2 to be blended into primary value.
    /// </summary>
    public readonly float FractionValueFromValue2 = 0;

    
    /// <summary>
    /// Miscellaneous information about the evaluator.
    /// </summary>
    public override EvaluatorInfo Info => ONNXFileName == null ? null : new EvaluatorInfo(new ONNXNet(ONNXFileName).NumParams);


    #region Statics

    // TODO: clean up this lookaside buffering
    static string lastONNXFileName;
    static long lastONNXBytesHash;
    static int lastBatchSize;
    static NNDeviceType lastDeviceType;
    static bool lastIsWDL;
    static bool lastUseTRT;
    static NNEvaluatorPrecision lastPrecision;
    static ONNXRuntimeExecutor lastExecutor;
    static ONNXRuntimeExecutor.NetTypeEnum lastType;

    #endregion


    public const int TPG_MODE_TOTAL_BYTES_ASSUMED  = 4060 + 782; // see DoEvaluateIntoBuffers

    public NNEvaluatorEngineONNX(string engineID, string onnxModelFileName, byte[] onnxModelBytes,
                                 NNDeviceType deviceType, int gpuID, bool useTRT,
                                 ONNXRuntimeExecutor.NetTypeEnum type, int batchSize,
                                 NNEvaluatorPrecision precision,
                                 bool isWDL, bool hasM, bool hasUncertaintyV, bool hasAction,
                                 string outputValue, string outputWDL, string outputPolicy, string outputMLH,
                                 bool valueHeadLogistic, bool scale50MoveCounter,
                                 bool movesEnabled = false, bool enableProfiling = false,
                                 bool useHistory = true, object options = null,
                                 bool hasValueSecondary = false,
                                 float temperatureValue1 = 1, float temperatureValue2 = 1, float fractionValueFromValue2 = 0,
                                 bool hasState = false)
    {
      EngineType = type == ONNXRuntimeExecutor.NetTypeEnum.Ceres ? "ONNX_DJE" : "ONNX_LZ0";
      EngineNetworkID = engineID;
      ONNXFileName = onnxModelFileName;
      BatchSize = batchSize;
      Precision = precision;
      this.isWDL = isWDL;
      this.hasValueSecondary = hasValueSecondary;
      this.hasM = hasM;
      this.hasUncertaintyV = hasUncertaintyV;
      this.hasAction = hasAction;
      this.HasState = hasState;
      DeviceType = deviceType;
      OutputValue = outputValue;
      OutputWDL = outputWDL;
      OutputPolicy = outputPolicy;
      OutputMLH = outputMLH;
      ValueHeadLogistic = valueHeadLogistic;
      Scale50MoveCounter = scale50MoveCounter;
      MovesEnabled = movesEnabled;
      UseHistory = useHistory;
      Options = options;
      TemperatureValue1 = temperatureValue1;
      TemperatureValue2 = temperatureValue2;
      FractionValueFromValue2 = fractionValueFromValue2;

      bool filesMatch = lastONNXFileName != null && lastONNXFileName == onnxModelFileName;
      bool bytesMatch = onnxModelBytes != null && ArrayUtils.ByteArrayStableHash(onnxModelBytes) == lastONNXBytesHash;

      const bool TRY_REUSE = false; // TODO: remove this completely, it is unsafe (?)
      if (TRY_REUSE && (filesMatch || bytesMatch) && lastBatchSize == batchSize && precision == lastPrecision
        && lastIsWDL == isWDL && lastType == type && deviceType == lastDeviceType && lastUseTRT == useTRT)
      {
        Executor = lastExecutor;
      }
      else
      {
        Console.WriteLine("Starting ONNX runtime against " + onnxModelFileName + " from " + onnxModelFileName + " with " + deviceType + " " + gpuID);

        string[] inputNames = type == ONNXRuntimeExecutor.NetTypeEnum.TPG
          ? (hasState ? ["squares", "prior_state.1"] : ["squares"])
          : ["/input/planes"];

        Executor = new ONNXRuntimeExecutor(engineID, onnxModelFileName, onnxModelBytes, inputNames,
                                           batchSize, type, precision, deviceType, gpuID, useTRT, enableProfiling);
        lastONNXFileName = onnxModelFileName;
        lastONNXBytesHash = onnxModelBytes == null ? 0 : ArrayUtils.ByteArrayStableHash(onnxModelBytes);
        lastDeviceType = deviceType;
        lastBatchSize = batchSize;
        lastIsWDL = isWDL;
        lastType = type;
        lastPrecision = precision;
        lastUseTRT = useTRT;
        lastExecutor = Executor;
      }
    }



    public static Action<IEncodedPositionBatchFlat, bool, Half[], short[]> ConverterToFlat = null;
    public static Func<object, byte[], int> ConverterToFlatFromTPG = null;

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

      int numConverted = ConverterToFlatFromTPG(positionsNativeInput, inputsPrimaryNative);

      // Convert bytes to float      
      for (int i=0;i<numConverted;i++)
      {
        // TODO: Consider improving performance (though this may not be an often used mode of executing TPG files).
        //       To speed up, consider vectorize this, or call a built-in function, or partly unroll loop.
        inputsPrimaryNativeF[i] = inputsPrimaryNative[i];
        inputsSecondaryNativeF[i] = inputsSecondaryNative[i];
      }

      const float TPG_DIVISOR = 100f; // TODO: receive this in constructor instead. Should refer to TPGSquareRecord.SQUARE_BYTES_DIVISOR.
      PositionEvaluationBatch ret = DoEvaluateBatch(default, inputsPrimaryNativeF, usesSecondaryInputs ? inputsSecondaryNativeF : null, 
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

        short[] legalMoveIndices = new short[batch.NumPos * MAX_MOVES];
        ConverterToFlat(batch, UseHistory, flatValuesAttention, legalMoveIndices);
#if NOT
        bool xPosMoveIsLegal(int posNum, int nnIndexNum)
        {
          if (batch.Moves.IsEmpty)
          {
            return true; // unable to disqualify
          }
          else
          {
            bool shouldFlip = batch.Positions.Span[posNum].SideToMove == SideType.Black;
            EncodedMove em = EncodedMove.FromNeuralNetIndex(nnIndexNum);
            
            ConverterMGMoveEncodedMove.FromTo mm = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMoveFromTo(em, shouldFlip);
            foreach (MGMove legalMove in batch.Moves.Span[posNum]) // TO DO: improve efficiency
            {
              if (legalMove.FromSquare.SquareIndexStartH1 == mm.From 
               && legalMove.ToSquare.SquareIndexStartH1 == mm.To)
              {
                return true;
              }
            }
            return false;
          }         
        }
        bool PosMoveIsLegal(int posNum, int nnIndexNum)
        {
          return true;

          // TODO: The code below doesn't quite work, fix someday
          //       though masking may be unnecessary.
          var ret = xPosMoveIsLegal(posNum, nnIndexNum);
          //Console.WriteLine(ret);
          return ret;
        }
#endif

        Func<int, int, bool> posMoveIsLegal = null; // PosMoveIsLegal
        PositionEvaluationBatch ret = DoEvaluateBatch(batch, flatValuesAttentionM, null, batch.NumPos, 
                                                      retrieveSupplementalResults, posMoveIsLegal, tpgDivisor:1);
        Debug.Assert(!retrieveSupplementalResults);
        return ret;
      }
      else
      {
        Func<int, int, bool> posMoveIsLegal = null; // PosMoveIsLegal
        bool PosMoveIsLegal(int posNum, int nnIndexNum)
        {
          return true; // ** TO DO? Maybe unnecessary?
          //return tpgRecordBuffer[(numBatchesDone * SUBBATCH_SIZE) + posNum].Policy[nnIndexNum] > 0;
        }

        int bufferLength = 112 * batch.NumPos * 64;
        Half[] flatValues = ArrayPool<Half>.Shared.Rent(bufferLength);

        batch.ValuesFlatFromPlanes(flatValues, false, Scale50MoveCounter);
        PositionEvaluationBatch ret = DoEvaluateBatch(batch, flatValues, null, batch.NumPos, retrieveSupplementalResults, posMoveIsLegal, tpgDivisor:1);

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
                                            Memory<Half> flatValuesPrimary, Memory<Half> flatValuesSecondary,
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
          result = Executor.Execute(IsWDL, HasState, flatValuesPrimary, flatValuesSecondary, numPos, alreadyConvertedToLZ0: true, tpgDivisor:tpgDivisor);

          // Apply move masking
          if (posMoveIsLegal != null) 
          {
            for (int i = 0; i < numPos; i++)
            {
              for (int j = 0; j < 1858; j++)
              {
                if (!posMoveIsLegal(i,j))
                {
                  throw new NotImplementedException(); // remediate next line
                  //result.PolicyVectors[i * 1858 + j] = -100f;
                }
              }
            }
          }
#if NO_LONGER_NEEDED_NOT_USING_96_FORMAT
          if (Executor.NetType == ONNXRuntimeExecutor.NetTypeEnum.TPG)
          {
            ConvertTPGPolicyToExpanded(batch, result);
          }
#endif
        }
      }

      if (HasState)
      {
        if (!warned)
        {
          Console.WriteLine("WARNING: NNEvaluatorEngineONNX needs remediation for state head");
          warned = true;
        }
      } 

#if DONE_BELOW_IN_NEXT_LINE
      // Set probability of illegal moves to 0.
      HashSet<int> legalIndices = new HashSet<int>(96);
      for (int pos=0; pos<numPos;pos++)
      {
        legalIndices.Clear();
        for (int i = 0; i <batch.Moves[pos].NumMovesUsed;i++)
        {
          EncodedMove encodedMove  = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(batch.Moves[pos].MovesArray[i]);
          legalIndices.Add(encodedMove.IndexNeuralNet);
        }

        for (int i=0;i<1858;i++)
        {
          if (!legalIndices.Contains(i))
          {
            result.PolicyVectors[pos][i] = -1E10f;
          }
        }
      }
#endif
#if NOT
      if (result.ActionLogisticVectors != null)
      {
        for (int i = 0; i < numPos; i++)
        {
          int offset = (1858 * 3 * i) + policyIndex * 3;
          Span<FP16> actionsSpan = ActionProbabilities.Span;

          float wLogit = actionsSpan[offset];
          float dLogit = actionsSpan[offset + 1];
          float lLogit = actionsSpan[offset + 2];

          return (actionsSpan[offset], actionsSpan[offset + 1], actionsSpan[offset + 2]);

          float max = MathF.Max(wLogit, MathF.Max(dLogit, lLogit));

          float w = MathF.Exp(wLogit - max);
          float d = MathF.Exp(dLogit - max);
          float l = MathF.Exp(lLogit - max);

          float mult = 1.0f / (w + d + l);

          return (w * mult, d * mult, l * mult);
        }
      }
#endif
      CompressedActionVector[] actions = null;
      if (!result.ActionLogisticVectors.IsEmpty && !haveWarned)
      {
        // TODO: initialize actions from result.ActionLogisticVectors
        Console.WriteLine("NNEvaluatorEngineONNX needs minor remediation to pass along converted ActionLogisticVectors below");
        actions = new CompressedActionVector[numPos];
        haveWarned = true;
      }

      Half[][] states = null;
      if (HasState)
      {
        // TODO: this is just a dummy, fill in someday
        states = new Half[numPos][];
        for (int i=0;i<numPos;i++)
        {
          states[i] = new Half[64 * 4];
        } 
      }


#if NOT
      Span<FP16> valueEvals, Span<FP16> valueEvals2, 
      Memory<FP16> policyProbs, Memory<CompressedActionVector> actionLogits,
      FP16[] m, FP16[] uncertaintyV, 
#endif

#if DEBUG
      if (ValueHeadLogistic)
      {
        float sum = IsWDL ? ((float)result.ValuesRaw.Span[0] + (float)result.ValuesRaw.Span[1] + (float)result.ValuesRaw.Span[2])
                          : (float)result.ValuesRaw.Span[0];
        if (Math.Abs(1.0 - sum) < 0.001f)
        {
          throw new Exception("Values expected logits, but look suspisciously like probabilities.");
        }

      }
      else
      {
        if ((float)result.ValuesRaw.Span[0] < 0)
        {
          throw new Exception("Negative probability found in non-logistic mode.");
        }
      }
#endif

      // NOTE: inefficient, above we convert from [] (flat) to [][] and here we convert back to []
      PositionEvaluationBatch ret =  new (IsWDL, HasM, HasUncertaintyV, HasAction, HasValueSecondary, HasState, numPos,
                                         MemoryMarshal.Cast<Float16, FP16>(result.ValuesRaw.Span),
                                         MemoryMarshal.Cast<Float16, FP16>(result.Values2Raw.Span),
                                         result.PolicyVectors,//*/result.PolicyFlat, 
                                         actions,
                                         MemoryMarshal.Cast<Float16, FP16>(result.MLH.Span),
                                         MemoryMarshal.Cast<Float16, FP16>(result.UncertaintyV.Span),
                                         MemoryMarshal.Cast<Float16, FP16>(result.ExtraStats0.Span),
                                         MemoryMarshal.Cast<Float16, FP16>(result.ExtraStats1.Span),
                                         new Memory<Half[]>(states), default,
                                         TemperatureValue1, TemperatureValue2, FractionValueFromValue2,
                                         ValueHeadLogistic, PositionEvaluationBatch.PolicyType.LogProbabilities, false, 
                                         batch, 
                                         stats);

//#if NOT
// ** Experimental test code, triggered by having FractionValueFromValue2 >  1
      bool ADJUST = FractionValueFromValue2 > 1 && !result.ExtraStats0.IsEmpty; // ***** TEMPORARY ******
      // flat: 0.8, 0.5
      const float THRESHOLD = 0.75f;
      float COEFF = 0.5f;
      if (ADJUST)
      {
        Span<Float16> spanExtraStats0 = result.ExtraStats0.Span;
        Span<Float16> spanExtraStats1 = result.ExtraStats1.Span;

        for (int i = 0; i < numPos; i++)
        {
          (float w, float l) = ((float)ret.W2.Span[i], (float)ret.L2.Span[i]);

          float v = w - l;
          if (v > THRESHOLD)
          {
            float qDn = (float)spanExtraStats1[i] - (float)spanExtraStats0[i];
            float adjustedV = v  - COEFF * qDn;
            float movedV = v - adjustedV;
            w -= movedV / 2;
            // possibly we could adjust d, but doesn't really matter
            l += movedV / 2;
          }
          else if (v < -THRESHOLD) 
          {
            float qUp = (float)spanExtraStats0[i] - (float)spanExtraStats1[i];
            float adjustedV = v + COEFF * qUp;
            float movedV = adjustedV - v;
            w += movedV / 2;
            // possibly we could adjust d, but doesn't really matter
            l -= movedV / 2;
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
