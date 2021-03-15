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

using Ceres.Chess.LC0NetInference;
using Ceres.Base;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.LC0.Batches;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;

#endregion

namespace Chess.Ceres.NNEvaluators
{
  /// <summary>
  /// NNEvaluator subclass which reads network definitions from ONNX file
  /// via the ONNX Runtime (using ONNXRuntimeExecutor).
  /// </summary>
  public class NNEvaluatorEngineONNX : NNEvaluator
  {
    /// <summary>
    /// Name of file containing ONNX network defition.
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
    /// Executor object to run ONNX network evaluation.
    /// </summary>
    public readonly ONNXRuntimeExecutor Executor;


    /// <summary>
    /// Types of input(s) required by the evaluator.
    /// </summary>
    public override InputTypes InputsRequired => InputTypes.Boards | InputTypes.Moves;


    public override bool IsWDL => isWDL;
    public override bool HasM => hasM;

    readonly bool isWDL;
    readonly bool hasM;

    #region Statics

    static string lastONNXFileName;
    static int lastBatchSize;
    static bool lastIsWDL;
    static ONNXRuntimeExecutor lastExecutor;
    static ONNXRuntimeExecutor.NetTypeEnum lastType;

    #endregion

    public NNEvaluatorEngineONNX(string engineID, string weightsFN, int gpuID, 
                                 ONNXRuntimeExecutor.NetTypeEnum type, int batchSize, bool isWDL, bool hasM)
    {
      if (batchSize > MAX_BATCH_SIZE) throw new ArgumentOutOfRangeException(nameof(batchSize), $"exceeds maximum of {MAX_BATCH_SIZE}");

      EngineType = type == ONNXRuntimeExecutor.NetTypeEnum.Ceres ? "ONNX_DJE" : "ONNX_LZ0";
      EngineNetworkID = engineID;
      ONNXFileName = weightsFN;
      BatchSize = batchSize;
      this.isWDL = isWDL;
      this.hasM = hasM;

      if (lastONNXFileName == weightsFN && lastBatchSize == batchSize
        && lastIsWDL == isWDL && lastType == type)
      {
        Executor = lastExecutor;
      }
      else
      {
        Console.WriteLine("Starting ONNX runtime against " + engineID + " from " + weightsFN + " with GPU " + gpuID);

        Executor = new ONNXRuntimeExecutor(weightsFN, batchSize, type, gpuID);
        lastONNXFileName = weightsFN;
        lastBatchSize = batchSize;
        lastIsWDL = isWDL;
        lastType = type;
        lastExecutor = Executor;
      }
    }


    /// <summary>
    /// Overrides worker method to evaluate a specified batch into internal buffers.
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public override IPositionEvaluationBatch EvaluateIntoBuffers(IEncodedPositionBatchFlat batch, bool retrieveSupplementalResults = false)
    {
      int bufferLength = 112 * batch.NumPos * 64;
      float[] flatValues = ArrayPool<float>.Shared.Rent(bufferLength);
        
      batch.ValuesFlatFromPlanes(flatValues);
      PositionEvaluationBatch ret = DoEvaluateBatch(flatValues, batch.NumPos, retrieveSupplementalResults);

      ArrayPool<float>.Shared.Return(flatValues);
      return ret;
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

    #region Internals

    /// <summary>
    /// Internal worker method to 
    /// </summary>
    /// <param name="flatValues"></param>
    /// <param name="numPos"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    PositionEvaluationBatch DoEvaluateBatch(float[] flatValues, int numPos, bool retrieveSupplementalResults)
    {
      if (retrieveSupplementalResults) throw new Exception("retrieveSupplementalResults not supported");

      ONNXRuntimeExecutorResultBatch result;
      TimingStats stats = new TimingStats();
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        lock (Executor)
        {
          result = Executor.Execute(IsWDL, flatValues, numPos, alreadyConvertedToLZ0: true);
        }
      }

      FP16[] mFP16 = null;
      if (HasM)
      {
        mFP16 = Array.ConvertAll<float, FP16>(result.MLH, m => (FP16)m);
      }
      // NOTE: inefficient, above we convert from [] (flat) to [][] and here we convert back to []
      return new PositionEvaluationBatch(IsWDL, HasM, numPos, result.ValuesRaw, result.PolicyFlat, mFP16, null,  true,
                                         PositionEvaluationBatch.PolicyType.LogProbabilities, false, stats);
    }

    #endregion

    protected override void DoShutdown()
    {
      Executor.Dispose();
    }

  }

}
