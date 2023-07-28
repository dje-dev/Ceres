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
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Collections.Concurrent;

using Ceres.Base.OperatingSystem;
using Ceres.Base.DataTypes;
using Ceres.Base.Threading;
using Ceres.Base.Benchmarking;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NetEvaluation.Batch;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNEvaluators.Internals;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.MoveGen;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNBackends.CUDA;

#endregion

namespace Ceres.Chess.NNEvaluators.CUDA
{
  /// <summary>
  /// NNEvaluator subclass which uses direct C# to CUDA to process LC0 networks
  /// (implementation based heavily on LC0 CUDA backends).
  /// </summary>
  public class NNEvaluatorCUDA : NNEvaluator
  {
    // TODO: Try to make this work. Currently 
    /// <summary>
    /// If the weights memory on GPU should be shared between two evaluators
    /// (created using the referenceEvaluator argument in the constructor)
    /// thereby reducing GPU memory consumption on the order of 25%.
    /// </summary>
    const bool TRY_SHARE_EVALUATORS = true;


    /// <summary>
    /// Index of GPU on which evaluation runs.
    /// </summary>
    public readonly int GPUID;


    /// <summary>
    /// Optional reference evaluator from which weights can be reused.
    /// </summary>
    public readonly NNEvaluatorCUDA ReferenceEvaluator;


    /// <summary>
    /// If Win/Draw/Loss head is present in the network.
    /// </summary>
    public override bool IsWDL => isWDL;


    /// <summary>
    /// If Moves-left head is present in the network.
    /// </summary>
    public override bool HasM => hasM;

    /// <summary>
    /// If Uncertainty of V  head is present in the network.
    /// </summary>
    public override bool HasUncertaintyV => false;

    /// <summary>
    /// If the network returns policy moves in the same order
    /// as the legal MGMoveList.
    /// </summary>
    public override bool PolicyReturnedSameOrderMoveList => true;


    readonly bool isWDL;
    readonly bool hasM;
    readonly int maxBatchSize;

    /// <summary>
    /// Types of input(s) required by the evaluator.
    /// </summary>
#if DEBUG
    public override InputTypes InputsRequired => InputTypes.Boards | InputTypes.Moves | InputTypes.Positions;
#else
    public override InputTypes InputsRequired => InputTypes.Boards | InputTypes.Moves;
#endif   

    CompressedPolicyVector[] policies;
    FP16[] l;
    FP16[] w;
    FP16[] m;

    /// <summary>
    /// Underlying neural network evaluator used.
    /// </summary>
    public readonly NNBackendLC0_CUDA Evaluator;

    static int numInstancesEverCreated = 0;

    int instanceNumber;

    #region Constructors 

    public NNEvaluatorCUDA(NNWeightsFileLC0 net, int gpuID,
                               int maxBatchSize = NNBackendLC0_CUDA.DEFAULT_MAX_BATCH_SIZE,
                               bool saveActivations = false,
                               NNEvaluatorPrecision precision = NNEvaluatorPrecision.FP16,
                               bool dumpTimings = false,
                               bool enableCUDAGraphs = true,
                               int graphBatchSizeDivisor = 1,
                               NNEvaluatorCUDA referenceEvaluator = null)
    {
      if (precision != NNEvaluatorPrecision.FP16)
      {
        throw new ArgumentException(nameof(precision), "Implementation restriction: only FP16 supported");
      }

      this.maxBatchSize = maxBatchSize;
      ReferenceEvaluator = referenceEvaluator;

      instanceNumber = numInstancesEverCreated++;

      if (!SoftwareManager.IsCUDAInstalled)
      {
        throw new Exception("GPU hardware with CUDA installation is required but not found.");
      }

      GPUID = gpuID;

      // Record net characteristics      
      isWDL = net.IsWDL;
      hasM = net.HasMovesLeft;

      // Allocate space for outputs
      policies = new CompressedPolicyVector[maxBatchSize];
      w = new FP16[maxBatchSize];
      l = isWDL ? new FP16[maxBatchSize] : null;
      m = isWDL ? new FP16[maxBatchSize] : null;

      Evaluator = new NNBackendLC0_CUDA(gpuID, net.Info.Net, saveActivations, maxBatchSize,
                                        dumpTimings, enableCUDAGraphs, graphBatchSizeDivisor,
                                        TRY_SHARE_EVALUATORS ? ReferenceEvaluator?.Evaluator : null);
    }


    /// <summary>
    /// Constructor when only one GPU is requested.
    /// </summary>
    /// <param name="networkID"></param>
    /// <param name="gpuID"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="saveActivations"></param>
    /// <param name="precision"></param>
    /// <param name="dumpTimings"></param>
    /// <param name="enableCUDAGraphs"></param>
    /// <param name="graphBatchSizeDivisor"></param>
    /// <param name="referenceEvaluator"></param>
    public NNEvaluatorCUDA(string networkID,
                               int gpuID = 0,
                               int maxBatchSize = NNBackendLC0_CUDA.DEFAULT_MAX_BATCH_SIZE,
                               bool saveActivations = false,
                               NNEvaluatorPrecision precision = NNEvaluatorPrecision.FP16,
                               bool dumpTimings = false,
                               bool enableCUDAGraphs = true,
                               int graphBatchSizeDivisor = 1,
                               NNEvaluatorCUDA referenceEvaluator = null)
      : this(NNWeightsFileLC0.LookupOrDownload(networkID), gpuID, maxBatchSize, saveActivations, precision, dumpTimings,
                                               enableCUDAGraphs, graphBatchSizeDivisor, referenceEvaluator)
    {
    }

    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => maxBatchSize;


    #endregion

    /// <summary>
    /// If this evaluator produces the same output as another specified evaluator.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <returns></returns>
    public override bool IsEquivalentTo(NNEvaluator evaluator) => EngineNetworkID == evaluator.EngineNetworkID;


    public void StartEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, int numPositions, bool retrieveSupplementalResults = false)
    {
      if (retrieveSupplementalResults && !Evaluator.SaveActivations)
      {
        throw new Exception("retrieveSupplementalResults requires that the evaluator was created with the saveActivations parameter true.");
      }

      // Prepare inputs for NN evaluation (unless not provided, indincating this is already complete)
      if (positions != null)
      {
        if (numPositions != positions.NumPos)
        {
          throw new ArgumentException(nameof(numPositions));
        }

        PrepareInputPositions(positions);
      }

      // Actually do the NN evaluation
      Evaluator.EvaluateNN(numPositions);

      NNEvaluatorStats.UpdateStatsForBatch(GPUID, numPositions);
    }

    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      StartEvaluateIntoBuffers(positions, positions.NumPos, retrieveSupplementalResults);
      const bool COPY_RESULTS = false;
      return GetPostprocessedBatch(positions, null, null, retrieveSupplementalResults, COPY_RESULTS);
    }


    IPositionEvaluationBatch GetPostprocessedBatch(IEncodedPositionBatchFlat positions, short[] numMoves, short[] moveIndices, bool retrieveSupplementalResults, bool copyResults)
    {
      Evaluator.ExtractActivations(positions.NumPos);

      ParallelOptions parallelOptions = ParallelUtils.ParallelOptions(positions.NumPos, NUM_POSITIONS_PER_THREAD_OUTPUT);
      Parallel.ForEach(Partitioner.Create(0, positions.NumPos), parallelOptions,
        range =>
        {
          PrepareOutputPositions(numMoves, moveIndices, range.Item1, range.Item2);
        });

      NNEvaluatorResultActivations[] activations = null;
      if (retrieveSupplementalResults)
      {
        throw new NotImplementedException();
#if NOT
        float[,] rawActivations = Evaluator.inputOutput.OutputValueHeadFC2;
        activations = new NNEvaluatorResultActivations[positions.NumPos];
        for (int i = 0; i < activations.Length; i++)
        {
          activations[i] = new NNEvaluatorResultActivations(i, null, rawActivations);
        }
#endif
      }

      return new PositionEvaluationBatch(IsWDL, HasM, HasUncertaintyV, positions.NumPos, policies, w, l, m, default, activations, new TimingStats(), copyResults);
    }


#region Optional Async support

    protected override Task DoLaunchEvaluateBatchAsync(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      Evaluator.InputsCopyToDeviceFinished.Reset();
      return Task.Run(() =>
      {
        try
        {
          int numPos = positions == null ? numPreparedPositions : positions.NumPos; 
          StartEvaluateIntoBuffers(positions, numPos, retrieveSupplementalResults);
        }
        catch (Exception ex)
        {
          Console.WriteLine("Exception in DoLaunchEvaluateBatchAsync " + ex);
          Console.WriteLine(ex.StackTrace);
        }
      });
    }

    public override IPositionEvaluationBatch GetLastAsyncBatchResult(IEncodedPositionBatchFlat positions, 
                                                                     short[] numMoves,
                                                                     short[] moveIndices,
                                                                     bool retrieveSupplementalResults,
                                                                     bool makeCopyOfResults)
    {
      return GetPostprocessedBatch(positions, numMoves, moveIndices, retrieveSupplementalResults, makeCopyOfResults);
    }

#endregion


    const int NUM_POSITIONS_PER_THREAD_INPUT = 48;
    const int NUM_POSITIONS_PER_THREAD_OUTPUT = 32;

    static bool haveWarnedMoveOverflow = false;

    int numPreparedPositions;
    public void PrepareInputPositions(IEncodedPositionBatchFlat batch)
    {
      if (batch.NumPos > MaxBatchSize)
      {
        new ArgumentOutOfRangeException($"batch.NumPos is too large, max {MaxBatchSize} versus actual {batch.NumPos}");
      }

      numPreparedPositions = batch.NumPos;

      int numPlanes = NNBackendInputOutput.NUM_INPUT_PLANES;
      Span<ulong> masksSource = batch.PosPlaneBitmaps.Span.Slice(0, batch.NumPos * numPlanes);
      Span<ulong> masksDest = Evaluator.inputOutput.InputBoardMasks.AsSpan().Slice(0, batch.NumPos * numPlanes);
      masksSource.CopyTo(masksDest);

      ParallelOptions parallelOptions = ParallelUtils.ParallelOptions(batch.NumPos, NUM_POSITIONS_PER_THREAD_INPUT);

      Parallel.ForEach(Partitioner.Create(0, batch.NumPos), parallelOptions,
        range =>
        {
          Span<MGMoveList> movesSpan = batch.Moves.Span;
          Span<byte> valuesSource = batch.PosPlaneValues.Span;
          Span<float> valuesDest = Evaluator.inputOutput.InputBoardValues.AsSpan();

          for (int i = range.Item1; i < range.Item2; i++)
          {
            // Determine legal move list
            MGMoveList movesLegal = movesSpan[i];
#if NOT
            MGMoveList movesLegal;
            if (movesSpan.Length <= i)
            {
              movesLegal = new MGMoveList();
              MGMoveGen.GenerateMoves(in batch.Positions[i], movesLegal);
            }
            else
            {
              movesLegal = movesSpan[i];
            }
#endif

            // Note that rarely there might be more legal moves than we can fit in our buffer;
            // in this case we just silently ignore some
            // TODO: consider if this could cause missing good moves, if we could prioritize somehow
            if (movesLegal.NumMovesUsed > NNBackendInputOutput.MAX_MOVES && !haveWarnedMoveOverflow)
            {
              Console.WriteLine($"Warning: moves overflow, {movesLegal.NumMovesUsed} legal moves in position exceeds Ceres max of {NNBackendInputOutput.MAX_MOVES}, truncating.");
              haveWarnedMoveOverflow = true;
            }

            int numMoves = Math.Min(NNBackendInputOutput.MAX_MOVES, movesLegal.NumMovesUsed);

            Span<short> moveIndicesSpan = Evaluator.inputOutput.InputMoveIndices.AsSpan();

            Evaluator.inputOutput.InputNumMovesUsed[i] = (short)numMoves;
            int baseOffsetMoves = NNBackendInputOutput.MAX_MOVES * i;

            // Handle the white/black conditions separately for performance reasons.
            bool isBlack = movesLegal.MovesArray[0].BlackToMove;
            if (isBlack)
            {
              for (int m = 0; m < numMoves; m++)
              {
                EncodedMove moveVal = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMoveBlack(movesLegal.MovesArray[m]);
                moveIndicesSpan[baseOffsetMoves + m] = (short)moveVal.IndexNeuralNet;
              }
            }
            else
            {
              for (int m = 0; m < numMoves; m++)
              {
                EncodedMove moveVal = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMoveWhite(movesLegal.MovesArray[m]);
                moveIndicesSpan[baseOffsetMoves + m] = (short)moveVal.IndexNeuralNet;
              }
            }

            int baseOffset = i * NNBackendInputOutput.NUM_INPUT_PLANES;
            for (int j = 0; j < NNBackendInputOutput.NUM_INPUT_PLANES; j++)
            {
              int offset = baseOffset + j;
              valuesDest[offset] = valuesSource[offset];
            }

          }
        });
    }



    [SkipLocalsInit]
    private void PrepareOutputPositions(short[] numMovesArray, short[] moveIndices, int startPos, int endPos)
    {
      // TODO: Improve performance of softmax.
      //       Or use AVX (https://github.com/reyoung/avx_mathfun)

      NNBackendInputOutput io = Evaluator.inputOutput;

      Span<float> policiesMasked = io.OutputPolicyHeadMasked.AsSpan();
      Span<short> moveIndicesSpan = moveIndices != null ? moveIndices.AsSpan() : io.InputMoveIndices.AsSpan();
      Span<short> numMovesSpan = numMovesArray != null ? numMovesArray.AsSpan() :  io.InputNumMovesUsed.AsSpan();

      Span<FP16> mlhSpan = Evaluator.mlhOutputBuffer.AsSpan();
      Span<FP16> wdlOutputBuffer = Evaluator.wdlOutputBuffer.AsSpan();

      for (int i = startPos; i < endPos; i++)
      {
        if (IsWDL)
        {
          int valueOffset = i * (IsWDL ? 3 : 1);

          // ...........................................................................
          float win = wdlOutputBuffer[valueOffset + 0];
          float draw = wdlOutputBuffer[valueOffset + 1];
          float loss = wdlOutputBuffer[valueOffset + 2];

          float max = MathF.Max(MathF.Max(win, draw), loss);

          win = MathF.Exp(win - max);
          draw = MathF.Exp(draw - max);
          loss = MathF.Exp(loss - max);

          float sum = win + draw + loss;
          if (float.IsNaN(sum))
          {
            throw new Exception($"Neural network backend returned NaN value head from evaluator {this}");
          }

          win /= sum;
          loss /= sum;

          w[i] = (FP16)win;
          l[i] = (FP16)loss;
        }
        else
        {
          w[i] = wdlOutputBuffer[i];
        }

        if (HasM)
        {
          m[i] = mlhSpan[i];
        }

        int numMoves = numMovesSpan[i];

        if (numMoves > 0)
        {
          int numMovesToSave = Math.Min(CompressedPolicyVector.NUM_MOVE_SLOTS, numMoves);

          int moveBaseOffset = NNBackendInputOutput.MAX_MOVES * i;

          // We need to sort here to make sure the highest P are in the first NUM_MOVE_SLOTS
          Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs
              = stackalloc PolicyVectorCompressedInitializerFromProbs.ProbEntry[numMoves];
          unsafe
          {
            for (int j = 0; j < numMoves; j++)
            {
              short moveCode = moveIndicesSpan[moveBaseOffset + j];
              float p = policiesMasked[moveBaseOffset + j];
              if (float.IsNaN(p))
              {
                throw new Exception($"Neural network backend return NaN in policy head from evaluator {this}");
              }

              probs[j] = new PolicyVectorCompressedInitializerFromProbs.ProbEntry(moveCode, p);
            }
          }

          PolicyVectorCompressedInitializerFromProbs.InitializeFromLogitProbsArray(ref policies[i], numMoves, numMovesToSave, probs);
        }
      }
    }

    protected override void DoShutdown()
    {
      Evaluator.Dispose();
    }


    public override string ToString()
    {
      return $"<NNEvaluatorCUDA {Evaluator}>";
    }

  }

}
