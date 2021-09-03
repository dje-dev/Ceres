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
    internal readonly NNBackendLC0_CUDA Evaluator;

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


    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (positions.NumPos > MaxBatchSize) new ArgumentOutOfRangeException($"batch.NumPos is too large, max {MaxBatchSize} versus actual {positions.NumPos}");

      if (retrieveSupplementalResults && !Evaluator.SaveActivations)
      {
        throw new Exception("retrieveSupplementalResults requires that the evaluator was created with the saveActivations parameter true.");
      }

      int numToPrepare = positions.NumPos;
      //Console.WriteLine("INPUTS " + System.Threading.Thread.CurrentThread.ManagedThreadId + " " + positions.NumPos);

      // Transform input information 
      PrepareInputPositions(positions);

      // Actually do the NN evaluation
      Evaluator.EvaluateNN(positions.NumPos);

      NNEvaluatorStats.UpdateStatsForBatch(GPUID, positions.NumPos);

      ParallelOptions parallelOptions = ParallelUtils.ParallelOptions(positions.NumPos, NUM_POSITIONS_PER_THREAD_OUTPUT);

      Parallel.ForEach(Partitioner.Create(0, positions.NumPos), parallelOptions,
        range =>
        {
          PrepareOutputPositions(range.Item1, range.Item2);
        });

      NNEvaluatorResultActivations[] activations = null;
      if (retrieveSupplementalResults)
      {
        float[,] rawActivations = Evaluator.inputOutput.OutputValueHeadFC2;
        activations = new NNEvaluatorResultActivations[positions.NumPos];
        for (int i = 0; i < activations.Length; i++)
        {
          activations[i] = new NNEvaluatorResultActivations(i, null, rawActivations);
        }
      }

      return new PositionEvaluationBatch(IsWDL, HasM, positions.NumPos, policies, w, l, m, activations, new TimingStats()); ;
    }


    const int NUM_POSITIONS_PER_THREAD_INPUT = 48;
    const int NUM_POSITIONS_PER_THREAD_OUTPUT = 32;

    private void PrepareInputPositions(IEncodedPositionBatchFlat batch)
    {
      int numPlanes = NNBackendInputOutput.NUM_INPUT_PLANES;
      Span<ulong> masksSource = batch.PosPlaneBitmaps.Slice(0, batch.NumPos * numPlanes);
      Span<ulong> masksDest = Evaluator.inputOutput.InputBoardMasks.AsSpan().Slice(0, batch.NumPos * numPlanes);
      masksSource.CopyTo(masksDest);

      ParallelOptions parallelOptions = ParallelUtils.ParallelOptions(batch.NumPos, NUM_POSITIONS_PER_THREAD_INPUT);

      Parallel.ForEach(Partitioner.Create(0, batch.NumPos), parallelOptions,
        range =>
        {
          Span<MGMoveList> movesSpan = batch.Moves;
          Span<byte> valuesSource = batch.PosPlaneValues;
          Span<float> valuesDest = Evaluator.inputOutput.InputBoardValues.AsSpan();

          for (int i = range.Item1; i < range.Item2; i++)
          {
            // Determine legal move list
            MGMoveList movesLegal = movesSpan[i];

            // Note that rarely there might be more legal moves than we can fit in our buffer;
            // in this case we just silently ignore some
            // TODO: consider if this could cause missing good moves, if we could prioritize somehow
            if (movesLegal.NumMovesUsed > NNBackendInputOutput.MAX_MOVES)
            {
              Console.WriteLine("Warning: move overflow");
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
    private void PrepareOutputPositions(int startPos, int endPos)
    {
      NNBackendInputOutput io = Evaluator.inputOutput;

      Span<float> policiesMasked = io.OutputPolicyHeadMasked.AsSpan();
      Span<short> moveIndicesSpan = io.InputMoveIndices.AsSpan();

      for (int i = startPos; i < endPos; i++)
      {
        int valueOffset = i * (IsWDL ? 3 : 1);

        if (IsWDL)
        {
          w[i] = (FP16)io.OutputValueHead[i, 0];
          if (float.IsNaN(w[i]))
          {
            throw new Exception($"Neural network backend returned NaN value head from evaluator {this}");
          }

          l[i] = (FP16)io.OutputValueHead[i, 2];
        }
        else
        {
          w[i] = (FP16)io.OutputValueHead[i, 0];
        }

        if (HasM)
        {
          m[i] = (FP16)io.OutputMovesLeftHead[i];
        }

        int numMoves = io.InputNumMovesUsed[i];

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

          PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(ref policies[i], numMoves, numMovesToSave, probs);
        }
      }
    }

    protected override void DoShutdown()
    {
      // Release layers, but only if no ReferenceEvaluator
      //       Evaluator.Dispose(); // **** TBD
    }


    public override string ToString()
    {
      return $"<NNEvaluatorCUDA {Evaluator}>";
    }

  }

}
