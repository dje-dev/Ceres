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
using System.IO;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNBackends.LC0EX;
using Ceres.Chess.NNEvaluators.Internals;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Chess.NNEvaluators.LC0EX
{
  /// <summary>
  /// NNEvaluator which runs an Lc0 network compiled into an lc0ex "neural executable"
  /// (precompiled per-GPU-architecture cubins plus a kernel graph, built offline by the
  /// lczero-triton lc0ex builder), with weights from the builder's carrier .pb.gz.
  ///
  /// lc0ex (the format, the kernels, the builder and the reference runtime) is the work of
  /// Kovax (GitHub: Kovax007) for Leela Chess Zero. The executor this evaluator calls
  /// (NNBackends/LC0EX) is a C# port of his runtime; see the provenance notes in those files.
  ///
  /// Pure C# on the CUDA driver API (see LC0EXExecutable / LC0EXExecution): no C++,
  /// no TensorRT, no cuBLAS. Weights are shared by all instances on the same GPU.
  ///
  /// Inputs are the Lc0 classical 112-plane encoding; outputs are policy logits (gathered to
  /// legal moves and softmaxed on the host), WDL logits (softmaxed on the host) and moves-left.
  ///
  /// An lc0ex executable holds programs of a few fixed batch sizes. Any batch size up to
  /// MaxBatchSize is accepted: the execution covers it with the cheapest sequence of
  /// program launches (see LC0EXExecution), so at most a few slots of padding are computed.
  /// </summary>
  public class NNEvaluatorLC0EX : NNEvaluator
  {
    /// <summary>
    /// Upper bound on legal moves in a chess position (218) rounded up.
    /// </summary>
    const int MAX_MOVES = 256;

    /// <summary>
    /// Number of positions decoded per thread when converting outputs.
    /// </summary>
    const int NUM_POSITIONS_PER_THREAD_OUTPUT = 32;

    /// <summary>
    /// CUDA device index on which this evaluator runs.
    /// </summary>
    public readonly int GPUID;

    /// <summary>
    /// Full path of the carrier (weights) file.
    /// </summary>
    public string CarrierFileName => execution.Executable.CarrierFileName;

    /// <summary>
    /// Full path of the lc0ex executable file.
    /// </summary>
    public string ArtifactFileName => execution.Executable.ArtifactFileName;

    readonly LC0EXExecution execution;
    readonly int maxBatchSize;
    readonly object evalLockObj = new();
    bool haveWarmedUp;

    // Result buffers (reused by every batch; a returned batch is valid until the next evaluation).
    readonly CompressedPolicyVector[] policies;
    readonly FP16[] w;
    readonly FP16[] l;
    readonly FP16[] m;

    // Batch being evaluated and the cached delegate over it (see DoEvaluateIntoBuffers).
    IEncodedPositionBatchFlat currentPositions;
    Action<int, int> decodeOutputsAction;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="carrierFileName">carrier .pb.gz holding the weights</param>
    /// <param name="artifactFileName">.lc0ex executable compiled for this GPU's architecture</param>
    /// <param name="gpuID">CUDA device index</param>
    /// <param name="maxBatchSize">largest batch accepted (any size; batches are covered by several program launches as needed)</param>
    /// <param name="useCUDAGraphs">if kernels are launched through instantiated CUDA graphs</param>
    public NNEvaluatorLC0EX(string carrierFileName, string artifactFileName, int gpuID, int maxBatchSize, bool useCUDAGraphs = true)
    {
      if (!SoftwareManager.IsCUDAInstalled)
      {
        throw new Exception("GPU hardware with CUDA installation is required but not found.");
      }
      if (maxBatchSize < 1)
      {
        throw new ArgumentOutOfRangeException(nameof(maxBatchSize));
      }

      GPUID = gpuID;
      this.maxBatchSize = maxBatchSize;

      LC0EXExecutable executable = LC0EXExecutable.Acquire(gpuID, artifactFileName, carrierFileName);
      execution = new LC0EXExecution(executable, maxBatchSize, useCUDAGraphs);  // takes ownership of the reference

      EngineNetworkID = $"LC0EX:{executable.CarrierFileName}|{executable.ArtifactFileName}";

      policies = new CompressedPolicyVector[maxBatchSize];
      w = new FP16[maxBatchSize];
      l = new FP16[maxBatchSize];
      m = new FP16[maxBatchSize];

      Warmup();
    }


    #region Properties

    public override bool IsWDL => true;
    public override bool HasM => execution.HasMovesLeft;
    public override bool HasUncertaintyV => false;
    public override bool HasUncertaintyP => false;
    public override bool HasAction => false;
    public override bool HasValueSecondary => false;

    /// <summary>
    /// Policies are returned in logit order (sorted), not the legal move list order.
    /// </summary>
    public override bool PolicyReturnedSameOrderMoveList => false;

    public override InputTypes InputsRequired => InputTypes.Boards | InputTypes.Moves;

    public override int MaxBatchSize => maxBatchSize;

    /// <summary>
    /// Batch sizes of the compiled programs in use (ascending).
    /// </summary>
    public int[] ProgramBatchSizes => execution.ProgramBatchSizes;


    /// <summary>
    /// Number of position slots actually computed for a batch of numPositions:
    /// the sum of the sizes of the programs launched to cover it (never reported
    /// as more than MaxBatchSize, the most a caller could fill).
    /// </summary>
    public override int PaddedBatchCapacity(int numPositions)
    {
      if (numPositions <= 0)
      {
        return 0;
      }
      return Math.Min(maxBatchSize, execution.PaddedCapacity(Math.Min(numPositions, maxBatchSize)));
    }


    EvaluatorInfo infoCached;

    /// <summary>
    /// Reports the carrier (weights) file size, used by search to gate batch sizing.
    /// </summary>
    public override EvaluatorInfo Info => infoCached ??= new EvaluatorInfo(0, execution.Executable.CarrierFileSizeBytes);


    /// <summary>
    /// Evaluators are equivalent if they run the same network, i.e. the same carrier.
    /// (The executable may differ: each GPU architecture needs its own build of it.)
    /// </summary>
    public override bool IsEquivalentTo(NNEvaluator evaluator)
      => evaluator is NNEvaluatorLC0EX other && CarrierFileName == other.CarrierFileName;

    #endregion


    /// <summary>
    /// Records throughput estimates (the benchmark runs a batch of 1 and one large batch).
    /// The graphs are already instantiated and every program has already been run,
    /// when its latency was measured at construction. Best effort.
    /// </summary>
    public override void Warmup()
    {
      if (haveWarmedUp)
      {
        return;
      }
      haveWarmedUp = true;

      try
      {
        CalcStatistics(computeBreaks: false);
      }
      catch (Exception exc)
      {
        Console.WriteLine($"Warning: NNEvaluatorLC0EX warmup/NPS estimation failed, using default estimates. {exc.Message}");
      }
    }


    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (retrieveSupplementalResults)
      {
        throw new NotSupportedException("NNEvaluatorLC0EX does not support retrieveSupplementalResults.");
      }

      int numPos = positions.NumPos;
      if (numPos > maxBatchSize)
      {
        throw new ArgumentOutOfRangeException(nameof(positions), $"Batch of {numPos} exceeds evaluator maximum {maxBatchSize}.");
      }
      if (!(Options.PolicyTemperature > 0))
      {
        throw new Exception($"NNEvaluatorLC0EX requires a positive policy temperature (was {Options.PolicyTemperature}).");
      }

      lock (evalLockObj)
      {
        // (The cached delegates read the batch from this field, avoiding a closure allocation per batch.)
        currentPositions = positions;
        try
        {
          CopyInputs(positions, numPos);

          // The whole batch is one device dispatch (the execution splits it into
          // program launches internally and waits for all of them once).
          // (The gate is installed by NNEvaluatorSet for a single-device evaluator. Under a multi-device
          // split it is not, which is the better choice here: holding a gate per device was measured
          // 8% slower in two-GPU search.)
          object deviceLock = DeviceExclusiveExecutionLock;
          BackendTimeTracker tracker = BackendTimeTracker;
          if (deviceLock != null)
          {
            lock (deviceLock)
            {
              RunTracked(tracker, numPos);
            }
          }
          else
          {
            RunTracked(tracker, numPos);
          }

          ParallelUtils.ForRange(0, numPos, NUM_POSITIONS_PER_THREAD_OUTPUT, decodeOutputsAction ??= DecodeOutputs);
        }
        finally
        {
          currentPositions = null;
        }

        NNEvaluatorStats.UpdateStatsForBatch(GPUID, numPos);

        return new PositionEvaluationBatch(IsWDL, HasM, HasUncertaintyV, HasUncertaintyP,
                                           HasAction, HasValueSecondary, HasState,
                                           numPos, policies, null, w, l, default, default, m, default,
                                           default, default, null, new TimingStats(), default, default, false);
      }
    }


    void RunTracked(BackendTimeTracker tracker, int count)
    {
      tracker?.EnterBackend();
      try
      {
        execution.Run(count);
      }
      finally
      {
        tracker?.ExitBackend();
      }
    }


    /// <summary>
    /// Copies the first count positions of the Lc0 packed planes into the execution's pinned inputs:
    /// masks verbatim, byte plane values widened to float (rule-50 plane raw, as Lc0 classical expects).
    /// </summary>
    void CopyInputs(IEncodedPositionBatchFlat positions, int count)
    {
      const int PLANES = LC0EXExecution.NUM_PLANES;
      positions.PosPlaneBitmaps.Span.Slice(0, count * PLANES).CopyTo(execution.InputMasks);
      TensorPrimitives.ConvertChecked<byte, float>(positions.PosPlaneValues.Span.Slice(0, count * PLANES),
                                                   execution.InputValues.Slice(0, count * PLANES));
    }


    /// <summary>
    /// Converts the raw outputs of positions [rangeStart, rangeEnd) of the current batch
    /// into the result buffers: WDL logits to probabilities, moves-left as is, and the policy
    /// logits of the legal moves (looked up at their index in the side-to-move frame) to a
    /// compressed policy vector.
    /// </summary>
    [SkipLocalsInit]
    void DecodeOutputs(int rangeStart, int rangeEnd)
    {
      Span<MGMoveList> movesSpan = currentPositions.Moves.Span;
      ReadOnlySpan<float> wdlLogits = execution.OutputWDLLogits;
      ReadOnlySpan<float> mlh = execution.OutputMLH;
      ReadOnlySpan<float> policyLogits = execution.OutputPolicyLogits;
      float inverseTemperature = 1.0f / Options.PolicyTemperature;
      bool hasM = HasM;
      Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probsBuffer
        = stackalloc PolicyVectorCompressedInitializerFromProbs.ProbEntry[MAX_MOVES];

      for (int i = rangeStart; i < rangeEnd; i++)
      {
        // WDL (softmax of logits).
        float win = wdlLogits[i * 3 + 0];
        float draw = wdlLogits[i * 3 + 1];
        float loss = wdlLogits[i * 3 + 2];
        float max = MathF.Max(MathF.Max(win, draw), loss);
        win = MathF.Exp(win - max);
        draw = MathF.Exp(draw - max);
        loss = MathF.Exp(loss - max);
        float sum = win + draw + loss;
        if (float.IsNaN(sum))
        {
          throw new Exception($"lc0ex backend returned NaN value head from evaluator {this}");
        }
        w[i] = (FP16)(win / sum);
        l[i] = (FP16)(loss / sum);

        // Moves left (raw plies).
        if (hasM)
        {
          float movesLeft = mlh[i];
          if (float.IsNaN(movesLeft))
          {
            throw new Exception($"lc0ex backend returned NaN moves left head from evaluator {this}");
          }
          m[i] = (FP16)MathF.Max(0, movesLeft);
        }

        // Policy: gather logits of the legal moves, scale by temperature, softmax via the initializer.
        MGMoveList legal = movesSpan[i];
        int count = legal == null ? 0 : Math.Min(MAX_MOVES, legal.NumMovesUsed);
        if (count > 0)
        {
          MGMove[] moves = legal.MovesArray;
          bool blackToMove = moves[0].BlackToMove;
          int numMovesToSave = Math.Min(CompressedPolicyVector.NUM_MOVE_SLOTS, count);
          Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs = probsBuffer.Slice(0, count);
          int logitBase = i * LC0EXExecution.NUM_POLICY;
          for (int j = 0; j < count; j++)
          {
            EncodedMove encodedMove = blackToMove ? ConverterMGMoveEncodedMove.MGChessMoveToEncodedMoveBlack(moves[j])
                                                  : ConverterMGMoveEncodedMove.MGChessMoveToEncodedMoveWhite(moves[j]);
            short moveCode = (short)encodedMove.IndexNeuralNet;
            float p = policyLogits[logitBase + moveCode] * inverseTemperature;
            if (float.IsNaN(p))
            {
              throw new Exception($"lc0ex backend returned NaN in policy head from evaluator {this}");
            }
            probs[j] = new PolicyVectorCompressedInitializerFromProbs.ProbEntry(moveCode, p);
          }

          SideType side = blackToMove ? SideType.Black : SideType.White;
          PolicyVectorCompressedInitializerFromProbs.InitializeFromLogitProbsArray(ref policies[i], side, count, numMovesToSave, probs);
        }
        else
        {
          policies[i] = default;
        }
      }
    }


    protected override void DoShutdown()
    {
      // Wait for any evaluation in progress (its buffers and stream are about to be freed).
      lock (evalLockObj)
      {
        execution.Dispose();
      }
    }


    public override string ToString()
      => $"<NNEvaluatorLC0EX GPU {GPUID} {Path.GetFileName(ArtifactFileName)} + {Path.GetFileName(CarrierFileName)} "
       + $"using programs [{string.Join(",", ProgramBatchSizes)}] max {maxBatchSize}>";
  }
}
