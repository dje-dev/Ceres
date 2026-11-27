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
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;

using Microsoft.ML.OnnxRuntime;

using Ceres.Base.Benchmarking;
using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Base.Threading;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNBackends.ONNXRuntime;
using Ceres.Chess.NNEvaluators.Internals;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Chess.NNEvaluators.Fry
{
  /// <summary>
  /// NNEvaluator for the Fry chess transformer (github.com/GilliamWong/chess-transformer)
  /// exported to ONNX, run through ONNX Runtime with the TensorRT (default) or CUDA execution provider.
  ///
  /// Network interface (differs from Lc0 in every respect):
  ///   input  "board"  int64 [B,70]: 64 square tokens in the side-to-move frame (square ^ 56 with Black to move;
  ///                  0 empty, 1-6 our P N B R Q K, 7-12 theirs), then turn (14 White / 13 Black),
  ///                  castling rights 15/16 for our O-O, our O-O-O, their O-O, their O-O-O,
  ///                  and en passant 17 + square (only if a legal en passant capture exists), else 17. No history.
  ///   output "policy" float [B,1968]: logits over Fry's own move table (side-to-move frame, explicit queen
  ///                  promotions, castling as the king's two-square move).
  ///   output "value"  float [B]: expected score for the side to move in [0,1].
  ///
  /// The value is sharpened as the Fry engine serves it, sigmoid(k * logit(E)), with k read from the ONNX
  /// metadata "value_sharpness" (override with option VSHARP=k), then converted to W/L with no draw mass
  /// beyond 1-|V| (the same mapping MCGS applies to any non-WDL value head). There is no moves-left head.
  /// 
  /// Example network string:
  ///   "FRY:fry-final-117.fp16.onnx|TRT=true;VSHARP=1.25;POLTEMP=0.9;RUNBATCH=256;OPT=3"
  /// </summary>
  public sealed class NNEvaluatorFry : NNEvaluator
  {
    public const int NUM_TOKENS = 70;
    public const int NUM_POLICY = 1968;

    /// <summary>
    /// Default largest batch passed to a single ONNX Runtime run, which is also the TensorRT profile's
    /// optimal and maximum shape; larger batches are evaluated in several runs. A TensorRT engine is
    /// fastest at its profile's optimal shape (H100 NVL, fp16: 6450 pos/s at 256 with a 256 profile,
    /// but 5100 at 256 and 6200 at 512 with a 512 profile), and 256 gave the highest peak.
    /// </summary>
    public const int DEFAULT_RUN_BATCH_SIZE = 256;

    const int MAX_MOVES = 256;
    const int NUM_POSITIONS_PER_THREAD = 32;
    const float MISSING_MOVE_LOGIT = -100f;

    // Token ids (model/encoding.py).
    const int OPPONENT_OFFSET = 6;
    const int TURN_TOKEN_OFFSET = 13;
    const int CASTLING_TOKEN_OFFSET = 15;
    const int EN_PASSANT_TOKEN_OFFSET = 17;

    static readonly string[] INPUT_NAMES = ["board"];
    static readonly string[] OUTPUT_NAMES = ["policy", "value"];


    /// <summary>
    /// Full path of the ONNX file.
    /// </summary>
    public readonly string ONNXFileName;

    /// <summary>
    /// CUDA device index.
    /// </summary>
    public readonly int GPUID;

    /// <summary>
    /// If the TensorRT execution provider is used (else CUDA).
    /// </summary>
    public readonly bool UseTensorRT;

    /// <summary>
    /// Exponent applied to the value's odds (Fry's "value_sharpness").
    /// </summary>
    public readonly float ValueSharpness;

    readonly InferenceSession session;
    readonly RunOptions runOptions;
    readonly int maxBatchSize;
    readonly int maxRunBatchSize;
    readonly object evalLockObj = new();
    bool haveWarmedUp;

    // Network input/output buffers (sized for maxBatchSize).
    readonly long[] inputTokens;
    readonly float[] outputPolicy;
    readonly float[] outputValue;

    // Result buffers (reused; a returned batch is valid until the next evaluation).
    readonly CompressedPolicyVector[] policies;
    readonly FP16[] w;
    readonly FP16[] l;

    // Batch being evaluated and cached delegates over it.
    IEncodedPositionBatchFlat currentPositions;
    Action<int, int> encodeInputsAction;
    Action<int, int> decodeOutputsAction;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="onnxFileName">Fry ONNX file (fp16 export recommended for TensorRT)</param>
    /// <param name="gpuID">CUDA device index</param>
    /// <param name="maxBatchSize">largest batch accepted (evaluated in runs of at most runBatchSize)</param>
    /// <param name="useTensorRT">if the TensorRT execution provider is used (else CUDA)</param>
    /// <param name="useFP16">if TensorRT may use fp16 kernels</param>
    /// <param name="valueSharpnessOverride">value sharpness to use instead of the one in the file's metadata</param>
    /// <param name="optimizationLevel">TensorRT builder optimization level</param>
    public NNEvaluatorFry(string onnxFileName, int gpuID, int maxBatchSize,
                          bool useTensorRT = true, bool useFP16 = true,
                          float? valueSharpnessOverride = null, int optimizationLevel = 3,
                          int runBatchSize = DEFAULT_RUN_BATCH_SIZE)
    {
      if (!File.Exists(onnxFileName))
      {
        throw new FileNotFoundException($"Fry ONNX file not found: {onnxFileName}");
      }
      if (maxBatchSize < 1)
      {
        throw new ArgumentOutOfRangeException(nameof(maxBatchSize));
      }

      ONNXFileName = Path.GetFullPath(onnxFileName);
      GPUID = gpuID;
      UseTensorRT = useTensorRT;
      this.maxBatchSize = maxBatchSize;
      if (runBatchSize < 1)
      {
        throw new ArgumentOutOfRangeException(nameof(runBatchSize));
      }
      maxRunBatchSize = Math.Min(maxBatchSize, runBatchSize);

      session = new InferenceSession(ONNXFileName, CreateSessionOptions(useFP16, optimizationLevel));
      runOptions = new RunOptions();
      ValidateInterface();

      ValueSharpness = valueSharpnessOverride ?? ReadValueSharpness();
      if (!(ValueSharpness > 0))
      {
        throw new Exception($"Fry value sharpness must be positive, got {ValueSharpness}");
      }

      inputTokens = new long[maxBatchSize * NUM_TOKENS];
      outputPolicy = new float[maxBatchSize * NUM_POLICY];
      outputValue = new float[maxBatchSize];
      policies = new CompressedPolicyVector[maxBatchSize];
      w = new FP16[maxBatchSize];
      l = new FP16[maxBatchSize];

      EngineNetworkID = $"FRY:{ONNXFileName}";
    }


    #region Properties

    public override bool IsWDL => true;
    public override bool HasM => false;
    public override bool HasUncertaintyV => false;
    public override bool HasUncertaintyP => false;
    public override bool HasAction => false;
    public override bool HasValueSecondary => false;

    /// <summary>
    /// Policies are returned sorted by probability, not in legal move list order.
    /// </summary>
    public override bool PolicyReturnedSameOrderMoveList => false;

    /// <summary>
    /// The current position (no history, no Lc0 planes) and its legal moves.
    /// </summary>
    public override InputTypes InputsRequired => InputTypes.Positions | InputTypes.Moves;

    public override int MaxBatchSize => maxBatchSize;

    EvaluatorInfo infoCached;
    public override EvaluatorInfo Info => infoCached ??= new EvaluatorInfo(0, new FileInfo(ONNXFileName).Length);

    public override bool IsEquivalentTo(NNEvaluator evaluator)
      => evaluator is NNEvaluatorFry other
      && ONNXFileName == other.ONNXFileName
      && ValueSharpness == other.ValueSharpness;

    #endregion


    #region Session setup

    SessionOptions CreateSessionOptions(bool useFP16, int optimizationLevel)
    {
      SessionOptions so = new SessionOptions();
      so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
      so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
      so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

      if (UseTensorRT)
      {
        CUDADevice device = CUDADevice.GetContext(GPUID);
        string cacheDir = ONNXExecutor.GetTRTEngineCacheDir(ONNXFileName);
        Directory.CreateDirectory(cacheDir);
        string cachePrefix = FileUtils.FileNameSanitized(Path.GetFileNameWithoutExtension(ONNXFileName))
                           + $"_{device.NumStreamingMultiprocessors}sm_cuda{device.CUDACapabilityMajor}.{device.CUDACapabilityMinor}"
                           + $"_bs1-{maxRunBatchSize}" + (useFP16 ? "_fp16" : "_fp32") + $"_opt{optimizationLevel}";

        using OrtTensorRTProviderOptions trtOptions = new();
        trtOptions.UpdateOptions(new Dictionary<string, string>
        {
          ["device_id"] = GPUID.ToString(),
          ["trt_max_workspace_size"] = (4L * 1024 * 1024 * 1024).ToString(),
          ["trt_fp16_enable"] = useFP16 ? "1" : "0",
          ["trt_builder_optimization_level"] = optimizationLevel.ToString(),
          ["trt_profile_min_shapes"] = "board:1x70",
          ["trt_profile_opt_shapes"] = $"board:{maxRunBatchSize}x70",
          ["trt_profile_max_shapes"] = $"board:{maxRunBatchSize}x70",
          ["trt_engine_cache_enable"] = "1",
          ["trt_engine_cache_path"] = cacheDir,
          ["trt_engine_cache_prefix"] = cachePrefix,
          ["trt_timing_cache_enable"] = "1",
          ["trt_timing_cache_path"] = cacheDir,
          ["trt_auxiliary_streams"] = "0",
        });
        so.AppendExecutionProvider_Tensorrt(trtOptions);
      }

      // CUDA either as the provider or as TensorRT's fallback for any nodes it does not take.
      using OrtCUDAProviderOptions cudaOptions = new();
      cudaOptions.UpdateOptions(new Dictionary<string, string> { ["device_id"] = GPUID.ToString() });
      so.AppendExecutionProvider_CUDA(cudaOptions);

      return so;
    }


    void ValidateInterface()
    {
      static string Shape(NodeMetadata m) => string.Join("x", m.Dimensions);

      if (!session.InputMetadata.TryGetValue("board", out NodeMetadata input)
       || input.ElementDataType != Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int64
       || input.Dimensions.Length != 2 || input.Dimensions[1] != NUM_TOKENS)
      {
        throw new Exception($"{ONNXFileName} is not a Fry network: expected input board int64 [batch,{NUM_TOKENS}]");
      }
      if (!session.OutputMetadata.TryGetValue("policy", out NodeMetadata policy)
       || policy.Dimensions.Length != 2 || policy.Dimensions[1] != NUM_POLICY)
      {
        throw new Exception($"{ONNXFileName} is not a Fry network: expected output policy [batch,{NUM_POLICY}]");
      }
      if (!session.OutputMetadata.TryGetValue("value", out NodeMetadata value) || value.Dimensions.Length != 1)
      {
        throw new Exception($"{ONNXFileName} is not a Fry network: expected output value [batch]");
      }
    }


    float ReadValueSharpness()
    {
      if (session.ModelMetadata.CustomMetadataMap.TryGetValue("value_sharpness", out string str)
       && float.TryParse(str, NumberStyles.Float, CultureInfo.InvariantCulture, out float sharpness))
      {
        return sharpness;
      }
      return 1.0f;
    }

    #endregion


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
        Console.WriteLine($"Warning: NNEvaluatorFry warmup/NPS estimation failed, using default estimates. {exc.Message}");
      }
    }


    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (retrieveSupplementalResults)
      {
        throw new NotSupportedException("NNEvaluatorFry does not support retrieveSupplementalResults.");
      }

      int numPos = positions.NumPos;
      if (numPos > maxBatchSize)
      {
        throw new ArgumentOutOfRangeException(nameof(positions), $"Batch of {numPos} exceeds evaluator maximum {maxBatchSize}.");
      }
      if (positions.Positions.IsEmpty || positions.Moves.IsEmpty)
      {
        throw new Exception("NNEvaluatorFry requires Positions and Moves in the batch.");
      }
      if (!(Options.PolicyTemperature > 0))
      {
        throw new Exception($"NNEvaluatorFry requires a positive policy temperature (was {Options.PolicyTemperature}).");
      }

      lock (evalLockObj)
      {
        currentPositions = positions;
        try
        {
          ParallelUtils.ForRange(0, numPos, NUM_POSITIONS_PER_THREAD, encodeInputsAction ??= EncodeInputs);

          object deviceLock = DeviceExclusiveExecutionLock;
          if (deviceLock != null)
          {
            lock (deviceLock)
            {
              RunNetwork(numPos);
            }
          }
          else
          {
            RunNetwork(numPos);
          }

          ParallelUtils.ForRange(0, numPos, NUM_POSITIONS_PER_THREAD, decodeOutputsAction ??= DecodeOutputs);
        }
        finally
        {
          currentPositions = null;
        }

        NNEvaluatorStats.UpdateStatsForBatch(GPUID, numPos);

        return new PositionEvaluationBatch(IsWDL, HasM, HasUncertaintyV, HasUncertaintyP,
                                           HasAction, HasValueSecondary, HasState,
                                           numPos, policies, null, w, l, default, default, default, default,
                                           default, default, null, new TimingStats(), default, default, false);
      }
    }


    void RunNetwork(int numPos)
    {
      BackendTimeTracker tracker = BackendTimeTracker;
      tracker?.EnterBackend();
      try
      {
        for (int start = 0; start < numPos; start += maxRunBatchSize)
        {
          int count = Math.Min(maxRunBatchSize, numPos - start);
          using OrtValue input = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                                                                      new Memory<long>(inputTokens, start * NUM_TOKENS, count * NUM_TOKENS),
                                                                      [count, NUM_TOKENS]);
          using OrtValue policy = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                                                                       new Memory<float>(outputPolicy, start * NUM_POLICY, count * NUM_POLICY),
                                                                       [count, NUM_POLICY]);
          using OrtValue value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                                                                      new Memory<float>(outputValue, start, count),
                                                                      [count]);
          session.Run(runOptions, INPUT_NAMES, [input], OUTPUT_NAMES, [policy, value]);
        }
      }
      finally
      {
        tracker?.ExitBackend();
      }
    }


    #region Encoding

    /// <summary>
    /// Writes the 70 input tokens of positions [rangeStart, rangeEnd) of the current batch.
    /// </summary>
    void EncodeInputs(int rangeStart, int rangeEnd)
    {
      Span<MGPosition> positionsSpan = currentPositions.Positions.Span;
      Span<MGMoveList> movesSpan = currentPositions.Moves.Span;
      for (int i = rangeStart; i < rangeEnd; i++)
      {
        EncodePosition(in positionsSpan[i], movesSpan[i], inputTokens.AsSpan(i * NUM_TOKENS, NUM_TOKENS));
      }
    }


    /// <summary>
    /// Fry's encode_board for a position, given its legal moves (used to decide en passant).
    /// </summary>
    public static void EncodePosition(in MGPosition position, MGMoveList legalMoves, Span<long> tokens)
    {
      bool blackToMove = position.BlackToMove;
      int flip = blackToMove ? 56 : 0;
      SideType us = blackToMove ? SideType.Black : SideType.White;

      for (int square = 0; square < 64; square++)
      {
        Piece piece = position.GetPieceAtSquare(new Square(square));
        tokens[square ^ flip] = piece.Type == PieceType.None ? 0
                              : (int)piece.Type + (piece.Side == us ? 0 : OPPONENT_OFFSET);
      }

      bool whiteOO = position.WhiteCanCastle && CastlingPiecesHome(in position, SideType.White, kingside: true);
      bool whiteOOO = position.WhiteCanCastleLong && CastlingPiecesHome(in position, SideType.White, kingside: false);
      bool blackOO = position.BlackCanCastle && CastlingPiecesHome(in position, SideType.Black, kingside: true);
      bool blackOOO = position.BlackCanCastleLong && CastlingPiecesHome(in position, SideType.Black, kingside: false);

      tokens[64] = TURN_TOKEN_OFFSET + (blackToMove ? 0 : 1);
      tokens[65] = CASTLING_TOKEN_OFFSET + ((blackToMove ? blackOO : whiteOO) ? 1 : 0);
      tokens[66] = CASTLING_TOKEN_OFFSET + ((blackToMove ? blackOOO : whiteOOO) ? 1 : 0);
      tokens[67] = CASTLING_TOKEN_OFFSET + ((blackToMove ? whiteOO : blackOO) ? 1 : 0);
      tokens[68] = CASTLING_TOKEN_OFFSET + ((blackToMove ? whiteOOO : blackOOO) ? 1 : 0);

      // En passant only when a legal en passant capture exists (python-chess has_legal_en_passant);
      // Ceres keeps the en passant file after every double push.
      int enPassant = 0;
      if (legalMoves != null)
      {
        MGMove[] moves = legalMoves.MovesArray;
        for (int j = 0; j < legalMoves.NumMovesUsed; j++)
        {
          if (moves[j].EnPassantCapture)
          {
            enPassant = moves[j].ToSquare.SquareIndexStartA1 ^ flip;
            break;
          }
        }
      }
      tokens[69] = EN_PASSANT_TOKEN_OFFSET + enPassant;
    }


    /// <summary>
    /// python-chess's cleaned castling rights: a right counts only while the king is on e1/e8 and that rook in its corner.
    /// </summary>
    static bool CastlingPiecesHome(in MGPosition position, SideType side, bool kingside)
    {
      int rank = side == SideType.White ? 0 : 7;
      Piece king = position.GetPieceAtSquare(Square.FromFileAndRank(4, rank));
      Piece rook = position.GetPieceAtSquare(Square.FromFileAndRank(kingside ? 7 : 0, rank));
      return king.Type == PieceType.King && king.Side == side
          && rook.Type == PieceType.Rook && rook.Side == side;
    }

    #endregion


    #region Decoding

    /// <summary>
    /// Converts the outputs of positions [rangeStart, rangeEnd) into the result buffers.
    /// </summary>
    [SkipLocalsInit]
    void DecodeOutputs(int rangeStart, int rangeEnd)
    {
      Span<MGMoveList> movesSpan = currentPositions.Moves.Span;
      float inverseTemperature = 1.0f / Options.PolicyTemperature;
      Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probsBuffer
        = stackalloc PolicyVectorCompressedInitializerFromProbs.ProbEntry[MAX_MOVES];

      for (int i = rangeStart; i < rangeEnd; i++)
      {
        // Value: expected score for the side to move, sharpened, as W/L.
        float score = outputValue[i];
        if (float.IsNaN(score))
        {
          throw new Exception($"Fry backend returned NaN value from evaluator {this}");
        }
        float v = 2 * Sharpen(score, ValueSharpness) - 1;
        w[i] = (FP16)MathF.Max(0, v);
        l[i] = (FP16)MathF.Max(0, -v);

        // Policy: gather the legal moves' logits (Fry table), store under their Lc0 index.
        MGMoveList legal = movesSpan[i];
        int count = legal == null ? 0 : Math.Min(MAX_MOVES, legal.NumMovesUsed);
        if (count == 0)
        {
          policies[i] = default;
          continue;
        }

        MGMove[] moves = legal.MovesArray;
        bool blackToMove = moves[0].BlackToMove;
        int logitBase = i * NUM_POLICY;
        Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs = probsBuffer.Slice(0, count);
        for (int j = 0; j < count; j++)
        {
          int fryIndex = FryMoveTable.IndexOf(moves[j]);
          float logit = fryIndex < 0 ? MISSING_MOVE_LOGIT : outputPolicy[logitBase + fryIndex] * inverseTemperature;
          if (float.IsNaN(logit))
          {
            throw new Exception($"Fry backend returned NaN in policy from evaluator {this}");
          }

          EncodedMove encodedMove = blackToMove ? ConverterMGMoveEncodedMove.MGChessMoveToEncodedMoveBlack(moves[j])
                                                : ConverterMGMoveEncodedMove.MGChessMoveToEncodedMoveWhite(moves[j]);
          probs[j] = new PolicyVectorCompressedInitializerFromProbs.ProbEntry((short)encodedMove.IndexNeuralNet, logit);
        }

        int numMovesToSave = Math.Min(CompressedPolicyVector.NUM_MOVE_SLOTS, count);
        PolicyVectorCompressedInitializerFromProbs.InitializeFromLogitProbsArray(ref policies[i], blackToMove ? SideType.Black : SideType.White,
                                                                                 count, numMovesToSave, probs);
      }
    }


    /// <summary>
    /// Fry's value sharpening: sigmoid(sharpness * logit(score)).
    /// </summary>
    public static float Sharpen(float score, float sharpness)
    {
      if (sharpness == 1 || score <= 0 || score >= 1)
      {
        return Math.Clamp(score, 0, 1);
      }
      return 1 / (1 + MathF.Pow((1 - score) / score, sharpness));
    }

    #endregion


    protected override void DoShutdown()
    {
      lock (evalLockObj)
      {
        session.Dispose();
        runOptions.Dispose();
      }
    }


    public override string ToString()
      => $"<NNEvaluatorFry GPU {GPUID} {(UseTensorRT ? "TensorRT" : "CUDA")} {Path.GetFileName(ONNXFileName)} "
       + $"sharpness {ValueSharpness} max {maxBatchSize}>";
  }
}
