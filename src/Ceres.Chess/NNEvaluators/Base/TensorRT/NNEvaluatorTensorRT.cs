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
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;
using Ceres.Base.Threading;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNBackends.ONNXRuntime;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Microsoft.ML.OnnxRuntime;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// TensorRT-based neural network evaluator for Ceres.
///
/// Supports:
/// - TPG (Ceres) networks with Half (FP16) or Byte (INT8) inputs
/// - LC0 (Leela Chess Zero) networks with Half (FP16) plane inputs
/// - Multi-GPU with intelligent batch splitting
/// - Exact batch size engines for optimal performance
/// </summary>
public class NNEvaluatorTensorRT : NNEvaluator
{
  const bool USE_HISTORY = true;
  const bool SUBSTITUTE_VALUE3_INTO_VALUE2_IF_FOUND = false;

  /// <summary>
  /// If true, loads TensorRT engines in parallel across GPUs.
  /// </summary>
  public const bool PARALLEL_ENGINE_LOAD_ENABLED = false; // Needs more testing


  /// <summary>
  /// Path to the ONNX model file.
  /// </summary>
  public readonly string ONNXFileName;

  /// <summary>
  /// Type of neural network (LC0 or TPG).
  /// </summary>
  public readonly ONNXNetExecutor.NetTypeEnum NetType;

  public readonly EnginePoolMode PoolMode;
  public readonly int[] BatchSizes;

  /// <summary>
  /// GPU IDs used by this evaluator.
  /// </summary>
  public readonly int[] GpuIDs;

  private readonly TensorRT trt;
  private MultiGPUEnginePool pool;
  private OutputTensorInfo[] outputInfos;
  private int largestEngineBatchSize;

  // Network outputs layout - sizes are per-position
  private int valueSize;
  private int value2Size;
  private int value3Size;
  private int policySize;
  private int mlhSize;
  private int uncVSize;
  private int uncPSize;

  // Tensor indices in output (for computing per-engine offsets)
  private int valueTensorIndex;
  private int value2TensorIndex;
  private int value3TensorIndex;
  private int policyTensorIndex;
  private int mlhTensorIndex;
  private int uncVTensorIndex;
  private int uncPTensorIndex;
  private int pieceMoveTensorIndex = -1;
  private int pieceCaptureTensorIndex = -1;
  private int punimSelfTensorIndex = -1;
  private int punimOpponentTensorIndex = -1;

  // Per-position sizes for ply-bin tensors
  private int pieceMoveSizePerPos;
  private int pieceCaptureSizePerPos;
  private bool hasPlyBinOutputs;

  // Per-position sizes for PUNIM tensors
  private int punimSelfSizePerPos;
  private int punimOpponentSizePerPos;
  private bool hasPunimOutputs;

  private readonly int inputElementsPerPosition;
  private readonly int outputElementsPerPosition;

  // Input mode: byte inputs or Half inputs with /100 normalization
  private readonly bool useByteInputs;

  // Shared buffers for TPG encoding
  private byte[] squareByteBuffer;
  private byte[] inputByteBuffer;
  private Half[] inputHalfBuffer;
  private int maxBatchSize;

  // Thread-static output buffer for parallel multi-GPU processing.
  // Each GPU thread gets its own buffer, eliminating lock contention.
  [ThreadStatic]
  private static float[] threadLocalOutputFloatBuffer;
  private int outputFloatBufferSize; // Required size, stored for thread-local allocation

  // Pre-allocated ply-bin result buffers
  private Half[] plyBinMoveBuffer;
  private Half[] plyBinCaptureBuffer;

  // Pre-allocated PUNIM result buffers
  private Half[] punimSelfBuffer;
  private Half[] punimOpponentBuffer;

  // Pre-allocated result buffers to reduce GC pressure
  private FP16[] wBuffer;
  private FP16[] lBuffer;
  private FP16[] w2Buffer;
  private FP16[] l2Buffer;
  private FP16[] blendedWBuffer;
  private FP16[] blendedLBuffer;
  private FP16[] mBuffer;
  private FP16[] uncVBuffer;
  private FP16[] uncPBuffer;
  private CompressedPolicyVector[] policiesBuffer;
  private ParallelOptions cachedParallelOptions;

  // Network capabilities (determined from output tensors)
  private readonly bool isWDL;
  private readonly bool hasM;
  private readonly bool hasUncertaintyV;
  private readonly bool hasUncertaintyP;
  private readonly bool hasValueSecondary;

  // Warmup tracking (static lock ensures only one Warmup runs at a time across all instances)
  private bool haveWarmedUp;

  /// <inheritdoc/>
  public override bool IsWDL => isWDL;

  /// <inheritdoc/>
  public override bool HasM => hasM;

  /// <inheritdoc/>
  public override bool HasUncertaintyV => hasUncertaintyV;

  /// <inheritdoc/>
  public override bool HasUncertaintyP => hasUncertaintyP;

  /// <inheritdoc/>
  public override bool HasAction => false;

  /// <inheritdoc/>
  public override bool HasValueSecondary => hasValueSecondary;

  /// <inheritdoc/>
  public override int MaxBatchSize => maxBatchSize;

  public override InputTypes InputsRequired => InputTypes.Positions | InputTypes.Boards | InputTypes.Moves | (HasState ? InputTypes.State : 0);



  /// <summary>
  /// Creates evaluator with pool of exact batch size engines for intelligent batch splitting.
  /// Uses MultiGPUEnginePool for multi-GPU support and optimized batch handling.
  /// </summary>
  /// <param name="onnxFileName">Path to ONNX model file</param>
  /// <param name="netType">Type of neural network (LC0 or TPG)</param>
  /// <param name="poolMode">Engine pool mode (Exact or Range)</param>
  /// <param name="batchSizes">Array of exact batch sizes for engines</param>
  /// <param name="gpuIDs">GPU IDs to use, defaults to [0]</param>
  /// <param name="useCudaGraphs">Enable CUDA graphs for faster inference</param>
  /// <param name="softMaxBatchSize">Soft max batch size (can exceed largest engine via splitting)</param>
  public NNEvaluatorTensorRT(string onnxFileName,
                             ONNXNetExecutor.NetTypeEnum netType,
                             EnginePoolMode poolMode,
                             int[] batchSizes,
                             int[] gpuIDs = null,
                             bool useCudaGraphs = false,
                             int softMaxBatchSize = 0,
                             int optimizationLevel = 3)
    : this(onnxFileName, netType, poolMode, batchSizes, null, gpuIDs, useCudaGraphs, softMaxBatchSize, optimizationLevel, forceBF16: false, refittable: false)
  {
  }


  /// <summary>
  /// Creates evaluator with pool of exact batch size engines for intelligent batch splitting.
  /// Uses MultiGPUEnginePool for multi-GPU support and optimized batch handling.
  /// </summary>
  /// <param name="onnxFileName">Path to ONNX model file</param>
  /// <param name="netType">Type of neural network (LC0 or TPG)</param>
  /// <param name="poolMode">Engine pool mode (Exact or Range)</param>
  /// <param name="batchSizes">Array of exact batch sizes for engines</param>
  /// <param name="buildOptions">Optional TensorRT build options (null uses default BF16)</param>
  /// <param name="gpuIDs">GPU IDs to use, defaults to [0]</param>
  /// <param name="useCudaGraphs">Enable CUDA graphs for faster inference</param>
  /// <param name="softMaxBatchSize">Soft max batch size (can exceed largest engine via splitting)</param>
  public NNEvaluatorTensorRT(string onnxFileName,
                             ONNXNetExecutor.NetTypeEnum netType,
                             EnginePoolMode poolMode,
                             int[] batchSizes,
                             TensorRTBuildOptions? buildOptions,
                             int[] gpuIDs = null,
                             bool useCudaGraphs = false,
                             int softMaxBatchSize = 0,
                             int optimizationLevel = 3,
                             bool forceBF16 = false,
                             bool refittable = false)
  {
    if (!File.Exists(onnxFileName))
    {
      throw new FileNotFoundException($"ONNX model file not found: {onnxFileName}");
    }

    ONNXFileName = onnxFileName;
    NetType = netType;
    PoolMode = poolMode;
    BatchSizes = batchSizes;
    GpuIDs = gpuIDs ?? [0];
    EngineNetworkID = System.IO.Path.GetFileNameWithoutExtension(onnxFileName);

    // Build per-GPU sizes: use provided sizesPerGPU or replicate batchSizes for all GPUs
    int[][] effectiveSizesPerGPU = new int[GpuIDs.Length][];
    for (int i = 0; i < GpuIDs.Length; i++)
    {
      effectiveSizesPerGPU[i] = AdjustToSM(GpuIDs[i], batchSizes);
    }

    string cacheDir = ONNXExecutor.GetTRTEngineCacheDir(onnxFileName);

    trt = TensorRT.Instance;

    if (!string.IsNullOrEmpty(cacheDir))
    {
      System.IO.Directory.CreateDirectory(cacheDir);
    }

    // Compute max batch size for soft limit (pre-pool creation estimate)
    int maxEngineBatchSizeAll = poolMode == EnginePoolMode.Range
                                          ? effectiveSizesPerGPU.Max(s => s[^1])
                                          : effectiveSizesPerGPU.Max(s => s.Max());
    maxBatchSize = softMaxBatchSize > 0 ? softMaxBatchSize : maxEngineBatchSizeAll;

    string graphsLabel = useCudaGraphs ? " [CUDA Graphs]" : "";
    Console.WriteLine($"Creating NNEvaluatorTensorRT for {onnxFileName}{graphsLabel}");
    Console.WriteLine($"  GPUs: [{string.Join(", ", GpuIDs)}]  Batch sizes: [{string.Join(", ", batchSizes)}]");

    TensorRTBuildOptions options;
    if (buildOptions.HasValue)
    {
      options = buildOptions.Value;
      options.UseCudaGraphs = useCudaGraphs ? 1 : 0;
    }
    else
    {
      options = TensorRTBuildOptions.Default;
      options.BuilderOptimizationLevel = optimizationLevel;
      if (optimizationLevel == 5)
      {
        options.TilingOptimizationLevel = 3; // Also do tiling search, but this can be slow.
      }
      options.UseFP16 = 1;
      options.UseBF16 = 0;
      options.FP32PostAttentionNorm = 1;

      // NOTE: BF16 is best on datacenter cards (works for all nets, no upcasting required)
      //       Consumer cards can use FP32+post attention norm which is high accuraacy for most nets
      if (netType == ONNXNetExecutor.NetTypeEnum.TPG)
      {
        //options.FP32SmolgenNorm = 1;
      }
      options.UseCudaGraphs = useCudaGraphs ? 1 : 0;
    }

    if (forceBF16)
    {
      options.UseBF16 = 1;
      options.UseFP16 = 0;
      options.FP32PostAttentionNorm = 0;
      options.FP32PostAttentionNormStrict = 0;
      options.FP32SmolgenNorm = 0;
    }

    if (refittable)
    {
      options.Refittable = 1;
    }

    options.Validate();

    Console.WriteLine($"  Build options: FP16={options.UseFP16}, BF16={options.UseBF16}, FP32PostAttentionNorm={options.FP32PostAttentionNorm}, UseCUDAGraphs={options.UseCudaGraphs}");

    const int MIN_BATCH_SIZE_PER_GPU = 6;
    pool = new MultiGPUEnginePool(trt, onnxFileName, effectiveSizesPerGPU, poolMode, options, 0, 0, GpuIDs, MIN_BATCH_SIZE_PER_GPU, cacheDir);

    // Query actual max batch size from pool AFTER construction (may differ if optimization rebuilt engines)
    largestEngineBatchSize = pool.MaxEngineBatchSize;

    inputElementsPerPosition = pool.InputElementsPerPosition;
    outputElementsPerPosition = pool.OutputElementsPerPosition;
    outputInfos = pool.GetOutputTensorInfo();
    useByteInputs = pool.UseByteInputs;

    string inputTensorName = pool.GetInputName(0);

    int tensorIndex = 0;
    foreach (OutputTensorInfo info in outputInfos)
    {
      int sizePerPos = (int)(info.Size / largestEngineBatchSize);

      // Normalize output name: strip "/output/" prefix (LC0 format) and map "wdl" to "value"
      string name = info.Name.ToLower();
      if (name.StartsWith("/output/"))
      {
        name = name["/output/".Length..];
      }
      if (name == "wdl")
      {
        name = "value";
      }

      switch (name)
      {
        case "value":
          valueTensorIndex = tensorIndex;
          valueSize = sizePerPos;
          break;
        case "value2":
          value2TensorIndex = tensorIndex;
          value2Size = sizePerPos;
          break;
        case "value3":
          value3TensorIndex = tensorIndex;
          value3Size = sizePerPos;
          break;
        case "policy":
          policyTensorIndex = tensorIndex;
          policySize = sizePerPos;
          break;
        case "mlh":
          mlhTensorIndex = tensorIndex;
          mlhSize = sizePerPos;
          break;
        case "unc":
        case "uncertainty_v":
        case "unc_v":
          uncVTensorIndex = tensorIndex;
          uncVSize = sizePerPos;
          break;
        case "uncertainty_p":
        case "unc_p":
          uncPTensorIndex = tensorIndex;
          uncPSize = sizePerPos;
          break;
        case "piece_move":
          pieceMoveTensorIndex = tensorIndex;
          pieceMoveSizePerPos = sizePerPos;
          break;
        case "piece_capture":
          pieceCaptureTensorIndex = tensorIndex;
          pieceCaptureSizePerPos = sizePerPos;
          break;
        case "punim_self":
          punimSelfTensorIndex = tensorIndex;
          punimSelfSizePerPos = sizePerPos;
          break;
        case "punim_opponent":
          punimOpponentTensorIndex = tensorIndex;
          punimOpponentSizePerPos = sizePerPos;
          break;
      }
      tensorIndex++;
    }

    isWDL = valueSize == 3;
    hasM = mlhSize > 0;
    hasUncertaintyV = uncVSize > 0;
    hasUncertaintyP = uncPSize > 0;
    hasValueSecondary = value2Size > 0 || (SUBSTITUTE_VALUE3_INTO_VALUE2_IF_FOUND && value3Size > 0);
    hasPlyBinOutputs = pieceMoveSizePerPos == 512 && pieceCaptureSizePerPos == 512;
    hasPunimOutputs = punimSelfSizePerPos == 8 && punimOpponentSizePerPos == 8;

    if (netType == ONNXNetExecutor.NetTypeEnum.TPG)
    {
      int bytesPerSquareRecord = TPGRecord.BYTES_PER_SQUARE_RECORD;
      squareByteBuffer = new byte[maxBatchSize * 64 * bytesPerSquareRecord];
    }

    if (useByteInputs)
    {
      inputByteBuffer = new byte[maxBatchSize * inputElementsPerPosition];
    }
    else
    {
      inputHalfBuffer = new Half[maxBatchSize * inputElementsPerPosition];
    }

    outputFloatBufferSize = (int)pool.MaxTotalOutputSize;

    // Pre-allocate result buffers
    wBuffer = new FP16[maxBatchSize];
    lBuffer = new FP16[maxBatchSize];
    w2Buffer = hasValueSecondary ? new FP16[maxBatchSize] : Array.Empty<FP16>();
    l2Buffer = hasValueSecondary ? new FP16[maxBatchSize] : Array.Empty<FP16>();
    blendedWBuffer = hasValueSecondary ? new FP16[maxBatchSize] : Array.Empty<FP16>();
    blendedLBuffer = hasValueSecondary ? new FP16[maxBatchSize] : Array.Empty<FP16>();
    mBuffer = hasM ? new FP16[maxBatchSize] : Array.Empty<FP16>();
    uncVBuffer = hasUncertaintyV ? new FP16[maxBatchSize] : Array.Empty<FP16>();
    uncPBuffer = hasUncertaintyP ? new FP16[maxBatchSize] : Array.Empty<FP16>();
    policiesBuffer = new CompressedPolicyVector[maxBatchSize];
    plyBinMoveBuffer = hasPlyBinOutputs ? new Half[maxBatchSize * 512] : Array.Empty<Half>();
    plyBinCaptureBuffer = hasPlyBinOutputs ? new Half[maxBatchSize * 512] : Array.Empty<Half>();
    punimSelfBuffer = hasPunimOutputs ? new Half[maxBatchSize * 8] : Array.Empty<Half>();
    punimOpponentBuffer = hasPunimOutputs ? new Half[maxBatchSize * 8] : Array.Empty<Half>();

    // Cache ParallelOptions to avoid allocation per batch
    cachedParallelOptions = new ParallelOptions
    {
      MaxDegreeOfParallelism = ParallelUtils.CalcMaxParallelism(maxBatchSize, 32)
    };

    // Important to capture possible CUDA graphs and timing statistics up front
    Warmup();
  }

  private static readonly object warmupLock = new();

  /// <summary>
  /// Performs warmup by running dummy evaluations at each defined batch size.
  /// Only executes on first call; subsequent calls are no-ops.
  /// </summary>

  public override void Warmup()
  {
    if (haveWarmedUp)
    {
      return;
    }

    lock (warmupLock)
    {
      if (haveWarmedUp)
      {
        return;
      }

      // Determine batch sizes to warm up based on pool mode
      int[] warmupSizes;
      if (PoolMode == EnginePoolMode.Range)
      {
        // For Range mode, warm up at each range maximum
        warmupSizes = BatchSizes;
      }
      else
      {
        // For Exact mode, warm up at each exact batch size
        warmupSizes = BatchSizes;
      }

      // Allocate dummy input buffers for warmup
      int maxWarmupSize = warmupSizes.Max();
      Half[] dummyOutputBuffer = new Half[maxWarmupSize * outputElementsPerPosition];

      foreach (int batchSize in warmupSizes)
      {
        if (useByteInputs)
        {
          byte[] dummyInput = new byte[batchSize * inputElementsPerPosition];
          pool.ProcessBytes(dummyInput, dummyOutputBuffer, batchSize);
        }
        else
        {
          Half[] dummyInput = new Half[batchSize * inputElementsPerPosition];
          pool.Process(dummyInput, dummyOutputBuffer, batchSize);
        }
      }

      haveWarmedUp = true;
    }
  }


  /// <summary>
  /// Converter function for converting positions to flat TPG format.
  /// Must be set before evaluation.
  /// </summary>
  public Action<NNEvaluatorOptions, IEncodedPositionBatchFlat, bool, Memory<byte>, Memory<Half>, short[]> ConverterToFlat { get; set; }
  public Func<NNEvaluatorOptions, object, Memory<byte>, int> ConverterToFlatFromTPG = null;


  /// <inheritdoc/>
  protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat batch, bool retrieveSupplementalResults = false)
  {
    int numPos = batch.NumPos;
    if (numPos > maxBatchSize)
    {
      throw new ArgumentException($"Batch size {numPos} exceeds maximum {maxBatchSize}");
    }

    if (NetType == ONNXNetExecutor.NetTypeEnum.LC0)
    {
      // LC0 path: convert positions to 112-plane format directly into inputHalfBuffer
      Memory<Half> inputBuffer = new Memory<Half>(inputHalfBuffer, 0, numPos * inputElementsPerPosition);
      batch.ConvertValuesToFlatFromPlanes(inputBuffer, false, true);
    }
    else
    {
      // TPG path: convert positions to flat TPG byte format
      if (ConverterToFlat == null)
      {
        throw new InvalidOperationException("ConverterToFlat must be set before evaluation for TPG networks");
      }

      Memory<byte> byteBuffer = new Memory<byte>(squareByteBuffer, 0, numPos * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);
      Memory<Half> emptyHalf = Memory<Half>.Empty;
      ConverterToFlat(Options, batch, USE_HISTORY, byteBuffer, emptyHalf, null);
    }

    return ProcessBatchWithPool(batch, numPos);
  }


  /// <summary>
  /// Process batch using MultiGPUEnginePool with callback-based tensor-major extraction.
  /// </summary>
  private PositionEvaluationBatch ProcessBatchWithPool(IEncodedPositionBatchFlat batch, int numPos)
  {
    // Reuse pre-allocated buffers (sized to maxBatchSize)
    FP16[] w = wBuffer;
    FP16[] l = lBuffer;
    FP16[] w2 = w2Buffer;
    FP16[] l2 = l2Buffer;
    FP16[] m = mBuffer;
    FP16[] uncV = uncVBuffer;
    FP16[] uncP = uncPBuffer;
    CompressedPolicyVector[] policies = policiesBuffer;
    CompressedActionVector[] actions = Array.Empty<CompressedActionVector>();
    Half[] plyBinMove = plyBinMoveBuffer;
    Half[] plyBinCapture = plyBinCaptureBuffer;
    Half[] punimSelf = punimSelfBuffer;
    Half[] punimOpponent = punimOpponentBuffer;

    // Get option values for value head temperature
    float valueHead1Temperature = Options?.ValueHead1Temperature ?? 1.0f;
    float valueHead2Temperature = Options?.ValueHead2Temperature ?? 1.0f;
    float valueHead1TemperatureScaling = Options?.Value1UncertaintyTemperatureScalingFactor ?? 0.0f;
    float valueHead2TemperatureScaling = Options?.Value2UncertaintyTemperatureScalingFactor ?? 0.0f;

    // Capture buffer size for thread-local allocation in handler
    int requiredBufferSize = outputFloatBufferSize;

    SubBatchOutputHandler handler = (int globalStartPosition, int positionCount, int engineBatchSize, Half[] rawOutput) =>
    {
      // Ensure thread-local buffer is allocated (each GPU thread gets its own buffer)
      if (threadLocalOutputFloatBuffer == null || threadLocalOutputFloatBuffer.Length < requiredBufferSize)
      {
        threadLocalOutputFloatBuffer = new float[requiredBufferSize];
      }

      // Vectorized conversion from Half to float using thread-local buffer
      TensorPrimitives.ConvertToSingle(rawOutput, threadLocalOutputFloatBuffer.AsSpan(0, rawOutput.Length));

      ExtractSubBatchResults(batch, globalStartPosition, positionCount, engineBatchSize,
                             threadLocalOutputFloatBuffer, w, l, w2, l2, m, uncV, uncP, policies,
                             plyBinMove, plyBinCapture, punimSelf, punimOpponent,
                             valueHead1Temperature, valueHead2Temperature,
                             valueHead1TemperatureScaling, valueHead2TemperatureScaling,
                             NetType == ONNXNetExecutor.NetTypeEnum.TPG);
    };

    if (NetType == ONNXNetExecutor.NetTypeEnum.LC0)
    {
      // LC0: inputHalfBuffer already contains the encoded planes from DoEvaluateIntoBuffers
      pool.ProcessWithHandler(inputHalfBuffer, numPos, handler);
    }
    else if (useByteInputs)
    {
      // TPG byte path
      pool.ProcessBytesWithHandler(squareByteBuffer, numPos, handler);
    }
    else
    {
      // TPG half path: convert byte buffer to half buffer with /100 scaling
      int elemsToCopy = numPos * inputElementsPerPosition;
      Memory<byte> sourceBytes = new Memory<byte>(squareByteBuffer, 0, elemsToCopy);
      Memory<Half> targetHalfs = new Memory<Half>(inputHalfBuffer, 0, elemsToCopy);
      TPGConvertersToFlat.CopyAndDivideSIMD(sourceBytes, targetHalfs, 100.0f);
      pool.ProcessWithHandler(inputHalfBuffer, numPos, handler);
    }

    // Apply value head blending into separate buffers if FractionValueHead2 is specified
    // This preserves original w/l (head 1) values, matching ONNX behavior where W1/L1 remain accessible
    float fractionValueHead2 = Options?.FractionValueHead2 ?? 0.0f;
    FP16[] finalW, finalL;

    if (fractionValueHead2 > 0.0f && hasValueSecondary)
    {
      // Blend into pre-allocated buffers, preserving original head 1 values
      float fractionValueHead1 = 1.0f - fractionValueHead2;
      FP16[] blendedW = blendedWBuffer;
      FP16[] blendedL = blendedLBuffer;

      for (int i = 0; i < numPos; i++)
      {
        blendedW[i] = (FP16)((float)w[i] * fractionValueHead1 + (float)w2[i] * fractionValueHead2);
        blendedL[i] = (FP16)((float)l[i] * fractionValueHead1 + (float)l2[i] * fractionValueHead2);
      }
      finalW = blendedW;
      finalL = blendedL;
    }
    else
    {
      // No blending - use head 1 values directly
      finalW = w;
      finalL = l;
    }

    return new PositionEvaluationBatch(
      isWDL: IsWDL,
      hasM: HasM,
      hasUncertaintyV: HasUncertaintyV,
      hasUncertaintyP: hasUncertaintyP,
      hasAction: false,
      hasValueSecondary: hasValueSecondary,
      hasState: false,
      numPos: numPos,
      policies: policies,
      actionProbabilties: actions,
      w: finalW,
      l: finalL,
      w2: w2,
      l2: l2,
      m: m,
      uncertaintyV: uncV,
      uncertaintyP: uncP,
      states: default,
      activations: default,
      stats: default,
      plyBinMoveProbs: hasPlyBinOutputs ? plyBinMove : default,
      plyBinCaptureProbs: hasPlyBinOutputs ? plyBinCapture : default,
      punimSelfProbs: hasPunimOutputs ? punimSelf : default,
      punimOpponentProbs: hasPunimOutputs ? punimOpponent : default);
  }


  /// <summary>
  /// Computes inverse hyperbolic tangent (artanh).
  /// </summary>
  private static float Atanh(float x) => 0.5f * MathF.Log((1f + x) / (1f - x));


  /// <summary>
  /// Applies softmax-based temperature scaling to WDL logits.
  /// </summary>
  private static (float W, float L) ApplySoftmaxTemperature(float vW, float vD, float vL, float invTemp)
  {
    float maxV = MathF.Max(vW, MathF.Max(vD, vL));
    float expW = MathF.Exp((vW - maxV) * invTemp);
    float expD = MathF.Exp((vD - maxV) * invTemp);
    float expL = MathF.Exp((vL - maxV) * invTemp);
    float sumV = expW + expD + expL;
    return (expW / sumV, expL / sumV);
  }


  /// <summary>
  /// Applies temperature scaling to a WDL (win/draw/loss) distribution,
  /// compressing or sharpening the value (W-L) while preserving the draw probability.
  ///
  /// Temperature operates in arctanh (logit) space on V = W - L, then redistributes
  /// W and L symmetrically around the fixed draw mass.
  ///
  /// Parameters:
  ///   w, d, l      - Original win/draw/loss probabilities (should sum to 1).
  ///   temperature  - Temperature. t=1 returns original values unchanged.
  ///                  t>1 compresses V toward 0 (more uncertain).
  ///                  t&lt;1 pushes V toward ±1 (more decisive).
  ///   drawFactor   - Controls how much the draw probability resists temperature [0, 1].
  ///                  0: drawish positions are heavily resistant to temperature changes,
  ///                     preserving the natural buffering effect of high draw probability.
  ///                  1: temperature acts with full force regardless of draw probability,
  ///                     as if operating purely in value space.
  ///                  Values in between blend smoothly between these two extremes.
  /// </summary>
  private static (float W, float L) ApplyTanhTemperature(float w, float d, float l, float temperature, float drawFactor)
  {
    float v = Math.Clamp(w - l, -0.9999f, 0.9999f);

    // Draws resist temperature change, modulated by drawFactor
    // drawFactor=1: full bypass (temperature acts with full force)
    // drawFactor=0: draws fully dampen temperature effect
    float damping = MathF.Pow(1f - d, 1f - drawFactor);
    float tEff = 1f + (temperature - 1f) * damping;

    float vt = MathF.Tanh(Atanh(v) / tEff);

    // Preserve original draw, redistribute W/L to hit new V
    float delta = 1f - d;
    float newW = (delta + vt) / 2f;
    float newL = (delta - vt) / 2f;
    return (newW, newL);
  }


  /// <summary>
  /// Extracts WDL values from logits and applies appropriate temperature scaling.
  /// When temperatureScaling > 0: uses softmax with dynamic temperature based on uncertainty.
  /// When temperatureScaling < 0: uses tanh-based scaling with dynamic temperature based on uncertainty.
  /// When temperatureScaling == 0: uses softmax with fixed base temperature.
  /// </summary>
  private static (float W, float L) ExtractAndScaleWDL(float vW, float vD, float vL,
                                                        float baseTemperature, float temperatureScaling,
                                                        float uncertaintyV, bool wdlIsLogistic)
  {
    if (temperatureScaling < 0)
    {
      // Tanh-based scaling: first convert logits to probabilities, then apply tanh scaling
      float effectiveTemp = baseTemperature + uncertaintyV * (-temperatureScaling);

      // Convert logits to probabilities first (softmax with temp=1)
      float maxV = MathF.Max(vW, MathF.Max(vD, vL));
      float expW = MathF.Exp(vW - maxV);
      float expD = MathF.Exp(vD - maxV);
      float expL = MathF.Exp(vL - maxV);
      float sumV = expW + expD + expL;
      float probW = expW / sumV;
      float probD = expD / sumV;
      float probL = expL / sumV;

      const float DRAW_FACTOR = 0.3f;
      return ApplyTanhTemperature(probW, probD, probL, effectiveTemp, DRAW_FACTOR);
    }
    else
    {
      // Softmax-based scaling
      float effectiveTemp = temperatureScaling > 0
                          ? baseTemperature + uncertaintyV * temperatureScaling
                          : baseTemperature;
      float invTemp = 1.0f / effectiveTemp;
      return ApplySoftmaxTemperature(vW, vD, vL, invTemp);
    }
  }


  /// <summary>
  /// Extract results from a sub-batch output buffer into the result arrays.
  /// Uses parallel processing for all per-position extractions (values, policies, etc.)
  /// for improved performance, matching the pattern used in the ONNX backend.
  /// </summary>
  private void ExtractSubBatchResults(IEncodedPositionBatchFlat batch, int startPos, int count, int engineBatchSize,
                                      float[] subBatchOutput,
                                      FP16[] w, FP16[] l, FP16[] w2, FP16[] l2, FP16[] m, FP16[] uncV, FP16[] uncP,
                                      CompressedPolicyVector[] policies,
                                      Half[] plyBinMove, Half[] plyBinCapture,
                                      Half[] punimSelf, Half[] punimOpponent,
                                      float valueHead1Temperature, float valueHead2Temperature,
                                      float valueHead1TemperatureScaling, float valueHead2TemperatureScaling,
                                      bool wdlIsLogistic)
  {
    // Compute tensor offsets once outside the parallel loop
    int valueOffset = 0;
    int value2Offset = 0;
    int value3Offset = 0;
    int policyOffset = 0;
    int mlhOffset = 0;
    int uncVOffset = 0;
    int uncPOffset = 0;
    int pieceMoveOffset = 0;
    int pieceCaptureOffset = 0;
    int punimSelfOffset = 0;
    int punimOpponentOffset = 0;

    int currentOffset = 0;
    for (int t = 0; t < outputInfos.Length; t++)
    {
      int sizePerPos = (int)(outputInfos[t].Size / largestEngineBatchSize);
      int tensorSize = engineBatchSize * sizePerPos;

      if (t == valueTensorIndex)
      {
        valueOffset = currentOffset;
      }
      else if (t == value2TensorIndex)
      {
        value2Offset = currentOffset;
      }
      else if (t == value3TensorIndex)
      {
        value3Offset = currentOffset;
      }
      else if (t == policyTensorIndex)
      {
        policyOffset = currentOffset;
      }
      else if (t == mlhTensorIndex)
      {
        mlhOffset = currentOffset;
      }
      else if (t == uncVTensorIndex)
      {
        uncVOffset = currentOffset;
      }
      else if (t == uncPTensorIndex)
      {
        uncPOffset = currentOffset;
      }
      else if (t == pieceMoveTensorIndex)
      {
        pieceMoveOffset = currentOffset;
      }
      else if (t == pieceCaptureTensorIndex)
      {
        pieceCaptureOffset = currentOffset;
      }
      else if (t == punimSelfTensorIndex)
      {
        punimSelfOffset = currentOffset;
      }
      else if (t == punimOpponentTensorIndex)
      {
        punimOpponentOffset = currentOffset;
      }

      // Align to 256-byte boundary (128 fp16 elements) to match
      // OUTPUT_TENSOR_ALIGN_ELEMS in TensorRTWrapper.cpp.
      // Required for CUDA graphs: small tensors (e.g. 8-element PUNIM)
      // would otherwise cause misaligned addresses during graph capture.
      const int ALIGN = 128;
      currentOffset += (tensorSize + ALIGN - 1) / ALIGN * ALIGN;
    }

    // Substitute value3 into value2 if enabled and value3 is present
    int effectiveValue2Offset = (SUBSTITUTE_VALUE3_INTO_VALUE2_IF_FOUND && value3Size > 0) ? value3Offset : value2Offset;

    // Get policy temperature once
    float policyTemperature = Options?.PolicyTemperature ?? 1.0f;

    // Capture batch data for parallel access
    ReadOnlyMemory<MGMoveList> moves = batch.Moves;
    ReadOnlyMemory<MGPosition> positions = batch.Positions;

    // Single parallel loop extracts all per-position results: values, M, uncertainties, and policies.
    // This improves cache locality and parallelism compared to separate sequential + parallel passes.
    Parallel.For(0, count, new ParallelOptions() { MaxDegreeOfParallelism = 1 + count / 48 }, /*cachedParallelOptions,*/ i =>
    {
      int resultIndex = startPos + i;

      // ===== Extract MLH =====
      if (hasM)
      {
        int posMlhOffset = mlhOffset + i * mlhSize;
        m[resultIndex] = (FP16)MathF.Max(subBatchOutput[posMlhOffset] * 100, 0); // NetTransformer.MLH_DIVISOR = 100
      }

      // ===== Extract Uncertainty V =====
      float uncertaintyV = 0;
      if (hasUncertaintyV)
      {
        int posUncVOffset = uncVOffset + i * uncVSize;
        uncertaintyV = subBatchOutput[posUncVOffset];
        uncV[resultIndex] = (FP16)uncertaintyV;
      }

      // ===== Extract Uncertainty P =====
      if (hasUncertaintyP)
      {
        int posUncPOffset = uncPOffset + i * uncPSize;
        uncP[resultIndex] = (FP16)subBatchOutput[posUncPOffset];
      }

      // ===== Extract Value Head 1 =====
      int posValueOffset = valueOffset + i * valueSize;
      if (isWDL)
      {
        if (wdlIsLogistic)
        {
          float vW = subBatchOutput[posValueOffset];
          float vD = subBatchOutput[posValueOffset + 1];
          float vL = subBatchOutput[posValueOffset + 2];

          (float wVal, float lVal) = ExtractAndScaleWDL(vW, vD, vL,
                                                         valueHead1Temperature, valueHead1TemperatureScaling,
                                                         uncertaintyV, wdlIsLogistic);
          w[resultIndex] = (FP16)wVal;
          l[resultIndex] = (FP16)lVal;
        }
        else
        {
          w[resultIndex] = (FP16)subBatchOutput[posValueOffset];
          l[resultIndex] = (FP16)subBatchOutput[posValueOffset + 2];
        }
      }
      else
      {
        float v = subBatchOutput[posValueOffset];
        w[resultIndex] = (FP16)((v + 1) * 0.5f);
        l[resultIndex] = (FP16)((1 - v) * 0.5f);
      }

      // ===== Extract Value Head 2 (if present) =====
      if (hasValueSecondary)
      {
        int posValue2Offset = effectiveValue2Offset + i * value2Size;
        float vW2 = subBatchOutput[posValue2Offset];
        float vD2 = subBatchOutput[posValue2Offset + 1];
        float vL2 = subBatchOutput[posValue2Offset + 2];

        (float w2Val, float l2Val) = ExtractAndScaleWDL(vW2, vD2, vL2,
                                                         valueHead2Temperature, valueHead2TemperatureScaling,
                                                         uncertaintyV, wdlIsLogistic: true);
        w2[resultIndex] = (FP16)w2Val;
        l2[resultIndex] = (FP16)l2Val;
      }

      // ===== Extract Policy =====
      int batchIndex = startPos + i;
      int posPolicyOffset = policyOffset + i * policySize;

      MGMoveList moveList = moves.Span[batchIndex];
      int numMoves = moveList.NumMovesUsed;
      if (numMoves == 0)
      {
        return;
      }

      ReadOnlySpan<float> policyLogits = subBatchOutput.AsSpan().Slice(posPolicyOffset, policySize);

      // Collect move indices and find max logit in a single pass
      Span<int> indices = stackalloc int[numMoves];
      Span<float> logits = stackalloc float[numMoves];
      float maxLogit = float.NegativeInfinity;

      for (int mv = 0; mv < numMoves; mv++)
      {
        MGMove move = moveList.MovesArray[mv];
        EncodedMove encoded = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(move);
        int nnIndex = encoded.IndexNeuralNet;
        indices[mv] = nnIndex;

        if (nnIndex >= 0 && nnIndex < policySize)
        {
          float logit = policyLogits[nnIndex];
          logits[mv] = logit;
          if (logit > maxLogit)
          {
            maxLogit = logit;
          }
        }
        else
        {
          logits[mv] = float.NegativeInfinity;
        }
      }

      // Apply softmax using TensorPrimitives for vectorization
      TensorPrimitives.Subtract(logits, maxLogit, logits);

      if (policyTemperature != 1.0f)
      {
        float invPolicyTemp = 1.0f / policyTemperature;
        TensorPrimitives.Multiply(logits, invPolicyTemp, logits);
      }

      TensorPrimitives.Exp(logits, logits);

      float sum = TensorPrimitives.Sum(logits);
      if (sum > 0)
      {
        float invSum = 1.0f / sum;
        TensorPrimitives.Multiply(logits, invSum, logits);
      }

      SideType side = positions.Span[batchIndex].SideToMove;
      CompressedPolicyVector.Initialize(ref policies[resultIndex], side, indices, logits, alreadySorted: false);

      // ===== Extract Ply-Bin Outputs (if present) =====
      if (hasPlyBinOutputs)
      {
        int destOffset = resultIndex * 512;

        // piece_move: softmax per square (8 bins), reverse bin order
        int posPieceMoveOffset = pieceMoveOffset + i * pieceMoveSizePerPos;
        for (int sq = 0; sq < 64; sq++)
        {
          int srcBase = posPieceMoveOffset + sq * 8;
          int dstBase = destOffset + sq * 8;

          float max = subBatchOutput[srcBase];
          for (int b = 1; b < 8; b++)
          {
            float v = subBatchOutput[srcBase + b];
            if (v > max) max = v;
          }

          float expSum = 0;
          Span<float> exps = stackalloc float[8];
          for (int b = 0; b < 8; b++)
          {
            float e = MathF.Exp(subBatchOutput[srcBase + b] - max);
            exps[b] = e;
            expSum += e;
          }

          float invSum = 1.0f / expSum;
          for (int b = 0; b < 8; b++)
          {
            plyBinMove[dstBase + b] = (Half)(exps[7 - b] * invSum);
          }
        }

        // piece_capture: softmax per square (8 bins), reverse bin order
        int posPieceCaptureOffset = pieceCaptureOffset + i * pieceCaptureSizePerPos;
        for (int sq = 0; sq < 64; sq++)
        {
          int srcBase = posPieceCaptureOffset + sq * 8;
          int dstBase = destOffset + sq * 8;

          float max = subBatchOutput[srcBase];
          for (int b = 1; b < 8; b++)
          {
            float v = subBatchOutput[srcBase + b];
            if (v > max) max = v;
          }

          float expSum = 0;
          Span<float> exps = stackalloc float[8];
          for (int b = 0; b < 8; b++)
          {
            float e = MathF.Exp(subBatchOutput[srcBase + b] - max);
            exps[b] = e;
            expSum += e;
          }

          float invSum = 1.0f / expSum;
          for (int b = 0; b < 8; b++)
          {
            plyBinCapture[dstBase + b] = (Half)(exps[7 - b] * invSum);
          }
        }
      }

      // ===== Extract PUNIM Outputs (if present) =====
      if (hasPunimOutputs)
      {
        int punimDestOffset = resultIndex * 8;

        // punim_self: softmax of 8 bins, reverse bin order
        int posPunimSelfOffset = punimSelfOffset + i * punimSelfSizePerPos;
        {
          float max = subBatchOutput[posPunimSelfOffset];
          for (int b = 1; b < 8; b++)
          {
            float v = subBatchOutput[posPunimSelfOffset + b];
            if (v > max) max = v;
          }

          float expSum = 0;
          Span<float> exps = stackalloc float[8];
          for (int b = 0; b < 8; b++)
          {
            float e = MathF.Exp(subBatchOutput[posPunimSelfOffset + b] - max);
            exps[b] = e;
            expSum += e;
          }

          float invSum = 1.0f / expSum;
          for (int b = 0; b < 8; b++)
          {
            punimSelf[punimDestOffset + b] = (Half)(exps[7 - b] * invSum);
          }
        }

        // punim_opponent: softmax of 8 bins, reverse bin order
        int posPunimOpponentOffset = punimOpponentOffset + i * punimOpponentSizePerPos;
        {
          float max = subBatchOutput[posPunimOpponentOffset];
          for (int b = 1; b < 8; b++)
          {
            float v = subBatchOutput[posPunimOpponentOffset + b];
            if (v > max) max = v;
          }

          float expSum = 0;
          Span<float> exps = stackalloc float[8];
          for (int b = 0; b < 8; b++)
          {
            float e = MathF.Exp(subBatchOutput[posPunimOpponentOffset + b] - max);
            exps[b] = e;
            expSum += e;
          }

          float invSum = 1.0f / expSum;
          for (int b = 0; b < 8; b++)
          {
            punimOpponent[punimDestOffset + b] = (Half)(exps[7 - b] * invSum);
          }
        }
      }
    });
  }



  /// <summary>
  /// Creates an NNEvaluatorTensorRT from an NNEvaluatorDef.
  /// Supports multi-GPU configurations when multiple devices are specified.
  /// </summary>
  /// <param name="def">The evaluator definition containing network and device specifications.</param>
  /// <param name="options">Optional evaluator options (if null, extracted from def.Options).</param>
  /// <param name="gpuIDs">Optional GPU IDs to use (if null, extracted from def.Devices).</param>
  /// <returns>A configured NNEvaluatorTensorRT instance.</returns>
  public static NNEvaluatorTensorRT FromDefinition(NNEvaluatorDef def, NNEvaluatorOptions options = null, int[] gpuIDs = null)
  {
    // Validate multi-device configuration
    if (def.Devices.Length > 1)
    {
      // Verify all devices have TensorRTNative as OverrideEngineType
      for (int i = 0; i < def.Devices.Length; i++)
      {
        string engineType = def.Devices[i].Device.OverrideEngineType;
        if (engineType == null || !engineType.Contains("TensorRTNative", StringComparison.OrdinalIgnoreCase))
        {
          throw new Exception($"All devices must have OverrideEngineType of 'TensorRTNative' for multi-GPU TensorRT. Device {i} has: {engineType ?? "null"}");
        }
      }

      // Verify all nets are the same
      var firstNet = def.Nets[0].Net;
      for (int i = 1; i < def.Nets.Length; i++)
      {
        if (!def.Nets[i].Net.Equals(firstNet))
        {
          throw new Exception($"All nets must be identical for multi-GPU TensorRT. Net {i} differs from Net 0.");
        }
      }
    }

    // Extract GPU IDs from devices if not provided
    if (gpuIDs == null)
    {
      gpuIDs = def.DeviceIndices;
    }

    // Get network definition (all should be the same, use first)
    NNEvaluatorNetDef netDef = def.Nets[0].Net;

    // Get options from def if not provided
    options = options ?? def.Options;

    string overrideFN = null;
    if (netDef.Type != NNEvaluatorType.Ceres)
    {
      string pathLc0Networks = CeresUserSettingsManager.Settings.DirLC0Networks;
      overrideFN = Path.Combine(CeresUserSettingsManager.Settings.DirLC0Networks, netDef.NetworkID);
      if (!overrideFN.ToUpper().Contains("ONNX"))
      {
        overrideFN += ".onnx";
      }
    }
    NNEvaluatorTensorRT trtNativeEngine = BuildEvaluator(netDef, gpuIDs, options,
                                                         netDef.Type == NNEvaluatorType.Ceres ? ONNXNetExecutor.NetTypeEnum.TPG
                                                                                              : ONNXNetExecutor.NetTypeEnum.LC0,
                                                         overrideFN);

    return trtNativeEngine;
  }


  public static NNEvaluatorTensorRT BuildEvaluator(NNEvaluatorNetDef netDef,
                                                     int[] gpuIDs,
                                                     NNEvaluatorOptions options,
                                                     ONNXNetExecutor.NetTypeEnum netType = ONNXNetExecutor.NetTypeEnum.TPG,
                                                     string overrideFileName = null)
  {
    // Determine network file path
    string netFileName = overrideFileName ?? netDef.NetworkID;
    string extUpper = System.IO.Path.GetExtension(netFileName).ToUpper();
    if (extUpper != ".ONNX" && extUpper != ".ENGINE" && extUpper != ".PLAN")
    {
      netFileName += ".onnx";
    }

    if (!System.IO.File.Exists(netFileName))
    {
      netFileName = System.IO.Path.Combine(CeresUserSettingsManager.Settings.DirCeresNetworks, netFileName);
    }
    if (!System.IO.File.Exists(netFileName))
    {
      throw new Exception($"Ceres net {netFileName} not found. Use valid full path or set source directory using DirCeresNetworks in Ceres.json");
    }

    //    if (optionsCeres.HeadOverrides != null)
    //    {
    //      throw new NotImplementedException("Ceres TensorRT Native evaluator does not yet support head overrides.");
    //    }
    bool EXACT_BATCHES = options.EnableCUDAGraphs;

    // Check if BF16 mode or refittable is requested via NNEvaluatorOptionsCeres
    TensorRTBuildOptions? buildOptions = null;

    //const bool ENABLE_GRAPHS = false;
    bool forceBF16 = options is NNEvaluatorOptionsCeres optionsCeres && optionsCeres.UseBF16;
    bool refittable = options is NNEvaluatorOptionsCeres optionsCeresRefit && optionsCeresRefit.Refittable;
    NNEvaluatorTensorRT trtNativeEngine = new(netFileName,
                                              netType,
                                              EXACT_BATCHES ? EnginePoolMode.Exact : EnginePoolMode.Range,
                                              EXACT_BATCHES ? [1, 8, 20, 42, 64, 88, 116, 240]
                                                            : [96, 1024],
                                              buildOptions,
                                              gpuIDs: gpuIDs,
                                              useCudaGraphs: EXACT_BATCHES,
                                              softMaxBatchSize: 1024,
                                              optimizationLevel: options.OptimizationLevel,
                                              forceBF16: forceBF16,
                                              refittable: refittable);
    trtNativeEngine.Options = options;

    EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true; // ** TODO: remove/rework

    if (netType == ONNXNetExecutor.NetTypeEnum.TPG)
    {
      trtNativeEngine.ConverterToFlatFromTPG = (opts, o, f1) => TPGConvertersToFlat.ConvertToFlatTPGFromTPG(opts, o, f1.Span);
      trtNativeEngine.ConverterToFlat = (opts, o, history, squaresBytes, squares, legalMoveIndices)
        => TPGConvertersToFlat.ConvertToFlatTPG(opts, o, history, squaresBytes, squares, legalMoveIndices);
    }

    return trtNativeEngine;
  }


  /// <summary>
  /// Modifies in place a sequence of batch sizes to be 
  /// aligned with exact SM count (if within a specified tolerance).
  /// </summary>
  /// <param name="anchorBatchSizes"></param>
  /// <returns></returns>
  public static int[] AdjustToSM(int deviceID, int[] anchorBatchSizes)
  {
    int[] adjustedSizes = new int[anchorBatchSizes.Length];

    // Query actual SM count from GPU device
    int smCount = TensorRTNative.GetMultiProcessorCount(deviceID);
    if (smCount < 0)
    {
      throw new Exception($"Failed to query SM count for GPU {deviceID}");
    }
    const int THRESHOLD_ADJUST_TO_SM_DISTANCE = 8;

    int[] bases = { smCount };

    for (int i = 0; i < anchorBatchSizes.Length; i++)
    {
      int original = anchorBatchSizes[i];
      int bestAdjusted = original;
      int bestDistance = int.MaxValue;

      foreach (int baseVal in bases)
      {
        if (baseVal < 1) continue;

        int lower = (original / baseVal) * baseVal;
        int upper = lower + baseVal;

        if (lower > 0)
        {
          int dist = Math.Abs(original - lower);
          if (dist <= THRESHOLD_ADJUST_TO_SM_DISTANCE && dist < bestDistance)
          {
            bestDistance = dist;
            bestAdjusted = lower;
          }
        }

        int distUpper = Math.Abs(original - upper);
        if (distUpper <= THRESHOLD_ADJUST_TO_SM_DISTANCE && distUpper < bestDistance)
        {
          bestDistance = distUpper;
          bestAdjusted = upper;
        }
      }

      adjustedSizes[i] = bestAdjusted;
    }

    return adjustedSizes;
  }


  /// <inheritdoc/>
  public override bool IsEquivalentTo(NNEvaluator evaluator) => evaluator is NNEvaluatorTensorRT other
                                                             && other.ONNXFileName == ONNXFileName;



  /// <inheritdoc/>
  protected override void DoShutdown()
  {
    pool?.Dispose();
    pool = null;

    // Note: trt is a shared singleton, do not dispose it here
  }
}
