#region Using directives

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Threading;
using Ceres.Base.Benchmarking;
using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnx;

#endregion

#region License notice

/*
 This file is part of the Ceres project at https://github.com/dje-dev/ceres.
 Copyright (C)2020- by David Elliott and the Ceres Authors.

 Ceres is free software under the terms of the GNU General Public License v3.0.
 You should have received a copy of the GNU General Public License
 along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

/// <summary>
/// Manages evaluation of neural networks using ONNX runtime.
/// 
/// Although ONNX the documentation stats that multiple threads can invoke the Run() method
/// on the same inference session object, we have single-instance buffers for inputs and outputs
/// and therefore take locks to enforce single-threaded access.
/// 
/// When CUDA graphs are enabled, IOBinding is used to bind GPU memory buffers which must remain
/// at the same memory addresses across invocations for optimal performance.
namespace Ceres.Chess.NNBackends.ONNXRuntime;

public class ONNXExecutor : IDisposable
{
  /// <summary>
  /// Name of underlying ONNX file;
  /// </summary>
  public readonly string ONNXFileName;

  /// <summary>
  /// ID (index) of GPU to use
  /// </summary>
  public readonly int GPUID;

  /// <summary>
  /// Precision width to use (32=FP32, 16=FP16, 8=FP8).
  /// </summary>
  public int PrecisionNumBits;

  /// <summary>
  /// Minimum supported batch size.
  /// </summary>
  public int MinBatchSize;

  /// <summary>
  /// If TensorRT execution provider should be used.
  /// </summary>
  public bool UseTensorRT;

  /// <summary>
  /// If CUDA graphs should be used (if the backend supports it).
  /// </summary>
  public bool EnableCUDAGraphs;

  /// <summary>
  /// If all outputs should be retained.
  /// </summary>
  public bool RetainRawInputs;


  /// <summary>
  /// If true then the client directly populates this evaluator's input buffers
  /// (therefore no copy is needed from inputs provided to Run).
  /// </summary>
  public bool InputBuffersArePrepopulated;

  /// <summary>
  /// The names of the sub-networks if the net is a specially prepared 
  /// Ceres multinet network (containing the string "multinet" in the file name).
  /// </summary>
  public string[] MultiNetNames { get; private set; }


  /// The weights to be used for inference of the sub-networks if the net is a specially 
  /// prepared Ceres multinet network (containing the string "multinet" in the file name).
  public float[] MultiNetWeights { get; private set; }

  /// <summary>
  /// Name of the LoRA adapter file (if any).
  /// </summary>
  public readonly string LoRAAdapterFileName;

  /// <summary>
  /// Stored ONNX model bytes for creating multiple sessions.
  /// </summary>
  private byte[] storedOnnxModelBytes;

  /// <summary>
  /// Stored session creation parameters for creating multiple sessions.
  /// </summary>
  private (string shortID, string[] inputNames, string nonBatchDimensions,
           int precisionNumBits, int gpuID, bool useTensorRT, int minBatchSize,
           int maxBatchSize, bool enableProfiling, string[] outputNamesToRetrieve) sessionCreationParams;

  /// <summary>
  /// Cache of sessions per batch size for CUDA graph support.
  /// Each batch size gets its own InferenceSession to enable separate CUDA graph capture.
  /// Key includes batch size range (min, max) and whether CUDA graphs are enabled.
  /// </summary>
  private Dictionary<(int min, int max, bool useCudaGraphs), SessionForBatchSize> sessionCache;

  /// <summary>
  /// Batch size anchors for sessions without CUDA graphs.
  /// Actual batch sizes will be rounded up to the nearest anchor.
  /// </summary>
  public readonly int[] BATCH_SIZE_ANCHORS_WITHOUT_GRAPH;

  /// <summary>
  /// Batch size anchors for sessions with CUDA graphs.
  /// Actual batch sizes will be rounded up to the nearest anchor.
  /// </summary>
  public readonly int[] BATCH_SIZE_ANCHORS_WITH_GRAPH;

  /// <summary>
  /// Version of BATCH_SIZE_ANCHORS_WITH_GRAPH adjusted such that
  /// the breakpoints are rounded up to represent the actual fixed size of the batch.
  /// </summary>
  private readonly int[] BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED;

  /// <summary>
  /// Underlying CUDA device which this evaluator uses.
  /// </summary>
  private CUDADevice cudaDevice;

  /// <summary>
  /// Size of the input data type (in bits).
  /// </summary>
  public readonly int InputsNumBits;

  /// <summary>
  /// If a single session can support multiple batch size profiles.
  /// (See notes; this is currently not supported in TensorRT (only TensorRT RTX).
  /// </summary>
  public bool UseMultipleProfilesPerSession { get; private set; }

  /// <summary>
  /// TensorRT optimization level (default is 3, maximum is 5).
  /// Engine build time is typically approximately 2x longer at 4 vs 3
  /// with a typically relatively small pickup in nps (0% to 5%).
  /// However using 4 or 5 increases the number of kernel timing attempts
  /// and has possibly been observed to decrease the frequence of failure
  /// to generate an FP16 engine when requested.
  /// </summary>
  public int OptimizationLevel { get; private set; } = 4;


  /// <summary>
  /// Execution is serialized by this lock object.
  /// Technically, ONNX runtime sessions are thread-safe
  /// so this might not be strictly necessary.
  /// </summary>
  readonly object lockObject = new object();

  int maxBatchSize;

  RunOptions runOptions;
  bool disposed;
  bool haveWarned = false;

  private IReadOnlyDictionary<string, NodeMetadata> inputsMetadata;
  private readonly object metadataLock = new object();

  /// <summary>
  /// Lazily initialized input metadata from the ONNX model.
  /// </summary>
  public IReadOnlyDictionary<string, NodeMetadata> InputsMetadata
  {
    get
    {
      if (inputsMetadata == null)
      {
        lock (metadataLock)
        {
          if (inputsMetadata == null)
          {
            // Use the session for smallest batch size
            SessionForBatchSize session = GetOrCreateSessionForBatchSize(MinBatchSize);
            inputsMetadata = FilterMetadata(session.Session.InputMetadata);
          }
        }
      }
      return inputsMetadata;
    }
  }

  public enum ONNXInputTypeEnum { Float32, Float16, Byte };

  static bool VERBOSE_LOGGING = false;

  /// <summary>
  /// Constructor.
  /// </summary>
  public ONNXExecutor(string shortID,
      string onnxFileName,
      byte[] onnxModelBytes,
      string[] inputNames,
      string nonBatchDimensions,
      int inputsNumBits,
      int precisionNumBits,
      bool inputBuffersArePrepopulated,
      int gpuID,
      bool useTensorRT,
      bool enableCUDAGraphs,
      int minBatchSize,
      int maxBatchSize,
      bool enableProfiling,
      bool retainRawOutputs,
      string[] outputNamesToRetrieve = null,
      string loraAdapterFileName = null)
  {
    if (!inputBuffersArePrepopulated)
    {
      throw new NotImplementedException("Currently only inputBuffersArePrepopulated=true is supported.");
    }

    if (onnxFileName != null && !File.Exists(onnxFileName))
    {
      throw new Exception("ONNX file not found: " + onnxFileName);
    }

    if (minBatchSize != 1)
    {
      throw new NotImplementedException(shortID + ": Currently only minBatchSize=1 is supported.");
    }

    if (precisionNumBits != 32 && precisionNumBits != 16)
    {
      throw new NotImplementedException();
    }

    ONNXFileName = onnxFileName;
    if (onnxFileName == null && onnxModelBytes == null)
    {
      throw new Exception("Must specify either onnxFileName or onnxModelBytes");
    }

    InputsNumBits = inputsNumBits;
    InputBuffersArePrepopulated = inputBuffersArePrepopulated;

    // Multiple engines only beneficial when using TensorRT
    bool multiEngineMode = useTensorRT;// && ONNXFileName != null && !ONNXFileName.ToLower().Contains("copy");
    UseMultipleProfilesPerSession = false;

    BATCH_SIZE_ANCHORS_WITHOUT_GRAPH = [48, 128];
    BATCH_SIZE_ANCHORS_WITH_GRAPH = enableCUDAGraphs && useTensorRT ? [12, 32, 56, 88] : null;

#if NOT
    // Possible ONNX bugs
    // Graph capture/replay only works one time: https://github.com/microsoft/onnxruntime/issues/22583
    // Device memory allocations only work on device with index 0: https://github.com/microsoft/onnxruntime/issues/24453
#endif

    if (BATCH_SIZE_ANCHORS_WITH_GRAPH != null)
    {
      BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED = new int[BATCH_SIZE_ANCHORS_WITH_GRAPH.Length];
      for (int i = 0; i < BATCH_SIZE_ANCHORS_WITH_GRAPH.Length; i++)
      {
        BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED[i] = Math.Min(maxBatchSize, BATCH_SIZE_ANCHORS_WITH_GRAPH[i] + 1);
      }
    }

    this.maxBatchSize = maxBatchSize;

    if (onnxModelBytes == null)
    {
      onnxModelBytes = File.ReadAllBytes(onnxFileName);
    }

    // Store ONNX model bytes and session creation parameters for lazy initialization
    storedOnnxModelBytes = onnxModelBytes;
    sessionCreationParams = (shortID, inputNames, nonBatchDimensions, precisionNumBits,
                             gpuID, useTensorRT, minBatchSize, maxBatchSize,
                             enableProfiling, outputNamesToRetrieve);

    ExtractMultinetMetadataIfApplicable(onnxFileName, onnxModelBytes);

    cudaDevice = CUDADevice.GetContext(gpuID);
    runOptions = new RunOptions();

    if (loraAdapterFileName != null)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Install LoRA adapter " + loraAdapterFileName);
      OrtLoraAdapter adapterCeres = OrtLoraAdapter.Create(loraAdapterFileName, null);
      runOptions.AddActiveLoraAdapter(adapterCeres);
    }

    GPUID = gpuID;
    PrecisionNumBits = precisionNumBits;
    MinBatchSize = minBatchSize;
    UseTensorRT = useTensorRT;
    EnableCUDAGraphs = enableCUDAGraphs;
    RetainRawInputs = retainRawOutputs;
    LoRAAdapterFileName = loraAdapterFileName;

    // On Linux it was found necessary to touch the instance before any of the operations below
    // to prevent error about a session object not being created.
    // https://github.com/microsoft/onnxruntime/issues/11572
    OrtEnv ortInstance = OrtEnv.Instance();

    // Initialize session cache
    sessionCache = new Dictionary<(int, int, bool), SessionForBatchSize>();

    Console.WriteLine($"ONNXExecutor initialized on GPU {GPUID} (sessions will be created on-demand)");
    lock (warmupLock) // Prevent overlapping stream capture
    {
      Warmup();
    }
  }


  static readonly object warmupLock = new();


  private void ExtractMultinetMetadataIfApplicable(string onnxFileName, byte[] onnxModelBytes)
  {
    if (onnxFileName == null || onnxFileName.ToUpper().Contains("MULTINET"))
    {
      using (new TimingBlock("ONNX ModelProto parse"))
      {
        ModelProto onnxProto = ModelProto.Parser.ParseFrom(onnxModelBytes);
        string multinetNames = Ceres.Base.Misc.ONNX.ONNXHelpers.GetMetadataValue(onnxProto, "Ceres_multinet_names");
        if (multinetNames != null)
        {
          MultiNetNames = multinetNames.Split(',');
        }

        string multinetWeights = Ceres.Base.Misc.ONNX.ONNXHelpers.GetMetadataValue(onnxProto, "Ceres_multinet_weights");
        if (multinetWeights != null)
        {
          MultiNetWeights = multinetWeights.Split(',').Select(float.Parse).ToArray();
        }

        if (MultiNetNames != null)
        {
          Console.WriteLine();
          Console.Write("LOADING Ceres MultiNet:");
          for (int i = 0; i < MultiNetNames.Length; i++)
          {
            Console.Write(" " + MultiNetNames[i] + "=" + MultiNetWeights[i]);
          }
        }
      }
    }
  }


  /// <summary>
  /// Filters metadata to exclude inputs with zero dimensions (used for LoRA adapters).
  /// </summary>
  private static IReadOnlyDictionary<string, NodeMetadata> FilterMetadata(IReadOnlyDictionary<string, NodeMetadata> metadata)
  {
    Dictionary<string, NodeMetadata> newDict = new();
    foreach (KeyValuePair<string, NodeMetadata> kvp in metadata)
    {
      if (!kvp.Value.Dimensions.Contains(0))
      {
        newDict.Add(kvp.Key, kvp.Value);
      }
    }
    return newDict;
  }


  /// <summary>
  /// Creates SessionOptions based on stored parameters.
  /// </summary>
  private SessionOptions CreateSessionOptions((string shortID, string[] inputNames, string nonBatchDimensions,
                                               int precisionNumBits, int gpuID, bool useTensorRT,
                                               int minBatchSize, int maxBatchSize,
                                               bool enableProfiling, string[] outputNamesToRetrieve) parameters,
                                               int minBatch, int maxBatch, bool useCudaGraphs = false)
  {
    var (shortID, inputNames, nonBatchDimensions, precisionNumBits,
         gpuID, useTensorRT, minBatchSize, maxBatchSize,
         enableProfiling, outputNamesToRetrieve) = parameters;

    SessionOptions so;

    if (gpuID < 0) // CPU
    {
      so = new SessionOptions();
    }
    else if (useTensorRT)
    {
      // Batches for CUDA graphs always execute full set 
      int minBatchSizetoUse = useCudaGraphs ? maxBatch : minBatch;
      so = CreateTensorRTSessionOptions(shortID, inputNames, nonBatchDimensions, gpuID,
                                        precisionNumBits, minBatchSizetoUse, maxBatch, useCudaGraphs);
    }
    else
    {
      so = CreateCUDASessionOptions(gpuID);
    }

    so.LogSeverityLevel = VERBOSE_LOGGING ? OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
                                          : OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;

    if (VERBOSE_LOGGING)
    {
      so.LogVerbosityLevel = 999;
      so.LogId = @"ort.log.txt";
    }

    if (enableProfiling)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, @"****************   NetExecutorONNXRuntime is profiling to c:\temp ....   ****************");
      so.EnableProfiling = true;
      so.ProfileOutputPathPrefix = @"c:\temp";
    }

    // N.B. Use of ORT_ENABLE_ALL might possibly exacerbate the
    //      nondeterminism of engine generation and inconsistent request for FP16 precision
    so.GraphOptimizationLevel = OptimizationLevel == 0 ? GraphOptimizationLevel.ORT_ENABLE_BASIC
                                                       : GraphOptimizationLevel.ORT_ENABLE_EXTENDED;

    // N.B. Do not use ORT_PARALLEL, this causes failures in ONNXRuntime when using GPUs with index other than 0.
    //      Probably related to this ONNXRuntime bug: https://github.com/microsoft/onnxruntime/issues/24453
    so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

    return so;
  }


  static string lastNotifiedTRTCacheDir;

  string GetTRTEngineCacheDir()
  {
    string directoryName = ONNXFileName == null ? Path.GetTempPath() : new FileInfo(ONNXFileName).DirectoryName;
    string trtSubdirectory = Path.Combine(directoryName, "trt_engines", Environment.MachineName);
    if (trtSubdirectory != lastNotifiedTRTCacheDir)
    {
      Console.WriteLine("TensorRT engines will be cached in: " + trtSubdirectory);
      lastNotifiedTRTCacheDir = trtSubdirectory;
    }

    return trtSubdirectory;
  }


  /// <summary>
  /// Creates TensorRT-specific session options.
  /// </summary>
  private SessionOptions CreateTensorRTSessionOptions(string shortID, string[] inputNames,
                                                      string nonBatchDimensions, int gpuID, int precisionNumBits,
                                                      int minBatch, int maxBatch, bool useCudaGraphs)
  {
    using OrtTensorRTProviderOptions trtProviderOptions = new OrtTensorRTProviderOptions();
    Dictionary<string, string> providerOptionsDict = new();

    providerOptionsDict["device_id"] = gpuID.ToString();
    providerOptionsDict["trt_max_workspace_size"] = (4L * 1024 * 1024 * 1024).ToString();

    if (inputNames != null)
    {
      string MakeShapeStr(int size)
      {
        bool firstTime = true;
        string ret = "";
        foreach (string inputName in inputNames)
        {
          if (!firstTime)
          {
#if DEBUG
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "WARNING as workaround for caching failure in ONNXRuntime,"
          + " shapes string will omit " + inputName);
#endif
            continue;
          }
          ret += (firstTime ? "" : ",") + inputName + $":{size}x{nonBatchDimensions}";
          firstTime = false;
        }
        return ret;
      }

      if (UseMultipleProfilesPerSession)
      {
        throw new Exception("Current implementation restriction: UseSingleProfile must be true.");
      }
      else
      {
        int optimalBatchSize;
        if (minBatch <= 64 && maxBatch > 512)
        {
          // Don't set optimal excessively high
          optimalBatchSize = Math.Min(maxBatch, 256);
        }
        else
        {
          // Use midpoint
          optimalBatchSize = Math.Min(maxBatch, (int)MathUtils.RoundedUp((minBatch + maxBatch) / 2, 2));
        }

        providerOptionsDict["trt_profile_min_shapes"] = MakeShapeStr(minBatch);
        providerOptionsDict["trt_profile_opt_shapes"] = MakeShapeStr(optimalBatchSize);
        providerOptionsDict["trt_profile_max_shapes"] = MakeShapeStr(maxBatch);
      }
    }

    // Engine cache configuration
    string trtSubdirectory = GetTRTEngineCacheDir();
    Directory.CreateDirectory(trtSubdirectory);

    bool EMBED = Environment.GetEnvironmentVariable("EMBED_TRT") == "1";
    if (EMBED)
    {
      providerOptionsDict["trt_ep_context_file_path"] = "./";
      providerOptionsDict["trt_dump_ep_context_model"] = "1";
      providerOptionsDict["trt_ep_context_embed_mode"] = "1";

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "NOTE: EMBED_TRT is set to 1. TensorRT engine will be embedded in the ONNX file _ctx.onnx.");
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "NOTE: the _ctx.onnx file will only be created only upon normal termination of this process.");
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "NOTE: For security reasons, this file is emitted in subdirectory trt_engines_embed of the working directory of this process.");
      Console.WriteLine();
    }


    const bool ENABLE_CACHING = true;
    if (ENABLE_CACHING)
    {
      providerOptionsDict["trt_engine_cache_enable"] = "1";
      providerOptionsDict["trt_engine_cache_path"] = trtSubdirectory;

      string shortIDNoPrefix = shortID.Split("|")[0]; // remove anything after pipe character (options string)
      string cachePrefix = FileUtils.FileNameSanitized(shortIDNoPrefix);

      cachePrefix += $"_bs{minBatch}-{maxBatch}" + (useCudaGraphs ? "-graph" : "");

      providerOptionsDict["trt_engine_cache_prefix"] = cachePrefix;
      providerOptionsDict["trt_timing_cache_enable"] = "1";
      providerOptionsDict["trt_timing_cache_path"] = trtSubdirectory;
      providerOptionsDict["trt_force_timing_cache"] = "1";
    }

    providerOptionsDict["trt_fp16_enable"] = precisionNumBits == 16 ? "1" : "0";
    providerOptionsDict["trt_builder_optimization_level"] = OptimizationLevel.ToString();
    providerOptionsDict["trt_cuda_graph_enable"] = useCudaGraphs ? "1" : "0";
    providerOptionsDict["trt_auxiliary_streams"] = "0";
    providerOptionsDict["trt_layer_norm_fp32_fallback"] = "1";

    // providerOptionsDict["trt_context_memory_sharing_enable"] = "1"; //returns error, not obviously faster
    //providerOptionsDict["trt_detailed_build_log"] = "1";

    trtProviderOptions.UpdateOptions(providerOptionsDict);
    return SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
  }


  /// <summary>
  /// Creates CUDA-specific session options.
  /// </summary>
  private SessionOptions CreateCUDASessionOptions(int gpuID)
  {
    using OrtCUDAProviderOptions cudaProviderOptions = new();
    Dictionary<string, string> providerOptionsDict = new();
    providerOptionsDict["device_id"] = gpuID.ToString();
    cudaProviderOptions.UpdateOptions(providerOptionsDict);
    return SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
  }


  /// <summary>
  /// Performs any initialization to prepare evaluator for delay-free execution.
  /// </summary>
  public void Warmup()
  {
    List<int> batchSizesToRunWithGraphs = new List<int>();

    _ = GetOrCreateSessionForBatchSize(1, true);

    // Get engines at/around the breakpoints to trigger creation if needed.
    foreach (int b in BATCH_SIZE_ANCHORS_WITHOUT_GRAPH)
    {
      _ = GetOrCreateSessionForBatchSize(b, true);
    }

    if (BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED != null)
    {
      foreach (int b in BATCH_SIZE_ANCHORS_WITH_GRAPH)
      {
        _ = GetOrCreateSessionForBatchSize(b);
        batchSizesToRunWithGraphs.Add(b);
      }
    }

    // Additionally we need to run once to capture the CUDA graphs
    // (over uninitialized buffers).
    // Seemingly CUDA capture operations must be strictly globally serialized.
    lock (cudaGraphSessionCaptureGlobalLock)
    {
      foreach (int batchSizeToRun in batchSizesToRunWithGraphs)
      {
        switch (InputsNumBits)
        {
          case 8:
            _ = RunWithIOBinding<byte, Float16>(null, batchSizeToRun, false);
            break;
          case 16:
            _ = RunWithIOBinding<Half, Float16>(null, batchSizeToRun, false);
            break;
          default:
            throw new NotImplementedException();
        }
      }
    }
  }

  static readonly object cudaGraphSessionCaptureGlobalLock = new();


  /// <summary>
  /// Returns the number of inputs as reported by the ONNX model metadata.
  /// </summary>
  public int NumInputs => InputsMetadata.Count;


  /// <summary>
  /// Common validation and input processing logic shared by both Run method overloads.
  /// </summary>
  /// <typeparam name="T">The input data type (byte or Half)</typeparam>
  /// <param name="inputs">Input array containing memory, shape, and metadata</param>
  /// <param name="batchSize">Batch size for processing</param>
  /// <returns>Processed input data with names and element counts</returns>
  private (Memory<T> input, int[] shape, string inputName, int numElements)[] ValidateAndProcessInputs<T>((Memory<T> input, int[] shape)[] inputs, int batchSize)
  {
    if (InputsMetadata.Count != inputs.Length)
    {
      throw new ArgumentException($"Expected {InputsMetadata.Count} inputs, received " + inputs.Length);
    }

    if (inputs[0].shape[0] > maxBatchSize)
    {
      throw new ArgumentException($"Batch size {inputs[0].shape[0]} exceeds maximum of {maxBatchSize}");
    }

    var inputsONNX = new (Memory<T> input, int[] shape, string inputName, int numElements)[InputsMetadata.Count];

    if (InputsMetadata.Count != 1)
    {
      if (!haveWarned)
      {
        // data type check below is only on first element
        Console.WriteLine("WARNING: Currently only single input ONNX files supported definitively.");
        haveWarned = true;
      }
    }

    int inputIndex = 0;
    foreach (KeyValuePair<string, NodeMetadata> iv in InputsMetadata)
    {
      (Memory<T> input, int[] shape) = inputs[inputIndex];
      string inputName = iv.Key;
      if (inputName == null)
      {
        throw new Exception("Unable to retrieve name of input");
      }

      int numElements = ONNXHelpers.ProductDimensions(shape, batchSize);
      Debug.Assert(input.Length == numElements); // caller to have passed the correct size

      inputsONNX[inputIndex] = (input, shape, inputName, numElements);
      inputIndex++;
    }

    return inputsONNX;
  }


  /// <summary>
  /// Evaluates the input.
  /// </summary>
  /// <param name="inputType"></param>
  /// <param name="inputs"></param>
  /// <param name="batchSize"></param>
  /// <returns></returns>
  public List<(string, Memory<Float16>)> Run(ONNXInputTypeEnum inputType, (Memory<byte> input, int[] shape)[] inputs, int batchSize)
  {
    (Memory<byte> input, int[] shape, string inputName, int numElements)[]
  inputsONNX = ValidateAndProcessInputs(inputs, batchSize);

    // TODO: Actually the precision of the network is defined by the net itself.
    //       So the inputIsFloat above should be used to determine this, and
    //       the caller should not be offered the chance to set the precision here
    //       (unless we decided to support auto-conversion of ONNX files here).
    if (inputType == ONNXInputTypeEnum.Byte)
    {
      return RunInputByteOutputFloat16(inputsONNX, batchSize);
    }
    else
    {
      throw new NotImplementedException("Unexpected ONNXInputTypeEnum" + inputType);
    }
  }


  /// <summary>
  /// Evaluates the input.
  /// </summary>
  /// <param name="inputType"></param>
  /// <param name="inputs"></param>
  /// <param name="batchSize"></param>
  /// <returns></returns>
  public List<(string, Memory<Float16>)> Run(ONNXInputTypeEnum inputType, (Memory<Half> input, int[] shape)[] inputs, int batchSize)
  {
    (Memory<Half> input, int[] shape, string inputName, int numElements)[]
      inputsONNX = ValidateAndProcessInputs(inputs, batchSize);

    // TODO: Actually the precision of the network is defined by the net itself.
    //       So the inputIsFloat above should be used to determine this, and
    //       the caller should not be offered the chance to set the precision here
    //       (unless we decided to support auto-conversion of ONNX files here).
    if (inputType == ONNXInputTypeEnum.Float16)
    {
      return RunInputHalfOutputFloat16(inputsONNX, batchSize);
    }
    else if (inputType == ONNXInputTypeEnum.Float32)
    {
      return RunOutputFloat(inputsONNX, batchSize);
    }
    else if (inputType == ONNXInputTypeEnum.Byte)
    {
      throw new Exception("Use the overloaded function for Byte instead");
    }
    else
    {
      throw new NotImplementedException("Unknown ONNXInputTypeEnum" + inputType);
    }
  }


  /// <summary>
  /// 
  /// 
  /// TO DO: eventually we could have a separate (more efficient) code path 
  ///        which is FP16 throughout rather than multiple conversions.
  /// </summary>
  /// <param name="input"></param>
  /// <param name="shape"></param>
  /// <param name="inputName"></param>
  /// <param name="numElements"></param>
  /// <returns></returns>
  internal List<(string, Memory<Float16>)> RunInputByteOutputFloat16((Memory<byte> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)

  {
    SessionForBatchSize session = GetOrCreateSessionForBatchSize(batchSize);
    return session.UsesCUDAGraphs ? RunWithIOBinding<byte, Float16>(inputs, batchSize)
                                  : RunWithDirectRun<byte, Float16>(inputs, batchSize);
  }


  /// <summary>
  /// 
  /// 
  /// TO DO: eventually we could have a separate (more efficient) code path 
  ///        which is FP16 throughout rather than multiple conversions.
  /// </summary>
  /// <param name="input"></param>
  /// <param name="shape"></param>
  /// <param name="inputName"></param>
  /// <param name="numElements"></param>
  /// <returns></returns>
  internal List<(string, Memory<Float16>)> RunInputHalfOutputFloat16((Memory<Half> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
  {
    SessionForBatchSize session = GetOrCreateSessionForBatchSize(batchSize);
    return session.UsesCUDAGraphs ? RunWithIOBinding<Half, Float16>(inputs, batchSize)
                                  : RunWithDirectRun<Half, Float16>(inputs, batchSize);
  }



  public Memory<TDest> InputBufferForBatchSize<T, TDest>(int inputIndex, int batchSize)
                         where T : unmanaged
                         where TDest : unmanaged
  {
    if (batchSize > maxBatchSize)
    {
      throw new ArgumentOutOfRangeException($"Batch size {batchSize} exceeds maximum of {maxBatchSize}");
    }

    int evaluatorBatchSize = Math.Max(MinBatchSize, batchSize);
    T[] rawArray = ((T[])GetOrCreateSessionForBatchSize(evaluatorBatchSize).InputBuffers[inputIndex]);
    int numElements = ONNXHelpers.ProductDimensions(InputsMetadata.ElementAt(inputIndex).Value.Dimensions, batchSize);

    return MemoryCasted.AsMemory<T, TDest>(rawArray).Slice(0, numElements);
  }


  /// <summary>
  /// Executes inference using IOBinding (required for CUDA graphs).
  /// </summary>
  private List<(string, Memory<Float16>)> RunWithIOBinding<TInput, TBuffer>(
    (Memory<TInput> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize, bool disableCUDAGraphs = false)
      where TInput : unmanaged
      where TBuffer : unmanaged
  {
    //Console.WriteLine("zzzIOBINDING " + batchSize + "  " + ONNXFileName);

    if (batchSize < MinBatchSize)
    {
      throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
    }
    List<(string, Memory<Float16>)> resultArrays = new();

    lock (lockObject)
    {
      SessionForBatchSize context = GetOrCreateSessionForBatchSize(batchSize, disableCUDAGraphs);

      cudaDevice.SetCurrent();

      //runOptions.AddRunConfigEntry("gpu_graph_id", "1");

      //context.IoBinding.ClearBoundInputs();

      // DJE: found necessary to rebind every time
      int inputVarIndex = 0;
      foreach (var (name, ortValue, shape, cudaBuffer, cudaBufferFloat16) in context.InputOrtValues)
      {
        Array array = context.InputBuffers[inputVarIndex];
        int numElements = (int)ONNXHelpers.ProductDimensionsLong(shape, batchSize);
        int numBytes = numElements * Marshal.SizeOf(typeof(TInput));

        switch (typeof(TInput))
        {
          case Type t when t == typeof(byte):
            cudaBuffer.CopyToDevice((byte[])array, 0, 0, numBytes);
            break;

          case Type t when t == typeof(Half):
            cudaBufferFloat16.CopyToDevice((Float16[])array, 0, 0, numBytes);
            break;

          default:
            throw new NotImplementedException();
        }

        context.IoBinding.BindInput(name, ortValue);
        inputVarIndex++;
      }

      // For CUDA graphs, bindings are already set up at session creation time
      // We just need to synchronize and run
      cudaDevice.Context.Synchronize();
      context.Session.RunWithBinding(runOptions, context.IoBinding);
      cudaDevice.Context.Synchronize();

      int outputVarIndex = 0;
      foreach (var (name, ortValue, shape, cudaBuffer) in context.OutputOrtValues)
      {
        int numElements = (int)ONNXHelpers.ProductDimensionsLong(shape, batchSize);
        int numBytes = numElements * Marshal.SizeOf(typeof(Float16));

        cudaBuffer.CopyToHost((Float16[])context.OutputBuffers[outputVarIndex].buffer, 0, 0, numBytes);
        outputVarIndex++;
      }

      // Package up restult array.
      for (int i = 0; i < context.OutputOrtValues.Count; i++)
      {
        var (name, ortValue, shape, cudaBuffer) = context.OutputOrtValues[i];
        NodeMetadata metadata = context.OutputBuffers[i].metadata;
        int usedElements = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);

        if (typeof(TBuffer) == typeof(Float16))
        {
          resultArrays.Add((name, new Memory<Float16>((Float16[])context.OutputBuffers[i].buffer, 0, usedElements)));
        }
        else if (typeof(TBuffer) == typeof(float))
        {
          // Convert to Float16 for return. TODO: improve efficiency (avoid allocation)
          float[] resultBuffer = (float[])context.OutputBuffers[i].buffer;
          Float16[] retBuffer16 = new Float16[usedElements];
          for (int j = 0; j < usedElements; j++)
          {
            retBuffer16[j] = (Float16)resultBuffer[j];
          }

          resultArrays.Add((name, new Memory<Float16>(retBuffer16, 0, usedElements)));
        }
        else
        {
          throw new NotSupportedException($"Unsupported output buffer type {typeof(TBuffer)}");
        }

      }
    }

    return resultArrays;
  }


  /// <summary>
  /// Executes inference using direct Run method with OrtValues (alternative to IOBinding).
  /// </summary>
  private List<(string, Memory<Float16>)> RunWithDirectRun<TInput, TBuffer>(
    (Memory<TInput> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
      where TInput : unmanaged
      where TBuffer : unmanaged
  {
    //    if (ONNXFileName.ToLower().Contains("copy")) Console.WriteLine("zzzDirectRun " + batchSize + " --------------------");
    if (batchSize < MinBatchSize)
    {
      throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
    }

    List<(string, Memory<Float16>)> resultArrays = new();

    lock (lockObject)
    {
      SessionForBatchSize context = GetOrCreateSessionForBatchSize(batchSize);

      // Create temporary OrtValues with actual batch size for this run
      List<OrtValue> inputOrtValuesList = new List<OrtValue>();
      List<OrtValue> outputOrtValuesList = new List<OrtValue>();

      List<string> inputNames = new List<string>(InputsMetadata.Count); // TODO: preallocate
      List<string> outputNames = new List<string>(); // TODO: preallocate

      try
      {
        // Create input OrtValues for this specific batch size
        for (int i = 0; i < context.InputOrtValues.Count; i++)
        {
          long[] shape = ONNXHelpers.ToLongArray(InputsMetadata[context.InputOrtValues[i].name].Dimensions, batchSize);

          inputOrtValuesList.Add(InputsNumBits switch
          {
            8 => OrtValue.CreateTensorValueFromMemory((byte[])context.InputBuffers[i], shape),
            16 => OrtValue.CreateTensorValueFromMemory((Float16[])context.InputBuffers[i], shape),
            32 => OrtValue.CreateTensorValueFromMemory((float[])context.InputBuffers[i], shape),
            _ => throw new NotSupportedException($"Unsupported InputsNumBits {InputsNumBits}"),
          });
          inputNames.Add(context.InputOrtValues[i].name);
        }

        for (int i = 0; i < context.OutputBuffers.Count; i++)
        {
          var (name, metadata, buffer) = context.OutputBuffers[i];
          long[] shape = ONNXHelpers.ToLongArray(metadata.Dimensions, batchSize);
          int count = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);

          OrtValue ortVal = ortVal = buffer switch
          {
            Float16[] f16 => OrtValue.CreateTensorValueFromMemory(f16, shape),
            float[] f32 => OrtValue.CreateTensorValueFromMemory(f32, shape),
            _ => throw new NotSupportedException($"Unsupported output buffer type for {name}"),
          };

          outputOrtValuesList.Add(ortVal);
          outputNames.Add(name);
        }

        // Run inference using direct Run method
        context.Session.Run(runOptions, inputNames, inputOrtValuesList, outputNames, outputOrtValuesList);

        // Read output buffers
        for (int i = 0; i < context.OutputBuffers.Count; i++)
        {
          Array cpuBuffer = context.OutputBuffers[i].buffer;
          int usedElements = ONNXHelpers.ProductDimensions(context.OutputBuffers[i].metadata.Dimensions, batchSize);

          if (cpuBuffer is Float16[] float16Buffer)
          {
            resultArrays.Add((context.OutputBuffers[i].name, new Memory<Float16>(float16Buffer, 0, usedElements)));
          }
          else if (cpuBuffer is float[] floatBuffer)
          {
            // TODO: improve efficiency
            Float16[] float16Result = new Float16[usedElements];
            for (int j = 0; j < usedElements; j++)
            {
              float16Result[j] = (Float16)floatBuffer[j];
            }
            resultArrays.Add((context.OutputBuffers[i].name, float16Result));
          }
        }
      }
      finally
      {
        outputOrtValuesList.ForEach(ortVal => ortVal?.Dispose());
        inputOrtValuesList.ForEach(ortVal => ortVal?.Dispose());
      }
    }

    return resultArrays;
  }


  /// <summary>
  /// Runs the network with float inputs (instead of Half).
  /// 
  /// Note that this accepts Half inputs and returns Half inputs, 
  /// but merely upcasts them to floats for ONNX runtime execution and then downcasts results.
  /// 
  /// This is inefficient and does not fully exploit the higher precision of float over Half
  /// (but is intended mostly for debugging purposes).
  /// </summary>
  /// <param name="inputs"></param>
  /// <param name="batchSize"></param>
  /// <returns></returns>
  private List<(string, Memory<Float16>)> RunOutputFloat((Memory<Half> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
  {
    if (batchSize < MinBatchSize)
    {
      throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
    }

    List<(string, Memory<Float16>)> resultArrays = new();

    // TODO: improve this method to be more like RunOutputHalf (copy or share logic)
    lock (lockObject)
    {
      SessionForBatchSize context = GetOrCreateSessionForBatchSize(batchSize);

      // Convert inputs to float buffers using pre-allocated inputBuffers32.
      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<Half> input, int[] shape, string inputName, int numElements) = inputs[i];
        float[] buffer = context.InputBuffers[i] as float[];
        Span<float> inputBufferSpanFloat = buffer.AsSpan(0, numElements);
        TensorPrimitives.ConvertToSingle(input.Span, inputBufferSpanFloat);
      }

      if (!context.UsesCUDAGraphs)
      {
        // Create temporary OrtValues for direct Run
        OrtMemoryInfo cpuMemInfo = OrtMemoryInfo.DefaultInstance;
        List<OrtValue> inputOrtValuesList = new List<OrtValue>();
        List<OrtValue> outputOrtValuesList2 = new List<OrtValue>();

        List<string> inputNames = new List<string>(InputsMetadata.Count);
        List<string> outputNames2 = new List<string>(); // TODO: preallocate

        try
        {
          // Create input OrtValues from InputBuffers (not from context.InputOrtValues which are null)
          for (int i = 0; i < context.InputBuffers.Count; i++)
          {
            string name = InputsMetadata.ElementAt(i).Key;
            float[] buf = (float[])context.InputBuffers[i];
            long[] shape = ONNXHelpers.ToLongArray(InputsMetadata.ElementAt(i).Value.Dimensions, batchSize);

            OrtValue ortValue = OrtValue.CreateTensorValueFromMemory(buf, shape);
            inputOrtValuesList.Add(ortValue);
            inputNames.Add(name);
          }

          // Create output OrtValues
          List<OrtValue> outputOrtValuesList = new List<OrtValue>();
          List<string> outputNames = new List<string>();

          try
          {
            for (int i = 0; i < context.OutputBuffers.Count; i++)
            {
              var (name, metadata, buffer) = context.OutputBuffers[i];
              long[] shape = ONNXHelpers.ToLongArray(metadata.Dimensions, batchSize);
              int count = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);

              float[] f32 = buffer as float[];
              Memory<float> mem = new Memory<float>(f32, 0, count);
              OrtValue ortVal = OrtValue.CreateTensorValueFromMemory<float>(cpuMemInfo, mem, shape);

              outputOrtValuesList.Add(ortVal);
              outputNames.Add(name);
            }

            // Run inference
            context.Session.Run(runOptions, inputNames, inputOrtValuesList, outputNames, outputOrtValuesList);

            // Convert results
            foreach (var (name, metadata, buffer) in context.OutputBuffers)
            {
              float[] floatBuffer = buffer as float[];
              int count = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);
              Float16[] halfResult = new Float16[count];
              for (int j = 0; j < count; j++)
              {
                halfResult[j] = (Float16)floatBuffer[j];
              }
              resultArrays.Add((name, halfResult));
            }
          }
          finally
          {
            foreach (var ortVal in outputOrtValuesList)
            {
              ortVal?.Dispose();
            }
          }
        }
        finally
        {
          foreach (var ortVal in inputOrtValuesList)
          {
            ortVal?.Dispose();
          }
        }
      }
      else
      {
        // Use IOBinding
        context.IoBinding.SynchronizeBoundInputs();
        context.Session.RunWithBinding(runOptions, context.IoBinding);
        context.IoBinding.SynchronizeBoundOutputs();

        foreach (var (name, metadata, buffer) in context.OutputBuffers)
        {
          float[] floatBuffer = buffer as float[];
          int count = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);
          Float16[] halfResult = new Float16[count];
          for (int j = 0; j < count; j++)
          {
            halfResult[j] = (Float16)floatBuffer[j];
          }
          resultArrays.Add((name, halfResult));
        }
      }
    }

    return resultArrays;
  }


  /// <summary>
  /// Ends profiling.
  /// </summary>
  public void EndProfiling()
  {
    foreach (SessionForBatchSize context in sessionCache.Values)
    {
      context.Session.EndProfiling();
    }
  }


  /// <summary>
  /// Returns a string description of this object.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return "<ONNXExecutor " + ONNXFileName + " (" + PrecisionNumBits + ")>";
  }


  /// <summary>
  /// Disposes of this object.
  /// </summary>
  public void Dispose()
  {
    if (!disposed)
    {
      // Dispose CUDA graph contexts
      if (sessionCache != null)
      {
        foreach (SessionForBatchSize context in sessionCache.Values)
        {
          context.Dispose();
        }
        sessionCache.Clear();
      }

      runOptions.Dispose();
      disposed = true;
    }
  }


  private (int min, int max, bool useCudaGraphs) FindBatchSizeBucket(int batchSize, bool useCudaGraphs)
  {

    // Select the appropriate anchors array
    int[] anchors = useCudaGraphs ? BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED
                                  : BATCH_SIZE_ANCHORS_WITHOUT_GRAPH;

    // If no anchors are configured, fallback to a single bucket spanning the full range
    if (anchors == null || anchors.Length == 0)
    {
      return (MinBatchSize, maxBatchSize, useCudaGraphs);
    }

    int currentMin = MinBatchSize;
    foreach (int anchor in anchors)
    {
      int currentMax = anchor - 1;
      if (batchSize >= currentMin && batchSize <= currentMax)
      {
        return (currentMin, currentMax, useCudaGraphs);
      }
      currentMin = anchor;
    }

    // If batchSize is larger than or equal to the last anchor
    int lastAnchor = anchors[^1];
    if (useCudaGraphs)
    {
      // Batch size exceeds CUDA graph anchors - fall back to non-CUDA-graph mode
      // and recursively find the appropriate bucket
      return FindBatchSizeBucket(batchSize, useCudaGraphs: false);
    }

    // Non-graph: catch-all bucket to global max
    return (currentMin, maxBatchSize, useCudaGraphs);
  }


  /// <summary>
  /// Gets or creates a session context for the specified batch size.
  /// Rounds up to the nearest batch size anchor.
  /// </summary>
  private SessionForBatchSize GetOrCreateSessionForBatchSize(int batchSize, bool disableCUDAGraphs = false)
  {
    int effectiveBatchSize = Math.Max(MinBatchSize, batchSize);

    bool useCudaGraphs = !disableCUDAGraphs && DetermineIfCUDAGraphsShouldBeUsed(effectiveBatchSize);

    (int min, int max, bool useCudaGraphs) bucketKey = FindBatchSizeBucket(batchSize, useCudaGraphs);
    if (sessionCache.TryGetValue(bucketKey, out var context))
    {
      return context;
    }

    if (UseTensorRT)
    {
      // Trigger calculation of the target cache directory (to force user output message to appear first).
      _ = GetTRTEngineCacheDir();
    }

    // Create new session context for this batch size anchor
    if (bucketKey.useCudaGraphs)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"[{GPUID}]: Creating new graph session for batch size {bucketKey.max} (requested: {batchSize}) ... ", endLine: false);
    }
    else
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"[{GPUID}]: Creating new non-graph session for batch size bucket [{bucketKey.min}..{bucketKey.max}] (requested: {batchSize}) ... ", endLine: false);
    }

    // Take a global lock during session creation for two reasons:
    //   - we don't want to try to create TensorRT engines for the same network simultaneously
    //     because this would interfere with performance profiling,
    //     and the second time a cached version should just be used rather than recreated
    //   - reduce likelihood of conflict with other initialization (e.g. other threads using ManagedCUDA).
    TimingStats stats = new();
    using (new TimingBlock(stats, TimingBlock.LoggingType.None))
    {
      lock (CUDADevice.InitializingCUDAContextLockObj)
      {
        context = CreateSessionForBatchSize(bucketKey.min, bucketKey.max, useCudaGraphs);
      }
    }
    sessionCache[bucketKey] = context;

    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "done in " + Math.Round(stats.ElapsedTimeSecs, 2) + " seconds.");

    if (context.UsesCUDAGraphs)
    {
      foreach (var (name, ortValue, _, cudaBuffer) in context.OutputOrtValues)
      {
        context.IoBinding.BindOutput(name, ortValue);
      }
    }

    return context;
  }


  private bool DetermineIfCUDAGraphsShouldBeUsed(int effectiveBatchSize)
  {
    // Determine if CUDA graphs should be used based on:
    // 1. CUDA graphs are configured (BATCH_SIZE_ANCHORS_WITH_GRAPH is not null)
    // 2. There exists a suitable batch size range
    // 3. The effective batch size is close enough to the range maximum (within 50% of anchor value)
    bool useCudaGraphs = false;
    if (BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED != null && BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED.Length > 0)
    {
      // Find the smallest anchor that can accommodate the effective batch size
      int? closestGraphMaxBatch = null;
      foreach (int anchor in BATCH_SIZE_ANCHORS_WITH_GRAPH_ADJUSTED)
      {
        if (effectiveBatchSize <= anchor)
        {
          closestGraphMaxBatch = anchor;
          break;
        }
      }

      // Use CUDA graphs if we found a suitable range and the batch size is within 40% of the anchor
      // (or if the unused positions would be very small, i.e., less than 12).
      if (closestGraphMaxBatch.HasValue)
      {
        float maxUnusedPositions = Math.Max(12, closestGraphMaxBatch.Value * 0.40f);
        if ((closestGraphMaxBatch.Value - effectiveBatchSize) <= maxUnusedPositions)
        {
          useCudaGraphs = true;
        }
      }
    }

    return useCudaGraphs;
  }


  /// <summary>
  /// Converts a ManagedCUDA CUdeviceptr to an IntPtr for use with ONNX Runtime.
  /// </summary>
  /// <param name="dptr"></param>
  /// <returns></returns>
  static IntPtr CUdeviceptrToIntPtr(CUdeviceptr dptr) => (IntPtr)(long)(ulong)dptr;


  /// <summary>
  /// Creates a new InferenceSession and SessionContext for a specific batch size.
  /// Each session gets its own TensorRT engine optimized for that batch size,
  /// enabling proper CUDA graph capture.
  /// </summary>
  private SessionForBatchSize CreateSessionForBatchSize(int minBatch, int maxBatch, bool useCudaGraphs)
  {
    var (shortID, inputNames, nonBatchDimensions, precisionNumBits,
         gpuID, useTensorRT, minBatchSize, maxBatchSize,
         enableProfiling, outputNamesToRetrieve) = sessionCreationParams;

    // Create session options
    using SessionOptions so = CreateSessionOptions(sessionCreationParams, minBatch, maxBatch, useCudaGraphs);

    // Create the session
    InferenceSession newSession;
    lock (CUDADevice.InitializingCUDAContextLockObj)
    {
      newSession = new InferenceSession(storedOnnxModelBytes, so);
    }

    // Extract and filter metadata from this session
    IReadOnlyDictionary<string, NodeMetadata> sessionInputMetadata = FilterMetadata(newSession.InputMetadata);
    IReadOnlyDictionary<string, NodeMetadata> sessionOutputMetadata = newSession.OutputMetadata;

    OrtIoBinding newIoBinding = newSession.CreateIoBinding();
    SessionForBatchSize context = new SessionForBatchSize
    {
      BatchSizeMin = minBatch,
      BatchSizeMax = maxBatch,
      UsesCUDAGraphs = useCudaGraphs,
      Session = newSession,
      IoBinding = newIoBinding,
      InputOrtValues = new List<(string, OrtValue, long[], CudaDeviceVariable<byte>, CudaDeviceVariable<Float16>)>(),
      OutputOrtValues = new List<(string, OrtValue, long[], CudaDeviceVariable<Float16>)>(),
      InputBuffers = new List<Array>(),
      OutputBuffers = new List<(string, NodeMetadata, Array)>()
    };

    // Determine memory allocation type based on CUDA graphs usage
    OrtMemoryInfo memInfo;
    OrtAllocator allocator = null;

    if (useCudaGraphs)
    {
      // For CUDA graphs, use GPU-allocated pinned memory
      memInfo = new(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, gpuID, OrtMemType.CpuOutput);
      allocator = new OrtAllocator(newSession, memInfo);
    }
    else
    {
      // Use CPU memory for non-CUDA graph scenarios
      memInfo = OrtMemoryInfo.DefaultInstance;
    }

    // Create OrtValues for inputs
    if (InputsNumBits < 32)
    {
      if (InputsNumBits == 8)
      {
        (string name, NodeMetadata metadata, bool isKnownShape, byte[] value)[] inputBuffersByte = ONNXHelpers.CreateBuffers<byte>(sessionInputMetadata, maxBatch);
        context.InputBuffers = inputBuffersByte.Select(b => b.value as Array).ToList();
        for (int i = 0; i < inputBuffersByte.Length; i++)
        {
          (string name, NodeMetadata metadata, bool isKnownShape, byte[] value) buffer = inputBuffersByte[i];
          long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

          OrtValue ortValue;
          CudaDeviceVariable<byte> cudaBuffer = default;
          if (useCudaGraphs)
          {
            // Allocate GPU memory for CUDA graphs
            //            ortValue = OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.UInt8, shape);
            long numElements = ONNXHelpers.ProductDimensions(buffer.metadata.Dimensions, maxBatch);
            cudaDevice.SetCurrent();
            cudaBuffer = new CudaDeviceVariable<byte>(numElements);

            // 5) Wrap existing DEVICE pointers as OrtValue tensors (no copies; ORT doesn't own memory)
            ortValue = OrtValue.CreateTensorValueWithData(memInfo, TensorElementType.UInt8, shape,
                                                          CUdeviceptrToIntPtr(cudaBuffer.DevicePointer), sizeof(byte) * numElements);
          }
          else
          {
            ortValue = default; // Will create on-demand in RunWithDirectRun
          }

          context.InputOrtValues.Add((buffer.name, ortValue, shape, cudaBuffer, default));
        }
      }
      else if (InputsNumBits == 16)
      {
        (string name, NodeMetadata metadata, bool isKnownShape, Float16[] value)[] inputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(sessionInputMetadata, maxBatch);
        context.InputBuffers = inputBuffers16.Select(b => b.value as Array).ToList();
        for (int i = 0; i < inputBuffers16.Length; i++)
        {
          (string name, NodeMetadata metadata, bool isKnownShape, Float16[] value) buffer = inputBuffers16[i];
          long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

          OrtValue ortValue;
          CudaDeviceVariable<Float16> cudaBufferFloat16 = default;

          if (useCudaGraphs)
          {
            long numElements = ONNXHelpers.ProductDimensions(buffer.metadata.Dimensions, maxBatch);
            cudaDevice.SetCurrent();
            cudaBufferFloat16 = new CudaDeviceVariable<Float16>(numElements);
            // 5) Wrap existing DEVICE pointers as OrtValue tensors (no copies; ORT doesn't own memory)
            ortValue = OrtValue.CreateTensorValueWithData(memInfo, TensorElementType.Float16, shape,
                                                          CUdeviceptrToIntPtr(cudaBufferFloat16.DevicePointer),
                                                          2 * numElements);
          }
          else
          {
            ortValue = default; // Will create on-demand in RunWithDirectRun
          }

          context.InputOrtValues.Add((buffer.name, ortValue, shape, default, cudaBufferFloat16));
        }
      }
      else
      {
        throw new Exception(shortID + ": Unsupported input precision (" + InputsNumBits + ")");
      }

      (string name, NodeMetadata metadata, bool isKnownShape, Float16[] value)[] outputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(sessionOutputMetadata, maxBatch, outputNamesToRetrieve);
      foreach (var outputBuffer in outputBuffers16)
      {
        context.OutputBuffers.Add((outputBuffer.name, outputBuffer.metadata, outputBuffer.value));
        long[] shape = ONNXHelpers.ToLongArray(outputBuffer.metadata.Dimensions, maxBatch);

        OrtValue ortValue;
        CudaDeviceVariable<Float16> cudaBuffer = default;

        if (useCudaGraphs)
        {
          // Allocate GPU memory for CUDA graphs
          //          ortValue = OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float16, shape);
          long numElements = ONNXHelpers.ProductDimensions(outputBuffer.metadata.Dimensions, maxBatch);
          cudaDevice.SetCurrent();
          cudaBuffer = new CudaDeviceVariable<Float16>(numElements);

          ortValue = OrtValue.CreateTensorValueWithData(memInfo, TensorElementType.Float16, shape,
                                                        CUdeviceptrToIntPtr(cudaBuffer.DevicePointer),
                                                        Marshal.SizeOf<Float16>() * numElements);
        }
        else
        {
          ortValue = default;
        }

        context.OutputOrtValues.Add((outputBuffer.name, ortValue, shape, cudaBuffer));
      }
    }
    else
    {
      // Use float32 buffers
      var inputBuffers32 = ONNXHelpers.CreateBuffers<float>(sessionInputMetadata, maxBatch);
      context.InputBuffers = inputBuffers32.Select(b => b.value as Array).ToList();
      for (int i = 0; i < inputBuffers32.Length; i++)
      {
        (string name, NodeMetadata metadata, bool isKnownShape, float[] value) buffer = inputBuffers32[i];
        long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

        OrtValue ortValue;
        CudaDeviceVariable<byte> cudaBuffer = default;

        if (useCudaGraphs)
        {
          ortValue = OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, shape);
          throw new Exception("need to adjust data type from byte to Float16 and allocate here");
          cudaDevice.SetCurrent();
          //cudaBuffer = new CudaDeviceVariable<float>(ONNXHelpers.ProductDimensions(buffer.metadata.Dimensions, maxBatch) * sizeof(float));
        }
        else
        {
          ortValue = default;
        }

        context.InputOrtValues.Add((buffer.name, ortValue, shape, cudaBuffer, default));
      }

      var outputBuffers32 = ONNXHelpers.CreateBuffers<float>(sessionOutputMetadata, maxBatch, outputNamesToRetrieve);
      foreach (var buffer in outputBuffers32)
      {
        context.OutputBuffers.Add((buffer.name, buffer.metadata, buffer.value));
        long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

        OrtValue ortValue;
        CudaDeviceVariable<Float16> cudaBuffer = default;

        if (useCudaGraphs)
        {
          ortValue = OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, shape);
          cudaDevice.SetCurrent();
          //cudaBuffer = new CudaDeviceVariable<Float16>(ONNXHelpers.ProductDimensions(buffer.metadata.Dimensions, maxBatch));
          throw new Exception("need to adjust data type from Float16 to float and allocate here");
        }
        else
        {
          ortValue = default;
        }

        context.OutputOrtValues.Add((buffer.name, ortValue, shape, cudaBuffer));
      }
    }

    // Store allocators in context so they can be disposed later
    context.Allocator = allocator;

    return context;
  }


  /// <summary>
  /// NOTE CURRENTLY NOT USED - need to wait for multiprofile support (currently TensorRT-RTX EP only)
  /// </summary>
  static (string min, string opt, string max) BuildTrtProfiles(string inputName, int[] anchors, string nonBatchDimensions, int globalMin, int globalMax)
  {
    if (!anchors.SequenceEqual(anchors.OrderBy(x => x)))
    {
      throw new ArgumentException("anchors must be increasing");
    }
    if (globalMin > anchors.First() || globalMax < anchors.Last())
    {
      throw new ArgumentException("global range must cover anchors");
    }

    // Partition [globalMin..globalMax] by midpoints between anchors
    List<int> mins = new(anchors.Length);
    List<int> opts = new(anchors);
    List<int> maxs = new(anchors.Length);

    mins.Add(globalMin);
    for (int i = 0; i < anchors.Length - 1; i++)
    {
      int boundary = (anchors[i] + anchors[i + 1]) / 2; // floor midpoint
      maxs.Add(boundary);
      mins.Add(boundary + 1);
    }
    maxs.Add(globalMax);

    string Join(string name, IEnumerable<int> batches) =>
    string.Join(",", batches.Select(b => $"{name}:{b}x{nonBatchDimensions}"));

    return (Join(inputName, mins), Join(inputName, opts), Join(inputName, maxs));

    // Example resulting strings
    // min : "squares:1x64x137,squares:9x64x137,squares:41x64x137,squares:97x64x137,squares:193x64x137"
    // opt : "squares:1x64x137,squares:16x64x137,squares:64x64x137,squares:128x64x137,squares:256x64x137"
    // max : "squares:8x64x137,squares:40x64x137,squares:96x64x137,squares:192x64x137,squares:256x64x137"
  }


  /// <summary>
  /// Context for a specific batch size session, including session and IOBinding.
  /// </summary>
  private class SessionForBatchSize : IDisposable
  {
    public int BatchSizeMin { get; set; }
    public int BatchSizeMax { get; set; }
    public bool UsesCUDAGraphs { get; set; }
    public InferenceSession Session { get; set; }
    public OrtIoBinding IoBinding { get; set; }
    public List<(string name, OrtValue ortValue, long[] shape, CudaDeviceVariable<byte> cudaBufferByte, CudaDeviceVariable<Float16> cudaBufferFloat16)> InputOrtValues { get; set; }
    public List<(string name, OrtValue ortValue, long[] shape, CudaDeviceVariable<Float16> cudaBuffer)> OutputOrtValues { get; set; }
    public List<Array> InputBuffers { get; set; }
    public List<(string name, NodeMetadata metadata, Array buffer)> OutputBuffers { get; set; }
    public OrtAllocator Allocator { get; set; }

    public void Dispose()
    {
      foreach ((string _, OrtValue ortValue, long[] _, CudaDeviceVariable<byte> cudaBuffer, CudaDeviceVariable<Float16> cudaBufferFloat16) in InputOrtValues)
      {
        ortValue?.Dispose();
        cudaBuffer?.Dispose();
        cudaBufferFloat16?.Dispose();
      }
      foreach ((string _, OrtValue ortValue, long[] _, CudaDeviceVariable<Float16> cudaBuffer) in OutputOrtValues)
      {
        ortValue?.Dispose();
        cudaBuffer?.Dispose();
      }

      InputOrtValues?.Clear();
      OutputOrtValues?.Clear();
      IoBinding?.Dispose();
      Session?.Dispose();
      Allocator?.Dispose();
    }
  }
}
