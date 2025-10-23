#region Using directives

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ceres.Base.Benchmarking;
using Ceres.Base.CUDA;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using Microsoft.ML.OnnxRuntime;
using Onnx;

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

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
  public readonly string[] MultiNetNames;


  /// The weights to be used for inference of the sub-networks if the net is a specially 
  /// prepared Ceres multinet network (containing the string "multinet" in the file name).
  public readonly float[] MultiNetWeights;

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
  /// Batch size anchors for CUDA graph sessions.
  /// Actual batch sizes will be rounded up to the nearest anchor.
  /// </summary>
  private readonly int[] BATCH_SIZE_ANCHORS;


  public readonly int InputsNumBits;


  /// <summary>
  /// Batch sizes below this threshold will use CUDA graphs (if TensorRT is enabled).
  /// Default value is 12. Set to 0 to disable CUDA graphs for all batch sizes.
  /// Set to int.MaxValue to enable CUDA graphs for all batch sizes.
  /// </summary>
  public int UseCUDAGraphsBelowBatchSize { get; private set; } = 0; // disable, non-functional currently (see comments below)

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
      int minBatchSize,
      int maxBatchSize,
      bool enableProfiling,
      bool retainRawOutputs,
      string[] outputNamesToRetrieve = null,
      string loraAdapterFileName = null)
  {
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
    bool multiEngineMode = useTensorRT;// ONNXFileName?.ToLower().Contains("copy") ?? false;
    UseMultipleProfilesPerSession = false;

    //      BATCH_SIZE_ANCHORS = testMode ? [16, 64] : [];
    BATCH_SIZE_ANCHORS = multiEngineMode ? [16, 64, 256] : [];
    UseCUDAGraphsBelowBatchSize = 0;  // disabled due to ONNXRuntime bugs (see below)


    if (UseCUDAGraphsBelowBatchSize > 0)
    {
      throw new NotImplementedException("CUDA graphs not currently supported due to ONNXRuntime bugs (see issues 22583 and 24453).");

      // Graph capture/replay only works one time: https://github.com/microsoft/onnxruntime/issues/22583
      // Device memory allocations only work on device with index 0: https://github.com/microsoft/onnxruntime/issues/24453
    }

    if (onnxFileName != null && !File.Exists(onnxFileName))
    {
      throw new Exception("ONNX file not found: " + onnxFileName);
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

    // Extract multinet metadata if applicable
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
    RetainRawInputs = retainRawOutputs;
    LoRAAdapterFileName = loraAdapterFileName;

    // Touch OrtEnv instance (required on Linux)
    OrtEnv ortInstance = OrtEnv.Instance();

    // Initialize session cache
    sessionCache = new Dictionary<(int, int, bool), SessionForBatchSize>();

    Console.WriteLine($"ONNXExecutor initialized on GPU {GPUID} (sessions will be created on-demand)");
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
      so = CreateTensorRTSessionOptions(shortID, inputNames, nonBatchDimensions, gpuID,
           precisionNumBits, minBatch, maxBatch, useCudaGraphs);
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


  /// <summary>
  /// Creates TensorRT-specific session options.
  /// </summary>
  private SessionOptions CreateTensorRTSessionOptions(string shortID, string[] inputNames,
            string nonBatchDimensions, int gpuID,
int precisionNumBits, int minBatch, int maxBatch, bool useCudaGraphs)
  {
    OrtTensorRTProviderOptions trtProviderOptions = new OrtTensorRTProviderOptions();
    Dictionary<string, string> providerOptionsDict = new();

    providerOptionsDict["device_id"] = gpuID.ToString();
    providerOptionsDict["trt_max_workspace_size"] = (8L * 1024 * 1024 * 1024).ToString();

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
        if (BATCH_SIZE_ANCHORS.Length == 0)
        {
          // Only one range with maximum probably very large (MaxBatchSize).
          // Don't use midpoint with MinBatchSize (would probably be atypically large).
          optimalBatchSize = Math.Min(maxBatch, 128);
        }
        else
        {
          // Use midpoint
          optimalBatchSize = (int)MathUtils.RoundedUp((minBatch + maxBatch) / 2, 2);
        }

        providerOptionsDict["trt_profile_min_shapes"] = MakeShapeStr(minBatch);
        providerOptionsDict["trt_profile_opt_shapes"] = MakeShapeStr((minBatch + maxBatch) / 2);
        providerOptionsDict["trt_profile_max_shapes"] = MakeShapeStr(maxBatch);
      }
    }

    // Engine cache configuration
    string directoryName = ONNXFileName == null ? Path.GetTempPath() : new FileInfo(ONNXFileName).DirectoryName;
    string trtSubdirectory = Path.Combine(directoryName, "trt_engines", Environment.MachineName);
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

    Console.WriteLine("TensorRT engines will be cached in: " + trtSubdirectory);

    providerOptionsDict["trt_engine_cache_enable"] = "1";
    providerOptionsDict["trt_engine_cache_path"] = trtSubdirectory;

    string cachePrefix = FileUtils.FileNameSanitized(shortID);
    cachePrefix += $"_gpu{GPUID}_bs{minBatch}-{maxBatch}";
    if (useCudaGraphs)
    {
      cachePrefix += "_graph";
    }

    providerOptionsDict["trt_engine_cache_prefix"] = cachePrefix;
    providerOptionsDict["trt_timing_cache_enable"] = "1";
    providerOptionsDict["trt_timing_cache_path"] = trtSubdirectory;
    providerOptionsDict["trt_force_timing_cache"] = "1";

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
    OrtCUDAProviderOptions cudaProviderOptions = new();
    Dictionary<string, string> providerOptionsDict = new();
    providerOptionsDict["device_id"] = gpuID.ToString();
    cudaProviderOptions.UpdateOptions(providerOptionsDict);
    return SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
  }


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
    var inputsONNX = ValidateAndProcessInputs(inputs, batchSize);

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
    if (batchSize < MinBatchSize)
    {
      throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
    }

    return RunWithIOBinding<byte, byte>(inputs, batchSize);
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
    if (batchSize < MinBatchSize)
    {
      throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
    }

    return RunWithIOBinding<Half, Float16>(inputs, batchSize);
  }


  public Memory<TDest> InputBufferForBatchSize<T, TDest>(int inputIndex, int batchSize)
                         where T : unmanaged
                         where TDest : unmanaged
  {
    int evaluatorBatchSize = Math.Max(MinBatchSize, batchSize);
    T[] rawArray = ((T[])GetOrCreateSessionForBatchSize(evaluatorBatchSize).InputBuffers[inputIndex]);
    int numElements = ONNXHelpers.ProductDimensions(InputsMetadata.ElementAt(inputIndex).Value.Dimensions, batchSize);

    return MemoryCasted.AsMemory<T, TDest>(rawArray).Slice(0, numElements);
  }


  /// <summary>
  /// Executes inference using IOBinding (required for CUDA graphs).
  /// </summary>
  private List<(string, Memory<Float16>)> RunWithIOBinding<TInput, TBuffer>(
    (Memory<TInput> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
      where TInput : unmanaged
      where TBuffer : unmanaged
  {
    List<(string, Memory<Float16>)> resultArrays = new();

    lock (lockObject)
    {
      SessionForBatchSize context = GetOrCreateSessionForBatchSize(batchSize);

      if (!InputBuffersArePrepopulated)
      {
        // Copy into backing CPU buffers
        for (int i = 0; i < inputs.Length; i++)
        {
          (Memory<TInput> input, int[] shape, string _, int numElements) = inputs[i];
          Array destArray = context.InputBuffers[i];

          if (typeof(TInput) == typeof(byte))
          {
            Span<byte> src = MemoryMarshal.Cast<TInput, byte>(input.Span);
            src.Slice(0, numElements).CopyTo(((byte[])destArray).AsSpan(0, numElements));
          }
          else if (typeof(TInput) == typeof(Half))
          {
            Span<Float16> src = MemoryMarshal.Cast<Half, Float16>(MemoryMarshal.Cast<TInput, Half>(input.Span));
            src.Slice(0, numElements).CopyTo(((Float16[])destArray).AsSpan(0, numElements));
          }
          else if (typeof(TInput) == typeof(float))
          {
            Span<float> src = MemoryMarshal.Cast<TInput, float>(input.Span);
            src.Slice(0, numElements).CopyTo(((float[])destArray).AsSpan(0, numElements));
          }
          else
          {
            throw new NotSupportedException($"Unsupported input type {typeof(TInput)}");
          }
        }
      }

      // If not using CUDA graphs, rebind with shapes for the current batch size
      if (!context.UsesCUDAGraphs)
      {
        // Clear previous bindings
        context.IoBinding.ClearBoundInputs();
        context.IoBinding.ClearBoundOutputs();

        OrtMemoryInfo cpuMemInfo = OrtMemoryInfo.DefaultInstance;

        // Bind per-run inputs with shape = batchSize
        for (int i = 0; i < context.InputOrtValues.Count; i++)
        {
          var (name, _, shape) = context.InputOrtValues[i];
          NodeMetadata meta = InputsMetadata[name];
          shape[0] = batchSize;
          int count = ONNXHelpers.ProductDimensions(meta.Dimensions, batchSize);

          if (InputsNumBits == 8)
          {
            byte[] buf = (byte[])context.InputBuffers[i];
            Memory<byte> mem = new Memory<byte>(buf, 0, count);
            using OrtValue ortVal = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, mem, shape);
            context.IoBinding.BindInput(name, ortVal);
          }
          else if (InputsNumBits == 16)
          {
            Float16[] buf = (Float16[])context.InputBuffers[i];
            Memory<Float16> mem = new Memory<Float16>(buf, 0, count);
            using OrtValue ortVal = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, mem, shape);
            context.IoBinding.BindInput(name, ortVal);
          }
          else // 32
          {
            float[] buf = (float[])context.InputBuffers[i];
            Memory<float> mem = new Memory<float>(buf, 0, count);
            using OrtValue ortVal = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, mem, shape);
            context.IoBinding.BindInput(name, ortVal);
          }
        }

        // Bind per-run outputs with shape = batchSize
        for (int i = 0; i < context.OutputBuffers.Count; i++)
        {
          var (name, metadata, buffer) = context.OutputBuffers[i];
          long[] shape = ONNXHelpers.ToLongArray(metadata.Dimensions, batchSize);

          if (buffer is Float16[] f16)
          {
            int count = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);
            Memory<Float16> mem = new Memory<Float16>(f16, 0, count);
            using OrtValue ortVal = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, mem, shape);
            context.IoBinding.BindOutput(name, ortVal);
          }
          else if (buffer is float[] f32)
          {
            int count = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);
            Memory<float> mem = new Memory<float>(f32, 0, count);
            using OrtValue ortVal = OrtValue.CreateTensorValueFromMemory<float>(cpuMemInfo, mem, shape);
            context.IoBinding.BindOutput(name, ortVal);
          }
          else
          {
            throw new NotSupportedException($"Unsupported output buffer type for {name}");
          }
        }
      }
      else
      {
        // CUDA graphs path: inputs/outputs are pre-bound at bucket max shape
      }

      context.IoBinding.SynchronizeBoundInputs();
      context.Session.RunWithBinding(runOptions, context.IoBinding);
      context.IoBinding.SynchronizeBoundOutputs();

      // Read only actualElements from output buffers
      for (int i = 0; i < context.OutputOrtValues.Count; i++)
      {
        string outputName = context.OutputOrtValues[i].name;
        NodeMetadata metadata = context.OutputBuffers[i].metadata;
        Array cpuBuffer = context.OutputBuffers[i].buffer;
        int actualElements = ONNXHelpers.ProductDimensions(metadata.Dimensions, batchSize);

        if (cpuBuffer is Float16[] float16Buffer)
        {
          resultArrays.Add((outputName, new Memory<Float16>(float16Buffer, 0, actualElements)));
        }
        else if (cpuBuffer is float[] floatBuffer)
        {
          Float16[] float16Result = new Float16[actualElements];
          for (int j = 0; j < actualElements; j++)
          {
            float16Result[j] = (Float16)floatBuffer[j];
          }
          resultArrays.Add((outputName, float16Result));
        }
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

    return resultArrays;
  }


  /// <summary>
  /// Ends profiling.
  /// </summary>
  public void EndProfiling()
  {
    foreach (var context in sessionCache.Values)
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
        foreach (var context in sessionCache.Values)
        {
          context.Dispose();
        }
        sessionCache.Clear();
      }

      runOptions.Dispose();
      disposed = true;
    }
  }


  private (int min, int max, bool useCudaGraphs) FindBatchSizeBucket(int batchSize)
  {
    int currentMin = MinBatchSize;
    foreach (int anchor in BATCH_SIZE_ANCHORS)
    {
      int currentMax = anchor - 1;
      if (batchSize >= currentMin && batchSize <= currentMax)
      {
        bool useCudaGraphs = batchSize < UseCUDAGraphsBelowBatchSize;
        return (currentMin, currentMax, useCudaGraphs);
      }
      currentMin = anchor;
    }

    // If batchSize is larger than or equal to the last anchor, it belongs to the last bucket.
    // The max of this bucket is the global maxBatchSize for the executor.
    bool useCudaGraphsForLast = batchSize < UseCUDAGraphsBelowBatchSize;
    return (currentMin, maxBatchSize, useCudaGraphsForLast);
  }


  /// <summary>
  /// Gets or creates a session context for the specified batch size.
  /// Rounds up to the nearest batch size anchor.
  /// </summary>
  private SessionForBatchSize GetOrCreateSessionForBatchSize(int batchSize)
  {
    (int min, int max, bool useCudaGraphs) bucketKey = FindBatchSizeBucket(batchSize);
    if (sessionCache.TryGetValue(bucketKey, out var context))
    {
      return context;
    }

    // Create new session context for this batch size anchor
    string graphStatus = bucketKey.useCudaGraphs ? "with CUDA graphs" : "without CUDA graphs";
    Console.WriteLine($"Creating new session for batch size bucket [{bucketKey.min}..{bucketKey.max}] {graphStatus} (requested: {batchSize})");
    context = CreateSessionForBatchSize(bucketKey.min, bucketKey.max);
    sessionCache[bucketKey] = context;

    // Bind inputs and outputs once. This is crucial for CUDA graphs.
    // The bindings will be reused for every run with this session context.
    foreach (var (name, ortValue, _) in context.InputOrtValues)
    {
      context.IoBinding.BindInput(name, ortValue);
    }
    foreach (var (name, ortValue, _) in context.OutputOrtValues)
    {
      context.IoBinding.BindOutput(name, ortValue);
    }

    return context;
  }

  /// <summary>
  /// Creates a new InferenceSession and SessionContext for a specific batch size.
  /// Each session gets its own TensorRT engine optimized for that batch size,
  /// enabling proper CUDA graph capture.
  /// </summary>
  private SessionForBatchSize CreateSessionForBatchSize(int minBatch, int maxBatch)
  {
    var (shortID, inputNames, nonBatchDimensions, precisionNumBits,
     gpuID, useTensorRT, minBatchSize, maxBatchSize,
         enableProfiling, outputNamesToRetrieve) = sessionCreationParams;

    // Determine if CUDA graphs should be enabled for this batch size range
    bool useCudaGraphs = maxBatch < UseCUDAGraphsBelowBatchSize;

    // Create session options
    SessionOptions so = CreateSessionOptions(sessionCreationParams, minBatch, maxBatch, useCudaGraphs);

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
      InputOrtValues = new List<(string, OrtValue, long[])>(),
      OutputOrtValues = new List<(string, OrtValue, long[])>(),
      InputBuffers = new List<Array>(),
      OutputBuffers = new List<(string, NodeMetadata, Array)>()
    };

    // Use CPU memory for OrtValues - IOBinding will handle GPU transfer
    OrtMemoryInfo cpuMemInfo = OrtMemoryInfo.DefaultInstance;

    // Create OrtValues for inputs - allocated on CPU
    if (InputsNumBits < 32)
    {
      if (InputsNumBits == 8)
      {
        var inputBuffersByte = ONNXHelpers.CreateBuffers<byte>(sessionInputMetadata, maxBatch);
        context.InputBuffers = inputBuffersByte.Select(b => b.value as Array).ToList();
        for (int i = 0; i < inputBuffersByte.Length; i++)
        {
          var buffer = inputBuffersByte[i];
          long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

          // Create OrtValue on CPU memory
          OrtValue ortValue = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, new Memory<byte>(buffer.value), shape);

          context.InputOrtValues.Add((buffer.name, ortValue, shape));
        }
      }
      else if (InputsNumBits == 16)
      {
        var inputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(sessionInputMetadata, maxBatch);
        context.InputBuffers = inputBuffers16.Select(b => b.value as Array).ToList();
        for (int i = 0; i < inputBuffers16.Length; i++)
        {
          var buffer = inputBuffers16[i];
          long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

          // Create OrtValue on CPU memory
          OrtValue ortValue = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, new Memory<Float16>(buffer.value), shape);

          context.InputOrtValues.Add((buffer.name, ortValue, shape));
        }
      }
      else
      {
        throw new Exception(shortID + ": Unsupported input precision (" + InputsNumBits + ")");
      }

      // Create OrtValues for outputs - allocated on CPU
      var outputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(sessionOutputMetadata, maxBatch, outputNamesToRetrieve);
      foreach (var buffer in outputBuffers16)
      {
        context.OutputBuffers.Add((buffer.name, buffer.metadata, buffer.value));
        long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

        // Create OrtValue on CPU memory
        OrtValue ortValue = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, new Memory<Float16>(buffer.value), shape);

        context.OutputOrtValues.Add((buffer.name, ortValue, shape));
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

        // Create OrtValue on CPU memory
        OrtValue ortValue = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, new Memory<float>(buffer.value), shape);

        context.InputOrtValues.Add((buffer.name, ortValue, shape));
      }

      var outputBuffers32 = ONNXHelpers.CreateBuffers<float>(sessionOutputMetadata, maxBatch, outputNamesToRetrieve);
      foreach (var buffer in outputBuffers32)
      {
        context.OutputBuffers.Add((buffer.name, buffer.metadata, buffer.value));
        long[] shape = ONNXHelpers.ToLongArray(buffer.metadata.Dimensions, maxBatch);

        // Create OrtValue on CPU memory
        OrtValue ortValue = OrtValue.CreateTensorValueFromMemory(cpuMemInfo, new Memory<float>(buffer.value), shape);

        context.OutputOrtValues.Add((buffer.name, ortValue, shape));
      }
    }

    Console.WriteLine($"Created session context for batch size {maxBatch} with {context.InputOrtValues.Count} inputs and {context.OutputOrtValues.Count} outputs");
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
    public List<(string name, OrtValue ortValue, long[] shape)> InputOrtValues { get; set; }
    public List<(string name, OrtValue ortValue, long[] shape)> OutputOrtValues { get; set; }
    public List<Array> InputBuffers { get; set; }
    public List<(string name, NodeMetadata metadata, Array buffer)> OutputBuffers { get; set; }

    public void Dispose()
    {
      foreach ((string _, OrtValue ortValue, long[] _) in InputOrtValues)
      {
        ortValue?.Dispose();
      }
      foreach ((string _, OrtValue ortValue, long[] _) in OutputOrtValues)
      {
        ortValue?.Dispose();
      }
      InputOrtValues?.Clear();
      OutputOrtValues?.Clear();
      IoBinding?.Dispose();
      Session?.Dispose();
    }
  }
}

/// <summary>
/// TODO: Move this into Ceres.Base
/// </summary>
public static class MemoryCasted
{
  public static Memory<TDest> AsMemory<TSrc, TDest>(TSrc[] array)
      where TSrc : unmanaged
      where TDest : unmanaged
  {
    if (array is null) { return Memory<TDest>.Empty; }
    if (Unsafe.SizeOf<TSrc>() != Unsafe.SizeOf<TDest>())
    {
      throw new NotSupportedException("TSrc and TDest must be the same size.");
    }
    // Types must be blittable / no refs.
    if (RuntimeHelpers.IsReferenceOrContainsReferences<TSrc>() ||
        RuntimeHelpers.IsReferenceOrContainsReferences<TDest>())
    {
      throw new NotSupportedException("TSrc/TDest must be unmanaged (no references).");
    }

    return new ReinterpretingArrayMemoryManager<TSrc, TDest>(array).Memory;
  }

  private sealed class ReinterpretingArrayMemoryManager<TFrom, TTo> : MemoryManager<TTo>
      where TFrom : unmanaged
      where TTo : unmanaged
  {
    private readonly TFrom[] _array;
    private MemoryHandle _pinned;
    private bool _isPinned;

    public ReinterpretingArrayMemoryManager(TFrom[] array)
    {
      _array = array ?? Array.Empty<TFrom>();
    }

    public override Span<TTo> GetSpan()
    {
      if (_array.Length == 0) { return Span<TTo>.Empty; }

      // Reinterpret ref to first element, then create a span of same element count.
      ref TFrom srcRef = ref MemoryMarshal.GetArrayDataReference(_array);
      ref TTo dstRef = ref Unsafe.As<TFrom, TTo>(ref srcRef);
      return MemoryMarshal.CreateSpan(ref dstRef, _array.Length);
    }

    public override unsafe MemoryHandle Pin(int elementIndex = 0)
    {
      if (_isPinned) { throw new InvalidOperationException("Already pinned."); }
      if ((uint)elementIndex > (uint)_array.Length)
      {
        throw new ArgumentOutOfRangeException(nameof(elementIndex));
      }

      // Pin the source array and adjust the pointer for the TTo element index.
      _pinned = new Memory<TFrom>(_array).Pin();
      _isPinned = true;

      byte* basePtr = (byte*)_pinned.Pointer;
      byte* adjPtr = basePtr + (nuint)elementIndex * (nuint)Unsafe.SizeOf<TTo>();

      // Tie lifetime to this manager so Unpin() is called on dispose.
      return new MemoryHandle(adjPtr, default, this);
    }

    public override void Unpin()
    {
      if (_isPinned)
      {
        _pinned.Dispose();
        _isPinned = false;
      }
    }

    protected override void Dispose(bool disposing)
    {
      if (disposing && _isPinned)
      {
        _pinned.Dispose();
        _isPinned = false;
      }
    }

  }
}
