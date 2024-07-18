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
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Linq;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Ceres.Base.CUDA;
using Ceres.Base.Benchmarking;
using Ceres.Base.Misc;

using Chess.Ceres.NNEvaluators;
using Microsoft.Extensions.Primitives;


#endregion


/// <summary>
/// Manages evaluation of neural networks using ONNX runtime.
/// 
/// Although ONNX the documentation stats that multiple threads can invoke the Run() method
/// on the same inference session object, we have single-instance buffers for inputs and outputs
/// and therefore take locks to enforce single-threaded access.
/// 
/// TODO: some of the clients of this class could possibly pass in a restricted list of outputNames
///       to eliminate overhead of retrieving values for outputs which may not be needed in some situations.
/// </summary>
namespace Ceres.Chess.LC0NetInference
{
  public class NetExecutorONNXRuntime : IDisposable
  {
    const int MAX_BATCH_SIZE = 1024;

    /// <summary>
    /// Name of underlying ONNX file;
    /// </summary>
    public readonly String ONNXFileName;

    /// <summary>
    /// Underlying ONNX runtime session
    /// </summary>
    public readonly InferenceSession Session;

    /// <summary>
    /// ID (index) of GPU to use
    /// </summary>
    public readonly int GPUID;

    /// <summary>
    /// Session data type precision to use.
    /// </summary>
    public readonly NNEvaluatorPrecision Precision;


    readonly object lockObject = new object();
    bool disposed;

    IReadOnlyDictionary<string, NodeMetadata> inputsMetadata;

    (string name, NodeMetadata metadata, Float16[] value)[] inputBuffers16;
    (string name, NodeMetadata metadata, Float16[] value)[] outputBuffers16;
    (string name, NodeMetadata metadata, float[] value)[] inputBuffers32;
    (string name, NodeMetadata metadata, float[] value)[] outputBuffers32;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="shortID"></param>
    /// <param name="onnxFileName"></param>
    /// <param name="onnxModelBytes"></param>
    /// <param name="inputNames"></param>
    /// <param name="gpuID"></param>
    /// <param name="useTRT"></param>
    /// <param name="enableProfiling"></param>
    public NetExecutorONNXRuntime(string shortID,
                                  string onnxFileName, byte[] onnxModelBytes,
                                  string[] inputNames,
                                  NNEvaluatorPrecision precision, int gpuID,
                                  bool useTRT, bool enableProfiling,
                                  string[] outputNamesToUse = null)
    {
      //     if (gpuID < 0 || gpuID > 16) throw new Exception($"Invalid GPU ID { gpuID}");
      ONNXFileName = onnxFileName;
      if (onnxFileName == null && onnxModelBytes == null)
      {
        throw new Exception("Must specify either onnxFileName or onnxModelBytes");
      }

      string directoryName = onnxFileName == null ? Path.GetTempPath() : new FileInfo(onnxFileName).DirectoryName;

      if (!File.Exists(onnxFileName))
      {
        throw new Exception("ONNX file not found: " + onnxFileName);
      }
      if (onnxModelBytes == null)
      {
        onnxModelBytes = File.ReadAllBytes(onnxFileName);
      }


      GPUID = gpuID;
      Precision = precision;

      // On Linux it was found necessary to touch the instance before any of the operations below
      // to prevent error about a session object not being created.
      // https://github.com/microsoft/onnxruntime/issues/11572
      OrtEnv ortInstance = OrtEnv.Instance();

      SessionOptions so = default;

      //        so.AppendExecutionProvider_CoreML();


      if (gpuID < 0) // CPU. TO DO: clean this up
      {
        so = new SessionOptions();
      }
      else if (useTRT)
      {
        const bool USE_DML = false; // This likely requires different ONNXRuntime nuget package
        if (USE_DML)
        {
          so = new SessionOptions();
          so.AppendExecutionProvider_DML(gpuID);
          so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        }
        else
        {
          OrtTensorRTProviderOptions trtProviderOptions = new OrtTensorRTProviderOptions();
          // TODO: this code has no effect for unknown reasons.
          var providerOptionsDict = new Dictionary<string, string>();
          providerOptionsDict["device_id"] = gpuID.ToString();
          providerOptionsDict["trt_max_workspace_size"] = "4294967296";

          if (inputNames != null)
          {
            string MakeShapeStr(int size) // construct a shape string of the form expected by the TensorRT execution provider.
            {
              bool firstTime = true;
              string ret = "";
              foreach (string inputName in inputNames)
              {
                ret += (firstTime ? "" : ",") + inputName + $":{size}x64x{ONNXRuntimeExecutor.TPG_BYTES_PER_SQUARE_RECORD}";
                firstTime = false;
              }
              return ret;
            }

            // N.B. Using trtexec we seem to see extreme variability of runtime depending on these shapes (espcially optimal)
            // For now we omit specifying them, which seems to do as well.
            if (true)
            {
              providerOptionsDict["trt_profile_min_shapes"] = MakeShapeStr(1);
              providerOptionsDict["trt_profile_opt_shapes"] = MakeShapeStr(16);
              providerOptionsDict["trt_profile_max_shapes"] = MakeShapeStr(1024);
            }
          }

          // Use timing and engine caches, located in a folder specific to this host.
          providerOptionsDict["trt_timing_cache_enable"] = "true";
          //providerOptionsDict["trt_force_timing_cache"] = "true";
          providerOptionsDict["trt_engine_cache_enable"] = "true";
          string trtSubdirectory = Path.Combine(directoryName, "trt_engines", Environment.MachineName);
          Directory.CreateDirectory(trtSubdirectory);
          providerOptionsDict["trt_engine_cache_path"] = trtSubdirectory;
          providerOptionsDict["trt_timing_cache_path"] = trtSubdirectory;

          if (shortID != null)
          {
            providerOptionsDict["trt_engine_cache_prefix"] = shortID;
          }

//          providerOptionsDict["trt_detailed_build_log"] = "1";
          providerOptionsDict["trt_fp16_enable"] = Precision == NNEvaluatorPrecision.FP16 ? "true" : "false";
          providerOptionsDict["trt_builder_optimization_level"] = "3"; // N.B. Level 5 may be buggy

//          providerOptionsDict["trt_cuda_graph_enable"] = "true"; // NOTE: may fail or yield bad output, requires entire graph to map onto ONNX nodes (?)
//          providerOptionsDict["trt_auxiliary_streams"] = "0";

          providerOptionsDict["trt_layer_norm_fp32_fallback"] = "1"; // possibly necessary otherwise terrible accuracy

          trtProviderOptions.UpdateOptions(providerOptionsDict);
          so = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
        }
      }
      else
      {
        // https://tomwildenhain-microsoft.github.io/onnxruntime/docs/execution-providers/CUDA-ExecutionProvider.html
        OrtCUDAProviderOptions cudaProviderOptions = new();

        Dictionary<string, string> providerOptionsDict = new();
        providerOptionsDict["device_id"] = gpuID.ToString();


//        providerOptionsDict["enable_cuda_graph"] = "1"; // NOTE: may fail, requires entire graph to map onto ONNX nodes

        cudaProviderOptions.UpdateOptions(providerOptionsDict);
        so = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);

        //Session = new InferenceSession(onnxFileName, SessionOptions.MakeSessionOptionWithTensorrtProvider(gpuID));
      }

#if DEBUG
      so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
#else
      so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
#endif

      bool VERBOSE = false;
      if (VERBOSE)
      {
        so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        so.LogVerbosityLevel = 999;
        so.LogId = "ort.log.txt";
      }


      if (enableProfiling)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "****************   NetExecutorONNXRuntime is profiling....   ****************");
        so.EnableProfiling = true;
        so.ProfileOutputPathPrefix = @"d:\temp";
      }

      // See: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
      so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL; // Possibly this is overkill and takes too long?
      so.ExecutionMode = ExecutionMode.ORT_PARALLEL;

      lock (CUDADevice.InitializingCUDAContextLockObj)
      {
        using (new TimingBlock($"ONNX InferenceSession create on model of size {onnxModelBytes.Length:N0} bytes"))
        {
          Session = new InferenceSession(onnxModelBytes, so);

          inputsMetadata = Session.InputMetadata;

          // Create input and output buffers.
          if (Precision == NNEvaluatorPrecision.FP32)
          {
            inputBuffers32 = ONNXHelpers.CreateBuffers<float>(Session.InputMetadata, MAX_BATCH_SIZE);
            outputBuffers32 = ONNXHelpers.CreateBuffers<float>(Session.OutputMetadata, MAX_BATCH_SIZE, outputNamesToUse);
          }
          else if (Precision == NNEvaluatorPrecision.FP16)
          {
            inputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(Session.InputMetadata, MAX_BATCH_SIZE);
            outputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(Session.OutputMetadata, MAX_BATCH_SIZE, outputNamesToUse);
          }
          else
          {
            throw new Exception("Unsupported precision");
          } 

        }
      }
    }


    bool haveWarned = false;

    /// <summary>
    /// Returns the number of inputs as reported by the ONNX model metadata.
    /// </summary>
    public int NumInputs => inputsMetadata.Count;


    /// <summary>
    /// Evaluates the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public List<(string, Memory<Float16>)> Run((Memory<Half> input, int[] shape)[] inputs, int batchSize, bool float16)
    {
      // Determine input name
      // NOTE: experienced strange lowlevel crash when tried to break out this into name retrieval into a separate method
      string inputName = null;

      if (inputsMetadata.Count != inputs.Length)
      {
        throw new ArgumentException($"Expected {inputsMetadata.Count} inputs, received " + inputs.Length);
      }

      if (inputs[0].shape[0] > MAX_BATCH_SIZE)
      {
        throw new ArgumentException($"Batch size {inputs[0].shape[0]} exceeds maximum of {MAX_BATCH_SIZE}");
      }

      var inputsONNX = new (Memory<Half> input, int[] shape, string inputName, int numElements)[inputsMetadata.Count];

      if (inputsMetadata.Count != 1)
      {
        if (!haveWarned)
        {
          // data type check below is only on first element
          Console.WriteLine("WARNING: Currently only single input ONNX files supported definitively.");
          haveWarned = true;
        }
//        throw new Exception("Currently only single input ONNX files supported.");
      }

      int inputIndex = 0;
      bool inputIsFloat = true;
      foreach (KeyValuePair<string, NodeMetadata> iv in inputsMetadata)
      {
        (Memory<Half> input, int[] shape) = inputs[inputIndex];
        inputName = iv.Key;
        if (inputName == null)
        {
          throw new Exception("Unable to retrieve name of input");
        }

        int numElements = ONNXHelpers.ProductDimensions(shape, batchSize);
        Debug.Assert(input.Length == numElements); // caller to have passed the correct size

        inputIsFloat = iv.Value.ElementType == typeof(float);

        inputsONNX[inputIndex] = (input, shape, inputName, numElements);
        inputIndex++;
      }

      // TODO: Actually the precision of the network is defined by the net itself.
      //       So the variableIsFloat above should be used to determine this, and
      //       the caller should not be offered the chance to set the precision here
      //       (unless we decided to support auto-conversion of ONNX files here).
      if (float16 || !inputIsFloat)
      {
        return RunFloat16(inputsONNX, batchSize);
#if NOT
        // TODO: Make more efficient, avoid conversion to FP16 which happens in RunFloat16
        var ret = new List<(string, Memory<Float16>)>();
        int i = 0;
        foreach ((string, Memory<Float16> spanx) rr in RunFloat16(inputsONNX, batchSize))
        {
          // TODO: eventually avoid this conversion (in the FP16 case)
          //          Half[] floats = new Half[rr.Item2.Length];
          //          TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast < Float16, Half > (rr.Item2.Span), floats);
          //          Memory<Half> mx = MemoryMarshal.Cast<Float16, Half>(new Float16[22]);

          ret[i] = new(rr.Item1, rr.Item2);
          i++;
        }
        return ret;
#endif
      }
      else
      {
        return RunFloat(inputsONNX, batchSize);
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
    internal List<(string, Memory<Float16>)> RunFloat16((Memory<Half> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
    {
      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<Half> input, int[] shape, string inputName, int numElements) = inputs[i];

        // Convert float inputs directly into the target Float16 ONNX buffer
        Span<Half> inputBufferSpanHalf = MemoryMarshal.Cast<Float16, Half>(inputBuffers16[i].value);
        input.Span.CopyTo(inputBufferSpanHalf);
        //TensorPrimitives.ConvertToHalf(MemoryMarshal.Cast<Half,Float16>(input.Span), inputBufferSpanHalf);
      }

      lock (lockObject)
      {

        //          using (new TimingBlock(ToString() + " RunFloat16 " + inputs[0].numElements))
        {
          RunOptions runOptions = new RunOptions();

          do
          {
            //            using (new TimingBlock(ToString() + " RunFloat16 " + inputs[0].numElements))
            {
              var inputBuffers = ONNXHelpers.CreateOrtValues(batchSize, inputBuffers16);
              var outputBuffers = ONNXHelpers.CreateOrtValues(batchSize, outputBuffers16);

              // Note that IOBinding is not used. As noted in the ONNX documentation,
              // there is not necessarily any benefit of using IOBinding over this simpler
              // method of passing the OrtValue inputs and outputs directly to the Run method.
              Session.Run(runOptions,
                          inputBuffers.names, inputBuffers.values,
                          outputBuffers.names, outputBuffers.values);
            }
          } while (false);// (batchSize > 200); // TEST CODE

        }
      }

      List<(string, Memory<Float16>)> resultArrays = new(outputBuffers16.Length);
      foreach ((string name, NodeMetadata metadata, Float16[] value) resultItem in outputBuffers16)
      {
        // Create a Memory over the ONNX buffer sized to the actual number of elements for this batch.
        Memory<Float16> memory = new Memory<Float16>(resultItem.value)[..ONNXHelpers.ProductDimensions(resultItem.metadata.Dimensions, batchSize)];
        resultArrays.Add((resultItem.name, memory));
      }

      return resultArrays;
    }


    /// <summary>
    /// Runs the network with float inputs (instead of Half).
    /// 
    /// Note that this accepts Half inputs and returns Half inputs, 
    /// but merely upcasts them to floats for ONNX runtime execution and then downcasts results.
    /// 
    /// This is in efficient and does not fully exploit the higher precision of float over Half
    /// (but is intended mostly for debugging purposes).
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    private List<(string, Memory<Float16>)> RunFloat((Memory<Half> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
    {
      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<Half> input, int[] shape, string inputName, int numElements) = inputs[i];
        Memory<float> inputFloat = new float[numElements];
        TensorPrimitives.ConvertToSingle(input.Span, inputFloat.Span);
        DenseTensor<float> inputTensor = new DenseTensor<float>(inputFloat.Slice(0, numElements), shape);
        inputsONNX.Add(NamedOnnxValue.CreateFromTensor(inputName, inputTensor));
      }

      IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runResult;
      lock (lockObject)
      {
        RunOptions ro = new RunOptions();
        ro.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        ro.LogVerbosityLevel = 999;
        runResult = Session.Run(inputsONNX);//, ro); // fails on second run, reshape error, may be a bug on ONNXruntime
      }

      List<(string, Memory<Float16>)> resultArrays = new(Session.OutputMetadata.Count);
      foreach (DisposableNamedOnnxValue resultValue in runResult)
      {
        DenseTensor<float> tensor = (DenseTensor<float>)resultValue.AsTensor<float>();
        //default;// tensor.Buffer.ToArray(); // TO DO: Avoid reallocation ?

        Float16[] values = new Float16[tensor.Buffer.Length];
        for (int i=0; i<tensor.Buffer.Length; i++)
        {
          values[i] = (Float16)tensor.GetValue(i); // Inefficient!
        } 
        resultArrays.Add((resultValue.Name, values));
      }
      return resultArrays;
    }


    public void EndProfiling()
    {
      Session.EndProfiling();
    }


    /// <summary>
    /// Returns a string description of this object.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return "<NetExecutorONNXRuntime " + ONNXFileName + ">"; 
    }


    public void Dispose()
    {
#if FEATURE_ONNX
      if (!disposed)
      {
        Session.Dispose();
        disposed = true;
      }
#endif
    }


  }

  public static class  ONNXHelpers
  {

    /// <summary>
    /// Allocates array of buffers to be used as either inputs or outputs, based on specified metadata.
    /// 
    /// NOTE: One possibility would have been to allocate underlying memory via ONNX,
    ///       opening the door to an on-device buffer allocation.
    ///       However allocatorCUDA_PINNED was tested and works but does not seem any faster (possibly slightly slower)
    ///       using (var ortMemInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, DEVICE_ID, OrtMemType.Default))
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="metadata"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="outputNamesToUse"></param>
    /// <returns></returns>
    public static (string name, NodeMetadata metadata, T[] value)[]
      CreateBuffers<T>(IReadOnlyDictionary<string, NodeMetadata> metadata, int maxBatchSize, string[] outputNamesToUse = null) where T : unmanaged
    {
      var buffers = new (string name, NodeMetadata metadata, T[] value)[metadata.Count];

      int i = 0;
      foreach (KeyValuePair<string, NodeMetadata> iv in metadata)
      {
        if (outputNamesToUse == null || outputNamesToUse.Contains(iv.Key))
        {

          int maxElements = ProductDimensions(iv.Value.Dimensions, maxBatchSize);
          buffers[i] = (iv.Key, iv.Value, new T[maxElements]);

          i++;
        }
      }
      return buffers;
    }


    public static (string[] names, OrtValue[] values) CreateOrtValues<T>(int batchSize, (string name, NodeMetadata metadata, T[] value)[] buffers) where T : unmanaged
    {
      (string[] names, OrtValue[] values) ret = new();

      // TODO: eliminate next two allocations
      ret.names = new string[buffers.Length];
      ret.values = new OrtValue[buffers.Length];

      int i = 0;
      foreach ((string name, NodeMetadata metadata, T[] value) in buffers)
      {
        OrtValue ortValue = OrtValue.CreateTensorValueFromMemory<T>(value, ToLongArray(metadata.Dimensions, batchSize));
        ret.names[i] = name;
        ret.values[i] = ortValue;
        i++;
      }

      return ret;
    }

    public static long[] ToLongArray(int[] values, long firstValue)
    {
      long[] ret = new long[values.Length];

      Debug.Assert(values[0] == -1);
      ret[0] = firstValue;

      for (int i = 1; i < values.Length; i++)
      {
        ret[i] = values[i];
      }
      return ret;
    }



    public static int ProductDimensions(int[] dims, int negativeOneFillInValue = 1)
    {
      int productNumElements = 1;
      foreach (int dimSize in dims)
      {
        productNumElements *= dimSize == -1 ? negativeOneFillInValue : dimSize;  
      }

      return productNumElements;
    }



  }
}


