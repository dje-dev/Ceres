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

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Ceres.Base.Misc;
using Ceres.Base.CUDA;
using Ceres.Base.Benchmarking;

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
namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  public class ONNXExecutor : IDisposable
  {
    /// <summary>
    /// Name of underlying ONNX file;
    /// </summary>
    public readonly string ONNXFileName;

    /// <summary>
    /// Underlying ONNX runtime session
    /// </summary>
    public readonly InferenceSession Session;

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
    /// Execution is serialized by this lock object.
    /// Technically, ONNX runtime sessions are thread-safe
    /// so this might not be strictly necessary.
    /// </summary>
    readonly object lockObject = new object();

    int maxBatchSize;

    RunOptions runOptions;
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
    /// <param name="nonBatchDimensions"></param>
    /// <param name="precisionNumBits"></param>
    /// <param name="gpuID"></param>
    /// <param name="useTRT"></param>
    /// <param name="minBatchSize"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="enableProfiling"></param>
    /// <param name="outputNamesToUse"></param>
    /// <exception cref="Exception"></exception>
    public ONNXExecutor(string shortID,
                        string onnxFileName, 
                        byte[] onnxModelBytes,
                        string[] inputNames,
                        string nonBatchDimensions,
                        int precisionNumBits, 
                        int gpuID,
                        bool useTRT, 
                        int minBatchSize,
                        int maxBatchSize,
                        bool enableProfiling,
                        string[] outputNamesToUse = null)
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

      if (!File.Exists(onnxFileName))
      {
        throw new Exception("ONNX file not found: " + onnxFileName);
      }

      this.maxBatchSize = maxBatchSize;
      string directoryName = onnxFileName == null ? Path.GetTempPath() : new FileInfo(onnxFileName).DirectoryName;

      if (onnxModelBytes == null)
      {
        onnxModelBytes = File.ReadAllBytes(onnxFileName);
      }


      runOptions = new RunOptions();

      GPUID = gpuID;
      PrecisionNumBits = precisionNumBits;
      MinBatchSize = minBatchSize;

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
          Dictionary<string, string> providerOptionsDict = new();
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
                if (!firstTime)
                {
                  // Testing with TensorRT 10.3 suggests caching fails if there is more than one input.
                  ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "WARNING as workaround for caching failure in ONNXRuntime,"
                    + " shapes string will omit " + inputName);
                  continue;
                }

                ret += (firstTime ? "" : ",") + inputName + $":{size}x{nonBatchDimensions}";
                firstTime = false;
              }
              return ret;
            }

            // If no profile is specified then TensorRT will build a new profile/engine
            // each time it sees a different batch size.
            // Although possibly faster to execute, these rebuilds would be far too slow.
            // Therefore we build a profile specifying the min, opt, and max batch sizes.
            //
            // Note that the actual performance is seemingly sensitive to the min/max batch sizes
            // in unpredictable ways. Seemingly large max batch sizes (e.g. over about 200)
            // often reduce performance.
            const bool USE_OMNI_PROFILE = true;
            if (USE_OMNI_PROFILE)
            {
              providerOptionsDict["trt_profile_min_shapes"] = MakeShapeStr(minBatchSize);
              providerOptionsDict["trt_profile_opt_shapes"] = MakeShapeStr(128);
              providerOptionsDict["trt_profile_max_shapes"] = MakeShapeStr(maxBatchSize); // N.B. see note above
            }
          }
#if NOT
Comments from onnxruntime source code:
 /*
   * Parse explicit min/max/opt profile shapes from provider options.
   *
   * The format of min/max/opt profile shapes is defined as below:
   * "input1:dim1xdim2...,input2:dim1xdim2...,...,input1:dim3xdim4...,input2:dim3xdim4...,..."
   *
   * (Note: if multiple shapes with same input name are specified, TRT EP will consider them as multiple profiles.
   *  Please refer to ParserProfileShapes() for more details)
   *
   */
#endif

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

          providerOptionsDict["trt_fp16_enable"] = PrecisionNumBits == 16 ? "true" : "false";
          providerOptionsDict["trt_builder_optimization_level"] = "3"; // N.B. Level 5 may be buggy

          // TODO: For graphs: use IO / Binding to bind input tensors in GPU memory.
          // See: https://github.com/microsoft/onnxruntime/issues/20050
          // "During inference, copy input to same address(input shape shall be the same) of the input used in the first inference run."
          //https://github.com/microsoft/onnxruntime/blob/4a196d15940b0f328735c888e2e861d67602ffcf/onnxruntime/python/tools/transformers/io_binding_helper.py#L212-L307
          // providerOptionsDict["trt_cuda_graph_enable"] = "true"; // NOTE: may fail or yield bad output, requires entire graph to map onto ONNX nodes (?)
          // providerOptionsDict["trt_auxiliary_streams"] = "0";

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


        cudaProviderOptions.UpdateOptions(providerOptionsDict);
        so = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
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
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, @"****************   NetExecutorONNXRuntime is profiling to c:\temp ....   ****************");
        so.EnableProfiling = true;
        so.ProfileOutputPathPrefix = @"c:\temp";
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
          if (PrecisionNumBits == 32)
          {
            inputBuffers32 = ONNXHelpers.CreateBuffers<float>(Session.InputMetadata, maxBatchSize);
            outputBuffers32 = ONNXHelpers.CreateBuffers<float>(Session.OutputMetadata, maxBatchSize, outputNamesToUse);
          }
          else if (PrecisionNumBits == 16)
          {
            inputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(Session.InputMetadata, maxBatchSize);
            outputBuffers16 = ONNXHelpers.CreateBuffers<Float16>(Session.OutputMetadata, maxBatchSize, outputNamesToUse);
          }
          else
          {
            throw new Exception("Unsupported precision (" + PrecisionNumBits + ")");
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

      if (inputs[0].shape[0] > maxBatchSize)
      {
        throw new ArgumentException($"Batch size {inputs[0].shape[0]} exceeds maximum of {maxBatchSize}");
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
      if (float16)
      {
        return RunFloat16(inputsONNX, batchSize);
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
      if (batchSize < MinBatchSize)
      {
        throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
      }

      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<Half> input, int[] shape, string inputName, int numElements) = inputs[i];

        // Cast float inputs directly into the target Float16 ONNX buffer
        Span<Half> inputBufferSpanHalf = MemoryMarshal.Cast<Float16, Half>(inputBuffers16[i].value);
        input.Span.CopyTo(inputBufferSpanHalf);
      }

      lock (lockObject)
      {
        (string[] names, OrtValue[] values) inputBuffers = ONNXHelpers.CreateOrtValues(batchSize, inputBuffers16);
        (string[] names, OrtValue[] values) outputBuffers = ONNXHelpers.CreateOrtValues(batchSize, outputBuffers16);

        // Note that IOBinding is not used. As noted in the ONNX documentation,
        // there is not necessarily any benefit of using IOBinding over this simpler
        // method of passing the OrtValue inputs and outputs directly to the Run method.
        Session.Run(runOptions,
                    inputBuffers.names, inputBuffers.values,
                    outputBuffers.names, outputBuffers.values);
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
    /// This is inefficient and does not fully exploit the higher precision of float over Half
    /// (but is intended mostly for debugging purposes).
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    private List<(string, Memory<Float16>)> RunFloat((Memory<Half> input, int[] shape, string inputName, int numElements)[] inputs, int batchSize)
    {
      if (batchSize < MinBatchSize)
      {
        throw new ArgumentException($"Batch size {batchSize} is less than minimum of {MinBatchSize}");
      }

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
        // TODO: make this code more simlar to RunFloat16, including use of inputBuffers32
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


    /// <summary>
    /// Ends profiling.
    /// </summary>
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
      return "<NetExecutorONNXRuntime " + ONNXFileName + " (" + PrecisionNumBits + ")>"; 
    }


    /// <summary>
    /// Disposes of this object.
    /// </summary>
    public void Dispose()
    {
      if (!disposed)
      {
        runOptions.Dispose();
        Session.Dispose();
        disposed = true;
      }
    }

  }
}