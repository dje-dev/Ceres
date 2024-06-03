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

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Ceres.Base.CUDA;
using Ceres.Base.Benchmarking;
using Ceres.Base.Misc;

using Chess.Ceres.NNEvaluators;

#endregion

/// <summary>
/// Manages evaluation of neural networks using ONNX runtime.
/// 
///
/// Docs say: " Key design decisions Multiple threads can invoke the Run() method on the same inference session object. 
///           See[API doc] (C_API.md) for more details.
///
/// NOTE: Float16 not supported, would need to have support in this file: csharp/src/Microsoft.ML.OnnxRuntime/NamedOnnxValue.cs 
///
/// </summary>
namespace Ceres.Chess.LC0NetInference
{
  public class NetExecutorONNXRuntime : IDisposable
  {
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

    readonly object lockObject = new object();
    bool disposed;

    /// <summary>
    /// Session data type precision to use.
    /// </summary>
    public readonly NNEvaluatorPrecision Precision;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="onnxFileName"></param>
    /// <param name="onnxModelBytes"></param>
    /// <param name="inputNames"></param>
    /// <param name="gpuID"></param>
    /// <param name="useTRT"></param>
    /// <param name="enableProfiling"></param>
    public NetExecutorONNXRuntime(string onnxFileName, byte[] onnxModelBytes,
                                  string[] inputNames,
                                  NNEvaluatorPrecision precision, int gpuID,
                                  bool useTRT, bool enableProfiling)
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
          providerOptionsDict["device_id"] = gpuID.ToString(); ;
          providerOptionsDict["trt_max_workspace_size"] = "2147483648";

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
            // N.B. For now using optimal batch size of 108, seemed fastest for a certain tested net.
            providerOptionsDict["trt_profile_min_shapes"] = MakeShapeStr(1);
            providerOptionsDict["trt_profile_opt_shapes"] = MakeShapeStr(108);
            providerOptionsDict["trt_profile_max_shapes"] = MakeShapeStr(1024);
          }
          //          providerOptionsDict["trt_profile_min_shapes"] = "squares:1x64x137" + (hasStateInput ? ",prior_state.1:1x64x4" : "");

          providerOptionsDict["trt_timing_cache_enable"] = "true";
          providerOptionsDict["trt_force_timing_cache"] = "true";

          providerOptionsDict["trt_engine_cache_enable"] = "true";

          providerOptionsDict["trt_engine_cache_path"] = Path.Combine(directoryName, "trt_engines");
          providerOptionsDict["trt_timing_cache_path"] = directoryName;
          // providerOptionsDict["trt_engine_cache_prefix"] = "Ceres";

          //trt_detailed_build_log=1
          providerOptionsDict["trt_fp16_enable"] = Precision == NNEvaluatorPrecision.FP16 ? "1" : "0";
          providerOptionsDict["trt_builder_optimization_level"] = "3";

          // providerOptionsDict["trt_cuda_graph_enable"] = "1"; // NOTE: may fail, requires entire graph to map onto ONNX nodes (?)

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
        //so.ProfileOutputPathPrefix = @"d:\temp";
      }

      // See: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
      so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL; // Possibly this is overkill and takes too long?
      so.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        
      lock (CUDADevice.InitializingCUDAContextLockObj)
      {
        using (new TimingBlock($"ONNX InferenceSession create on model of size {onnxModelBytes.Length:N0} bytes"))
        {
          Session = new InferenceSession(onnxModelBytes, so);
        }
      }
    }


    bool haveWarned = false;


    /// <summary>
    /// Evaluates the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public List<(string, float[])> Run((Memory<float> input, int[] shape)[] inputs, bool float16)
    {
      // Determine input name
      // NOTE: experienced strange lowlevel crash when tried to break out this into name retrieval into a separate method
      string inputName = null;
      IReadOnlyDictionary<string, NodeMetadata> inputsMetadata = Session.InputMetadata;
      if (inputsMetadata.Count != inputs.Length)
      {
        throw new ArgumentException($"Expected {inputsMetadata.Count} inputs, received " + inputs.Length);
      }

#if NOT
// see: great code here https://gist.github.com/pranavsharma/f3c3ced552cada00fb556734c6967711
//      var inputMeta = session.InputMetadata;
//      float[] inputData = LoadTensorFromFile("C:\\Users\\prs\\source\\repos\\GH8332\\bench.in");
      var input_tensor = new DenseTensor<float>(inputData, inputMeta["data_0"].Dimensions);
      var output_mem_info = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

      var io_binding = session.CreateIoBinding();
      io_binding.BindInput("data_0", FixedBufferOnnxValue.CreateFromTensor(input_tensor));
      io_binding.BindOutputToDevice("softmaxout_1", output_mem_info);

      session.RunWithBinding(run_options, io_binding);
#endif

      var inputsONNX = new (Memory<float> input, int[] shape, string inputName, int numElements)[inputsMetadata.Count];

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
        (Memory<float> input, int[] shape) = inputs[inputIndex];
        inputName = iv.Key;
        if (inputName == null)
        {
          throw new Exception("Unable to retrieve name of input");
        }

        int numElements = 1;
        foreach (int dimSize in shape)
        {
          numElements *= dimSize;
        }

        if (input.Length != numElements)
        {
          // Resize the input (the caller may have passed an oversized buffer for efficiency).
          input = input.Slice(0, numElements);
        }

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
        // TODO: Make more efficient, avoid conversion to FP16 which happens in RunFloat16
        return RunFloat16(inputsONNX);
      }
      else
      {
        return RunFloat(inputsONNX);
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
    internal List<(string, float[])> RunFloat16((Memory<float> input, int[] shape, string inputName, int numElements)[] inputs)
    {
      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<float> input, int[] shape, string inputName, int numElements) = inputs[i];
        Span<float> inputSpan = input.Span;
        Float16[] inputFloat16 = new Float16[numElements];

        TensorPrimitives.ConvertToHalf(inputSpan, MemoryMarshal.Cast<Float16, Half>(inputFloat16));

        DenseTensor<Float16> inputTensor = new DenseTensor<Float16>(inputFloat16, shape);
        inputsONNX.Add(NamedOnnxValue.CreateFromTensor(inputName, inputTensor));
      }

      IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runResult;
      lock (lockObject)
      {
        if (inputs[0].numElements > 28768)
        {
          while (false) // TEST CODE
          {
            using (new TimingBlock(ToString() + " RunFloat16 " + inputs[0].numElements))
              runResult = Session.Run(inputsONNX);
          }
        } 
        runResult = Session.Run(inputsONNX);
      }

      List<(string, float[])> resultArrays = new(Session.OutputMetadata.Count);
      foreach (DisposableNamedOnnxValue resultValue in runResult)
      {
        DenseTensor<Float16> tensor = (DenseTensor<Float16>)resultValue.AsTensor<Float16>();
        Span<Float16> valuesFloat16 = tensor.Buffer.Span;

        float[] resultArray = new float[valuesFloat16.Length];
        TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast<Float16, Half>(valuesFloat16), resultArray);

        resultArrays.Add((resultValue.Name, resultArray));
      }

      return resultArrays;
    }


    private List<(string, float[])> RunFloat((Memory<float> input, int[] shape, string inputName, int numElements)[] inputs)
    {
      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<float> input, int[] shape, string inputName, int numElements) = inputs[i];
        DenseTensor<float> inputTensor = new DenseTensor<float>(input.Slice(0, numElements), shape);
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

      List<(string, float[])> resultArrays = new(Session.OutputMetadata.Count);
      foreach (DisposableNamedOnnxValue resultValue in runResult)
      {
        DenseTensor<float> tensor = (DenseTensor<float>)resultValue.AsTensor<float>();
        float[] values = tensor.Buffer.ToArray(); // TO DO: Avoid reallocation ?
        resultArrays.Add((resultValue.Name, values));
      }
      return resultArrays;
    }
#if NOT
    DenseTensor<float> inputTensor = new DenseTensor<float>(input.Slice(0, numElements), shape);
      List<NamedOnnxValue> inputs = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

      IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runResult = default;

        lock (lockObject)
        {
          runResult = Session.Run(inputs);
        }

      float[][] resultArrays = new float[Session.OutputMetadata.Count][];
      int i = 0;
      foreach (DisposableNamedOnnxValue resultValue in runResult)
      {
        DenseTensor<float> tensor = (DenseTensor<float>)resultValue.AsTensor<float>();
        float[] values = tensor.Buffer.ToArray(); // TO DO: Avoid reallocation ?
        resultArrays[i] = values;
        i++;
      }
      return resultArrays;
    }
#endif

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
}


#if NOT

// Python example. This successfully runs a model, but very slowly.
//  (do this first from command line: set CUDA_VISIBLE_DEVICES=3)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy.random
import onnxruntime
import time

import onnx

session = onnxruntime.InferenceSession(r"d:\converted\run1_62242.onnx", None)
x = numpy.random.random((8192,112, 64))
x = x.astype(numpy.float32)
res = session.run(['value_head:0','policy_head:0'], {'Placeholder:0': x})

for i in range(3):
  t1 = time.time()
  res = session.run(['value_head:0','policy_head:0'], {'Placeholder:0': x})
  t2 = time.time()
  print('time', t2 - t1)

#endif
