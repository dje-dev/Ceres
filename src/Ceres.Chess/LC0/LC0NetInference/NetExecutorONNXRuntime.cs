#define FEATURE_ONNX
#if NOT
// requires packages: (actually, probably possible and best to install the Gpu package only)
// NOTE: the Gpu version may have a dependency on a specific version of CUDA \
//       and fail to load onnxruntime.dll otherwise
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.7.0" />
    <PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.7.1" />
    <PackageReference Include="Microsoft.ML.OnnxRuntime.Managed" Version="1.7.1" />
#endif
#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#define CUDA 

#region Using directives

using System;
using System.Collections.Generic;
using System.IO;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Chess.Ceres.NNEvaluators;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

#if FEATURE_ONNX
#endif

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
#if FEATURE_ONNX
    /// <summary>
    /// Underlying ONNX runtime session
    /// </summary>
    public readonly InferenceSession Session;
#endif

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
    /// <param name="gpuID"></param>
    /// <param name="batchSize"></param>
    public NetExecutorONNXRuntime(string onnxFileName, NNEvaluatorPrecision precision, int gpuID, bool useTRT)
    {
      // https://codechina.csdn.net/mirrors/microsoft/onnxruntime/-/blob/skottmckay/CreateHelperForGeneratingIdsForUseInCompilingEPs/docs/execution_providers/TensorRT-ExecutionProvider.md
      //     if (gpuID < 0 || gpuID > 16) throw new Exception($"Invalid GPU ID { gpuID}");
#if FEATURE_ONNX
      GPUID = gpuID;
      Precision = precision;

      // On Linux it was found necessary to touch the instance before any of the operations below
      // to prevent error about a session object not being created.
      // https://github.com/microsoft/onnxruntime/issues/11572
      OrtEnv ortInstance = OrtEnv.Instance();
      
      SessionOptions so = default;

      //        so.AppendExecutionProvider_DML();
      //        so.AppendExecutionProvider_CoreML();


      if (gpuID < 0) // CPU. TO DO: clean this up
      {
        so = new SessionOptions();
      }
      else if (useTRT)
      {
        OrtTensorRTProviderOptions trtProviderOptions = new OrtTensorRTProviderOptions();

        // TODO: this code has no effect for unknown reasons.
        var providerOptionsDict = new Dictionary<string, string>();
        providerOptionsDict["device_id"] = gpuID.ToString(); ;
        providerOptionsDict["trt_engine_cache_enable"] = "1";
        providerOptionsDict["trt_engine_cache_path"] = new FileInfo(onnxFileName).DirectoryName;
        providerOptionsDict["trt_fp16_enable"] = Precision == NNEvaluatorPrecision.FP16 ? "1" : "0";
        trtProviderOptions.UpdateOptions(providerOptionsDict);
        //trt_cache_path="/path/to/cache"
        //        providerOptionsDict["enable_cuda_graph"] = "1"; // NOTE: this requires entire graph to map onto ONNX nodes

#if NOT
('TensorrtExecutionProvider', {
'device_id': 0,
'trt_max_workspace_size': 2147483648,
'trt_fp16_enable': True,
'trt_dla_enable': False,
'trt_engine_cache_enable': False,
'trt_engine_cache_path':'./trtcache',
}),
#endif
        // TODO: Someday remove this. In theory, the above two assignments should work (but seems to be ignored).
        // WARNING: This makes a global change that could impact other threads. ****************
        Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1");
        Environment.SetEnvironmentVariable("ORT_TENSORRT_CACHE_PATH", new FileInfo(onnxFileName).DirectoryName);
        Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", Precision == NNEvaluatorPrecision.FP16 ? "1" : "0");


//        so = SessionOptions.MakeSessionOptionWithTensorrtProvider(gpuID);
        so = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
      }
      else
      {
        OrtCUDAProviderOptions cudaProviderOptions = new ();

        var providerOptionsDict = new Dictionary<string, string>();
        providerOptionsDict["device_id"] = gpuID.ToString();
        //        providerOptionsDict["gpu_mem_limit"] = "2147483648";
        //        providerOptionsDict["arena_extend_strategy"] = "kSameAsRequested";
        //        providerOptionsDict["cudnn_conv_algo_search"] = "DEFAULT";
        //        providerOptionsDict["do_copy_in_default_stream"] = "1";
        //        providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";
        //        providerOptionsDict["cudnn_conv1d_pad_to_nc1d"] = "1";
//        providerOptionsDict["enable_cuda_graph"] = "1"; // NOTE: this requires entire graph to map onto ONNX nodes
        
        cudaProviderOptions.UpdateOptions(providerOptionsDict);
        so = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);

        //Session = new InferenceSession(onnxFileName, SessionOptions.MakeSessionOptionWithTensorrtProvider(gpuID));
      }

      bool VERBOSE = false;
      if (VERBOSE)
      {
        so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        so.LogVerbosityLevel = 999;
        so.LogId = "ort.log.txt";
      }

      so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
    

      Session = new InferenceSession(onnxFileName, so);

#else
        throw new Exception("NetExecutorONNXRuntine feature is not enabled.");
#endif

    }


    /// <summary>
    /// Evaluates the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public float[][] Run((Memory<float> input, int[] shape)[] inputs, bool float16)
    {
#if FEATURE_ONNX
      // Determine input name
      // NOTE: experienced strange lowlevel crash when tried to break out this into name retrieval into a separate method
      string inputName = null;
      IReadOnlyDictionary<string, NodeMetadata> inputsMetadata = Session.InputMetadata;
      if (inputsMetadata.Count != inputs.Length)
      {
        throw new ArgumentException($"Expected {inputsMetadata.Count} inputs, received " + inputs.Length);
      }

      var inputsONNX = new (Memory<float> input, int[] shape, string inputName, int numElements)[inputsMetadata.Count];

      if (inputsMetadata.Count != 1)
      {
        // data type check below is only on first element
        throw new Exception("Currently only single input ONNX files supported."); 
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
          throw new Exception($"Unexpected number of elements {numElements} {input.Length} {shape.ToString()}");
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

#else
      return default;
#endif
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
    private float[][] RunFloat16((Memory<float> input, int[] shape, string inputName, int numElements)[] inputs)
    {
      List<NamedOnnxValue> inputsONNX = new(inputs.Length);

      for (int i = 0; i < inputs.Length; i++)
      {
        (Memory<float> input, int[] shape, string inputName, int numElements) = inputs[i];
        Span<float> inputSpan = input.Span;
        Float16[] inputFloat16 = new Float16[numElements];

        for (int ix = 0; ix < numElements; ix++)
        {
          inputFloat16[ix] = new FP16(inputSpan[ix]).Value;
        }

        DenseTensor<Float16> inputTensor = new DenseTensor<Float16>(inputFloat16, shape);
        inputsONNX.Add(NamedOnnxValue.CreateFromTensor(inputName, inputTensor));
      }

      IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runResult;
      lock (lockObject)
      {
        runResult = Session.Run(inputsONNX);
      }

      float[][] resultArrays = new float[Session.OutputMetadata.Count][];
      int iResult = 0;
      foreach (DisposableNamedOnnxValue resultValue in runResult)
      {
        DenseTensor<Float16> tensor = (DenseTensor<Float16>)resultValue.AsTensor<Float16>();
        Float16[] valuesFP16 = tensor.Buffer.ToArray();
        resultArrays[iResult++] = Array.ConvertAll<Float16, float>(valuesFP16, f => FP16.FromRaw((ushort)f));
      }
      return resultArrays;
    }


    private float[][] RunFloat((Memory<float> input, int[] shape, string inputName, int numElements)[] inputs)
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
        runResult = Session.Run(inputsONNX);
      }

      float[][] resultArrays = new float[Session.OutputMetadata.Count][];
      int ix = 0;
      foreach (DisposableNamedOnnxValue resultValue in runResult)
      {
        DenseTensor<float> tensor = (DenseTensor<float>)resultValue.AsTensor<float>();
        float[] values = tensor.Buffer.ToArray(); // TO DO: Avoid reallocation ?
        resultArrays[ix] = values;
        ix++;
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
