//#define FEATURE_ONNX
#if NOT
// requires packages:
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
using System.Diagnostics;
using System.Collections.Generic;
using Ceres.Base;


#if FEATURE_ONNX
// needs     <PackageReference Include="Microsoft.ML.OnnxRuntime.Managed" Version="1.6.0" />

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
    /// Constructor.
    /// </summary>
    /// <param name="onnxFileName"></param>
    /// <param name="gpuID"></param>
    /// <param name="batchSize"></param>
    public NetExecutorONNXRuntime(string onnxFileName, int gpuID)
    {
      //     if (gpuID < 0 || gpuID > 16) throw new Exception($"Invalid GPU ID { gpuID}");
#if FEATURE_ONNX
#if CUDA
      if (gpuID == -999) // CPU. TO DO: clean this up
        Session = new InferenceSession(onnxFileName);
      else if (gpuID == -1)
        Session = new InferenceSession(onnxFileName, SessionOptions.MakeSessionOptionWithCudaProvider());
      else
      {
#if NOT
//Yields error: Unable to find an entry point named 'OrtSessionOptionsAppendExecutionProvider_Tensorrt' in DLL 'onnxruntime'.

        SessionOptions options = new SessionOptions();
        options.AppendExecutionProvider_Tensorrt(gpuID);
//        options.AppendExecutionProvider_CUDA();
        Session = new InferenceSession(onnxFileName, options);
#endif
        Session = new InferenceSession(onnxFileName, SessionOptions.MakeSessionOptionWithCudaProvider(/*gpuID*/));
      }
#else
        Session = new InferenceSession(onnxFileName);
#endif
#else
      throw new Exception("NetExecutorONNXRuntine feature is not enabled.");
#endif
      GPUID = gpuID;
    }


    /// <summary>
    /// Evaluates the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public float[][] Run(Memory<float> input, int[] shape)
    {
#if FEATURE_ONNX
      // Determine input name
      // NOTE: experienced strange lowlevel crash when tried to break out this into name retrieval into a separate method
      string inputName = null;
      IReadOnlyDictionary<string, NodeMetadata> inputsMetadata = Session.InputMetadata;
      if (inputsMetadata.Count > 1) throw new ArgumentException("Expected only one input, found " + inputsMetadata.Count);
      foreach (KeyValuePair<string, NodeMetadata> iv in inputsMetadata)
      {
        inputName = iv.Key;
        break; // get only first
      }
      if (inputName == null) throw new Exception("Unable to retrieve name of input");

      int numElements = 1;
      foreach (int dimSize in shape) numElements *= dimSize;
      if (input.Length < numElements) throw new Exception($"Unexpected number of elements {numElements} {input.Length} {shape.ToString()}");

      //System.Memory Memory<float> inputAsMemory = (Memory<float>)input;
      DenseTensor<float> inputTensor = new DenseTensor<float>(input.Slice(0, numElements), shape);
      List<NamedOnnxValue> inputs = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

      IDisposableReadOnlyCollection<DisposableNamedOnnxValue> runResult;

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
#else
      return default;
#endif
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
