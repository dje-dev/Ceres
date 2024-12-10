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
using System.Linq;

using System.Collections.Generic;
using System.Runtime.InteropServices;

using Onnx;
using Google.Protobuf;
using Google.Protobuf.Collections;
using System.IO;

#endregion

namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  /// <summary>
  /// Generates new ONNX files which contain the averaged weights over multiple ONNX files.
  /// </summary>
  public static class ONNXFileAveraging
  {
    const bool DUMP_MAX_ABS_WEIGHTS_BY_LAYER = false; // currently only shows those >2


    /// <summary>
    /// Creates an equal-weighted average of the weights of the specified ONNX models.
    /// </summary>
    /// <param name="outputPath"></param>
    /// <param name="modelPaths"></param>
    /// <exception cref="ArgumentException"></exception>
    public static void CreateAveragedFile(string outputPath, params string[] modelPaths)
    {
      if (modelPaths == null || modelPaths.Length < 2)
      {
        throw new ArgumentException("At least two model paths must be provided.", nameof(modelPaths));
      }

      if (File.Exists(outputPath))
      {
        throw new ArgumentException("Output file already exists.", nameof(outputPath));
      } 

      // Load all models and prepare their initializers for averaging
      List<(float weight, RepeatedField<TensorProto> tensors)> modelsAndInitializers = new List<(float weight, RepeatedField<TensorProto> tensors)>();
      float weight = 1f / modelPaths.Length;

      foreach (string modelPath in modelPaths)
      {
        var model = Base.Misc.ONNX.ONNXHelpers.LoadModel(modelPath);
        modelsAndInitializers.Add((weight, model.Graph.Initializer));
      }

      // Compute the average of the initializers
      IEnumerable<TensorProto> averagedInitializers = AverageInitializers(modelsAndInitializers.ToArray());

      // Start with the first model as a base and update its initializers
      ModelProto baseModel = Base.Misc.ONNX.ONNXHelpers.LoadModel(modelPaths[0]);
      baseModel.Graph.Initializer.Clear();
      baseModel.Graph.Initializer.AddRange(averagedInitializers);

      // Save the new averaged model
      Base.Misc.ONNX.ONNXHelpers.SaveModel(baseModel, outputPath);

      Console.WriteLine($"New ONNX model {outputPath} created with averaged parameters from {modelPaths.Length} files.");
    }


    /// <summary>
    /// Internal method to average the initializers of multiple models.
    /// </summary>
    /// <param name="items"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    /// <exception cref="NotImplementedException"></exception>
    static IEnumerable<TensorProto> AverageInitializers(params (float weight, RepeatedField<TensorProto> tensors)[] items)
    {
      List<TensorProto> averagedInitializers = new List<TensorProto>();

      RepeatedField<TensorProto> rootTensors = items[0].tensors;

      foreach (TensorProto tensorRoot in rootTensors)
      {
        TensorProto avgTensor = default;
        byte[] newData = new byte[tensorRoot.RawData.Length];
        Span<Half> newHalfData = MemoryMarshal.Cast<byte, Half>(newData);

        foreach ((float weight, RepeatedField<TensorProto> tensors) item in items)
        {
          TensorProto tensorThisItem = item.tensors.FirstOrDefault(t => tensorRoot.Name == t.Name);
          if (tensorThisItem == null)
          {
            throw new Exception($"Tensor {tensorRoot.Name} not found in all models.");
          }

          if (avgTensor == null)
          {
            avgTensor = new()
            {
              Name = tensorRoot.Name,
              DataType = tensorRoot.DataType,
              Dims = { tensorRoot.Dims }
            };
          }

          if (tensorThisItem.DataType == (int)TensorProto.Types.DataType.Float16)
          {
            ReadOnlySpan<Half> value = MemoryMarshal.Cast<byte, Half>(tensorThisItem.RawData.Span);
            float maxAbs = 0;
            float minAbs = float.MaxValue;
            float sumAbs = 0;
            for (int i = 0; i < value.Length; i++)
            {
              newHalfData[i] += (Half)(item.weight * (float)value[i]);
              if (DUMP_MAX_ABS_WEIGHTS_BY_LAYER)
              {
                minAbs = value[i] == Half.Zero ? minAbs : Math.Min(minAbs, Math.Abs((float)value[i]));
                maxAbs =  Math.Max(maxAbs, Math.Abs((float)value[i]));
                sumAbs+= Math.Abs((float)value[i]);
              }                      
            }

            if (DUMP_MAX_ABS_WEIGHTS_BY_LAYER)
            {
              float avgAbs = sumAbs / value.Length;
              if (avgAbs < 1E-4 || minAbs > 2 || maxAbs > 2)
              {
                Console.WriteLine(avgAbs + " " + minAbs + " " + maxAbs + "  " + tensorRoot.Name);
              }
            }
          }
          else
          {
            throw new NotImplementedException("Unable to process: " + tensorThisItem.DataType + " " + tensorThisItem.Name);
          }

        }

        avgTensor.RawData = ByteString.CopyFrom(newData);
        averagedInitializers.Add(avgTensor);

//        Console.WriteLine(avgTensor.Name + " " + avgTensor.RawData.Length / 2);
      }

      return averagedInitializers;
    }


  }

#if NOT

	static (T[] sum, T min, T max) SumMinMax<T>(params (T weight, T[] data)[] raw1) where T : struct, INumber<T>
	{
		T min = default, max = default;
		T[] sum = new T[raw1[0].data.Length];

		for (int a = 0; a < raw1.Length; a++)
		{
			T acc = T.Zero;

			for (int i = 0; i < raw1[a].data.Length; i++)
			{
				T thisValue = raw1[a].data[i];

				acc += raw1[a].weight * thisValue;

				if (i == 0 || thisValue < min)
				{
					min = thisValue;
				}
				if (i == 0 || thisValue > max)
				{
					max = thisValue;
				}
			}

			sum[a] = acc;
		}
		return (sum, min, max);
	}
}

	static INumber<T>[] AverageTensorData(TensorProto tensor1, TensorProto tensor2) where T: INumber<T>
	{
		if (tensor1.DataType == (int)Onnx.TensorProto.Types.DataType.Float16)
		{
			(System.Half[] sum, System.Half min, System.Half max) sumMinMax = SumMinMax<System.Half>(
	((Half)0.5f, MemoryMarshal.Cast<byte, System.Half>(tensor1.RawData.Span).ToArray()),
	((Half)0.5f, MemoryMarshal.Cast<byte, System.Half>(tensor2.RawData.Span).ToArray())
);
return sumMinMax.sum as INumber<T>[];
		}
		else if (tensor1.DataType == (int)Onnx.TensorProto.Types.DataType.Float)
		{
			(float[] sum, float min, float max) sumMinMax = SumMinMax<float>(
  			(0.5f, MemoryMarshal.Cast<byte, float>(tensor1.RawData.Span).ToArray()),
	  		(0.5f, MemoryMarshal.Cast<byte, float>(tensor2.RawData.Span).ToArray())
			);
			return sumMinMax.sum as INumber<T>[];


//			Console.WriteLine($"{min,-8:F3}, {max,-8:F3} {tensor1.Name}");            return sum;
//			TensorPrimitives.Add(spanFloat1, spanFloat2, sumFloat);
		}
		else if (tensor1.DataType == (int)Onnx.TensorProto.Types.DataType.Double)
		{
			throw new NotImplementedException();
//			(double[] sum, double min, double max) = SumMinMax<double>(tensor1.RawData.Span, tensor2.RawData.Span);
//			return Array.ConvertAll(sum, item => (float)item);
		}
		else
		{
			throw new NotImplementedException();
			throw new NotSupportedException("Only float and double data types are supported in this example.");
		}
	}
}
#endif

#if NOT

static (T[] sum, T min, T max) SumMinMax<T>(ReadOnlySpan<byte> raw1, ReadOnlySpan<byte> raw2) where T: struct, INumber<T>
{
		ReadOnlySpan<T> spanFloat1 = MemoryMarshal.Cast<byte, T>(raw1);
		ReadOnlySpan<T> spanFloat2 = MemoryMarshal.Cast<byte, T>(raw2);
		
		T[] sumFloat = new T[spanFloat1.Length];
		sumFloat = Sum(spanFloat1, spanFloat2);	

		return (Sum(spanFloat1, spanFloat2), 
				T.Min(Min(spanFloat1), Min(spanFloat2)),
				T.Max(Max(spanFloat1), Max(spanFloat2)));
	}

#endif
}
