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

using Google.Protobuf.Collections;

using Onnx;
using System.IO;
using Google.Protobuf;

#endregion

namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  public static class ONNXFileAveraging
  {
    public static void Main()
    {
      string modelPath1 = @"e:\cout\nets\HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_511.onnx";
      string modelPath2 = @"e:\cout\nets\HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072.onnx";
      string outputPath = @"e:\cout\nets\combox_511_final.onnx";

      // Load both ONNX models
      var model1 = LoadOnnxModel(modelPath1);
      var model2 = LoadOnnxModel(modelPath2);

      // Compute the average of the initializers
      var averagedInitializers = AverageInitializers((0.5f, model1.Graph.Initializer), (0.5f, model2.Graph.Initializer));

      // Replace the initializers in model1 with the averaged initializers
      model1.Graph.Initializer.Clear();
      model1.Graph.Initializer.AddRange(averagedInitializers);

      // Save the new model
      SaveOnnxModel(model1, outputPath);

      Console.WriteLine("New ONNX model 'combo.onnx' created with averaged parameters.");
    }


    static IEnumerable<TensorProto> AverageInitializers(params (float weight, RepeatedField<TensorProto> tensors)[] items)
    {
      var averagedInitializers = new List<TensorProto>();

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
            for (int i = 0; i < value.Length; i++)
            {
              newHalfData[i] += (Half)(item.weight * (float)value[i]);
            }

          }
          else
          {
            throw new NotImplementedException();
          }
        }

        avgTensor.RawData = ByteString.CopyFrom(newData);
        averagedInitializers.Add(avgTensor);

        Console.WriteLine(avgTensor.Name + " " + avgTensor.RawData.Length / 2);
      }

      return averagedInitializers;
    }


    static ModelProto LoadOnnxModel(string modelPath)
    {
      using var fileStream = File.OpenRead(modelPath);
      return ModelProto.Parser.ParseFrom(fileStream);
    }

    static void SaveOnnxModel(ModelProto model, string outputPath)
    {
      File.Delete(outputPath);
      using var output = File.Create(outputPath);
      model.WriteTo(output);
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
