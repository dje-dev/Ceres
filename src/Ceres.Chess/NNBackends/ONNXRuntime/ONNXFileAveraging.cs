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
using System.Threading.Tasks;

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
    /// Creates an equal-weighted average of the specified ONNX models, with optional min/max discarding.
    /// </summary>
    public static void CreateAveragedFile(string outputPath, bool discardMinAndMaxValues, params string[] modelPaths)
    {
      (string, float)[] modelPathsAndWeights = modelPaths.Select(p => (p, 1f / modelPaths.Length)).ToArray(); // equal weights
      CreateAveragedFile(outputPath, discardMinAndMaxValues, modelPathsAndWeights);
    }


    /// <summary>
    /// Creates an equal-weighted average of the specified ONNX models, with optional min/max discarding.
    /// </summary>
    public static void CreateAveragedFile(string outputPath,  params string[] modelPaths) =>  CreateAveragedFile(outputPath, false, modelPaths);
    


    /// <summary>
    /// Creates a weighted average of the specified ONNX models, with optional min/max discarding.
    /// </summary>
    /// <param name="outputPath">Path for the output ONNX file.</param>
    /// <param name="discardMinAndMaxValues">If true, discards the largest and smallest values before averaging.</param>
    /// <param name="modelPaths">Tuples of (string modelPath, float weight).</param>
    /// <exception cref="ArgumentException"></exception>
    public static void CreateAveragedFile(string outputPath, bool discardMinAndMaxValues, params (string, float)[] modelPaths)
    {
      if (modelPaths.Length < 3 && discardMinAndMaxValues)
      {
        throw new ArgumentException("At least three model paths must be provided with discardMinAndMaxValues.", nameof(modelPaths));
      }

      if (File.Exists(outputPath))
      {
        throw new ArgumentException("Output file already exists.", nameof(outputPath));
      }

      // Load all models and prepare their initializers for averaging
      List<(float weight, RepeatedField<TensorProto> tensors)> modelsAndInitializers
          = new List<(float weight, RepeatedField<TensorProto> tensors)>();

      foreach ((string path, float wgt) in modelPaths)
      {
        ModelProto model = Base.Misc.ONNX.ONNXHelpers.LoadModel(path);
        modelsAndInitializers.Add((wgt, model.Graph.Initializer));
      }

      // Compute the average of the initializers (assume two different methods exist, 
      // one with discard and one without)
      IEnumerable<TensorProto> averagedInitializers =  AverageInitializersMulti(discardMinAndMaxValues, modelsAndInitializers.ToArray());

      // Start with the first model as a base and update its initializers
      ModelProto baseModel = Base.Misc.ONNX.ONNXHelpers.LoadModel(modelPaths[0].Item1);
      baseModel.Graph.Initializer.Clear();
      baseModel.Graph.Initializer.AddRange(averagedInitializers);

      // Save the new averaged model
      Base.Misc.ONNX.ONNXHelpers.SaveModel(baseModel, outputPath);

      Console.WriteLine($"New ONNX model {outputPath} created with weighted averaged parameters from {modelPaths.Length} files.");
    }


    /// <summary>
    /// Computes a weighted average of the given 2D inputs (per "column" index),
    /// optionally removing the min and max values across the N inputs for each column.
    /// 
    /// If removeMinAndMax is true, then for each column j, we exclude from the
    /// weighted average any inputs[i][j] that match the minimum or maximum in that column.
    /// 
    /// Throws if removeMinAndMax is true and fewer than 3 total "rows" (i.e. inputs.Length < 3).
    /// </summary>
    /// <param name="inputs">2D array of Half, where each inputs[i] has the same length.</param>
    /// <param name="weights">Array of float weights, one weight per row in inputs.</param>
    /// <param name="removeMinAndMax">Whether to discard each column's min and max from the average.</param>
    /// <returns>A 1D array of Half, where each index corresponds to the column-averaged result.</returns>
    /// <exception cref="ArgumentException"></exception>
    public static void ComputedWtdAveragesWithDiscard(Span<Half> result, Half[][] inputs, float[] weights, bool removeMinAndMax)
    {
      if (inputs == null || weights == null)
      {
        throw new ArgumentException("Input arrays cannot be null.");
      }
      if (inputs.Length != weights.Length)
      {
        throw new ArgumentException("Number of input rows must match number of weights.");
      }
      if (inputs.Length == 0)
      {
        throw new ArgumentException("No input data provided.");
      }
      if (removeMinAndMax && inputs.Length < 3)
      {
        throw new ArgumentException("At least three rows are required to remove min and max.");
      }

      // Check that all rows have the same length
      int widthInputs = inputs[0].Length;
      for (int i = 1; i < inputs.Length; i++)
      {
        if (inputs[i].Length != widthInputs)
        {
          throw new ArgumentException("All input rows must have the same number of columns.");
        }
      }

      Half[] tempResult = new Half[widthInputs];

      // For each column j, gather N values from the N rows
      //      for (int j = 0; j < widthInputs; j++)
      Parallel.For(0, widthInputs, new ParallelOptions() { MaxDegreeOfParallelism = 16 }, j =>
            {
              float minVal = float.MaxValue;
              float maxVal = float.MinValue;

              if (removeMinAndMax)
              {
                // 1) Find min and max across the N inputs for column j
                for (int i = 0; i < inputs.Length; i++)
                {
                  float current = (float)inputs[i][j];
                  if (current < minVal)
                  {
                    minVal = current;
                  }
                  if (current > maxVal)
                  {
                    maxVal = current;
                  }
                }
              }

              // 2) Sum up (value * weight) and sum of weights, excluding min and max if requested
              float sumWeighted = 0f;
              float sumWeights = 0f;
              for (int i = 0; i < inputs.Length; i++)
              {
                float current = (float)inputs[i][j];
                if (removeMinAndMax && (current == minVal || current == maxVal))
                {
                  // Skip these
                  continue;
                }
                sumWeighted += current * weights[i];
                sumWeights += weights[i];
              }

              // 3) Compute the final average for this column
              float finalAverage = 0f;
              if (sumWeights > 1e-12f)
              {
                finalAverage = sumWeighted / sumWeights;
              }

              // 4) Store as Half in the result array
              tempResult[j] = (Half)finalAverage;
            });

            tempResult.CopyTo(result);
    }


    /// <summary>
    /// Internal method to average the initializers of multiple models.
    /// </summary>
    /// <param name="items"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    /// <exception cref="NotImplementedException"></exception>
    static IEnumerable<TensorProto> AverageInitializersMulti(bool discardMinMax, params (float weight, RepeatedField<TensorProto> tensors)[] items)
    {
      List<TensorProto> averagedInitializers = new List<TensorProto>();

      RepeatedField<TensorProto> rootTensors = items[0].tensors;
      float[] weights = items.Length == 1 ? new float[] { 1f } : items.Select(i => i.weight).ToArray();

      foreach (TensorProto tensorRoot in rootTensors)
      {
        TensorProto avgTensor = new()
        {
          Name = tensorRoot.Name,
          DataType = tensorRoot.DataType,
          Dims = { tensorRoot.Dims }
        };

        // Handle integer tensors separately - just copy from first model (no averaging for indices/config data)
        if (tensorRoot.DataType == (int)TensorProto.Types.DataType.Int32 ||
            tensorRoot.DataType == (int)TensorProto.Types.DataType.Int64)
        {
          avgTensor.RawData = tensorRoot.RawData;
          averagedInitializers.Add(avgTensor);
          continue;
        }

        if (tensorRoot.DataType != (int)TensorProto.Types.DataType.Float16)
        {
          throw new NotImplementedException("Unable to process: " + tensorRoot.DataType + " " + tensorRoot.Name);
        }

        Half[][] inputs = new Half[items.Length][];
        int index = 0;
        foreach ((float weight, RepeatedField<TensorProto> tensors) item in items)
        {
          TensorProto tensorThisItem = item.tensors.FirstOrDefault(t => tensorRoot.Name == t.Name);
          if (tensorThisItem == null)
          {
            throw new Exception($"Tensor {tensorRoot.Name} not found in all models.");
          }

          if (tensorThisItem.DataType != (int)TensorProto.Types.DataType.Float16)
          {
            throw new Exception($"Tensor {tensorRoot.Name} has inconsistent data types across models.");
          }

          inputs[index++] = MemoryMarshal.Cast<byte, Half>(tensorThisItem.RawData.Span).ToArray();
        }

        // Create the newData buffer
        byte[] newData = new byte[tensorRoot.RawData.Length];

        // Pin the array, so GC won't relocate it during our operations
        GCHandle handle = GCHandle.Alloc(newData, GCHandleType.Pinned);
        try
        {
          // Safely cast to Span<Half> after pinning
          Span<Half> newHalfData = MemoryMarshal.Cast<byte, Half>(newData.AsSpan());

          // Call your function on the newly pinned span
          ComputedWtdAveragesWithDiscard(newHalfData, inputs, weights, discardMinMax);
        }
        finally
        {
          // Free the handle once you're done
          handle.Free();
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
