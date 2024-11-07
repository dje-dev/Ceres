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

using Onnx;
using System;
using System.IO;
using System.Linq;

using Google.Protobuf;

#endregion

namespace Ceres.Base.Misc.ONNX
{
  /// <summary>
  /// ONNX data types.
  /// See: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto.
  /// </summary>
  public enum ONNXDataType
  {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
    FLOAT4E2M1 = 23
  }


  /// <summary>
  /// Miscellaneous helper functions for working with ONNX models.
  /// </summary>
  public static class ONNXHelpers
  {

    /// <summary>
    /// Loads an ONNX model from a file.
    /// </summary>
    /// <param name="modelPath"></param>
    /// <returns></returns>
    public static ModelProto LoadModel(string modelPath)
    {
      using FileStream fileStream = File.OpenRead(modelPath);
      return ModelProto.Parser.ParseFrom(fileStream);
    }


    /// <summary>
    /// Saves an ONNX model to a file.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="outputPath"></param>
    public static void SaveModel(ModelProto model, string outputPath)
    {
      File.Delete(outputPath);
      using FileStream output = File.Create(outputPath);
      model.WriteTo(output);
    }


    /// <summary>
    /// Returns the number of parameters in the model.
    /// </summary>
    public static long NumParameters(ModelProto model)
    {
      int numNodes = 0;
      long numParams = 0;
      GraphProto graph = model.Graph;

      foreach (TensorProto init in graph.Initializer)
      {
        numNodes++;

        int typeSize = ((TensorProto.Types.DataType)init.DataType) switch
        {
          TensorProto.Types.DataType.Float => 4,
          TensorProto.Types.DataType.Float16 => 2,
          TensorProto.Types.DataType.Bfloat16 => 2,
          TensorProto.Types.DataType.Int32 => 4,
          TensorProto.Types.DataType.Int64 => 8,
          _ => throw new NotImplementedException()
        };

        numParams += init.CalculateSize() / typeSize;
      }

      return numParams;
    }


    /// <summary>
    /// Creates a TensorShapeProto from a list of dimensions.
    /// </summary>
    /// <param name="dims"></param>
    /// <returns></returns>
    public static TensorShapeProto MakeTensorShape(params long[] dims)
    {
      TensorShapeProto tsp = new();
      foreach (var d in dims)
      {
        if (d == -1)
        {
          tsp.Dim.Add(new TensorShapeProto.Types.Dimension() { DimParam = "batch_size" });
        }
        else
        {
          tsp.Dim.Add(new TensorShapeProto.Types.Dimension() { DimValue = d });
        }
      }
      return tsp;
    }


    /// <summary>
    /// Adds a metadata key-value pair to the model.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="key"></param>
    /// <param name="value"></param>
    public static void AddMetadataValue(ModelProto model, string key, string value)
    {
      StringStringEntryProto existingEntry = model.MetadataProps.FirstOrDefault(entry => entry.Key == key);

      if (existingEntry != null)
      {
        existingEntry.Value = value;
      }
      else
      {
        model.MetadataProps.Add(new StringStringEntryProto
        {
          Key = key,
          Value = value
        });
      }
    }


    /// <summary>
    /// Returns the value of a metadata key in the model (or null if not found).
    /// </summary>
    /// <param name="model"></param>
    /// <param name="key"></param>
    /// <returns></returns>
    public static string GetMetadataValue(ModelProto model, string key)
    {
      foreach (StringStringEntryProto entry in model.MetadataProps)
      {
        if (entry.Key == key)
        {
          return entry.Value;
        }
      }
      return null;
    }
  }
}
