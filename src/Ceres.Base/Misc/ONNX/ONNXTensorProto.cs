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

using Onnx;

using Ceres.Base.DataType;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Google.Protobuf;

#endregion

namespace Ceres.Base.Misc.ONNX
{
  /// <summary>
  /// Wrapper for Onnx.TensorProto objects with convenience methods.
  /// </summary>
  public class ONNXTensorProto
  {
    public readonly TensorProto Proto;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="proto"></param>
    public ONNXTensorProto(TensorProto proto) => Proto = proto;

    /// <summary>
    /// The name of the tensor.
    /// </summary>
    public string Name => Proto.Name;

    /// <summary>
    /// Data type of the tensor.
    /// </summary>
    public TensorProto.Types.DataType DataType => Proto.DataType();

    /// <summary>
    /// Documentation string for the tensor.
    /// </summary>
    public string DocString => Proto.DocString;

    /// <summary>
    /// Tensor contents as a float array.
    /// </summary>
    public float[] DataFloat => GetTensorData<float>(Proto).ToArray();

    /// <summary>
    /// Tensor contents as a 2D float array.
    /// </summary>
    public float[,] DataFloat2D => GetTensorData2D<float>();

    /// <summary>
    /// Tensor contents as a double array.
    /// </summary>
    public double[] DataDouble => GetTensorData<double>(Proto).ToArray();

    /// <summary>
    /// Tensor contents as a 2D double array.
    /// </summary>
    public int[] DataInt => GetTensorData<int>(Proto).ToArray();

    /// <summary>
    /// Tensor contents as a 2D int array.
    /// </summary>
    public int[] DataLong => GetTensorData<int>(Proto).ToArray();


    /// <summary>
    /// Data contents of the tensor as a 1D array.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    unsafe T[,] GetTensorData2D<T>() where T : struct
    {
      if (Dims.Length != 2)
      {
        throw new Exception($"Requested tensor {Name} has dimension {Dims.Length} not 2D");
      }

      T[] raw = GetTensorData<T>(Proto).ToArray();
      return ArrayUtils.To2D<T>(raw, (int)Proto.Dims[1]);
    }


    /// <summary>
    /// Returns the size of the tensor in bytes.
    /// </summary>
    public int RawSizeBytes => Proto.CalculateSize();
    
   
    /// <summary>
    /// Returns the dimensions of the tensor.
    /// </summary>
    public long[] Dims
    {
      get
      {
        var dims = Proto.Dims;
        long[] ret = new long[dims.Count];
        for (int i = 0; i < ret.Length; i++)
        {
          ret[i] = dims[i];
        }
        return ret;
      }
    }

    #region Helper methods
    static unsafe ReadOnlySpan<T> GetTensorData<T>(TensorProto tensor) where T : struct
      => tensor.DataType() switch
      {
        // NOTE: Long lines accepted below for structure
        TensorProto.Types.DataType.Float => GetTensorData<T, float>(tensor.FloatData as IReadOnlyList<T>, tensor.RawData),
        TensorProto.Types.DataType.Double => GetTensorData<T, double>(tensor.DoubleData as IReadOnlyList<T>, tensor.RawData),
        TensorProto.Types.DataType.Int32 => GetTensorData<T, int>(tensor.Int32Data as IReadOnlyList<T>, tensor.RawData),
        TensorProto.Types.DataType.Int64 => GetTensorData<T, long>(tensor.Int64Data as IReadOnlyList<T>, tensor.RawData),
        // TODO: StringData
        _ => throw new NotImplementedException()
      };


    // Based on code from OnnxSharp (method FormatValuesOrStats).
    static unsafe ReadOnlySpan<T> GetTensorData<T, T1>(IReadOnlyList<T> values, ByteString rawData) where T : struct
    {
      if (typeof(T) != typeof(T1))
      {
        throw new Exception("type mismatch");
      }

      // Data may be either in strongly typed field (e.g. FloatData) or in RawBytes, determine which.
      bool useRawData = values.Count == 0 && rawData.Length > 0;
      ReadOnlySpan<T> rawValues = MemoryMarshal.Cast<byte, T>(rawData.Span);

      return useRawData ? rawValues : values.ToArray();
    }

    #endregion
  }

}
