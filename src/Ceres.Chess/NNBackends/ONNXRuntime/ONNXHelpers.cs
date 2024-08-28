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
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

using Microsoft.ML.OnnxRuntime;

#endregion

namespace Ceres.Chess.NNBackends.ONNXRuntime
{
  /// <summary>
  /// Miscellaneous static helper methods for ONNX operations.
  /// </summary>
  public static class ONNXHelpers
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
    public static (string name, NodeMetadata metadata, bool isKnownShape, T[] value)[]
      CreateBuffers<T>(IReadOnlyDictionary<string, NodeMetadata> metadata, int maxBatchSize, string[] outputNamesToUse = null) where T : unmanaged
    {
      var buffers = new (string name, NodeMetadata metadata, bool isKnownShape, T[] value)[metadata.Count];

      int i = 0;
      foreach (KeyValuePair<string, NodeMetadata> iv in metadata)
      {
        if (outputNamesToUse == null || outputNamesToUse.Contains(iv.Key))
        {

          int maxElements = ProductDimensions(iv.Value.Dimensions, maxBatchSize);
          bool isKnownShape = iv.Value.Dimensions.Length > 0 && Array.LastIndexOf(iv.Value.Dimensions, -1) <= 0; // if any dimension other than batch size unknown
          buffers[i] = (iv.Key, iv.Value, isKnownShape, new T[maxElements]);

          i++;
        }
      }
      return buffers;
    }


    public static (string[] names, OrtValue[] values) CreateOrtValues<T>(int batchSize, (string name, NodeMetadata metadata, bool isKnownShape, T[] value)[] buffers) where T : unmanaged
    {
      (string[] names, OrtValue[] values) ret = new();

      // TODO: eliminate next two allocations
      ret.names = new string[buffers.Length];
      ret.values = new OrtValue[buffers.Length];

      int i = 0;
      foreach ((string name, NodeMetadata metadata, bool isKnownShape, T[] value) in buffers)
      {
        // If not known shape, the value is a dummy since we wouldn't know the appropriate size.
        long[] shape = isKnownShape ? ToLongArray(metadata.Dimensions, batchSize) : [0];
        OrtValue ortValue = OrtValue.CreateTensorValueFromMemory(value, shape);
        ret.names[i] = name;
        ret.values[i] = ortValue;
        i++;
      }

      return ret;
    }

    public static long[] ToLongArray(int[] values, long firstValue)
    {
      long[] ret = new long[values.Length];

      if (values[0] == -1)
      {
        ret[0] = firstValue;
      }

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
