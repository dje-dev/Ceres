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

#endregion

namespace Ceres.Base.Misc.ONNX
{
  /// <summary>
  /// Miscellaneous helper functions for working with ONNX models.
  /// </summary>
  public static class ONNXHelpers
  {
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

  }
}
