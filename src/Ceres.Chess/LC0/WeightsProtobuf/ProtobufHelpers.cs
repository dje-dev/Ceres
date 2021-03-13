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

using Ceres.Base.DataType;
using Pblczero;
using ProtoBuf;
using System;
using System.IO;

#endregion

namespace Ceres.Chess.LC0.WeightsProtobuf
{
  /// <summary>
  /// Set of static helper methods for working with 
  /// LC0 weights protobuf files.
  /// </summary>
  public static class ProtobufHelpers
  {
    /// <summary>
    /// Retrieves the weights values from a specified layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <returns></returns>
    public static float[] GetLayerLinear16(Weights.Layer layer)
    {
      float[] ret = new float[layer.Params.Length / 2];

      for (int i = 0; i < ret.Length; i++)
        ret[i] = GetLayerLinear16Single(layer, i);
      return ret;
    }


    /// <summary>
    /// Sets the weights values in a specified layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="index"></param>
    /// <param name="value"></param>
    public static void SetLayerLinear16(Weights.Layer layer, int index, float value)
    {
      if (value < layer.MinVal) value = layer.MinVal;
      if (value > layer.MaxVal) value = layer.MaxVal;

      float width = (layer.MaxVal - layer.MinVal) / 65535.0f;
      float offset = value - layer.MinVal;
      float increment = MathF.Round(offset / width, 0);
      byte b0 = (byte)(increment % 256);
      byte b1 = (byte)(increment / 256);

      layer.Params[index * 2] = b0;
      layer.Params[index * 2 + 1] = b1;
    }


    /// <summary>
    /// Gets a single layer vector within a speciifed layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="index"></param>
    /// <returns></returns>
    public static float GetLayerLinear16Single(Weights.Layer layer, int index)
    {
      byte[] b = new byte[2];
      b[0] = layer.Params[index * 2];
      b[1] = layer.Params[index * 2 + 1];
      float v1 = 256 * b[1] + b[0];
      float v1a = layer.MinVal + v1 * (layer.MaxVal - layer.MinVal) / 65535.0f;
      return v1a;
    }


    /// <summary>
    /// Sets a layer vector within a specified layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="values"></param>
    public static void SetLayerLinear16(Weights.Layer layer, float[] values)
    {
      if (layer.Params.Length != values.Length *  2) throw new System.Exception("not expected size");

      for (int i = 0; i < values.Length; i++)
        SetLayerLinear16(layer, i, values[i]);
    }


    /// <summary>
    /// Rewrites a protobuf file to another file, with specified modfications.
    /// </summary>
    /// <param name="fnOrg"></param>
    /// <param name="fnRewritten"></param>
    /// <param name="layerMap"></param>
    /// <param name="valueCalc"></param>
    public static void RewriteLayerInNet(string fnOrg, string fnRewritten,
                                         Func<LC0ProtobufNet, Weights.Layer> layerMap,
                                         Func<int, float, float> valueCalc)
    {
      // Read from disk
      LC0ProtobufNet pbn = new LC0ProtobufNet(fnOrg);
      Weights.Layer layer = layerMap(pbn);

      // Get current layer values and get them rewritten
      float[] values = ProtobufHelpers.GetLayerLinear16(layer);
      for (int i = 0; i < values.Length; i++)
        values[i] = valueCalc(i, values[i]);

      // Set value of this layer and serialize whole net to bytes
      ProtobufHelpers.SetLayerLinear16(layer, values);
      byte[] bytes = SerializationUtils.ProtoSerialize<Net>(pbn.Net);

      // Write to disk
      if (File.Exists(fnRewritten)) File.Delete(fnRewritten);
      File.WriteAllBytes(fnRewritten, bytes);
    }

  }
}
