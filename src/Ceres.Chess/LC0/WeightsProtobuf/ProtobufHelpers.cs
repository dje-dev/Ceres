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
using Ceres.Base.DataTypes;
using Pblczero;
using ProtoBuf;
using System;
using System.IO;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Chess.LC0.WeightsProtobuf
{
  /// <summary>
  /// Set of static helper methods for working with 
  /// LC0 weights protobuf files.
  /// </summary>
  public static class ProtobufHelpers
  {
    static bool haveWarned = false;

    /// <summary>
    /// Retrieves the weights values from a specified layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="scale">the linear scaling factor use to encode the layer</param>
    /// <returns></returns>
    public static float[] GetLayerLinear16(Weights.Layer layer, out float scaleUsed)
    {
      float[] ret = new float[layer.Params.Length / 2];

      float minVal = layer.MinVal;
      float maxVal = layer.MaxVal;
      scaleUsed = (maxVal - minVal) / 65535.0f;
      
      if (scaleUsed > 1 && !haveWarned)
      {
        Console.Write($"** Warning: network weights encountered outside expected range, layer min = {minVal} max = {maxVal}. ");
        Console.WriteLine("Further warnings will be suppressed.");
        haveWarned = true;
      }

      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = GetLayerLinear16Single(layer, i, minVal, scaleUsed);
      }

      return ret;
    }


    /// <summary>
    /// Retrieves the weights values from a specified layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <returns></returns>
    public static FP16[] GetLayerLinear16FP16(Weights.Layer layer)
    {
      FP16[] ret = new FP16[layer.Params.Length / 2];

      for (int i = 0; i < ret.Length; i++)
      {
        ret[i] = (FP16)GetLayerLinear16Single(layer, i);
      }

      return ret;
    }


    /// <summary>
    /// Sets the weights values in a specified layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="index"></param>
    /// <param name="value"></param>
    public static void SetLayerLinear16(Weights.Layer layer, float[] values, float newMin, float newMax)
    {
      // Update bounds
      layer.MinVal = newMin;
      layer.MaxVal = newMax;

      // Rewrite shifted/scaled values
      float width = (newMax - newMin) / 65535.0f;
      for (int i = 0; i < values.Length; i++)
      {
        float value = values[i];

        float offset = value - layer.MinVal;
        float increment = MathF.Round(offset / width, 0);
        byte b0 = (byte)(increment % 256);
        byte b1 = (byte)(increment / 256);

        layer.Params[i * 2] = b0;
        layer.Params[i * 2 + 1] = b1;
      }
    }


    /// <summary>
    /// Gets a single layer vector within a speciifed layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="index"></param>
    /// <returns></returns>
    public static float GetLayerLinear16Single(Weights.Layer layer, int index)
    {
      return GetLayerLinear16Single(layer, index, layer.MinVal, (layer.MaxVal - layer.MinVal) / 65535.0f);
    }

    /// <summary>
    /// Gets a single layer vector within a speciifed layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="index"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GetLayerLinear16Single(Weights.Layer layer, int index, float minVal, float scale)
    {
      byte b0 = layer.Params[index * 2];
      byte b1 = layer.Params[index * 2 + 1];
      float v1 = 256 * b1 + b0;
      float v1a = minVal + v1 * scale;
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

      // Compute new min and max over array
      float newMin = float.MaxValue;
      float newMax = float.MinValue;
      for (int i = 0; i < values.Length; i++)
      {
        float value = values[i];

        if (value < newMin)
        {
          newMin = value;
        }

        if (value > newMax)
        {
          newMax = value;
        }
      }

      // Update layer
      SetLayerLinear16(layer, values, newMin, newMax);
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
      LC0ProtobufNet pbn = LC0ProtobufNet.LoadedNet(fnOrg);
      Weights.Layer layer = layerMap(pbn);

      // Get current layer values and get them rewritten
      float[] values = ProtobufHelpers.GetLayerLinear16(layer, out _);
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
