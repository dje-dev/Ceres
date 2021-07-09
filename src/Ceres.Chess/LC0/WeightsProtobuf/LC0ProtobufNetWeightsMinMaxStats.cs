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
using System.Reflection;

#endregion

namespace Ceres.Chess.LC0.WeightsProtobuf
{
  /// <summary>
  /// Uses reflection to find all Layer objects within an Pblczero.Weights object
  /// and collects statistics on the min/max values in each layer.
  /// </summary>
  public class LC0ProtobufNetWeightsMinMaxStats
  {
    /// <summary>
    /// Root object.
    /// </summary>
    public readonly Pblczero.Weights Weights;

    /// <summary>
    /// Set of collected layers (with their min/max.
    /// </summary>
    public List<(string, float, float)> Layers = new();

    /// <summary>
    /// Global min value across all layers.
    /// </summary>
    public float Min { private set; get; } = float.MaxValue;

    /// <summary>
    /// Global max value across all layers.
    /// </summary>
    public float Max { private set; get; } = float.MinValue;

    /// <summary>
    /// Name of layer having Min value.
    /// </summary>
    public string MinLayerName { private set; get; } = null;

    /// <summary>
    /// Name of layer having Min value.
    /// </summary>
    public string MaxLayerName { private set; get; } = null;


    // Track already processed object to avoid infinite recursion due to cycles.
    HashSet<object> seenObjects = new HashSet<object>();

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="weights"></param>
    public LC0ProtobufNetWeightsMinMaxStats(Pblczero.Weights weights)
    {
      Weights = weights;
      Collect("", Weights);
    }

    /// <summary>
    /// Use recursion to collect all properties which are Layers.
    /// </summary>
    void Collect(string name, object obj)
    {
      Type type = obj.GetType();
      FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
      PropertyInfo[] props = type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
      foreach (var prop in props)
      {
        bool isResiduals = prop.PropertyType == typeof(List<Pblczero.Weights.Residual>);
        bool isLayer = prop.PropertyType == typeof(Pblczero.Weights.Layer);

        object valueObj = null;
        try
        {
          valueObj = prop.GetValue(obj, null);
        }
        catch (Exception ee)
        {
        }

        if (isResiduals)
        {
          List<Pblczero.Weights.Residual> list = valueObj as List<Pblczero.Weights.Residual>;
          int count = 0;
          foreach (var residual in list)
          {
            Collect(name + "." + prop.Name + "[" + count++ + "]", residual);
          }
        }
        if (valueObj != null)
        {
          if (prop.PropertyType == typeof(Pblczero.Weights.Layer))
          {
            Pblczero.Weights.Layer value = prop.GetValue(obj, null) as Pblczero.Weights.Layer;
            string thisName = (name + "." + prop).Substring(1);
            Layers.Add((thisName, value.MinVal, value.MaxVal));
            if (value.MinVal < Min)
            {
              Min = value.MinVal;
              MinLayerName = thisName;
            }
            if (value.MaxVal > Max)
            {
              Max = value.MaxVal;
              MaxLayerName = thisName;
            }
          }
          else
          {
            if (!seenObjects.Contains(valueObj))
            {
              seenObjects.Add(valueObj);
              Collect(name + "." + prop.Name, valueObj);
            }

          }
        }
      }
    }
  }

}
