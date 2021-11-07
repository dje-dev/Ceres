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
using System.Linq;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications.Iternal;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Chess.NNEvaluators.Specifications
{
  /// <summary>
  /// Manages parsing of a device specification string for neural net evaluation which 
  /// specifies the set of devices to be used in evaluation.
  /// 
  /// Examples: 
  ///   "GPU:0"
  ///   "GPU:0@0.5,1@0.5"
  public record NNDevicesSpecificationString
  {
    /// <summary>
    /// Method used to combine (possibly) multiple devices.
    /// </summary>
    public readonly NNEvaluatorDeviceComboType ComboType;

    /// <summary>
    /// List of devices to be used, and their fractional weights to be used 
    /// when dividing batches across the devices.
    /// </summary>
    public readonly List<(NNEvaluatorDeviceDef, float)> Devices;

    /// <summary>
    /// If evaluator is globally shared under a specified name, the name used to identify it.
    /// </summary>
    public readonly string SharingName;


    /// <summary>
    /// Constructor which parses a device specification string.
    /// </summary>
    /// <param name="netString"></param>
    public NNDevicesSpecificationString(string deviceString)
    {
      const string exampleString = ", valid examples: GPU:0 or GPU:0,1 or GPU:0@0.75,1@0.25:POOLED";
      
      Devices = new List<(NNEvaluatorDeviceDef, float)>();

      string[] equalsParts = deviceString.Split("=");
      if (equalsParts.Length == 2)
      {
        SharingName = equalsParts[1];
      }
      else if (equalsParts.Length > 2)
      {
        throw new Exception("Device specification string has too many equals parts " + deviceString);
      }

      string[] deviceStringParts = equalsParts[0].Split(":");

      if (deviceStringParts.Length is not (2 or 3))
      {
        throw new Exception($"{deviceString} not valid device specification string { exampleString}");
      }

      // Device specification such as "GPU:0,1" or "GPU:0,1:POOLED"
      string deviceTypeStr = deviceStringParts[0].ToUpper();
      if (deviceTypeStr != "GPU")
      { 
        throw new Exception($"{deviceString} not valid, device specification expected to begin with 'GPU:'");
      }

      List<(string, NNEvaluatorPrecision, float, float, float)> deviceParts = OptionsParserHelpers.ParseCommaSeparatedWithOptionalWeights(deviceStringParts[1], false);

      foreach (var device in deviceParts)
      {
        Devices.Add((new NNEvaluatorDeviceDef(NNDeviceType.GPU, int.Parse(device.Item1)), device.Item3));
      }

      ComboType = Devices.Count == 1 ? NNEvaluatorDeviceComboType.Single
                                     : NNEvaluatorDeviceComboType.Split;

      // Possibly switch combination type to pooled if requested
      if (deviceStringParts.Length == 3)
      {
        if (deviceStringParts[2].ToUpper() != "POOLED")
        {
          throw new Exception("Unexpected third part, expected POOLED");
        }

        ComboType = NNEvaluatorDeviceComboType.Pooled;        
      }
    }


    /// <summary>
    /// Returns if specified device string is valid.
    /// </summary>
    /// <param name="deviceString"></param>
    /// <returns></returns>
    public static bool IsValid(string deviceString)
    {
      try
      {
        new NNDevicesSpecificationString(deviceString);
        return true;
      }
      catch
      {
        return false;
      }
    }

    /// <summary>
    /// Static helper method to convert specification components into equivalent specification string.
    /// </summary>
    /// <param name="comboType"></param>
    /// <param name="devices"></param>
    /// <returns></returns>
    static string ToSpecificationStringComplex(NNEvaluatorDeviceComboType comboType,
                                               IEnumerable<(NNEvaluatorDeviceDef, float)> devices)
    {
      string ret = $"Devices: {comboType} ";
      foreach (var net in devices)
        ret += $"({net.Item1} {net.Item2}) ";
      return ret;
    }

    /// <summary>
    /// Static helper method to convert specification components into equivalent specification string.
    /// </summary>
    /// <param name="comboType"></param>
    /// <param name="devices"></param>
    /// <returns></returns>
    public static string ToSpecificationString(NNEvaluatorDeviceComboType comboType,
                                               IEnumerable<(NNEvaluatorDeviceDef, float)> devices)
    {
      int count = 0;
      // TODO: Currently support conversion back to original specification string only for simple cases
      if ((comboType == NNEvaluatorDeviceComboType.Single
        || comboType == NNEvaluatorDeviceComboType.Split
        || comboType == NNEvaluatorDeviceComboType.Pooled)
       && devices.First().Item1.Type == NNDeviceType.GPU)
      {
        string str = $"Device=GPU:";
        foreach ((NNEvaluatorDeviceDef device, float weight) in devices)
        {
          if (count > 0) str += ",";
          str += $"{device.DeviceIndex}";
          if (weight != 1) str+= $"@{weight}";
          count++;
        }
        if (comboType == NNEvaluatorDeviceComboType.Pooled)
        {
          str += ":POOLED";
        }
        return str;
      }
      else
      {
        return ToSpecificationStringComplex(comboType, devices);
      }      
    }

  }
}
