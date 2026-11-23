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
using System.Globalization;
using System.Linq;

using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Remote;
using Ceres.Chess.NNEvaluators.Specifications.Internal;

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
  ///   "GPU:0,1@spark2-fast"        (remote evaluation on spark2-fast)
  ///   "GPU:0,1@spark2-fast/50060"  (remote with custom port)
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
    /// Optional arbitrary string which indicates a requested engine type (execution engine)
    /// to be used with this device (e.g. TensorRT).
    /// </summary>
    public readonly string OverrideEngineType;


    /// <summary>
    /// If evaluator is globally shared under a specified name, the name used to identify it.
    /// </summary>
    public readonly string SharingName;


    /// <summary>
    /// Constructor which parses a device specification string.
    /// 
    /// Examples:
    ///  "gpu:0"
    ///  "gpu:0,1"
    ///  "gpu:0,1:POOLED",
    ///  "GPU:0@0.66,1@0.34"
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

      // "+"-joined device groups, e.g. "GPU:0#TensorRTNative+GPU:0,1#TensorRTNative@HOP".
      // Each group is parsed with the existing syntax and tagged with its GroupIndex.
      // The factory then builds one sub-evaluator per group (one connection per remote host)
      // and combines them with a calibrated, latency-aware balanced split.
      if (equalsParts[0].Contains('+'))
      {
        string[] groupStrings = equalsParts[0].Split('+', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (groupStrings.Length < 2)
        {
          throw new Exception($"Device group specification requires at least two groups: {deviceString}");
        }

        for (int g = 0; g < groupStrings.Length; g++)
        {
          NNDevicesSpecificationString groupSpec = new(groupStrings[g]);
          if (groupSpec.ComboType == NNEvaluatorDeviceComboType.Pooled)
          {
            throw new Exception($"POOLED not supported within a device group: {groupStrings[g]}");
          }

          foreach ((NNEvaluatorDeviceDef devDef, float weight) in groupSpec.Devices)
          {
            if (devDef.Type != NNDeviceType.GPU)
            {
              throw new Exception($"Only GPU devices supported in device groups: {groupStrings[g]}");
            }
            devDef.GroupIndex = g;
            Devices.Add((devDef, weight));
          }
        }

        ComboType = NNEvaluatorDeviceComboType.Split;
        return;
      }

      string[] deviceStringParts = equalsParts[0].Split(":");

      if (deviceStringParts[0].ToUpper() == "CPU")
      {
        Devices.Add((new NNEvaluatorDeviceDef(NNDeviceType.CPU, 0, false, null, null, null), 1));
        ComboType = NNEvaluatorDeviceComboType.Single;
        return;
      }

      if (deviceStringParts.Length is not (2 or 3))
      {
        throw new Exception($"{deviceString} not valid device specification string {exampleString}");
      }

      // Device specification such as "GPU:0,1" or "GPU:0,1:POOLED"
      string deviceTypeStr = deviceStringParts[0].ToUpper();
      if (deviceTypeStr != "GPU")
      {
        throw new Exception($"{deviceString} not valid, device specification expected to begin with 'GPU:'");
      }

      // Extract @hostname from device indices string if present.
      // Disambiguation: if the value after the last '@' is NOT a valid float, it's a hostname.
      // Example: "0,1@spark2-fast" -> hostname="spark2-fast", devicesStr="0,1"
      // Example: "0@0.75,1@0.25@myhost" -> hostname="myhost", devicesStr="0@0.75,1@0.25"
      string remoteHost = null;
      int remotePort = NNRemoteProtocol.DEFAULT_PORT;
      int remoteStreamIndex = 0;
      string devicesStr = deviceStringParts[1];

      int lastAt = devicesStr.LastIndexOf('@');
      if (lastAt >= 0)
      {
        string afterAt = devicesStr.Substring(lastAt + 1);
        if (!float.TryParse(afterAt, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
        {
          // Not a weight -- it's a hostname, possibly with a "/port" suffix and (rarely written by hand,
          // but emitted by ToSpecificationString) a further "/stream" suffix, e.g. "station/50055/2".
          devicesStr = devicesStr.Substring(0, lastAt);

          string[] hostParts = afterAt.Split('/');
          if (hostParts.Length > 3 || hostParts[0].Length == 0)
          {
            throw new Exception($"Invalid remote host specification '@{afterAt}' in {deviceString}, expected @host[/port[/stream]]");
          }

          remoteHost = hostParts[0];
          if (hostParts.Length >= 2)
          {
            remotePort = int.Parse(hostParts[1], CultureInfo.InvariantCulture);
          }
          if (hostParts.Length == 3)
          {
            remoteStreamIndex = int.Parse(hostParts[2], CultureInfo.InvariantCulture);
          }
        }
      }

      var deviceSpec = OptionsParserHelpers.ParseDeviceOptions(devicesStr);
      OverrideEngineType = deviceSpec.Item2;

      foreach (var device in deviceSpec.Item1)
      {
        int? maxBatchSize = device.maxBatchSize;
        int? optimalBatchSize = device.optimalBatchSize;
        List<(int, int[])> predefinedPartitions = null;
        if (device.batchSizesFileName != null)
        {
          var parsedBatchFile = BatchSizeFileParser.ParseFromFile(device.batchSizesFileName);
          maxBatchSize = parsedBatchFile.maxBatchSize;
          predefinedPartitions = parsedBatchFile.predefinedPartitions;
        }
        NNEvaluatorDeviceDef devDef = new NNEvaluatorDeviceDef(NNDeviceType.GPU, int.Parse(device.deviceID), false,
                                              maxBatchSize, optimalBatchSize, deviceSpec.overrideEngineType,
                                              predefinedPartitions);
        // Set remote host if extracted from the device string.
        if (remoteHost != null)
        {
          devDef.RemoteHost = remoteHost;
          devDef.RemotePort = remotePort;
          devDef.StreamIndex = remoteStreamIndex;
        }
        Devices.Add((devDef, device.weight));
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
      // Device groups are rendered back as "+"-joined group specifications.
      // The same rendering is used for a (non-grouped) specification whose devices all live on one
      // remote host, e.g. "GPU:0,1#TensorRTNative@HOP/50055" (the general rendering below omits the host,
      // which would make an engine launched from this string silently evaluate locally).
      NNEvaluatorDeviceDef firstDevice = devices.First().Item1;
      bool allOnOneRemoteHost = firstDevice.RemoteHost != null
                             && devices.All(d => d.Item1.RemoteHost == firstDevice.RemoteHost
                                              && d.Item1.RemotePort == firstDevice.RemotePort
                                              && d.Item1.StreamIndex == firstDevice.StreamIndex);
      if (devices.Any(d => d.Item1.GroupIndex >= 0) || (allOnOneRemoteHost && comboType != NNEvaluatorDeviceComboType.Pooled))
      {
        IEnumerable<string> groupStrs = devices.GroupBy(d => d.Item1.GroupIndex).OrderBy(g => g.Key).Select(g =>
        {
          NNEvaluatorDeviceDef first = g.First().Item1;
          string engine = first.OverrideEngineType != null ? "#" + first.OverrideEngineType : "";
          string stream = first.StreamIndex != 0 ? "/" + first.StreamIndex : "";
          string remote = first.RemoteHost != null ? $"@{first.RemoteHost}/{first.RemotePort}{stream}" : "";
          return $"GPU:{string.Join(",", g.Select(d => d.Item1.DeviceIndex))}{engine}{remote}";
        });
        return "Device=" + string.Join("+", groupStrs);
      }

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

          if (weight != 1)
          {
            str += $"@{weight}";
          }

          if (device.PredefinedOptimalBatchPartitions != null)
          {
            str += $"[<batch_size_file>]";
          }

          if (device.MaxBatchSize.HasValue)
          {
            if (device.MaxBatchSize.HasValue)
            {
              str += $"[{device.OptimalBatchSize.Value}..{device.MaxBatchSize.Value}]";
            }
            else
            {
              str += $"[{device.MaxBatchSize.Value}]";
            }
          }

          if (device.OverrideEngineType != null)
          {
            str += $"#{device.OverrideEngineType}";
          }

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
