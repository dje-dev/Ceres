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
using Ceres.Base.Misc;
using Pblczero;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading;

#endregion

namespace Ceres.Chess.LC0.WeightsProtobuf
{
  /// <summary>
  /// Represents an underlying Leela Chess Zero protobuf weights file.
  /// 
  /// NOTE: This class is based upon the LC0ProtobufDef class which is  machine generated
  /// from the protobuf files used by Leela Chess Zero (chunk.proto and net.proto).
  /// 
  /// To generate (or regenerate with newer version) this LC0ProtobufDef class, there are two options:
  ///   1. Use the online translator tool to generate C# classes: https://protogen.marcgravell.com/
  ///      This requires only to concatenate the two proto files (net and chunk), 
  ///      then remove the header from chunk and run the online generator.
  ///   2. Alternately a dotnet tool can be installed and run protogen on tensor.proto, for example:
  ///        dotnet tool install --global protobuf-net.Protogen
  ///        xcopy *.proto tensorflow\core\framework
  ///        for %f in (*.proto) do protogen --csharp_out=Ceres  -I. %f
  /// </summary>
  public class LC0ProtobufNet
  {
    /// <summary>
    /// Underling machine generated protobuf definition.
    /// </summary>
    public readonly Net Net;

    /// <summary>
    /// Number of convolutional filters.
    /// </summary>
    public int NumFilters => Net.Weights.Residuals[0].Conv1.BnMeans.Params.Length / 2;

    /// <summary>
    /// Number of SE blocks (layers).
    /// </summary>
    public int NumBlocks => Net.Weights.Residuals.Count;

    /// <summary>
    /// If the network supports the MLH (moves left head).
    /// </summary>
    public bool HasMovesLeft => Net.Format.NetworkFormat?.MovesLeft != null &&
                                Net.Format.NetworkFormat?.MovesLeft != NetworkFormat.MovesLeftFormat.MovesLeftNone;

    /// <summary>
    /// If the network supports the WDL (win/draw/loss) head.
    /// </summary>
    public bool HasWDL => Net.Format.NetworkFormat?.Output == NetworkFormat.OutputFormat.OutputWdl;


    /// <summary>
    /// Squeeze and Excitation ratio.
    /// </summary>
    public int? SERatio => (Net.Weights.Residuals[0].Se?.B2.Params.Length
                          / Net.Weights.Residuals[0].Se?.B1.Params.Length) / 2;

    /// <summary>
    /// Number of policy channels.
    /// </summary>
    public int? NumPolicyChannels => Net.Weights.Policy.BnBetas?.Params.Length / 2;

    /// <summary>
    /// Statistics about min/max values within weights layers.
    /// </summary>
    public LC0ProtobufNetWeightsMinMaxStats Stats => new LC0ProtobufNetWeightsMinMaxStats(Net.Weights);


    static ConcurrentDictionary<string, LC0ProtobufNet> cachedNets = new();

    /// <summary>
    /// Try to avoid concurrent loading of same nets.
    /// </summary>
    static ConcurrentDictionary<string, string> loadingNets = new();

    /// <summary>
    /// Loads and returns network stored in file with specified name.
    /// </summary>
    /// <param name="fn"></param>
    /// <returns></returns>
    public static LC0ProtobufNet LoadedNet(string fn)
    {
      LC0ProtobufNet net = null;

      while (loadingNets.ContainsKey(fn))
      {
        // Already being loaded.
        Thread.Sleep(50);
      }

      // TODO: consider having a bounded size of the cache to avoid excess memory use.
      loadingNets.TryAdd(fn, fn);
      if (!cachedNets.TryGetValue(fn, out net))
      {
        net = cachedNets[fn] = new LC0ProtobufNet(fn);
      }
      loadingNets.TryRemove(new KeyValuePair<string, string>(fn, fn));

      return net;
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="fn"></param>
    private LC0ProtobufNet(string fn)
    {
      if (!File.Exists(fn))
      {
        throw new ArgumentException($"No such protobuf file found {fn}");
      }

      // Read data from file, decompressing if necessary.
      byte[] data = FileUtils.IsZippedFile(fn) ? CompressionUtils.GetDecompressedBytes(fn) 
                                               : File.ReadAllBytes(fn);

      Net = SerializationUtils.ProtoDeserialize<Net>(data);

      if (Net == null)
      {
        throw new Exception($"Failure reading/parsing net {fn}");
      }

      if (Net.Format.NetworkFormat != null && 
          Net.Format.NetworkFormat.Input != NetworkFormat.InputFormat.InputClassical112Plane)
      {
        throw new Exception($"Only network format InputClassical112Plane is supported, not {Net.Format.NetworkFormat.Input} in {fn}.");
      }
    }


    /// <summary>
    /// Dumps summay of network characteristics to Console.
    /// </summary>
    public void Dump()
    {
      Console.WriteLine(this);

      Console.WriteLine("Filters         : " + Net.Weights.Residuals[0].Conv1.BnMeans.Params.Length / 2);
      Console.WriteLine("Blocks          : " + Net.Weights.Residuals.Count);

      Console.WriteLine("SE Ratio        : " + SERatio);
      Console.WriteLine("Policy channels : " + NumPolicyChannels);

      if (Net.Format.NetworkFormat != null)
      {
        Console.WriteLine("  " + Net.Format.NetworkFormat.Input);
        Console.WriteLine("  " + Net.Format.NetworkFormat.Network);
        Console.WriteLine("  " + Net.Format.NetworkFormat.Output);
        Console.WriteLine("  " + Net.Format.NetworkFormat.Policy);
        Console.WriteLine("  " + Net.Format.NetworkFormat.Value);
      }
      else
        Console.WriteLine("  Network format: cannot be determined (not saved in this version of the protobuf)");

      Console.WriteLine();
      Console.WriteLine($"LC0Params      { Net.TrainingParams.Lc0Params}");
      Console.WriteLine($"Learning rate  { Net.TrainingParams.LearningRate,10:F6}");
      Console.WriteLine($"Steps          { Net.TrainingParams.TrainingSteps, 10:N0}");
      Console.WriteLine($"MSE            { Net.TrainingParams.MseLoss, 10:F5}");
      Console.WriteLine($"Policy Loss    { Net.TrainingParams.PolicyLoss, 10:F2}");
      Console.WriteLine($"Accuracy       { Net.TrainingParams.Accuracy,  10:F2}%");
    }

  }
}
