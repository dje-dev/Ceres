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

//using HDF.PInvoke;
//using Hdf5DotNetTools;
//using System.Text.Json;

#endregion

namespace Ceres.Base.ANN
{
  /// <summary>
  ///  Constructs an ANNDef given names of files containing
  ///  a nework structure and network weights, as saved by Keras.
  /// </summary>
  public static class ANNLoaderKeras
  {
    // Two options for HDF5 access:
    //  - use Hdf5DotNetTools (build from source at https://github.com/reyntjesr/Hdf5DotnetTools)
    //    (currently used approach, but is bulky and has extraneous error messages)
    //  - better yet, switch to a lighter weight implementation such as https://github.com/silkfire/LiteHDF
    public static ANNDef ReadANNFromKeras(string fnH5Weights, string fnJSONStructure)
    {
      throw new NotImplementedException();
    }

#if NOT

Needs project reference:

    <PackageReference Include="HDF.PInvoke.NETStandard" Version="1.10.502" />

  <ItemGroup>
    <Reference Include="Hdf5DotnetTools">
      <HintPath>..\..\..\Hdf5DotnetTools\Hdf5DotNetTools\bin\Release\netstandard2.0\Hdf5DotnetTools.dll</HintPath>
    </Reference>
  </ItemGroup>

    public static ANNDef ReadANNFromKeras(string fnH5Weights, string fnJSONStructure)
    {
      ANNDef ann = new ANNDef();

      Console.WriteLine("** LOAD file " + fnH5Weights);
      if (!File.Exists(fnH5Weights)) throw new Exception("*** File not found");

      const bool READ_ONLY = true;
      long file = Hdf5.OpenFile(fnH5Weights, READ_ONLY);
      if (file < 0) throw new Exception("unable to find/open file " + fnH5Weights);

      bool inputLayerIsSparse = false;

      JsonDocument o = JsonDocument.Parse(File.ReadAllText(fnJSONStructure));
      List<(string, ANNDef)> subnets = new();

      inputLayerIsSparse = ExtractNetwork(ann, subnets, file, inputLayerIsSparse, o.RootElement, null);
      if (subnets.Count > 0) ann.InputSubnetworks = subnets;
      H5G.close(file);

      return ann;

    }

    private static bool ExtractNetwork(ANNDef ann, List<(string,ANNDef)> subnets, long file, 
                                       bool inputLayerIsSparse, JsonElement o, string path)
    {
      foreach (var obj in o.GetProperty("config").GetProperty("layers").EnumerateArray())
      {
        string className = obj.GetProperty("class_name").GetString();

        string name = obj.GetProperty("config").GetProperty("name").GetString();// (string)obj["name"];

        string innerPath = path == null ? ("/" + name) : (path + "/" + name);

        Console.Write(name + " " + className);

        if (className == "InputLayer")
        {
          inputLayerIsSparse = obj.GetProperty("config").GetProperty("sparse").GetBoolean();
        }
        else if (className == "DenseNormal")
        {
          // TODO: the path is hardcoded here because it seems to have a different convention.
          //       This works, but wuold fail if this were a subnetwork.
          ExtractDense(ann, file, inputLayerIsSparse, obj, "/dense_normal/dense_normal/dense_2/", "linear", 2);
          ann.OutputLayerUncertainty = ANNDef.OutputLayerUncertaintyType.DenseNormal;
        }
        else if (className == "DenseNormalGamma")
        {
          // TODO: the path is hardcoded here because it seems to have a different convention.
          //       This works, but wuold fail if this were a subnetwork.
          //dataset    /dense_normal_gamma/dense_normal_gamma/dense_2/bias:0
          //dataset    /dense_normal_gamma/dense_normal_gamma/dense_2/kernel:0
          ExtractDense(ann, file, inputLayerIsSparse, obj, "/dense_normal_gamma/dense_normal_gamma/dense_2/", "linear", 4);
          ann.OutputLayerUncertainty = ANNDef.OutputLayerUncertaintyType.DenseNormalGamma;
        }
        else if (className == "Dense")
        {
          ExtractDense(ann, file, inputLayerIsSparse, obj, innerPath);
        }
        else if (className == "Sequential")
        {
          if (subnets == null) throw new Exception("too much nesting");

          ANNDef annSubnet = new ANNDef();

//          long subgroupID = H5G.open(file, name);// "Root");

          ExtractNetwork(annSubnet, null, file, false, obj, innerPath);
          subnets.Add((name, annSubnet));
          //Console.WriteLine("Unknown "  + className + " " + name);
        }
//        Console.WriteLine();


      }
      return inputLayerIsSparse;
    }


    private static void ExtractDense(ANNDef ann, long file, bool inputLayerIsSparse, JsonElement obj, 
                                     string path, string? overrideActivation = null, int? overrideUnits = null)
    {
      // NOTE: use utility to see structure:
      //    h5dump --contents djechess_bi2.h5
      bool found = ((float[])Hdf5.ReadDataset<float>(file, path + "/bias:0")).Length > 0;
      if (!found)
      {
        // Hack, for some reason the group is appears twice sometimes (at root level)
        string firstPart = path.Substring(1);
        path = firstPart + "/" + path;
      }
      float[] bias = (float[])Hdf5.ReadDataset<float>(file, path + "/bias:0");
      //var weight_names = Hdf5.ReadStringAttributes(groupId, "weight_names"); // fails, in Python:  [n.decode('utf8') for n in g.attrs['weight_names']]

      float[,] weights = (float[,])Hdf5.ReadDataset<float>(file, path + "/kernel:0");
      int widthIn = weights.GetLength(0);
      int widthOut = weights.GetLength(1);
      Console.Write(" " + widthIn + "X" + widthOut);

      string activation = overrideActivation ?? obj.GetProperty("config").GetProperty("activation").GetString();
      int units = overrideUnits ?? obj.GetProperty("config").GetProperty("units").GetInt32(); 
      Console.Write(" " + units + " " + activation);
      string bnGroupName = path.Replace("_pre", "") + "_bn";
      bool bnExists = Hdf5.GroupExists(file, bnGroupName);
//      Console.WriteLine("group exists " + bnExists);
      float[] gamma = null;
      float[] beta = null;
      float[] moving_mean = null;
      float[] moving_variance = null;
      if (bnExists)
      {
        var groupIdBN = H5G.open(file, bnGroupName);
        gamma = (float[])Hdf5.ReadDataset<float>(groupIdBN, bnGroupName + "/gamma:0");
        beta = (float[])Hdf5.ReadDataset<float>(groupIdBN, bnGroupName + "/beta:0");
        moving_mean = (float[])Hdf5.ReadDataset<float>(groupIdBN, bnGroupName + "/moving_mean:0");
        moving_variance = (float[])Hdf5.ReadDataset<float>(groupIdBN, bnGroupName + "/moving_variance:0");
        Console.Write(" " + gamma + " " + beta + " " + moving_mean + " " + moving_variance);
        Hdf5.CloseGroup(groupIdBN);
      }

      ANNLayerDef.LayerSubtype subtype = ann.Layers.Count == 0 && inputLayerIsSparse ? ANNLayerDef.LayerSubtype.SparseBinaryInput : ANNLayerDef.LayerSubtype.Normal;

      ANNLayerDef.ActivationType activationType;
      if (activation.ToLower() == "linear")
        activationType = ANNLayerDef.ActivationType.Linear;
      else if (activation.ToLower() == "swish" || activation.ToLower() == "swish_keras")
        activationType = ANNLayerDef.ActivationType.Swish;
      else if (activation.ToLower() == "elu")
        activationType = ANNLayerDef.ActivationType.ELU;
      else if (activation.ToLower() == "relu")
        activationType = ANNLayerDef.ActivationType.RELU;
      else if (activation.ToLower() == "leakyrelu")
        activationType = ANNLayerDef.ActivationType.LeakyRELU;
      else if (activation.ToLower() == "selu")
        activationType = ANNLayerDef.ActivationType.SELU;
      else
        throw new Exception("unknown activation " + activation);

      ANNLayerDef thisLayer = ann.AddLayer(new ANNLayerDef(ann, path, widthIn, widthOut, weights, bias, activationType, subtype,
                                                           moving_mean, moving_variance, beta, gamma));
    }

#endif
  }
}
