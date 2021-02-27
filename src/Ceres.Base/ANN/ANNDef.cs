#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using Microsoft.Win32;
using System;
using System.Collections.Generic;

#endregion

namespace Ceres.Base.ANN
{
  /// <summary>
  /// Defines the format and weights associated with an artificial nerual network.
  /// </summary>
  [Serializable]
  public class ANNDef
  {
    /// <summary>
    /// Sequential layers of the network.
    /// </summary>
    public List<ANNLayerDef> Layers = new List<ANNLayerDef>();

    // Optionally a sequence of subnetworks which are interposed
    // between input the main layers of this network.
    // It is currently assumed that the input to this main network
    // is the concatenation of the output of the input subnetworks.
    public List<(string, ANNDef)> InputSubnetworks;

    /// <summary>
    /// Returns the cardinality of the expected input.
    /// </summary>
    public int WidthInput  => Layers[0].WidthIn;

    /// <summary>
    /// Returns the cardinality of the output layer.
    /// </summary>
    public int WidthOutput => Layers[Layers.Count-1].WidthOut;

    /// <summary>
    /// If the final layer is an evidential regression layer
    /// to measure uncertainty of output.
    /// 
    /// See "Deep Evidential Regression" by Amini et al.
    /// </summary>
    public enum OutputLayerUncertaintyType
    {
      None,
      DenseNormal,
      DenseNormalGamma
    }

    public OutputLayerUncertaintyType OutputLayerUncertainty = OutputLayerUncertaintyType.None;


    /// <summary>
    /// Adds specified layer definition to sequence of network layers.
    /// </summary>
    /// <param name="layer"></param>
    /// <returns></returns>
    public ANNLayerDef AddLayer(ANNLayerDef layer)
    {
      Layers.Add(layer);
      return layer;
    }

    /// <summary>
    /// Gets or sets if the first layer (inputs) is sparsely encoded.
    /// </summary>
    /// <param name="inputLayerMaxCountNonzeroPerRow"></param>
    public void SetInputsSparseBinary(int inputLayerMaxCountNonzeroPerRow)
    {
      Layers[0].InputLayerMaxCountNonzeroPerRow = inputLayerMaxCountNonzeroPerRow;
      Layers[0].Subtype = ANNLayerDef.LayerSubtype.SparseBinaryInput;
    }


    [ThreadStatic]
    static Random dropoutRand = new Random();


    /// <summary>
    /// Applies dropout to specified set ouf output activations.
    /// </summary>
    /// <param name="activations"></param>
    /// <param name="dropoutFraction"></param>
    static void ApplyDropout(float[] activations, float dropoutFraction)
    {
      if (dropoutFraction > 0)
      {
        float mult = 1.0f / (1.0f - dropoutFraction);
        for (int i = 0; i < activations.Length; i++)
        {
          if (dropoutRand.NextDouble() < dropoutFraction)
            activations[i] = 0.0f;
          else
            activations[i] *= mult; // rescale so keep save average magnitude.
        }
      }
    }



    /// <summary>
    /// Runs a specified batch of samples thru the network and returns output layer.
    /// </summary>
    /// <param name="numInputs"></param>
    /// <param name="input"></param>
    /// <param name="result"></param>
    /// <param name="dropoutFraction"></param>
    /// <returns></returns>
    public float[,] ComputeBatch(int numInputs, float[,] input, ANNCalcResult result, float dropoutFraction = 0.0f)
    {
      if (result.MultiCount < numInputs) throw new ArgumentException($"ANNCalcResult is too small to receive result {numInputs}");
      if (dropoutFraction != 0) throw new NotImplementedException();

      int batchSize = input.GetLength(0);
      if (numInputs != batchSize) throw new Exception("Batch size does not match input matrix dimension");
      if (InputSubnetworks != null)
      {
        // Allocate matrix to hold concatenated output of all subnetworks
        float[,] subnetsOutput = new float[batchSize, Layers[0].WidthIn];

        // Run each subnetwork
        int sumSubnetworkInputCounts = 0;
        int sumSubnetworksOutputCounts = 0;
        foreach ((string subnetName, ANNDef subnetDef) in InputSubnetworks)
        {
          // Extract inputs just for this subnetwork
          float[,] subnetInput = new float[batchSize, subnetDef.WidthInput];
          for (int i = 0; i < batchSize; i++)
          {
            for (int k = 0; k < subnetDef.WidthInput; k++)
              subnetInput[i, k] = input[i, sumSubnetworkInputCounts + k];
          }

          // Evaluate
          ANNCalcResult subResult = new ANNCalcResult(subnetDef, numInputs);
          subnetDef.ComputeBatch(numInputs, subnetInput, subResult, dropoutFraction);

          // Finally copy output of subnetowrk into our outputs
          for (int i = 0; i < batchSize; i++)
            for (int k = 0; k < subnetDef.WidthOutput; k++)
              subnetsOutput[i, k + sumSubnetworksOutputCounts] = subResult.LayerOutputBuffersMulti[^1][i, k];

          sumSubnetworkInputCounts += subnetDef.WidthInput;
          sumSubnetworksOutputCounts += subnetDef.WidthOutput;
        }

        if (sumSubnetworksOutputCounts != Layers[0].WidthIn)
          throw new Exception("Layer size mismatch");

        // Now switch over our inputs to be these subnet outputs
        input = subnetsOutput;
      }

      // Compute first
      Layers[0].ComputeMulti(numInputs, input, result.LayerOutputBuffersMulti[0]);

      // Compute any hidden
      for (int i = 1; i < Layers.Count; i++)
      {
        //using (new TimingContext(WidthInput + " CPU " + input.GetLength(0), SHOW_STATS))
        Layers[i].ComputeMulti(numInputs, result.LayerOutputBuffersMulti[i - 1], result.LayerOutputBuffersMulti[i]);
      }

      float[,] ths = result.LayerOutputBuffersMulti[^1];

      if (OutputLayerUncertainty == OutputLayerUncertaintyType.DenseNormal)
      {
        float[,] mus = new float[ths.GetLength(0), 1];
        for (int i=0; i<ths.GetLength(0);i++)
        {
          mus[i, 0] = ths[i, 0];

          float uncertainty = MathF.Log(MathF.Exp(ths[i, 1]) + 1) + 1E-6f;
        }
        return mus;
      }
      else if (OutputLayerUncertainty == OutputLayerUncertaintyType.DenseNormalGamma)
      {
        float[,] mus = new float[ths.GetLength(0), 1];
        for (int i = 0; i < ths.GetLength(0); i++)
        {
          mus[i, 0] = ths[i, 0];

          // See Equation 5 in section 3.2
          float v = MathF.Log(MathF.Exp(ths[i, 1]) + 1);
          float alpha = 1 + MathF.Log(MathF.Exp(ths[i, 2]) + 1);
          float beta = MathF.Log(MathF.Exp(ths[i, 3]) + 1);
          float aleatoricUncertainty = MathF.Sqrt(beta / (alpha - 1));
          float epistemicUncertainty = MathF.Sqrt(beta / (v * (alpha - 1)));

        }
        return mus;
      }

      // Return final
      return ths;

    }

  }
}
