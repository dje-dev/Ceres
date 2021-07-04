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
using ManagedCuda;
using Pblczero;
using Ceres.Base.Math;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  internal class NNBackendCUDALayers
  {
    /// <summary>
    /// List of underlying layers.
    /// </summary>
    internal List<BaseLayerCUDA> Layers = new List<BaseLayerCUDA>();
    internal BaseLayerCUDA LastLayer => Layers[Layers.Count - 1];

    /// <summary>
    /// Network dimension (number of convolutional filters).
    /// </summary>
    public readonly int NumFilters;

    /// <summary>
    /// Network dimension (number of residual blocks).
    /// </summary>
    public readonly int NumBlocks;

    /// <summary>
    /// If the network as a WDL (win/draw/loss) head.
    /// </summary>
    public readonly bool HasWDL;

    /// <summary>
    /// If the network as an MLH (moves left) head.
    /// </summary>
    public readonly bool HasMLH;

    /// <summary>
    /// If the network uses squeeze and excitation feature.
    /// </summary>
    public readonly bool HasSE;

    /// <summary>
    /// If the policy head uses convolutional layer.
    /// </summary>
    public readonly bool PolicyIsConvolutional;

    /// <summary>
    /// If some of the output activation should be captured and returned.
    /// </summary>
    public readonly bool SaveActivations;

    internal NNBackendCUDALayers referenceLayers;
    internal CudaKernel expandPlanesKernel;
    internal CudaKernel maskedMovesKernel;


    /// <summary>
    /// A "fingerprint" of one of the layers (using a hash function)
    /// used an approximate way to testing two instances for (probable) equality.
    /// </summary>
    float valueLayer1BiasSum;

   
    public NNBackendCUDALayers(NNBackendExecContext context, Net net, LC0LegacyWeights weights, 
                               bool saveActivations, NNBackendCUDALayers referenceLayers)
    {
      SaveActivations = saveActivations;
      NumFilters = (int)weights.input.biases.Length;
      NumBlocks = (int)weights.residual.Length;
      HasSE = weights.residual[0].has_se;

      this.referenceLayers = referenceLayers;

      // Record the hash of one of the arrays of weights as fingerprint of these weights.
      valueLayer1BiasSum = StatUtils.Sum(weights.ip1_val_b);


      if (net.Format.NetworkFormat == null)
      {
        HasWDL = false;
        HasMLH = false;
        PolicyIsConvolutional = false;
      }
      else
      {
        HasWDL = net.Format.NetworkFormat.Value == NetworkFormat.ValueFormat.ValueWdl;
        HasMLH = net.Format.NetworkFormat.MovesLeft == NetworkFormat.MovesLeftFormat.MovesLeftV1;
        PolicyIsConvolutional = net.Format.NetworkFormat.Policy == NetworkFormat.PolicyFormat.PolicyConvolution;
      }

      InitKernels(context);
    }


    void InitKernels(NNBackendExecContext context)
    {
      const string ptxFileName = @"common_kernels.ptx";
      const string ptxFileNameCeres = @"ceres_kernels.ptx";

      const string maskMovesKernelName = "_ZN5ceres21copyMaskedMovesKernelEP6__halfPsPfi";
      maskedMovesKernel = context.Device.GetKernel(context.PTXAssembly, ptxFileNameCeres, maskMovesKernelName);

      const string expandPlanesKernelName = "_ZN6lczero13cudnn_backend29expandPlanes_kernel_Fp16_NCHWEP6__halfPKyPKfi";
      expandPlanesKernel = context.Device.GetKernel(context.PTXAssembly, ptxFileName, expandPlanesKernelName);

      //const string shiftConvertKernelName = "_ZN5ceres18shiftConvertKernelEP6__halfPcffi";
      //shiftConvertKernel = Context.GetKernel(ptxFileNameCeres, shiftConvertKernelName);
    }


    bool weightsAlreadyLoaded = false;

    internal void BuildNetworkAndLoadWeights(NNBackendExecContext execContext, LC0LegacyWeights weights, int kNumInputPlanes)
    {
      if (weightsAlreadyLoaded)
      {
        if (valueLayer1BiasSum != StatUtils.Sum(weights.ip1_val_b))
        {
          throw new Exception("Weights appear different in reused NNBackendCUDALayers.");
        }
        return;
      }
      else
      {
        DoBuildNetworkAndLoadWeights(execContext, weights, kNumInputPlanes);
      }
    }


    internal void DoBuildNetworkAndLoadWeights(NNBackendExecContext execContext, LC0LegacyWeights weights, int kNumInputPlanes)
    {
      // Build the network, and copy the weights to GPU memory.

      // Input.
      FusedWinogradConvSELayerCUDA inputConv = new(execContext, "InputConv1", Layers.Count, NumFilters, 8, 8, null, kNumInputPlanes, true, true, false, false, 0, true);
      inputConv.LoadWeights(execContext.Stream, weights.input.weights, weights.input.biases);
      Layers.Add(inputConv);

      for (int block = 0; block < NumBlocks; block++)
      {
        int se_k = HasSE ? (int)weights.residual[block].se.b1.Length : 0;

        ResidualBlockBaseCUDA layer = new ResidualBlockFusedCUDA(execContext, "residual_fused_" + block, Layers.Count, LastLayer, NumFilters, HasSE, se_k, block == 0, block == (NumBlocks - 1));

        layer.LoadWeights0(execContext.Stream, weights.residual[block].conv1.weights, weights.residual[block].conv1.biases);
        layer.LoadWeights1(execContext.Stream, weights.residual[block].conv2.weights, weights.residual[block].conv2.biases);

        if (HasSE)
        {
          layer.LoadSEWeights(weights.residual[block].se.w1, weights.residual[block].se.b1,
                              weights.residual[block].se.w2, weights.residual[block].se.b2);
        }

        Layers.Add(layer);
      }

      BaseLayerCUDA resi_last_ = LastLayer;

      // Policy head.
      if (PolicyIsConvolutional)
      {
        FusedWinogradConvSELayerCUDA conv1;
        conv1 = new(execContext, "policy_conv1", Layers.Count, NumFilters, 8, 8, resi_last_, NumFilters, true, true, false, false, 0, false);
        conv1.LoadWeights(execContext.Stream, weights.policy1.weights, weights.policy1.biases);
        Layers.Add(conv1);

        int pol_channels = weights.policy.biases.Length;

        FusedWinogradConvSELayerCUDA conv2;
        conv2 = new(execContext, "policy_conv2", Layers.Count, pol_channels, 8, 8, LastLayer, NumFilters, false, true, false, false, 0, false);
        conv2.LoadWeights(execContext.Stream, weights.policy.weights, weights.policy.biases);
        Layers.Add(conv2);

        LayerPolicyMapCUDA policymap = new(execContext, "policy_map", Layers.Count, LastLayer, 1858, 1, 1, 73 * 8 * 8);
        policymap.LoadWeights(PolicyMap.ConvolutionPolicyMap);
        Layers.Add(policymap);
      }
      else
      {
        LayerConv1CUDA convPol = new(execContext, "policy_conv", Layers.Count, resi_last_, weights.policy.biases.Length, 8, 8, NumFilters, true, true);
        convPol.LoadWeights(weights.policy.weights, weights.policy.biases);
        Layers.Add(convPol);

        LayerFCCUDA FCPol = new(execContext, "policy_fc", Layers.Count, LastLayer, weights.ip_pol_b.Length, 1, 1, false, true);
        FCPol.LoadWeights(weights.ip_pol_w, weights.ip_pol_b);
        Layers.Add(FCPol);
      }

      BaseLayerCUDA policy_out_ = LastLayer;

      // Value head.
      LayerConv1CUDA convVal = new(execContext, "value_conv", Layers.Count, resi_last_, weights.value.biases.Length, 8, 8, NumFilters, true, true);
      convVal.LoadWeights(weights.value.weights, weights.value.biases);
      Layers.Add(convVal);

      LayerFCCUDA FCVal1 = new(execContext, "value_fc1", Layers.Count, LastLayer, weights.ip1_val_b.Length, 1, 1, true, true);
      FCVal1.LoadWeights(weights.ip1_val_w, weights.ip1_val_b);
      Layers.Add(FCVal1);

      bool fc2_tanh = !HasWDL;

      LayerFCCUDA FCVal2 = new(execContext, "value_fc2", Layers.Count, LastLayer, weights.ip2_val_b.Length, 1, 1, false, true, fc2_tanh);
      FCVal2.LoadWeights(weights.ip2_val_w, weights.ip2_val_b);
      Layers.Add(FCVal2);

      var value_out_ = LastLayer;

      // Moves left head
      BaseLayerCUDA moves_left_out_ = null;
      if (HasMLH)
      {
        LayerConv1CUDA convMov = new(execContext, "mlh_conv", Layers.Count, resi_last_, weights.moves_left.biases.Length, 8, 8, NumFilters, true, true);
        convMov.LoadWeights(weights.moves_left.weights, weights.moves_left.biases);
        Layers.Add(convMov);

        LayerFCCUDA FCMov1 = new(execContext, "mlh_fc1", Layers.Count, LastLayer, weights.ip1_mov_b.Length, 1, 1, true, true);
        FCMov1.LoadWeights(weights.ip1_mov_w, weights.ip1_mov_b);
        Layers.Add(FCMov1);

        LayerFCCUDA FCMov2 = new(execContext, "mlh_fc2", Layers.Count, LastLayer, 1, 1, 1, true, true);
        FCMov2.LoadWeights(weights.ip2_mov_w, weights.ip2_mov_b);
        Layers.Add(FCMov2);

        moves_left_out_ = LastLayer;
      }
      
      weightsAlreadyLoaded = true;
    }


    internal void RunLayers(CudaStream stream, int batchSize, 
                            NNInputCudaVariables inputs, 
                            NNOutputCudaVariables outputs)
    {
      int l = 0;

      // Input.
      Layers[l++].Eval(stream, batchSize, inputs.Tensors[1], inputs.Tensors[0], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // input conv

      // Residual blocks
      for (int block = 0; block < NumBlocks; block++)
      {
        Layers[l++].Eval(stream, batchSize, inputs.Tensors[2], inputs.Tensors[1], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);
      }

      // Policy head.
      if (PolicyIsConvolutional)
      {
        Layers[l++].Eval(stream, batchSize, inputs.Tensors[0], inputs.Tensors[2], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // policy conv1
        Layers[l++].Eval(stream, batchSize, inputs.Tensors[1], inputs.Tensors[0], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // policy conv2
        Layers[l++].Eval(stream, batchSize, outputs.PolicyOut, inputs.Tensors[1], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // policy map layer
      }
      else
      {
        Layers[l++].Eval(stream, batchSize, inputs.Tensors[0], inputs.Tensors[2], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // pol conv
        Layers[l++].Eval(stream, batchSize, outputs.PolicyOut, inputs.Tensors[0], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // pol FC
      }


      // value head
      Layers[l++].Eval(stream, batchSize, inputs.Tensors[0], inputs.Tensors[2], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // value conv
      Layers[l++].Eval(stream, batchSize, outputs.ValueHeadFC2Out, inputs.Tensors[0], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // value FC1


      Layers[l++].Eval(stream, batchSize, outputs.ValueOut, outputs.ValueHeadFC2Out, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // value FC2    

      if (HasMLH)
      {
        // Moves left head
        Layers[l++].Eval(stream, batchSize, inputs.Tensors[0], inputs.Tensors[2], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // moves conv
        Layers[l++].Eval(stream, batchSize, inputs.Tensors[1], inputs.Tensors[0], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // moves FC1

        // Moves left FC2
        // TODO: consider fusing the bias-add of FC2 with format conversion.
        Layers[l++].Eval(stream, batchSize, outputs.MLHOut, inputs.Tensors[1], inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);
      }
    }


    // TODO: these should be unique to each executor (?)
    public void DumpTimings()
    {
      Console.WriteLine();
      Console.WriteLine("Network Timings (milliseconds)");
      for (int i = 0; i < Layers.Count; i++)
      {
        BaseLayerCUDA layer = Layers[i];
        Console.WriteLine(layer.ToString());
      }
    }
  }

}
