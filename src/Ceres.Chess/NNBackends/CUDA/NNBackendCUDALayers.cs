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
using System.Diagnostics;
using Ceres.Base.DataTypes;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  internal class NNBackendCUDALayers : IDisposable
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
    /// If the policy head uses attentino mechanism.
    /// </summary>
    public readonly bool PolicyIsAttention;

    public readonly bool attn_body_;
    public readonly int  num_encoder_blocks_;

    /// <summary>
    /// Default activation function.
    /// </summary>
    public readonly BaseLayerCUDA.ActivationFunction DefaultActivation;

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

    public readonly NNBackendExecContext Context;

    public readonly int DeviceComputeCapabilityMajor;

    public NNBackendCUDALayers(NNBackendExecContext context, int deviceComputeCapabilityMajor,
                               Net net, LC0LegacyWeights weights, 
                               bool saveActivations, NNBackendCUDALayers referenceLayers)
    {
      Context = context;
      DeviceComputeCapabilityMajor = deviceComputeCapabilityMajor;
      SaveActivations = saveActivations;
      NumFilters = (int)weights.input.biases.Length;
      NumBlocks = (int)weights.residual.Length;
      HasSE = NumBlocks > 0 && weights.residual[0].has_se;


      num_encoder_blocks_ = weights.encoder == null ? 0 :weights.encoder.Length;
      attn_body_ = (num_encoder_blocks_ > 0);
      if (attn_body_)
      {
        Debug.Assert(weights.ip_emb_b.Length > 0);
Console.WriteLine($"\nNum encoder blocks: {num_encoder_blocks_}, num policy encoder blocks: {weights.policyEncoders.Length}\n");
      }


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
        PolicyIsAttention = net.Format.NetworkFormat.Policy == NetworkFormat.PolicyFormat.PolicyAttention;
        if (net.Format.NetworkFormat.default_activation == NetworkFormat.DefaultActivation.DefaultActivationRelu)
        {
          DefaultActivation = BaseLayerCUDA.ActivationFunction.RELU;
        }
        else if (net.Format.NetworkFormat.default_activation == NetworkFormat.DefaultActivation.DefaultActivationMish)
        {
          DefaultActivation = BaseLayerCUDA.ActivationFunction.MISH;
        }
        else
        {
          throw new Exception("Unsupported activation function " + net.Format.NetworkFormat.default_activation);
        }
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
        execContext.Stream2.Synchronize();
      }
    }


    internal void DoBuildNetworkAndLoadWeights(NNBackendExecContext execContext, LC0LegacyWeights weights, int kNumInputPlanes)
    {
      BaseLayerCUDA.ActivationFunction activation = DefaultActivation;

      // Build the network, and copy the weights to GPU memory.


      BaseLayerCUDA resi_last_ = null;
      BaseLayerCUDA encoder_last_ = null;

      if (NumBlocks > 0)
      {
        // Input.
        FusedWinogradConvSELayerCUDA inputConv = new(execContext, "InputConv1", Layers.Count, NumFilters, 8, 8,
                                                      null, kNumInputPlanes, true, false, false, 0, true, activation);
        inputConv.LoadWeights(execContext.Stream2, weights.input.weights, weights.input.biases);
        Layers.Add(inputConv);

        for (int block = 0; block < NumBlocks; block++)
        {
          int se_k = HasSE ? (int)weights.residual[block].se.b1.Length : 0;

          ResidualBlockBaseCUDA layer = new ResidualBlockFusedCUDA(execContext, DeviceComputeCapabilityMajor,
                                                                   "residual_fused_" + block, Layers.Count, LastLayer,
                                                                   NumFilters, HasSE, se_k, block == 0,
                                                                   block == (NumBlocks - 1), execContext.SharedMemPerBlock, activation);

          layer.LoadWeights0(execContext.Stream2, weights.residual[block].conv1.weights, weights.residual[block].conv1.biases);
          layer.LoadWeights1(execContext.Stream2, weights.residual[block].conv2.weights, weights.residual[block].conv2.biases);

          if (HasSE)
          {
            layer.LoadSEWeights(weights.residual[block].se.w1, weights.residual[block].se.b1,
                                weights.residual[block].se.w2, weights.residual[block].se.b2);
          }

          Layers.Add(layer);
        }

        resi_last_ = LastLayer;
      }

      if (attn_body_)
      {
        throw new NotImplementedException();
        BaseLayerCUDA attention_body = default;// new AttentionBody(weights, scratch_mem_, act, numBlocks_,
                                               //                   numBlocks_ > 0 ? kNumFilters : kInputPlanes);
        Layers.Add(attention_body);

        encoder_last_ = LastLayer;
      }

      // Policy head.
      if (PolicyIsAttention)
      {
        int embeddingOpSize = weights.ip_pol_b.Length;
        int wqOpSize = weights.ip2_pol_b.Length;
        int wkOptSize = weights.ip3_pol_b.Length;
        int numEncoderHeads = weights.numPolicyEncoderHeads;

        AttentionPolicyHead attentionPolicy = null;
        attentionPolicy = new (execContext, "policy_conv1", Layers.Count, weights, 64 * 64 + 24 * 8, 1, 1, LastLayer, attn_body_, DefaultActivation);
        Layers.Add(attentionPolicy);

        LayerPolicyMapCUDA policymap = new(execContext, "policy_map", Layers.Count, LastLayer, 1858, 1, 1, 64 * 64 + 8 * 24, true);
        policymap.LoadWeights(PolicyMap.AttentionPolicyMap);
        Layers.Add(policymap);

#if NOT
        auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
            getLastLayer(), kNumOutputPolicy, 1, 1, 64 * 64 + 8 * 24, true);
        policymap->LoadWeights(kAttnPolicyMap, scratch_mem_);
        network_.emplace_back(std::move(policymap));


        auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
            getLastLayer(), kNumOutputPolicy, 1, 1, 73 * 8 * 8, false);
        policymap->LoadWeights(kConvPolicyMap, scratch_mem_);
        network_.emplace_back(std::move(policymap));
#endif
      }
      else if (PolicyIsConvolutional)
      {
        Debug.Assert(!attn_body_);  // not supported with attention body

        FusedWinogradConvSELayerCUDA conv1;
        conv1 = new(execContext, "policy_conv1", Layers.Count, NumFilters, 8, 8, resi_last_, NumFilters, true, false, false, 0, false, activation);
        conv1.LoadWeights(execContext.Stream, weights.policy1.weights, weights.policy1.biases);
        Layers.Add(conv1);

        int pol_channels = weights.policy.biases.Length;

        FusedWinogradConvSELayerCUDA conv2;
        conv2 = new(execContext, "policy_conv2", Layers.Count, pol_channels, 8, 8, LastLayer, NumFilters,  true, false, false, 0, false, BaseLayerCUDA.ActivationFunction.NONE);
        conv2.LoadWeights(execContext.Stream2, weights.policy.weights, weights.policy.biases);
        Layers.Add(conv2);

        LayerPolicyMapCUDA policymap = new(execContext, "policy_map", Layers.Count, LastLayer, 1858, 1, 1, 73 * 8 * 8, false);
        policymap.LoadWeights(PolicyMap.ConvolutionPolicyMap);
        Layers.Add(policymap);
      }
      else
      {
        Debug.Assert(!attn_body_);  // not supported with attention body

        LayerConv1CUDA convPol = new(execContext, "policy_conv", Layers.Count, resi_last_, weights.policy.biases.Length, 8, 8, NumFilters, true, activation);
        convPol.LoadWeights(weights.policy.weights, weights.policy.biases);
        Layers.Add(convPol);

        LayerFCCUDA FCPol = new(execContext, "policy_fc", Layers.Count, LastLayer, weights.ip_pol_b.Length, 1, 1, true, BaseLayerCUDA.ActivationFunction.NONE);
        FCPol.LoadWeights(weights.ip_pol_w, weights.ip_pol_b);
        Layers.Add(FCPol);
      }

      // Value head.
      if (attn_body_)
      {
        throw new NotImplementedException();
        BaseLayerCUDA valueEmbedding = default;// new LayerEmbedding(encoder_last_, weights.ip_val_w, weights.ip_val_b, scratch_mem_, act);
        Layers.Add(valueEmbedding);
      }
      else
      {
        LayerConv1CUDA convVal = new(execContext, "value_conv", Layers.Count, resi_last_, weights.value.biases.Length, 8, 8, NumFilters, true, activation);
        convVal.LoadWeights(weights.value.weights, weights.value.biases);
        Layers.Add(convVal);

        LayerFCCUDA FCVal1 = new(execContext, "value_fc1", Layers.Count, LastLayer, weights.ip1_val_b.Length, 1, 1, true, activation);
        FCVal1.LoadWeights(weights.ip1_val_w, weights.ip1_val_b);
        Layers.Add(FCVal1);

        bool fc2_tanh = !HasWDL;
        BaseLayerCUDA.ActivationFunction activationValue2 = fc2_tanh ? BaseLayerCUDA.ActivationFunction.TANH : BaseLayerCUDA.ActivationFunction.NONE;
        LayerFCCUDA FCVal2 = new(execContext, "value_fc2", Layers.Count, LastLayer, weights.ip2_val_b.Length, 1, 1, true, activationValue2);
        FCVal2.LoadWeights(weights.ip2_val_w, weights.ip2_val_b);
        Layers.Add(FCVal2);
      }


      // Moves left head
      if (HasMLH)
      {
        if (attn_body_)
        {
          throw new NotImplementedException();
          BaseLayerCUDA mlhEmbedding = default;// new EmbeddingLayer( encoder_last_, weights.ip_mov_w, weights.ip_mov_b, scratch_mem_, act);
          Layers.Add(mlhEmbedding);
        }
        else
        {
          LayerConv1CUDA convMov = new(execContext, "mlh_conv", Layers.Count, resi_last_, weights.moves_left.biases.Length, 8, 8, NumFilters, true, activation);
          convMov.LoadWeights(weights.moves_left.weights, weights.moves_left.biases);
          Layers.Add(convMov);

          LayerFCCUDA FCMov1 = new(execContext, "mlh_fc1", Layers.Count, LastLayer, weights.ip1_mov_b.Length, 1, 1, true, activation);
          FCMov1.LoadWeights(weights.ip1_mov_w, weights.ip1_mov_b);
          Layers.Add(FCMov1);

          LayerFCCUDA FCMov2 = new(execContext, "mlh_fc2", Layers.Count, LastLayer, 1, 1, 1, true, activation);
          FCMov2.LoadWeights(weights.ip2_mov_w, weights.ip2_mov_b);
          Layers.Add(FCMov2);
        }
      }     

      weightsAlreadyLoaded = true;
    }


    internal void RunLayers(CudaStream stream, int batchSize, 
                            NNInputCudaVariables inputs, 
                            NNOutputCudaVariables outputs)
    {
      int l = 0;

      CudaDeviceVariable<FP16> flow = inputs.Tensors[0];
      CudaDeviceVariable<FP16> spare1 = inputs.Tensors[1];
      CudaDeviceVariable<FP16> spare2 = inputs.Tensors[2];

      if (NumBlocks > 0)
      {
        // Input.
        Layers[l++].Eval(stream, batchSize, inputs.Tensors[1], inputs.Tensors[0], null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // input conv

        // Residual blocks
        for (int block = 0; block < NumBlocks; block++)
        {
          Layers[l++].Eval(stream, batchSize, inputs.Tensors[2], inputs.Tensors[1], null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);
        }


        flow = inputs.Tensors[2];
        spare1 = inputs.Tensors[0];
        spare2 = inputs.Tensors[1];
      }

      if (attn_body_)
      {
        throw new NotImplementedException();
#if not
        network_[l++]->Eval(batchSize, tensor_mem[1],
                            (numBlocks_ > 0) ? tensor_mem[2] : tensor_mem[0],
                            (numBlocks_ > 0) ? tensor_mem[0] : tensor_mem[2],
                            scratch_mem, scratch_size_, nullptr,
                            cublas, stream);  // Entire attention body of the network
#endif
        flow = inputs.Tensors[1];
        spare1 = inputs.Tensors[0];
        spare2 = inputs.Tensors[2];
      }
      // Policy head.
      if (PolicyIsAttention)
      {
#if NOT
      network_[l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], tensor_mem[1], scratch_mem, scratch_size_, nullptr, cublas, stream);  // Entire Attention policy head except for the policy map
      network_[l++]->Eval(batchSize, tensor_mem[1], tensor_mem[0], nullptr, scratch_mem, scratch_size_, nullptr, cublas, stream);  // policy map layer
      copyTypeConverted(opPol, (half*)(tensor_mem[1]), batchSize * kNumOutputPolicy, stream);  // POLICY output
#endif
        Layers[l++].Eval(stream, batchSize, spare1, flow, spare2, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);// Entire Attention policy head except for the policy map
        Layers[l++].Eval(stream, batchSize, outputs.PolicyOut, spare1, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf); // policy map layer
      }
      else if (PolicyIsConvolutional)
      {
#if NOT
      network_[l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], nullptr, scratch_mem, scratch_size_, nullptr, cublas, stream);  // policy conv1
      network_[l++]->Eval(batchSize, tensor_mem[1], tensor_mem[0], nullptr, scratch_mem, scratch_size_, nullptr, cublas, stream);  // policy conv2
      network_[l++]->Eval(batchSize, tensor_mem[0], tensor_mem[1], nullptr, scratch_mem, scratch_size_, nullptr, cublas, stream);  // policy map layer
      copyTypeConverted(opPol, (half*)(tensor_mem[0]), batchSize * kNumOutputPolicy, stream);  // POLICY output
#endif
        Layers[l++].Eval(stream, batchSize, spare1, flow, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // policy conv1
        Layers[l++].Eval(stream, batchSize, spare2, spare1, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // policy conv2
        Layers[l++].Eval(stream, batchSize, outputs.PolicyOut, spare2, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // policy map layer
      }
      else
      {
        Layers[l++].Eval(stream, batchSize, spare1, flow, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // pol conv
        Layers[l++].Eval(stream, batchSize, outputs.PolicyOut, spare1, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // pol FC
      }


      // value head
      Layers[l++].Eval(stream, batchSize, spare1, flow, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // value conv
      Layers[l++].Eval(stream, batchSize, outputs.ValueHeadFC2Out, spare1, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // value FC1

      Layers[l++].Eval(stream, batchSize, outputs.ValueOut, outputs.ValueHeadFC2Out, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // value FC2    

      if (HasMLH)
      {
        // Moves left head
        Layers[l++].Eval(stream, batchSize, spare1, flow, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // moves conv
        Layers[l++].Eval(stream, batchSize, spare2, spare1, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);  // moves FC1

        // Moves left FC2
        // TODO: consider fusing the bias-add of FC2 with format conversion.
        Layers[l++].Eval(stream, batchSize, outputs.MLHOut, spare2, null, inputs.Scratch, inputs.ScratchSizeBytes, inputs.ScratchSecondHalf);
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

    public void Dispose()
    {
      foreach (BaseLayerCUDA layer in Layers)
      {
        layer.Dispose();
      }
    }
  }

}
