﻿#region License notice

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
using ManagedCuda;
using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class AttentionPolicyHead : BaseLayerCUDA, IDisposable
  {
    bool attentionBody;
    int embeddingOpSize;
    int wqOpSize;
    int wkOptSize;
    int numEncoderHeads;
    int policy_d_model_;

    CudaDeviceVariable<FP16> ip_pol_w_;
    CudaDeviceVariable<FP16> ip_pol_b_;
    CudaDeviceVariable<FP16> ip2_pol_w_;
    CudaDeviceVariable<FP16> ip2_pol_b_;
    CudaDeviceVariable<FP16> ip3_pol_w_;
    CudaDeviceVariable<FP16> ip3_pol_b_;
    CudaDeviceVariable<FP16> ip4_pol_w_;

    CudaDeviceVariable<FP16> wqk_w;
    CudaDeviceVariable<FP16> wqk_b;

//    AttentionPolicyEncoderWeights[] encoderWeights;

    LayerEncoder[] encoderLayers;


    public AttentionPolicyHead(NNBackendExecContext parent, string name, int layerIndex,
                              LC0LegacyWeights weights,
                              int c, int h, int w, 
                              BaseLayerCUDA inputLayer, bool attentionBody, ActivationFunction activation)
      : base(parent, name, layerIndex, c, h, w, inputLayer, 
             attentionBody ? activation : ActivationFunction.SELU //// HACK : old networks without attention body (e.g: T79 use hardcoded SELU activations)
            )
    {
      this.attentionBody = attentionBody;
      embeddingOpSize = weights.ip_pol_b.Length;
      wqOpSize = weights.ip2_pol_b.Length;
      wkOptSize = weights.ip3_pol_b.Length;
      policy_d_model_ = wqOpSize;

      numEncoderHeads = weights.numPolicyEncoderHeads;

      ip_pol_w_ = LoadedWeights(weights.ip_pol_w);
      ip_pol_b_ = LoadedWeights(weights.ip_pol_b);
      ip2_pol_w_ = LoadedWeights(weights.ip2_pol_w);
      ip2_pol_b_ = LoadedWeights(weights.ip2_pol_b);
      ip3_pol_w_ = LoadedWeights(weights.ip3_pol_w);
      ip3_pol_b_ = LoadedWeights(weights.ip3_pol_b);
      ip4_pol_w_ = LoadedWeights(weights.ip4_pol_w);

      // big allocation to hold wq and wk weights one after the other
      int elements = weights.ip2_pol_w.Length;
      int size = elements * Marshal.SizeOf<FP16>() * 2;
      wqk_w = new CudaDeviceVariable<FP16>(elements * 2);
      wqk_w.CopyToDevice(ip2_pol_w_, 0, 0,      size / 2);
      wqk_w.CopyToDevice(ip3_pol_w_, 0, size/2, size/2);

      elements = weights.ip2_pol_b.Length;
      size = elements * Marshal.SizeOf<FP16>() * 2;
      wqk_b = new CudaDeviceVariable<FP16>(elements * 2);
      wqk_b.CopyToDevice(ip2_pol_b_, 0, 0, size / 2);
      wqk_b.CopyToDevice(ip3_pol_b_, 0, size / 2, size / 2);

      if (weights.policyEncoders.Length > 0)
      {
        LoadSoftmaxKernels();
        LoadLayerNormKernel();

        encoderLayers = new LayerEncoder[weights.policyEncoders.Length];
        for (int i = 0; i < encoderLayers.Length; i++)
        {
          EncoderWeights encoderWeights = new EncoderWeights(in weights.policyEncoders[i]);
          const int LAYER_INDEX = 0; // ???
          encoderLayers[i] = new LayerEncoder(parent, encoderWeights, "PolAttnEnc" + i, weights, LAYER_INDEX, 
                                              null, numEncoderHeads, embeddingOpSize, 1.0f, Activation);
        }
      }
    }

   

    CudaKernel kernelAddVectors;
    CudaKernel kernelPromotionLogits;
    CudaKernel kernelNCHWtoNHWC;

    public override void LoadKernels()
    {
      const string kn = "_ZN6lczero13cudnn_backend17addVectors_kernelI6__halfEEvPT_S4_S4_iiiNS0_18ActivationFunctionE";
      kernelAddVectors = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, kn);

      const string knPromotion = "_ZN6lczero13cudnn_backend23promotion_logits_kernelI6__halfEEviPT_PKS3_S6_S6_";
      kernelPromotionLogits = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, knPromotion);

      const string knNCHWtoNHWC = "_ZN6lczero13cudnn_backend17NCHWtoNHWC_kernelI6__halfS2_EEvPT_PKT0_iiiiii";
      kernelNCHWtoNHWC = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, knNCHWtoNHWC);

      LoadAddBiasBatchedKernels();
    }


    protected override void DoEval(CudaStream stream, int N,
                                   CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input,
                                   CudaDeviceVariable<FP16> input2, 
                                   CudaDeviceVariable<FP16> scratch, long scratchSizeBytes,
                                   CudaDeviceVariable<FP16> scratchSecondHalf = null)
    {
      CudaDeviceVariable<FP16> scratch0 = scratch;
      CudaDeviceVariable<FP16> scratch1 = input2;
      CudaDeviceVariable<FP16> scratch2 = new CudaDeviceVariable<FP16>(output.DevicePointer + scratchSizeBytes / (2 * Marshal.SizeOf<FP16>()));
      CudaDeviceVariable<FP16> scratch3 = new CudaDeviceVariable<FP16>(scratch1.DevicePointer + scratchSizeBytes / (2 * Marshal.SizeOf<FP16>()));

      int inputC = input_.C;
      if (!attentionBody)
      {
        int numElements = N * inputC * 8 * 8;
        const int BLOCK_DIM_NCHW = 256;
        kernelNCHWtoNHWC.GridDimensions = CUDAUtils.DivUp(numElements, BLOCK_DIM_NCHW);
        kernelNCHWtoNHWC.BlockDimensions = BLOCK_DIM_NCHW;
        LaunchKernel(stream, kernelNCHWtoNHWC, scratch0.DevicePointer, input.DevicePointer, N, inputC, N, inputC, 8, 8, stream.Stream.Pointer);
      }

      CudaDeviceVariable<FP16> pol_embedding = scratch1;

      unsafe
      {
        half halfZero = new half(0);
        void* ptrHalfZero = &halfZero;
        IntPtr ipHalfZero = (IntPtr)ptrHalfZero;
        half halfOne = new half(1);
        void* ptrHalfOne = &halfOne;
        IntPtr ipHalfOne = (IntPtr)ptrHalfOne;
        int num_outputs = embeddingOpSize;
        int num_inputs = inputC;
        int batch = N * 64;

        CudaBlasNativeMethods.cublasGemmEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                   num_outputs, batch, num_inputs,
                                   ipHalfOne, ip_pol_w_.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                   attentionBody ? input.DevicePointer : scratch0.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                   ipHalfZero, 
                                   pol_embedding.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs,
                                   ComputeType.Compute16F, GemmAlgo.Default);
        AddBiasBatched(GetAddBiasBatchedKernel(Activation), pol_embedding, pol_embedding, ip_pol_b_, 1, batch, num_outputs, stream);

      }

      // 2. Encoder layers
      for (int encoderHeadIndex = 0; encoderLayers != null && encoderHeadIndex < encoderLayers.Length; encoderHeadIndex++)
      {
#if NOT
    protected override void DoEval(CudaStream stream, int N, 
                                   CudaDeviceVariable<FP16> scratch1, CudaDeviceVariable<FP16> scratch0, 
                                   CudaDeviceVariable<FP16> scratch2,
                                   CudaDeviceVariable<FP16> scratch3, long scratch_size, 
                                   CudaDeviceVariable<FP16> scratchSecondHalf)
#endif
        encoderLayers[encoderHeadIndex].Eval(stream, N, scratch1, scratch0, scratch2, scratch3, 0, null);
      }

      CudaDeviceVariable<FP16> wq;
      CudaDeviceVariable<FP16> wk;

      int final_num_inputs = embeddingOpSize;
      int final_num_outputs = policy_d_model_;
      int final_batch = N * 64;
      wq = scratch0;
      wk = new CudaDeviceVariable<FP16>(wq.DevicePointer + final_num_outputs * final_batch * Marshal.SizeOf<FP16>());

      float factorPolicyDModel = 1.0f / MathF.Sqrt((float)policy_d_model_);

      unsafe
      {
        half halfZero = new half(0);
        half halfOne = new half(1);
        half halfFactor = new half(factorPolicyDModel);
        void* ptrHalfZero = &halfZero;
        void* ptrHalfOne = &halfOne;
        void* ptrHalfFactor = &halfFactor;
        IntPtr ipHalfZero = (IntPtr)ptrHalfZero;
        IntPtr ipHalfOne = (IntPtr)ptrHalfOne;
        IntPtr ipHalfFactor = (IntPtr)ptrHalfFactor;

        CudaBlasNativeMethods.cublasGemmStridedBatchedEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                                 final_num_outputs, final_batch, final_num_inputs,
                                                 ipHalfOne,
                                                 wqk_w.DevicePointer, cudaDataType.CUDA_R_16F, final_num_inputs, final_num_inputs*final_num_outputs,
                                                 scratch1.DevicePointer, cudaDataType.CUDA_R_16F, final_num_inputs, 0,
                                                 ipHalfZero,
                                                 wq.DevicePointer, cudaDataType.CUDA_R_16F, final_num_outputs, final_num_outputs*final_batch,
                                                 2, cudaDataType.CUDA_R_16F, GemmAlgo.Default);
        AddBiasBatched(kernelAddBiasBatchedNone, wq, wq, wqk_b, 2, final_batch, final_num_outputs, stream);
#if NOT
      cublasXGemmStridedBatched<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
          num_outputs, batch, num_inputs, 
          1.0f,
          wqk_w_, num_inputs, num_inputs * num_outputs, 
          scratch1, num_inputs, 0,
          0.0f, 
          wq, num_outputs, num_outputs * batch, 2);

      addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs,  NONE, stream);
#endif

        CudaBlasNativeMethods.cublasGemmStridedBatchedEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                                         (int)64, (int)64, policy_d_model_,
                                                         ipHalfFactor,
                                                         wk.DevicePointer, cudaDataType.CUDA_R_16F, policy_d_model_, 64 * policy_d_model_,
                                                         wq.DevicePointer, cudaDataType.CUDA_R_16F, policy_d_model_, 64 * policy_d_model_,
                                                         ipHalfZero, 
                                                         output.DevicePointer, cudaDataType.CUDA_R_16F, 64, 64 * 64 + 8 * 24,
                                                         N, cudaDataType.CUDA_R_16F, GemmAlgo.Default);
      }

      kernelPromotionLogits.GridDimensions = N;
      kernelPromotionLogits.BlockDimensions = new ManagedCuda.VectorTypes.dim3(24, 8, 1);

      int promotionsOffsetInBytes = Marshal.SizeOf<FP16>() * 64 * 64;
      CudaDeviceVariable<FP16> promotion_logits = new CudaDeviceVariable<FP16>(output.DevicePointer + promotionsOffsetInBytes);
      LaunchKernel(stream, kernelPromotionLogits, policy_d_model_,
                   promotion_logits.DevicePointer, 
                   wk.DevicePointer, 
                   ip4_pol_w_.DevicePointer, 
                   output.DevicePointer,
                   stream.Stream.Pointer);
    }


    void AddVectors(CudaStream stream, CudaDeviceVariable<FP16> c, CudaDeviceVariable<FP16> a, CudaDeviceVariable<FP16> b, 
                    int size, int asize, int bsize, ActivationFunction activation)
    {
      const int BLOCK_DIM = 256;
      int blocks = CUDAUtils.DivUp(size, BLOCK_DIM);
      kernelAddVectors.BlockDimensions = BLOCK_DIM;
      kernelAddVectors.GridDimensions = blocks;

      LaunchKernel(stream, kernelAddVectors, c, a, b, size, asize, bsize, (int)activation, stream.Stream.Pointer);
    }



    void FCLayer(CudaStream stream, int num_outputs, int batch, int num_inputs,
                 CudaDeviceVariable<FP16> weights, CudaDeviceVariable<FP16> biases,
                 CudaDeviceVariable<FP16> scratchX, CudaDeviceVariable<FP16> scratchY,
                 ActivationFunction activation)
    {
      CublasStatus err = CudaBlasNativeMethods.cublasHgemm(
                           Parent.CuBlas.CublasHandle,
                           Operation.Transpose, Operation.NonTranspose,
                           (int)num_outputs,
                           (int)batch,
                           (int)num_inputs,
                           ref CUDAUtils.halfOne,
                           weights.DevicePointer,
                           (int)num_inputs,
                           scratchX.DevicePointer,
                           num_inputs,
                           ref CUDAUtils.halfZero,
                           scratchY.DevicePointer,
                           (int)num_outputs);
      if (err != CublasStatus.Success)
      {
        throw new Exception("LayerFCCUDA failure " + err);
      }

      // TODO: consider using AddVectors method instead.
      const int BLOCK_DIM = 256;
      kernelAddVectors.BlockDimensions = BLOCK_DIM;
      kernelAddVectors.GridDimensions = CUDAUtils.DivUp(num_outputs * batch, BLOCK_DIM);

      LaunchKernel(stream, kernelAddVectors, scratchY.DevicePointer, biases.DevicePointer, scratchY.DevicePointer,
                 num_outputs * batch,
                 num_outputs,
                 num_outputs * batch,
                 (int)activation, stream.Stream.Pointer);
    }

    public override void Dispose()
    {
      ip_pol_w_?.Dispose();
      ip_pol_b_?.Dispose();
      ip2_pol_w_?.Dispose();
      ip2_pol_b_?.Dispose();
      ip3_pol_w_?.Dispose();
      ip3_pol_b_?.Dispose();

      wqk_w?.Dispose();
      wqk_b?.Dispose();

      if (encoderLayers != null)
      {
        for (int i = 0; i < encoderLayers.Length; i++)
        {
          encoderLayers[i].Dispose();
        }
      }
    }

  }
}
