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

using ManagedCuda;
using Ceres.Base.DataTypes;
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using ManagedCuda.CudaBlas;
using ManagedCuda.BasicTypes;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class LayerAttentionBody : BaseLayerCUDA, IDisposable
  {
    int num_res_blocks;
    int input_c;
    int embedding_op_size_;
    int encoder_head_count_;
    LayerEncoder[] encoders;

    CudaDeviceVariable<FP16> ip_emb_w_;
    CudaDeviceVariable<FP16> ip_emb_b_;    // "embedding" layer in net body

    public LayerAttentionBody(NNBackendExecContext parent,
                              string name, LC0LegacyWeights weights, int layerIndex, BaseLayerCUDA ip,
                              int num_res_blocks, int input_c, ActivationFunction activation)
             : base(parent, name, layerIndex, weights.ip_emb_b.Length, 8, 8, ip, activation)
    {
      this.num_res_blocks = num_res_blocks;
      this.input_c = input_c;
      embedding_op_size_ = weights.ip_emb_b.Length;
      encoder_head_count_ = weights.encoder_head_count;

      ip_emb_w_ = LoadedWeights(weights.ip_emb_w);
      ip_emb_b_ = LoadedWeights(weights.ip_emb_b);

      int num_encoders = weights.encoder.Length;
      float alpha = (float)MathF.Pow(2.0f * num_encoders, 0.25f);
      encoders = new LayerEncoder[num_encoders];
      for (int i=0; i<num_encoders;i++)
      {
        EncoderWeights encoderWeights = new EncoderWeights(in weights.encoder[i]);
        encoders[i] = new LayerEncoder(parent, encoderWeights, "attn_enc_" + i, weights, -1, null, 
                                       encoder_head_count_, embedding_op_size_, alpha, activation);
      }
    }

    public override void LoadKernels()
    {
      LoadAttentionPreprocess();
      LoadAddBiasBatchedKernels();
    }

    protected override void DoEval(CudaStream stream, int N,
                                   CudaDeviceVariable<FP16> output,
                                   CudaDeviceVariable<FP16> input, CudaDeviceVariable<FP16> input2,
                                   CudaDeviceVariable<FP16> scratch, long scratchSizeBytes, CudaDeviceVariable<FP16> scratchSecondHalf = null)
    {
      CudaDeviceVariable<FP16> scratch0 = scratch;
      CudaDeviceVariable<FP16> scratch1 = output;
      CudaDeviceVariable<FP16> scratch2 = input2;
      CudaDeviceVariable<FP16> scratch3 = new CudaDeviceVariable<FP16>(scratch2.DevicePointer + scratchSizeBytes / 2);

      int inputC = input_c;
      if (num_res_blocks == 0)
      {
        Debug.Assert(inputC == kInputPlanes);
        // AttentionPreprocess(int N, CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input, CudaStream stream)
        AttentionPreprocess(N, scratch0, input, stream);
        inputC += kNumPosEncodingChannels;
      }
      else
      {
        // #redirect flow through encoder blocks
        throw new NotImplementedException();
        //convertNCHWtoNHWC(scratch0, input, N, inputC, N, inputC, 8, 8);
      }

      // 1. square embedding (fully connected layer)
      // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
      CudaDeviceVariable<FP16>  embedding = scratch1;
      int num_outputs = embedding_op_size_;
      int num_inputs = inputC;
      int batch = N * 64;

      unsafe
      {
        half halfZero = new half(0);
        half halfOne = new half(1);
        void* ptrHalfZero = &halfZero;
        void* ptrHalfOne = &halfOne;
        IntPtr ipHalfZero = (IntPtr)ptrHalfZero;
        IntPtr ipHalfOne = (IntPtr)ptrHalfOne;

        CublasStatus gemmStatus = CudaBlasNativeMethods.cublasGemmEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                         num_outputs, batch, num_inputs,
                                         ipHalfOne, ip_emb_w_.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                         scratch0.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                         ipHalfZero, embedding.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs,
                                         ComputeType.Compute16F, GemmAlgo.Default);
        Debug.Assert(gemmStatus == CublasStatus.Success);
      }

      AddBiasBatched(GetAddBiasBatchedKernel(Activation), embedding, embedding, ip_emb_b_, 1, batch, num_outputs, stream);

      // 2. Encoder layers
      foreach (LayerEncoder encoder in encoders)
      {
        encoder.Eval(stream, N, scratch1, scratch0, scratch2, scratch3, 0, null);
      }
    }

    public override void Dispose()
    {
      ip_emb_w_.Dispose();
      ip_emb_b_.Dispose();

      foreach (LayerEncoder encoder in encoders)
      {
        encoder.Dispose();
      }
    }
  }
}
