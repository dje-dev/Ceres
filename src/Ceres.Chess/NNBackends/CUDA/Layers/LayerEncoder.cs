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
using ManagedCuda.CudaBlas;
using ManagedCuda.BasicTypes;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class LayerEncoder : BaseLayerCUDA
  {
    internal EncoderWeights encoderWeights;

    int encoder_heads;
    int embedding_op_size_;
    float alpha;

    public LayerEncoder(NNBackendExecContext parent, EncoderWeights encoderWeights,
                        string name, LC0LegacyWeights weights, int layerIndex,
                        BaseLayerCUDA ip, int heads, int size, float alpha)
       : base(parent, name, layerIndex, 0, 0, 0, ip, ActivationFunction.NONE)
    {
      encoder_heads = heads;
      embedding_op_size_ = size;
      this.alpha = alpha;

      this.encoderWeights = encoderWeights;
    }


    public override void LoadKernels()
    {
      LoadAddBiasBatchedKernels();
      LoadSoftmaxKernels();
      LoadLayerNormKernel();
      //      string kn = "_ZN6lczero13cudnn_backend16policyMap_kernelI6__halfEEvPT_PKS3_PKsiiii";
      //      kernelPolicyMap = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, kn);
    }


    public void LoadWeights(short[] cpuWeight)
    {
    }

    protected override void DoEval(CudaStream stream, int N, 
                                   CudaDeviceVariable<FP16> scratch1, CudaDeviceVariable<FP16> scratch0, 
                                   CudaDeviceVariable<FP16> scratch2,
                                   CudaDeviceVariable<FP16> scratch3, long scratch_size, 
                                   CudaDeviceVariable<FP16> scratchSecondHalf)
    {
      EncoderWeights enc = encoderWeights; // shorter alias

      int d_model = enc.mha_q_size;
      int depth = d_model / encoder_heads;

      CudaDeviceVariable<FP16> mha_q;
      CudaDeviceVariable<FP16> mha_k;
      CudaDeviceVariable<FP16> mha_v;

      unsafe
      {
        int num_inputs = embedding_op_size_;
        int num_outputs = d_model;
        int batch = N * 64;

        mha_q = scratch0;
        mha_k = new CudaDeviceVariable<FP16>(mha_q.DevicePointer + num_outputs * batch * Marshal.SizeOf<FP16>()); // mha_q + num_outputs * batch;
        mha_v = new CudaDeviceVariable<FP16>(mha_k.DevicePointer + num_outputs * batch * Marshal.SizeOf<FP16>());// mha_k + num_outputs * batch;
        half halfZero = new half(0);
        half halfOne = new half(1);
        void* ptrHalfZero = &halfZero;
        void* ptrHalfOne = &halfOne;
        IntPtr ipHalfZero = (IntPtr)ptrHalfZero;
        IntPtr ipHalfOne = (IntPtr)ptrHalfOne;

        CudaBlasNativeMethods.cublasGemmStridedBatchedEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                                         num_outputs, batch, num_inputs,
                                                         ipHalfOne, enc.mha_qkv_w.DevicePointer, cudaDataType.CUDA_R_16F,
                                                         num_inputs, num_inputs * num_outputs,
                                                         scratch1.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs, 0,
                                                         ipHalfZero, mha_q.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs, num_outputs * batch, 3,
                                                         cudaDataType.CUDA_R_16F, GemmAlgo.Default);

        AddBiasBatched(kernelAddBiasBatchedNone, mha_q, mha_q, enc.mha_qkv_b, 3, batch, num_outputs, stream);

        int encoder_dff = enc.ffn_dense1_size;

        // shape(k)[-1] = depth
        float factor = 1.0f / MathF.Sqrt((float)depth);

        // matmul_qk = tf.matmul(q, k, transpose_b=True)
        for (int i = 0; i < this.encoder_heads; i++)
        {
          unsafe
          {
            half halfFactor = new half(factor);
            void* ptrHalfFactor = &halfFactor;
            IntPtr ipHalfFactor = (IntPtr)ptrHalfFactor;

            int offset = i * depth;
            // layout of the output: encoder_heads_ * Batch * 64 * 64
            int outOffset = i * N * 64 * 64;

            CudaDeviceVariable<FP16> ptr1 = new CudaDeviceVariable<FP16>(mha_k.DevicePointer + offset * Marshal.SizeOf<FP16>());
            CudaDeviceVariable<FP16> ptr2 = new CudaDeviceVariable<FP16>(mha_q.DevicePointer + offset * Marshal.SizeOf<FP16>());
            CudaDeviceVariable<FP16> ptr3 = new CudaDeviceVariable<FP16>(scratch2.DevicePointer + outOffset * Marshal.SizeOf<FP16>());
            CudaBlasNativeMethods.cublasGemmStridedBatchedEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                                 64, 64, depth,
                                                 ipHalfFactor,
                                                 ptr1.DevicePointer, cudaDataType.CUDA_R_16F, d_model, 64 * d_model,
                                                 ptr2.DevicePointer, cudaDataType.CUDA_R_16F, d_model, 64 * d_model,
                                                 ipHalfZero,
                                                 ptr3.DevicePointer, cudaDataType.CUDA_R_16F, 64, 64 * 64, N,
                                                 cudaDataType.CUDA_R_16F, GemmAlgo.Default);
          }
        }


        // attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
        // attention_weights -> scratch2
        Softmax(encoder_heads * N * 64, 64, scratch2, scratch2, stream);

        // output = tf.matmul(attention_weights, v)
        for (int i = 0; i < encoder_heads; i++)
        {
          int offset = i * depth;  // for output and "v" matrix
                                   // layout: encoder_heads_ * Batch*64*64
          int weightsOffset = i * N * 64 * 64;
          CudaDeviceVariable<FP16> ptr1 = new CudaDeviceVariable<FP16>(mha_v.DevicePointer + offset * Marshal.SizeOf<FP16>());
          CudaDeviceVariable<FP16> ptr2 = new CudaDeviceVariable<FP16>(scratch2.DevicePointer + weightsOffset * Marshal.SizeOf<FP16>());
          CudaDeviceVariable<FP16> ptr3 = new CudaDeviceVariable<FP16>(scratch3.DevicePointer + offset * Marshal.SizeOf<FP16>());
          CudaBlasNativeMethods.cublasGemmStridedBatchedEx(Parent.CuBlas.CublasHandle, Operation.NonTranspose, Operation.NonTranspose,
                                               depth, 64, 64,
                                               ipHalfOne,
                                               ptr1.DevicePointer, cudaDataType.CUDA_R_16F, d_model, 64 * d_model,
                                               ptr2.DevicePointer, cudaDataType.CUDA_R_16F, 64, 64 * 64,
                                               ipHalfZero,
                                               ptr3.DevicePointer, cudaDataType.CUDA_R_16F, d_model, 64 * d_model,
                                               N, cudaDataType.CUDA_R_16F, GemmAlgo.Default);
        }

        // #final dense layer (mha_dense), scratch3 -> scratch2
        num_inputs = d_model;
        num_outputs = embedding_op_size_;
        batch = N * 64;
        CudaBlasNativeMethods.cublasGemmEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                           num_outputs, batch, num_inputs,
                                           ipHalfOne, enc.mha_dense_w.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                           scratch3.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                           ipHalfZero, scratch2.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs,
                                           ComputeType.Compute16F, GemmAlgo.Default);

        // LN1: skip connection and layer normalization (also bias add of prev gemm)
        // scratch2/scratch1 -> scratch0
        LayerNorm(N * 64, embedding_op_size_, scratch0, scratch2,
                        enc.mha_dense_b, scratch1, enc.ln1_gammas,
                        enc.ln1_betas, 1e-6f, alpha, stream);

        // #FFN dense 1, scratch0 -> scratch1
        num_inputs = embedding_op_size_;
        num_outputs = encoder_dff;
        batch = N * 64;
        CudaBlasNativeMethods.cublasGemmEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                           num_outputs, batch, num_inputs,
                                           ipHalfOne,
                                           enc.ffn_dense1_w.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                           scratch0.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                           ipHalfZero,
                                           scratch1.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs,
                                           ComputeType.Compute16F, GemmAlgo.Default);

        AddBiasBatched(kernelAddBiasBatchedSELU, scratch1, scratch1, enc.ffn_dense1_b, 1, batch, num_outputs, stream);

        // #FFN dense 2, scratch1 -> scratch2
        num_inputs = encoder_dff;
        num_outputs = embedding_op_size_;
        batch = N * 64;
        CudaBlasNativeMethods.cublasGemmEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                           num_outputs, batch, num_inputs,
                                           ipHalfOne,
                                           enc.ffn_dense2_w.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                           scratch1.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                           ipHalfZero,
                                           scratch2.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs,
                                           ComputeType.Compute16F, GemmAlgo.Default);

        LayerNorm(N * 64, embedding_op_size_, scratch1, scratch2,
                    enc.ffn_dense2_b, scratch0, enc.ln2_gammas,
                    enc.ln2_betas, 1e-6f, alpha, stream);
      }
    }
  }
}
