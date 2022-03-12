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
  public class AttentionPolicyHead : BaseLayerCUDA
  {
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


    public AttentionPolicyHead(NNBackendExecContext parent, string name, int layerIndex,
                               LC0LegacyWeights weights,
                               int c, int h, int w, 
                               BaseLayerCUDA inputLayer, ActivationFunction activation)
      : base(parent, name, layerIndex, c, h, w, inputLayer, activation)
    {
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
    }


    CudaKernel kernelAddVectors;
    CudaKernel kernelPromotionLogits;
    CudaKernel kernelNCHWtoNHWC;
    CudaKernel kernelAddBiasBatched;

    public override void LoadKernels()
    {
      const string kn = "_ZN6lczero13cudnn_backend17addVectors_kernelI6__halfEEvPT_S4_S4_iiiNS0_18ActivationFunctionE";
      kernelAddVectors = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, kn);

      const string knPromotion = "_ZN6lczero13cudnn_backend23promotion_logits_kernelI6__halfEEviPT_PKS3_S6_S6_";
      kernelPromotionLogits = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, knPromotion);

        const string knNCHWtoNHWC = "_ZN6lczero13cudnn_backend17NCHWtoNHWC_kernelI6__halfS2_EEvPT_PKT0_iiiiii";
      kernelNCHWtoNHWC = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, knNCHWtoNHWC);

      kernelAddBiasBatched = GetAddBiasBatchedKernel(ActivationFunction.NONE);
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
      int numElements = N * inputC * 8 * 8;
      const int BLOCK_DIM_NCHW = 256;
      kernelNCHWtoNHWC.GridDimensions = CUDAUtils.DivUp(numElements, BLOCK_DIM_NCHW);
      kernelNCHWtoNHWC.BlockDimensions = BLOCK_DIM_NCHW;
      LaunchKernel(stream, kernelNCHWtoNHWC, scratch0.DevicePointer, input.DevicePointer, N, inputC, N, inputC, 8, 8, stream.Stream.Pointer);

#if NOT
template void convertNCHWtoNHWC<half, half>(half* output_tensor,
                                            const half* input_tensor, int Nin,
                                            int Cin, int Nout, int Cout, int H,
                                            int W);
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor, int Nin, int Cin, int Nout, int Cout, int H, int W) 
{
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);

  NCHWtoNHWC_kernel<<<blocks, blockSize>>>(output_tensor, input_tensor, 
                                           Nin, Cin, Nout, Cout, H, W);
}
#endif

#if NOT
      int inputC = this->input_->GetC();
  convertNCHWtoNHWC(scratch1, input, N, inputC, N, inputC, 8, 8);

  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
#endif



#if NOT
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                            num_inputs, 1.0f, (const DataType*)ip_pol_w_,
                          num_inputs, scratch1, num_inputs, 0.0f, scratch0,
                          num_outputs);
      addVectors(scratch0, (DataType*)ip_pol_b_, scratch0, num_outputs * batch,
                 num_outputs, num_outputs * batch, SELU, stream);
#endif
    CudaDeviceVariable<FP16> pol_embedding = scratch1;
    FCLayer(stream, embeddingOpSize, N * 64, input_.C,           ip_pol_w_,  ip_pol_b_, scratch0, pol_embedding, ActivationFunction.SELU);

      //    FCLayer(stream, policy_d_model_, N * 64, embeddingOpSize,  ip2_pol_w_, ip2_pol_b_, scratch0, scratch1, ActivationFunction.NONE);
      //    FCLayer(stream, policy_d_model_, N * 64, embeddingOpSize,  ip3_pol_w_, ip3_pol_b_, scratch0, scratch2, ActivationFunction.NONE);

      CudaDeviceVariable<FP16> wq;
      CudaDeviceVariable<FP16> wk;
      unsafe
      {
        half halfZero = new half(0);
        void* ptrHalfZero = &halfZero;
        IntPtr ipHalfZero = (IntPtr)ptrHalfZero;
        half halfOne = new half(1);
        void* ptrHalfOne = &halfOne;
        IntPtr ipHalfOne = (IntPtr)ptrHalfOne;

        int num_inputs = embeddingOpSize;
        int num_outputs = policy_d_model_;
        int batch = N * 64;
        wq = scratch0;
        wk = new CudaDeviceVariable<FP16>(wq.DevicePointer + Marshal.SizeOf<FP16>() * num_outputs * batch);
#if NOT
        CudaBlasNativeMethods.cublasGemmStridedBatchedEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                                         (int)64, (int)64, policy_d_model_,
                                                         ipHalfFactor,
                                                        scratch2.DevicePointer, cudaDataType.CUDA_R_16F, policy_d_model_, 64 * policy_d_model_,
                                                         scratch1.DevicePointer, cudaDataType.CUDA_R_16F, policy_d_model_, 64 * policy_d_model_,
                                                         ipHalfZero,
                                                         output.DevicePointer, cudaDataType.CUDA_R_16F, 64, 64 * 64 + 8 * 24,
                                                         N, cudaDataType.CUDA_R_16F, GemmAlgo.Default);
#endif
        CudaBlasNativeMethods.cublasGemmStridedBatchedEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                                 num_outputs, batch, num_inputs,
                                                 ipHalfOne,
                                                 wqk_w.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs, num_inputs * num_outputs,
                                                 scratch1.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs, 0,
                                                 ipHalfZero,
                                                 wq.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs, num_outputs*batch,
                                                 2, cudaDataType.CUDA_R_16F, GemmAlgo.Default);
#if NOT
        cublasXGemmStridedBatched<DataType>(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            num_outputs, batch, num_inputs, 
            1.0f,
            wqk_w_, num_inputs, num_inputs * num_outputs, 
            scratch1, num_inputs, 0,
            0.0f, 
            wq, num_outputs, num_outputs * batch, 
            2);
#endif
        AddBiasBatched(kernelAddBiasBatched, wq, wq, wqk_b, 2, batch, num_outputs, stream);
//        addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs, NONE, stream);
      }



      float factor = 1.0f / MathF.Sqrt((float)policy_d_model_);

#if NOT
    // A/B, and M/N are swapped for row-major to col-major transform
    // leave 8*24 after each batch to interleave promotion_logits (computed
    // later below)
    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64 /*M*/, 64 /*N*/,
        policy_d_model_ /*K*/,
        factor,  // to handle "/ tf.math.sqrt(dk)"
        scratch2 /*A*/, policy_d_model_ /*LDA*/,
        64 * policy_d_model_, /*strideA*/
        scratch1 /*B*/, policy_d_model_ /*LDB*/,
        64 * policy_d_model_, /*strideB*/
        0.0f, output /*C*/,   // output (policy_attn_logits)
        64 /*LDC*/, 64 * 64 + 8 * 24 /*strideC*/, N);
#endif

#if NOT
      public static extern CublasStatus cublasGemmStridedBatchedEx(CudaBlasHandle handle, Operation transa, Operation transb, 
        int m, int n, int k, 
        CUdeviceptr alpha, CUdeviceptr A, cudaDataType Atype, int lda, long strideA, 
        CUdeviceptr B, cudaDataType Btype, int ldb, long strideB, 
        CUdeviceptr beta, 
        CUdeviceptr C, cudaDataType Ctype, int ldc, long strideC, 
        int batchCount, ComputeType computeType, GemmAlgo algo);

            unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        handle, transa, transb, 
        m, n, k, 
        &alpha_h, 
        A, CUDA_R_16F, lda, strideA,
        B, CUDA_R_16F, ldb, strideB, 
        &beta_h, 
        C, CUDA_R_16F, ldc, strideC,
        batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));

#endif
      unsafe
      {
        half halfZero = new half(0);
        half halfFactor = new half(factor);
        void* ptrHalfZero = &halfZero;
        void* ptrHalfFactor = &halfFactor;
        IntPtr ipHalfZero = (IntPtr)ptrHalfZero;
        IntPtr ipHalfFactor = (IntPtr)ptrHalfFactor;

        //        public static extern CublasStatus cublasGemmStridedBatchedEx(CudaBlasHandle handle, Operation transa, Operation transb, int m, int n, int k, CUdeviceptr alpha, CUdeviceptr A, cudaDataType Atype, int lda, long strideA, CUdeviceptr B, cudaDataType Btype, int ldb, long strideB, CUdeviceptr beta, CUdeviceptr C, cudaDataType Ctype, int ldc, long strideC, int batchCount, ComputeType computeType, GemmAlgo algo);
        //        public static extern CublasStatus cublasGemmStridedBatchedEx(CudaBlasHandle handle, Operation transa, Operation transb, int m, int n, int k, IntPtr alpha,      CUdeviceptr A, cudaDataType Atype, int lda, long strideA, CUdeviceptr B, cudaDataType Btype, int ldb, long strideB, IntPtr beta,      CUdeviceptr C, cudaDataType Ctype, int ldc, long strideC, int batchCount, cudaDataType computeType, GemmAlgo algo);

#if NOT
cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64 /*M*/, 64 /*N*/,
        policy_d_model_ /*K*/,
        factor,  // to handle "/ tf.math.sqrt(dk)"
        scratch2 /*A*/, policy_d_model_ /*LDA*/,
        64 * policy_d_model_, /*strideA*/
        scratch1 /*B*/, policy_d_model_ /*LDB*/,
        64 * policy_d_model_, /*strideB*/
        0.0f, output /*C*/,   // output (policy_attn_logits)
        64 /*LDC*/, 64 * 64 + 8 * 24 /*strideC*/, N);
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
      //      ReportCUBLASErrors(cublasGemmStridedBatchedEx(
      //          handle, transa, transb, m, n, k, &alpha_h, A, CUDA_R_16F, lda, strideA,
      //          B, CUDA_R_16F, ldb, strideB, &beta_h, C, CUDA_R_16F, ldc, strideC,
      //          batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));

#if NOT
  ComputePromotionLogits<DataType>(N, policy_d_model_, 
                                   promotion_logits, scratch2, ip4_pol_w_, output, 
                                   stream);

void ComputePromotionLogits(int N, int C, 
                            T* output, const T* keys, const T* ppo, const T* policy_attn_logits,
                            cudaStream_t stream) 
  dim3 blockDim(24, 8, 1);
  promotion_logits_kernel<T><<<N, blockDim, 0, stream>>>(C, output, keys, ppo, policy_attn_logits);
}

.entry _ZN6lczero13cudnn_backend23promotion_logits_kernelI6__halfEEviPT_PKS3_S6_S6_(
.entry _ZN6lczero13cudnn_backend23promotion_logits_kernelIfEEviPT_PKS2_S5_S5_(
#endif

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

      const int BLOCK_DIM = 256;
      kernelAddVectors.BlockDimensions = BLOCK_DIM;
      kernelAddVectors.GridDimensions = CUDAUtils.DivUp(num_outputs * batch, BLOCK_DIM);

      LaunchKernel(stream, kernelAddVectors, scratchY.DevicePointer, biases.DevicePointer, scratchY.DevicePointer,
                 num_outputs * batch,
                 num_outputs,
                 num_outputs * batch,
                 (int)activation, stream.Stream.Pointer);
    }

  }
}
