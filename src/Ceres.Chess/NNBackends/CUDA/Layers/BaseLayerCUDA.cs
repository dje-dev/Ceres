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
using System.Diagnostics;
using System.Runtime.InteropServices;

using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;

using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;

#endregion

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// The Layer objects only hold memory for weights, biases, etc
  /// memory for input and output tensors is provided by caller of Eval.
  /// </summary>
  public abstract class BaseLayerCUDA
  {
    public readonly NNBackendExecContext Parent;

    public TimeSpan LastExecutionTime;
    public float LastKernelExecutionTimeMS = 0;
    public float SumKernelExecutionTimeMS = 0;

    /// <summary>
    /// Unique descriptive name.
    /// </summary>
    public readonly string Name;

    /// <summary>
    /// Index of layer within network.
    /// </summary>
    public readonly int LayerIndex;

    /// <summary>
    /// Number of channels.
    /// </summary>
    public int C { get; }

    /// <summary>
    /// Width.
    /// </summary>
    public int W { get; }

    /// <summary>
    /// Height.
    /// </summary>
    public int GetH { get; }

    /// <summary>
    /// Input layer.
    /// </summary>
    protected readonly BaseLayerCUDA input_;

    protected const string FP16_KERNELS_PTX_NAME = @"fp16_kernels.ptx";

    public float Sum = 0;
    public float Min = float.MaxValue;
    public float Max = float.MinValue;
    public float First = float.NaN;
    public float Last = float.NaN;
    public bool IdenticalForAllPositions;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="name"></param>
    /// <param name="layerIndex"></param>
    /// <param name="c"></param>
    /// <param name="h"></param>
    /// <param name="w"></param>
    /// <param name="inputLayer"></param>
    public BaseLayerCUDA(NNBackendExecContext parent, string name, int layerIndex, int c, int h, int w, BaseLayerCUDA inputLayer)
    {
      Parent = parent;
      Name = name;
      C = c;
      W = w;
      GetH = h;
      input_ = inputLayer;
      LayerIndex = layerIndex;

      LoadKernels();
    }

    /// <summary>
    /// Number of output values for a batch of given size.
    /// </summary>
    /// <param name="N"></param>
    /// <returns></returns>
    public int GetOutputSize(int N) => N * C * GetH * W;


    /// <summary>
    /// Virtual method that initializes layer by loading any necessary kernels.
    /// </summary>
    public abstract void LoadKernels();


    protected void LaunchKernel(CudaStream stream, CudaKernel kernel, params object[] args)
    {
      //Ticks = DateTime.Now.Ticks;
      if (Parent.DumpTimings)
      {
        CUDATimingBlock tb;
        using (tb = new CUDATimingBlock("Kernel", stream, false))
        {
          kernel.RunAsync(stream.Stream, args);
          stream.Synchronize();
        }
        LastKernelExecutionTimeMS = tb.RuntimeMS;
        SumKernelExecutionTimeMS += LastKernelExecutionTimeMS;
      }
      else
      {
        kernel.RunAsync(stream.Stream, args);
      }
    }

    /// <summary>
    /// Worker method actually compute evaluate layer.
    /// </summary>
    /// <param name="N"></param>
    /// <param name="output"></param>
    /// <param name="input"></param>
    /// <param name="scratch"></param>
    /// <param name="scratchSizeBytes"></param>

    protected abstract void DoEval(CudaStream stream, int N,
                                   CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input,
                                   CudaDeviceVariable<FP16> scratch, long scratchSizeBytes,
                                   CudaDeviceVariable<FP16> scratchSecondHalf = null);

    public void Eval(CudaStream stream, int N, CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input,
                     CudaDeviceVariable<FP16> scratch, long scratchSizeBytes,
                     CudaDeviceVariable<FP16> scratchSecondHalf)
    {
      if (Parent.DumpTimings)
      {
        DateTime start = DateTime.Now;
        DoEval(stream, N, output, input, scratch, scratchSizeBytes, scratchSecondHalf);
        LastExecutionTime = DateTime.Now - start;
      }
      else
      {
        DoEval(stream, N, output, input, scratch, scratchSizeBytes, scratchSecondHalf);
      }

      if (Parent.DumpTimings)
      {
        stream.Synchronize();
        FP16[] data = new FP16[GetOutputSize(N)];
        output.CopyToHost(data, 0, 0, Marshal.SizeOf<FP16>() * data.Length);

        First = data[0];
        Last = data[GetOutputSize(1) * (N - 1)];

        int sizeEach = GetOutputSize(1);
        IdenticalForAllPositions = true;
        for (int i = 0; i < N; i++)
        {
          for (int j = 0; j < sizeEach; j++)
          {
            if (data[i * sizeEach + j] != data[j])
            {
              IdenticalForAllPositions = false;
            }
          }
        }

        Min = float.MaxValue;
        Max = float.MinValue;
        Sum = 0;

        for (int i=0; i<data.Length;i++)
        {
          float val = data[i];
          Sum += val;
          if (val > Max)
          {
            Max = val;
          }
          if (val < Min)
          {
            Min = val;
          }
        }

      }
    }


    #region Helpers
    protected static CudaDeviceVariable<FP16> LoadedWeights(float[] weights, int? checkSize = default)
    {
      Debug.Assert(!checkSize.HasValue || checkSize.Value == weights.Length);

      CudaDeviceVariable<FP16> ret = new CudaDeviceVariable<FP16>(weights.Length);
      ret.CopyToDevice(FP16.ToFP16(weights));
      return ret;
    }

    internal void cublasRowMajorMatrixMul(CudaDeviceVariable<FP16> A, /*A inut */
                                           CudaDeviceVariable<FP16> B,  /*B weights*/
                                           CudaDeviceVariable<FP16> Out,  /*Out output*/
                                           int M, int N, int K, int batchSize,
                                           bool forceStrideAZero = false)
    {
      if (false)
      {
        // For Int8:
        //   - only this combination works CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I
        //   - but it seems slower (!)
        const bool INT8 = false;
        cudaDataType typeA = INT8 ? cudaDataType.CUDA_R_8I : cudaDataType.CUDA_R_16F;
        cudaDataType typeB = INT8 ? cudaDataType.CUDA_R_8I : cudaDataType.CUDA_R_16F;
        cudaDataType typeC = INT8 ? cudaDataType.CUDA_R_32I : cudaDataType.CUDA_R_16F;
        ComputeType typeCompute = INT8 ? ComputeType.Compute32I : ComputeType.Compute16F;

        throw new Exception("Probably needs remediation to turn on PointerMode=Host below to not conflict");
#if NOT
        CUDA.BLAS.PointerMode = PointerMode.Device;
        CublasStatus errx = default;
        if (true)//batchSize != 36)
        {
          errx = CudaBlasNativeMethods.cublasGemmStridedBatchedEx
                        (CUDA.BLAS.CublasHandle, Operation.NonTranspose, Operation.NonTranspose,
                        N, M, K, // SHOULD BE N, M, K
                        CUDA.halfOneDevice.DevicePointer,
                        B.DevicePointer, typeA, N, N * K,
                        A.DevicePointer, typeB, K, forceStrideAZero ? 0 : K * M,
                        CUDA.halfZeroDevice.DevicePointer,
                        Out.DevicePointer, typeC, N, N * M,
                        batchSize, typeCompute, GemmAlgo.DefaultTensorOp);
        }
        else
        {         
          // ok
          int lda = M;
          int ldb = K;
          int ldc = M;
          for (int i = 0; i < 36; i++)
          {
            errx = CudaBlasNativeMethods.cublasGemmEx
                          (CUDA.BLAS.CublasHandle, Operation.NonTranspose, Operation.NonTranspose,
                          N, M, K, // SHOULD BE N, M, K
                          CUDA.halfOneDevice.DevicePointer,
                          B.DevicePointer, typeA, lda,//N * K,
                          A.DevicePointer, typeB, ldb,//forceStrideAZero ? 0 : K * M,
                          CUDA.halfZeroDevice.DevicePointer,
                          Out.DevicePointer, typeC, ldc,//N * M,
                          typeCompute, GemmAlgo.DefaultTensorOp); // maybe Algo13TensorOp good
          }
        }
        if (errx != CublasStatus.Success) throw new Exception($"CUDA cublasHgemmStridedBatched error: {errx}");

        CUDA.BLAS.PointerMode = PointerMode.Host;
#endif
      }
      else
      {
//        CudaEvent start = new CudaEvent();
//        CudaEvent stop = new CudaEvent();
//        start.Record();

        // About 40% of total runtime is in this call
        CublasStatus err = CudaBlasNativeMethods.cublasHgemmStridedBatched
                            (Parent.CuBlas.CublasHandle, Operation.NonTranspose, Operation.NonTranspose,
                            N, M, K, // SHOULD BE N, M, K
                            ref CUDAUtils.halfOne,
                            B.DevicePointer, N, N * K,
                            A.DevicePointer, K, forceStrideAZero ? 0 : K * M,
                            ref CUDAUtils.halfZero,
                            Out.DevicePointer, N, N * M, batchSize);


//        stop.Record();
//        stop.Synchronize();

//        float cudaElapsedTime = CudaEvent.ElapsedTime(start, stop);
//        Console.WriteLine("elapsed " + cudaElapsedTime + "ms");

        if (err != CublasStatus.Success)
        {
          throw new Exception($"CUDA cublasHgemmStridedBatched error: {err}");
        }
      }
#if NOT
      // 1 element test, success
      CublasStatus err = CudaBlasNativeMethods.cublasHgemmStridedBatched
                          (CUDA.BLAS.CublasHandle, Operation.NonTranspose, Operation.NonTranspose,
                          1,1,1,
                          ref CUDA.halfOne,
                          B.DevicePointer, 1,
                          1, A.DevicePointer, 1, 0,
                          ref CUDA.halfZero,
                          Out.DevicePointer,
                          1, 1, batchSize);

#endif
      //Possibly use this version which allows setting the algo to DefaultTensorOp?
      /*
      var err = CudaBlasNativeMethods.cublasGemmStridedBatchedEx(BlasHandle, Operation.NonTranspose, Operation.NonTranspose,
                                                         N, M,  K,
                                                         halfOne.DevicePointer, input.DevicePointer, cudaDataType.CUDA_R_16F, N,
                                                         N * K, weights.DevicePointer, cudaDataType.CUDA_R_16F, K, 0,
                                                         halfZero.DevicePointer, output.DevicePointer, cudaDataType.CUDA_R_16F,
                                                         N, N * M, batchSize,
                                                         cudaDataType.CUDA_R_16F,
                                                         GemmAlgo.Default); // ** or DefaultTensorOp?
            if (err != CublasStatus.Success) throw new Exception($"CUDA cublasHgemmStridedBatched error: {err}");

      */
#if NOT
      public static extern CublasStatus cublasGemmStridedBatchedEx(
                                                                 CUdeviceptr alpha,  /* host or device pointer */
                                                                 CUdeviceptr A, cudaDataType Atype,
                                                                 int lda, long strideA,   /* purposely signed */
                                                                 CUdeviceptr B,cudaDataType Btype,
                                                                 int ldb, long strideB,
                                                                 CUdeviceptr beta,   /* host or device pointer */
                                                                 CUdeviceptr C, cudaDataType Ctype,
                                                                 int ldc, long strideC, int batchCount,
                                                                 cudaDataType computeType, GemmAlgo algo);
#endif

      // dimensions of matrix A = M x K
      // dimensions of matrix B = K x N
      // dimensions of output   = M x N  // cublas supports only col major output to multiply row major matrices, use the trick below

    }

#endregion

    public override string ToString()
    {
      return $"{Name,12}  {LastExecutionTime.TotalMilliseconds,6:F2}  {LastKernelExecutionTimeMS,6:F2}   OutSize: {GetOutputSize(1),8:F0}  Sum: {Sum,10:F4}  First {First,10:F4} Last: {Last,10:F4}  Min: {Min,10:F4}  Max: {Max,10:F4}  {(IdenticalForAllPositions?"SAME" : "DIFF")}";
    }

  }

}
