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
using ManagedCuda.BasicTypes;

using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class ResidualBlockFusedCUDA : ResidualBlockBaseCUDA
  {
    // The launcher preallocates and pins the parameters to pass
    // reducing memory allocations (considerably) and 
    // improving speed (slightly).
    public const bool USE_LAUNCHER = true;

    bool IsBig => C > 384;

    // PERFORMANCE ANALYSIS
    // 65% of the runtime is in the two matmuls, 35% in the two kernel calls
    // Within the kernel, it is the SE calculation part that is the most expensive
    const string knFirstBlockInput = "_ZN6lczero13cudnn_backend21InputTransform_kernelI6__halfLb1EEEviiPKT_PS3_";

    string knInputOutputPre => IsBig ? "_ZN6lczero13cudnn_backend43OutputInputTransformKernel_fp16_shmem_boardILb0ELb1ELb1ELb0EEEviiiP6__halfPKS2_S3_S5_S5_S5_S5_S5_"
                                       : "_ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb0ELb1ELb1ELb0EEEviiiPT_PKS3_S4_S6_S6_S6_S6_S6_";
    string knNotLastSE => IsBig ? "_ZN6lczero13cudnn_backend43OutputInputTransformKernel_fp16_shmem_boardILb1ELb1ELb1ELb1EEEviiiP6__halfPKS2_S3_S5_S5_S5_S5_S5_"
                                : "_ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb1ELb1ELb1ELb1EEEviiiPT_PKS3_S4_S6_S6_S6_S6_S6_";
    string knNotLastNotSE => IsBig ? "_ZN6lczero13cudnn_backend43OutputInputTransformKernel_fp16_shmem_boardILb0ELb1ELb1ELb1EEEviiiP6__halfPKS2_S3_S5_S5_S5_S5_S5_"
                                    : "_ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb0ELb1ELb1ELb1EEEviiiPT_PKS3_S4_S6_S6_S6_S6_S6_";

    const string knLastSE = "_ZN6lczero13cudnn_backend22OutputTransform_kernelI6__halfLb1ELb1ELb1ELb1ELb1ELb0EEEviiiPT_PKS3_S6_S6_S6_S6_S6_S6_";
    const string knLastNotSE = "_ZN6lczero13cudnn_backend22OutputTransform_kernelI6__halfLb0ELb1ELb1ELb1ELb1ELb0EEEviiiPT_PKS3_S6_S6_S6_S6_S6_S6_";

    const string knNoSENotLast = "_ZN6lczero13cudnn_backend42OutputTransform_relu_InputTransform_kernelI6__halfLb1ELb1ELb0EEEviiPT_PKS3_S4_S6_";

    string Activated(string baseName) => baseName.Replace("<ACTIVATION>", Activation == ActivationFunction.MISH ? "MISH" : "RELU");

    int NUM_SHARED_BYTES => IsBig ? (72 * 1024) : 0;  // SharedMemSize

    public override void LoadKernels()
    {
      CudaKernel kernelNotLastSE = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, knNotLastSE);
      CudaKernel kernelNotLastNotSE = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, knNotLastNotSE);
      inputTransformKernel = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, knFirstBlockInput);
      inputOutputKernelPre = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, Activated(knInputOutputPre));

      inputOutputKernelPre.BlockDimensions = C;

      kernelSENotLast = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, Activated(knNoSENotLast));

      if (IsBig)
      {
        inputOutputKernelPre.MaxDynamicSharedSizeBytes = NUM_SHARED_BYTES;
        kernelNotLastSE.MaxDynamicSharedSizeBytes = NUM_SHARED_BYTES;
        kernelNotLastNotSE.MaxDynamicSharedSizeBytes = NUM_SHARED_BYTES;
      }

      Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, knFirstBlockInput);
      Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, knLastSE);
      Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, knLastNotSE);
    }


    public readonly bool HasSE;
    public readonly int SEK;
    public readonly bool FirstBlock;
    public readonly bool LastBlock;

    CudaDeviceVariable<FP16> biases0;
    CudaDeviceVariable<FP16> biases1;
    CudaDeviceVariable<FP16> transformedWeights0;
    CudaDeviceVariable<FP16> transformedWeights1;

    CudaKernel inputTransformKernel; // does not depend on activation


    CudaKernel kernelLastSE;
    CudaKernel kernelLastNotSE;
    CudaKernel kernelNotLastSE;
    CudaKernel kernelNotLastNotSE;

    CudaKernel inputOutputKernelPre;
    CudaKernel kernelSENotLast;


    public ResidualBlockFusedCUDA(NNBackendExecContext parent, string name, int layerIndex,
                                  BaseLayerCUDA inputLayer,
                                  int C, bool se, int se_k, bool first, bool last, int sharedMemSize, ActivationFunction activation)
      : base(parent, name, layerIndex, C, 8, 8, inputLayer, se, se_k, sharedMemSize, activation)
    {
      if (C > 512)
      {
        // This limit is definitive since data structures are sized to 512 max (see cuda_common.h).
        throw new Exception("Maximum number of channels supported is 512");
      }

      HasSE = se;
      SEK = se_k;
      FirstBlock = first;
      LastBlock = last;

      kernelLastSE = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, Activated(knLastSE));
      kernelLastNotSE = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, Activated(knLastNotSE));
      kernelNotLastSE = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, Activated(knNotLastSE));
      kernelNotLastNotSE = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, Activated(knNotLastNotSE));

      if (NNBackendLC0_CUDA.BLASLT && NNBackendLC0_CUDA.BLASLT_N > 0)
      {
#if FEATURE_BLASLT
        batchMultiplier = new (Parent.Device.Context, Parent.CuBlasLT, 
                              NNBackendLC0_CUDA.BLASLT_USE_LOOP, base.C,
                              NNBackendLC0_CUDA.BLASLT_N * 4, base.C, 36);
#else
        throw new NotImplementedException();
#endif
      }
    }


    public override void LoadWeights0(CudaStream stream, float[] weights, float[] bias)
    {
      if (Parent.ReferenceLayers != null)
      {
        ResidualBlockFusedCUDA refLayer = Parent.ReferenceLayers.Layers[LayerIndex] as ResidualBlockFusedCUDA;
        biases0 = refLayer.biases0;
        transformedWeights0 = refLayer.transformedWeights0;
      }
      else
      {
        biases0 = CudaHalf(bias);
        transformedWeights0 = ToFiltered(stream, weights, C, C);
      }
    }

    public override void LoadWeights1(CudaStream stream, float[] weights, float[] bias)
    {
      if (Parent.ReferenceLayers != null)
      {
        ResidualBlockFusedCUDA refLayer = Parent.ReferenceLayers.Layers[LayerIndex] as ResidualBlockFusedCUDA;
        biases1 = refLayer.biases1;
        transformedWeights1 = refLayer.transformedWeights1;
      }
      else
      {
        biases1 = CudaHalf(bias);
        transformedWeights1 = ToFiltered(stream, weights, C, C);
      }
    }


#if FEATURE_BLASLT
    GEMMStridedBatched batchMultiplier = null;
#endif

    CUDAKernelLauncher inputOutputLauncher;
    CUDAKernelLauncher inputOutputNoSELauncher;
    CUDAKernelLauncher outputLauncher;


    protected override void DoEval(CudaStream stream, int N,
                                   CudaDeviceVariable<FP16> output,
                                   CudaDeviceVariable<FP16> input,
                                   CudaDeviceVariable<FP16> scratch,
                                   long scratch_size,
                                   CudaDeviceVariable<FP16> scratchSecondHalf)
    {
      // normally:
      // - "output" initially contains the transformed input, 
      //    and after this layer, it contains the transformed input for next layer
      // - "input" contains the original/untransformed input
      // special cases:
      //   - for first_block_, input is real input (untransformed)
      //   - for last_block_, output is the final output of this block (untransformed)

      // Split the scratch space into two parts - use first part for holding
      // transformed input and second part for transformed output.

      CudaDeviceVariable<FP16> transformed_input = scratch;
      CudaDeviceVariable<FP16> transformed_output = scratchSecondHalf;

      #region Preparatory

      if (FirstBlock)
      {
        // Possibly duplicate code as in FusedWinogradSELayerCUDA
        inputTransformKernel.GridDimensions = N;
        inputTransformKernel.BlockDimensions = C;
        // InputTransform<DataType>(N, c_input_, transformed_input, input);
        LaunchKernel(stream, inputTransformKernel, N, C, input.DevicePointer, transformed_input.DevicePointer, stream.Stream.Pointer);

        cublasRowMajorMatrixMul(transformed_input, transformedWeights0, transformed_output, N * 4, C, C, 36);
      }
      else
      {
        if (true)
        {
#if FEATURE_BLASLT
        if (batchMultiplier != null)
            {
              if (N != NNBackendLC0_CUDA.BLASLT_N) throw new Exception("Wrong N");
              batchMultiplier.Execute(stream, transformedWeights0, output, transformed_output, false);
            }
            else
            {
#else
          cublasRowMajorMatrixMul(output, transformedWeights0, transformed_output, N * 4, C, C, 36);
#endif
        }
        else
        {
          throw new NotImplementedException(); // see code at bottom of file
        }
      }

      #endregion

      #region InputOutput

      const bool USE_OPTIMIZED_NON_SE_IO_KERNEL = true;
      if (USE_OPTIMIZED_NON_SE_IO_KERNEL && !LastBlock)
      {
        const int kOpInpTransformBlockSize = 64;
        kernelSENotLast.BlockDimensions = kOpInpTransformBlockSize;
        kernelSENotLast.GridDimensions = new ManagedCuda.VectorTypes.dim3(CUDAUtils.DivUp(C, kOpInpTransformBlockSize), N, 1);

        if (USE_LAUNCHER && inputOutputNoSELauncher == null)
        {
          inputOutputNoSELauncher = new(kernelSENotLast, stream, 0,
                                    new object[] {N, C,
                                                  transformed_input.DevicePointer, transformed_output.DevicePointer,
                                                  (IntPtr)0, biases0.DevicePointer });
        }

        if (inputOutputNoSELauncher != null)
        {
          inputOutputNoSELauncher.Parms.ObjRef<int>(0) = N;
          inputOutputNoSELauncher.LaunchAsync();
        }
        else
        {
          LaunchKernel(stream, kernelSENotLast, N, C,
                       transformed_input.DevicePointer, transformed_output.DevicePointer,
                       (IntPtr)0, biases0.DevicePointer);
        }
      }
      else
      {
        inputOutputKernelPre.GridDimensions = N;
        inputOutputKernelPre.BlockDimensions = C;

        // with relu, use_bias (not use_se, not skip)
        if (USE_LAUNCHER && inputOutputLauncher == null)
        {
          inputOutputLauncher = new(inputOutputKernelPre, stream, NUM_SHARED_BYTES,
                                    new object[] {N, C, 0,
                                  transformed_input.DevicePointer,
                                  transformed_output.DevicePointer,
                                  (IntPtr)0, biases0.DevicePointer,
                                  (IntPtr)0, (IntPtr)0,
                                  (IntPtr)0, (IntPtr)0, stream.Stream.Pointer });
        }

        if (inputOutputLauncher != null)
        {
          inputOutputLauncher.Parms.ObjRef<int>(0) = N;
          inputOutputLauncher.LaunchAsync();
        }
        else
        {
          LaunchKernel(stream, inputOutputKernelPre, N, C, 0,
                      transformed_input.DevicePointer, transformed_output.DevicePointer,
                      (IntPtr)0, biases0.DevicePointer,
                      (IntPtr)0, (IntPtr)0,
                      (IntPtr)0, (IntPtr)0, stream.Stream.Pointer);
        }
      }

      #endregion

      // "transformed_input" tensor now contains transformed input for the next convolution

#if FEATURE_BLASLT
      if (batchMultiplier != null)
      {
        if (N != NNBackendLC0_CUDA.BLASLT_N) throw new Exception("wrong N");

        batchMultiplier.Execute(stream, transformedWeights1, transformed_input, transformed_output, false);
      }
      else
      {
#endif
      cublasRowMajorMatrixMul(transformed_input, transformedWeights1, transformed_output, N * 4, C, C, 36);

      // Verify support for this combination of network/hardware.
      if (LastBlock && HasSE)
      {
        const int kMaxResBlockFusingChannels = 384;  // limit on num_filters
        const int kMaxResBlockFusingSeKFp16Ampere = 512;  // (use a different kernel with reduced register pressure)
        const int kMaxResBlockFusingSeFp16AmpereSmem = 72 * 1024;  // shared memory used by the special kernel

        bool allowFusing = (C <= kMaxResBlockFusingChannels)
                        || ((SharedMemSize >= kMaxResBlockFusingSeFp16AmpereSmem) &&
                            (C <= kMaxResBlockFusingSeKFp16Ampere));
        if (!allowFusing)
        {
          // TODO: support this case (for 512b networks on hardware without sufficient shared memory.
          throw new Exception("Ceres limitation: network not supported on this hardware, see possible remediation in source.");
#if NOT
// Need to add translation of this code to support this case
OutputTransform<DataType, true, true, true, true, true, true>(
            N, C, se_k_, (DataType*)input, transformed_output, input,
            biases1_, w1_, b1_, w2_, b2_, stream);
        InputTransform<DataType, true>(N, C, output, (DataType*)input, stream);
#endif
        }
      }


      CudaKernel kernelOutputInput;
      if (LastBlock)
      {
        kernelOutputInput = HasSE ? kernelLastSE : kernelLastNotSE;
      }
      else
      {
        kernelOutputInput = HasSE ? kernelNotLastSE : kernelNotLastNotSE;
      }

      int numSharedBytes = 0;
      if (!LastBlock)
      {
        numSharedBytes = NUM_SHARED_BYTES;
      }

      kernelOutputInput.GridDimensions = N;
      kernelOutputInput.BlockDimensions = C;

      CUdeviceptr DUMMY = default;
      if (!base.HasSE)
      {
        // Although not actually used in non-SE mode,
        // we have to pass some valid address to the kernel to prevent error.
        DUMMY = scratch.DevicePointer;
      }

      if (USE_LAUNCHER && outputLauncher == null)
      {
        outputLauncher = new(kernelOutputInput, stream, numSharedBytes,
                             new object[] {N, C, SEK,
                                           output.DevicePointer, transformed_output.DevicePointer,
                                           input.DevicePointer, biases1.DevicePointer,
                                           HasSE ? Weights1.DevicePointer : DUMMY, HasSE ? Biases1.DevicePointer : DUMMY,
                                           HasSE ? Weights2.DevicePointer : DUMMY, HasSE ? Biases2.DevicePointer : DUMMY,
                             stream.Stream.Pointer}
                             );
      }

      if (outputLauncher != null)
      {
        outputLauncher.Parms.ObjRef<int>(0) = N;
        outputLauncher.LaunchAsync();
      }
      else
      {
        throw new NotImplementedException("No longer support USE_LAUNCHER=false, setting of shared memory must be remediated.");

        // This kernel is the slowest part of this block
        // NOTE: the PTX may look like it fails to take advantage of FMA operations,
        //       but in fact a postprocessing step does this, explicit FMA was no faster.
        //      using (new TimingBlockCUDA("residual block kernelOutputInput", stream))
        LaunchKernel(stream, kernelOutputInput, N, C, SEK,
                            output.DevicePointer, transformed_output.DevicePointer,
                            input.DevicePointer, biases1.DevicePointer,
                            HasSE ? Weights1.DevicePointer : DUMMY, HasSE ? Biases1.DevicePointer : DUMMY,
                            HasSE ? Weights2.DevicePointer : DUMMY, HasSE ? Biases2.DevicePointer : DUMMY, stream.Stream.Pointer);
      }

      // "output" tensor now contains transformed input for the next
      // convolution
    }

  }
}


#if NOT
          using (new TimingBlock("10_000"))
          {
            for (int i = 0; i < 1000_000; i++)
            {
              cublasRowMajorMatrixMul(output, transformed_weights0_, transformed_output, N * 4, C, C, 36);

              FP16[] correct = new FP16[384 * 1024 * 36];
              transformed_output.CopyToHost(correct);

              batchMultiplier.Execute(stream, transformed_weights0_, output, transformed_output, false);

              FP16[] attempt = new FP16[384 * 1024 * 36];
              transformed_output.CopyToHost(attempt);
              for (int ix = 0; ix < attempt.Length; ix++)
              {
                if (Math.Abs(correct[ix] - attempt[ix]) > 1E-5)
                {
                  Console.WriteLine("bad ");
                }
              }
              Console.WriteLine("ok");
            }
          }
#endif
