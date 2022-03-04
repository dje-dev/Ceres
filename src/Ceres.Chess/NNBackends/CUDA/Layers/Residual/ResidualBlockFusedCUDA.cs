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
    const string knInput = "_ZN6lczero13cudnn_backend21InputTransform_kernelI6__halfLb1EEEviiPKT_PS3_";

    static string BoolStr(bool b) => b ? "1" : "0";

    static string GetOutputTransformKernelName(bool useSE, ActivationFunction activation, bool useBias, bool useSkip, bool skipInputNHCW, bool outputNHCW)
    {
      //template <typename T, bool use_se, bool relu, bool use_bias, bool use_skip, bool skipInput_nhcw, bool output_nhcw>
      //fp16_kernels.ptx:.visible .entry _ZN6lczero13cudnn_backend22OutputTransform_kernelI6__halfLb1ELNS0_18ActivationFunctionE1ELb1ELb1ELb0ELb0EEEviiiPT_PKS4_S7_S7_S7_S7_S7_S7_

      const string CALL = "_ZN6lczero13cudnn_backend22OutputTransform_kernelI6__halfLb!1ELNS0_18ActivationFunctionE!2ELb!3ELb!4ELb!5ELb!6EEEviiiPT_PKS4_S7_S7_S7_S7_S7_S7_";
      return CALL.Replace("!1", BoolStr(useSE))
                 .Replace("!2", ((int)activation).ToString())
                 .Replace("!3", BoolStr(useBias))
                 .Replace("!4", BoolStr(useSkip))
                 .Replace("!5", BoolStr(skipInputNHCW))
                 .Replace("!6", BoolStr(outputNHCW));
    }

    string GetOutputInputTransformKernelName(bool useSE, ActivationFunction activation, bool useBias, bool useSkip)
    {
      if(IsBig)
      {
        //template <bool use_se, ActivationFunction activation, bool use_bias, bool use_skip>
        //fp16_kernels.ptx:.visible .entry _ZN6lczero13cudnn_backend43OutputInputTransformKernel_fp16_shmem_boardILb1ELNS0_18ActivationFunctionE1ELb1ELb1EEEviiiP6__halfPKS3_S4_S6_S6_S6_S6_S6_(

        const string CALL = "_ZN6lczero13cudnn_backend43OutputInputTransformKernel_fp16_shmem_boardILb!1ELNS0_18ActivationFunctionE!2ELb!3ELb!4EEEviiiP6__halfPKS3_S4_S6_S6_S6_S6_S6_";
        return CALL.Replace("!1", BoolStr(useSE))
                   .Replace("!2", ((int)activation).ToString())
                   .Replace("!3", BoolStr(useBias))
                   .Replace("!4", BoolStr(useSkip));
      }
      else
      {
        //fp16_kernels.ptx:.visible .entry _ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb1ELNS0_18ActivationFunctionE1ELb1ELb1EEEviiiPT_PKS4_S5_S7_S7_S7_S7_S7_

        const string CALL = "_ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb!1ELNS0_18ActivationFunctionE!2ELb!3ELb!4EEEviiiPT_PKS4_S5_S7_S7_S7_S7_S7_";
        return CALL.Replace("!1", BoolStr(useSE))
                   .Replace("!2", ((int)activation).ToString())
                   .Replace("!3", BoolStr(useBias))
                   .Replace("!4", BoolStr(useSkip));
      }
    }


    int NUM_SHARED_BYTES => IsBig ? (72 * 1024) : 0;  // SharedMemSize

    public override void LoadKernels()
    {
      inputTransformKernel = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, knInput);
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


    CudaKernel kernelOutputInputPre;

    CudaKernel kernelOutput;
    CudaKernel kernelOutputInput;

    CUDAKernelLauncher preLauncher;
    CUDAKernelLauncher postLauncher;

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

      kernelOutputInputPre = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, GetOutputInputTransformKernelName(false, Activation, true, false));

      kernelOutputInput = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, GetOutputInputTransformKernelName(HasSE, Activation, true, true));
      kernelOutput = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, GetOutputTransformKernelName(HasSE, Activation, true, true, true, false));

      kernelOutputInputPre.BlockDimensions = C;
      if (IsBig)
      {
        kernelOutputInputPre.MaxDynamicSharedSizeBytes = NUM_SHARED_BYTES;
      }

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

#if NOT
  if (act_ == RELU) 
  {
    OutputInputTransform<DataType, false, RELU, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  }
  else if (act_ == MISH) 
  {
    OutputInputTransform<DataType, false, MISH, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  }
#endif
        kernelOutputInputPre.GridDimensions = N;
        kernelOutputInputPre.BlockDimensions = C;

        // with relu, use_bias (not use_se, not skip)
        if (USE_LAUNCHER && preLauncher == null)
        {
          preLauncher = new(kernelOutputInputPre, stream, NUM_SHARED_BYTES,
                                    new object[] {N, C, 0,
                                  transformed_input.DevicePointer,
                                  transformed_output.DevicePointer,
                                  (IntPtr)0, biases0.DevicePointer,
                                  (IntPtr)0, (IntPtr)0,
                                  (IntPtr)0, (IntPtr)0, stream.Stream.Pointer });
        }

        if (preLauncher != null)
        {
          preLauncher.Parms.ObjRef<int>(0) = N;
          preLauncher.LaunchAsync();
        }
        else
        {
          LaunchKernel(stream, kernelOutputInputPre, N, C, 0,
                      transformed_input.DevicePointer, transformed_output.DevicePointer,
                      (IntPtr)0, biases0.DevicePointer,
                      (IntPtr)0, (IntPtr)0,
                      (IntPtr)0, (IntPtr)0, stream.Stream.Pointer);
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


      CudaKernel kernelPost = LastBlock ? kernelOutput : kernelOutputInput;

      int numSharedBytes = 0;
      if (!LastBlock)
      {
        numSharedBytes = NUM_SHARED_BYTES;
      }

      kernelPost.GridDimensions = N;
      kernelPost.BlockDimensions = C;

      CUdeviceptr DUMMY = default;
      if (!base.HasSE)
      {
        // Although not actually used in non-SE mode,
        // we have to pass some valid address to the kernel to prevent error.
        DUMMY = scratch.DevicePointer;
      }

      if (USE_LAUNCHER && postLauncher == null)
      {
        postLauncher = new(kernelPost, stream, numSharedBytes,
                             new object[] {N, C, SEK,
                                           output.DevicePointer, transformed_output.DevicePointer,
                                           input.DevicePointer, biases1.DevicePointer,
                                           HasSE ? Weights1.DevicePointer : DUMMY, HasSE ? Biases1.DevicePointer : DUMMY,
                                           HasSE ? Weights2.DevicePointer : DUMMY, HasSE ? Biases2.DevicePointer : DUMMY,
                             stream.Stream.Pointer}
                             );
      }

      if (postLauncher != null)
      {
        postLauncher.Parms.ObjRef<int>(0) = N;
        postLauncher.LaunchAsync();
      }
      else
      {
        throw new NotImplementedException("No longer support USE_LAUNCHER=false, setting of shared memory must be remediated.");

        // This kernel is the slowest part of this block
        // NOTE: the PTX may look like it fails to take advantage of FMA operations,
        //       but in fact a postprocessing step does this, explicit FMA was no faster.
        //      using (new TimingBlockCUDA("residual block kernelOutputInput", stream))
        LaunchKernel(stream, kernelPost, N, C, SEK,
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


