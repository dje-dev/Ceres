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

    // PERFORMANCE ANALYSIS
    // 65% of the runtime is in the two matmuls, 35% in the two kernel calls
    // Within the kernel, it is the SE calculation part that is the most expensive
    string resource => (C > 384) ? @"fp16_kernels_big.ptx" :  @"fp16_kernels.ptx";

    string knFirstBlockInput = "_ZN6lczero13cudnn_backend21InputTransform_kernelI6__halfLb1EEEviiPKT_PS3_";
    string knInputOutputPre = "_ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb0ELb1ELb1ELb0EEEviiiPT_PKS3_S4_S6_S6_S6_S6_S6_";

    string knLastSE       = "_ZN6lczero13cudnn_backend22OutputTransform_kernelI6__halfLb1ELb1ELb1ELb1ELb1ELb0EEEviiiPT_PKS3_S6_S6_S6_S6_S6_S6_";
    string knLastNotSE    = "_ZN6lczero13cudnn_backend22OutputTransform_kernelI6__halfLb0ELb1ELb1ELb1ELb1ELb0EEEviiiPT_PKS3_S6_S6_S6_S6_S6_S6_";
    string knNotLastSE    = "_ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb1ELb1ELb1ELb1EEEviiiPT_PKS3_S4_S6_S6_S6_S6_S6_";
    string knNotLastNotSE = "_ZN6lczero13cudnn_backend45OutputTransform_SE_relu_InputTransform_kernelI6__halfLb0ELb1ELb1ELb1EEEviiiPT_PKS3_S4_S6_S6_S6_S6_S6_";
    
    public override void LoadKernels()
    {
      Parent.Device.GetKernel(Parent.PTXAssembly, resource, knFirstBlockInput);
      Parent.Device.GetKernel(Parent.PTXAssembly, resource, knInputOutputPre);

      Parent.Device.GetKernel(Parent.PTXAssembly, resource, knLastSE);
      Parent.Device.GetKernel(Parent.PTXAssembly, resource, knLastNotSE);
      Parent.Device.GetKernel(Parent.PTXAssembly, resource, knNotLastSE);
      Parent.Device.GetKernel(Parent.PTXAssembly, resource, knNotLastNotSE);
    }


    public readonly bool HasSE;
    public readonly int SEK;
    public readonly bool FirstBlock;
    public readonly bool LastBlock;

    CudaDeviceVariable<FP16> biases0;
    CudaDeviceVariable<FP16> biases1;
    CudaDeviceVariable<FP16> transformedWeights0;
    CudaDeviceVariable<FP16> transformedWeights1;
  

  public ResidualBlockFusedCUDA(NNBackendExecContext parent, string name, int layerIndex, 
                                BaseLayerCUDA inputLayer, 
                                int C, bool se, int se_k, bool first, bool last)
      : base(parent, name, layerIndex, C, 8, 8, inputLayer, se, se_k)
    {
      HasSE = se;
      SEK = se_k;
      FirstBlock = first;
      LastBlock = last;

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

      if (FirstBlock)
      {
        // Possibly duplicate code as in FusedWinogradSELayerCUDA
        CudaKernel inputTransformKernel = Parent.Device.GetKernel(Parent.PTXAssembly, resource, knFirstBlockInput);

        inputTransformKernel.GridDimensions = N;
        inputTransformKernel.BlockDimensions = C;
        // InputTransform<DataType>(N, c_input_, transformed_input, input);
        LaunchKernel(stream, inputTransformKernel, N, C, input.DevicePointer, transformed_input.DevicePointer);

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
          throw new NotImplementedException();
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
        }
      }

      // with reslu, use_bias (not use_se, not skip)
      CudaKernel inputOutputKernelPre = Parent.Device.GetKernel(Parent.PTXAssembly, resource, knInputOutputPre);
      inputOutputKernelPre.GridDimensions = N;
      inputOutputKernelPre.BlockDimensions = C;

      if (USE_LAUNCHER && inputOutputLauncher == null)
      {
        inputOutputLauncher = new(inputOutputKernelPre, stream,
                                  new object[] {N, C, 0,
                                  transformed_input.DevicePointer, 
                                  transformed_output.DevicePointer,
                                  (IntPtr)0, biases0.DevicePointer,
                                  (IntPtr)0, (IntPtr)0,
                                  (IntPtr)0, (IntPtr)0 });
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
                    (IntPtr)0, (IntPtr)0);
      }

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
      

      string outputKN;
      if (LastBlock)
      {
        outputKN = HasSE ? knLastSE : knLastNotSE;
      }
      else
      {
        outputKN = HasSE ? knNotLastSE : knNotLastNotSE;
      }


      CudaKernel kernelOutputInput = Parent.Device.GetKernel(Parent.PTXAssembly, resource, outputKN);
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
        outputLauncher = new(kernelOutputInput, stream,
                             new object[] {N, C, SEK,
                                           output.DevicePointer, transformed_output.DevicePointer,
                                           input.DevicePointer, biases1.DevicePointer,
                                           HasSE ? Weights1.DevicePointer : DUMMY, HasSE ? Biases1.DevicePointer : DUMMY,
                                           HasSE ? Weights2.DevicePointer : DUMMY, HasSE ? Biases2.DevicePointer : DUMMY});
      }

      if (outputLauncher != null)
      {
        outputLauncher.Parms.ObjRef<int>(0) = N;
        outputLauncher.LaunchAsync();
      }
      else
      {
        // This kernel is the slowest part of this block
        // NOTE: the PTX may look like it fails to take advantage of FMA operations,
        //       but in fact a postprocessing step does this, explicit FMA was no faster.
        //      using (new TimingBlockCUDA("residual block kernelOutputInput", stream))
        LaunchKernel(stream, kernelOutputInput, N, C, SEK,
                            output.DevicePointer, transformed_output.DevicePointer,
                            input.DevicePointer, biases1.DevicePointer,
                            HasSE ? Weights1.DevicePointer : DUMMY, HasSE ? Biases1.DevicePointer : DUMMY,
                            HasSE ? Weights2.DevicePointer : DUMMY, HasSE ? Biases2.DevicePointer : DUMMY);
      }

      // "output" tensor now contains transformed input for the next
      // convolution
    }
    
  }
}
