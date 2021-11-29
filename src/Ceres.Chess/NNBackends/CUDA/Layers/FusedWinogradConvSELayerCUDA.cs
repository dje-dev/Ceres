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
using ManagedCuda;
using Ceres.Base.DataTypes;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class FusedWinogradConvSELayerCUDA : BlockWithWinogradCUDA
  {
    CudaDeviceVariable<FP16> biases;
    CudaDeviceVariable<FP16> transformedWeights;  // After winograd transform.

    public readonly int NumInputChannels;
    public readonly bool UseRELU;
    public readonly bool UseBias;
    public readonly bool WithSkipConnection;
    public readonly bool UseSE;
    public readonly int SEChannelCount;
    public readonly bool OpNCHW;

    public FusedWinogradConvSELayerCUDA(NNBackendExecContext parent, string name, int layerIndex, 
                                        int c, int h, int w, 
                                        BaseLayerCUDA inputLayer,
                                        int Cin, bool relu, bool bias, bool skip_add, bool se, int se_k,
                                        bool opNHCW)
      : base(parent, name, layerIndex, c, h, w, inputLayer)
    {
      if (se & !skip_add)
      {
        throw new NotImplementedException();
      }

      NumInputChannels = Cin;
      UseRELU = relu;
      UseBias = bias;
      WithSkipConnection = skip_add;
      UseSE = se;
      SEChannelCount = se_k;
      OpNCHW = opNHCW;
    }

    public void LoadWeights(CudaStream stream, float[] weights, float[] biases)
    {
      Debug.Assert(biases.Length == C);
      Debug.Assert(weights.Length == NumInputChannels * C * 3 * 3);
      if (Parent.ReferenceLayers != null)
      {
        FusedWinogradConvSELayerCUDA refLayer = Parent.ReferenceLayers.Layers[LayerIndex] as FusedWinogradConvSELayerCUDA;
        this.biases = refLayer.biases;
        this.transformedWeights = refLayer.transformedWeights;
      }
      else
      {
        this.biases = FP16.ToFP16(biases);

#if NOT
      float[] transformedWeights = new float[4 * c_input_ * GetC * 3 * 3];
      // Attempt at using C# version, but it doesn't return correct answers
      // probably because the CPU (C#) version transposes output (see comment in WinogradFilterTransformF)
      using (new TimingBlock("WinogradFilterTransformF"))
        WinogradFilter.WinogradFilterTransformF(transformedWeights, weights, GetC, c_input_);
      transformed_weights_ = FP16.ToFP16(transformedWeights);

      Console.WriteLine("Warning: used invalid transormed weights in FusedWinogradConvSELayerCUDA");
      return;
      // TODO the cuda kerneel below ???
#endif

        transformedWeights = ToFiltered(stream, weights, NumInputChannels, C);
      }
    }


    CudaDeviceVariable<FP16> transformed_output;
    protected override void DoEval(CudaStream stream, int N, CudaDeviceVariable<FP16> output, 
                              CudaDeviceVariable<FP16> input,
                              CudaDeviceVariable<FP16> scratch, long scratchSizeBytes,
                              CudaDeviceVariable<FP16> scratchSecondHalf)
    {

      CudaDeviceVariable<FP16> transformed_input = scratch;

      kernelInputTransform.GridDimensions = N;
      kernelInputTransform.BlockDimensions = NumInputChannels;
      LaunchKernel(stream, kernelInputTransform, N, NumInputChannels, input.DevicePointer, transformed_input.DevicePointer);

      if (transformed_output == null)
      {
        transformed_output = new CudaDeviceVariable<FP16>(scratch.DevicePointer + scratchSizeBytes / 2, false);
      }

      cublasRowMajorMatrixMul(transformed_input, transformedWeights, transformed_output, N * 4, C, NumInputChannels, 36);

      CudaKernel kernel;

      // Only a small subset of the possible combinations of flags
      // are actually used, support only the required subset.
      Debug.Assert(!UseSE);
      Debug.Assert(UseBias);
      Debug.Assert(!WithSkipConnection);

      if (UseRELU && OpNCHW)
      {
        kernel = kernelOutputReluNHCW;
      }
      else if (UseRELU && !OpNCHW)
      {
        kernel = kernelOutputReluNotNHCW;
      }
      else if (!UseRELU && !OpNCHW)
      {
        kernel = kernelOutputNotReluNotNHCW;
      }
      else
        throw new NotImplementedException();

      kernel.GridDimensions = N;
      kernel.BlockDimensions = C;

      LaunchKernel(stream, kernel, N, C, (int)0, output.DevicePointer,
                   transformed_output.DevicePointer,
                   (IntPtr)0, biases.DevicePointer,
                   (IntPtr)0, (IntPtr)0, (IntPtr)0, (IntPtr)0);      
    }


    CudaKernel kernelInputTransform;
    CudaKernel kernelOutputReluNHCW;
    CudaKernel kernelOutputReluNotNHCW;
    CudaKernel kernelOutputNotReluNotNHCW;

    public override void LoadKernels()
    {
      // Possibly same as in ResidualBlockCUDA
      string resource = FP16_KERNELS_PTX_NAME;

      const string knInputTransform = "_ZN6lczero13cudnn_backend21InputTransform_kernelI6__halfLb0EEEviiPKT_PS3_";
      kernelInputTransform = Parent.Device.GetKernel(Parent.PTXAssembly, resource, knInputTransform);

      const string BASE_KN = "_ZN6lczero13cudnn_backend22OutputTransform_kernelI6__halfLb0ELb__RELU__ELb1ELb0ELb0ELb__NHCW__EEEviiiPT_PKS3_S6_S6_S6_S6_S6_S6_";
      string MakeName(bool relu, bool nhcw) => BASE_KN.Replace("__RELU__", relu ? "1" : "0")
                                                      .Replace("__NHCW__", nhcw ? "1" : "0");

      kernelOutputReluNHCW       = Parent.Device.GetKernel(Parent.PTXAssembly, resource, MakeName(true, true));
      kernelOutputReluNotNHCW    = Parent.Device.GetKernel(Parent.PTXAssembly, resource, MakeName(true, false));
      kernelOutputNotReluNotNHCW = Parent.Device.GetKernel(Parent.PTXAssembly, resource, MakeName(false, false));
    }
  }

}
