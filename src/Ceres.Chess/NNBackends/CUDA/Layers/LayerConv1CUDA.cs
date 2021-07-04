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
  public class LayerConv1CUDA : BaseLayerCUDA
  {
    /// <summary>
    /// Number of channels.
    /// </summary>
    public readonly int C;

    /// <summary>
    /// If RELU activations used.
    /// </summary>
    bool UseRELU;

    /// <summary>
    /// If bias weights included.
    /// </summary>
    bool UseBias;

    // Constants
    CudaDeviceVariable<FP16> halfOne;
    CudaDeviceVariable<FP16> halfZero;

    CudaDeviceVariable<FP16> biasesCUDA;
    CudaDeviceVariable<FP16> weightsCUDA;

    
    public LayerConv1CUDA(NNBackendExecContext parent, string name, int layerIndex, 
                          BaseLayerCUDA ip, int c, int h, int w, 
                          int Cin,
                          bool relu = false, bool bias = false)
      : base(parent, name, layerIndex, c, h, w, ip)
    {
      C = Cin;
      UseRELU = relu;
      UseBias = bias;

      halfOne = new CudaDeviceVariable<FP16>(1);
      halfOne.CopyToDevice(new FP16[] { 1 });
      halfZero = new CudaDeviceVariable<FP16>(1);
      halfZero.CopyToDevice(new FP16[] { 0 });

    }

    CudaKernel kernel;
    public override void LoadKernels()
    {
      const string knAddBias = "_ZN6lczero13cudnn_backend19addBias_NCHW_kernelI6__halfEEvPT_S4_S4_iiiib";
      kernel = Parent.Device.GetKernel(Parent.PTXAssembly, @"common_kernels.ptx", knAddBias);
    }


    public void LoadWeights(float[] pfilter, float[] pBias)
    {
      if (Parent.ReferenceLayers != null)
      {
        LayerConv1CUDA refLayer = Parent.ReferenceLayers.Layers[LayerIndex] as LayerConv1CUDA;
        biasesCUDA = refLayer.biasesCUDA;
        weightsCUDA = refLayer.weightsCUDA;
      }
      else
      {
        biasesCUDA = LoadedWeights(pBias, base.C);
        weightsCUDA = LoadedWeights(pfilter, base.C * C);
      }
    }


    protected override void DoEval(CudaStream stream, int N, CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input,
                                   CudaDeviceVariable<FP16> scratch, long scratch_size,
                                   CudaDeviceVariable<FP16> scratchSecondHalf)
    {
      cublasRowMajorMatrixMul(weightsCUDA, input, output, base.C, GetH * W, C, N, true);

      if (UseBias)
      {
        const int kBlockDimension = 256;
        int size = N * base.C * GetH * W;
        kernel.BlockDimensions = kBlockDimension;
        kernel.GridDimensions = CUDAUtils.DivUp(size, kBlockDimension);

        LaunchKernel(stream, kernel, output.DevicePointer, 
                     output.DevicePointer, biasesCUDA.DevicePointer,
                     N, base.C, GetH, W, UseRELU ? 1 : 0);
      }
      else if (UseRELU)
      {
        // Not currently used. If it were, look in LayerFCCUDA for exampe of calling the addVectors kernel.
        throw new Exception("Unimplemented.");
        //addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W, 0, use_relu_, false, false);
      }
    }

  }

}
