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

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// Layer that maps between policy representations.
  /// </summary>
  public class LayerPolicyMapCUDA : BaseLayerCUDA
  {
    int usedSize;
    CudaDeviceVariable<short> indices;

    public LayerPolicyMapCUDA(NNBackendExecContext parent, string name, int layerIndex,
                              BaseLayerCUDA ip, int c, int h, int w, int usedSize)
      : base(parent, name, layerIndex, c, h, w, ip)
    {
      this.usedSize = usedSize;
    }

    CudaKernel kernelPolicyMap;


    public override void LoadKernels()
    {
      string kn = "_ZN6lczero13cudnn_backend16policyMap_kernelI6__halfEEvPT_PKS3_PKsiiii";
      kernelPolicyMap = Parent.Device.GetKernel(Parent.PTXAssembly, @"common_kernels.ptx", kn);
    }


    public void LoadWeights(short[] cpuWeight)
    {
      if (Parent.ReferenceLayers != null)
      {
        LayerPolicyMapCUDA refLayer = Parent.ReferenceLayers.Layers[LayerIndex] as LayerPolicyMapCUDA;
        indices = refLayer.indices;
      }
      else
      {
        indices = cpuWeight;
      }
    }


    protected override void DoEval(CudaStream stream, int N, CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input, 
                                   CudaDeviceVariable<FP16> scratch, long scratch_size, CudaDeviceVariable<FP16> scratchSecondHalf)
    {
      int inputSize = input_.C * input_.GetH * input_.W;
      int outputSize = C * GetH * W;

      string kn = "_ZN6lczero13cudnn_backend16policyMap_kernelI6__halfEEvPT_PKS3_PKsiiii";
      CudaKernel kernelPolicyMap = Parent.Device.GetKernel(Parent.PTXAssembly, @"common_kernels.ptx", kn);

      // Each thread processes one input element
      // Only some of the threads (with valid mapping) write output
      const int kBlockSize = 256;
      int kBlocks = CUDAUtils.DivUp(N * usedSize, kBlockSize);
      kernelPolicyMap.BlockDimensions = kBlockSize;
      kernelPolicyMap.GridDimensions = kBlocks;

      LaunchKernel(stream, kernelPolicyMap, output.DevicePointer, input.DevicePointer, indices.DevicePointer, N, inputSize, usedSize, outputSize);
    }


  }
}
