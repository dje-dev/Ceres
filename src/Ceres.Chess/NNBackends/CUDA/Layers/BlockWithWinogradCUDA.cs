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

using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;
using ManagedCuda;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public abstract class BlockWithWinogradCUDA : BaseLayerCUDA
  {
    CudaKernel kernelFilterTransform;

    public BlockWithWinogradCUDA(NNBackendExecContext parent, string name, int layerIndex,
                                 int c, int h, int w, 
                                 BaseLayerCUDA inputLayer)
      : base(parent, name, layerIndex, c, h, w, inputLayer)
      {
      }

    internal CudaDeviceVariable<FP16> ToFiltered(CudaStream stream, float[] weights, int c_input_, int C)
    {
      // NOTE: These repeated large allocations are somewhat time consuming.
      //       Consider optimizing by moving to single allocation with suballocations.
      int filteredSize = 4 * c_input_ * C * 3 * 3;
      CudaDeviceVariable<FP16> filteredWeights = new CudaDeviceVariable<FP16>(filteredSize);

      using (CudaDeviceVariable<FP16> weightsGPU = FP16.ToFP16Approx(weights))
      {
        // Run winograd transform kernel for the filter
        if (kernelFilterTransform == null)
        {
          const string kn = "_ZN6lczero13cudnn_backend22filterTransform_kernelI6__halfEEviiiPT_PKS3_";
          kernelFilterTransform = Parent.Device.GetKernel(Parent.PTXAssembly, FP16_KERNELS_PTX_NAME, kn);
        }

        // Each thread processes entire filter block (input 3x3 elements -> output 6x6 elements)
        const int BLOCK_SIZE = 64;
        int kBlocks = CUDAUtils.DivUp(C * c_input_, BLOCK_SIZE);
        kernelFilterTransform.BlockDimensions = BLOCK_SIZE;
        kernelFilterTransform.GridDimensions = kBlocks;

        // FilterTransform(int N, int C, T* transformedFilter, const T* filter)
        kernelFilterTransform.RunAsync(stream.Stream, C, c_input_, C * c_input_, filteredWeights.DevicePointer, weightsGPU.DevicePointer);

        return filteredWeights;
      }
    }

  }
}
