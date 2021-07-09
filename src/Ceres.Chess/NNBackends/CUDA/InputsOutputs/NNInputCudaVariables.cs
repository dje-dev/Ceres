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
using System.Runtime.InteropServices;
using Ceres.Base.DataTypes;
using ManagedCuda;
using ManagedCuda.BasicTypes;

#endregion

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// Collection of CudaDeviceVariables used to provide the inputs to the neural network.
  /// </summary>
  internal record NNInputCudaVariables : IDisposable
  {
    public readonly long ScratchSizeBytes;

    /// <summary>
    /// Memory are used for temporary calculations.
    /// </summary>
    public CudaDeviceVariable<FP16> Scratch;

    /// <summary>
    /// View into the second half of the Scratch variable.
    /// </summary>
    public CudaDeviceVariable<FP16> ScratchSecondHalf;

    /// <summary>
    /// Array of tensors used for the layer inputs/outputs.
    /// </summary>
    public CudaDeviceVariable<FP16>[] Tensors;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="scratchSizeElements"></param>
    /// <param name="tensorsSizeElements"></param>
    public NNInputCudaVariables(long scratchSizeElements, long tensorsSizeElements)
    {
      Scratch = new CudaDeviceVariable<FP16>(scratchSizeElements);
      ScratchSizeBytes = scratchSizeElements * Marshal.SizeOf<FP16>();
      CUdeviceptr addr = (CUdeviceptr)(Scratch.DevicePointer + ScratchSizeBytes / 2);
      ScratchSecondHalf = new CudaDeviceVariable<FP16>(addr, false, ScratchSizeBytes / 2);

      Tensors = new CudaDeviceVariable<FP16>[3];
      for (int i = 0; i < 3; i++)
      {
        Tensors[i] = new CudaDeviceVariable<FP16>(tensorsSizeElements);
      }
    }

    
    bool disposed = false;

    /// <summary>
    /// Disposes of associated CUDA memory objects.
    /// </summary>
    public void Dispose()
    {
      if (!disposed)
      {
        Scratch.Dispose();
        ScratchSecondHalf.Dispose();
        for (int i=0; i<Tensors.Length;i++)
        {
          Tensors[i].Dispose();
        }

        disposed = true;
      }
    }
  }

}
