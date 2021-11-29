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
using Ceres.Base.DataTypes;
using ManagedCuda;
using ManagedCuda.BasicTypes;

#endregion

namespace Ceres.Base.CUDA
{
  /// <summary>
  /// Wrapper for a single CUDA kernel that facilitates
  /// rapid launching of the kernel, using a set of
  /// preallocated and pinned objects for data transfer.
  /// </summary>
  public class CUDAKernelLauncher : IDisposable
  {
    /// <summary>
    /// The associated kernel.
    /// </summary>
    public readonly CudaKernel Kernel;

    /// <summary>
    /// The stream on which this kernel will execute.
    /// </summary>
    public readonly CudaStream Stream;

    /// <summary>
    /// Number of bytes of shared memory to be used by the kernel.
    /// </summary>
    public readonly int SharedMemoryNumBytes;

    /// <summary>
    /// 
    /// </summary>
    public readonly PinnedObjectArray Parms;

    public CUDAKernelLauncher(CudaKernel kernel, CudaStream stream, int sharedMemoryNumBytes, params object[] parameters)
    {
      Kernel = kernel;
      Stream = stream;
      SharedMemoryNumBytes = sharedMemoryNumBytes;

      Parms = new PinnedObjectArray(parameters);
    }

    public void LaunchAsync()
    {
      // TODO: verify shared zero
      CUResult res = DriverAPINativeMethods.Launch.cuLaunchKernel(Kernel.CUFunction,// _function, 
                                                                  Kernel.GridDimensions.x, Kernel.GridDimensions.y, Kernel.GridDimensions.z,
                                                                  Kernel.BlockDimensions.x, Kernel.BlockDimensions.y, Kernel.BlockDimensions.z,
                                                                  (uint)SharedMemoryNumBytes, Stream.Stream, Parms.Pointers, null);
      if (res != CUResult.Success)
      {
        throw new Exception($"Failure in cuLaunchKernel: {res}");
      }
    }

    ~CUDAKernelLauncher()
    {
      Dispose();
    }

    public void Dispose()
    {
      Parms.Dispose();

    }
  }

}
