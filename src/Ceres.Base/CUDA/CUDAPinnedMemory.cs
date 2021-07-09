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

using ManagedCuda;
using ManagedCuda.BasicTypes;

#endregion

namespace Ceres.Base.CUDA
{
  /// <summary>
  /// Wrapper around CudaPageLockedHostMemory for holding
  /// CUDA memory allocation at a fixed address to allow
  /// asynchronous memory copies isolated to one stream.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class CUDAPinnedMemory<T> : IDisposable where T : unmanaged
  {
    /// <summary>
    /// Total number of allocated elements.
    /// </summary>
    public readonly int NumElements;

    /// <summary>
    /// Returns the underlying CudaPageLockedHostMemory object.
    /// </summary>
    public CudaPageLockedHostMemory<T> Memory { get; private set; }

    /// <summary>
    /// Returns a span into the pinned memory.
    /// </summary>
    /// <returns></returns>
    public unsafe Span<T> AsSpan() => new Span<T>((T*)Memory.PinnedHostPointer, NumElements);

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="numElements"></param>
    /// <param name="hostToDeviceOnly"></param>
    public CUDAPinnedMemory(int numElements, bool hostToDeviceOnly = false)
    {
      NumElements = numElements;
      Memory = new CudaPageLockedHostMemory<T>(numElements, hostToDeviceOnly ? CUMemHostAllocFlags.WriteCombined
                                                                             : CUMemHostAllocFlags.None);
    }

    /// <summary>
    /// Copies data from host to device memory.
    /// </summary>
    /// <param name="dest">Destination pointer to host memory</param>
    public void CopyToDeviceAsync<T>(CudaDeviceVariable<T> dest,
                                     int numElements, CudaStream stream) where T : unmanaged
    {
      if (numElements > dest.Size) throw new ArgumentOutOfRangeException("Array too short");
      SizeT aSizeInBytes = numElements * Marshal.SizeOf<T>();

      CUResult res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(dest.DevicePointer, Memory.PinnedHostPointer, aSizeInBytes, stream.Stream);
      if (res != CUResult.Success)
      {
        throw new CudaException(res);
      }
    }


    /// <summary>
    /// Copy data from device to host memory.
    /// </summary>
    /// <param name="dest">Destination pointer to host memory</param>
    public void CopyToHostAsync<T>(CudaDeviceVariable<T> data,
                                   int numElements, CudaStream stream) where T : unmanaged
    {
      if (numElements > NumElements) throw new ArgumentOutOfRangeException("Array too short");
      SizeT aSizeInBytes = numElements * Marshal.SizeOf<T>();

      CUResult res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(Memory.PinnedHostPointer, data.DevicePointer, aSizeInBytes, stream.Stream);
      if (res != CUResult.Success)
      {
        throw new CudaException(res);
      }
    }

    /// <summary>
    /// Destructor.
    /// </summary>
    ~CUDAPinnedMemory()
    {
      Dispose();
    }


    /// <summary>
    /// Diposes underlying memory object.
    /// </summary>
    public void Dispose()
    {
      if (Memory != null)
      {
        Memory.Dispose();
        Memory = null;
      }
    }
  }

}
