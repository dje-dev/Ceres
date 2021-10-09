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

using ManagedCuda;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Threading;

#endregion

namespace Ceres.Base.CUDA
{
  /// <summary>
  /// Manages the state of a single CUDA device
  /// including the underlying CudaContext and various sychronization objects,
  /// and facilitating kernel loading.
  /// </summary>
  public class CUDADevice
  {
    /// <summary>
    /// Index of underlying GPU device.
    public readonly int GPUID;


    /// <summary>
    /// Associated (default) context for this device.
    /// </summary>
    public CudaContext Context;


    /// <summary>
    /// Lock to prevent concurrent evaluation.
    /// </summary>
    public readonly object ExecLockObj = new object();


    // It seems not possible to capture a graph on any device
    // while any other device is active. Therefore we 
    // use a reader/writer lock to permit only one writer at a time
    // i.e. only one thread can be capturing at one time.

    // Note that this seems device-wide and any concurrent activity will cause failure,
    // including for example from a LC0LibraryNNEvaluator in another thread of this process.
    static ReaderWriterLockSlim graphCaptureRWLock = new(LockRecursionPolicy.SupportsRecursion);
    public ReaderWriterLockSlim GraphCaptureRWLock => graphCaptureRWLock;



    /// <summary>
    /// Constructor for a context for a specified device.
    /// </summary>
    /// <param name="gpuID"></param>
    public CUDADevice(int gpuID)
    {
      //int deviceCount = CudaContext.GetDeviceCount();

      GPUID = gpuID;
      Context = new CudaContext(gpuID, true);
    }

    [ThreadStatic] static CudaContext currentContext;

    public void SetCurrent()
    {
     if (!object.ReferenceEquals(currentContext, Context))
      {
        Context.SetCurrent();
        currentContext = Context;
      }
    }

    #region Kernel loading
  
    ConcurrentDictionary<string, CudaKernel> cachedKernels = new();
    public CudaKernel GetKernel(Assembly assembly, string resource, string kernelName)
    {
      CudaKernel ret;
      if (!cachedKernels.TryGetValue(resource + kernelName, out ret))
      {
        ret = cachedKernels[resource + kernelName] = DoLoadKernel(assembly, Context, resource, kernelName);
      }
      return ret;
    }


    const string BASE_RESOURCE_NAME = @"Ceres.Chess.NNBackends.CUDA.Kernels.PTX.";

    static Dictionary<string, Stream> resourceStreams = new Dictionary<string, Stream>();

    public CudaKernel DoLoadKernel(Assembly assembly, CudaContext context, string resource, string kernelName)
    {
      string resourceName = BASE_RESOURCE_NAME + resource;
      Stream stream;
      string key = assembly.FullName + resource;
      if (!resourceStreams.TryGetValue(key, out stream))
      {
        stream = resourceStreams[key] = assembly.GetManifestResourceStream(resourceName);
        if (stream == null) throw new Exception($"Kernel {resourceName} not found in embedded resource.");
      }

      return context.LoadKernelPTX(stream, kernelName);
    }

    #endregion

    #region Statics

    public static ConcurrentDictionary<int, CUDADevice> contexts = new();

    public static CUDADevice GetContext(int gpuID)
    {
      CUDADevice device = null;
      if (!contexts.TryGetValue(gpuID, out device))
      {
        lock (contexts)
        {
          if (!contexts.TryGetValue(gpuID, out device))
          {
            if (gpuID >= CudaContext.GetDeviceCount())
            {
              throw new Exception($"CUDA device does not exist: {gpuID}");
            }

            device = contexts[gpuID] = new CUDADevice(gpuID);
          }
        }
      }
      return device;
    }

    #endregion


    public override string ToString()
    {
      return $"<CUDADevice GPU {GPUID} on CUDA context {Context.Context.Pointer}>";
    }

  }
}
