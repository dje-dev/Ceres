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

using Ceres.Base.DataTypes;
using ManagedCuda;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

#endregion

namespace Ceres.Base.CUDA
{
  /// <summary>
  /// Manages the state of a single CUDA device
  /// including the underlying CudaContext and various sychronization objects,
  /// and facilitating kernel loading.
  /// </summary>
  public class CUDADevice : IDisposable
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

        if (ret == null)
        {
          throw new Exception($"CUDA kernel {kernelName} not found in resource {resource}");
        }
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


    /// <summary>
    /// Debugging diagnostic helper that outputs checksum and first element from a CudaDeviceVariable<FP16>.
    /// </summary>
    /// <param name="description"></param>
    /// <param name="data"></param>
    public void DumpVariableChecksum(string description, CudaDeviceVariable<FP16> data, int numItems)
    {
      // Make sure asynchronous computation finishes.
      Context.Synchronize();

      // Retrieve raw data from GPU.
      if (data.Size < numItems)
      {
        numItems = data.Size;
      }

      FP16[] raw = new FP16[numItems];
      data.CopyToHost(raw, 0, 0, numItems * Marshal.SizeOf<FP16>());

      // Compute a checksum, giving odd and even elements different multipliers
      // to make values at least somewhat position dependent.
      float acc = 0;
      for (int i = 0; i < numItems; i++)
      {
        float val = raw[i];
        if (!haveWarnedNaN && float.IsNaN(val))
        {
          Console.WriteLine("NaN at " + i + " in " + description);
          haveWarnedNaN = true;
        }
        acc += (i % 2 == 0 ? 1 : 3) * raw[i];
      }

      Console.WriteLine("CHECKSUM: " + description + " " + acc + " first=" + raw[0]);
    }
    bool haveWarnedNaN = false;

#if EQUIVALENT_CPP
  void Dump(cudaStream_t stream, char* desc, const half* data, int numElements) 
  { 
    cudaStreamSynchronize(stream);
  
    half* raw = new half[numElements];
    cudaMemcpy((half*)(&raw[0]), data, numElements * 2, cudaMemcpyDeviceToHost);

    float acc = 0;
    for (int i = 0; i < numElements; i++) {
      acc += (i % 2 == 0 ? 1 : 3) * (float)raw[i];
    }

    float first = (float)raw[0];
    printf("CHECKSUM: %s %f first= %f\r\n", desc, acc, first);
  }
#endif

    #region Statics

    public static ConcurrentDictionary<int, CUDADevice> contexts = new();
    private bool disposedValue;

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

    #region Dispose

    public override string ToString()
    {
      return $"<CUDADevice GPU {GPUID} on CUDA context {Context.Context.Pointer}>";
    }

    protected virtual void Dispose(bool disposing)
    {
      if (!disposedValue)
      {
        if (disposing)
        {
          Context.Dispose();
        }

        disposedValue = true;
      }
    }

    public void Dispose()
    {
      Dispose(disposing: true);
      GC.SuppressFinalize(this);
    }

    #endregion
  }
}
