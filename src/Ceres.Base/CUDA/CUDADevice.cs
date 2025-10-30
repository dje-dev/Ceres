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
using Ceres.Base.OperatingSystem;
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
  /// including the underlying CudaContext and various synchronization objects,
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
    public PrimaryContext Context;


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
    /// Object on which a lock should be taken during the time in which the CUDA context is being initialized.
    /// This includes the situations where user code is initializing the context, or where the context is being
    /// initialized on another thread by external libraries invoked by user code (for example ONNXRuntime).
    /// </summary>
    public static readonly object InitializingCUDAContextLockObj = new object();


    /// <summary>
    /// Constructor for a context for a specified device.
    /// 
    /// N.B. The argument createNew is set to false in CudaContext constructor.
    ///      This avoids initialization Exception if the context was already created elsewhere
    ///      (for example, the ONNX runtime might have been loaded in-process and initialized CUDA).
    /// </summary>
    /// <param name="gpuID"></param>
    internal CUDADevice(int gpuID)
    {
      lock (InitializingCUDAContextLockObj)
      {
        GPUID = gpuID;
        Context = new PrimaryContext(gpuID);

        CudaDeviceProperties deviceProperties = Context.GetDeviceInfo();
        driverVersionMajor = deviceProperties.DriverVersion.Major;
        driverVersionMinor = deviceProperties.DriverVersion.Minor;
      }
    }


    static int driverVersionMajor = -1;
    static int driverVersionMinor = -1;

    /// <summary>
    /// Returns the major and minor version of CUDA installed.
    /// </summary>
    /// <returns></returns>
    public static (int majorVersion, int minorVersion) GetCUDAVersion()
    {
      if (!SoftwareManager.IsCUDAInstalled)
      {
        return default;
      }

      if (driverVersionMajor == -1)
      {
        // No device ever initialized; force this now (then release).
        using (CUDADevice context = CUDADevice.GetContext(0))
        {
        }
      }

      return (driverVersionMajor, driverVersionMinor);
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
      SetLazyLoading();

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


    static bool haveSetLazyLoading = false;

    /// <summary>
    /// Setting CUDA_MODULE_LOADING = LAZY is supposed to reduce load time and
    /// memory usage(starting with 11.7, and even more so with 11.8).

    /// With 11.7 on Linux using Ceres.Indeed, loading 2 networks(T70, T80)
    /// on all 4 GPUs takes 8.98sec and 4903k without but only 7.53sec and 4477k.

    /// However setting the environment variable programmatically at initialization(before call to cudaInit())
    /// surprisingly doesn't work for me; one has to set it from shell before launch.
    /// </summary>
    static void SetLazyLoading()
    {
      if (!haveSetLazyLoading)
      {
        // TODO: this does not actualy seem to have any effect; see above.
        System.Environment.SetEnvironmentVariable("CUDA_MODULE_LOADING", "LAZY");

        haveSetLazyLoading = true;
      }
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
          contexts.Remove(GPUID, out _);
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
