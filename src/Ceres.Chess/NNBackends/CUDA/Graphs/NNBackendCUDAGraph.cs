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

#endregion

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// Wraps a single CUDA Graph object used to capture 
  /// an NN evaluation for a batch of a specified fixed size.
  /// 
  /// Capturing and instantiating a graph seems 
  /// to require about 3 milliseconds and consumes some GPU memory.
  /// </summary>
  internal class NNBackendCUDAGraph : IDisposable
  {
    /// <summary>
    /// Parent device context.
    /// </summary>
    public readonly CUDADevice Device;

    /// <summary>
    /// The implicit batch size for which the captured graph applies.
    /// </summary>
    public readonly int BatchSize;


    #region Private data

    /// <summary>
    /// Underlying CUDA graph execution object.
    /// </summary>
    CUgraphExec exec = default;

    /// <summary>
    /// Underlying CUDA graph root node.
    /// </summary>
    CUgraphNode node = default;

    /// <summary>
    /// If graph objects have been released.
    /// </summary>
    bool disposed = false;


    #endregion


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="device"></param>
    /// <param name="batchSize"></param>
    public NNBackendCUDAGraph(CUDADevice device, int batchSize)
    {
      Device = device;
      BatchSize = batchSize;
    }

    /// <summary>
    /// Begins the capture of the graph.
    /// A write lock at the device level is taken.
    /// </summary>
    /// <param name="stream"></param>
    internal void BeginCaptureGraph(CudaStream stream)
    {
      exec = default;
      node = default;
      Device.GraphCaptureRWLock.EnterWriteLock();
      CUDAUtils.Check(DriverAPINativeMethods.Streams.cuStreamBeginCapture(stream.Stream, CUstreamCaptureMode.Global));
    }


    /// <summary>
    /// Ends capture of the graph.
    /// The write lock at the device level is released.
    /// </summary>
    /// <param name="stream"></param>
    internal void EndCaptureGraph(CudaStream stream)
    {
      CUgraph graphObj = default;

      CUDAUtils.Check(DriverAPINativeMethods.Streams.cuStreamEndCapture(stream.Stream, ref graphObj));
      CUDAUtils.Check(DriverAPINativeMethods.GraphManagment.cuGraphInstantiate(ref exec, graphObj, ref node, null, 0));
      CUDAUtils.Check(DriverAPINativeMethods.GraphManagment.cuGraphDestroy(graphObj));
      Device.GraphCaptureRWLock.ExitWriteLock();
    }


    /// <summary>
    /// Executes the captured graph.
    /// </summary>
    /// <param name="stream"></param>
    /// <param name="batchSize"></param>
    internal void RunGraph(CudaStream stream, int batchSize)
    {
      if (batchSize > BatchSize)
      {
        throw new Exception("incorrect batch size");
      }

      CUDAUtils.Check(DriverAPINativeMethods.GraphManagment.cuGraphLaunch(exec, stream.Stream));
    }

    #region Disposal

    /// <summary>
    /// Destructor to release graph objects.
    /// </summary>
    ~NNBackendCUDAGraph()
    {
      Dispose();
    }


    /// <summary>
    /// Dispose method to release graph objects.
    /// </summary>
    public void Dispose()
    {
      if (!disposed)
      {
        DriverAPINativeMethods.GraphManagment.cuGraphExecDestroy(exec);
        DriverAPINativeMethods.GraphManagment.cuGraphDestroyNode(node);
        disposed = true;
      }
    }

    #endregion
  }

}
