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
using System.Collections.Generic;
using ManagedCuda;

#endregion

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// Manages a pool of NNBackendCUDAGraph objects
  /// each of which is created for a specified batch size.
  /// </summary>
  internal class NNBackendCUDAGraphSet : IDisposable
  {
    /// <summary>
    /// Delegate called during graph capture
    /// which performs the CUDA operations. 
    /// </summary>
    /// <param name="batchSize"></param>
    public delegate void GraphBuilder(CudaStream stream, int batchSize);

    /// <summary>
    /// Set of batch size breaks at which new graphs are created.
    /// </summary>
    public readonly int[] BatchSizes;

    /// <summary>
    /// Method that can be called in streaming mode to capture 
    /// CUDA operations to be present in the graph.
    /// </summary>
    public GraphBuilder Builder;

    /// <summary>
    /// Cache of graphs already created (indexed by maximum batch size).
    /// </summary>
    Dictionary<int, NNBackendCUDAGraph> graphs = new ();


    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="builder"></param>
    /// <param name="batchSizes"></param>
    public NNBackendCUDAGraphSet(GraphBuilder builder, int[] batchSizes)
    {
      Builder = builder;
      BatchSizes = batchSizes;
    }

    /// <summary>
    /// Returns the smallest batch size in the set of valid sizes
    /// which is large enough to contain the batch of specified size.
    /// </summary>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    int SizeToUse(int batchSize)
    {
      for (int i = 0; i < BatchSizes.Length; i++)
      {
        if (BatchSizes[i] >= batchSize)
        {
          return BatchSizes[i];
        }
      }
      return -1;
    }


    /// <summary>
    /// Returns if it will be necessary to construct a 
    /// new graph if a batch size of a specified size is requested.
    /// </summary>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    public bool GraphForBatchSizeNeedsConstruction(int batchSize)
    {
      int sizeToUse = SizeToUse(batchSize);

      if (sizeToUse == -1) return false;

      return !graphs.ContainsKey(sizeToUse);
    }


    /// <summary>
    /// Returns a graph able to evaluate a batch of specified size,
    /// or null if no such graph is available.
    /// </summary>
    /// <param name="batchSize"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    public NNBackendCUDAGraph GetGraphForBatchSize(NNBackendExecContext execContext, int batchSize)
    {
      int sizeToUse = SizeToUse(batchSize);

      if (sizeToUse == -1)
      {
        return null;
      }

      if (!graphs.ContainsKey(sizeToUse))
      {
        // Create a graph for this batch size.
        NNBackendCUDAGraph graph = new NNBackendCUDAGraph(execContext.Device, sizeToUse);

        // Record CUDA operations to evaluate batch of this size.
        graph.BeginCaptureGraph(execContext.Stream);
        Builder(execContext.Stream, sizeToUse);
        graph.EndCaptureGraph(execContext.Stream);

        // Save graph for future use.
        graphs[sizeToUse] = graph;
      }

      return graphs[sizeToUse];
    }


    #region Disposal

    ~NNBackendCUDAGraphSet()
    {
      Dispose();
    }


    bool disposed = true;
    public void Dispose()
    {
      if (!disposed)
      {
        foreach (NNBackendCUDAGraph graph in graphs.Values)
        {
          graph.Dispose();
        }
        disposed = true;
      }
    }

    #endregion
  }

}
