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

#endregion

namespace Ceres.Base.CUDA
{
  /// <summary>
  /// Implements disposable object which captures and optionally displays
  /// elapsed time between construction and Dispose of the object.
  /// </summary>
  public class CUDATimingBlock : IDisposable
  {
    /// <summary>
    /// Textual description of operation within block.
    /// </summary>
    public readonly string Description;

    /// <summary>
    /// If the timing statistics should be output to Console at end of block.
    /// </summary>
    public readonly bool OutputConsole;

    /// <summary>
    /// Total execution time in milliseconds of block.
    /// </summary>
    public float RuntimeMS { private set; get; }


    CudaEvent start;
    CudaEvent stop;
    CudaStream stream;

    /// <summary>
    /// Constructs CUDATimingBlock to track CUDA operations on a specified stream
    /// within a block, with specified desciptive name.
    /// </summary>
    /// <param name="desc"></param>
    /// <param name="stream"></param>
    /// <param name="outputConsole"></param>
    public CUDATimingBlock(string desc, CudaStream stream, bool outputConsole = true)
    {
      Description = desc;
      OutputConsole = outputConsole;

      this.stream = stream;
      start = new CudaEvent();
      stop = new CudaEvent();

      start.Record(stream.Stream);
    }


    protected virtual void Dispose(bool disposing)
    {
      stop.Record(stream.Stream);
      stop.Synchronize();

      float cudaElapsedTime = CudaEvent.ElapsedTime(start, stop);
      if (OutputConsole) Console.WriteLine("CUDA " + Description + " elapsed " + cudaElapsedTime + "ms");
      start.Dispose();
      stop.Dispose();
      RuntimeMS = cudaElapsedTime;
    }


    public void Dispose()
    {
      // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
      Dispose(disposing: true);
      GC.SuppressFinalize(this);
    }
  }


}
