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
using System.Reflection;

using ManagedCuda;
using ManagedCuda.CudaBlas;

using Ceres.Base.CUDA;

#endregion

namespace Ceres.Chess.NNBackends.CUDA
{

  /// <summary>
  /// Captures all the state variables related to CUDA
  /// as used by a specific instance of an NN backend.
  /// </summary>
  public record NNBackendExecContext
  {
    /// <summary>
    /// Underlying CUDA on device on which operations execute.
    /// </summary>
    public readonly CUDADevice Device;

    /// <summary>
    /// Stream used when executing operations by this backend.
    /// </summary>
    public readonly CudaStream Stream;

    /// <summary>
    /// CUBLAS handle associated with this backend.
    /// </summary>
    public readonly CudaBlas CuBlas;

    /// <summary>
    /// CUBLAS LT handle associated with this backend.
    /// </summary>
    public readonly CudaBlasLTHandle CuBlasLT;

    /// <summary>
    /// The assembly which contains the resources with the PTX text.
    /// </summary>
    public readonly Assembly PTXAssembly;

    /// <summary>
    /// If the layer-by-layer timing debug information should be dumped to Console.
    /// </summary>
    public readonly bool DumpTimings;

    /// <summary>
    /// Optional peer layer object which already
    /// is initialized with same set of weights (which can be reused).
    /// </summary>
    internal NNBackendCUDALayers ReferenceLayers;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="stream"></param>
    /// <param name="cuBlas"></param>
    /// <param name="cuBlasLT"></param>
    /// <param name="ptxAssembly"></param>
    public NNBackendExecContext(CUDADevice context, CudaStream stream, 
                                CudaBlas cuBlas, CudaBlasLTHandle cuBlasLT, 
                                Assembly ptxAssembly, bool dumpTimings)
    {
      Device = context ?? throw new ArgumentNullException(nameof(context));
      Stream = stream ?? throw new ArgumentNullException(nameof(stream));
      CuBlas = cuBlas;
      CuBlasLT = cuBlasLT;
      PTXAssembly = ptxAssembly ?? throw new ArgumentNullException(nameof(ptxAssembly));
      DumpTimings = dumpTimings;
    }


    public override string ToString()
    {
      return $"<NNBackendExecContext on GPU {Device.GPUID} {(ReferenceLayers == null ? "without" : "with")} reference layer>";
    }

  }

}
