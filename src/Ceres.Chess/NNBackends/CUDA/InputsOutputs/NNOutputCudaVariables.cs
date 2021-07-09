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

#endregion

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// Collection of CudaDeviceVariables used to receive the outputs of the network.
  /// </summary>
  internal record NNOutputCudaVariables : IDisposable
  {
    /// <summary>
    /// Output of the policy head.
    /// </summary>
    public CudaDeviceVariable<FP16> PolicyOut;

    /// <summary>
    /// Output of the penultimate value head layer.
    /// </summary>
    public CudaDeviceVariable<FP16> ValueHeadFC2Out;

    /// <summary>
    /// Output of the value head.
    /// </summary>
    public CudaDeviceVariable<FP16> ValueOut;

    /// <summary>
    /// Output of the MLH head.
    /// </summary>
    public CudaDeviceVariable<FP16> MLHOut;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxBatchSize"></param>
    /// <param name="wdl"></param>
    /// <param name="mlh"></param>
    public NNOutputCudaVariables(int maxBatchSize, bool wdl, bool mlh)
    {
      PolicyOut = new CudaDeviceVariable<FP16>(maxBatchSize * NNBackendInputOutput.NUM_OUTPUT_POLICY);
      ValueOut = new CudaDeviceVariable<FP16>(maxBatchSize * (wdl ? 3 : 1));
      ValueHeadFC2Out = new CudaDeviceVariable<FP16>(maxBatchSize * 128);

      if (mlh)
      {
        MLHOut = new CudaDeviceVariable<FP16>(maxBatchSize * 1);
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
        PolicyOut.Dispose();
        ValueOut.Dispose();
        MLHOut?.Dispose();
        ValueHeadFC2Out.Dispose();

        disposed = true;
      }
    }
  }

}
