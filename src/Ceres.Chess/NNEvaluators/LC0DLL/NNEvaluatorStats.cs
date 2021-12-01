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

using System.Runtime.CompilerServices;
using System.Threading;

#endregion

namespace Ceres.Chess.NNEvaluators.Internals
{
  /// <summary>
  /// Manager of global statistics about total number of 
  /// batches/positions evaluated by each GPU cumulatively.
  /// 
  /// TODO: deal with tracking CPU evaluations separately?
  /// </summary>
  public static class NNEvaluatorStats
  {
    public const int MAX_GPUS = 16;

    /// <summary>
    /// Total number of batches evaluated using each GPU
    /// </summary>
    public static int[] TotalBatchesPerGPU;

    /// <summary>
    /// Total number of positions evaluated using each GPU
    /// </summary>
    public static long[] TotalPosEvaluationsPerGPU;

    /// <summary>
    /// Updates statistics to reflect that a batch with specified number of positions was evaluated.
    /// </summary>
    /// <param name="gpuID"></param>
    /// <param name="numPositions"></param>
    public static void UpdateStatsForBatch(int gpuID, int numPositions)
    {
      Interlocked.Add(ref TotalPosEvaluationsPerGPU[gpuID], numPositions);
      Interlocked.Increment(ref TotalBatchesPerGPU[gpuID]);
    }

    /// <summary>
    /// Total number of positions evaluated across all GPUs.
    /// </summary>
    public static long TotalPosEvaluations
    {
      get
      {
        long acc = 0;
        for (int i=0; i< MAX_GPUS; i++)
        {
          acc += TotalPosEvaluationsPerGPU[i];
        }
        return acc;
      }
    }

    #region Internals

    [ModuleInitializer]
    internal static void ClassInitialize()
    {
      TotalBatchesPerGPU = new int[MAX_GPUS];
      TotalPosEvaluationsPerGPU = new long[MAX_GPUS];
    }

    #endregion
  }
}
