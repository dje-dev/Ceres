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

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Build options for TensorRT engine compilation.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public record struct TensorRTBuildOptions
{
  /// <summary>
  /// Builder optimization level (0-5, default 3).
  /// </summary>
  public int BuilderOptimizationLevel;

  /// <summary>
  /// Tiling optimization level (-1 for default, 0-5 otherwise).
  /// For large batch sizes (e.g. 128 or above) on some GPUs
  /// using values or 5 can yield 5% to 15% speedups.
  /// </summary>
  public int TilingOptimizationLevel;

  /// <summary>
  /// Use spin wait (1 = true (default), 0 = false).
  /// </summary>
  public int UseSpinWait;

  /// <summary>
  /// Use CUDA graphs (1 = true, 0 = false (default)).
  /// </summary>
  public int UseCudaGraphs;

  /// <summary>
  /// Use FP16 precision (1 = true (default), 0 = false).
  /// </summary>
  public int UseFP16;

  /// <summary>
  /// Use BF16 precision (1 = true, 0 = false (default)).
  /// </summary>
  public int UseBF16;

  /// <summary>
  /// Use FP8 precision (1 = true, 0 = false (default)).
  /// </summary>
  public int UseFP8;

  /// <summary>
  /// Use best available precision (1 = true, 0 = false (default)).
  /// </summary>
  public int UseBest;

  /// <summary>
  /// Minimum batch size for optimization profile (0 = use batchSize).
  /// </summary>
  public int MinBatchSize;

  /// <summary>
  /// Optimal batch size for optimization profile (0 = use batchSize).
  /// </summary>
  public int OptBatchSize;

  /// <summary>
  /// Maximum batch size for optimization profile (0 = use batchSize).
  /// </summary>
  public int MaxBatchSize;

  /// <summary>
  /// Force FP32 precision for RMSNorm/LayerNorm layers (1 = true, 0 = false (default)).
  /// This can improve numerical stability for models with normalization layers.
  /// </summary>
  public int ForceRMSNormFP32;

  /// <summary>
  /// Returns default build options.
  /// </summary>
  public static TensorRTBuildOptions Default => new TensorRTBuildOptions
  {
    BuilderOptimizationLevel = 3,
    TilingOptimizationLevel = -1,
    UseSpinWait = 1,
    UseCudaGraphs = 0,
    UseFP16 = 1,
    UseBF16 = 0,
    UseFP8 = 0,
    UseBest = 0,
    MinBatchSize = 0,
    OptBatchSize = 0,
    MaxBatchSize = 0,
    ForceRMSNormFP32 = 0
  };


  /// <summary>
  /// Validates the build options and throws if invalid.
  /// </summary>
  /// <exception cref="ArgumentException">Thrown when options are invalid.</exception>
  public readonly void Validate()
  {
    if (UseFP16 != 0 && UseBF16 != 0)
    {
      throw new ArgumentException("Cannot enable both UseFP16 and UseBF16 simultaneously.");
    }
  }
}
