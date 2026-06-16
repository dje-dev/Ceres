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

#endregion

namespace Ceres.MCGS.Search;

public static class OptimalBatchSizeCalculator
{
  /// <summary>
  /// Override logic for batch size calculations.
  /// </summary>
  public delegate int BatchSizeOverride(int currentTreeSize, int maxBatchSize);

  /// <summary>
  /// Optional delegate to override logic in CalcOptimalBatchSize.
  /// </summary>
  public static BatchSizeOverride OverrideBatchSizeFunc;


  /// <summary>
  /// Calculates the target batch size to be used given specified 
  /// current search tree characteristics.
  /// 
  /// Large batch sizes have the disadvantage of reducing node selection quality,
  /// i.e. the nodes selected are increasingly away from the optimal CPUCT choices
  /// due to the effect of collisions.
  /// 
  /// Smaller batch sizes have two disadvantages:
  ///   - slower NN evaluation because with very small batch sizes the fraction of time 
  ///     consumed by GPU launch latency is relatively high, and
  ///   - more CPU overhead because the opportunity for parallelism in selecting the batch members is reduced
  ///     because parallelism overhead is prohibitively high with small concurrent sub-batch selection
  /// </summary>
  /// <param name="estimatedTotalSearchNodes"></param>
  /// <param name="currentGraphSize"></param>
  /// <param name="overlapInUse"></param>
  /// <param name="maxBatchSize"></param>
  /// <param name="batchSizeMultiplier"></param>
  /// <param name="enableEarlySmallBatchSizes"></param>
  /// <param name="numDevicesInEvaluator">Number of compute devices the evaluator spans (raises the cap by 128 per extra device).</param>
  /// <param name="netFileSizeBytes">Size in bytes of the network file (&lt;= 0 if unknown); larger nets get a smaller cap.</param>
  /// <returns></returns>
  public static int CalcOptimalBatchSize(int estimatedTotalSearchNodes, int currentGraphSize,
                                         bool overlapInUse,
                                         int maxBatchSize, float batchSizeMultiplier,
                                         bool enableEarlySmallBatchSizes,
                                         int numDevicesInEvaluator, long netFileSizeBytes)
  {
    if (OverrideBatchSizeFunc is not null)
    {
      return OverrideBatchSizeFunc(currentGraphSize, maxBatchSize);
    }

    // Cap maximum batch size based on network file size and device count.
    // Try to keep batch size small (for better selection quality)
    // but allow larger for smaller nets or more devices where this improves backend eps significantly
    const long NET_FILE_SIZE_THRESHOLD_BIG = 100_000_000;
    const int BATCH_SIZE_BIG = 768;
    const int BATCH_SIZE_MAX_DEFAULT = 512;
    const int BATCH_SIZE_INCREMENT_PER_DEVICE = 128;  
    int baseMax = (netFileSizeBytes > 0 && netFileSizeBytes <= NET_FILE_SIZE_THRESHOLD_BIG) ? BATCH_SIZE_BIG : BATCH_SIZE_MAX_DEFAULT;
    int deviceCount = numDevicesInEvaluator > 0 ? numDevicesInEvaluator : 1;
    maxBatchSize = Math.Min(maxBatchSize, baseMax + BATCH_SIZE_INCREMENT_PER_DEVICE * (deviceCount - 1));
    
    // Use batch size of 1 for extremely small searches.
    if (estimatedTotalSearchNodes < 10)
    {
      return 1;
    }

    // If tree size is very small, hand code reasonable batch size
    // such that we don't have many excessively small batches.
    // TODO: We should almost certainly be less aggressive at small node counts!
    switch (currentGraphSize)
    {
      case <= 4:
        return 1;
      case <= 7:
        return 2;
      case <= 13:
        return 3;
    }


    // Coefficients that determine maximum batch size as a function of current graph size.
    // Note that batch sizes are very sensitive to these coefficients.
    const float MULT = 3.2f;
    const float POW = 0.45f;
    int value = (int)(MULT * MathF.Pow(currentGraphSize, POW));

    if (overlapInUse)
    {
      // If we are overlapping we have two batches in flight at once,
      // since the total number in flight will include 2 selectors.
      // Since there will be some collisions across batches,
      // we need to adjust downward the batch size.
      value = (int)(value * 0.80f);
    }

    // Never allow batch size to be more than 10% of current number of nodes (or 20% for smaller trees)
    float MAX_FRACTION = currentGraphSize < 100 ? 0.2f : 0.1f; // was 0.10

    const float EXTRA_MULTPLIER = 1.2f; // made slightly more aggressive June 2026
    value = (int)(batchSizeMultiplier * Math.Min(EXTRA_MULTPLIER * value, (int)(MAX_FRACTION * currentGraphSize)));

    return Math.Clamp(value, 1, maxBatchSize);    
  }
}
