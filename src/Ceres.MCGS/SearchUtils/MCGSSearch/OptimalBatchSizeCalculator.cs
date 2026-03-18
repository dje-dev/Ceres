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
  /// <param name="dualSelectorsInUse"></param>
  /// <param name="maxBatchSize"></param>
  /// <param name="batchSizeMultiplier"></param>
  /// <param name="paramsSearch"></param>
  /// <returns></returns>
  public static int CalcOptimalBatchSize(int estimatedTotalSearchNodes, int currentGraphSize,
                                           bool overlapInUse,
                                           int maxBatchSize, float batchSizeMultiplier,
                                           bool enableEarlySmallBatchSizes)
  {
    if (OverrideBatchSizeFunc is not null)
    {
      return OverrideBatchSizeFunc(currentGraphSize, maxBatchSize);
    }

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
    value = (int)(batchSizeMultiplier * Math.Min(value, (int)(MAX_FRACTION * currentGraphSize)));

    return Math.Clamp(value, 1, maxBatchSize);    
  }
}
