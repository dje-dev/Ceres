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

using Ceres.Chess.NNEvaluators;
using Ceres.MCTS.Params;
using System;

#endregion

namespace Ceres.MCTS.Search
{
  internal static class OptimalBatchSizeCalculator
  {
    /// <summary>
    /// Calculates the target batch size to be used given specified 
    /// current search tree characteristics.
    /// 
    /// Large batch sizes have the disadvantage of reducing node selection quality,
    /// i.e. the nodes selected are increasingly away from the optimal CPUCT choices
    /// due to the effect of collisions.
    /// 
    /// Smaller batch sizes have two disdvantages:
    ///   - slower NN evaluation because with very small batch sizes the fraction of time 
    ///     consumed by GPU launch latency is relatively high, and
    ///   - more CPU overhead because the opportunity for parallelism in selecting the batch members is reduced
    ///     because parallelism overhead is prohibitively high with small concurrent sub-batch selection
    /// </summary>
    /// <param name="estimatedTotalSearchNodes"></param>
    /// <param name="currentTreeSize"></param>
    /// <param name="overlapInUse"></param>
    /// <param name="dualSelectorsInUse"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="batchSizeMultiplier"></param>
    /// <param name="paramsSearch"></param>
    /// <returns></returns>
    internal static int CalcOptimalBatchSize(int estimatedTotalSearchNodes, int currentTreeSize,
                                             bool overlapInUse, bool dualSelectorsInUse,
                                             int maxBatchSize, float batchSizeMultiplier,
                                             ParamsSearch paramsSearch)
    {
      // Follow pure MCTS for extremely small searches.
      if (estimatedTotalSearchNodes < 10) return 1;

      // If tree size is very small, hand code reasonable batch size
      // such that we don't have many excessively small batches.
      switch (currentTreeSize)
      {
        case <= 4:
          return 1;
        case <= 7:
          return 2;
        case <= 13:
          return 3;
      }


      const float MULT = 3.2f;
      const float POW = 0.45f;
      int value = (int)(MULT * MathF.Pow(currentTreeSize, POW));

      if (overlapInUse)
      {
        // If we are overlapping we have two batches in flight at once,
        // since the total number in flight will include 2 selectors.
        // Since there will be some collisions across batches,
        // we need to adjust downward the batch size.
        if (dualSelectorsInUse)
        {
          // Only modestly decrease since the dual selectors 
          // will have different CPUCTs and therefore not totally compete.
          value = (int)(value * 0.80f);
        }
        else
        {
          // Reduce batch size to larger degree since two overlapped selectors diretly competing.
          value = (int)(value * 0.60f);
        }
      }

      // Never allow batch size to be more than 10% of current number of nodes (or 20% for smaller trees)
      float MAX_FRACTION = currentTreeSize < 100 ? 0.2f : 0.1f; // was 0.10
      value = (int)(batchSizeMultiplier * Math.Min(value, (int)(MAX_FRACTION * currentTreeSize)));

      if (value < 1) // Cannot allow zero!
      {
        return 1;
      }
      else if (value > maxBatchSize)
      {
        return maxBatchSize;
      }
      else
      {
        return value;
      }
    }

  }
}
