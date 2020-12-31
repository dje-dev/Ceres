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
    /// <param name="totalNumNodes"></param>
    /// <param name="currentNumNodes"></param>
    /// <param name="overlapInUse"></param>
    /// <param name="dualSelectorsInUse"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="batchSizeMultiplier"></param>
    /// <returns></returns>
    internal static int CalcOptimalBatchSize(int totalNumNodes, int currentNumNodes, 
                        bool overlapInUse, bool dualSelectorsInUse, 
                        int maxBatchSize, float batchSizeMultiplier = 1.0f)
    {
      // Follow pure MCTS for extremely small searches.
      if (totalNumNodes < 10) return 1;

      // If tree size is very small, hand code reasonable batch size
      // such that we don't have many excessively small batches.
      if (currentNumNodes <= 4) return 1;
      if (currentNumNodes <= 7) return 2;
      if (currentNumNodes <= 13) return 3;

      // At larger tree sizes, we have two components,
      // one of which has a low exponent and becomes meaningful only at larger tree sizes.
      // Note that play quality (with small number of nodes per moves, e.g. 5000) 
      // is quite sensitive to changes in these parameters, with larger clearly worse.
      float part1 = 0.2f * MathF.Pow(currentNumNodes, 0.65f);
      float part2 = 2.0f * MathF.Pow(currentNumNodes, 0.45f);

      int value = (int)(part1 + part2);

      if (overlapInUse && !dualSelectorsInUse)
      {
        // If we are overlapping we have to do only 1/2 the batch in each overlap,
        // since the total number in flight will include 2 selectors.
        // The exception is if we are using dual selectors, 
        // in which case we consider this to provide sufficient 
        // collision avoidance so we don't divide in half.
        value /= 2;
      }

      // Never allow batch size to be more than 10% of current number of nodes (or 20% for smaller trees)
      float MAX_FRACTION = currentNumNodes < 100 ? 0.2f : 0.1f; // was 0.10
      value = (int)(batchSizeMultiplier * Math.Min(value, (int)(MAX_FRACTION * currentNumNodes)));

      if (value < 1) // Cannot allow zero!
        return 1;
      else if (value > maxBatchSize)
        return maxBatchSize;
      else
        return value;
    }

  }
}
