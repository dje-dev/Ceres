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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Ceres.Base.DataTypes;
using Ceres.Base.Environment;
using Ceres.Base.Threading;
using Ceres.Chess.Diagnostics;
using Ceres.Chess.PositionEvalCaching;
using Ceres.MCTS.Environment;
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Leaf evaluator that consults a PositionEvalCache to 
  /// try to return an already evaluated and stored result.
  /// </summary>
  public sealed class LeafEvaluatorCache : LeafEvaluatorBase
  {
    #region Statistics tracking

    internal static AccumulatorMultithreaded NumHits;
    internal static AccumulatorMultithreaded NumMisses;
    internal static AccumulatorMultithreaded NumHitsOldGeneration;

    public static float HitRatePct => 100.0f * (float)NumHits.Value / (float)(NumHits.Value + NumMisses.Value);

    #endregion

    /// <summary>
    /// Underlying cache containing evaluation information.
    /// </summary>
    public readonly PositionEvalCache Cache;


    /// <summary>
    /// Constructor for a cache built on top of specified PositionEvalCache.
    /// </summary>
    /// <param name="cache"></param>
    public LeafEvaluatorCache(PositionEvalCache cache)
    {
      Cache = cache;
    }


    /// <summary>
    /// Virtual worker method that attempts to resolve the evaluation of a specified node from the cache.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    protected override LeafEvaluationResult DoTryEvaluate(MCTSNode node)
    {
      PositionEvalCacheEntry cacheEntry = default;
      bool inCache = Cache.TryLookupFromHash(node.StructRef.ZobristHash, ref cacheEntry);

      if (inCache)
      {
        if (CeresEnvironment.MONITORING_METRICS)
        {
          NumHits.Add(1, node.Index);
          //MCTSEventSource.TestMetric1++;

          if (node.StructRef.IsOldGeneration)
          {
            NumHitsOldGeneration.Add(1, node.Index);
          }
        }

        Debug.Assert(!float.IsNaN(cacheEntry.WinP + cacheEntry.LossP));

        LeafEvaluationResult result = new LeafEvaluationResult(cacheEntry.TerminalStatus, cacheEntry.WinP, cacheEntry.LossP, cacheEntry.M);
        result.PolicySingle = cacheEntry.Policy;
        return result;
      }
      else
      {
        if (CeresEnvironment.MONITORING_METRICS)
        {
          NumMisses.Add(1, node.Index);
        }

        return default;
      }
    }


    [ModuleInitializer]
    internal static void ModuleInitialize()
    {
      NumHits.Initialize();
      NumMisses.Initialize();
    }

  }
}
