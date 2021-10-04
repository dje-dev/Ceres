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
using Ceres.Chess.PositionEvalCaching;
using Ceres.Base;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Iteration;
using Ceres.Base.Benchmarking;


#endregion

namespace Ceres.MCTS.Search.IteratedMCTS
{
  /// <summary>
  /// Manages running an "iterated" MCTS search which runs multiple iterations
  /// starting from the root with a small amount of information carried over from the prior iteration
  /// (bending the NN policy priors toward the emprical visit distribution).
  /// </summary>
  /// 
#if COMMENT
  Promising feature because:
    1. It has potential to reduce memory consumption,
    2. It could be useful in building "opening book"
    3. It may improve play quality, but unclear:

       In match Ceres vs Ceres (T60, 10 seconds per move, no contempt)
       loses by 10 Elo (0/104/3) with this IMCTSSchedule
       (and yet in suite tests this IMCTSSchedule seems to help)
       // -10    6     4    107    6:13:30   600   193   441.64 442.36     18,812,157    9,620,575     88     Draw  Ceres_2    0 104   3   
      tournamentDef.searchParams2.IMCTSSchedule =
                               new IteratedMCTSDef(IteratedMCTSDef.TreeModificationType.ClearNodeVisits,
                               new IteratedMCTSStepDef(0.75f, 0.15f, 0.75f),
                               new IteratedMCTSStepDef(0.25f, float.NaN, float.NaN));
#endif

#if COMMENT
    Somewhat related research idea is to smoothly transition from MCTS averaging to min-max
    by using Weight and EffectiveN fields at node structure and
      - scaling P by Weight (toward 0 which is no exploration), and
      - periodically lowering weight on nodes that are markedly inferior to best sibling,
        and backing out their impact on predecessors (back out some W and reduce effective N multiplicatively up tree)
    Not yet fully implemented or tested
#endif

  public class IteratedMCTSSearchManager
  {
    /// <summary>
    /// Runs all iterations of the iterated search.
    /// </summary>
    /// <param name="manager"></param>
    /// <param name="progressCallback"></param>
    /// <param name="iterationsDefinition"></param>
    /// <returns></returns>
    public (TimingStats, MCTSNode) IteratedSearch(MCTSManager manager, 
                                                  MCTSManager.MCTSProgressCallback progressCallback, 
                                                  IteratedMCTSDef iterationsDefinition)
    {
      TimingStats fullSearchTimingStats = new TimingStats();
      MCTSNode fullSearchNode = default;

      using (new TimingBlock(fullSearchTimingStats, TimingBlock.LoggingType.None))
      {
        iterationsDefinition.SetForSearchLimit(manager.SearchLimit);

        // Temporarily disable the primary/secondary pruning aggressivenss so we get pure policy distribution
        bool saveEarlyStop = manager.Context.ParamsSearch.FutilityPruningStopSearchEnabled;
        float saveSecondaryAgg = manager.Context.ParamsSearch.MoveFutilityPruningAggressiveness;
        manager.Context.ParamsSearch.FutilityPruningStopSearchEnabled = false;
        manager.Context.ParamsSearch.MoveFutilityPruningAggressiveness = 0;

        string cacheFileName = $"Ceres.imcts_{DateTime.Now.Ticks}_.cache";

        // Loop thru the steps (except last one)
        PositionEvalCache lastCache = null;
        for (int step = 0; step < iterationsDefinition.StepDefs.Length - 1; step++)
        {
          IteratedMCTSStepDef thisStep = iterationsDefinition.StepDefs[step];

          // On the second and subsequent steps configure so that we reuse the case saved from prior iteration
          if (step > 0 && iterationsDefinition.TreeModification == IteratedMCTSDef.TreeModificationType.DeleteNodesMoveToCache)
          {
            manager.Context.EvaluatorDef.CacheMode = PositionEvalCache.CacheMode.MemoryAndDisk;
            manager.Context.EvaluatorDef.CacheFileName = null;
            manager.Context.EvaluatorDef.PreloadedCache = lastCache;
          }

          // Set the search limit as requested for this step
          manager.SearchLimit = thisStep.Limit;

          // Run this step of the search (disable progress callback)
          (TimingStats, MCTSNode) stepResult = manager.DoSearch(thisStep.Limit, null);

          // Extract a cache with a small subset of nodes with largest N and a blended policy
          int minN = (int)(thisStep.NodeNFractionCutoff * manager.Root.N);
          if (minN < 100) minN = int.MaxValue; // do not modify policies on very small trees
          lastCache = IteratedMCTSBlending.ModifyNodeP(manager.Context.Root, minN, thisStep.WeightFractionNewPolicy, iterationsDefinition.TreeModification);
          //cache.SaveToDisk(cacheFileName);

          if (iterationsDefinition.TreeModification == IteratedMCTSDef.TreeModificationType.ClearNodeVisits)
          {
            const bool MATERIALIZE_TRANSPOSITIONS = true; // ** TODO: can we safely remove this?
            manager.ResetTreeState(MATERIALIZE_TRANSPOSITIONS);
          }
        }

        // Restore original pruning aggressiveness
        manager.Context.ParamsSearch.FutilityPruningStopSearchEnabled = saveEarlyStop;
        manager.Context.ParamsSearch.MoveFutilityPruningAggressiveness = saveSecondaryAgg;
        manager.SearchLimit = iterationsDefinition.StepDefs[^1].Limit; // TODO: duplicated with the next call?

        (TimingStats, MCTSNode) finalResult = manager.DoSearch(iterationsDefinition.StepDefs[^1].Limit, progressCallback);
        fullSearchNode = finalResult.Item2;
      }

      return (fullSearchTimingStats, fullSearchNode);
    }

  }
}
