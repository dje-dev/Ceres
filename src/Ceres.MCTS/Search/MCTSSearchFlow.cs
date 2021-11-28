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
using Ceres.Chess.NNEvaluators;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;

#endregion

namespace Ceres.MCTS.Search
{
  public partial class MCTSSearchFlow
  {
    public readonly MCTSManager Manager;
    public readonly MCTSIterator Context;

    public readonly MCTSNNEvaluator BlockNNEval1;
    public readonly MCTSNNEvaluator BlockNNEval2;

    /// <summary>
    /// Optional evaluator used as a secondary evaluator (experimental).
    /// </summary>
    public readonly MCTSNNEvaluator BlockNNEvalSecondaryNet;

    public readonly MCTSApply BlockApply;


    MCTSBatchParamsManager[] batchingManagers;

    public MCTSSearchFlow(MCTSManager manager, MCTSIterator context)
    {
      Manager = manager;
      Context = context;

      int numSelectors = context.ParamsSearch.Execution.FlowDirectOverlapped ? 2 : 1;
      batchingManagers = new MCTSBatchParamsManager[numSelectors];

      for (int i = 0; i < numSelectors; i++)
      {
        batchingManagers[i] = new MCTSBatchParamsManager(manager.Context.ParamsSelect.UseDynamicVLoss);
      }

      bool shouldCache = context.EvaluatorDef.CacheMode != Chess.PositionEvalCaching.PositionEvalCache.CacheMode.None;

      string instanceID = "0";

      const bool LOW_PRIORITY_PRIMARY = false;

      LeafEvaluatorNN nodeEvaluatorSecondary = null;
      if (context.EvaluatorDefSecondary != null)
      {
        const bool CACHE_SECONDARY = false; // TODO: support this someday (in its own cache)?
        nodeEvaluatorSecondary = new LeafEvaluatorNN(context.EvaluatorDef,
                                                     context.NNEvaluators.EvaluatorSecondary,
                                                     CACHE_SECONDARY, false, null, null, null);
        BlockNNEvalSecondaryNet = new MCTSNNEvaluator(nodeEvaluatorSecondary, false);
      }

      // If a secondary evaluator is configured
      // and the definition requests early nodes in the search to use this evaluator,
      // then create the LeafEvaluatorNN to also contain this secondary evaluator.
      // TODO: one evaluator is shared (not overlapped), is this a significant performance drag?
      NNEvaluator evaluatorSecondaryToUseInPrimaryEvaluators = null;
      if (Context.ParamsSearch.ParamsSecondaryEvaluator?.InitialTreeNodesForceSecondary > 0)
      {
        if (nodeEvaluatorSecondary == null)
        {
          throw new Exception("Secondary network must be specified if ParamsSecondaryEvaluator are defined.");
        }

        evaluatorSecondaryToUseInPrimaryEvaluators = context.NNEvaluators.EvaluatorSecondary;
      }

      //Params.Evaluator1 : Params.Evaluator2;
      LeafEvaluatorNN nodeEvaluator1 = new LeafEvaluatorNN(context.EvaluatorDef, context.NNEvaluators.Evaluator1, shouldCache,
                                                           LOW_PRIORITY_PRIMARY, context.Tree.PositionCache, null, evaluatorSecondaryToUseInPrimaryEvaluators);
      BlockNNEval1 = new MCTSNNEvaluator(nodeEvaluator1, true);

      if (context.ParamsSearch.Execution.FlowDirectOverlapped)
      {
        // Create a second evaluator (configured like the first) on which to do overlapping.
        LeafEvaluatorNN nodeEvaluator2 = new LeafEvaluatorNN(context.EvaluatorDef, context.NNEvaluators.Evaluator2, shouldCache,
                                                             false, context.Tree.PositionCache, null, evaluatorSecondaryToUseInPrimaryEvaluators);
        BlockNNEval2 = new MCTSNNEvaluator(nodeEvaluator2, true);
      }

      BlockApply = new MCTSApply(context.FirstMoveSampler);

      // TODO: Reconsider this. Current dynamic statistics calculation is disabled because:
      //         - it slows startup
      //         - in the current implementation, it  triggers both executors to initialize even if only 1 needed
      //         - it is noisy to estimate on the fly
      //         - it probably does not interact well with the fixed sized batches of graphs
      const bool ENABLE_CALC_STATS = false;
      if (ENABLE_CALC_STATS && context.ParamsSearch.Execution.SmartSizeBatches)
      {
        context.NNEvaluators.CalcStatistics(true, 1f);
      }
    }

  }
}

