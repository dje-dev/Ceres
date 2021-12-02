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
using System.Runtime.CompilerServices;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Iteration;

#endregion

namespace Ceres.MCTS.NNEvaluators
{
  internal class NNEvaluatorComboPhased
  {
    /// <summary>
    /// Installs "COMBO_PHASED" as custom network type which is an NNEvaluatorDynamic
    /// and uses one network for first part of search then switches to second network
    /// after a specified fraction of tree search is complete.
    /// 
    /// The string consists of two network IDs separated by semicolons,
    /// where the first network string optionally ends "%" followed by a  
    /// phase transition fraction (default 0.5 if not spsecified).

    /// Example usage (also customized nets and fraction below):
    ///   "COMBO_PHASED:69637%0.4;703810"
    /// </summary>
    static void InstallComboPhasedFactory()
    {
      static NNEvaluator Builder(string netID1, int gpuID, NNEvaluator referenceEvaluator)
      {
        string[] netIDs = netID1.Split(";");
        if (netIDs.Length != 2)
        {
          throw new NotImplementedException("COMBO_PHASED evaluators must consist of two comma separated network IDs");
        }

        float baseSplitFraction = 0.5f;
        if (netIDs[0].Contains("%"))
        {
          string[] netThenFraction = netIDs[0].Split("%");
          netIDs[0] = netThenFraction[0];
          baseSplitFraction = float.Parse(netThenFraction[1]);
        }

        // Construct a compound evaluator which does both fast and slow (potentially parallel)
        NNEvaluator[] evaluators = new NNEvaluator[] {NNEvaluator.FromSpecification(netIDs[0], $"GPU:{gpuID}"),
                                                      NNEvaluator.FromSpecification(netIDs[1], $"GPU:{gpuID}")};

        NNEvaluatorDynamic dyn = new NNEvaluatorDynamic(evaluators,
          delegate (IEncodedPositionBatchFlat batch)
          {
            batch.PositionsUseSecondaryEvaluator = false; // default assumption

            // Return 0 if not yet initialized (happens only during evaluator during warmup phase).
            if (MCTSManager.ThreadSearchContext == null)
            {
              MCTSManager.NumSecondaryBatches++;
              MCTSManager.NumSecondaryEvaluations += batch.NumPos;
              batch.PositionsUseSecondaryEvaluator = true;
              return 0;
            }

            MCTSIterator manager = MCTSManager.ThreadSearchContext;
            float thisSplitFraction = baseSplitFraction;

            if (manager.Manager.RootNWhenSearchStarted == 0 
             && manager.Manager.FractionSearchCompleted < baseSplitFraction)
              {
                MCTSManager.NumSecondaryBatches++;
                MCTSManager.NumSecondaryEvaluations += batch.NumPos;
                batch.PositionsUseSecondaryEvaluator = true;
                return 0;
              }
              else
              {
                // Note that secondary net never used when in tree reuse mode
                // because tree is already mostly formed.
                // However the reuse manager may be more aggressive in restarting 
                // searches from scratch in this mode.
                return 1;
              }
           
          });

        return dyn;
      }

      NNEvaluatorFactory.ComboPhasedFactory = Builder;
    }


    [ModuleInitializer]
    public static void Install()
    {
      InstallComboPhasedFactory();
    }
  }
}
