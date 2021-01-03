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
using Ceres.Base.Math;
using Ceres.MCTS.Managers;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Manages identification of moves at root which can have
  /// further search suspended due to futility - impossibility
  /// or improbability of further visits changing the final selected best move.
  /// 
  /// Pruning of primary and/or secondary moves is possible
  /// (where pruning of primary moves effectively shuts down the whole search).
  /// </summary>
  public class MCTSFutilityPruning
  {
    /// <summary>
    /// Associated search manager.
    /// </summary>
    public readonly MCTSManager Manager;

    /// <summary>
    /// Associated context.
    /// </summary>
    public MCTSIterator Context => Manager.Context;

    /// <summary>
    /// Helper method which returns root node of search.
    /// </summary>
    public MCTSNode Root => Context.Root;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="manager"></param>
    /// <param name="context"></param>
    public MCTSFutilityPruning(MCTSManager manager, MCTSIterator context)
    {
      Manager = manager;
    }


    /// <summary>
    /// Updates internal pruning statistics based on current state of search tree.
    /// </summary>
    internal void UpdatePruningFlags()
    {
      // Exit if early stopping not enabled
      if (Context.ParamsSearch.MoveFutilityPruningAggressiveness == 0f) return;

      // Because estimates of nodes remaining are noisy for searces with time limits,
      // we conservatively decline to set MinNToVisit unless we are 
      // reasonably close to the end of the search (and thus had sufficient time to get accurate statistics)
      if (Manager.FractionSearchRemaining > 0.25) return;
      if (Manager.NumStepsTakenThisSearch < 100) return;

      if (Context.RootMovesArePruned == null) Context.RootMovesArePruned = new bool[Root.NumPolicyMoves];

      DoSetEarlyStopMoveSecondaryFlags();
     }


    /// <summary>
    /// Worker method that actually computes and updates the statistic.
    /// </summary>
    private void DoSetEarlyStopMoveSecondaryFlags()
    {
      if (Manager.Root.NumChildrenExpanded == 0) return;

      float aggressiveness = Context.ParamsSearch.MoveFutilityPruningAggressiveness;
      if (aggressiveness > 1.0f) throw new Exception("Ceres configuration error: maximum value of EarlyStopMoveSecondaryAggressiveness is 1.0.");

      float MIN_BEST_N_FRAC_REQUIRED = ManagerChooseRootMove.MIN_FRAC_N_REQUIRED_MIN(Context);

      // Calibrate aggressiveness such that :
      //   - at maximum value of 1.0 we assume 50% visits go to second best move(s)
      //   - at reasonable default value 0.5 we assume 75% of visits to go second best move(s) 
      float aggressivenessMultiplier = 1.0f / (1.0f - aggressiveness * 0.6f);

      int? numRemainingSteps = Manager.EstimatedNumVisitsRemaining();

      // Can't make any determiniation if we can't estimate how many steps left
      if (numRemainingSteps is null) return;

      int numStepsTotal = (int)(numRemainingSteps / Manager.FractionSearchRemaining);

      MCTSNode[] nodesSortedN = Root.ChildrenSorted(n => -n.N);
      MCTSNode bestNNode = nodesSortedN[0];

      int minN = Root.N / 5;// (int)(bestQNode.N * MIN_BEST_N_FRAC_REQUIRED);
      MCTSNode[] nodesSortedQ = Root.ChildrenSorted(n => n.N < minN ? int.MaxValue : (float)n.Q);
      MCTSNode bestQNode = nodesSortedQ[0];

      float bestQ = (float)bestQNode.Q;


      ManagerChooseRootMove bestMoveChoser = new(Context.Root, false, Context.ParamsSearch.MLHBonusFactor);
      Span<MCTSNodeStructChild> children = Root.Ref.Children;
      for (int i = 0; i < Root.Ref.NumChildrenExpanded; i++)
      {
        // Never shut down second best move unless the whole search is eligible to shut down
        if (nodesSortedN.Length > 1)
        {
          MCTSNode secondBestMove = Context.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopN ? nodesSortedN[1] : nodesSortedQ[1];
          bool isSecondBestMove = children[i].Move == secondBestMove.PriorMove;
          if (isSecondBestMove && !Context.ParamsSearch.FutilityPruningStopSearchEnabled)
          {
            Context.RootMovesArePruned[i] = false;
            continue;
          }
        }

        float earlyStopGapRaw;
        if (Context.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopN)
        {
          earlyStopGapRaw = nodesSortedN[0].N - children[i].N;
        }
        else
        {
          int minNRequired = (int)(bestQNode.N * MIN_BEST_N_FRAC_REQUIRED);
          earlyStopGapRaw = minNRequired - children[i].N;
        }

        float earlyStopGapAjusted = earlyStopGapRaw * aggressivenessMultiplier;
        bool earlyStopSimple = earlyStopGapRaw > numRemainingSteps;

        if (MCTSDiagnostics.DumpSearchFutilityShutdown
          && earlyStopSimple
          && !Context.RootMovesArePruned[i]
          && NumberOfNotShutdownChildren() < 3)
        {
          Console.WriteLine();
          Console.WriteLine($"\r\nShutdown {children[i].Move} [{children[i].N}] at root N  {Context.Root.N} with remaning {numRemainingSteps}"
                          + $" due to raw gapN {earlyStopGapRaw} adusted to {earlyStopGapAjusted} in mode {Context.ParamsSearch.BestMoveMode} aggmult {aggressivenessMultiplier}");
          DumpDiagnosticsMoveShutdown();
        }

        Context.RootMovesArePruned[i] = earlyStopSimple;
        // Console.WriteLine(i + $" EarlyStopMoveSecondary(simple) gap={gapToBest} adjustedGap={inflatedGap} remaining={numRemainingSteps} ");
      }


      // TODO: log this
      //Console.WriteLine($"{Context.RemainingTime,5:F2}sec remains at N={Root.N}, setting  minN to {minN} number still considered {count} " +
      //                  $"using EstimatedNPS {Context.EstimatedNPS} with nodes remaining {Context.EstimatedNumStepsRemaining()} " +
      //                  statsStr);

    }


    /// <summary>
    /// Dumps diagnostic information related to the futiltiy statistics.
    /// </summary>
    public void DumpDiagnosticsMoveShutdown()
    {
      Manager.DumpTimeInfo();
      Context.Root.Dump(1, 1);
      Console.WriteLine();
    }


    /// <summary>
    /// Returns the number of children at root which are not in a shutdown state.
    /// </summary>
    /// <returns></returns>
    public int NumberOfNotShutdownChildren()
    {
      if (Context.RootMovesArePruned == null) return Context.Root.NumPolicyMoves;

      int count = 0;
      for (int i = 0; i < Root.NumPolicyMoves; i++)
      {
        if (!Context.RootMovesArePruned[i])
          count++;
      }

      return count;
    }


#if EXPERIMENTAL    
    private bool PossiblyEarlyStopMoveSecondary(int childIndex, in MCTSNodeStruct child, int numRemainingSteps,
                                                MCTSNode bestMove, float qAdjustment)
    {
      if (Root.N > 500)
      {
        if (Context.ParamsSelect.CPUCT2 != 0) throw new NotImplementedException(); // need to add to NumVisitsToEqualize method

        int neededSteps = VisitsToEqualizeCalculator.NumVisitsToEqualize(
                 Context.ParamsSelect.UCTNonRootNumeratorExponent,
                 Context.ParamsSelect.UCTRootDenominatorExponent,
                 Root.N,
                 bestMove.P, (float)-bestMove.Q, bestMove.N,
                 child.P, (float)-child.Q, child.N,
                 StatUtils.Bounded(-(float)bestMove.Q - qAdjustment, -1, 1),
                 StatUtils.Bounded(-(float)child.Q + qAdjustment, -1, 1));

        return neededSteps > numRemainingSteps;
      }
      else
        return false;
    }

    /// <summary>
    /// Number of visits to a node required to make it catch up to a specified Q value,
    /// given an assume value of Q to be repeated on each visit.
    /// </summary>
    /// <param name="qNewVisits"></param>
    /// <param name="nStart"></param>
    /// <param name="qStart"></param>
    /// <param name="qTarget"></param>
    /// <returns></returns>
    static int NumVisitsToCatchUp(float qNewVisits, float nStart, float qStart, float qTarget)
    {
      if (qStart >= qTarget)
        return 0;
      else if (qNewVisits < qTarget)
        return int.MaxValue; // no way to catch up
      else
      {

        double num = (nStart * qStart) - (nStart * qTarget);
        double den = qTarget - qNewVisits;
        return (int)(num / den);
      }
    }
#endif

  }
}


