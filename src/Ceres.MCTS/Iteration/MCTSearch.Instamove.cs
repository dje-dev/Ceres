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

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Managers;
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Internal helper method for MCTSearch relating to instant move handling.
  /// </summary>
  public partial class MCTSearch
  {
    /// <summary>
    /// Returns instant move that should be made, if any, due to either:
    ///   - best move is obvious (and search tree is already of a reasonable size)
    ///   - tree is already very big (from reuse)
    /// </summary>
    /// <param name="priorManager"></param>
    /// <param name="newRoot"></param>
    /// <returns></returns>
    (MCTSManager, MGMove, TimingStats) CheckInstamove(MCTSManager priorManager, 
                                                      SearchLimit searchLimit, 
                                                      MCTSNode newRoot)
    {
      if (newRoot != null
        && priorManager != null
        && newRoot.N > 100
        && newRoot.NumChildrenExpanded > 1
        && priorManager.Context.ParamsSearch.FutilityPruningStopSearchEnabled
        && priorManager.Context.ParamsSearch.EnableInstamoves)
      {
        // Don't make too many instamoves in a row because
        // time control state has been improved by the instamoves, 
        // so further search may now be warranted.
        // (Also, for some reason, humans tend to find it annoying/suspicious!).
        const int MAX_CONSECUTIVE_INSTAMOVES = 3;
        if (CountSearchContinuations >= MAX_CONSECUTIVE_INSTAMOVES)
          return default;

        if (double.IsNaN(priorManager.EstimatedNPS)) return default;

        MCTSNode lastSearchRoot = priorManager.Root;
        float baselineTreeSize = lastSearchRoot.N;

        int estNewVisitsThisMove = searchLimit.EstNumNodes((int)priorManager.EstimatedNPS, true);
        float ratioNewToBaseline = ((float)newRoot.N + (float)estNewVisitsThisMove) / (float)baselineTreeSize;
        float thresholdRatioNewToCurrent = 1.4f - (0.10f * CountSearchContinuations);
        bool treeIsBigEnough = ratioNewToBaseline < thresholdRatioNewToCurrent;
        if (!treeIsBigEnough) return default;

        // Possibly veto the instamove if there is a second-best move that could
        // catch up to best move if the planned search were conducted.
        BestMoveInfo bestMoveInfo = newRoot.BestMoveInfo(false);
        if (bestMoveInfo.BestMove.IsNull) return default;

        MCTSNode[] childrenSortedQ = newRoot.ChildrenSorted(n => (float)n.Q);
        MCTSNode[] childrenSortedN = newRoot.ChildrenSorted(n => -n.N);
        float nGap = Math.Abs(childrenSortedQ[0].N - childrenSortedQ[1].N);
        float qGap = (float)(Math.Abs(childrenSortedQ[0].Q - childrenSortedQ[1].Q));
        //float biggestN = Math.Max(bestMoveInfo.BestN, bestMoveInfo.BestNSecond);
        float minNRequiredToChange = nGap / ManagerChooseRootMove.MinFractionNToUseQ(newRoot, qGap);
        bool couldCatchUp = (estNewVisitsThisMove * 0.25f) > minNRequiredToChange;
        if (couldCatchUp)
          MCTSEventSource.TestCounter1++;
        else
          MCTSEventSource.TestMetric1++;

        if (couldCatchUp) return default;
        //Console.WriteLine("catchup " + couldCatchUp + " visits: " + estNewVisitsThisMove + " " + minNRequiredToChange);

        // Log.Info($"\r\nInstamove already {CountSearchContinuations} sureness {surenessFactor} " +
        //          $"Tree size factor {treeSizeFactor} N={bestMoveInfo.BestN} N2={bestMoveInfo.BestNSecond}");
        InstamoveCount++;
        CountSearchContinuations++;

        return (priorManager, newRoot.BestMoveInfo(false).BestMove, null);        
      }
      return default;
    }

  }
}
