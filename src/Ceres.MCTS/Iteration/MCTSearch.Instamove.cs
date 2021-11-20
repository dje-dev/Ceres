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

using Ceres.Chess;
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
    bool CheckInstamove(MCTSManager priorManager, SearchLimit searchLimit, 
                        MCTSNode newRoot, ManagerTreeReuse.Method reuseMethod)
    {    
      // Do quick checks to see if instamove not possible/desirable.
      if (newRoot.IsNull
       || priorManager == null
       || newRoot.N <= 100
       || newRoot.NumChildrenExpanded <= 1)
      {
        return false;
      }

      // Never instamove when tablebase available
      // (because complex logic with contextual knowledge
      // needed to avoid falling into draw by repetitions, etc.)
      (GameResult result, Chess.MoveGen.MGMove immediateMove) = priorManager.TryGetTablebaseImmediateMove(newRoot);
      if (result != GameResult.Unknown)
      {
        return false;
      }

      if (reuseMethod == ManagerTreeReuse.Method.ForceInstamove
       || CheckInstamoveFutility(priorManager, searchLimit, newRoot, reuseMethod))
      {
        InstamoveCount++;
        CountSearchContinuations++;

        return true;
      }
      else
      {
        return false;
      }

    }


    const bool VERBOSE = false;

    bool CheckInstamoveFutility(MCTSManager priorManager,
                                SearchLimit searchLimit,
                                MCTSNode newRoot,
                                ManagerTreeReuse.Method reuseMethodIfNoInstamove)
    {
      if (!priorManager.Context.ParamsSearch.FutilityPruningStopSearchEnabled
        ||!priorManager.Context.ParamsSearch.EnableInstamoves)
      {
        return false;
      }

      // Don't make too many instamoves in a row because
      // time control state has been improved by the instamoves, 
      // so further search may now be warranted.
      // (Also, for some reason, humans tend to find it annoying/suspicious!).
      const int MAX_CONSECUTIVE_INSTAMOVES = 2;
      if (CountSearchContinuations >= MAX_CONSECUTIVE_INSTAMOVES)
      {
        return false;
      }

      if (double.IsNaN(priorManager.EstimatedNPS))
      {
        return false;
      }

      float nRatioNewRootVersusLastSearchTree = (float)newRoot.N / priorManager.Root.N;
      float nRatioThreshold;
      if (reuseMethodIfNoInstamove == ManagerTreeReuse.Method.KeepStoreSwapRoot)
      {
        // Swap root is inexpensive so be less inclined to instamove
        nRatioThreshold = 0.5f;
      }
      else
      {
        nRatioThreshold = 0.3f;
      }

      if (nRatioNewRootVersusLastSearchTree < nRatioThreshold)
      {
        if (VERBOSE) Console.WriteLine("nRatioNewRootVersusLastSearchTree " + nRatioNewRootVersusLastSearchTree);
        return false;
      }

//      float ratioNewToCurrent = ((float)newRoot.N + (float)estNewVisitsThisMove) / (float)newRoot.N;
//      float thresholdRatioNewToCurrent = 1.4f - (0.10f * CountSearchContinuations);
//      bool treeIsBigEnough = ratioNewToCurrent < thresholdRatioNewToCurrent;
//      if (!treeIsBigEnough)
//      {
//        return false;
//      }

      BestMoveInfo bestMoveInfo = newRoot.BestMoveInfo(false);
      if (bestMoveInfo.BestMove.IsNull)
      {
        return false;
      }

      // Possibly veto the instamove if there is a second-best move that could
      // catch up to best move if the planned search were conducted.
      MCTSNode[] childrenSortedQ = newRoot.ChildrenSorted(n => n.N == 0 ? float.MaxValue : (float)n.Q);
      MCTSNode[] childrenSortedN = newRoot.ChildrenSorted(n => -n.N);

      // Never instamove if the visit count of most visited node was
      // not much higher than visit count of second-best node.
      const float THRESHOLD_MIN_N_RATIO = 0.70f;
      if (newRoot.NumChildrenExpanded > 1)
      {
        float fracNBestToNSecondBest = (float)childrenSortedN[0].N / childrenSortedN[1].N;
        if (fracNBestToNSecondBest < THRESHOLD_MIN_N_RATIO)
        {
          if (VERBOSE) Console.WriteLine("fracNBestToNSecondBest " + fracNBestToNSecondBest);
          return false;
        }
      }

      // Don't instamove if TopN and TopQ moves differ.
      if (childrenSortedN[0] != childrenSortedQ[0])
      {
        if (VERBOSE) Console.WriteLine("different Q/N");
        return false;
      }

      float nGap = Math.Abs(childrenSortedQ[0].N - childrenSortedQ[1].N);
      float qGap = (float)(Math.Abs(childrenSortedQ[0].Q - childrenSortedQ[1].Q));
      float minNRequiredToChange = nGap / ManagerChooseBestMove.MinFractionNToUseQ(newRoot, qGap);
      int estNewVisitsThisMove = searchLimit.EstNumNodes(newRoot.N, (int)priorManager.EstimatedNPS, true);
      bool couldCatchUp = (estNewVisitsThisMove * 0.25f) > minNRequiredToChange;

      // Don't instamove if the second-best move looks close to catching up.
      if (couldCatchUp)
      {
        if (VERBOSE) Console.WriteLine("coudCatchUp " + minNRequiredToChange +  " " + estNewVisitsThisMove);
        return false;
      }

      // Instamove looks appropriate.
      if (VERBOSE) Console.WriteLine($"INSTAMOVE {newRoot.N} {priorManager.Root.N}   { nGap}  { qGap}");
      if (VERBOSE) Console.WriteLine();
      return true;
    }

  }
}
