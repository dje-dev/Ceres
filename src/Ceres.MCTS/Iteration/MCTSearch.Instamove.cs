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
        && priorManager.Context.ParamsSearch.FutilityPruningStopSearchEnabled
        && priorManager.Context.ParamsSearch.EnableInstamoves)
      {
        // Don't make too many instamoves in a row because
        // time control state has been improved by the instamoves, 
        // so further search may now be warranted.
        const int MAX_CONSECUTIVE_INSTAMOVES = 2;
        if (CountSearchContinuations >= MAX_CONSECUTIVE_INSTAMOVES)
          return default;

        if (double.IsNaN(priorManager.EstimatedNPS)) return default;

        int estNewVisitsThisMove = searchLimit.EstNumNodes((int)priorManager.EstimatedNPS, true);
        float treeSizeFactor = (float)newRoot.N / (float)estNewVisitsThisMove;

        BestMoveInfo bestMoveInfo = newRoot.BestMoveInfo(false);

        // The "sureness" that a move is best and not needing more search
        // is the product of treeSizeFactor and TopMovesNRatio
        float surenessFactor = treeSizeFactor * MathF.Min(10, bestMoveInfo.TopMovesNRatio);

        // Key parameter which determines how "sure" the top move
        // must be to selected as an instamove.
        // The threshold increases with the number of already-done continuations,
        // to gently discourage too many consecutive instamoves unless they are compelling.
        float instamoveSurenessThreshold = 4 + (2 * CountSearchContinuations);

        if (!bestMoveInfo.BestMove.IsNull)
        {
          if (surenessFactor >= instamoveSurenessThreshold)
          {
            InstamoveCount++;
            CountSearchContinuations++;
            return (priorManager, newRoot.BestMoveInfo(false).BestMove, null);
          }
        }
      }
      return default;
    }

  }
}
