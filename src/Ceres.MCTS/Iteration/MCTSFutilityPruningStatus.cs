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


#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Futility pruning status of a given move at root.
  /// </summary>
  public enum MCTSFutilityPruningStatus
  {
    /// <summary>
    /// Move is not pruned (eligible for more search visits).
    /// </summary>
    NotPruned,

    /// <summary>
    /// Move is pruned because not specified in set of restricted search moves.
    /// </summary>
    PrunedDueToSearchMoves,

    /// <summary>
    /// Move is pruned because judged not to be a plausible candidate for best move.
    /// </summary>
    PrunedDueToFutility,

    /// <summary>
    /// Searching using only WDL tablebases and position is a win 
    /// but move is not one of the winning moves.
    /// </summary>
    PrunedDueToTablebaseNotWinning
  }
}


