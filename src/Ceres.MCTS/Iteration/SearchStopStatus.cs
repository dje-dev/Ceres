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
  public partial class MCTSManager
  {
    /// <summary>
    /// Status of search (either continuing or reason if was stopped).
    /// </summary>
    public enum SearchStopStatus
    {
      /// <summary>
      /// Search is not stopped.
      /// </summary>
      Continue,

      /// <summary>
      /// Only one legal move, search not required.
      /// </summary>
      OnlyOneLegalMove,

      /// <summary>
      /// Flag requesting external stop (e.g. from UCI manager) was set.
      /// </summary>
      ExternalStopRequested,

      /// <summary>
      /// Panic low time limit reached.
      /// </summary>
      PanicTimeTooLow,

      /// <summary>
      /// Search limit (e.g. max number of nodes) reached.
      /// </summary>
      SearchLimitExceeded,

      /// <summary>
      /// The MaxTreeVisits field in the SearchLimit was set
      /// and the search tree has reached that limit.
      /// </summary>
      MaxTreeVisitsExceeded,

      /// <summary>
      /// The MaxTreeNodes field in the SearchLimit was set
      /// and the store backing the search tree has reached that limit.
      /// </summary>
      MaxTreeAllocatedNodesExceeded,

      /// <summary>
      /// All top-level moves were pruned.
      /// </summary>
      FutilityPrunedAllMoves,

      /// <summary>
      /// Best move is chosen from existing reused tree.
      /// </summary>
      Instamove,

      /// <summary>
      /// Root position in tablebase and optimal move available.
      /// </summary>
      TablebaseImmediateMove
    }

  }
}

