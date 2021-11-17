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


namespace Ceres.Chess
{
  /// <summary>
  /// The type of resource which is limited in a search (e.g. nods or time).
  /// </summary>
  public enum SearchLimitType
  {
    /// <summary>
    /// Number of nodes (visits to leaf positions) to be made per move.
    /// </summary>
    NodesPerMove,

    /// <summary>
    /// Number of final nodes in tree at end of search 
    /// (or more, if search started with already more).
    /// </summary>
    NodesPerTree,

    /// <summary>
    /// Total number of nodes (visits to leaf positions) for all moves.
    /// </summary>
    NodesForAllMoves,

    /// <summary>
    /// Seconds for this move.
    /// </summary>
    SecondsPerMove,

    /// <summary>
    /// Total number of seconds for all moves.
    /// </summary>
    SecondsForAllMoves
  };
}
