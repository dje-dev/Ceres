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
  /// Static class holding a set of miscellaneous flags or methods
  /// related to diagnostic features.
  /// </summary>
  public static class MCTSDiagnostics
  {
    /// <summary>
    /// If verbose information related to search futility shutdown
    /// should be output console.
    /// </summary>
    public static bool DumpSearchFutilityShutdown = false;

    /// <summary>
    /// If a verbose summary of moves played should be output to Console
    /// at the end of a tournament game.
    /// </summary>
    public static bool TournamentDumpEngine2EndOfGameSummary = false;

    /// <summary>
    /// If the tree integrity verification runs at the end of each search.
    /// </summary>
    public static bool VerifyTreeIntegrityAtSearchEnd = false;
  }
}


