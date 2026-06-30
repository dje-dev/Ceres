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

namespace Ceres.MCGS.Managers.Limits;

/// <summary>
/// Interface implemented by classes which manage the
/// determination of move limits (nodes or time) for single moves
/// within the context of a whole game and associated time allotments.
/// </summary>
public interface IManagerGameLimit
{
  /// <summary>
  /// Determines how much time or nodes resource to
  /// allocate to the the current move in a game subject to
  /// a limit on total numbrer of time or nodes over 
  /// some number of moves (or possibly all moves).
  /// </summary>
  /// <param name="search"></param>
  /// <param name="inputs"></param>
  /// <returns></returns>
  public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs, bool applyEarlySmoothing = true);


  /// <summary>
  /// If set, the manager should build a human-readable diagnostic string describing how each move's
  /// allocation was derived and attach it to the outputs (ManagerGameLimitOutputs.DiagnosticText), so
  /// callers (e.g. the diagnostic game move-log) can record it near the move header. This is
  /// independent of any console dumping, which is governed separately (and unconditionally) by
  /// MCGSParamsFixed.DUMP_LIMIT_CALC. Default implementation is a no-op for managers that do not
  /// produce diagnostics.
  /// </summary>
  public bool CaptureDiagnostics { get => false; set { } }


  /// <summary>
  /// Method called periodically to determine if an instamove would be
  /// acceptable given the current state.
  /// </summary>
  /// <param name="search"></param>
  /// <param name="newRoot"></param>
  /// <param name="inputs"></param>
  /// <returns></returns>
  public LimitsManagerInstamoveDecision CheckInstamove(ManagerGameLimitInputs inputs)
    => LimitsManagerInstamoveDecision.NoDecision;


  /// <summary>
  /// Method called periodically to determine if the limits manager
  /// requests the search be terminated soon or immediately.
  /// </summary>
  /// <param name="search"></param>
  /// <param name="inputs"></param>
  /// <returns></returns>
  public bool CheckStopSearch(/*MCTSearch search, */ManagerGameLimitInputs inputs) => false;
}
