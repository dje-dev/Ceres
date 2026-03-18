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
using System.Diagnostics;
using System.Runtime.CompilerServices;

using Ceres.Base.DataTypes;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Phases.Evaluation;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.PathEvaluators;

/// <summary>
/// Structure that transiently holds the output of a selection terminator
/// (until the result can be transferred into an MCGS node).
/// </summary>
public readonly struct SelectTerminationInfo
{
  /// <summary>
  /// Side current to move.
  /// </summary>
  public readonly SideType Side;

  /// <summary>
  /// Reason why the path was terminated at the final node.
  /// </summary>
  public readonly MCGSPathTerminationReason TerminationReason;

  /// <summary>
  /// Terminal status of node.
  /// </summary>
  public readonly GameResult GameResult;

  /// <summary>
  /// Policy win probability percentage.
  /// </summary>
  public readonly FP16 WinP;

  /// <summary>
  /// Policy loss probability percentage.
  /// </summary>
  public readonly FP16 LossP;

  /// <summary>
  /// Moves left value (if any).
  /// </summary>
  public readonly FP16 M;

  /// <summary>
  /// Uncertainty of V.
  /// </summary>
  public readonly FP16 UncertaintyV;

  /// <summary>
  /// Uncertainty of P.
  /// </summary>
  public readonly FP16 UncertaintyP;

  /// <summary>
  /// Fortress probability metric: minimum P(NEVER) over all pawn squares.
  /// </summary>
  public readonly FP16 FortressP;


#if ACTION_ENABLED
  /// <summary>
  /// Optional states information about the node.
  /// </summary>
  public readonly Half[] State;
#endif


  /// <summary>
  /// Policy draw probability percentage.
  /// </summary>
  public float DrawP
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    get
    {
      float winP = WinP.ToFloat;
      float lossP = LossP.ToFloat;
      float v = winP - lossP;
      return ParamsSelect.VIsForcedResult(v) ? 0 : 1.0f - (winP + lossP);
    }
  }


  /// <summary>
  /// Value estimate (logistic).
  /// </summary>
  public float V => WinP.ToFloat - LossP.ToFloat;

  /// <summary>
  /// If the structure has been initialized.
  /// </summary>
  public bool IsNull => GameResult == GameResult.NotInitialized;



  /// <summary>
  /// Constructor (including policy/action/state).
  /// </summary>
  /// <param name="terminatorType"></param>
  /// <param name="side"></param>
  /// <param name="terminationReason"></param>
  /// <param name="terminalStatus"></param>
  /// <param name="winP"></param>
  /// <param name="lossP"></param>
  /// <param name="m"></param>
  /// <param name="uncertaintyV"></param>
  /// <param name="uncertaintyP"></param>
  /// <param name="state"></param>
  /// <param name="fortressP"></param>
  public SelectTerminationInfo(SideType side,
                               MCGSPathTerminationReason terminationReason, GameResult terminalStatus,
                               FP16 winP, FP16 lossP, FP16 m, FP16 uncertaintyV, FP16 uncertaintyP,
                               Half[] state, FP16 fortressP = default)
  {
    Debug.Assert(terminalStatus != GameResult.NotInitialized);

    Side = side;
    TerminationReason = terminationReason;
    GameResult = terminalStatus;

    WinP = winP;
    LossP = lossP;
    M = m;

    UncertaintyV = uncertaintyV;
    UncertaintyP = uncertaintyP;
    FortressP = fortressP;

#if ACTION_ENABLED
    State = state;
#endif
  }


  /// <summary>
  /// Constructor (without action/policy/state).
  /// </summary>
  /// <param name="terminatorType"></param>
  /// <param name="side"></param>
  /// <param name="terminationReason"></param>
  /// <param name="gameResult"></param>
  /// <param name="winP"></param>
  /// <param name="lossP"></param>
  /// <param name="m"></param>
  /// <param name="uncertaintyV"></param>
  /// <param name="uncertaintyP"></param>
  /// <param name="fortressP"></param>
  public SelectTerminationInfo(SideType side,
                               MCGSPathTerminationReason terminationReason, 
                               GameResult gameResult, 
                               FP16 winP, FP16 lossP, FP16 m, FP16 uncertaintyV, FP16 uncertaintyP,
                               FP16 fortressP = default)
  {
    Debug.Assert(gameResult != GameResult.NotInitialized);
    Debug.Assert(gameResult != GameResult.Draw || (winP == 0 && lossP == 0) 
                                               || (winP == EvaluatorSyzygy.BLSSED_WIN_LOSS_MAGNITUDE_FP16 && lossP == 0)
                                               || (lossP == EvaluatorSyzygy.BLSSED_WIN_LOSS_MAGNITUDE_FP16 && winP == 0));
    Debug.Assert(gameResult != GameResult.Checkmate || winP >= 1 || lossP >= 1);

    Side = side;
    TerminationReason = terminationReason;
    GameResult = gameResult;

    WinP = winP;
    LossP = lossP;
    M = m;

    UncertaintyV = uncertaintyV;
    UncertaintyP = uncertaintyP;
    FortressP = fortressP;
  }




  /// <summary>
  /// Returns string representation.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<SelectTerminationInfo {(IsNull ? "(null)>" 
                                             : $"{Side}  reason {TerminationReason} as {GameResult} V={WinP - LossP,6:F3}>")}";
  }

}
