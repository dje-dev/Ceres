
#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using Ceres.Base.DataTypes;
using Ceres.Base.Environment;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Phases.Evaluation;

/// <summary>
/// Selection terminator which probes Syzygy tablebases 
/// and returns optimal WDL evaluation if found.
/// </summary>
public sealed class EvaluatorSyzygy
{
  /// <summary>
  /// Maximum cardinality (number of pieces) supported by tablebases.
  /// </summary>
  public int MaxCardinality { init; get; }

  /// <summary>
  /// Number of probe successes.
  /// </summary>
  internal AccumulatorMultithreaded NumHits;

  /// <summary>
  /// 
  /// </summary>
  public readonly ISyzygyEvaluatorEngine Evaluator;


  internal static readonly float BLESSED_WIN_LOSS_MAGNITUDE = 0.05f;
  internal static readonly FP16 BLSSED_WIN_LOSS_MAGNITUDE_FP16 = (FP16)BLESSED_WIN_LOSS_MAGNITUDE;



  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="paths">Set of paths in which tablebase are found; if null then Ceres.json settings used.</param>
  /// <param name="forceNoTablebaseTerminals"></param>
  public EvaluatorSyzygy(string paths, bool forceNoTablebaseTerminals = false)
  {
    Evaluator = SyzygyEvaluatorPool.GetSessionForPaths(paths);
    MaxCardinality = Evaluator.MaxCardinality;

    if (forceNoTablebaseTerminals)
    {
      throw new NotImplementedException();
    }

    NumHits.Initialize();
  }


  internal bool Lookup(MCGSPath path, in Position pos, ref SelectTerminationInfo terminationInfo)
  {
    // Do not enable at root, since we need need to choose actual best move
    // which is not available without at least one level of search, since
    // DTZ access is not yet implemented.
    //
    // TODO: when implement DTZ access then we would know which move to make immediate
    //       then this would not be needed
    if (path.NumVisitsInPath == 1)
    {
      return false;
    }

    Evaluator.ProbeWDL(in pos, out SyzygyWDLScore score, out SyzygyProbeState resultCode);
    if (resultCode == SyzygyProbeState.Fail || resultCode == SyzygyProbeState.ChangeSTM)
    {
      return false;
    }

    NumHits.Add(1, pos.PiecesShortHash);

    switch (score)
    {
      // TODO: use information about distance to draw/mate to fill in the M
      // and also argument to WinPForProvenWin/WinPForProvenLoss
      case SyzygyWDLScore.WDLDraw:
        terminationInfo = new SelectTerminationInfo(pos.SideToMove, MCGSPathTerminationReason.TerminalEdge, GameResult.Draw, 0, 0, -1, 0, 0);
        return true;

      case SyzygyWDLScore.WDLWin:
        terminationInfo = new SelectTerminationInfo( pos.SideToMove, MCGSPathTerminationReason.TerminalEdge, GameResult.Checkmate, (FP16)ParamsSelect.WinPForProvenWin(1, false), 0, 1, 0, 0);
        return true;

      case SyzygyWDLScore.WDLLoss:
        terminationInfo = new SelectTerminationInfo(pos.SideToMove, MCGSPathTerminationReason.TerminalEdge, GameResult.Checkmate, 0, (FP16)ParamsSelect.LossPForProvenLoss(1, false), 1, 0,  0);
        return true;

      case SyzygyWDLScore.WDLCursedWin:
        // Score as almost draw, just slightly positive (since opponent might err in obtaining draw)
        terminationInfo = new SelectTerminationInfo(pos.SideToMove, MCGSPathTerminationReason.TerminalEdge, GameResult.Draw, BLSSED_WIN_LOSS_MAGNITUDE_FP16, 0, 100, 0, 0);
        return true;

      case SyzygyWDLScore.WDLBlessedLoss:
        // Score as almost draw, just slightly negative (since we might err in obtaining draw)
        terminationInfo = new SelectTerminationInfo(pos.SideToMove, MCGSPathTerminationReason.TerminalEdge, GameResult.Draw, 0, BLSSED_WIN_LOSS_MAGNITUDE_FP16, 100, 0, 0);
        return true;

      default:
        throw new Exception("Internal error: unknown Syzygy tablebase result code");
    }
  }


  public static bool PosIsTablebaseWinWithNoDTZAvailable(string tablebasePaths, PositionWithHistory priorMoves)
  {
    // Check if this is the unusual situation of a tablebase hit
    // but only WDL and not DTZ available.
    Position startPos = priorMoves.FinalPosition;
    bool forceNoTablebaseTerminals = false;
    if (startPos.PieceCount <= 7 && tablebasePaths != null) // TODO: remove hardcoding of the PieceCount maximum here
    {
      ISyzygyEvaluatorEngine evaluator = SyzygyEvaluatorPool.GetSessionForPaths(tablebasePaths);
      if (startPos.PieceCount <= evaluator.MaxCardinality)
      {
        MGMove ret = evaluator.CheckTablebaseBestNextMove(in startPos, out WDLResult result,
          out List<(MGMove, short)> fullWinningMoveList, out bool winningMoveListOrderedByDTM);

        if (result == WDLResult.Win && !winningMoveListOrderedByDTM)
        {
          // No DTZ were available to guide search, must start a new graph
          // and perform actual NN search to find the win.
          forceNoTablebaseTerminals = true;
        }
      }
    }

    return forceNoTablebaseTerminals;
  }

}
