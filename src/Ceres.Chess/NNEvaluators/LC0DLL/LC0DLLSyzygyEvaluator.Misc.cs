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

using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.UserSettings;
using System;
using System.Collections.Generic;
using System.Diagnostics;

#endregion

namespace Ceres.Chess.NNEvaluators.LC0DLL
{

  public delegate MGMove CheckTablebaseBestNextMoveDelegate(in Position currentPos, out WDLResult result, 
                                                            out List<(MGMove, short)> fullWinningMoveList, 
                                                            out bool winningMoveListOrderedByDTM);

  /// <summary>
  /// Miscellaneous helper methods related to Syzygy tablebase 
  /// probing routines  exposed by the LC0 DLL.

  /// </summary>
  public partial class LC0DLLSyzygyEvaluator : IDisposable, ISyzygyEvaluatorEngine
  {

    public MGMove CheckTablebaseBestNextMoveViaDTZ(in Position currentPos,
                                                   out WDLResult result, 
                                                   out List<(MGMove, short)> moveList, 
                                                   out short dtz,
                                                   bool returnOnlyWinningMoves = true,
                                                   bool succeedIfIncompleteDTZInfo = false)
    {
      if (succeedIfIncompleteDTZInfo)
      {
        throw new NotImplementedException();
      }  

      moveList = null;

      ProbeWDL(in currentPos, out SyzygyWDLScore score, out SyzygyProbeState probeResult);
      if (!(probeResult == SyzygyProbeState.Ok || probeResult == SyzygyProbeState.ZeroingBestMove))
      {
        result = WDLResult.Unknown;
        dtz = 0;
        return default(MGMove);
      }

      switch (score)
      {
        case SyzygyWDLScore.WDLLoss:
        case SyzygyWDLScore.WDLBlessedLoss:
          // This is probably actually a loss, but not way to represent that with this enum.
          // TODO: Clean this up.
          result = WDLResult.Unknown;
          break;

        case SyzygyWDLScore.WDLWin:
        case SyzygyWDLScore.WDLCursedWin:
          result = WDLResult.Win;
          break;

        case SyzygyWDLScore.WDLDraw:
          result = WDLResult.Draw;
          break;

        default:
          throw new Exception($"Unexpected WDLScore {score} {currentPos.FEN}");
      }

      // Try to find best move (quickest mate, else draw if possible).
      int dtzMove = ProbeDTZ(in currentPos);
      if (dtzMove < 0)
      {
        result = WDLResult.Loss;
        dtz = 0;
        return default(MGMove);
      }
      else
      {
        EncodedMove encodedMove = new EncodedMove((ushort)dtzMove);
        MGMove mgMove = MGMoveConverter.ToMGMove(in currentPos, encodedMove);
        dtz = -1; // not supported
        return mgMove;
      }
    }

    public int? NumWDLTablebaseFiles => null;

    public int? NumDTZTablebaseFiles => null;


    #region Tests

    static void Check(LC0DLLSyzygyEvaluator eval, string fen, WDLResult result, string moveStr)
    {
      MGMove move = (eval as ISyzygyEvaluatorEngine).CheckTablebaseBestNextMove(Position.FromFEN(fen), out WDLResult gr, out _, out _ );
      Debug.Assert(result == gr);
      if (gr != WDLResult.Unknown)
      {
        Debug.Assert(moveStr == move.ToString());
      }
    }


    public static void RunUnitTests()
    {
      //  CeresUserSettingsManager.LoadFromDefaultFile();

      LC0DLLSyzygyEvaluator eval = new(0);
      eval.Initialize(CeresUserSettingsManager.Settings.TablebaseDirectory);
      //Check(eval, "8/8/6R1/8/7k/8/6K1/8 w - - 0 1", GameResult.Checkmate, "Rg6-g3");
      Check(eval, "k7/P6R/2K5/8/7P/1r6/8/8 w - -", WDLResult.Win, "Rh7-h8"); // DTM 46
      Check(eval, "8/8/8/8/5kp1/P7/8/1K1N4 w - -", WDLResult.Win, "Kb1-c2"); // DTM 50
      Check(eval, "8/1k6/1p1r4/5K2/8/8/8/2R5 w - -", WDLResult.Draw, "Kf5-e4");
      Check(eval, "4kq2/8/8/8/8/8/8/4K3 w - - 0 1", WDLResult.Loss, null);
      Check(eval, Position.StartPosition.FEN, WDLResult.Unknown, null);
    }

    #endregion
  }

}
