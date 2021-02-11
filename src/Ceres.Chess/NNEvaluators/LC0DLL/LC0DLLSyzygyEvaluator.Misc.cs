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
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using System;
using System.Diagnostics;

#endregion

namespace Ceres.Chess.NNEvaluators.LC0DLL
{
  public delegate MGMove CheckTablebaseBestNextMoveDelegate(in Position currentPos, out GameResult result);

  /// <summary>
  /// Miscellaneous helper methods related to Syzygy tablebase 
  /// probing routines  exposed by the LC0 DLL.

  /// </summary>
  public partial class LC0DLLSyzygyEvaluator : IDisposable
  {
    /// <summary>
    /// 
    /// TODO: someday make use of DTZ information
    /// </summary>
    /// <param name="currentPos"></param>
    /// <returns></returns>
    public MGMove CheckTablebaseBestNextMove(in Position currentPos, out GameResult result)
    {
      // Finding best move is only possible if the number of pieces
      // at most one more than the max tablebase cardinality
      // (because the next move considered might be a capture reducing count by 1)
      if (currentPos.PieceCount > MaxCardinality + 1)
      {
        result = GameResult.Unknown;
        return default;
      }

      if (DTZAvailable)
      {
        return CheckTablebaseBestNextMoveViaDTZ(in currentPos, out result);
      }
      else
      {
        return CheckTablebaseBestNextMoveViaDTM(in currentPos, out result);
      }
    }

   
    MGMove CheckTablebaseBestNextMoveViaDTZ(in Position currentPos, out GameResult result)
    {
      ProbeWDL(in currentPos, out WDLScore score, out ProbeState probeResult);
      if (!(probeResult == ProbeState.Ok || probeResult == ProbeState.ZeroingBestMove))
      {
        result = GameResult.Unknown;
        return default(MGMove);
      }

      switch (score)
      {
        case WDLScore.WDLLoss:
        case WDLScore.WDLBlessedLoss:
          // This is probably actually a loss, but not way to represent that with this enum.
          // TODO: Clean this up.
          result = GameResult.Unknown;
          break;

        case WDLScore.WDLWin:
        case WDLScore.WDLCursedWin:
          result = GameResult.Checkmate;
          break;

        case WDLScore.WDLDraw:
          result = GameResult.Draw;
          break;

        default:
          throw new Exception($"Unexpected WDLScore {score} {currentPos.FEN}");
      }

      // Try to find best move (quickest mate, else draw if possible).
      int dtzMove = ProbeDTZ(in currentPos);
      if (dtzMove < 0)
      {
        result = GameResult.Unknown;
        return default(MGMove);
      }
      else
      {
        EncodedMove encodedMove = new EncodedMove((ushort)dtzMove);
        MGMove mgMove = MGMoveConverter.ToMGMove(in currentPos, encodedMove);
        return mgMove;
      }
    }


    MGMove CheckTablebaseBestNextMoveViaDTM(in Position currentPos, out GameResult result)
    {
      // First check for immediate winning or drawing moves known by TB probe
      MGMove winningMove = default;
      MGMove winningCursedMove = default;
      MGMove drawingMove = default;
      bool allNextPositionsInTablebase = true;

      // Generate all possible next moves and look up in tablebase
      foreach ((MGMove move, Position nextPos) in PositionsGenerator1Ply.GenPositions(currentPos))
      {
        ProbeWDL(in nextPos, out WDLScore score, out ProbeState probeResult);
        if (!(probeResult == ProbeState.Ok || probeResult == ProbeState.ZeroingBestMove))
        {
          allNextPositionsInTablebase = false;
          continue;
        }

        switch (score)
        {
          case WDLScore.WDLBlessedLoss: // blessed loss for the opponent
            winningCursedMove = move;
            break;

          case WDLScore.WDLLoss: // loss for the opponent
            winningMove = move;
            break;

          case WDLScore.WDLDraw:
            drawingMove = move;
            break;

          default:
            break;
        }
      }

      MGMove bestMove = default;
      if (winningMove != default(MGMove))
      {
        // If we found a winning move, definitely make it 
        bestMove = winningMove;
        result = GameResult.Checkmate;
      }
      else if (winningCursedMove != default(MGMove))
      {
        // If we found a cursed winning move, we might as well try making it
        bestMove = winningCursedMove;
        result = GameResult.Draw;
      }
      else if (allNextPositionsInTablebase && drawingMove != default(MGMove))
      {
        // If we were able to find all next positions and none are winning, 
        // then take (any) drawing move if we found it
        bestMove = drawingMove;
        result = GameResult.Draw;
      }
      else
        result = GameResult.Unknown;

      return bestMove;
    }


    #region Tests

    static void Check(LC0DLLSyzygyEvaluator eval, string fen, GameResult result, string moveStr)
    {
      MGMove move = eval.CheckTablebaseBestNextMove(Position.FromFEN(fen), out GameResult gr);
      Debug.Assert(result == gr);
      if (gr != GameResult.Unknown)
      {
        Debug.Assert(moveStr == move.ToString());
      }
    }


    public static void RunUnitTests()
    {
      //  CeresUserSettingsManager.LoadFromDefaultFile();

      LC0DLLSyzygyEvaluator eval = new(0, CeresUserSettingsManager.Settings.TablebaseDirectory);
      //Check(eval, "8/8/6R1/8/7k/8/6K1/8 w - - 0 1", GameResult.Checkmate, "Rg6-g3");
      Check(eval, "k7/P6R/2K5/8/7P/1r6/8/8 w - -", GameResult.Checkmate, "Rh7-h8"); // DTM 46
      Check(eval, "8/8/8/8/5kp1/P7/8/1K1N4 w - -", GameResult.Checkmate, "Kb1-c2"); // DTM 50
      Check(eval, "8/1k6/1p1r4/5K2/8/8/8/2R5 w - -", GameResult.Draw, "Kf5-e4");
      Check(eval, "4kq2/8/8/8/8/8/8/4K3 w - - 0 1", GameResult.Unknown, null);
    }

    #endregion
  }

}
