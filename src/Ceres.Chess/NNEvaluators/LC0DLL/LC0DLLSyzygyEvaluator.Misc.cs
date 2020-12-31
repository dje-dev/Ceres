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
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using System;

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
  }

}