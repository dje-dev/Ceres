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
using System.Collections.Generic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using static Ceres.Chess.NNEvaluators.LC0DLL.LC0DLLSyzygyEvaluator;

#endregion

namespace Ceres.Chess.NNEvaluators.LC0DLL
{
  /// <summary>
  /// Interface shared by all Syzygy evaluators (and some shared logic).
  /// </summary>
  public interface ISyzygyEvaluatorEngine
  {
    /// <summary>
    /// Maximum number of pieces of available tablebase positions.
    /// </summary>
    public int MaxCardinality { get; }


    /// <summary>
    /// If the DTZ files supported by the engine and potentially usable
    /// (if the necessary tablebase files are found for a given piece combination).
    /// The case of partial DTZ availability is not generally supported
    /// (not guaranteed to produce correct play).
    /// </summary>
    public bool DTZAvailable { get;}

    /// <summary>
    /// The number of wdl tablebase files available, if known.
    /// </summary>
    public int? NumWDLTablebaseFiles { get; }

    /// <summary>
    /// The number of DTZ tablebase files available, if known.
    /// </summary>
    public int? NumDTZTablebaseFiles { get; }

    /// <summary>
    /// Initializes tablebases to use a specied set of paths.
    /// </summary>
    /// <param name="paths"></param>
    /// <returns></returns>
    public bool Initialize(string paths);


    /// <summary>
    /// Probes the win/draw/loss information for a specified position.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="score"></param>
    /// <param name="result"></param>
    public void ProbeWDL(in Position pos, out WDLScore score, out ProbeState result);

    
    /// <summary>
    /// Probes the distance to zero tables to find the suggested move to play.
    /// </summary>
    /// <param name="currentPos"></param>
    /// <param name="result"></param>
    /// <returns></returns>
    public MGMove CheckTablebaseBestNextMoveViaDTZ(in Position currentPos, out GameResult result, out List<MGMove> fullWinningMoveList);


    /// <summary>
    /// Shuts down the evaluators (releasing associated resources).
    /// </summary>
    public void Dispose();


    const bool VERBOSE = false;

    /// <summary>
    /// Returns best winning move (if any) from a specified position
    /// (and possibly a list of all possible winning moves, with shortest mates first).
    /// </summary>
    /// <param name="currentPos"></param>
    /// <param name="result"></param>
    /// <param name="fullWinningMoveList"></param>
    /// <returns></returns>
    public MGMove CheckTablebaseBestNextMove(in Position currentPos, out GameResult result, out List<MGMove> fullWinningMoveList, out bool winningMoveListOrderedByDTM)
    {
      fullWinningMoveList = null;

      // TODO: Ponder if this could be done for PieceCount == MaxCardinality + 1
      if (currentPos.PieceCount > MaxCardinality)
      {
        result = GameResult.Unknown;
        winningMoveListOrderedByDTM = false;
        return default;
      }

      if (DTZAvailable)
      {
        // Try to use DTZ table, which may may not work (depending on file availability).
        MGMove dtzMove = CheckTablebaseBestNextMoveViaDTZ(in currentPos, out result, out fullWinningMoveList);

        if (result != GameResult.Unknown)
        {
          if (VERBOSE)
          {
            Console.WriteLine($"\r\nCheckTablebaseBestNextMove via DTZ yields {result} {dtzMove} with winning list "
                                       + $"size {fullWinningMoveList?.Count} in position {currentPos.FEN}");
          }

          winningMoveListOrderedByDTM = true;
          return dtzMove;
        }
      }

      // Fall thru to use WDL
      MGMove ret =  CheckTablebaseBestNextMoveViaWDL(in currentPos, out result, out fullWinningMoveList);
      if (VERBOSE)
      {
        Console.WriteLine($"\r\nCheckTablebaseBestNextMove via WDL yields {result} {ret} with winning list "
                                   + $"size {fullWinningMoveList?.Count} in position {currentPos.FEN}");
      }
      winningMoveListOrderedByDTM = false;
      return ret;
    }


    MGMove CheckTablebaseBestNextMoveViaWDL(in Position currentPos, out GameResult result, out List<MGMove> fullWinningMoveList)
    {
      fullWinningMoveList = new();

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
            fullWinningMoveList.Add(move);
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
