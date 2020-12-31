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
using Ceres.Chess.MoveGen.Converters;
using System;
using System.Collections.Generic;

#endregion

namespace Ceres.Chess.Positions
{
  /// <summary>
  /// Helper method/class which enumerates positions available 
  /// after a single move is made from a specified position.
  /// 
  /// </summary>
  public static class PositionsGenerator1Ply
  {
    /// <summary>
    /// Generates the set of moves and positions possible as the next move in a position.
    /// </summary>
    /// <param name="startPos"></param>
    /// <param name="moveToIncludeFilter"></param>
    /// <returns></returns>
    public static IEnumerable<(MGMove, Position)> GenPositions(Position startPos, Predicate<MGMove> moveToIncludeFilter = null)
    {
      MGPosition posMG = MGChessPositionConverter.MGChessPositionFromFEN(startPos.FEN); // TODO: more efficient?
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in posMG, moves);

      for (int i = 0; i < moves.NumMovesUsed; i++)
      {
        // Only consider captures (would reduce number of pieces into range of tablebase)
        if (moveToIncludeFilter == null || moveToIncludeFilter(moves.MovesArray[i]))
        {
          // Make this move and get new Position
          MGPosition newPosMG = new MGPosition(posMG);
          newPosMG.MakeMove(moves.MovesArray[i]);
          Position newPos = MGChessPositionConverter.PositionFromMGChessPosition(in newPosMG);

          yield return (moves.MovesArray[i], newPos);
        }
      }
    }
  }
}