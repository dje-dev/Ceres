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
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Textual;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Static helper methods to convert from string to MGMove.
  /// </summary>
  public static class MGMoveFromString
  {
    /// <summary>
    /// Parses a move string from a specified starting position
    /// (either algebraic or SAN format is accepted).
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="moveStr"></param>
    /// <returns></returns>
    public static MGMove ParseMove(MGPosition pos, string moveStr)
    {
      if (!TryParseMoveCoordinateOrAlgebraic(pos, moveStr, out MGMove move))
      {
        Position position = MGChessPositionConverter.PositionFromMGChessPosition(in pos);
        PositionWithMove mfp = SANParser.FromSAN(moveStr, in position);
        return MGMoveConverter.MGMoveFromPosAndMove(in position, mfp.Move);
      }
      else
        return move;
    }


    /// <summary>
    /// Attempts to parse a move string in coordinate or long algebraic format.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="moveStr"></param>
    /// <param name="move"></param>
    /// <returns></returns>
    private static bool TryParseMoveCoordinateOrAlgebraic(MGPosition pos, string moveStr, out MGMove move)
    {
      moveStr = moveStr.ToLower();

      // Sometimes promotions to Knight use the "k" instead of expected "n"
      if (moveStr.EndsWith("k"))
        moveStr = moveStr.Substring(0, moveStr.Length - 1) + "n";

      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in pos, moves);
      foreach (MGMove moveTry in moves.MovesArray)
      {
        // Accept moves in any of multiple formats, including Chess 960 (for castling variation)
        if (String.Equals(moveTry.MoveStr(MGMoveNotationStyle.LC0Coordinate), moveStr, StringComparison.OrdinalIgnoreCase)
         || String.Equals(moveTry.MoveStr(MGMoveNotationStyle.LC0Coordinate960Format), moveStr, StringComparison.OrdinalIgnoreCase)
         || String.Equals(moveTry.MoveStr(MGMoveNotationStyle.LongAlgebraic), moveStr, StringComparison.OrdinalIgnoreCase))
        {
          move = moveTry;
          return true;
        }           
      }

      move = default;
      return false;
    }
  }


}
