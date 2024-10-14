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

using System.Runtime.CompilerServices;

using Ceres.Chess.EncodedPositions.Basic;

#endregion

namespace Ceres.Chess.MoveGen.Converters
{
  public static class MoveInMGMovesArrayLocator
  {
    /// <summary>
    /// Returns the index of a specified move (starting search at specified index), or -1 if not found.
    /// Optimized for performance.
    /// </summary>
    /// <param name="moves"></param>
    /// <param name="move"></param>
    /// <param name="startIndex"></param>
    /// <param name="numMovesUsed"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int FindMoveIndex(MGMove[] moves, MGMove move, int startIndex, int numMovesUsed)
    {
      byte moveToSquare = move.ToSquareIndex;
      for (int i = startIndex; i < numMovesUsed; i++)
      {
        if (moves[i].ToSquareIndex == moveToSquare && moves[i] == move)
        {
          return i;
        }
      }

      return -1;
    }


    /// <summary>
    /// Returns the index of a specified move (starting search at specified index), or -1 if not found.
    /// Optimized for performance.
    /// </summary>
    /// <param name="moves"></param>
    /// <param name="moveSquares"></param>
    /// <param name="startIndex"></param>
    /// <param name="numMovesUsed"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FindMoveIndex(MGMove[] moves, ConverterMGMoveEncodedMove.FromTo moveSquares, int startIndex, int numMovesUsed)
    {
      short fromAndTo = moveSquares.FromAndToCombined;

      // partially unroll the loop to improve instruction-level parallelism
      int i = startIndex;
      while (i < numMovesUsed - 3)
      {
        if (moves[i].FromAndToCombined == fromAndTo) return i;
        if (moves[i + 1].FromAndToCombined == fromAndTo) return i + 1;
        if (moves[i + 2].FromAndToCombined == fromAndTo) return i + 2;
        if (moves[i + 3].FromAndToCombined == fromAndTo) return i + 3;

        i += 4;
      }

      while (i < numMovesUsed)
      {
        if (moves[i].FromAndToCombined == fromAndTo)
        {
          return i;
        }

        i++;
      }

      return -1;
    }


    /// <summary>
    /// Returns the index of a specified move (starting search at specified index), or -1 if not found.
    /// </summary>
    /// <param name="posMG"></param>
    /// <param name="legalMoveArray"></param>
    /// <param name="thisPolicyMove"></param>
    /// <param name="countPolicyMovesProcessed"></param>
    /// <param name="numMovesUsed"></param>
    /// <param name="blackToMove"></param>
    /// <returns></returns>
    public static int FindMoveInMGMoves(in MGPosition posMG, MGMove[] legalMoveArray, EncodedMove thisPolicyMove, int countPolicyMovesProcessed, int numMovesUsed, bool blackToMove)
    {
      PieceType pieceMoving = posMG.PieceMoving(thisPolicyMove);

      if (pieceMoving == PieceType.Pawn && thisPolicyMove.ToSquare.IsRank8) // pawn promotion
      {
        // This slower code path is necessary because in this case of promotion the (from, to) are not sufficient to uniquely identify move
        MGMove policyMoveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(thisPolicyMove, in posMG);

        // Find this array of possible moves (be careful to only search the part of the array currently in use)
        return FindMoveIndex(legalMoveArray, policyMoveMG, countPolicyMovesProcessed, numMovesUsed);
      }
      else
      {
        ConverterMGMoveEncodedMove.FromTo mgMoveFromTo = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMoveFromTo(thisPolicyMove, blackToMove);
        return FindMoveIndex(legalMoveArray, mgMoveFromTo, countPolicyMovesProcessed, numMovesUsed);
      }
    }


  }

}
