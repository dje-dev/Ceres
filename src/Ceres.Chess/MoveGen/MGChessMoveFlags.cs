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

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Representation of a move (as used by the move generator subsystem).
  /// </summary>
  /// 

  public partial struct MGMove
  {
    [Flags]
    public enum MGChessMoveFlags : ushort
    {
      None = 0,
      BlackToMove = 1 << 0,
      Check = 1 << 1,
      Capture = 1 << 2,
      EnPassantCapture = 1 << 3,
      DoublePawnMove = 1 << 4,
      CastleShort = 1 << 5,
      CastleLong = 1 << 6,
      PromoteKnight = 1 << 7,
      PromoteBishop = 1 << 8,
      PromoteRook = 1 << 9,
      PromoteQueen = 1 << 10,
      NoMoreMoves = 1 << 11, // not used
      Piece = 0b1111 << 12 // 4 bits
//      IllegalMove = 1 << 16,
//      MoveCount = 0b1111_1111 << 17, // 8 bits // not used
//      Unused = 0b1111111 << 25, // 7 bits
    }


  }

}