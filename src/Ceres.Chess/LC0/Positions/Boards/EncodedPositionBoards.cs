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
using System.Runtime.InteropServices;
using Ceres.Base.DataTypes;

#endregion

namespace Ceres.Chess.LC0.Boards
{
  /// <summary>
  /// Encoced position as an array of boards including current position and history boards.
  /// </summary>
  [Serializable()]
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  public readonly unsafe struct EncodedPositionBoards : IEquatable<EncodedPositionBoards>
  {
    public const int NUM_MOVES_HISTORY = 8;

    #region Raw structure data

    //  All planes (including history) where the later moves are at lower indices (slot 0 is current position)
    // Perhaps ideally we would use a fixed buffer, but this is not allowed in C# 7 (see above)
    //    public fixed LZTrainingPositionRawInputPlaneSet HistoryPlanes[NUM_MOVES_HISTORY];
    public readonly EncodedPositionBoard History_0;
    public readonly EncodedPositionBoard History_1;
    public readonly EncodedPositionBoard History_2;
    public readonly EncodedPositionBoard History_3;
    public readonly EncodedPositionBoard History_4;
    public readonly EncodedPositionBoard History_5;
    public readonly EncodedPositionBoard History_6;
    public readonly EncodedPositionBoard History_7;

    #endregion

    /// <summary>
    /// Constructor form set of history boards.
    /// </summary>
    /// <param name="history0"></param>
    /// <param name="history1"></param>
    /// <param name="history2"></param>
    /// <param name="history3"></param>
    /// <param name="history4"></param>
    /// <param name="history5"></param>
    /// <param name="history6"></param>
    /// <param name="history7"></param>
    public EncodedPositionBoards(EncodedPositionBoard history0, EncodedPositionBoard history1, EncodedPositionBoard history2, EncodedPositionBoard history3,
                                 EncodedPositionBoard history4, EncodedPositionBoard history5, EncodedPositionBoard history6, EncodedPositionBoard history7)
    {
      History_0 = history0;
      History_1 = history1;
      History_2 = history2;
      History_3 = history3;
      History_4 = history4;
      History_5 = history5;
      History_6 = history6;
      History_7 = history7;

    }


    /// <summary>
    /// Mirrors all board planes in place.
    /// </summary>
    public unsafe void MirrorBoardsInPlace()
    {
      History_0.MirrorPlanesInPlace();
      History_1.MirrorPlanesInPlace();
      History_2.MirrorPlanesInPlace();
      History_3.MirrorPlanesInPlace();
      History_4.MirrorPlanesInPlace();
      History_5.MirrorPlanesInPlace();
      History_6.MirrorPlanesInPlace();
      History_7.MirrorPlanesInPlace();
    }

    /// <summary>
    /// Set a speciifed history board to a specified value.
    /// </summary>
    /// <param name="boardIndex"></param>
    /// <param name="board"></param>
    public void SetBoard(int boardIndex, EncodedPositionBoard board)
    {
      Debug.Assert(boardIndex >= 0 && boardIndex < 8);
      fixed (EncodedPositionBoard* boards = &History_0)
      {
        boards[boardIndex] = board;
      }
    }


    /// <summary>
    /// Determines if en passant opportunity 
    /// exists between two specified consecutive history boards.
    /// </summary>
    /// <param name="currentBoard"></param>
    /// <param name="priorBoard"></param>
    /// <returns></returns>
    public static PositionMiscInfo.EnPassantFileIndexEnum EnPassantOpportunityBetweenBoards(EncodedPositionBoard currentBoard, 
                                                                                           EncodedPositionBoard priorBoard)
    {
      EncodedPositionBoardPlane pawnsPrior = priorBoard.OurPawns;
      EncodedPositionBoardPlane pawnsCurrent = currentBoard.TheirPawns;

      // We look for the boards differing only because a pawn moved from rank 7 to rank 5 (from our perspective)
      for (int i = 8; i <16; i++)
      {
        if (
            priorBoard.TheirPawns.BitIsSet(64 - i)  &&
            !priorBoard.TheirPawns.BitIsSet(64 - i - 16) &&
            !currentBoard.TheirPawns.BitIsSet(64 - i) &&
            currentBoard.TheirPawns.BitIsSet(64 - i - 16)  
           )
        {
          return (PositionMiscInfo.EnPassantFileIndexEnum)new Square(16 - i).File;
        }
      }

      return PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
    }

    public override bool Equals(object obj)
    {
      if (obj is EncodedPositionBoards)
        return Equals((EncodedPositionBoards)obj);
      else
        return false;
    }

    public bool Equals(EncodedPositionBoards other)
    {
      return this.History_0 == other.History_0
          && this.History_1 == other.History_1
          && this.History_2 == other.History_2
          && this.History_3 == other.History_3
          && this.History_4 == other.History_4
          && this.History_5 == other.History_5
          && this.History_6 == other.History_6
          && this.History_7 == other.History_7;
    }

    public override int GetHashCode()
    {
      return HashCode.Combine(History_0, History_1, History_2, History_3, History_4, History_5, History_6, History_7);
    }
  }
}



