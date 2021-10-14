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
using System.Runtime.InteropServices;
using System.Text;
using Ceres.Base.Environment;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Represents the miscellaneous state information associated with a Position
  /// (other than the position of the pieces).
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly struct PositionMiscInfo : IEquatable<PositionMiscInfo>
  {
    public enum HashMove50Mode { Ignore, Value, ValueBoolIfAbove98 };

    [Flags]
    enum CastlingFlagsEnum : byte
    {
      None = 0,
      WhiteCanOO = 1 << 0,
      WhiteCanOOO = 1 << 1,
      BlackCanOO = 1 << 2,
      BlackCanOOO = 1 << 3
    };

    public enum EnPassantFileIndexEnum : byte
    {
      FileA = 0, FileB = 1, FileC = 2, FileD = 3,
      FileE = 4, FileF = 5, FileG = 6, FileH = 7,
      FileNone = 15
    };

    #region Raw data

    /// <summary>
    /// Flags possibly indicating en passant capture rights exist on one of the files.
    /// </summary>
    readonly byte EnPassantFileIndexAndCastlingFlags;

    /// <summary>
    /// The side to move.
    /// </summary>
    public readonly SideType SideToMove;

    /// <summary>
    /// 50 move counter - number of ply since last move that resets this counter (capture, etc.)
    /// </summary>
    public readonly byte Move50Count;

    /// <summary>
    /// Number of times this position has been repeated in the sequence.
    /// </summary>
    public readonly byte RepetitionCount;

    /// <summary>
    /// Ply number within sequence.
    /// </summary>
    public readonly short MoveNum;

    #endregion


    /// <summary>
    /// Returns a mirrored version of this PositionMiscInfo.
    /// </summary>
    public PositionMiscInfo Mirrored
    {
      get
      {
        if (CastlingRightsAny)
          throw new Exception("Cannot mirror position with castling rights due to non-equivalence.");

        EnPassantFileIndexEnum mirroredEnPassant = EnPassantFileIndexEnum.FileNone;
        if (EnPassantFileIndex != EnPassantFileIndexEnum.FileNone)
          mirroredEnPassant = (EnPassantFileIndexEnum)(7 - EnPassantFileIndex);

        return new PositionMiscInfo(false, false, false, false, SideToMove, Move50Count, RepetitionCount, MoveNum, mirroredEnPassant);
      }
    }


    /// <summary>
    /// Sets the repetition count to a specified value.
    /// </summary>
    /// <param name="count"></param>
    public unsafe void SetRepetitionCount(int count)
    {
      fixed (byte* repCount = &RepetitionCount)
        *repCount = (byte)count;
    }

    /// <summary>
    /// Returns if white can castle short.
    /// </summary>
    public bool WhiteCanOO => (EnPassantFileIndexAndCastlingFlags & (int)CastlingFlagsEnum.WhiteCanOO) != 0;

    /// <summary>
    /// Returns if white can castle long.
    /// </summary>
    public bool WhiteCanOOO => (EnPassantFileIndexAndCastlingFlags & (int)CastlingFlagsEnum.WhiteCanOOO) != 0;

    /// <summary>
    /// Returns if black can castle short.
    /// </summary>
    public bool BlackCanOO => (EnPassantFileIndexAndCastlingFlags & (int)CastlingFlagsEnum.BlackCanOO) != 0;

    /// <summary>
    /// Returns if black can castle long.
    /// </summary>
    public bool BlackCanOOO => (EnPassantFileIndexAndCastlingFlags & (int)CastlingFlagsEnum.BlackCanOOO) != 0;


    /// <summary>
    /// Returns if either side retains any castling rights.
    /// </summary>
    public bool CastlingRightsAny => (EnPassantFileIndexAndCastlingFlags & (int)0b1111) != 0;


    internal static char[] EPFileChars = new[] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' };

    /// <summary>
    /// Returns if there are any en passant rights associated with the position.
    /// </summary>
    public bool EnPassantRightsPresent => EnPassantFileIndex != EnPassantFileIndexEnum.FileNone;

    /// <summary>
    /// Returns the index of the file on which
    /// en passant rights are present (if any).
    /// </summary>
    public EnPassantFileIndexEnum EnPassantFileIndex => (EnPassantFileIndexEnum)((int)EnPassantFileIndexAndCastlingFlags >> 4);

    /// <summary>
    /// Returns the character representing the file on which 
    /// en passant rights are present (if any).
    /// </summary>
    public char EnPassantFileChar => EPFileChars[(int)EnPassantFileIndex];


    /// <summary>
    /// Returns a string representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      StringBuilder sb = new StringBuilder();
      sb.Append("{");

      sb.Append(SideToMove.ToString() + " to move, ");

      if (WhiteCanOO) sb.Append("w:OO ");
      if (WhiteCanOOO) sb.Append("w:OOO ");
      if (BlackCanOO) sb.Append("b:OO ");
      if (BlackCanOOO) sb.Append("b:OOO ");

      sb.Append(" R50:" + Move50Count);
      sb.Append(" Rep:" + RepetitionCount);
      sb.Append(" Ply:" + MoveNum);

      sb.Append("}");
      return sb.ToString();
    }


    /// <summary>
    /// Returns hash code.
    /// </summary>
    /// <returns></returns>
    public override int GetHashCode()
    {
      return HashCode.Combine(EnPassantFileIndexAndCastlingFlags, SideToMove,
                              Move50Count, RepetitionCount, MoveNum);
    }


    /// <summary>
    /// Returns a stable hashcode over the parts of this info
    /// which are used to determine position equality (for neural network evaluation).
    /// Note that the MoveCount is is not used because some nets do not utilize this info (e.g. Leela).
    /// 
    /// NOTE: this should be aligned with the implementation in LZPositionMiscInfoP(BlackCanOO ? 1 : 0)osition
    /// </summary>
    public int HashPosition(HashMove50Mode mode, bool includeRepetitions = true)
    {
      byte move50Part = 0;
      if (mode == HashMove50Mode.Value)
      {
        move50Part = Move50Count;
      }
      else if (mode == HashMove50Mode.ValueBoolIfAbove98)
      {
        // We use two levels, distinguishing between
        //   - more than 98 ply, so about to be a possible draw
        //   - more than 80 ply, getting close and drawish but not yet imminent
        if (Move50Count >= 98)
        {
          move50Part = 2;
        }
        else if (Move50Count > 80)
        {
          move50Part = 1;
        }
      }

      int repetitionPart;
      if (includeRepetitions)
      {
#if NOT
        // NOTE: attempts to rework this yielded circa 
        // -10Elo change (when tested without tablebases)
        if (CeresEnvironment.TEST_MODE)
          repetitionPart = RepetitionCount >= 2 ? 2 : 0; // -15 +/- 11 with no tablebases
        //repetitionPart = RepetitionCount >= 1 ? 1 : 0; // -12+/- 9 with no tablebases, 70k
        else
#endif
        repetitionPart = RepetitionCount >= 2 ? 2 : RepetitionCount;
      }
      else
      {
        repetitionPart = 0;
      }

      if (SideToMove == SideType.White)
      {
        return HashCode.Combine((BlackCanOO ? 1 : 0), (BlackCanOOO ? 1 : 0),
                                (WhiteCanOO ? 1 : 0), (WhiteCanOOO ? 1 : 0),
                                 move50Part, repetitionPart, SideToMove);
      }
      else
      {
        return HashCode.Combine((WhiteCanOO ? 1 : 0), (WhiteCanOOO ? 1 : 0),
                                (BlackCanOO ? 1 : 0), (BlackCanOOO ? 1 : 0),
                                move50Part, repetitionPart, SideToMove);
      }
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="whiteCanCastleOO"></param>
    /// <param name="whiteCanCastleOOO"></param>
    /// <param name="blackCanCastleOO"></param>
    /// <param name="blackCanCastleOOO"></param>
    /// <param name="sideToMove"></param>
    /// <param name="move50Count"></param>
    /// <param name="repetitionCount"></param>
    /// <param name="moveNum"></param>
    /// <param name="enPassantColIndex"></param>
    public PositionMiscInfo(bool whiteCanCastleOO, bool whiteCanCastleOOO,
                            bool blackCanCastleOO, bool blackCanCastleOOO,
                            SideType sideToMove, int move50Count, int repetitionCount, int moveNum, EnPassantFileIndexEnum enPassantColIndex)
    {
      int epc = 0;
      if (whiteCanCastleOO) epc |= (int)CastlingFlagsEnum.WhiteCanOO;
      if (whiteCanCastleOOO) epc |= (int)CastlingFlagsEnum.WhiteCanOOO;
      if (blackCanCastleOO) epc |= (int)CastlingFlagsEnum.BlackCanOO;
      if (blackCanCastleOOO) epc |= (int)CastlingFlagsEnum.BlackCanOOO;

      epc |= (byte)enPassantColIndex * 16;
      EnPassantFileIndexAndCastlingFlags = (byte)epc;

      SideToMove = sideToMove;
      Move50Count = move50Count > 255 ? (byte)255 : (byte)move50Count;
      RepetitionCount = repetitionCount > 3 ? (byte)3 : (byte)repetitionCount;
      MoveNum = (short)moveNum;
    }


    /// <summary>
    /// Tests for equality with another PositionMiscInfo
    /// (using chess semantics which implies that the MoveNum is irrelevant).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public bool Equals(PositionMiscInfo other)
    {
      return SideToMove == other.SideToMove
        && EnPassantFileIndexAndCastlingFlags == other.EnPassantFileIndexAndCastlingFlags
        && Move50Count == other.Move50Count
        && RepetitionCount == other.RepetitionCount;
    }

  }

}
