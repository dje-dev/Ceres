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

using Ceres.Base;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.Textual;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  public readonly unsafe struct EncodedPositionWithHistory : IEquatable<EncodedPositionWithHistory>
  {
    public const int NUM_MISC_PLANES = 8;
    public const int NUM_PLANES_TOTAL = (EncodedPositionBoards.NUM_MOVES_HISTORY * EncodedPositionBoard.NUM_PLANES_PER_BOARD) + NUM_MISC_PLANES;

    //  All planes including history (112 planes * 8 bytes)
    public readonly EncodedPositionBoards BoardsHistory;

    // Miscellaneous information
    public readonly EncodedTrainingPositionMiscInfo MiscInfo;

    const int LC0BoardSizeInBytes = 104;

    public override int GetHashCode() => HashCode.Combine(BoardsHistory.GetHashCode(), MiscInfo.GetHashCode());


    /// <summary>
    /// Constructor from a set of EncodedPositionBoards.
    /// </summary>
    /// <param name="boardHistory"></param>
    /// <param name="miscInfo"></param>
    private EncodedPositionWithHistory(EncodedPositionBoards boardHistory, EncodedTrainingPositionMiscInfo miscInfo)
    {
      BoardsHistory = boardHistory;
      MiscInfo = miscInfo;

      // if (Marshal.SizeOf<LZBoard>() != LZBoardSizeInBytes) throw new Exception("Internal error, incorrect board size");
    }


    /// <summary>
    /// Returns another EncodedPositionWithHistory contianing this position mirrored.
    /// </summary>
    public EncodedPositionWithHistory Mirrored
    {
      get
      {
        return new EncodedPositionWithHistory(new EncodedPositionBoards(BoardsHistory.History_0.Mirrored,
                                                     BoardsHistory.History_1.Mirrored,
                                                     BoardsHistory.History_2.Mirrored,
                                                     BoardsHistory.History_3.Mirrored,
                                                     BoardsHistory.History_4.Mirrored,
                                                     BoardsHistory.History_5.Mirrored,
                                                     BoardsHistory.History_6.Mirrored,
                                                     BoardsHistory.History_7.Mirrored),
                                                     MiscInfo);
      }
    }


    /// <summary>
    /// Returns the piece character for a FEN corresponding to specified square.
    /// </summary>
    /// <param name="planes"></param>
    /// <param name="index"></param>
    /// <param name="weAreWhite"></param>
    /// <param name="emptySquareChar"></param>
    /// <returns></returns>
    internal static string FENCharAt(EncodedPositionBoard planes, int index, bool weAreWhite, string emptySquareChar)
    {
      if (planes.OurPawns.BitIsSet(index)) return weAreWhite ? "P" : "p";
      if (planes.OurKnights.BitIsSet(index)) return weAreWhite ? "N" : "n";
      if (planes.OurBishops.BitIsSet(index)) return weAreWhite ? "B" : "b";
      if (planes.OurRooks.BitIsSet(index)) return weAreWhite ? "R" : "r";
      if (planes.OurQueens.BitIsSet(index)) return weAreWhite ? "Q" : "q";
      if (planes.OurKing.BitIsSet(index)) return weAreWhite ? "K" : "k";

      if (planes.TheirPawns.BitIsSet(index)) return weAreWhite ? "p" : "P";
      if (planes.TheirKnights.BitIsSet(index)) return weAreWhite ? "n" : "N";
      if (planes.TheirBishops.BitIsSet(index)) return weAreWhite ? "b" : "B";
      if (planes.TheirRooks.BitIsSet(index)) return weAreWhite ? "r" : "R";
      if (planes.TheirQueens.BitIsSet(index)) return weAreWhite ? "q" : "Q";
      if (planes.TheirKing.BitIsSet(index)) return weAreWhite ? "k" : "K";

      return emptySquareChar;
    }

    /// <summary>
    /// Gets the part of a FEN string corrrespondong to a specifid row on a specified plane in this position.
    /// </summary>
    /// <param name="startIndex"></param>
    /// <param name="planes"></param>
    /// <param name="weAreWhite"></param>
    /// <returns></returns>
    internal static string GetRowString(int startIndex, EncodedPositionBoard planes, bool weAreWhite)
    {
      string ret = "";
      for (int i = 7; i >= 0; i--)
      {
        string thisChar = FENCharAt(planes, startIndex + i, weAreWhite, ".");
        ret += thisChar;
      }
      return ret;
    }


    /// <summary>
    /// Returns string with text representations of a specified history board.
    /// </summary>
    /// <param name="historyIndex"></param>
    /// <returns></returns>
    public string BoardPictureForHistoryBoard(int historyIndex)
    {
      EncodedPositionBoard planes = GetPlanesForHistoryBoard(historyIndex);
      return planes.GetBoardPicture(true);
    }


    /// <summary>
    /// Returns string with text representations of all history boards from left to right.
    /// </summary>
    /// <returns></returns>
    public string BoardPictures()
    {
      string[] rows = new string[] { "", "", "", "", "", "", "", "", "", "" };

      for (int i = 0; i < EncodedPositionBoards.NUM_MOVES_HISTORY; i++)
      {
        EncodedPositionBoard planes = GetPlanesForHistoryBoard(i);
        string lines = planes.GetBoardPicture(true);
        string[] parsed = lines.Split('\n');


        for (int line = 0; line < parsed.Length; line++)
          rows[line] += parsed[line] + "  ";
      }

      // Concatenate the lines across all boards
      StringBuilder allLines = new StringBuilder();
      for (int i = 0; i < rows.Length; i++)
        allLines.AppendLine(rows[i]);

      return allLines.ToString();
    }

    /// <summary>
    /// Returns the Position correspondong to the final board.
    /// </summary>
    public Position FinalPosition => Position.FromFEN(FENForHistoryBoard(0));


    /// <summary>
    /// Returns the FEN string correpsonding to the position at specified history board.
    /// </summary>
    /// <param name="historyIndex"></param>
    /// <returns></returns>
    public string FENForHistoryBoard(int historyIndex)
    {
      EncodedPositionBoard planes = GetPlanesForHistoryBoard(historyIndex);

      // KQkq - 0 1
      bool weAreWhite = MiscInfo.InfoPosition.SideToMove == 0;
      string fen = MiscInfo.InfoPosition.SideToMove == 0 ? planes.GetFEN(weAreWhite) : planes.Reversed.GetFEN(weAreWhite);
      if (historyIndex != 0) return fen;

      fen = fen + (weAreWhite ? " w" : " b");
      fen = fen + " ";

      string castling = "";
      if ((weAreWhite ? MiscInfo.InfoPosition.Castling_US_OO : MiscInfo.InfoPosition.Castling_Them_OO) > 0) castling += "K";
      if ((weAreWhite ? MiscInfo.InfoPosition.Castling_US_OOO : MiscInfo.InfoPosition.Castling_Them_OOO) > 0) castling += "Q";
      if ((weAreWhite ? MiscInfo.InfoPosition.Castling_Them_OO : MiscInfo.InfoPosition.Castling_US_OO) > 0) castling += "k";
      if ((weAreWhite ? MiscInfo.InfoPosition.Castling_Them_OOO : MiscInfo.InfoPosition.Castling_US_OOO) > 0) castling += "q";
      if (castling == "") castling = "-";

      PositionMiscInfo.EnPassantFileIndexEnum enPassant = PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
      if (historyIndex < 7)
      {
        EncodedPositionBoard planesPriorBoard = GetPlanesForHistoryBoard(historyIndex + 1);
        enPassant = EncodedPositionBoards.EnPassantOpportunityBetweenBoards(planes, planesPriorBoard);
      }

      string epTarget = "-";
      if (enPassant != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
        epTarget = PositionMiscInfo.EPFileChars[(int)enPassant] + (weAreWhite ? "6" : "3");

      fen = fen + castling + " " + epTarget + " " + MiscInfo.InfoPosition.Rule50Count;// + " " + (1 + MiscInfo.InfoPosition.MoveCount); // Sometimes 2 dashes?

      return fen;
    }

    /// <summary>
    /// Overwrites the MiscInfo with a specified value.
    /// </summary>
    /// <param name="miscInfo"></param>
    public void SetMiscInfo(EncodedTrainingPositionMiscInfo miscInfo)
    {
      fixed (EncodedTrainingPositionMiscInfo* p = &MiscInfo)
      {
        *p = miscInfo;
      }
    }

    
    public void SetHistoryPlanes(in EncodedPositionBoard plane, int firstBoardIndex, int numBoards)
    {
      int lastBoardIndex = firstBoardIndex + numBoards - 1;
      if (lastBoardIndex >= EncodedPositionBoards.NUM_MOVES_HISTORY) throw new ArgumentOutOfRangeException("Incorrect number of history positions");

      fixed (EncodedPositionBoard* p = &BoardsHistory.History_0)
      {
        // TODO: memory copy for speed?
        for (int i = firstBoardIndex; i <= lastBoardIndex; i++)
          p[i] = plane;
      }

    }


    /// <summary>
    /// Extracts a specified number of history planes into an output array.
    /// </summary>
    /// <param name="numHistoryPos"></param>
    /// <param name="dest"></param>
    /// <param name="destOffset"></param>
    public void ExtractPlanesValuesIntoArray(int numHistoryPos, ulong[] dest, int destOffset)
    {
      // TO DO: Consider memory copy instead
      int offset = destOffset;

      fixed (EncodedPositionBoardPlane* boardPlanes = &BoardsHistory.History_0.OurPawns)
      fixed (ulong* destPlanes = &dest[destOffset])
      {
        int LEN = numHistoryPos * EncodedPositionBoard.NUM_PLANES_PER_BOARD * sizeof(ulong);
        Buffer.MemoryCopy(boardPlanes, destPlanes, LEN, LEN);
      }

    }



    /// <summary>
    /// Sets the history planes from a Span of EncodedPositionBoard.
    /// </summary>
    /// <param name="planes"></param>
    public void SetHistoryPlanes(Span<EncodedPositionBoard> planes)
    {
      Debug.Assert(planes.Length == EncodedPositionBoards.NUM_MOVES_HISTORY);

      uint bytes = (uint)(EncodedPositionBoards.NUM_MOVES_HISTORY * LC0BoardSizeInBytes);
      fixed (EncodedPositionBoard* planesPtr = &planes[0])
      fixed (EncodedPositionBoard* p = &BoardsHistory.History_0)
      {
        Buffer.MemoryCopy(planesPtr, p, bytes, bytes);
        // Unsafe.CopyBlockUnaligned(p, planesPtr, bytes); seems slower
      }
    }



    /// <summary>
    /// Returns the EncodedPositionBoard corresponding to a specified history baord.
    /// </summary>
    /// <param name="historyIndex"></param>
    /// <returns></returns>
    public EncodedPositionBoard GetPlanesForHistoryBoard(int historyIndex)
    {
      if (historyIndex == 0) return BoardsHistory.History_0;
      else if (historyIndex == 1) return BoardsHistory.History_1;
      else if (historyIndex == 2) return BoardsHistory.History_2;
      else if (historyIndex == 3) return BoardsHistory.History_3;
      else if (historyIndex == 4) return BoardsHistory.History_4;
      else if (historyIndex == 5) return BoardsHistory.History_5;
      else if (historyIndex == 6) return BoardsHistory.History_6;
      else if (historyIndex == 7) return BoardsHistory.History_7;
      else
        throw new Exception("bad history index " + historyIndex);

    }



    [ThreadStatic]
    static EncodedPositionBoard[] scratchBoards;


    /// <summary>
    /// Returns the scratch temporary array of EncodedPositionBoard for use by this thread.
    /// </summary>
    /// <returns></returns>
    static EncodedPositionBoard[] ScratchBoards()
    {
      if (scratchBoards == null) scratchBoards = new EncodedPositionBoard[EncodedPositionBoards.NUM_MOVES_HISTORY];
      return scratchBoards;
    }



    /// <summary>
    /// Sets the boards from a Span of Position indicating full history.
    /// </summary>
    /// <param name="sequentialPositions">sequence of positions, with the last entry being the latest move in the sequence</param>
    /// <param name="fillInMissingPlanes"></param>
    public void SetFromSequentialPositions(Span<Position> sequentialPositions, bool fillInMissingPlanes)
    {
      int LAST_POSITION_INDEX = sequentialPositions.Length - 1;

      // All the positions must be from the perspective of the side to move (last position)
      SideType sideToMove = sequentialPositions[LAST_POSITION_INDEX].MiscInfo.SideToMove;

      // Setting miscellaneous planes is easy; take from last position
      SetMiscFromPosition(sequentialPositions[LAST_POSITION_INDEX].MiscInfo, sideToMove);

      // Cache the first position in sequence from our perspective (which would be used for fill)
      bool firstBoardHadRepetition = sequentialPositions[0].MiscInfo.RepetitionCount > 0;
      EncodedPositionBoard fillBoardFromOurPerspective = EncodedPositionBoard.GetBoard(in sequentialPositions[0], sideToMove, firstBoardHadRepetition);

      Span<EncodedPositionBoard> boards = ScratchBoards();// stackalloc LZBoard[LZBoardsHistory.NUM_MOVES_HISTORY];

      SideType lastPosSide = default; // not used first time through the the loop
      for (int i = 0; i < EncodedPositionBoards.NUM_MOVES_HISTORY; i++)
      {
        if (i >= sequentialPositions.Length)
        {
          // We are past the number of boards supplied. Fill in board (only if requested)
          if (fillInMissingPlanes)
            boards[i] = fillBoardFromOurPerspective;
          else
            boards[i].Clear(); // must clear the bits since we are reusing a scratch area which may have remnants from prior position
        }
        else
        {
          // Put last positions first in board array
          ref Position thisPos = ref sequentialPositions[LAST_POSITION_INDEX - i];

          boards[i] = EncodedPositionBoard.GetBoard(in thisPos, sideToMove, thisPos.MiscInfo.RepetitionCount > 0);

#if DEBUG
          // Make sure the sides alternates between moves
          if (i > 0 && lastPosSide == thisPos.MiscInfo.SideToMove)
            throw new Exception("Sequential positions are expected to be on alternating sides");
#endif
          lastPosSide = thisPos.MiscInfo.SideToMove;
        }
      }

      SetHistoryPlanes(boards);
    }



    /// <summary>
    /// Sets the MiscInfo substructure based on specified PositionMiscInfo.
    /// </summary>
    /// <param name="posMiscInfo"></param>
    /// <param name="desiredFromSidePerspective"></param>
    public void SetMiscFromPosition(PositionMiscInfo posMiscInfo, SideType desiredFromSidePerspective)
    {
      EncodedPositionMiscInfo miscInfo = GetMiscFromPosition(posMiscInfo, desiredFromSidePerspective);
      MiscInfo.SetMisc(miscInfo.Castling_US_OOO, miscInfo.Castling_US_OO,
                       miscInfo.Castling_Them_OOO, miscInfo.Castling_Them_OO,
                       (byte)miscInfo.SideToMove, miscInfo.Rule50Count);

    }


    /// <summary>
    /// Returns an EncodedPositionMiscInfo which corresponds to a specified PositionMiscInfo
    /// from a specified perspective.
    /// </summary>
    /// <param name="posMiscInfo"></param>
    /// <param name="desiredFromSidePerspective"></param>
    /// <returns></returns>
    public static EncodedPositionMiscInfo GetMiscFromPosition(PositionMiscInfo posMiscInfo, SideType desiredFromSidePerspective)
    {
      bool weAreWhite = posMiscInfo.SideToMove != SideType.Black;

      byte ToBool(bool val) => val ? (byte)1 : (byte)0;

      byte castling_US_OOO = ToBool(weAreWhite ? posMiscInfo.WhiteCanOOO : posMiscInfo.BlackCanOOO);
      byte castling_US_OO = ToBool(weAreWhite ? posMiscInfo.WhiteCanOO : posMiscInfo.BlackCanOO);
      byte castling_Them_OOO = ToBool(!weAreWhite ? posMiscInfo.WhiteCanOOO : posMiscInfo.BlackCanOOO);
      byte castling_Them_OO = ToBool(!weAreWhite ? posMiscInfo.WhiteCanOO : posMiscInfo.BlackCanOO);

      byte sideToMove = posMiscInfo.SideToMove == SideType.White ? (byte)0 : (byte)1; // White = 0, Black = 1

      // "FEN does not represent sufficient information to decide whether a draw by threefold repetition may be legally claimed 
      // or a draw offer may be accepted; for that, a different format such as Extended Position Description is needed.

      // ** TO DO: NOTE: we don't try to set repetition below, we  could. In LZ0 we see (encoder.cc) that Repetitions is either all 1's (if position was repeated) else all 0's

      //bool flip = pos.MiscInfo.SideToMove != desiredFromSidePerspective; // WRONG: we flip above at top of this method pos.MiscInfo.SideToMove != desiredFromSidePerspective;

      byte rule50 = posMiscInfo.Move50Count > (byte)255 ? (byte)255 : (byte)posMiscInfo.Move50Count;

      return new EncodedPositionMiscInfo(castling_US_OOO, castling_US_OO,
                                            castling_Them_OOO, castling_Them_OO,
                                           (EncodedPositionMiscInfo.SideToMoveEnum)sideToMove, rule50);
    }

    
    /// <summary>
    /// Converts an EncodedTrainingPosition into a Position.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static Position PositionFromEncodedTrainingPosition(in EncodedTrainingPosition pos)
    {
      return PositionFromEncodedPosition(pos.PositionWithBoardsMirrored.Mirrored);
    }


    /// <summary>
    /// Returns a Position which is equivalent to a specified EncodedPositionWithHistory.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static Position PositionFromEncodedPosition(in EncodedPositionWithHistory pos)
    {
      EncodedPositionBoard board0 = pos.BoardsHistory.History_0;

      EncodedPositionMiscInfo sourceMiscInfo = pos.MiscInfo.InfoPosition;

      EncodedPositionBoard board1 = pos.BoardsHistory.History_1;
      PositionMiscInfo.EnPassantFileIndexEnum epColIndex = EncodedPositionBoards.EnPassantOpportunityBetweenBoards(board0, board1);

      Position pc;
      if (sourceMiscInfo.SideToMove == EncodedPositionMiscInfo.SideToMoveEnum.White)
      {
        PositionMiscInfo miscInfo = new PositionMiscInfo(sourceMiscInfo.Castling_US_OO == 1 ? true : false, sourceMiscInfo.Castling_US_OOO == 1 ? true : false,
                                                         sourceMiscInfo.Castling_Them_OO == 1 ? true : false, sourceMiscInfo.Castling_Them_OOO == 1 ? true : false,
                                                         sourceMiscInfo.SideToMove == EncodedPositionMiscInfo.SideToMoveEnum.White ? SideType.White : SideType.Black,
                                                         sourceMiscInfo.Rule50Count, (int)board0.Repetitions.Data, 0, epColIndex);

        return new Position(board0.OurKing.Data, board0.OurQueens.Data, board0.OurRooks.Data, board0.OurBishops.Data, board0.OurKnights.Data, board0.OurPawns.Data,
                            board0.TheirKing.Data, board0.TheirQueens.Data, board0.TheirRooks.Data, board0.TheirBishops.Data, board0.TheirKnights.Data, board0.TheirPawns.Data,
                            in miscInfo);
      }
      else
      {
        PositionMiscInfo miscInfo = new PositionMiscInfo(sourceMiscInfo.Castling_Them_OO == 1 ? true : false, sourceMiscInfo.Castling_Them_OOO == 1 ? true : false,
                                                         sourceMiscInfo.Castling_US_OO == 1 ? true : false, sourceMiscInfo.Castling_US_OOO == 1 ? true : false,
                                                         sourceMiscInfo.SideToMove == EncodedPositionMiscInfo.SideToMoveEnum.White ? SideType.White : SideType.Black,
                                                         sourceMiscInfo.Rule50Count, (int)board0.Repetitions.Data, 0, epColIndex);
        board0 = board0.Reversed;
        return new Position(board0.TheirKing.Data, board0.TheirQueens.Data, board0.TheirRooks.Data, board0.TheirBishops.Data, board0.TheirKnights.Data, board0.TheirPawns.Data,
                            board0.OurKing.Data, board0.OurQueens.Data, board0.OurRooks.Data, board0.OurBishops.Data, board0.OurKnights.Data, board0.OurPawns.Data,
                            in miscInfo);
      }
    }



    /// <summary>
    /// Returns an instance equivalent to a specified Position (possibly with history fill in).
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="fillInHistoryPlanes"></param>
    /// <returns></returns>
    public static EncodedPositionWithHistory FromPosition(in Position pos, bool fillInHistoryPlanes = true)
    {
      EncodedPositionWithHistory posRaw = new EncodedPositionWithHistory();
      posRaw.SetFromPosition(in pos, fillInHistoryPlanes, pos.MiscInfo.SideToMove);
      return posRaw;
    }


    /// <summary>
    /// Returns an instance equivalent to a specified position represented as a FEN (possibly with history fill in).
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="fillInHistoryPlanes"></param>
    /// <returns></returns>
    public static EncodedPositionWithHistory FromFEN(string fen, bool fillInHistoryPlanes = true)
    {
      FENParseResult parsed = FENParser.ParseFEN(fen);

      EncodedPositionWithHistory pos = new EncodedPositionWithHistory();
      pos.SetFromPosition(parsed.AsPosition, fillInHistoryPlanes, parsed.MiscInfo.SideToMove);
      return pos;
    }


    /// <summary>
    /// Initializes from a specified Position, possibly with history fill in.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="fillInHistoryPlanes"></param>
    /// <param name="desiredFromSidePerspective"></param>
    public void SetFromPosition(in Position pos, bool fillInHistoryPlanes, SideType desiredFromSidePerspective)
    {
      // Games and FENs should start with move number 1, so a value of 0 is invalid
      // Although training inputs may or may not be sensitive to his value, we still reject as invalid
      Debug.Assert(pos.MiscInfo.MoveNum > 0);

      SetMiscFromPosition(pos.MiscInfo, desiredFromSidePerspective);

      EncodedPositionBoard planes = EncodedPositionBoard.GetBoard(in pos, desiredFromSidePerspective, pos.MiscInfo.RepetitionCount > 0);

      bool hasEnPassant = pos.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
      if (hasEnPassant)
      {
        // Determine the prior position, before the move that created en passant rights.
        Position priorPosWithEnPassantUndone = pos.PosWithEnPassantUndone();

        // Get corresopnding board, flipped so it is from perspective of our side.
        EncodedPositionBoard planesPreEnPassant = EncodedPositionBoard.GetBoard(in priorPosWithEnPassantUndone, desiredFromSidePerspective, pos.MiscInfo.RepetitionCount > 1);

        if (fillInHistoryPlanes)
        {
          SetHistoryPlanes(in planes, 0, 1);
          SetHistoryPlanes(in planesPreEnPassant, 1, EncodedPositionBoards.NUM_MOVES_HISTORY - 1);
        }
        else
        {
          SetHistoryPlanes(in planes, 0, 1);
          SetHistoryPlanes(in planesPreEnPassant, 1, 2);
          ClearFillHistoryPlanes(2);
        }
      }
      else
      {
        if (fillInHistoryPlanes)
        {
          SetHistoryPlanes(in planes, 0, EncodedPositionBoards.NUM_MOVES_HISTORY);
        }
        else
        {
          SetHistoryPlanes(in planes, 0, 1);
          ClearFillHistoryPlanes(1);
        }
      }

#if NOT
      if (fillInHistoryPlanes)
        SetHistoryPlanes(planes, 0, LZBoardsHistory.NUM_MOVES_HISTORY);
      else
      {
        LZBoard emptyPlanes = new LZBoard();
        SetHistoryPlanes(planes, 0,  1);
        SetHistoryPlanes(emptyPlanes, 1, LZBoardsHistory.NUM_MOVES_HISTORY - 1); // TO DO: possibly unnneeded if initialized empty?
      }

#endif
      // TO DO: en passant (?)
#if NOT
      ss.Append(st.epSquare == SquareS.SQ_NONE ? " - " : " " + Types.square_to_string(st.epSquare) + " ");
      ss.Append(st.rule50).Append(" ").Append(1 + (gamePly - (sideToMove == ColorS.BLACK ? 1 : 0)) / 2);
#endif
    }

    private void ClearFillHistoryPlanes(int startIndex)
    {
      // Fill in other history planes
      // We don't skip this step, because:
      //   - avoids possible problem that this structure may be reused, and we need to reinitialize everything, and
      //   - for exact agreement with Leela encoding, the are some entries with bitmap=0 but value = 1.0f 
      //     and although the semantics wouldn't be wrong (product is zero), our internal unit tests expect exact agreement

      EncodedPositionBoard emptyPlanes = new EncodedPositionBoard();
      SetHistoryPlanes(in emptyPlanes, startIndex, EncodedPositionBoards.NUM_MOVES_HISTORY - startIndex);
    }


    /// <summary>
    /// Implements test for equality with another EncodedPositionWithHistory.
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public bool Equals(EncodedPositionWithHistory other) => MiscInfo.Equals(other.MiscInfo) && BoardsHistory.Equals(other.BoardsHistory);
  }


}


