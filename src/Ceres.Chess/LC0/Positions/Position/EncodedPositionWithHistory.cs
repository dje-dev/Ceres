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
using System.Text;

using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Textual;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.Positions;

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
    }


    /// <summary>
    /// Initializes from a specified single Position (without history).
    /// </summary>
    /// <param name="pos"></param>
    public EncodedPositionWithHistory(in Position pos)
    {
      SetFromPosition(in pos);
    }



    #region Access helpers

    /// <summary>
    /// Converts to a PositionWithHistory object.
    /// </summary>
    /// <param name="maxHistoryPositions"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public readonly PositionWithHistory ToPositionWithHistory(int maxHistoryPositions)
    {
      int numAdded = 0;
      Span<Position> positions = stackalloc Position[maxHistoryPositions];
      for (int i = maxHistoryPositions - 1; i >= 0; i--)
      {
        if (!GetPlanesForHistoryBoard(i).IsEmpty)
        {
          positions[numAdded++] = HistoryPosition(i);
        }
      }

      // First position may be incorrect (missing en passant)
      // since the prior history move not available to detect.
      // TODO: Try to infer this from the move actually played.
      const bool EARLIEST_POSITION_MAY_BE_MISSING_EN_PASSANT = true;
      return new PositionWithHistory(positions.Slice(0, numAdded), EARLIEST_POSITION_MAY_BE_MISSING_EN_PASSANT, false);
    }


    readonly MGMove ToMGMove(EncodedMove move)
    {
      MGPosition lastPosMG = FinalPosition.ToMGPosition;
      MGMove playedMoveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, in lastPosMG);
      return playedMoveMG;
    }


    /// <summary>
    /// Returns the best move according to the training search.
    /// </summary>
    public readonly MGMove BestMove => ToMGMove(MiscInfo.InfoTraining.BestMove);

    /// <summary>
    /// Returns the played move in the training game.
    /// </summary>
    public readonly MGMove PlayedMove => ToMGMove(MiscInfo.InfoTraining.PlayedMove);

    #endregion

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
    internal static string GetRowString(int startIndex, in EncodedPositionBoard planes, bool weAreWhite)
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
        {
          rows[line] += parsed[line] + "  ";
        }
      }

      // Concatenate the lines across all boards
      StringBuilder allLines = new StringBuilder();
      for (int i = 0; i < rows.Length; i++)
      {
        allLines.AppendLine(rows[i]);
      }

      return allLines.ToString();
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
      if (lastBoardIndex >= EncodedPositionBoards.NUM_MOVES_HISTORY)
      {
        throw new ArgumentOutOfRangeException("Incorrect number of history positions");
      }

      fixed (EncodedPositionBoard* p = &BoardsHistory.History_0)
      {
        // TODO: memory copy for speed?
        for (int i = firstBoardIndex; i <= lastBoardIndex; i++)
        {
          p[i] = plane;
        }
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

    /// <summary>
    /// Performs basic validation checks on integrity of the history positions.
    /// </summary>
    public void Validate()
    {
      if (GetPlanesForHistoryBoard(0).IsEmpty)
      {
        throw new Exception("First position is empty");
      }

      for (int i = 1; i < NUM_MISC_PLANES; i++)
      {
        if (!GetPlanesForHistoryBoard(i).IsEmpty)
        { 
          // Considering castling, maximum of 4 different piece placements could change in a single move.
          if (GetPlanesForHistoryBoard(i - 1).NumDifferentPiecePlacements(GetPlanesForHistoryBoard(i)) > 4)
          {
            int numDifferent = GetPlanesForHistoryBoard(i - 1).NumDifferentPiecePlacements(GetPlanesForHistoryBoard(i));
            throw new Exception("History planes do not look sufficiently similar, num piece placements different : " + numDifferent);
          }

          // Check for reachability
          Position posCur = HistoryPosition(i-1);
          Position posPrior = HistoryPosition(i);
          if (!MGPositionReachability.IsProbablyReachable(posPrior.ToMGPosition, posCur.ToMGPosition))
          {
            throw new Exception("FinalPosition does not appear reachable from prior position " + posPrior.FEN + " --> " + posCur.FEN);
          }
        }
      }
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
    /// <param name="fillInMissingPlanes">if history planes should be filled in if incomplete (typically necessary)</param>
    public void SetFromSequentialPositions(Span<Position> sequentialPositions, bool fillInMissingPlanes = true)
    {
      int LAST_POSITION_INDEX = sequentialPositions.Length - 1;

      // All the positions must be from the perspective of the side to move (last position)
      SideType sideToMove = sequentialPositions[LAST_POSITION_INDEX].MiscInfo.SideToMove;
      
      // Setting miscellaneous planes is easy; take from last position
      SetMiscFromPosition(sequentialPositions[LAST_POSITION_INDEX].MiscInfo);

      // Cache the first position in sequence from our perspective (which would be used for any possible fill)
      EncodedPositionBoard fillBoardFromOurPerspective = EncodedPositionBoard.FromPosition(in sequentialPositions[0], sideToMove);

      Span<EncodedPositionBoard> boards = ScratchBoards();

      SideType lastPosSide = default; // not used first time through the the loop
      for (int i = 0; i < EncodedPositionBoards.NUM_MOVES_HISTORY; i++)
      {
        if (i >= sequentialPositions.Length)
        {
          // We are past the number of boards supplied. Fill in board (only if requested)
          if (fillInMissingPlanes)
          {
            boards[i] = fillBoardFromOurPerspective;
          }
          else
          {
            boards[i].Clear(); // must clear the bits since we are reusing a scratch area which may have remnants from prior position
          }
        }
        else
        {
          // Put last positions first in board array
          ref Position thisPos = ref sequentialPositions[LAST_POSITION_INDEX - i];

          boards[i] = EncodedPositionBoard.FromPosition(in thisPos, sideToMove);

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
    /// Initializes from a specified single Position (without history).
    /// </summary>
    /// <param name="pos"></param>
    public void SetFromPosition(in Position pos) => SetFromSequentialPositions([pos], true);


    /// <summary>
    /// Scans all history positions and if any found, fills in all subsequent 
    /// with last populated history position
    /// </summary>
    public void FillInEmptyPlanes()
    {
      for (int b = 1; b < EncodedPositionBoards.NUM_MOVES_HISTORY; b++)
      {
        if (GetPlanesForHistoryBoard(b).IsEmpty)
        {
          FillStartingAtPlane(b);
          return;
        }
      }
    }


    /// <summary>
    /// Replicates the position prior to specified index into all subsequent history boards.
    /// </summary>
    /// <param name="firstEmptyBoardIndex"></param>
    void FillStartingAtPlane(int firstEmptyBoardIndex)
    {
      EncodedPositionBoard lastBoard = GetPlanesForHistoryBoard(firstEmptyBoardIndex - 1);
      for (int b = firstEmptyBoardIndex; b < EncodedPositionBoards.NUM_MOVES_HISTORY; b++)
      {
        BoardsHistory.SetBoard(b, lastBoard);
      }
    }


    /// <summary>
    /// Sets the MiscInfo substructure based on specified PositionMiscInfo.
    /// </summary>
    /// <param name="posMiscInfo"></param>
    public void SetMiscFromPosition(PositionMiscInfo posMiscInfo)
    {
      EncodedPositionMiscInfo miscInfo = GetMiscFromPosition(posMiscInfo);
      MiscInfo.SetMisc(miscInfo.Castling_US_OOO, miscInfo.Castling_US_OO,
                       miscInfo.Castling_Them_OOO, miscInfo.Castling_Them_OO,
                       (byte)miscInfo.SideToMove, miscInfo.Rule50Count);

    }


    /// <summary>
    /// Returns an EncodedPositionMiscInfo which corresponds to a specified PositionMiscInfo
    /// from a specified perspective.
    /// </summary>
    /// <param name="posMiscInfo"></param>
    /// <returns></returns>
    public static EncodedPositionMiscInfo GetMiscFromPosition(PositionMiscInfo posMiscInfo)
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

    

    static bool CheckMovedTheirPiece(in EncodedPositionBoardPlane planeAfter, in EncodedPositionBoardPlane planeBefore, ref Square destSquare)
    {
      int theirMovedPlane = System.Numerics.BitOperations.LeadingZeroCount((ulong)(planeAfter.Bits.Data & ~planeBefore.Bits.Data));
      if (theirMovedPlane < 64)
      {
        destSquare = new Square(63 - theirMovedPlane, Square.SquareIndexType.BottomToTopRightToLeft);
        return true;
      }
      return false;
    }

    public readonly (PieceType pieceType, Square fromSquare, Square toSquare, bool wasCastle) LastMoveInfoFromSideToMovePerspective()
    {
      return LastMoveInfoFromSideToMovePerspective(BoardsHistory.History_0, BoardsHistory.History_1);
    }


    static bool HAVE_WARNED = false;
    public static (PieceType pieceType, Square fromSquare, Square toSquare, bool wasCastle) LastMoveInfoFromSideToMovePerspective(in EncodedPositionBoard board0, in EncodedPositionBoard board1)
    {
      if (!HAVE_WARNED)
      {
        Console.WriteLine("LastMoveInfoFromSideToMovePerspective Method needs to be retested *after changes to mirroring policy)");
        HAVE_WARNED = true;
      }

      Square sourceSquare = default;
      Square destSquare = default;

      // Check King first (before rook) to check for possible castling
      if (CheckMovedTheirPiece(in board0.TheirKing, in board1.TheirKing, ref destSquare))
      {
        Square destSquareRook = default;
        bool wasCastle = CheckMovedTheirPiece(in board0.TheirRooks, in board1.TheirRooks, ref destSquareRook);

        CheckMovedTheirPiece(in board1.TheirKing, in board0.TheirKing, ref sourceSquare);
        return (PieceType.King, sourceSquare, destSquare, wasCastle);
      }

      if (CheckMovedTheirPiece(in board0.TheirQueens, in board1.TheirQueens, ref destSquare))
      {
        CheckMovedTheirPiece(in board1.TheirQueens, in board0.TheirQueens, ref sourceSquare);
        return (PieceType.Queen, sourceSquare, destSquare, false);
      }

      if (CheckMovedTheirPiece(in board0.TheirRooks, in board1.TheirRooks, ref destSquare))
      {
        CheckMovedTheirPiece(in board1.TheirRooks, in board0.TheirRooks, ref sourceSquare);
        return (PieceType.Rook, sourceSquare, destSquare, false);
      }
      if (CheckMovedTheirPiece(in board0.TheirBishops, in board1.TheirBishops, ref destSquare))
      {
        CheckMovedTheirPiece(in board1.TheirBishops, in board0.TheirBishops, ref sourceSquare);
        return (PieceType.Bishop, sourceSquare, destSquare, false);
      }
      if (CheckMovedTheirPiece(in board0.TheirKnights, in board1.TheirKnights, ref destSquare))
      {
        CheckMovedTheirPiece(in board1.TheirKnights, in board0.TheirKnights, ref sourceSquare);
        return (PieceType.Knight, sourceSquare, destSquare, false);
      }
      if (CheckMovedTheirPiece(in board0.TheirPawns, in board1.TheirPawns, ref destSquare))
      {
        CheckMovedTheirPiece(in board1.TheirPawns, in board0.TheirPawns, ref sourceSquare);
        return (PieceType.Pawn, sourceSquare, destSquare, false);
      }

      return default;
    }


    /// <summary>
    /// Returns if the history position at specified inde is empty.
    /// </summary>
    /// <param name="historyIndex"></param>
    /// <returns></returns>
    public bool HistoryPositionIsEmpty(int historyIndex) => BoardsHistory.BoardAtIndex(historyIndex).IsEmpty;

    /// <summary>
    /// Returns a Position which is equivalent to a specified EncodedPositionWithHistory.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public Position HistoryPosition(int historyIndex)
    {
      bool history0IsWhite = MiscInfo.WhiteToMove;
      bool thisHistoryPositionIsWhite = (historyIndex % 2 == 0) == history0IsWhite;

      EncodedPositionBoard board = BoardsHistory.BoardAtIndex(historyIndex);
      if (board.IsEmpty)
      {
        throw new ArgumentException($"The history position {historyIndex} is empty.");
      }

      if (!history0IsWhite)
      {
        board = board.ReversedAndFlipped;
      }

      // First determine en passant.
      PositionMiscInfo.EnPassantFileIndexEnum enPassant = PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
      if (historyIndex < 7)
      {
        // Read the encoded boards directly, will always be from same perspective (ours).
        EncodedPositionBoard planesPriorBoard = GetPlanesForHistoryBoard(historyIndex + 1);
        EncodedPositionBoard planesCurBoard = GetPlanesForHistoryBoard(historyIndex);

        if (!planesPriorBoard.IsEmpty)
        {
          if ((historyIndex % 2 == 1))
          {
            planesPriorBoard = planesPriorBoard.ReversedAndFlipped;
            planesCurBoard = planesCurBoard.ReversedAndFlipped;

          }

#if NOT
          if (planesCurBoard.NumDifferentPiecePlacements(in planesPriorBoard) > 4)
          {
            Console.Write("bad " + planesCurBoard.NumDifferentPiecePlacements(in planesPriorBoard));
          }
#endif
          enPassant = EncodedPositionBoards.EnPassantOpportunityBetweenBoards(planesCurBoard, in planesPriorBoard);
        }
      }

      int repetitionCount = board.Repetitions.Data > 0 ? 1 : 0;

      // First create the position without the miscellaneous info set.
      PositionMiscInfo miscInfo = default;

      Position pos = new Position(board.OurKing.Data, board.OurQueens.Data, board.OurRooks.Data, board.OurBishops.Data, board.OurKnights.Data, board.OurPawns.Data,
                                  board.TheirKing.Data, board.TheirQueens.Data, board.TheirRooks.Data, board.TheirBishops.Data, board.TheirKnights.Data, board.TheirPawns.Data,
                                  in miscInfo);

      bool whiteCanCastleOO = true;
      bool blackCanCastleOO = true;
      bool whiteCanCastleOOO = true;
      bool blackCanCastleOOO = true;
      int rule50 = 0;

      if (historyIndex == 0) // The Rule50 and castling can only be definitively determined for the first position.
      {
        rule50 = MiscInfo.InfoPosition.Rule50Count;

        if (thisHistoryPositionIsWhite)
        {
          whiteCanCastleOO = MiscInfo.InfoPosition.Castling_US_OO == 1;
          whiteCanCastleOOO = MiscInfo.InfoPosition.Castling_US_OOO == 1;
          blackCanCastleOO = MiscInfo.InfoPosition.Castling_Them_OO == 1;
          blackCanCastleOOO = MiscInfo.InfoPosition.Castling_Them_OOO == 1;
        }
        else
        {
          whiteCanCastleOO = MiscInfo.InfoPosition.Castling_Them_OO == 1;
          whiteCanCastleOOO = MiscInfo.InfoPosition.Castling_Them_OOO == 1;
          blackCanCastleOO = MiscInfo.InfoPosition.Castling_US_OO == 1;
          blackCanCastleOOO = MiscInfo.InfoPosition.Castling_US_OOO == 1;
        }
      }
      else
      {
        // Some calculations are necessary to compute plausible en passant and castling rights.
        // TODO: the PieceOnSquare calculations below could be made more efficient with direct bitmap operations.

        // Best we can do is assume castling rights exist if the king and rook are in their initial positions.
        if (pos.PieceOnSquare(SquareNames.E1) != new Piece(SideType.White, PieceType.King))
        {
          whiteCanCastleOO = false;
          whiteCanCastleOOO = false;
        }
        else
        {
          if (pos.PieceOnSquare(SquareNames.A1) != new Piece(SideType.White, PieceType.Rook))
          {
            whiteCanCastleOOO = false;
          }

          if (pos.PieceOnSquare(SquareNames.H1) != new Piece(SideType.White, PieceType.Rook))
          {
            whiteCanCastleOO = false;
          }
        }

        if (pos.PieceOnSquare(SquareNames.E8) != new Piece(SideType.Black, PieceType.King))
        {
          blackCanCastleOO = false;
          blackCanCastleOOO = false;
        }
        else
        {
          if (pos.PieceOnSquare(SquareNames.A8) != new Piece(SideType.Black, PieceType.Rook))
          {
            blackCanCastleOOO = false;
          }

          if (pos.PieceOnSquare(SquareNames.H8) != new Piece(SideType.Black, PieceType.Rook))
          {
            blackCanCastleOO = false;
          }
        }
      }

      // NOTE: move number cannot be determined, set value to 2 (which will translate to 1 ply in FEN) since 0 is considered invalid.
      miscInfo = new PositionMiscInfo(whiteCanCastleOO, whiteCanCastleOOO, blackCanCastleOO, blackCanCastleOOO,
                                    thisHistoryPositionIsWhite ? SideType.White : SideType.Black,
                                    rule50, repetitionCount, 2, enPassant);
      pos.SetMiscInfo(miscInfo);

      return pos;
    }



    /// <summary>
    /// Returns a Position which for the current move (last history position).
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public readonly Position FinalPosition => HistoryPosition(0);


    /// <summary>
    /// Returns an instance equivalent to a specified Position (possibly with history fill in).
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="fillInHistoryPlanes">if history planes should be filled in if incomplete (typically necessary)</param>
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
    /// <param name="fillInHistoryPlanes">if history planes should be filled in if incomplete (typically necessary)</param>
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
    /// <param name="fillInHistoryPlanes">if history planes should be filled in if incomplete (typically necessary)</param>
    /// <param name="desiredFromSidePerspective"></param>
    public void SetFromPosition(in Position pos, bool fillInHistoryPlanes, SideType desiredFromSidePerspective)
    {
      // Games and FENs should start with move number 1, so a value of 0 is invalid
      // Although training inputs may or may not be sensitive to his value, we still reject as invalid
      Debug.Assert(pos.MiscInfo.MoveNum > 0);

      SetMiscFromPosition(pos.MiscInfo);

      EncodedPositionBoard planes = EncodedPositionBoard.FromPosition(in pos, desiredFromSidePerspective);

      bool hasEnPassant = pos.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
      if (hasEnPassant)
      {
        // Determine the prior position, before the move that created en passant rights.
        Position priorPosWithEnPassantUndone = pos.PosWithEnPassantUndone();

        // Get corresponding board, flipped so it is from perspective of our side.
        EncodedPositionBoard planesPreEnPassant = EncodedPositionBoard.FromPosition(in priorPosWithEnPassantUndone, desiredFromSidePerspective);

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


