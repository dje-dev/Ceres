#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region License

/* 
License Note
   
This code originated from Github repository from Judd Niemann
and is licensed with the MIT License.

This version is modified by David Elliott, including a translation to C# and 
some moderate modifications to improve performance and modularity.
*/

/*

MIT License

Copyright(c) 2016-2017 Judd Niemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#region Using directives

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

#endregion

using BitBoard = System.UInt64;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Move generator.
  /// 
  /// Note that we dont' set the check flag (_FLAG_CHECKS_IN_MOVE_GENERATION_), 
  /// for two reasons:
  ///   - it's expensive (slows move generation by 30%), and
  ///   - the code does not yet properly handle computing this for promotions
  /// </summary>
  public static partial class MGMoveGen
  {
    public static ulong MoveGenCount = 0;

    public static void GenerateMoves(in MGPosition P, MGMoveList moves)
    {
      Debug.Assert((~(P.A | P.B | P.C) & P.D) == 0); // Should not be any "black" empty squares

#if DEBUG
      MoveGenCount++;
#endif

      if (P.BlackToMove)
        DoGenBlackMoves(in P, moves, MoveGenMode.AllMoves);
      else
        DoGenWhiteMoves(in P, moves, MoveGenMode.AllMoves);
    }

    static void AddWhiteKnightMoves(in MGPosition P, MGMoveList moves, byte q, BitBoard WhiteFree)
    {
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight8[q] & WhiteFree, MGPositionConstants.WKNIGHT);
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight7[q] & WhiteFree, MGPositionConstants.WKNIGHT);
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight1[q] & WhiteFree, MGPositionConstants.WKNIGHT);
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight2[q] & WhiteFree, MGPositionConstants.WKNIGHT);
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight3[q] & WhiteFree, MGPositionConstants.WKNIGHT);
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight4[q] & WhiteFree, MGPositionConstants.WKNIGHT);
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight5[q] & WhiteFree, MGPositionConstants.WKNIGHT);
      AddWhiteMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight6[q] & WhiteFree, MGPositionConstants.WKNIGHT);
    }

    static void AddBlackKnightMoves(in MGPosition P, MGMoveList moves, byte q, BitBoard BlackFree)
    {
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight1[q] & BlackFree, MGPositionConstants.BKNIGHT);
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight2[q] & BlackFree, MGPositionConstants.BKNIGHT);
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight3[q] & BlackFree, MGPositionConstants.BKNIGHT);
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight4[q] & BlackFree, MGPositionConstants.BKNIGHT);
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight5[q] & BlackFree, MGPositionConstants.BKNIGHT);
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight6[q] & BlackFree, MGPositionConstants.BKNIGHT);
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight7[q] & BlackFree, MGPositionConstants.BKNIGHT);
      AddBlackMoveToListIfLegal(in P, moves, q, MGPositionConstants.MoveKnight8[q] & BlackFree, MGPositionConstants.BKNIGHT);
    }

    [ThreadStatic]
    static MGMoveList movesTemp;

    public enum MoveGenMode {  AllMoves, AtLeastOneMoveIfAnyExists };


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool AtLeastOneLegalMoveExists(in MGPosition P)
    {
      if (movesTemp == null) movesTemp = new MGMoveList();
      movesTemp.NumMovesUsed = 0;

      if (P.BlackToMove)
        DoGenBlackMoves(in P, movesTemp, MoveGenMode.AtLeastOneMoveIfAnyExists);
      else
        DoGenWhiteMoves(in P, movesTemp, MoveGenMode.AtLeastOneMoveIfAnyExists);

      return movesTemp.NumMovesUsed > 0;
    }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void DoGenWhiteMoves(in MGPosition P, MGMoveList moves, MoveGenMode mode)
    {
      Debug.Assert(moves.NumMovesUsed == 0);

      BitBoard occupied = P.A | P.B | P.C;                // all squares occupied by something
      BitBoard pABCTemp = (P.A & P.B & ~P.C);
      BitBoard whiteOccupied = (occupied & ~P.D) & ~pABCTemp; // all squares occupied by W, excluding EP Squares
      BitBoard blackOccupied = occupied & P.D;              // all squares occupied by B, including Black EP Squares
      BitBoard whiteFree;       // all squares where W is free to move
      whiteFree = pABCTemp  // any EP square 
        | ~(occupied)       // any vacant square
        | (~P.A & P.D)      // Black Bishop, Rook or Queen
        | (~P.B & P.D);     // Black Pawn or Knight

      Debug.Assert(whiteOccupied != 0);

      BitBoard square;
      BitBoard currentSquare;
      BitBoard A, B, C;
      MGMove M = new MGMove(); // Dummy Move object used for setting flags.

      BitBoard occupiedYetToBeProcessed = occupied;
      byte q;
      do
      {
        // For efficiency, exit immediately if we are in "at least one move" mode and we have seen one or more moves
        if (mode == MoveGenMode.AtLeastOneMoveIfAnyExists && moves.NumMovesUsed > 0) return;

        q = (byte)System.Numerics.BitOperations.TrailingZeroCount(occupiedYetToBeProcessed);
        if (q < 64)
        {
          currentSquare = 1UL << (int)q;
          occupiedYetToBeProcessed = occupiedYetToBeProcessed ^ currentSquare;

          if ((whiteOccupied & currentSquare) == 0)
            continue; // square empty - nothing to do
          A = P.A & currentSquare;
          B = P.B & currentSquare;
          C = P.C & currentSquare;

          if (A != 0)
          {
            if (B == 0)
            {
              if (C == 0)
              {
                // single move forward 
                square = MGPositionConstants.MoveUp[q] & whiteFree & ~blackOccupied /* pawns can't capture in forward moves */;
                if ((square & MGPositionConstants.RANK8) != 0)
                  AddWhitePromotionsToListIfLegal(in P, moves, q, square, MGPositionConstants.WPAWN);
                else
                {
                  // Ordinary Pawn Advance
                  AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WPAWN);
                  /* double move forward (only available from 2nd Rank */
                  if ((currentSquare & MGPositionConstants.RANK2) != 0)
                  {
                    square = MGMoveGenFillFunctions.MoveUpSingleOccluded(square, whiteFree) & ~blackOccupied;
                    M.Flags = 0;
                    M.DoublePawnMove = true; // this flag will cause ChessPosition::performMove() to set an ep square in the position
                    AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WPAWN, M.Flags);
                  }
                }
                // generate Pawn Captures:
                square = MGPositionConstants.MoveUpLeft[q] & whiteFree & blackOccupied;
                if ((square & MGPositionConstants.RANK8) != 0)
                  AddWhitePromotionsToListIfLegal(in P, moves, q, square, MGPositionConstants.WPAWN);
                else
                {
                  // Ordinary Pawn Capture to Left
                  AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WPAWN);
                }
                square = MGPositionConstants.MoveUpRight[q] & whiteFree & blackOccupied;
                if ((square & MGPositionConstants.RANK8) != 0)
                  AddWhitePromotionsToListIfLegal(in P, moves, q, square, MGPositionConstants.WPAWN);
                else
                {
                  // Ordinary Pawn Capture to right
                  AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WPAWN);
                }
                continue;
              }
              else
              {
                AddWhiteKnightMoves(in P, moves, q, whiteFree);
                continue;
              }
            } // Ends if (B == 0)

            else /* B != 0 */
            {
              if (C != 0)
              {
                square = MGPositionConstants.MoveUp[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);

                square = MGPositionConstants.MoveRight[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);
                
                bool isWhiteInCheck = IsWhiteInCheck(P.A, P.B, P.C, P.D);
                BitBoard rookPos = ~P.A & ~P.B & P.C & ~P.D;

                // Conditionally generate O-O move:
                if (
                  (currentSquare == MGPositionConstants.WHITEKINGPOS) &&          // King is in correct Position AND
                  (P.WhiteCanCastle) &&               // White still has castle rights AND
                  (MGPositionConstants.WHITECASTLEZONE & occupied) == 0 &&        // Castle Zone (f1,g1) is clear AND
                  ((rookPos & MGPositionConstants.WHITEKRPOS) != 0) && // KRook is in correct Position AND
                  (!moves.MovesArray[moves.NumMovesUsed].IllegalMove) &&          // Last generated move (1 step to right) was legal AND
                  !isWhiteInCheck                                                 // King is not in Check
                  )
                {
                  // OK to Castle
                  square = MGPositionConstants.G1;                    // Move King to g1
                  M.Flags = 0;
                  M.CastleShort = true;
                  AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING, M.Flags);
                }

                square = MGPositionConstants.MoveDown[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);

                square = MGPositionConstants.MoveLeft[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);

                // Conditionally generate O-O-O move:
                if (
                  (currentSquare == MGPositionConstants.WHITEKINGPOS) &&          // King is in correct Position AND
                  (P.WhiteCanCastleLong) &&             // White still has castle-long rights AND
                  (MGPositionConstants.WHITECASTLELONGZONE & occupied) == 0 &&    // Castle Long Zone (b1,c1,d1) is clear AND	
                  ((rookPos & MGPositionConstants.WHITEQRPOS) != 0) && // QRook is in correct Position AND
                  (!moves.MovesArray[moves.NumMovesUsed].IllegalMove) &&          // Last generated move (1 step to left) was legal AND
                  !isWhiteInCheck                                                 // King is not in Check
                  )
                {
                  // Ok to Castle Long
                  square = MGPositionConstants.C1;                    // Move King to c1
                  M.Flags = 0;
                  M.CastleLong = true;
                  AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING, M.Flags);
                }

                square = MGPositionConstants.MoveUpRight[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);
                square = MGPositionConstants.MoveDownRight[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);
                square = MGPositionConstants.MoveDownLeft[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);
                square = MGPositionConstants.MoveUpLeft[q] & whiteFree;
                AddWhiteMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.WKING);

                continue;
              } // ENDS if (C != 0)
              else
              {
                // en passant square - no action to be taken, but continue loop
                continue;
              }
            } // Ends else
          } // ENDS if (A !=0)

          BitBoard SolidBlackPiece = P.D & ~(P.A & P.B); // All black pieces except enpassants and black king

          if (B != 0)
          {
            // Piece can do diagonal moves (it's either a B or Q)
            ulong piece = C == 0 ? MGPositionConstants.WBISHOP : MGPositionConstants.WQUEEN;

            square = currentSquare;
            do
            {
              // Diagonal UpRight 
              square = MGMoveGenFillFunctions.MoveUpRightSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);

            square = currentSquare;
            do
            { /* Diagonal DownRight */
              square = MGMoveGenFillFunctions.MoveDownRightSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);

            square = currentSquare;
            do
            {
              // Diagonal DownLeft 
              square = MGMoveGenFillFunctions.MoveDownLeftSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);

            square = currentSquare;
            do
            {
              // Diagonal UpLeft 
              square = MGMoveGenFillFunctions.MoveUpLeftSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);
          }

          if (C != 0)
          {
            // Piece can do straight moves (it's either a R or Q)
            ulong piece = B == 0 ? MGPositionConstants.WROOK : MGPositionConstants.WQUEEN;

            square = currentSquare;
            do
            {
              // Up 
              square = MGMoveGenFillFunctions.MoveUpSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);

            square = currentSquare;
            do
            {
              // Right 
              square = MGMoveGenFillFunctions.MoveRightSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);

            square = currentSquare;
            do
            {
              // Down 
              square = MGMoveGenFillFunctions.MoveDownSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);

            square = currentSquare;
            do
            {
              // Left
              square = MGMoveGenFillFunctions.MoveLeftSingleOccluded(square, whiteFree);
              AddWhiteMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidBlackPiece) != 0);
          }

        }
        else
          break;
      } while (true);

//      moves.MovesArray[0].MoveCount = (byte)(moves.NumMovesUsed);

      // Create 'no more moves' move to mark end of list:	
      moves.MovesArray[moves.NumMovesUsed].FromSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].ToSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].Piece = 0;
      moves.MovesArray[moves.NumMovesUsed].NoMoreMoves = true;
    }



    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void AddWhiteMoveToListIfLegal(in MGPosition P, MGMoveList moves, byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      if (to != 0)
      {
        DoAddWhiteMoveToListIfLegal(in P, moves, fromsquare, to, piece, flags);
      }
    }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void DoAddWhiteMoveToListIfLegal(in MGPosition P, MGMoveList moves, byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      moves.InsureMoveArrayHasRoom(1);

      ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

      thisMove.FromSquareIndex = fromsquare;
      thisMove.ToSquareIndex = MGMoveGenFillFunctions.GetSquareIndex(to);
      thisMove.Flags = flags;
      thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

      BitBoard O = (ulong)~((1UL << fromsquare) | (ulong)to);

      BitBoard QA, QB, QC, QD;

      if (thisMove.CastleShort)
      {
        QA = P.A ^ 0x000000000000000a;
        QB = P.B ^ 0x000000000000000a;
        QC = P.C ^ 0x000000000000000f;
        QD = P.D;
        //			Q.D &= 0xfffffffffffffff0;	// clear colour of e1,f1,g1,h1 (make white)
      }
      else if (thisMove.CastleLong)
      {
        QA = P.A ^ 0x0000000000000028;
        QB = P.B ^ 0x0000000000000028;
        QC = P.C ^ 0x00000000000000b8;
        QD = P.D;
        //			Q.D &= 0xffffffffffffff07;	// clear colour of a1,b1,c1,d1,e1 (make white)
      }
      else
      {
        // clear old and new square:  
        QA = P.A & O;
        QB = P.B & O;
        QC = P.C & O;
        QD = P.D & O;

        // Populate new square (Branchless method):
        QA |= (piece & 1) << thisMove.ToSquareIndex;
        QB |= ((piece & 2) >> 1) << thisMove.ToSquareIndex;
        QC |= ((piece & 4) >> 2) << thisMove.ToSquareIndex;
        QD |= ((piece & 8) >> 3) << thisMove.ToSquareIndex;

        // Test for capture:
        BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings:
        if ((to & P.D) != 0)
        {
          if ((to & ~PAB) != 0) // Only considered a capture if dest is not an enpassant or king.
            thisMove.Capture = true;

          else if ((piece == MGPositionConstants.WPAWN) && (to & P.D & PAB & ~P.C) != 0)
          {
            thisMove.EnPassantCapture = true;
            // remove the actual pawn (as well as the ep square)
            to >>= 8;
            QA &= ~to;
            QB &= ~to;
            QC &= ~to;
            QD &= ~to;
          }
        }
      }

      if (!IsWhiteInCheck(QA, QB, QC, QD))                   // Does proposed move put our (white) king in Check ?
      {
        moves.NumMovesUsed++;     // Advancing the pointer means that the 
                                  // move is now added to the list.
                                  // (pointer is ready for next move)
        moves.MovesArray[moves.NumMovesUsed].Flags = 0;

#if _FLAG_CHECKS_IN_MOVE_GENERATION
        if (IsBlackInCheck(QA, QB, QC, QD))                  // Does the move put enemy (black) king in Check ?
          moves.MovesArray[moves.NumMovesUsed].Check = true;
#endif
      }
      else
        thisMove.IllegalMove = true;

    }


    static void AddWhitePromotionsToListIfLegal(in MGPosition P, MGMoveList moves, 
                                                byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
   {
    if (to != 0)
    {
      moves.InsureMoveArrayHasRoom(4);

      ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

      thisMove.FromSquareIndex = fromsquare;
      thisMove.ToSquareIndex = MGMoveGenFillFunctions.GetSquareIndex(to);
      thisMove.Flags = flags;
      thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

      BitBoard O = ~((1UL << fromsquare) | to);

      // clear old and new square
      BitBoard QA = P.A & O;
      BitBoard QB = P.B & O;
      BitBoard QC = P.C & O;
      BitBoard QD = P.D & O;

      // Populate new square with Queen:
      QB |= to;
      QC |= to;
    // Q.D |= to;  (DJE) **** TO DO: This line is present in the black version, why was it not present here??? possibly this is correct?

      // Test for capture:
      BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings:
      moves.MovesArray[moves.NumMovesUsed].Capture = (to & P.D & ~PAB) != 0;

        if (!IsWhiteInCheck(QA, QB, QC, QD))                   // Does proposed move put our (white) king in Check ?
        {
#if NOT
          // TODO: Currently we do not set the Check flag for promotion moves,
          //       since this is complex
#if _FLAG_CHECKS_IN_MOVE_GENERATION
          //      NOTE: (DJE)why is similar code not replicated in the 3 additional places below, like we see for the black version of the code?
          // To-do: Promote to new piece (can potentially check opponent)
          if (IsBlackInCheck(QA, QB, QC, QD))                  // Does the move put enemy (black) king in Check ?
            moves.MovesArray[moves.NumMovesUsed].Check = true;
#endif
#endif

          // make an additional 3 copies (there are four promotions)
          moves.MovesArray[moves.NumMovesUsed + 1] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 2] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 3] = thisMove;

          // set Promotion flags accordingly:
          moves.MovesArray[moves.NumMovesUsed].PromoteKnight = true;
          moves.MovesArray[moves.NumMovesUsed + 1].PromoteBishop = true;
          moves.MovesArray[moves.NumMovesUsed + 2].PromoteRook = true;
          moves.MovesArray[moves.NumMovesUsed + 3].PromoteQueen = true;

          moves.MovesArray[moves.NumMovesUsed + 4].Flags = 0;
          moves.NumMovesUsed += 4;
        }
        else
          moves.MovesArray[moves.NumMovesUsed].IllegalMove = true;
        }
      }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void DoGenBlackMoves(in MGPosition P, MGMoveList moves, MoveGenMode mode)
    {
      Debug.Assert(moves.NumMovesUsed == 0);

      BitBoard occupied = P.A | P.B | P.C;                // all squares occupied by something
      BitBoard pABCTemp = (P.A & P.B & ~P.C);
      BitBoard blackOccupied = P.D & ~pABCTemp;         // all squares occupied by B, excluding EP Squares
      BitBoard whiteOccupied = (occupied & ~P.D);             // all squares occupied by W, including white EP Squares
      BitBoard blackFree;       // all squares where B is free to move
      blackFree = pABCTemp  // any EP square 
        | ~(occupied)       // any vacant square
        | (~P.A & ~P.D)       // White Bishop, Rook or Queen
        | (~P.B & ~P.D);      // White Pawn or Knight

      Debug.Assert(blackOccupied != 0);

      BitBoard square;
      BitBoard currentSquare;
      BitBoard A, B, C;
      MGMove M = new MGMove(); // Dummy Move object used for setting flags.

      BitBoard occupiedYetToBeProcessed = occupied;
      byte q;
      do
      {
        // For efficiency, exit immediately if we are in "at least one move" mode and we have seen one or more moves
        if (mode == MoveGenMode.AtLeastOneMoveIfAnyExists && moves.NumMovesUsed > 0) return;

        q = (byte)System.Numerics.BitOperations.TrailingZeroCount(occupiedYetToBeProcessed);
        if (q < 64)
        {
          currentSquare = 1UL << (int)q;
          occupiedYetToBeProcessed = occupiedYetToBeProcessed ^ currentSquare;
          if ((blackOccupied & currentSquare) == 0)
            continue; // square empty - nothing to do

          A = P.A & currentSquare;
          B = P.B & currentSquare;
          C = P.C & currentSquare;

          if (A != 0)
          {
            if (B == 0)
            {
              if (C == 0)
              {
                // single move forward 
                square = MGPositionConstants.MoveDown[q] & blackFree & ~whiteOccupied /* pawns can't capture in forward moves */;
                if ((square & MGPositionConstants.RANK1) != 0)
                  AddBlackPromotionsToListIfLegal(in P, moves, q, square, MGPositionConstants.BPAWN);
                else
                {
                  // Ordinary Pawn Advance
                  AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BPAWN);
                  /* double move forward (only available from 7th Rank */
                  if ((currentSquare & MGPositionConstants.RANK7) != 0)
                  {
                    square = MGMoveGenFillFunctions.MoveDownSingleOccluded(square, blackFree) & ~whiteOccupied;
                    M.Flags = 0;
                    M.DoublePawnMove = true; // this flag will cause ChessPosition::performMove() to set an ep square in the position
                    AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BPAWN, M.Flags);
                  }
                }
                // generate Pawn Captures:
                square = MGPositionConstants.MoveDownLeft[q] & blackFree & whiteOccupied;
                if ((square & MGPositionConstants.RANK1) != 0)
                  AddBlackPromotionsToListIfLegal(in P, moves, q, square, MGPositionConstants.BPAWN);
                else
                {
                  // Ordinary Pawn Capture to Left
                  AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BPAWN);
                }
                square = MGPositionConstants.MoveDownRight[q] & blackFree & whiteOccupied;
                if ((square & MGPositionConstants.RANK1) != 0)
                  AddBlackPromotionsToListIfLegal(in P, moves, q, square, MGPositionConstants.BPAWN);
                else
                {
                  // Ordinary Pawn Capture to right
                  AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BPAWN);
                }
                continue;
              }
              else
              {
                AddBlackKnightMoves(in P, moves, q, blackFree);
                continue;
              }
            } // ENDS if (B ==0)

            else /* B != 0 */
            {
              if (C != 0)
              {
                square = MGPositionConstants.MoveUp[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);

                square = MGPositionConstants.MoveRight[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);

                bool isBlackInCheck = IsBlackInCheck(P.A, P.B, P.C, P.D);
                BitBoard rookPos = ~P.A & ~P.B & P.C & P.D;

                // Conditionally generate O-O move:
                if (
                  (currentSquare == MGPositionConstants.BLACKKINGPOS) &&          // King is in correct Position AND
                  (P.BlackCanCastle) &&               // Black still has castle rights AND
                  (MGPositionConstants.BLACKCASTLEZONE & occupied) == 0 &&        // Castle Zone (f8,g8) is clear	AND
                  ((rookPos & MGPositionConstants.BLACKKRPOS) != 0) &&  // KRook is in correct Position AND
                  (!moves.MovesArray[moves.NumMovesUsed].IllegalMove) &&          // Last generated move (1 step to right) was legal AND
                  !isBlackInCheck                                                 // King is not in Check
                  )
                {
                  // OK to Castle
                  square = MGPositionConstants.G8; // Move King to g8
                  M.Flags = 0;
                  M.CastleShort = true;
                  AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING, M.Flags);
                }

                square = MGPositionConstants.MoveDown[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);

                square = MGPositionConstants.MoveLeft[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);

                // Conditionally generate O-O-O move:
                if (
                  (currentSquare == MGPositionConstants.BLACKKINGPOS) &&        // King is in correct Position AND
                  (P.BlackCanCastleLong) &&                                     // Black still has castle-long rights AND
                  ((rookPos & MGPositionConstants.BLACKQRPOS) != 0) &&          // QRook is in correct Position AND
                  (!moves.MovesArray[moves.NumMovesUsed].IllegalMove) &&        // Last generated move (1 step to left) was legal AND
                  (MGPositionConstants.BLACKCASTLELONGZONE & occupied) == 0 &&  // Castle Long Zone (b8,c8,d8) is clear
                  !isBlackInCheck                                               // King is not in Check
                  )
                {
                  // OK to castle Long
                  square = MGPositionConstants.C8;                    // Move King to c8
                  M.Flags = 0;
                  M.CastleLong = true;
                  AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING, M.Flags);
                }

                square = MGPositionConstants.MoveUpRight[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);
                square = MGPositionConstants.MoveDownRight[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);
                square = MGPositionConstants.MoveDownLeft[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);
                square = MGPositionConstants.MoveUpLeft[q] & blackFree;
                AddBlackMoveToListIfLegal(in P, moves, q, square, MGPositionConstants.BKING);
                continue;
              } // ENDS if (C != 0)
              else
              {
                // en passant square - no action to be taken, but continue loop
                continue;
              }
            } // Ends else
          } // ENDS if (A !=0)

          BitBoard SolidWhitePiece = whiteOccupied & ~(P.A & P.B); // All white pieces except enpassants and white king

          if (B != 0)
          {
            // Piece can do diagonal moves (it's either a B or Q)
            ulong piece = C == 0 ? MGPositionConstants.BBISHOP : MGPositionConstants.BQUEEN;

            square = currentSquare;
            do
            {
              // Diagonal UpRight 
              square = MGMoveGenFillFunctions.MoveUpRightSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);

            square = currentSquare;
            do
            {
              // Diagonal DownRight 
              square = MGMoveGenFillFunctions.MoveDownRightSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);

            square = currentSquare;
            do
            {
              // Diagonal DownLeft 
              square = MGMoveGenFillFunctions.MoveDownLeftSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);

            square = currentSquare;
            do
            {
              // Diagonal UpLeft 
              square = MGMoveGenFillFunctions.MoveUpLeftSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);
          }

          if (C != 0)
          {
            // Piece can do straight moves (it's either a R or Q)
            ulong piece = (B == 0) ? MGPositionConstants.BROOK : MGPositionConstants.BQUEEN;

            square = currentSquare;
            do
            {
              // Up
              square = MGMoveGenFillFunctions.MoveUpSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);

            square = currentSquare;
            do
            {
              // Right 
              square = MGMoveGenFillFunctions.MoveRightSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);

            square = currentSquare;
            do
            {
              // Down 
              square = MGMoveGenFillFunctions.MoveDownSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);

            square = currentSquare;
            do
            {
              // Left
              square = MGMoveGenFillFunctions.MoveLeftSingleOccluded(square, blackFree);
              AddBlackMoveToListIfLegal(in P, moves, q, square, piece);
            } while ((square & ~SolidWhitePiece) != 0);
          }
        }
        else
          break;
      } while (true);

//      moves.MovesArray[0].MoveCount = (byte)moves.NumMovesUsed;

      // Create 'no more moves' move to mark end of list:	
      moves.MovesArray[moves.NumMovesUsed].FromSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].ToSquareIndex = 0;
      moves.MovesArray[moves.NumMovesUsed].Piece = 0;
      moves.MovesArray[moves.NumMovesUsed].NoMoreMoves = true;

    }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void AddBlackMoveToListIfLegal(in MGPosition P, MGMoveList moves, byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      if (to != 0)
      {
        DoAddBlackMoveToListIfLegal(in P, moves, fromsquare, to, piece, flags);
      }
    }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void DoAddBlackMoveToListIfLegal(in MGPosition P, MGMoveList moves, byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags = 0)
    {
      moves.InsureMoveArrayHasRoom(1);

      ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

      thisMove.FromSquareIndex = fromsquare;
      thisMove.ToSquareIndex = MGMoveGenFillFunctions.GetSquareIndex(to);
      thisMove.Flags = flags;
      thisMove.BlackToMove = true;
      thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

      BitBoard O = (ulong)~((1UL << fromsquare) | (ulong)to);

      BitBoard QA, QB, QC, QD;

      if (thisMove.CastleShort)
      {
        QA = P.A ^ 0x0a00000000000000;
        QB = P.B ^ 0x0a00000000000000;
        QC = P.C ^ 0x0f00000000000000;
        QD = P.D ^ 0x0f00000000000000;
      }
      else if (thisMove.CastleLong)
      {
        QA = P.A ^ 0x2800000000000000;
        QB = P.B ^ 0x2800000000000000;
        QC = P.C ^ 0xb800000000000000;
        QD = P.D ^ 0xb800000000000000;
      }
      else
      {
        // clear old and new square  
        QA = P.A & O;
        QB = P.B & O;
        QC = P.C & O;
        QD = P.D & O;

        // Populate new square (Branchless method):
        QA |= (piece & 1) << thisMove.ToSquareIndex;
        QB |= ((piece & 2) >> 1) << thisMove.ToSquareIndex;
        QC |= ((piece & 4) >> 2) << thisMove.ToSquareIndex;
        QD |= ((piece & 8) >> 3) << thisMove.ToSquareIndex;

        // Test for capture:
        BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings
        BitBoard WhiteOccupied = (P.A | P.B | P.C) & ~P.D;
        if ((to & WhiteOccupied) != 0)
        {
          if ((to & ~PAB) != 0) // Only considered a capture if dest is not an enpassant or king.
            thisMove.Capture = true;

          //            else if ((piece == PositionConstants.BPAWN) && (to & WhiteOccupied & PAB & ~P.C))
          else if ((piece == MGPositionConstants.BPAWN) && (to & WhiteOccupied & PAB & ~P.C) != 0)
          {
            thisMove.EnPassantCapture = true;
            // remove the actual pawn (as well as the ep square)
            to <<= 8;
            QA &= ~to; // clear the pawn's square
            QB &= ~to;
            QC &= ~to;
            QD &= ~to;
          }
        }
      }

      if (!IsBlackInCheck(QA, QB, QC, QD)) // Does proposed move put our (black) king in Check ?
      {
        moves.NumMovesUsed++;     // Advancing the pointer means that the 
                                  // move is now added to the list.
                                  // (pointer is ready for next move)
        moves.MovesArray[moves.NumMovesUsed].Flags = 0;
#if _FLAG_CHECKS_IN_MOVE_GENERATION
        if (IsWhiteInCheck(QA, QB, QC, QD))  // Does the move put enemy (white) king in Check ?
          moves.MovesArray[moves.NumMovesUsed].Check = true;
#endif

      }
      else
        thisMove.IllegalMove = true;
    }


    static void AddBlackPromotionsToListIfLegal(in MGPosition P, MGMoveList moves, byte fromsquare, BitBoard to, ulong piece, MGMove.MGChessMoveFlags flags=0)
    {
      if (to != 0)
      {
        moves.InsureMoveArrayHasRoom(4);

        ref MGMove thisMove = ref moves.MovesArray[moves.NumMovesUsed];

        thisMove.FromSquareIndex = fromsquare;
        thisMove.ToSquareIndex = MGMoveGenFillFunctions.GetSquareIndex(to);
        thisMove.Flags = flags;
        thisMove.BlackToMove = true;
        thisMove.Piece = (MGPositionConstants.MCChessPositionPieceEnum)piece;

        BitBoard O = ~((1UL << fromsquare) | to);

        // clear old and new square  
        BitBoard QA = P.A & O;
        BitBoard QB = P.B & O;
        BitBoard QC = P.C & O;
        BitBoard QD = P.D & O;

        // Populate new square with Queen:
        QB |= to;
        QC |= to;
        QD |= to;

        // Test for capture:
        BitBoard PAB = (P.A & P.B); // Bitboard containing EnPassants and kings
        BitBoard WhiteOccupied = (P.A | P.B | P.C) & ~P.D;
        moves.MovesArray[moves.NumMovesUsed].Capture = (to & WhiteOccupied & ~PAB) != 0;

        if (!IsBlackInCheck(QA, QB, QC, QD)) // Does proposed move put our (black) king in Check ?
        {
          // make an additional 3 copies for underpromotions
          moves.MovesArray[moves.NumMovesUsed + 1] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 2] = thisMove;
          moves.MovesArray[moves.NumMovesUsed + 3] = thisMove; 

          // set Promotion flags accordingly:
          moves.MovesArray[moves.NumMovesUsed].PromoteQueen = true;

#if NOT
          // TODO: Currently we do not set the Check flag for promotion moves,
          //       since this is complex
#if _FLAG_CHECKS_IN_MOVE_GENERATION
          if (IsWhiteInCheck(QA, QB, QC, QD))  // Does the move put enemy (white) king in Check ?
            moves.MovesArray[moves.NumMovesUsed].Check = true;
#endif
#endif
          
          moves.MovesArray[moves.NumMovesUsed + 1].PromoteRook = true;
          moves.MovesArray[moves.NumMovesUsed + 2].PromoteBishop = true;
          moves.MovesArray[moves.NumMovesUsed + 3].PromoteKnight = true;
          moves.MovesArray[moves.NumMovesUsed + 4].Flags = 0;
          moves.NumMovesUsed += 4;
        }
        else
          moves.MovesArray[moves.NumMovesUsed].IllegalMove = true;
      }
    }

    static BitBoard GenWhiteAttacks(MGPosition Z)
    {
      BitBoard Occupied = Z.A | Z.B | Z.C;
      BitBoard Empty = (Z.A & Z.B & ~Z.C) | // All EP squares, regardless of colour
        ~Occupied;              // All Unoccupied squares

      BitBoard PotentialCapturesForWhite = Occupied & Z.D; // Black Pieces (including Kings)

      BitBoard A = Z.A & ~Z.D;        // White A-Plane
      BitBoard B = Z.B & ~Z.D;        // White B-Plane 
      BitBoard C = Z.C & ~Z.D;        // White C-Plane

      BitBoard S = C & ~A;        // Straight-moving Pieces
      BitBoard D = B & ~A;        // Diagonal-moving Pieces
      BitBoard K = A & B & C;       // King
      BitBoard P = A & ~B & ~C;     // Pawns
      BitBoard N = A & ~B & C;      // Knights

      BitBoard StraightAttacks = MGMoveGenFillFunctions.MoveUpSingleOccluded(MGMoveGenFillFunctions.FillUpOccluded(S, Empty), Empty | PotentialCapturesForWhite);
      StraightAttacks |= MGMoveGenFillFunctions.MoveRightSingleOccluded(MGMoveGenFillFunctions.FillRightOccluded(S, Empty), Empty | PotentialCapturesForWhite);
      StraightAttacks |= MGMoveGenFillFunctions.MoveDownSingleOccluded(MGMoveGenFillFunctions.FillDownOccluded(S, Empty), Empty | PotentialCapturesForWhite);
      StraightAttacks |= MGMoveGenFillFunctions.MoveLeftSingleOccluded(MGMoveGenFillFunctions.FillLeftOccluded(S, Empty), Empty | PotentialCapturesForWhite);

      BitBoard DiagonalAttacks = MGMoveGenFillFunctions.MoveUpRightSingleOccluded(MGMoveGenFillFunctions.FillUpRightOccluded(D, Empty), Empty | PotentialCapturesForWhite);
      DiagonalAttacks |= MGMoveGenFillFunctions.MoveDownRightSingleOccluded(MGMoveGenFillFunctions.FillDownRightOccluded(D, Empty), Empty | PotentialCapturesForWhite);
      DiagonalAttacks |= MGMoveGenFillFunctions.MoveDownLeftSingleOccluded(MGMoveGenFillFunctions.FillDownLeftOccluded(D, Empty), Empty | PotentialCapturesForWhite);
      DiagonalAttacks |= MGMoveGenFillFunctions.MoveUpLeftSingleOccluded(MGMoveGenFillFunctions.FillUpLeftOccluded(D, Empty), Empty | PotentialCapturesForWhite);

      BitBoard KingAttacks = MGMoveGenFillFunctions.FillKingAttacksOccluded(K, Empty | PotentialCapturesForWhite);
      BitBoard KnightAttacks = MGMoveGenFillFunctions.FillKnightAttacksOccluded(N, Empty | PotentialCapturesForWhite);
      BitBoard PawnAttacks = MGMoveGenFillFunctions.MoveUpLeftRightSingle(P) & (Empty | PotentialCapturesForWhite);

      return (StraightAttacks | DiagonalAttacks | KingAttacks | KnightAttacks | PawnAttacks);
    }

    static BitBoard GenBlackAttacks(MGPosition Z)
    {
      BitBoard Occupied = Z.A | Z.B | Z.C;
      BitBoard Empty = (Z.A & Z.B & ~Z.C) | // All EP squares, regardless of colour
                      ~Occupied;            // All Unoccupied squares

      BitBoard PotentialCapturesForBlack = Occupied & ~Z.D; // White Pieces (including Kings)

      BitBoard A = Z.A & Z.D;       // Black A-Plane
      BitBoard B = Z.B & Z.D;       // Black B-Plane 
      BitBoard C = Z.C & Z.D;       // Black C-Plane

      BitBoard S = C & ~A;        // Straight-moving Pieces
      BitBoard D = B & ~A;        // Diagonal-moving Pieces
      BitBoard K = A & B & C;       // King
      BitBoard P = A & ~B & ~C;     // Pawns
      BitBoard N = A & ~B & C;      // Knights

      BitBoard StraightAttacks = MGMoveGenFillFunctions.MoveUpSingleOccluded(MGMoveGenFillFunctions.FillUpOccluded(S, Empty), Empty | PotentialCapturesForBlack);
      StraightAttacks |= MGMoveGenFillFunctions.MoveRightSingleOccluded(MGMoveGenFillFunctions.FillRightOccluded(S, Empty), Empty | PotentialCapturesForBlack);
      StraightAttacks |= MGMoveGenFillFunctions.MoveDownSingleOccluded(MGMoveGenFillFunctions.FillDownOccluded(S, Empty), Empty | PotentialCapturesForBlack);
      StraightAttacks |= MGMoveGenFillFunctions.MoveLeftSingleOccluded(MGMoveGenFillFunctions.FillLeftOccluded(S, Empty), Empty | PotentialCapturesForBlack);

      BitBoard DiagonalAttacks = MGMoveGenFillFunctions.MoveUpRightSingleOccluded(MGMoveGenFillFunctions.FillUpRightOccluded(D, Empty), Empty | PotentialCapturesForBlack);
      DiagonalAttacks |= MGMoveGenFillFunctions.MoveDownRightSingleOccluded(MGMoveGenFillFunctions.FillDownRightOccluded(D, Empty), Empty | PotentialCapturesForBlack);
      DiagonalAttacks |= MGMoveGenFillFunctions.MoveDownLeftSingleOccluded(MGMoveGenFillFunctions.FillDownLeftOccluded(D, Empty), Empty | PotentialCapturesForBlack);
      DiagonalAttacks |= MGMoveGenFillFunctions.MoveUpLeftSingleOccluded(MGMoveGenFillFunctions.FillUpLeftOccluded(D, Empty), Empty | PotentialCapturesForBlack);

      BitBoard KingAttacks = MGMoveGenFillFunctions.FillKingAttacksOccluded(K, Empty | PotentialCapturesForBlack);
      BitBoard KnightAttacks = MGMoveGenFillFunctions.FillKnightAttacksOccluded(N, Empty | PotentialCapturesForBlack);
      BitBoard PawnAttacks = MGMoveGenFillFunctions.MoveDownLeftRightSingle(P) & (Empty | PotentialCapturesForBlack);

      return (StraightAttacks | DiagonalAttacks | KingAttacks | KnightAttacks | PawnAttacks);
    }


    static bool IsWhiteInCheck(BitBoard ZA, BitBoard ZB, BitBoard ZC, BitBoard ZD)
    {
      BitBoard ZAandZB = ZA & ZB;
      BitBoard WhiteKing = ZAandZB & ZC & ~ZD;

      BitBoard V = (ZAandZB & ~ZC) | // All EP squares, regardless of colour
        WhiteKing |           // White King
        ~(ZA | ZB | ZC);       // All Unoccupied squares

      BitBoard A = ZA & ZD;       // Black A-Plane
      BitBoard B = ZB & ZD;       // Black B-Plane 
      BitBoard C = ZC & ZD;       // Black C-Plane

      BitBoard S = C & ~A;        // Straight-moving Pieces
      BitBoard D = B & ~A;        // Diagonal-moving Pieces

      BitBoard K = A & B & C;       // King
      BitBoard AnotB = A & ~B;
      BitBoard P = AnotB & ~C;     // Pawns
      BitBoard N = AnotB & C;      // Knights

      BitBoard X = MGMoveGenFillFunctions.FillStraightAttacksOccluded(S, V);
      X |= MGMoveGenFillFunctions.FillDiagonalAttacksOccluded(D, V);
      X |= MGMoveGenFillFunctions.FillKingAttacks(K);
      X |= MGMoveGenFillFunctions.FillKnightAttacks(N);
      X |= MGMoveGenFillFunctions.MoveDownLeftRightSingle(P);
      return (X & WhiteKing) != 0;
    }

    static bool IsBlackInCheck(BitBoard ZA, BitBoard ZB, BitBoard ZC, BitBoard ZD)
    {
      BitBoard ZAandZB = ZA & ZB;

      BitBoard BlackKing = ZAandZB & ZC & ZD;
      BitBoard V = (ZAandZB & ~ZC) | // All EP squares, regardless of colour
        BlackKing |                     // Black King
        ~(ZA | ZB | ZC);             // All Unoccupied squares

      BitBoard A = ZA & ~ZD;      // White A-Plane
      BitBoard B = ZB & ~ZD;      // White B-Plane 
      BitBoard C = ZC & ~ZD;      // White C-Plane

      BitBoard S = C & ~A;        // Straight-moving Pieces
      BitBoard D = B & ~A;        // Diagonal-moving Pieces
      BitBoard K = A & B & C;       // King
      BitBoard AnotB = A & ~B;
      BitBoard P = AnotB & ~C;     // Pawns
      BitBoard N = AnotB & C;      // Knights

      BitBoard X = MGMoveGenFillFunctions.FillStraightAttacksOccluded(S, V);
      X |= MGMoveGenFillFunctions.FillDiagonalAttacksOccluded(D, V);
      X |= MGMoveGenFillFunctions.FillKingAttacks(K);
      X |= MGMoveGenFillFunctions.FillKnightAttacks(N);
      X |= MGMoveGenFillFunctions.MoveUpLeftRightSingle(P);

      return (X & BlackKing) != 0;
    }

    public static bool IsInCheck(in MGPosition P, bool bIsBlack)
    {
      return bIsBlack ? IsBlackInCheck(P.A, P.B, P.C, P.D) 
                      : IsWhiteInCheck(P.A, P.B, P.C, P.D);
    }


  }
}
