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

#endregion

#region Using directives

using Ceres.Chess.Textual.PgnFileTools;
using System;
using System.Diagnostics;
using System.Threading.Tasks.Sources;
using static Ceres.Chess.MoveGen.Converters.ConverterMGMoveEncodedMove;
using BitBoard = System.UInt64;

#endregion

namespace Ceres.Chess.MoveGen
{
  public partial struct MGPosition
  {
    /// <summary>
    /// Returns a new MGPosition which results from 
    /// making a specified move from a specified starting position.
    /// </summary>
    /// <param name="mgPos"></param>
    /// <param name="move"></param>
    /// <returns></returns>
    public static MGPosition MGPosAfterMove(MGPosition mgPos, MGMove move)
    {
      mgPos.MakeMove(move);
      return mgPos;
    }


    /// <summary>
    /// Modifies this position to reflect a specified move having been made.
    /// </summary>
    /// <param name="M"></param>
    public void MakeMove(MGMove M)
    {
      DoMakeMove(M);
      SwitchSides();
    }


    /// <summary>
    /// Worker method to make the move (but not switch sides).
    /// </summary>
    /// <param name="M"></param>
    void DoMakeMove(MGMove M)
    {
      Debug.Assert(M.Piece != MGPositionConstants.MCChessPositionPieceEnum.None);

      BitBoard O, To;

      To = 1UL << M.ToSquareIndex;
      O = ~((1UL << M.FromSquareIndex) | To);
      BitBoard nFromSquare = M.FromSquareIndex;
      BitBoard nToSquare = M.ToSquareIndex;

      // Increment ply count
      MoveNumber++;

      //reset opponents castling rights if there is a capture of one of his rooks
      if (M.Capture)
      {
        if (M.BlackToMove)
        {
          if (M.ToSquareIndex == rookInfo.WhiteKRInitPlacement)
          {
            WhiteCanCastle = false;
            WhiteForfeitedCastle = true;
          }
          else if (M.ToSquareIndex == rookInfo.WhiteQRInitPlacement)
          {
            WhiteCanCastleLong = false;
            WhiteForfeitedCastleLong = true;
          }
        }
        else
        {
          if (M.ToSquareIndex == rookInfo.BlackKRInitPlacement + 56)
          {
            BlackCanCastle = false;
            BlackForfeitedCastle = true;
          }
          else if (M.ToSquareIndex == rookInfo.BlackQRInitPlacement + 56)
          {
            BlackCanCastleLong = false;
            BlackForfeitedCastleLong = true;
          }
        }
      }

      // Update Rule 50 count (upon any pawn move or capture).
      // (Update for castling is performed below as well).
      if (M.Capture || (byte)M.Piece == MGPositionConstants.WPAWN
                    || (byte)M.Piece == MGPositionConstants.BPAWN)
      {
        Rule50Count = 0;
      }
      else
      {
        Rule50Count++;
      }

      // clear any enpassant squares
      BitBoard EnPassant = A & B & (~C);
      if (EnPassant != 0)
      {
        A &= ~EnPassant;
        B &= ~EnPassant;
        C &= ~EnPassant;
        D &= ~EnPassant;

#if MG_USE_HASH
        ulong nEPSquare;
        nEPSquare = MGMoveGenFillFunctions.GetSquareIndex(EnPassant);
        if (BlackToMove)
          HK ^= MGZobristKeySet.zkPieceOnSquare[MGChessPositionConstants.WENPASSANT][nEPSquare]; // Remove EP from nEPSquare
        else
          HK ^= MGZobristKeySet.zkPieceOnSquare[MGChessPositionConstants.BENPASSANT][nEPSquare]; // Remove EP from nEPSquare
#endif
      }


      // APPLY CASTLING MOVES:
      // we use magic XOR-tricks to do the job ! :

      // The following is for O-O:
      //
      //    K..R          .RK.
      // A: 1000 ^ 1010 = 0010
      // B: 1000 ^ 1010 = 0010
      // C: 1001 ^ 1111 = 0110
      // For Black:
      // D: 1001 ^ 1111 = 0110
      // For White:
      // D &= 0xfffffffffffffff0 (ie clear colour of affected squares from K to KR)

      // The following is for O-O-O:
      //
      //    R... K...                      ..KR ....
      // A: 0000 1000 ^ 0010 1000 (0x28) = 0010 0000
      // B: 0000 1000 ^ 0010 1000 (0x28) = 0010 0000
      // C: 1000 1000 ^ 1011 1000 (0xB8) = 0011 0000
      // For Black:
      // D: 1000 1000 ^ 1011 1000 (0xB8) = 0011 0000
      // For White:
      // D &= 0xffffffffffffff07 (ie clear colour of affected squares from QR to K) 

      if ((byte)M.Piece == MGPositionConstants.BKING)
      {
        if (M.CastleShort)
        {
          BitBoard kingToSq = 144115188075855872; //g8 in decimals
          BitBoard kingIdx = 57; //g8 represented as index from h1..a1
          BitBoard kingPos = nFromSquare == kingIdx ? 0 : (1UL << (int)nFromSquare) | kingToSq;
          BitBoard rookPos = 1UL << (int)nToSquare;
          rookPos = rookPos == 288230376151711744 ? 0 : rookPos | 288230376151711744;
          BitBoard kingAndRooks = kingPos == rookPos ? 0 : kingPos | rookPos;
          if (rookPos == kingPos)
          {
            kingAndRooks = 0;
          }
          else if ((rookPos & kingPos) != 0)
          {
            kingAndRooks = kingPos ^ rookPos;
          }
          A ^= kingPos;
          B ^= kingPos;
          C ^= kingAndRooks;
          D ^= kingAndRooks;

#if MG_USE_HASH
          HK ^= MGZobristKeySet.zkDoBlackCastle;
	    		if (BlackCanCastleLong) HK ^= MGZobristKeySet.zkBlackCanCastleLong; // conditionally flip black castling long
#endif
          Flags |= FlagsEnum.BlackDidCastle;
          Flags &= ~FlagsEnum.BlackCanCastle;
          Flags &= ~FlagsEnum.BlackCanCastleLong;
          return;
        }
        else if (M.CastleLong)
        {
          BitBoard kingToSq = 2305843009213693952; //c8 in decimals
          BitBoard kingIdx = 61; //c8 represented as index from h1..a8
          BitBoard kingPos = nFromSquare == kingIdx ? 0 : (1UL << (int)nFromSquare) | kingToSq;
          BitBoard rookPos = 1UL << (int)nToSquare;
          rookPos = rookPos == 1152921504606846976 ? 0 : rookPos | 1152921504606846976;
          kingPos = nFromSquare == nToSquare ? 0 : kingPos;
          BitBoard kingAndRooks = kingPos == rookPos ? 0 : kingPos | rookPos;

          if (rookPos == kingPos)
          {
            kingAndRooks = 0;
          }
          else if ((rookPos & kingPos) != 0)
          {
            kingAndRooks = kingPos ^ rookPos;
          }

          A ^= kingPos;
          B ^= kingPos;
          C ^= kingAndRooks;
          D ^= kingAndRooks;

#if MG_USE_HASH
          HK ^= MGZobristKeySet.zkDoBlackCastleLong;
  			  if (BlackCanCastle) HK ^= MGZobristKeySet.zkBlackCanCastle; // conditionally flip black castling
#endif
          BlackDidCastleLong = true;
          BlackCanCastle = false;
          BlackCanCastleLong = false;
          return;
        }
        else
        {
          // ordinary king move
          if (BlackCanCastle || BlackCanCastleLong)
          {
            // Black could have castled, but chose to move the King in a non-castling move
#if MG_USE_HASH
   				 if(BlackCanCastle) HK ^= MGZobristKeySet.zkBlackCanCastle;	// conditionally flip black castling
   				 if(BlackCanCastleLong) HK ^= MGZobristKeySet.zkBlackCanCastleLong;  // conditionally flip black castling long
#endif
            BlackForfeitedCastle = true;
            BlackForfeitedCastleLong = true;
            BlackCanCastle = false;
            BlackCanCastleLong = false;
          }
        }
      }

      else if ((byte)M.Piece == MGPositionConstants.WKING)
      {
        if (M.CastleShort)
        {
          BitBoard rookPos = 1UL << (int)nToSquare;
          rookPos = rookPos == 4 ? 0 : rookPos | 4;
          BitBoard kingToSq = 2; //g1 in decimals
          BitBoard kingIdx = 1; //g1 represented as index from h1..a8
          BitBoard kingPos = nFromSquare == kingIdx ? 0 : (1UL << (int)nFromSquare) | kingToSq;
          BitBoard kingAndRooks = kingPos | rookPos;
          if (rookPos == kingPos)
          {
            kingAndRooks = 0;
          }
          else if ((rookPos & kingPos) != 0)
          {
            kingAndRooks = kingPos ^ rookPos;
          }
          A ^= kingPos;
          B ^= kingPos;
          C ^= kingAndRooks;

#if MG_USE_HASH
			HK^=MGZobristKeySet.zkDoWhiteCastle;
			if (WhiteCanCastleLong) HK ^= MGZobristKeySet.zkWhiteCanCastleLong; // conditionally flip white castling long
#endif
          WhiteDidCastle = true;
          WhiteCanCastle = false;
          WhiteCanCastleLong = false;
          return;
        }

        if (M.CastleLong)
        {
          BitBoard rookPos = 1UL << (int)nToSquare;
          rookPos = rookPos == 16 ? 0 : rookPos | 16;
          BitBoard kingToSq = 32; //c1 in decimals
          BitBoard kingIdx = 5; //c1 represented as index from h1..a8
          BitBoard kingPos = nFromSquare == kingIdx ? 0 : (1UL << (int)nFromSquare) | kingToSq;
          BitBoard kingAndRooks = kingPos == rookPos ? 0 : kingPos | rookPos;
          if (rookPos == kingPos)
          {
            kingAndRooks = 0;
          }
          else if ((rookPos & kingPos) != 0)
          {
            kingAndRooks = kingPos ^ rookPos;
          }

          A ^= kingPos;
          B ^= kingPos;
          C ^= kingAndRooks;

#if MG_USE_HASH
			HK^=MGZobristKeySet.zkDoWhiteCastleLong;
			if (WhiteCanCastle) HK ^= MGZobristKeySet.zkWhiteCanCastle; // conditionally flip white castling
#endif
          WhiteDidCastleLong = true;
          WhiteCanCastle = false;
          WhiteCanCastleLong = false;
          return;

        }

        else
        {
          // ordinary king move
          if (WhiteCanCastle)
          {
#if MG_USE_HASH
				HK ^= MGZobristKeySet.zkWhiteCanCastle;	// flip white castling
#endif
            WhiteForfeitedCastle = true;
            WhiteCanCastle = false;

          }
          if (WhiteCanCastleLong)
          {
#if MG_USE_HASH
				HK ^= MGZobristKeySet.zkWhiteCanCastleLong;	// flip white castling long
#endif
            WhiteForfeitedCastleLong = true;
            WhiteCanCastleLong = false;
          }
        }
      }

      // LOOK FOR FORFEITED CASTLING RIGHTS DUE to ROOK MOVES:
      else if ((byte)M.Piece == MGPositionConstants.BROOK)
      {
        if (M.FromSquareIndex == (rookInfo.BlackKRInitPlacement + 56))
        {
          // Black moved K-side Rook and forfeits right to castle K-side
          if (BlackCanCastle)
          {
            BlackForfeitedCastle = true;
#if MG_USE_HASH
				    HK ^= MGZobristKeySet.zkBlackCanCastle;	// flip black castling		
#endif
            BlackCanCastle = false;
          }
        }
        //else if ((1LL<<nFromSquare) & BLACKQRPOS)
        else if (M.FromSquareIndex == (rookInfo.BlackQRInitPlacement + 56))
        {
          // Black moved the QS Rook and forfeits right to castle Q-side
          if (BlackCanCastleLong)
          {
            BlackForfeitedCastleLong = true;
#if MG_USE_HASH
				HK ^= MGZobristKeySet.zkBlackCanCastleLong;	// flip black castling long
#endif
            BlackCanCastleLong = false;
          }
        }
      }

      else if ((byte)M.Piece == MGPositionConstants.WROOK)
      {
        if (M.FromSquareIndex == rookInfo.WhiteKRInitPlacement)
        {
          // White moved K-side Rook and forfeits right to castle K-side
          if (WhiteCanCastle)
          {
            WhiteForfeitedCastle = true;
#if MG_USE_HASH
				HK ^= MGZobristKeySet.zkWhiteCanCastle;	// flip white castling BROKEN !!!
#endif
            WhiteCanCastle = false;
          }
        }
        //	else if((1LL<<nFromSquare) & WHITEQRPOS)
        else if (M.FromSquareIndex == rookInfo.WhiteQRInitPlacement)
        {
          // White moved the QSide Rook and forfeits right to castle Q-side
          if (WhiteCanCastleLong)
          {
            WhiteForfeitedCastleLong = true;
#if MG_USE_HASH
				HK ^= MGZobristKeySet.zkWhiteCanCastleLong;	// flip white castling long
#endif
            WhiteCanCastleLong = false;
          }
        }
      }

      // Ordinary Captures	////
      if (M.Capture)
      {
        ulong capturedpiece;

#if NOWAY
        //defined (_WIN64) && defined (_USE_BITTEST_INSTRUCTION)
		const long long d = D;
		const long long c = C;
		const long long b = B;
		const long long a = A;

		// using BitTest Intrinsic:
		capturedpiece = _bittest64(&d, nToSquare) << 3
			| _bittest64(&c, nToSquare) << 2
			| _bittest64(&b, nToSquare) << 1
			| _bittest64(&a, nToSquare);		
#else
        // find out which piece has been captured:

        // Branchless version:
        BitBoard bbCap = ((D & To) >> (int)nToSquare) << 3
            | ((C & To) >> (int)nToSquare) << 2
            | ((B & To) >> (int)nToSquare) << 1
            | ((A & To) >> (int)nToSquare);
        capturedpiece = (ulong)bbCap;
#endif

#if MG_USE_HASH
		// Update Hash
		  HK ^= MGZobristKeySet.zkPieceOnSquare[capturedpiece][nToSquare]; // Remove captured Piece
#endif
      }

      // Render "ordinary" moves:
      A &= O;
      B &= O;
      C &= O;
      D &= O;
      // Populate new square (Branchless method):
      A |= (BitBoard)(((ulong)M.Piece & 1) << M.ToSquareIndex);
      B |= (BitBoard)((((ulong)M.Piece & 2) >> 1) << M.ToSquareIndex);
      C |= (BitBoard)((((ulong)M.Piece & 4) >> 2) << M.ToSquareIndex);
      D |= (BitBoard)((((ulong)M.Piece & 8) >> 3) << M.ToSquareIndex);

#if MG_USE_HASH
	// Update Hash
	HK ^= MGZobristKeySet.zkPieceOnSquare[(int)M.Piece][nFromSquare]; // Remove piece at From square
	HK ^= MGZobristKeySet.zkPieceOnSquare[(int)M.Piece][nToSquare]; // Place piece at To Square
#endif

      // Promotions - Change the piece:

      if (M.PromoteBishop)
      {
        A &= ~To;
        B |= To;
#if MG_USE_HASH
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BPAWN : MGChessPositionConstants.WPAWN][nToSquare]; // Remove pawn at To square
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BBISHOP : MGChessPositionConstants.WBISHOP][nToSquare]; // place Bishop at To square
#endif
        M.Piece = M.BlackToMove ? MGPositionConstants.MCChessPositionPieceEnum.BlackBishop : MGPositionConstants.MCChessPositionPieceEnum.WhiteBishop;
        //
        return;

      }

      else if (M.PromoteKnight)
      {
        C |= To;
#if MG_USE_HASH
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BPAWN : MGChessPositionConstants.WPAWN][nToSquare]; // Remove pawn at To square
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BKNIGHT : MGChessPositionConstants.WKNIGHT][nToSquare];// place Knight at To square
#endif
        M.Piece = M.BlackToMove ? MGPositionConstants.MCChessPositionPieceEnum.BlackKnight : MGPositionConstants.MCChessPositionPieceEnum.WhiteKnight;
        //
        return;
      }

      else if (M.PromoteRook)
      {
        A &= ~To;
        C |= To;
#if MG_USE_HASH
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BPAWN : MGChessPositionConstants.WPAWN][nToSquare]; // Remove pawn at To square
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BROOK : MGChessPositionConstants.WROOK][nToSquare];	// place Rook at To square
#endif
        M.Piece = M.BlackToMove ? MGPositionConstants.MCChessPositionPieceEnum.BlackRook : MGPositionConstants.MCChessPositionPieceEnum.WhiteRook;
        //
        return;
      }

      else if (M.PromoteQueen)
      {
        A &= ~To;
        B |= To;
        C |= To;
#if MG_USE_HASH
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BPAWN : MGChessPositionConstants.WPAWN][nToSquare]; // Remove pawn at To square
		HK ^= MGZobristKeySet.zkPieceOnSquare[M.BlackToMove ? MGChessPositionConstants.BQUEEN : MGChessPositionConstants.WQUEEN][nToSquare];	// place Queen at To square
#endif
        M.Piece = M.BlackToMove ? MGPositionConstants.MCChessPositionPieceEnum.BlackQueen : MGPositionConstants.MCChessPositionPieceEnum.WhiteQueen;
        //
        return;
      }

      // For Double-Pawn Moves, set EP square:	////
      else if (M.DoublePawnMove)
      {
        // Set EnPassant Square
        if (M.BlackToMove)
        {
          To <<= 8;
          A |= To;
          B |= To;
          C &= ~To;
          D |= To;

#if MG_USE_HASH
			HK ^= MGZobristKeySet.zkPieceOnSquare[MGChessPositionConstants.BENPASSANT][nToSquare + 8];	// Place Black EP at (To+8)
#endif
        }
        else
        {
          To >>= 8;
          A |= To;
          B |= To;
          C &= ~To;
          D &= ~To;

#if MG_USE_HASH
HK ^= MGZobristKeySet.zkPieceOnSquare[MGChessPositionConstants.WENPASSANT][nToSquare - 8];	// Place White EP at (To-8)
#endif
        }
        //
        return;
      }

      // En-Passant Captures	////
      else if (M.EnPassantCapture)
      {
        // remove the actual pawn (it is different to the capture square)
        if (M.BlackToMove)
        {
          To <<= 8;
          A &= ~To; // clear the pawn's square
          B &= ~To;
          C &= ~To;
          D &= ~To;
          //	material -= 100; // perft doesn't care
#if MG_USE_HASH
HK ^= MGZobristKeySet.zkPieceOnSquare[MGChessPositionConstants.WPAWN][nToSquare + 8]; // Remove WHITE Pawn at (To+8)
#endif
        }
        else
        {
          To >>= 8;
          A &= ~To;
          B &= ~To;
          C &= ~To;
          D &= ~To;
          //	material += 100; // perft doesn't care
#if MG_USE_HASH
HK ^= MGZobristKeySet.zkPieceOnSquare[MGChessPositionConstants.BPAWN][nToSquare - 8]; // Remove BLACK Pawn at (To-8)
#endif
        }
      }

    }

    /// <summary>
    /// Switches side to move.
    /// </summary>
    private void SwitchSides()
    {
      BlackToMove = !BlackToMove;
#if MG_USE_HASH
  HK ^= MGZobristKeySet.zkBlackToMove;
#endif
      return;
    }

    public void Clear()
    {
      A = B = C = D = 0;
      Flags = 0;
#if MG_USE_HASH
  HK = 0;
#endif
    }

  }

}
