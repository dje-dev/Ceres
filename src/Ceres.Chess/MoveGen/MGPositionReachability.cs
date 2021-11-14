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
using System.Numerics;
using Ceres.Base.DataTypes;

#endregion


namespace Ceres.Chess.MoveGen
{
  public static class MGPositionReachability
  {
    const bool VERBOSE = false;

    /// <summary>
    /// Returns if the posFuture is a chess position which is probably 
    /// reachable from pCurrent.
    /// 
    /// Note that heuristics are used to determine a best guess of whether
    /// the position is most likely reachable, but are not definitive.
    /// For example:
    ///   - the number of pieces of a given type (e.g. Queen) for a given player
    ///     does not usually increase at a later move (but this is rarely possible with promotion)
    ///   - pawns do not move backward so generally if a pawn is past a given rank 
    ///     no pawn will typically be seen an earlier rank later in the game
    ///     (but this is occasionally possible with pawn captures from adjacent files)
    ///     
    /// </summary>
    /// <param name="posCurrent"></param>
    /// <param name="posFuture"></param>
    /// <returns></returns>
    public static bool IsProbablyReachable(in MGPosition posCurrent, in MGPosition posFuture)
    {
      var cur = CalcReachabilityInfo(in posCurrent);
      var pre = CalcReachabilityInfo(in posFuture);

      bool ok = true;

      // Quick check by looking to see if number of pieces for either side has increased.
      if (cur.numPiecesBlack < pre.numPiecesBlack
       || cur.numPiecesWhite < pre.numPiecesWhite)
      {
        if (VERBOSE) Console.WriteLine("NumPiecesPrecheck " + cur.numPiecesBlack + " " + pre.numPiecesBlack);
        ok = false;
      }

      // Look in detail at each specific piece type for decreases.
      else if (AnyPieceCountLessThan(in posCurrent, in posFuture))
      {
        if (VERBOSE) Console.WriteLine("AnyPieceCountLessThan");
        ok = false;
      }

      // Look for pawns that appear to have moved backwards.
      else if (WhitePawnsFail(cur.whitePawns, pre.whitePawns))
      {
        if (VERBOSE) Console.WriteLine("white pawns");
        ok = false;
      }

      else if (BlackPawnsFail(cur.blackPawns, pre.blackPawns))
      {
        if (VERBOSE) Console.WriteLine("black pawns");
        ok = false;
      }

      // Look for gaining castling rights.
      else if ((posCurrent.WhiteDidCastle && !posFuture.WhiteDidCastle)
            || (posCurrent.WhiteDidCastleLong && !posFuture.WhiteDidCastleLong)
            || (posCurrent.BlackDidCastle && !posFuture.BlackDidCastle)
            || (posCurrent.BlackDidCastleLong && !posFuture.BlackDidCastleLong))
      {
        if (VERBOSE) Console.WriteLine("Castle");
        ok = false;
      }

      //  Console.WriteLine(ok ? "ok" : "fail");
      return ok;
    }

    #region Helpers

    static int FirstWhiteRankOccupiedOnFile(BitVector64 bv, int file)
    {
      for (int rank = 1; rank < 7; rank++)
      {
        if (bv.BitIsSet(ToSquare(rank, file)))
        {
          return rank;
        }
      }
      return int.MaxValue;
    }

    static int FirstBlackRankOccupiedOnFile(BitVector64 bv, int file)
    {
      for (int rank = 6; rank > 0; rank--)
      {
        if (bv.BitIsSet(ToSquare(rank, file)))
        {
          return 7 - rank;
        }
      }
      return int.MaxValue;
    }

    /// <summary>
    /// Returns if any white pawns on any file appear subsequently less advanced.
    /// </summary>
    /// <param name="posCurrent"></param>
    /// <param name="posFuture"></param>
    /// <returns></returns>
    static bool WhitePawnsFail(BitVector64 posCurrent, BitVector64 posFuture)
    {
      for (int file = 0; file <= 7; file++)
      {
        int rankCur = FirstWhiteRankOccupiedOnFile(posCurrent, file);
        int rankOther = FirstWhiteRankOccupiedOnFile(posFuture, file);
        if (rankOther == 1 && rankCur != 1)
        {
          // Pawn moved off 2nd rank can never get back
          return true;
        }

        if (rankOther < rankCur)
        {
          int adjacentLeft = file == 0 ? int.MaxValue : FirstWhiteRankOccupiedOnFile(posCurrent, file - 1);
          int adjacentRight = file >= 7 ? int.MaxValue : FirstWhiteRankOccupiedOnFile(posCurrent, file + 1);
          bool possiblyFromAdjacent = adjacentLeft < rankCur || adjacentRight < rankCur;

          if (!possiblyFromAdjacent)
          {
            return true;
          }
        }
      }
      return false;
    }

    /// <summary>
    /// Returns if any black pawns on any file appear subsequently less advanced.
    /// </summary>
    /// <param name="posCurrent"></param>
    /// <param name="posFuture"></param>
    /// <returns></returns>
    static bool BlackPawnsFail(BitVector64 posCurrent, BitVector64 posFuture)
    {
      for (int file = 0; file < 7; file++)
      {
        int rankCur = FirstBlackRankOccupiedOnFile(posCurrent, file);
        int rankOther = FirstBlackRankOccupiedOnFile(posFuture, file);
        if (rankOther == 1 && rankCur != 1)
        {
          // Pawn moved off 2nd rank can never get back
          return true;
        }

        if (rankOther < rankCur)
        {
          int adjacentLeft = file == 0 ? int.MaxValue : FirstBlackRankOccupiedOnFile(posCurrent, file - 1);
          int adjacentRight = file >= 7 ? int.MaxValue : FirstBlackRankOccupiedOnFile(posCurrent, file + 1);
          bool possiblyFromAdjacent = adjacentLeft < rankCur || adjacentRight < rankCur;
          if (!possiblyFromAdjacent)
          {
            return true;
          }
        }
      }
      return false;
    }


    static int ToSquare(int rank, int file) => rank * 8 + file;


    /// <summary>
    /// Computes various reachability related data.
    /// </summary>
    /// <param name="P"></param>
    /// <returns></returns>
    static (int numPiecesWhite, int numPiecesBlack, BitVector64 whitePawns, BitVector64 blackPawns) 
      CalcReachabilityInfo(in MGPosition P)
    {
      ulong occupied = P.A | P.B | P.C;                    // all squares occupied by something
      ulong pABCTemp = (P.A & P.B & ~P.C);
      ulong blackOccupied = P.D & ~pABCTemp;               // all squares occupied by B, excluding EP Squares
      ulong whiteOccupied = (occupied & ~P.D) & ~pABCTemp; // all squares occupied by W, excluding EP Squares

      ulong whitePawns = ~P.D & ~P.C & ~P.B & P.A;
      ulong blackPawns = P.D & ~P.C & ~P.B & P.A;

      return (BitOperations.PopCount(whiteOccupied), BitOperations.PopCount(blackOccupied),
              new BitVector64(whitePawns), new BitVector64(blackPawns));
    }


    /// <summary>
    /// Returns if for any piece type the count has decreased.
    /// </summary>
    /// <param name="posCur"></param>
    /// <param name="posOther"></param>
    /// <returns></returns>
    static bool AnyPieceCountLessThan(in MGPosition posCur, in MGPosition posOther)
    {
      // White pawn
      if (BitOperations.PopCount(~posOther.D & ~posOther.C & ~posOther.B & posOther.A)
        > BitOperations.PopCount(~posCur.D & ~posCur.C & ~posCur.B & posCur.A))
      {
        return true;
      }

      // White bishop
      if (BitOperations.PopCount(~posOther.D & ~posOther.C & posOther.B & ~posOther.A)
        > BitOperations.PopCount(~posCur.D & ~posCur.C & posCur.B & ~posCur.A))
      {
        return true;
      }

      // White rook
      if (BitOperations.PopCount(~posOther.D & posOther.C & ~posOther.B & ~posOther.A)
        > BitOperations.PopCount(~posCur.D & posCur.C & ~posCur.B & ~posCur.A))
      {
        return true;
      }

      // White knight
      if (BitOperations.PopCount(~posOther.D & posOther.C & ~posOther.B & posOther.A)
        > BitOperations.PopCount(~posCur.D & posCur.C & ~posCur.B & posCur.A))
      {
        return true;
      }

      // White queen
      if (BitOperations.PopCount(~posOther.D & posOther.C & posOther.B & ~posOther.A)
        > BitOperations.PopCount(~posCur.D & posCur.C & posCur.B & ~posCur.A))
      {
        return true;
      }


      // Black pawn
      if (BitOperations.PopCount(posOther.D & ~posOther.C & ~posOther.B & posOther.A)
        > BitOperations.PopCount(posCur.D & ~posCur.C & ~posCur.B & posCur.A))
      {
        return true;
      }

      // Black bishop
      if (BitOperations.PopCount(posOther.D & ~posOther.C & posOther.B & ~posOther.A)
        > BitOperations.PopCount(posCur.D & ~posCur.C & posCur.B & ~posCur.A))
      {
        return true;
      }

      // Black rook
      if (BitOperations.PopCount(posOther.D & posOther.C & ~posOther.B & ~posOther.A)
        > BitOperations.PopCount(posCur.D & posCur.C & ~posCur.B & ~posCur.A))
      {
        return true;
      }

      // Black knight
      if (BitOperations.PopCount(posOther.D & posOther.C & ~posOther.B & posOther.A)
        > BitOperations.PopCount(posCur.D & posCur.C & ~posCur.B & posCur.A))
      {
        return true;
      }

      // Black queen
      if (BitOperations.PopCount(posOther.D & posOther.C & posOther.B & ~posOther.A)
        > BitOperations.PopCount(posCur.D & posCur.C & posCur.B & ~posCur.A))
      {
        return true;
      }

      return false;
    }

    #endregion
  }

}
