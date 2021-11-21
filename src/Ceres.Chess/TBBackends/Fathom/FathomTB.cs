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
using System.Collections.Generic;
using System.Diagnostics;
using Ceres.Chess.MoveGen;

#endregion

#if NOT

TO DO: possibly deal with positions which would be won/lost except DTZ is too big for position.
See the file syzygy.h (struct SyzygyTb) of the open source Arasan chess engine
for an example of a simple interface used by a chess engine to the underlying Fathom code.

However it seems like (in search.cpp) the probe_wdl is only called for move 50 counter equal to 0,
so maybe this engine is not doing this optimally?

#endif

namespace Ceres.Chess.TBBackends.Fathom
{
  /// <summary>
  /// WDL probe result (from perspective of player to move).
  /// </summary>
  public enum FathomWDLResult
  {
    Loss,
    BlessedLoss,
    Draw,
    CursedWin,
    Win,
    Failure = -1
  };

  public static class FathomTB
  {
    /// <summary>
    /// The tablebase can be probed for any position where #pieces <= MaxPieces.    
    /// </summary>
    public static int MaxPieces => probe != null ? probe.TB_LARGEST : 0;

    static FathomProbe probe = null;


    /// <summary>
    /// Initialize the tablebase.
    /// </summary>
    /// <param name="paths">one or more paths to tablebase files (separated by ';' on Windows or ':' on Linux)</param>
    /// <returns>if initialization was successful</returns>
    public static bool Initialize(string paths)
    {
      // Currently only little-endian supported.
      // (a few small support functions were not
      // transliterated from the C to support big-endian).
      Debug.Assert(BitConverter.IsLittleEndian);

      if (probe != null)
      {
        throw new Exception("FathomTB already initialized, call Release before attempting reinitialization.");
      }

      probe = new FathomProbe();
      probe.tb_init(paths);

      return true;
    }

    /// <summary>
    /// Frees any resources allocated by Initialize.
    /// </summary>
    public static void Release()
    {
      probe?.tb_free();
      probe = null;
    }


    /// <summary>
    /// Probes the Win/Draw/Loss (WDL) table.
    /// </summary>
    /// <param name="fen">FEN of the position</param>
    /// <returns>the result of the lookup, or Error if not found</returns>
    public static FathomWDLResult ProbeWDL(in Position pos)
    {
      //      if (_rule50 != 0)        return TB_RESULT_FAILED;

      if (!PreparePos(in pos, out FathomPos fathomPos))
      {
        return FathomWDLResult.Failure;
      }

      int v;
      int success = 0;
      try
      {
        v = probe.probe_wdl(in fathomPos, out success);
      }
      catch (Exception exc)
      {
        Console.WriteLine($"Exception in Fathom probe_wdl {pos.FEN} of {exc}");
        return FathomWDLResult.Failure;
      }

      if (success == 0)
      {
        return FathomWDLResult.Failure;
      }
      else
      {
        return (FathomWDLResult)(v + 2);
      }
    }


    /*
     * RETURN:
     * - A TB_RESULT value comprising:
     *   1) The WDL value (TB_GET_WDL)
     *   2) The suggested move (TB_GET_FROM, TB_GET_TO, TB_GET_PROMOTES, TB_GET_EP)
     *   3) The DTZ value (TB_GET_DTZ)
     *   The suggested move is guaranteed to preserved the WDL value.
     *
     *   Otherwise:
     *   1) TB_RESULT_STALEMATE is returned if the position is in stalemate.
     *   2) TB_RESULT_CHECKMATE is returned if the position is in checkmate.
     *   3) TB_RESULT_FAILED is returned if the probe failed.
     */

    static bool PreparePos(in Position pos, out FathomPos fathomPos)
    {
      fathomPos = FathomPos.FromPosition(in pos);

#if DEBUG
      FathomPos fp2 = default;
      Debug.Assert(FathomFENParsing.parse_FEN(ref fp2, pos.FEN) && fp2.Equals(fathomPos));
#endif

      if (fathomPos.castling != 0)
      {
        return false;
      }
      return true;
    }



    public static int ProbeDTZOnly(in Position pos, out int success)
    {
      if (!PreparePos(in pos, out FathomPos fathomPos))
      {
        success = 0; // failure 
        return -1;
      };

      int ret = probe.probe_dtz(in fathomPos, out success);
      if (ret == 0)
      {
        return -1;
      }
      else
      {
        return ret;
      }
    }

    /// <summary>
    /// A possible winning move from a DTZ probe (and associated DTZ).
    /// </summary>
    public struct DTZMove
    {
      public readonly MGMove Move;
      public int DistanceToZero;
      public FathomWDLResult WDL;

      public DTZMove(MGMove move, int distanceToZero, FathomWDLResult wdl)
      {
        Move = move;
        DistanceToZero = distanceToZero;
        WDL = wdl;
      }
    }

    /// <summary>
    /// Probes the Distance to Zero table (DTZ),
    /// intended for probing at root.
    /// 
    /// The Fathom documentation describes this as not thread safe,
    /// therefore the implementation is protected with a lock.
    /// 
    /// TODO: verify if this is true or the restriction can be lifted.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="minDTZ">the DTZ of the move having minimum DTZ value</param>
    /// <param name="results">if not null then the set of possible moves is populated</param>
    /// <returns>the suggested move (guaranteed to preserve the WDL value </returns>
    public static FathomProbeMove ProbeDTZ(in Position pos, out int minDTZ, out List<DTZMove> results)
    {
      minDTZ = -1;

      // Currently feature disabled, to enable allocate this array
      uint[] resultsArray = new uint[FathomMoveGen.TB_MAX_MOVES];

      if (!PreparePos(in pos, out FathomPos fathomPos))
      {
        results = null;
        return new FathomProbeMove() { Result = FathomWDLResult.Failure };
      }

      uint res;
      lock (dtzLockObj)
      {
        res = probe.tb_probe_root(fathomPos.white, fathomPos.black, fathomPos.kings,
                                  fathomPos.queens, fathomPos.rooks, fathomPos.bishops, fathomPos.knights, fathomPos.pawns,
                                  fathomPos.rule50, fathomPos.castling, fathomPos.ep, fathomPos.turn, resultsArray);
      }

      if (res == uint.MaxValue)
      {
        results = null;
        return new FathomProbeMove() { Result = FathomWDLResult.Failure };
      }

      minDTZ = int.MaxValue;
      results = new List<DTZMove>();
      for (int i=0; i<resultsArray.Length;i++)
      {
        uint result = resultsArray[i];
        if (result == uint.MaxValue)
        {
          break;
        }

        uint dtz = FathomProbe.TB_GET_DTZ((int)result);
        if (dtz < minDTZ)
        {
          minDTZ = (int)dtz;
        }

        FathomWDLResult wdl = (FathomWDLResult)FathomProbe.TB_GET_WDL((int)result);

        results.Add(new DTZMove(ToMGMove(result, in pos), (int)dtz, wdl));
      }
      
      MGMove move = ToMGMove(res, in pos);
      return new FathomProbeMove()
      {
        Result = (FathomWDLResult)FathomProbe.TB_GET_WDL((int)res),
        Move = move
      };
    }


    /// <summary>
    /// Converts a move from Fathom representation to MGMove.
    /// </summary>
    /// <param name="fathomResult"></param>
    /// <param name="position"></param>
    /// <returns></returns>
    private static MGMove ToMGMove(uint fathomResult, in Position position)
    {
      if (fathomResult == FathomProbe.TB_RESULT_STALEMATE 
       || fathomResult == FathomProbe.TB_RESULT_CHECKMATE)
      {
        return default;
      }

      static Square ToSquare(int s) => Square.FromFileAndRank(FathomMoveGen.file(s), FathomMoveGen.rank(s));

      MGMove.MGChessMoveFlags flags = MGMove.MGChessMoveFlags.None;

      Square fromSquare = ToSquare((int)FathomProbe.TB_GET_FROM((int)fathomResult));
      Square toSquare = ToSquare((int)FathomProbe.TB_GET_TO((int)fathomResult));

      Piece piece = position.PieceOnSquare(fromSquare);
      Piece takenPiece = position.PieceOnSquare(toSquare);

      if (takenPiece.Type != PieceType.None)
      {
        flags |= MGMove.MGChessMoveFlags.Capture;
      }
      if (FathomProbe.TB_GET_EP((int)fathomResult) != 0)
      {
        flags |= MGMove.MGChessMoveFlags.EnPassantCapture;
      }

      // Set promotes flag
      flags |= FLAGS_MAP[FathomProbe.TB_GET_PROMOTES((int)fathomResult)];


      MGPositionConstants.MCChessPositionPieceEnum pieceEnum;
      int[] offsets = new int[] { 0, 1, 5, 2, 4, 6, 7 }; // Map from PieceType to MCChessPositionPieceEnum
      if (piece.Side == SideType.White)
      {
        pieceEnum = (MGPositionConstants.MCChessPositionPieceEnum)offsets[(int)piece.Type];
      }
      else
      {
        pieceEnum = (MGPositionConstants.MCChessPositionPieceEnum)(8 + offsets[(int)piece.Type]);
      }

      MGMove move = new MGMove((byte)fromSquare.SquareIndexStartH1,
                               (byte)toSquare.SquareIndexStartH1,
                              pieceEnum,
                              flags);
      return move;
    }

    static readonly MGMove.MGChessMoveFlags[] FLAGS_MAP = new MGMove.MGChessMoveFlags[]
  {
          MGMove.MGChessMoveFlags.None,
          MGMove.MGChessMoveFlags.PromoteQueen,
          MGMove.MGChessMoveFlags.PromoteRook,
          MGMove.MGChessMoveFlags.PromoteBishop,
          MGMove.MGChessMoveFlags.PromoteKnight
  };


    static object dtzLockObj = new ();

#if NOT

struct TbRootMove
    {
      TbMove move;
      TbMove pv[TB_MAX_PLY];
      uint pvSize;
      int32_t tbScore, tbRank;
    };

    struct TbRootMoves
    {
      uint size;
      struct TbRootMove moves[TB_MAX_MOVES];
};

    /*
     * Use the DTZ tables to rank and score all root moves.
     * INPUT: as for tb_probe_root
     * OUTPUT: TbRootMoves structure is filled in. This contains
     * an array of TbRootMove structures.
     * Each structure instance contains a rank, a score, and a
     * predicted principal variation.
     * RETURN VALUE:
     *   non-zero if ok, 0 means not all probes were successful
     *
     */
    public static int tb_probe_root_dtz(
        ulong _white,
        ulong _black,
        ulong _kings,
        ulong _queens,
        ulong _rooks,
        ulong _bishops,
        ulong _knights,
        ulong _pawns,
        uint _rule50,
        uint _castling,
        uint _ep,
        bool _turn,
        bool hasRepeated,
        bool useRule50,
    struct TbRootMoves *_results);

/*
// Use the WDL tables to rank and score all root moves.
// This is a fallback for the case that some or all DTZ tables are missing.
 * INPUT: as for tb_probe_root
 * OUTPUT: TbRootMoves structure is filled in. This contains
 * an array of TbRootMove structures.
 * Each structure instance contains a rank, a score, and a
 * predicted principal variation.
 * RETURN VALUE:
 *   non-zero if ok, 0 means not all probes were successful
 *
 */
public static int tb_probe_root_wdl(ulong _white,
    ulong _black,
    ulong _kings,
    ulong _queens,
    ulong _rooks,
    ulong _bishops,
    ulong _knights,
    ulong _pawns,
    uint _rule50,
    uint _castling,
    uint _ep,
    bool _turn,
    bool useRule50,
    TbRootMoves results)
    {
      return -1;
    }
#endif

  }

}