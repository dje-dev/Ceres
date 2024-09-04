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

using Ceres.Base.DataType;

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Binary structure representing one square on board from a training position.
  /// All positions are stored from the perspective of side to move.
  /// 
  /// Bytes are used to represent the values so they can be compactly represented.
  /// However fixed divisor (such as 10 or 100) is typically applied after the bytes are sent 
  /// to the computing device (e.g. GPU) but before the neural network sees the values.
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public unsafe struct TPGSquareRecord
  {
    /// <summary>
    /// All bytes encoded within this record are ultimately
    /// converted to floating point values and divided by this divisor
    /// when used in the neural network.
    /// </summary>
    public const float SQUARE_BYTES_DIVISOR = ByteScaled.SCALING_FACTOR;

    public static bool IsOurPiece(ReadOnlySpan<ByteScaled> pieceTypeOneHot)
    {
      byte combinedValue = (byte)(pieceTypeOneHot[1].RawValue | pieceTypeOneHot[2].RawValue
                                  | pieceTypeOneHot[3].RawValue | pieceTypeOneHot[4].RawValue
                                  | pieceTypeOneHot[5].RawValue | pieceTypeOneHot[6].RawValue);
      return combinedValue > 0;
    }


    public static (PieceType pieceType, bool isOurPiece) GetPieceInfo(ReadOnlySpan<ByteScaled> pieceTypeOneHot)
    {
      if (pieceTypeOneHot[0].Value > 0) return (default, true);
      if (pieceTypeOneHot[1].Value > 0) return (PieceType.Pawn, true);
      if (pieceTypeOneHot[2].Value > 0) return (PieceType.Knight, true);
      if (pieceTypeOneHot[3].Value > 0) return (PieceType.Bishop, true);
      if (pieceTypeOneHot[4].Value > 0) return (PieceType.Rook, true);
      if (pieceTypeOneHot[5].Value > 0) return (PieceType.Queen, true);
      if (pieceTypeOneHot[6].Value > 0) return (PieceType.King, true);

      if (pieceTypeOneHot[7].Value > 0) return (PieceType.Pawn, false);
      if (pieceTypeOneHot[8].Value > 0) return (PieceType.Knight, false);
      if (pieceTypeOneHot[9].Value > 0) return (PieceType.Bishop, false);
      if (pieceTypeOneHot[10].Value > 0) return (PieceType.Rook, false);
      if (pieceTypeOneHot[11].Value > 0) return (PieceType.Queen, false);
      if (pieceTypeOneHot[12].Value > 0) return (PieceType.King, false);

      throw new Exception("Internal error: TPGRecord square was not initialized");
    }


    public Square GetSquare() => TPGRecordUtils.ToSquare(RankEncoding, FileEncoding);

    public readonly bool IsOccupied => PieceTypeHistory(0)[0].Value == 0;

    public override string ToString()
    {
      string str = $"[-{PlySinceLastMove} ply] ";
      str += (IsOurPiece(PieceTypeHistory(0)) ? "Our " : "Opponent ") + GetPieceInfo(PieceTypeHistory(0)).pieceType + " on " + GetSquare();

      str += " EP= " + this.IsEnPassant;
      str += " Move50= " + Move50Count;

      str += " Hist= ";
      for (int i = 0; i < NUM_HISTORY_POS; i++)
      {
        (PieceType pieceType, bool isOurPiece) histInfo = GetPieceInfo(PieceTypeHistory(i));
        str += histInfo.isOurPiece ? histInfo.pieceType : histInfo.pieceType.ToString().ToLower() + " ";
      }

      str += "Reps= ";
      for (int i = 0; i < NUM_HISTORY_POS; i++)
      {
        if (HistoryRepetitionCounts[i].Value > 0)
        {
          float repValue = HistoryRepetitionCounts[i].Value;
          str += HistoryRepetitionCounts[i].Value > 0 ? repValue : " ";
        }
      }

      return str;
    }


    #region Raw fields

    /// <summary>
    /// Total number of positions (current and prior) encoded.
    /// </summary>
    public const int NUM_HISTORY_POS = 8;

    /// <summary>
    /// Length of the one-hot piece encodings.
    /// </summary>
    const int NUM_BYTES_PER_HISTORY_PLANE = 13; // Empty square, 6 white pieces, 6 black pieces

    /// <summary>
    /// Total number of bytes used for the piece encoding for all history planes.
    /// </summary>
    const int TOTAL_BYTES_HISTORY_POS = NUM_HISTORY_POS * NUM_BYTES_PER_HISTORY_PLANE;

    /// <summary>
    /// Array indicating which piece (if any) is on the square for each of thie history positions.
    /// Conceptually this is a 2-D array of [NUM_HISTORY_POS, NUM_BYTES_PER_HISTORY_PLANE] 
    /// where the values are one-hot encodings of the enumeration Ceres.Chess.PieceType.
    /// </summary>
    fixed byte pieceTypeHistoryAllHistoryPositions[NUM_HISTORY_POS * NUM_BYTES_PER_HISTORY_PLANE]; // N.B. Actually represented as ByteScaled

    /// <summary>
    /// Binary (1 or 0) indicating if each of the history planes has nonzero repetition counter.
    /// </summary>
    fixed byte historyRepetitionCounts[NUM_HISTORY_POS];

    /// <summary>
    /// Binary (1 or 0) indicating if the side to play retains kingside castling rights.
    /// </summary>
    public ByteScaled CanOO;

    /// <summary>
    /// Binary (1 or 0) indicating if the side to play retains queenside castling rights.
    /// </summary>
    public ByteScaled CanOOO;

    /// <summary>
    /// Binary (1 or 0) indicating if the opponent retains kingside castling rights.
    /// </summary>
    public ByteScaled OpponentCanOO;

    /// <summary>
    /// Binary (1 or 0) indicating if the opponent retains queenside castling rights.
    /// </summary>
    public ByteScaled OpponentCanOOO;

    /// <summary>
    /// Counter of ply for the purposes of 50 move rule in chess
    /// (ply since last castling, pawn move, or capture).
    /// </summary>
    public ByteScaled Move50Count;

    /// <summary>
    /// The number of plies since the last move of the piece on this square.
    /// Not used.
    /// </summary>
    public ByteScaled PlySinceLastMove;

    /// <summary>
    /// Binary (1 or 0) indicating if the side to play has en passant rights for any legal move.
    /// </summary>
    public ByteScaled IsEnPassant;

    /// <summary>
    /// Sum of all future positive blunders (negative optimal moves played by opponent) in the game.
    /// Scaled by 100 (e.g. cumulative 0.17 blunders would be 17).
    /// For inference (where this is unknowable) typically a nonzero but small value is used (e.g. 3).
    /// </summary>
    public ByteScaled QPositiveBlunders;

    /// <summary>
    /// Sum of all future negative blunders (negative optimal moves played by side to play) in the game.
    /// Scaled by 100 (e.g. cumulative 0.17 blunders would be 17).
    /// For inference (where this is unknowable) typically a nonzero but small value is used (e.g. 3).
    /// </summary>
    public ByteScaled QNegativeBlunders;

    /// <summary>
    /// One-hot encoding of the rank of this square starting at back rank = 0.
    /// </summary>
    fixed byte rankEncoding[8];

    /// <summary>
    /// One-hot encoding of the file of this square, starting at A=0.
    /// </summary>
    fixed byte fileEncoding[8];

    #endregion

    public ReadOnlySpan<ByteScaled> PieceTypeHistory(int historyPosIndex)
    {
      ReadOnlySpan<ByteScaled> fullSpan = MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateReadOnlySpan(ref pieceTypeHistoryAllHistoryPositions[0], TOTAL_BYTES_HISTORY_POS));
      return fullSpan.Slice(historyPosIndex * NUM_BYTES_PER_HISTORY_PLANE, NUM_BYTES_PER_HISTORY_PLANE);
    }

    public Span<ByteScaled> PieceTypeHistorySetter(int historyPosIndex)
    {
      Span<ByteScaled> fullSpan = MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateSpan(ref pieceTypeHistoryAllHistoryPositions[0], TOTAL_BYTES_HISTORY_POS));
      return fullSpan.Slice(historyPosIndex * NUM_BYTES_PER_HISTORY_PLANE, NUM_BYTES_PER_HISTORY_PLANE);
    }

    public ReadOnlySpan<ByteScaled> HistoryRepetitionCounts => MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateReadOnlySpan(ref historyRepetitionCounts[0], NUM_HISTORY_POS));
    public Span<ByteScaled> HistoryRepetitionCountsSetter => MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateSpan(ref historyRepetitionCounts[0], NUM_HISTORY_POS));
    public ReadOnlySpan<ByteScaled> RankEncoding => MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateReadOnlySpan(ref rankEncoding[0], 8));
    public Span<ByteScaled> RankEncodingSetter => MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateSpan(ref rankEncoding[0], 8));

    public ReadOnlySpan<ByteScaled> FileEncoding => MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateReadOnlySpan(ref fileEncoding[0], 8));
    public Span<ByteScaled> FileEncodingSetter => MemoryMarshal.Cast<byte, ByteScaled>(MemoryMarshal.CreateSpan(ref fileEncoding[0], 8));


    public void Validate()
    {
      // TODO: fix      Debug.Assert(IsOurPiece.Value == 0 || IsEnPassant.Value == 0); // only makes sense for opponent piece to be en passant
    }

    static bool haveWarned = false;

    static internal unsafe void WritePosPieces(in Position pos,
                                               in Position historyPos1,
                                               in Position historyPos2,
                                               in Position historyPos3,
                                               in Position historyPos4,
                                               in Position historyPos5,
                                               in Position historyPos6,
                                               in Position historyPos7,
                                               Span<TPGSquareRecord> squareRecords,
                                               Span<byte> pliesSinceLastPieceMoveBySquare,
                                               bool emitPlySinceLastMovePerSquare,
                                               float qNegativeBlunders, float qPositiveBlunders)
    {
      bool hasEnPassant = pos.MiscInfo.EnPassantRightsPresent;
      bool sawEnPassant = false;

      // N.B. Assumed that the squareRecords start out cleared (verified below in debug mode).
      bool weAreWhite = pos.MiscInfo.SideToMove == SideType.White;

      byte canOO, canOOO, opponentCanOO, opponentCanOOO;

      if (weAreWhite)
      {
        canOO = pos.MiscInfo.WhiteCanOO ? (byte)1 : (byte)0;
        canOOO = pos.MiscInfo.WhiteCanOOO ? (byte)1 : (byte)0;
        opponentCanOO = pos.MiscInfo.BlackCanOO ? (byte)1 : (byte)0;
        opponentCanOOO = pos.MiscInfo.BlackCanOOO ? (byte)1 : (byte)0;
      }
      else
      {
        canOO = pos.MiscInfo.BlackCanOO ? (byte)1 : (byte)0;
        canOOO = pos.MiscInfo.BlackCanOOO ? (byte)1 : (byte)0;
        opponentCanOO = pos.MiscInfo.WhiteCanOO ? (byte)1 : (byte)0;
        opponentCanOOO = pos.MiscInfo.WhiteCanOOO ? (byte)1 : (byte)0;
      }

      int i = 0;
      bool foundTargetSquareFromPriorMove = false;
      Debug.Assert(TPGRecord.NUM_SQUARES == 64);
      for (int squareNum = 0; squareNum < 64; squareNum++)
      {
        Square squareFromPos = new Square(squareNum, Square.SquareIndexType.BottomToTopLeftToRight);
        Piece piece = pos.PieceOnSquare(squareFromPos);

        TPGSquareRecord pieceRecord = default;

        bool isOurPiece = piece.Side == pos.SideToMove;

        pieceRecord.CanOO.Value = canOO;
        pieceRecord.CanOOO.Value = canOOO;
        pieceRecord.OpponentCanOO.Value = opponentCanOO;
        pieceRecord.OpponentCanOOO.Value = opponentCanOOO;

        pieceRecord.Move50Count.Value = TPGRecordEncoding.Move50CountEncoded(pos.MiscInfo.Move50Count);

        pieceRecord.QNegativeBlunders.Value = Math.Min(ByteScaled.MAX_VALUE, qNegativeBlunders);
        pieceRecord.QPositiveBlunders.Value = Math.Min(ByteScaled.MAX_VALUE, qPositiveBlunders);

        if (emitPlySinceLastMovePerSquare)
        {
          if (pliesSinceLastPieceMoveBySquare != null)
          {
            pieceRecord.PlySinceLastMove.Value = TPGRecordEncoding.PliesSinceLastMoveEncoded(pliesSinceLastPieceMoveBySquare[squareNum]);
          }
          else
          {
            // Fill in a reasonable default.
            pieceRecord.PlySinceLastMove.Value = TPGRecordEncoding.PliesSinceLastMoveEncoded(TPGRecordEncoding.DEFAULT_PLIES_SINCE_LAST_PIECE_MOVED_IF_STARTPOS);
          }
        }

        // Check if en passant enabled.
        if (hasEnPassant
         && piece.Type == PieceType.Pawn
         && squareFromPos.Rank == (weAreWhite ? 4 : 3)
         && squareFromPos.File == (int)pos.MiscInfo.EnPassantFileIndex)
        {
          if (piece.Side == SideType.White == weAreWhite)
          {
            throw new Exception("Internal error, en passant pawn of wrong color found.");
          }

          pieceRecord.IsEnPassant.Value = 1;
          sawEnPassant = true;
          //          Console.WriteLine(pos.SideToMove + " " +  /*piece.Square.Rank + " " + piece.Square.File +*/ " " + (int)pos.MiscInfo.EnPassantFileIndex + " " + pos.MiscInfo.EnPassantFileChar);
        }


        static void WritePieceHistory(int index, SideType ourSide, in Position thisPos, Square squareFromPos,
                                     Span<ByteScaled> targetSetter, Span<ByteScaled> repetitionCountsSetter)
        {
          Piece piece = thisPos.PieceOnSquare(squareFromPos);
          TPGRecordUtils.WritePieceEncoding(piece.Side == ourSide, piece.Type, targetSetter);

          // NOTE: The LC0 training data either turns the repetition on (all ones) or not (all zeros)
          //       regardless of the actual repetition count.  Therefore we use a maximum value of 1 here,
          //       even if these positions came thru another path and had more complete repetition data.
          repetitionCountsSetter[index].Value = Math.Min((byte)1, thisPos.MiscInfo.RepetitionCount);
        }

        // Write piece on this square for this position and all history positions.
        Span<ByteScaled> historyReptitionCountSetter = pieceRecord.HistoryRepetitionCountsSetter;
        WritePieceHistory(0, pos.SideToMove, in pos, squareFromPos, pieceRecord.PieceTypeHistorySetter(0), historyReptitionCountSetter);
        WritePieceHistory(1, pos.SideToMove, in historyPos1, squareFromPos, pieceRecord.PieceTypeHistorySetter(1), historyReptitionCountSetter);
        WritePieceHistory(2, pos.SideToMove, in historyPos2, squareFromPos, pieceRecord.PieceTypeHistorySetter(2), historyReptitionCountSetter);
        WritePieceHistory(3, pos.SideToMove, in historyPos3, squareFromPos, pieceRecord.PieceTypeHistorySetter(3), historyReptitionCountSetter);
        WritePieceHistory(4, pos.SideToMove, in historyPos4, squareFromPos, pieceRecord.PieceTypeHistorySetter(4), historyReptitionCountSetter);
        WritePieceHistory(5, pos.SideToMove, in historyPos5, squareFromPos, pieceRecord.PieceTypeHistorySetter(5), historyReptitionCountSetter);
        WritePieceHistory(6, pos.SideToMove, in historyPos6, squareFromPos, pieceRecord.PieceTypeHistorySetter(6), historyReptitionCountSetter);
        WritePieceHistory(7, pos.SideToMove, in historyPos7, squareFromPos, pieceRecord.PieceTypeHistorySetter(7), historyReptitionCountSetter);

        // Write position of square on board.
        bool needsReversal = pos.SideToMove == SideType.Black;
        Square squareInTPG = needsReversal ? squareFromPos.Reversed : squareFromPos;
        TPGRecordUtils.WriteSquareEncoding(squareInTPG, pieceRecord.RankEncodingSetter, pieceRecord.FileEncodingSetter);

        if (pos.SideToMove == SideType.White)
        {
          squareRecords[i] = pieceRecord;
        }
        else
        {
          // This isn't strictly necessary since the square location is coded internally and
          // transformers are invariant to position of the sequence item.
          // However this may be more intuitive when debugging (dumping squares) so we see side-to-play squares first in array.
          squareRecords[63 - i] = pieceRecord;
        }
        i++;
      }


      if (hasEnPassant && !sawEnPassant)
      {
        throw new Exception("Internal error, en passant right were present but no opponent pawn was found where expected");
      }
    }


    /// <summary>
    /// Dumps a human-readable representation of the underlying bytes to the Console.
    /// </summary>
    public void Dump()
    {
      Console.WriteLine("Dump of raw square bytes");
      Console.WriteLine();
      Console.WriteLine("pieceTypeHistoryAllHistoryPositions:");
      for (int i = 0; i < NUM_HISTORY_POS; i++)
      {
        for (int j = 0; j < NUM_BYTES_PER_HISTORY_PLANE; j++)
        {
          Console.Write(pieceTypeHistoryAllHistoryPositions[i * NUM_BYTES_PER_HISTORY_PLANE + j] + " ");
        }
        Console.WriteLine();
      }

      Console.WriteLine("\nhistoryRepetitionCounts:");
      fixed (byte* ptr = historyRepetitionCounts)
      {
        for (int i = 0; i < NUM_HISTORY_POS; i++)
        {
          Console.Write(ptr[i] + " ");
        }
        Console.WriteLine();
      }

      Console.WriteLine("\nCanOO:\n" + CanOO);
      Console.WriteLine("\nCanOOO:\n" + CanOOO);
      Console.WriteLine("\nOpponentCanOO:\n" + OpponentCanOO);
      Console.WriteLine("\nOpponentCanOOO:\n" + OpponentCanOOO);
      Console.WriteLine("\nMove50Count:\n" + Move50Count);
      Console.WriteLine("\nPlySinceLastMove:\n" + PlySinceLastMove);
      Console.WriteLine("\nIsEnPassant:\n" + IsEnPassant);
      Console.WriteLine("\nQPositiveBlunders:\n" + QPositiveBlunders);
      Console.WriteLine("\nQNegativeBlunders:\n" + QNegativeBlunders);

      Console.WriteLine("\nrankEncoding:");
      fixed (byte* ptr = rankEncoding)
      {
        for (int i = 0; i < 8; i++)
        {
          Console.Write(ptr[i] + " ");
        }
        Console.WriteLine();

      }

      Console.WriteLine("\nfileEncoding:");
      fixed (byte* ptr = fileEncoding)
      {
        for (int i = 0; i < 8; i++)
        {
          Console.Write(ptr[i] + " ");
        }
        Console.WriteLine();
      }
    }
  }

}
