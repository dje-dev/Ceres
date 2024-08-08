#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Zstandard.Net;

using Ceres.Base.DataType;
using Ceres.Base.DataTypes;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Static helper methods related to TPG records.
  /// </summary>
  public static class TPGRecordUtils
  {
    /// <summary>
    /// Writes the binary encoding of a specified value into a specified span of bytes, starting a specified offset.
    /// </summary>
    /// <param name="value"></param>
    /// <param name="maxBits"></param>
    /// <param name="bytes"></param>
    /// <param name="offset"></param>
    /// <exception cref="NotImplementedException"></exception>
    static internal void WriteBinaryEncoding(byte value, int maxBits, Span<ByteScaled> bytes, int offset = 0)
    {
      if (maxBits > 8 || maxBits < 1)
      {
        throw new ArgumentOutOfRangeException(nameof(maxBits), "maxBits must be between 1 and 8.");
      }

      if (maxBits == 6)
      {
        // Common case, fully unrolled.
        BitVector64 bv = new BitVector64(value);

        bytes[0 + offset].Value = bv.BitIsSet(0) ? 1 : 0;
        bytes[1 + offset].Value = bv.BitIsSet(1) ? 1 : 0;
        bytes[2 + offset].Value = bv.BitIsSet(2) ? 1 : 0;
        bytes[3 + offset].Value = bv.BitIsSet(3) ? 1 : 0;
        bytes[4 + offset].Value = bv.BitIsSet(4) ? 1 : 0;
        bytes[5 + offset].Value = bv.BitIsSet(5) ? 1 : 0;
      }
      else
      {
        for (int i = 0; i < maxBits; i++)
        {
          bytes[i + offset].Value = (value & 1 << i) != 0 ? (byte)1 : (byte)0;
        }
      }
    }


    /// <summary>
    /// Writes representation of rank/file of a specified square into a specified span of bytes.
    /// </summary>
    /// <param name="sq"></param>
    /// <param name="ranks"></param>
    /// <param name="files"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void WriteSquareEncoding(Square sq, Span<ByteScaled> ranks, Span<ByteScaled> files)
    {
      ranks[sq.Rank].Value = 1;
      files[sq.File].Value = 1;
    }


    /// <summary>
    /// Writes representation of a specified piece into a specified span of bytes.
    /// </summary>
    /// <param name="isOurPiece"></param>
    /// <param name="pieceType"></param>
    /// <param name="pieces"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void WritePieceEncoding(bool isOurPiece, PieceType pieceType, Span<ByteScaled> pieces)
    {
      if (pieceType == PieceType.None)
      {
        pieces[0].Value = 1;
      }
      else if (isOurPiece)
      {
        pieces[(int)pieceType].Value = 1;
      }
      else
      {
        pieces[(int)pieceType + 6].Value = 1;
      }
    }

    static unsafe internal void WritePieceEncoding(MGPositionConstants.MCChessPositionPieceEnum pieceType, Span<ByteScaled> pieces) => WriteBinaryEncoding((byte)pieceType, 3, pieces);

    /// <summary>
    /// Returns the square represented by a specified spans of rank/file one-hot encodings.
    /// </summary>
    /// <param name="ranks"></param>
    /// <param name="files"></param>
    /// <returns></returns>
    public static unsafe Square ToSquare(ReadOnlySpan<ByteScaled> ranks, ReadOnlySpan<ByteScaled> files)
    {
      int rank = -1;
      int file = -1;

      for (int i = 0; i < 8; i++)
      {
        if (ranks[i].Value != 0)
        {
          rank = i;
        }
        if (files[i].Value != 0)
        {
          file = i;
        }
      }

      Debug.Assert(rank >= 0 && file >= 0);

      return new Square((byte)(rank * 8 + file), Square.SquareIndexType.BottomToTopLeftToRight);
    }


    /// <summary>
    /// Returns the en passant rights status between two boards representing last 2 positions.
    /// </summary>
    /// <param name="priorBoard"></param>
    /// <param name="currentBoard"></param>
    /// <returns></returns>
    public static PositionMiscInfo.EnPassantFileIndexEnum EnPassantOpportunityBetweenBoards(in Position priorBoard, in Position currentBoard)
    {
      bool currentWhiteIsToMove = currentBoard.IsWhite;
      SideType currentSide = currentWhiteIsToMove ? SideType.White : SideType.Black;
      SideType otherSide = currentSide.Reversed();
      for (int file = 0; file < 8; file++)
      {
        Square expectedPawnSourceSquare = Square.FromFileAndRank(file, otherSide == SideType.White ? 1 : 6);
        Square expectedPawnTargetSquare = Square.FromFileAndRank(file, otherSide == SideType.White ? 3 : 4);
        bool sawOpponentPawnOnSourceSquare = priorBoard.PieceOnSquare(expectedPawnSourceSquare) == new Piece(otherSide, PieceType.Pawn);
        bool sawOpponentPawnOnTargetSquare = currentBoard.PieceOnSquare(expectedPawnTargetSquare) == new Piece(otherSide, PieceType.Pawn);
        bool didNotSeeOpponentPawnOnTargetSquareBeforeMove = priorBoard.PieceOnSquare(expectedPawnTargetSquare).Type == PieceType.None;
        bool didNotSeeOpponentPawnOnSourceSquareAfterMove = currentBoard.PieceOnSquare(expectedPawnSourceSquare).Type == PieceType.None;
        if (sawOpponentPawnOnSourceSquare && sawOpponentPawnOnTargetSquare
         && didNotSeeOpponentPawnOnSourceSquareAfterMove && didNotSeeOpponentPawnOnTargetSquareBeforeMove)
        {
          return (PositionMiscInfo.EnPassantFileIndexEnum)file;
        }
      }
      return PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
    }


    /// <summary>
    /// Returns the en passant rights status between two boards representing last 2 positions.
    /// </summary>
    /// <param name="tpgRecord"></param>
    /// <param name="priorHistoryIndex"></param>
    /// <param name="currentHistoryIndex"></param>
    /// <returns></returns>
    internal static PositionMiscInfo.EnPassantFileIndexEnum EnPassantOpportunityBetweenTPGRecords(in TPGRecord tpgRecord, int priorHistoryIndex, int currentHistoryIndex)
    {
      Position currentBoard = tpgRecord.HistoryPosition(currentHistoryIndex, false);
      Position priorBoard = tpgRecord.HistoryPosition(priorHistoryIndex, false);
      bool currentWhiteIsToMove = tpgRecord.IsWhiteToMove == 1 == (currentHistoryIndex % 2 == 0);
      Debug.Assert(currentWhiteIsToMove == currentBoard.IsWhite);

      PositionMiscInfo.EnPassantFileIndexEnum ret = EnPassantOpportunityBetweenBoards(priorBoard, currentBoard);
      return ret;
    }


    #region Read/Write utilities

    public static void WriteToZSTFile(string fileName, TPGRecord[] records, int maxRecords = int.MaxValue)
    {
      if (maxRecords > records.Length)
      {
        maxRecords = records.Length;
      }

      FileStream es = new FileStream(fileName, FileMode.Create, FileAccess.Write);
      using (ZstandardStream stream = new(es, CompressionMode.Compress))
      {
        ReadOnlySpan<byte> bufferAsBytes = MemoryMarshal.Cast<TPGRecord, byte>(records.AsSpan().Slice(0, maxRecords));
        stream.Write(bufferAsBytes);
      }
    }

    [ThreadStatic]
    static byte[] tempBufferBytes = null;

    static byte[] GetTempBuffer(int minSize)
    {
      if (tempBufferBytes == null || tempBufferBytes.Length < minSize)
      {
        tempBufferBytes = new byte[minSize];
      }
      return tempBufferBytes;
    }


    public static TPGRecord[] ReadFromZSTFile(string fileName, int numToRead)
    {
      // TODO: Avoid the repeated 
      byte[] rawBuffer = GetTempBuffer(numToRead * Marshal.SizeOf<TPGRecord>());
      TPGRecord[] tpgRecordBuffer = new TPGRecord[numToRead];
      int numReadRaw = 0;
      FileStream es = new FileStream(fileName, FileMode.Open, FileAccess.Read);
      using (ZstandardStream stream = new(es, CompressionMode.Decompress))
      {
        // Uncompressed read
        numReadRaw = StreamUtils.ReadFromStream(stream, rawBuffer, ref tpgRecordBuffer, numToRead);
        if (numReadRaw < numToRead)
        {
          throw new Exception("Input file of insufficient size: " + fileName);
        }
      }
      return tpgRecordBuffer;
    }


    #endregion
  }

}
