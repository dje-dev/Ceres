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
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.LC0.Boards
{
  /// <summary>
  /// Represents a set of 13 piece planes to represent one position, 
  /// binary as part of a raw position within LZ training data files.
  /// </summary>
  [Serializable()]
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  public unsafe readonly struct EncodedPositionBoard : IEquatable<EncodedPositionBoard>
  {
    public const int NUM_PLANES_PER_BOARD = 13;
    public const int NUM_PIECE_PLANES_PER_BOARD = NUM_PLANES_PER_BOARD - 1;

    public enum Planes
    {
      OurPawns,
      OurKnights,
      OurBishops,
      OurRooks,
      OurQueens,
      OurKing,

      TheirPawns,
      TheirKnights,
      TheirBishops,
      TheirRooks,
      TheirQueens,
      TheirKing,

      Repetitions
    }

    #region Raw structure data

    public readonly EncodedPositionBoardPlane OurPawns;
    public readonly EncodedPositionBoardPlane OurKnights;
    public readonly EncodedPositionBoardPlane OurBishops;
    public readonly EncodedPositionBoardPlane OurRooks;
    public readonly EncodedPositionBoardPlane OurQueens;
    public readonly EncodedPositionBoardPlane OurKing;

    public readonly EncodedPositionBoardPlane TheirPawns;
    public readonly EncodedPositionBoardPlane TheirKnights;
    public readonly EncodedPositionBoardPlane TheirBishops;
    public readonly EncodedPositionBoardPlane TheirRooks;
    public readonly EncodedPositionBoardPlane TheirQueens;
    public readonly EncodedPositionBoardPlane TheirKing;

    public readonly EncodedPositionBoardPlane Repetitions;

    #endregion

    public static int PlaneStartIndex(EncodedPositionBoardPlane.PlanesType type) => (int)type * 64;

    /// <summary>
    /// Constructor from explicit set of encoded planes.
    /// </summary>
    /// <param name="ourPawns"></param>
    /// <param name="ourKnights"></param>
    /// <param name="ourBishops"></param>
    /// <param name="ourRooks"></param>
    /// <param name="ourQueens"></param>
    /// <param name="ourKing"></param>
    /// <param name="theirPawns"></param>
    /// <param name="theirKnights"></param>
    /// <param name="theirBishops"></param>
    /// <param name="theirRooks"></param>
    /// <param name="theirQueens"></param>
    /// <param name="theirKing"></param>
    /// <param name="repetitions"></param>
    public EncodedPositionBoard(EncodedPositionBoardPlane ourPawns, EncodedPositionBoardPlane ourKnights,
                   EncodedPositionBoardPlane ourBishops, EncodedPositionBoardPlane ourRooks,
                   EncodedPositionBoardPlane ourQueens, EncodedPositionBoardPlane ourKing,
                   EncodedPositionBoardPlane theirPawns, EncodedPositionBoardPlane theirKnights,
                   EncodedPositionBoardPlane theirBishops, EncodedPositionBoardPlane theirRooks,
                   EncodedPositionBoardPlane theirQueens, EncodedPositionBoardPlane theirKing,
                   EncodedPositionBoardPlane repetitions)
    {
      OurPawns = ourPawns;
      OurKnights = ourKnights;
      OurBishops = ourBishops;
      OurRooks = ourRooks;
      OurQueens = ourQueens;
      OurKing = ourKing;

      TheirPawns = theirPawns;
      TheirKnights = theirKnights;
      TheirBishops = theirBishops;
      TheirRooks = theirRooks;
      TheirQueens = theirQueens;
      TheirKing = theirKing;
      Repetitions = repetitions;
    }

    const long ALL_ONES_LONG = -1;


    /// <summary>
    /// Constructor from Spans of "ours" and "theirs" planes.
    /// </summary>
    /// <param name="spanOurPawnsKnightsBishipsRooksQueensKing"></param>
    /// <param name="spanTheirPawnsKnightsBishipsRooksQueensKing"></param>
    /// <param name="repetitions"></param>
    public EncodedPositionBoard(Span<BitVector64> spanOurPawnsKnightsBishipsRooksQueensKing,
                   Span<BitVector64> spanTheirPawnsKnightsBishipsRooksQueensKing,
                   bool repetitions)
    {
      // Suppress definite assigment error
      // TODO: In .NET 5 use the new API:
      //   Unsafe.SkipInit<T>(out T value);
      fixed (void* pThis = &this) { }

      fixed (void* targetOurPiecesPtr = &OurPawns)
      {
        Span<BitVector64> targetOurPieces = new Span<BitVector64>(targetOurPiecesPtr, 6);
        spanOurPawnsKnightsBishipsRooksQueensKing.CopyTo(targetOurPieces);
      }

      fixed (void* targetTheirPiecesPtr = &TheirPawns)
      {
        Span<BitVector64> targetTheirPieces = new Span<BitVector64>(targetTheirPiecesPtr, 6);
        spanTheirPawnsKnightsBishipsRooksQueensKing.CopyTo(targetTheirPieces);
      }

      Repetitions = repetitions ? new EncodedPositionBoardPlane(ALL_ONES_LONG) : default;
    }

    public EncodedPositionBoard(BitVector64 ourPawns, BitVector64 ourKnights,
                   BitVector64 ourBishops, BitVector64 ourRooks,
                   BitVector64 ourQueens, BitVector64 ourKing,
                   BitVector64 theirPawns, BitVector64 theirKnights,
                   BitVector64 theirBishops, BitVector64 theirRooks,
                   BitVector64 theirQueens, BitVector64 theirKing,
                   bool repetitions)
    {
      OurPawns = new EncodedPositionBoardPlane(ourPawns);
      OurKnights = new EncodedPositionBoardPlane(ourKnights);
      OurBishops = new EncodedPositionBoardPlane(ourBishops);
      OurRooks = new EncodedPositionBoardPlane(ourRooks);
      OurQueens = new EncodedPositionBoardPlane(ourQueens);
      OurKing = new EncodedPositionBoardPlane(ourKing);

      TheirPawns = new EncodedPositionBoardPlane(theirPawns);
      TheirKnights = new EncodedPositionBoardPlane(theirKnights);
      TheirBishops = new EncodedPositionBoardPlane(theirBishops);
      TheirRooks = new EncodedPositionBoardPlane(theirRooks);
      TheirQueens = new EncodedPositionBoardPlane(theirQueens);
      TheirKing = new EncodedPositionBoardPlane(theirKing);
      Repetitions = repetitions ? new EncodedPositionBoardPlane(ALL_ONES_LONG) : default;
    }

    public EncodedPositionBoard Mirrored => new EncodedPositionBoard(this, true, false);
    public EncodedPositionBoard Reversed => new EncodedPositionBoard(this, false, true);

    public EncodedPositionBoard ReversedAndMirrored => new EncodedPositionBoard(this, true, true);


    /// <summary>
    /// Copy constructor (with optional mirror and/or reverse applied).
    /// </summary>
    /// <param name="other"></param>
    /// <param name="mirror"></param>
    /// <param name="reverse"></param>
    private EncodedPositionBoard(EncodedPositionBoard other, bool mirror, bool reverse)
    {
      if (reverse & mirror)
      {
        OurPawns = other.OurPawns.Reversed.Mirrored;
        OurKnights = other.OurKnights.Reversed.Mirrored;
        OurBishops = other.OurBishops.Reversed.Mirrored;
        OurRooks = other.OurRooks.Reversed.Mirrored;
        OurQueens = other.OurQueens.Reversed.Mirrored;
        OurKing = other.OurKing.Reversed.Mirrored;

        TheirPawns = other.TheirPawns.Reversed.Mirrored;
        TheirKnights = other.TheirKnights.Reversed.Mirrored;
        TheirBishops = other.TheirBishops.Reversed.Mirrored;
        TheirRooks = other.TheirRooks.Reversed.Mirrored;
        TheirQueens = other.TheirQueens.Reversed.Mirrored;
        TheirKing = other.TheirKing.Reversed.Mirrored;
      }
      else
      if (reverse)
      {
        OurPawns = other.OurPawns.Reversed;
        OurKnights = other.OurKnights.Reversed;
        OurBishops = other.OurBishops.Reversed;
        OurRooks = other.OurRooks.Reversed;
        OurQueens = other.OurQueens.Reversed;
        OurKing = other.OurKing.Reversed;

        TheirPawns = other.TheirPawns.Reversed;
        TheirKnights = other.TheirKnights.Reversed;
        TheirBishops = other.TheirBishops.Reversed;
        TheirRooks = other.TheirRooks.Reversed;
        TheirQueens = other.TheirQueens.Reversed;
        TheirKing = other.TheirKing.Reversed;
      }
      else
      {
        OurPawns = mirror ? other.OurPawns.Mirrored : other.OurPawns;
        OurKnights = mirror ? other.OurKnights.Mirrored : other.OurKnights;
        OurBishops = mirror ? other.OurBishops.Mirrored : other.OurBishops;
        OurRooks = mirror ? other.OurRooks.Mirrored : other.OurRooks;
        OurQueens = mirror ? other.OurQueens.Mirrored : other.OurQueens;
        OurKing = mirror ? other.OurKing.Mirrored : other.OurKing;

        TheirPawns = mirror ? other.TheirPawns.Mirrored : other.TheirPawns;
        TheirKnights = mirror ? other.TheirKnights.Mirrored : other.TheirKnights;
        TheirBishops = mirror ? other.TheirBishops.Mirrored : other.TheirBishops;
        TheirRooks = mirror ? other.TheirRooks.Mirrored : other.TheirRooks;
        TheirQueens = mirror ? other.TheirQueens.Mirrored : other.TheirQueens;
        TheirKing = mirror ? other.TheirKing.Mirrored : other.TheirKing;
      }
      Repetitions = other.Repetitions;
    }


    /// <summary>
    /// Resets all planes to zero values.
    /// </summary>
    public void Clear()
    {
      unsafe
      {
        fixed (EncodedPositionBoardPlane* ths = &this.OurPawns)
        {
          Unsafe.InitBlockUnaligned(ths, 0, (uint)Marshal.SizeOf<EncodedPositionBoard>());
        }
      }
    }


    /// <summary>
    /// Converts to an array of bytes, each representing
    /// a single bit (inverse of FromExpandedBytes).
    /// </summary>
    /// <returns></returns>
    public void SetExpandedBytes(byte[] buffer, int startIndex)
    {
      OurPawns.SetBytesRepresentation(buffer, 64 * 0 + startIndex);
      OurKnights.SetBytesRepresentation(buffer, 64 * 1 + startIndex);
      OurBishops.SetBytesRepresentation(buffer, 64 * 2 + startIndex);
      OurRooks.SetBytesRepresentation(buffer, 64 * 3 + startIndex);
      OurQueens.SetBytesRepresentation(buffer, 64 * 4 + startIndex);
      OurKing.SetBytesRepresentation(buffer, 64 * 5 + startIndex);
      TheirPawns.SetBytesRepresentation(buffer, 64 * 6 + startIndex);
      TheirKnights.SetBytesRepresentation(buffer, 64 * 7 + startIndex);
      TheirBishops.SetBytesRepresentation(buffer, 64 * 8 + startIndex);
      TheirRooks.SetBytesRepresentation(buffer, 64 * 9 + startIndex);
      TheirQueens.SetBytesRepresentation(buffer, 64 * 10 + startIndex);
      TheirKing.SetBytesRepresentation(buffer, 64 * 11 + startIndex);
      Repetitions.SetBytesRepresentation(buffer, 64 * 12 + startIndex);
    }


    /// <summary>
    /// Converts to an array of bytes, each representing
    /// a single bit (inverse of FromExpandedBytes).
    /// </summary>
    /// <returns></returns>
    public byte[] ToExpandedBytes()
    {
      ulong[] decoded = new ulong[EncodedPositionBoard.NUM_PLANES_PER_BOARD];
      ExtractPlanesValuesIntoArray(decoded, 0);

      byte[] ret = new byte[64 * EncodedPositionBoard.NUM_PLANES_PER_BOARD];
      int index = 0;
      for (int i = 0; i < decoded.Length; i++)
      {
        BitVector64 bv = new BitVector64(decoded[i]);
        for (int j = 0; j < 64; j++)
        {
          ret[index++] = (byte)(bv.BitIsSet(j) ? 1 : 0);
        }
      }
      return ret;
    }

    /// <summary>
    /// Converts an array of bytes containing the consecutive bitboards
    /// of an EncodedPositionBoard which were expanded (each bit to a byte).
    /// Inverse of ToExpandedBytes.
    /// </summary>
    /// <param name="bytes"></param>
    /// <returns></returns>
    public static EncodedPositionBoard FromExpandedBytes(byte[] bytes)
    {
      return new(
        BitVector64.FromExpandedBytes(bytes, 64 * 0),
        BitVector64.FromExpandedBytes(bytes, 64 * 1),
        BitVector64.FromExpandedBytes(bytes, 64 * 2),
        BitVector64.FromExpandedBytes(bytes, 64 * 3),
        BitVector64.FromExpandedBytes(bytes, 64 * 4),
        BitVector64.FromExpandedBytes(bytes, 64 * 5),

        BitVector64.FromExpandedBytes(bytes, 64 * 6),
        BitVector64.FromExpandedBytes(bytes, 64 * 7),
        BitVector64.FromExpandedBytes(bytes, 64 * 8),
        BitVector64.FromExpandedBytes(bytes, 64 * 9),
        BitVector64.FromExpandedBytes(bytes, 64 * 10),
        BitVector64.FromExpandedBytes(bytes, 64 * 11),
        bytes[^1] == 1  // The 64 bytes are either all 1's or all 0's.
        );
    }


    /// <summary>
    /// Extracts all planes into an array of ulong.
    /// </summary>
    /// <param name="dest"></param>
    /// <param name="destIndex">starting index to receive values</param>
    public void ExtractPlanesValuesIntoArray(ulong[] dest, int destIndex)
    {
      const uint BYTES = sizeof(ulong) * EncodedPositionBoard.NUM_PLANES_PER_BOARD;

      unsafe
      {
        fixed (EncodedPositionBoardPlane* ths = &this.OurPawns)
        fixed (ulong* destPtr = &dest[destIndex])
        {
          Buffer.MemoryCopy(ths, destPtr, BYTES, BYTES);
        }
      }
      destIndex += EncodedPositionBoard.NUM_PLANES_PER_BOARD;
    }


    /// <summary>
    /// Performs integrity check on structure definition.
    /// </summary>
    public static void Validate()
    {
      if (Marshal.SizeOf(typeof(EncodedPositionBoardPlane)) != 8) throw new Exception("Unexpected LZBoardPlane size");
      if (Marshal.SizeOf(typeof(EncodedPositionBoard)) != 8 * EncodedPositionBoard.NUM_PLANES_PER_BOARD) throw new Exception("Unexpected LZBoard size");
    }


    /// <summary>
    /// Returns new board which is reversed and flipped.
    /// </summary>
    public EncodedPositionBoard ReversedAndFlipped
    {
      get
      {
        return new EncodedPositionBoard(TheirPawns.Reversed, TheirKnights.Reversed, TheirBishops.Reversed, TheirRooks.Reversed, TheirQueens.Reversed, TheirKing.Reversed,
                           OurPawns.Reversed, OurKnights.Reversed, OurBishops.Reversed, OurRooks.Reversed, OurQueens.Reversed, OurKing.Reversed,
                           Repetitions);
      }
    }


    /// <summary>
    /// Returns a new board which is flipped.
    /// </summary>
    public EncodedPositionBoard Flipped
    {
      get
      {
        return new EncodedPositionBoard(TheirPawns, TheirKnights, TheirBishops, TheirRooks, TheirQueens, TheirKing,
                           OurPawns, OurKnights, OurBishops, OurRooks, OurQueens, OurKing,
                           Repetitions);
      }
    }


    /// <summary>
    /// Decodes board into corresponding FEN.
    /// </summary>
    /// <param name="weAreWhite"></param>
    /// <returns></returns>
    public string GetFEN(bool weAreWhite)
    {
      string fen = "";

      for (int r = 7; r >= 0; r--)
      {
        if (r != 7) fen += "/";
        int numBlank = 0;
        int startIndex = r * 8;
        for (int c = 7; c >= 0; c--)
        {
          string thisChar = EncodedPositionWithHistory.FENCharAt(this, startIndex + (7 - c), weAreWhite, " ");
          if (thisChar == " ")
            numBlank++;
          else
          {
            if (numBlank > 0)
              fen += numBlank;
            numBlank = 0;
            fen += thisChar;
          }
        }

        if (numBlank > 0) fen += numBlank;
      }

      return fen;
    }

    /// <summary>
    /// Returns ASCII representation of board.
    /// </summary>
    /// <param name="weAreWhite"></param>
    /// <returns></returns>
    public string GetBoardPicture(bool weAreWhite)
    {
      string fen = "";

      for (int r = 7; r >=0; r--)
        fen = fen + EncodedPositionWithHistory.GetRowString(r * 8, this, weAreWhite) + "\r\n";

      fen += "Reps = " + Repetitions.Data + "\r\n";
      return fen;
    }


    /// <summary>
    /// Returns total number of pieces on the board.
    /// </summary>
    public int CountPieces
    {
      get
      {
        return (OurQueens.NumberBitsSet + TheirQueens.NumberBitsSet) +
                (OurRooks.NumberBitsSet + TheirRooks.NumberBitsSet) +
                (OurBishops.NumberBitsSet + TheirBishops.NumberBitsSet) +
                (OurKnights.NumberBitsSet + TheirKnights.NumberBitsSet) +
                (OurPawns.NumberBitsSet + TheirPawns.NumberBitsSet) +
                (OurKing.NumberBitsSet + TheirKing.NumberBitsSet);
      }
    }


    /// <summary>
    /// Returns number of relative points by which side
    /// is ahead in points (using the typical approximate point values of pieces).
    /// </summary>
    public float RelativePointsUs
    {
      get
      {
        float points = 9.0F * (OurQueens.NumberBitsSet - TheirQueens.NumberBitsSet) +
                       5.0F * (OurRooks.NumberBitsSet - TheirRooks.NumberBitsSet) +
                       3.0F * (OurBishops.NumberBitsSet - TheirBishops.NumberBitsSet) +
                       3.0F * (OurKnights.NumberBitsSet - TheirKnights.NumberBitsSet) +
                       1.0F * (OurPawns.NumberBitsSet - TheirPawns.NumberBitsSet);
        return points;
      }

    }


    public bool HasPawns => OurPawns.Data != 0 || TheirPawns.Data != 0;
    public bool HasKnights => OurKnights.Data != 0 || TheirKnights.Data != 0;
    public bool HasBishops => OurBishops.Data != 0 || TheirBishops.Data != 0;
    public bool HasRooks => OurRooks.Data != 0 || TheirRooks.Data != 0;
    public bool HasQueens => OurQueens.Data != 0 || TheirQueens.Data != 0;


    #region Overrides

    public static bool operator ==(EncodedPositionBoard lhs, EncodedPositionBoard rhs) => lhs.Equals(rhs);   

    public static bool operator !=(EncodedPositionBoard lhs, EncodedPositionBoard rhs) => !lhs.Equals((EncodedPositionBoard)rhs);
    

    public override int GetHashCode()
    {
      int part1 = HashCode.Combine(OurPawns, OurKnights, OurBishops, OurRooks, OurQueens, OurKing);
      int part2 = HashCode.Combine(TheirPawns, TheirKnights, TheirBishops, TheirRooks, TheirQueens, TheirKing);

      return HashCode.Combine(part1, part2, Repetitions);
    }


    public bool Equals(EncodedPositionBoard other)
    {
      return OurPawns.Data == other.OurPawns.Data &&
        OurKnights.Data == other.OurKnights.Data &&
        OurBishops.Data == other.OurBishops.Data &&
        OurRooks.Data == other.OurRooks.Data &&
        OurQueens.Data == other.OurQueens.Data &&
        OurKing.Data == other.OurKing.Data &&

        TheirPawns.Data == other.TheirPawns.Data &&
        TheirKnights.Data == other.TheirKnights.Data &&
        TheirBishops.Data == other.TheirBishops.Data &&
        TheirRooks.Data == other.TheirRooks.Data &&
        TheirQueens.Data == other.TheirQueens.Data &&
        TheirKing.Data == other.TheirKing.Data &&

        Repetitions.Data == other.Repetitions.Data;
    }

    #endregion

    const bool CONVERT_VIA_MG_POSITION = false; // performance better when this is false, equivalent behavior

    public static EncodedPositionBoard GetBoard(in Position pos, SideType desiredFromSidePerspective, bool isRepetition)
    {
      if (CONVERT_VIA_MG_POSITION)
      {
        MGPosition mgPos = MGChessPositionConverter.MGChessPositionFromFEN(pos.FEN);
        return GetBoard(in mgPos, desiredFromSidePerspective, isRepetition);
      }
      else
      {
        // PositionCompressed are always stored from perspective of white
        bool alreadyFromCorrectPerspective = (desiredFromSidePerspective == SideType.White);

        Span<BitVector64> bitmaps = stackalloc BitVector64[16];
        pos.InitializeBitmaps(bitmaps, !alreadyFromCorrectPerspective);

        return GetBoard(bitmaps, desiredFromSidePerspective, isRepetition);
      }
    }

    public unsafe void MirrorPlanesInPlace()
    {
      fixed (void* bitmapsR = &this.OurPawns)
      {
        ulong* bitmaps = (ulong*)bitmapsR;
        for (int i=0; i< NUM_PIECE_PLANES_PER_BOARD; i++)
          bitmaps[i] = BitVector64.Mirror(bitmaps[i]);
      }
    }

    public unsafe static void SetBoard(ref EncodedPositionBoard board, in MGPosition mgPos, SideType desiredFromSidePerspective, bool isRepetition)
    {
      // PositionCompressed are always stored from perspective of white
      bool alreadyFromCorrectPerspective = desiredFromSidePerspective == SideType.White;

      ulong A = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.A) : BitVector64.Mirror(BitVector64.Reverse(mgPos.A));
      ulong B = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.B) : BitVector64.Mirror(BitVector64.Reverse(mgPos.B));
      ulong C = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.C) : BitVector64.Mirror(BitVector64.Reverse(mgPos.C));
      ulong D = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.D) : BitVector64.Mirror(BitVector64.Reverse(mgPos.D));

      ulong xCA = ~C & A & ~B; // DJE
      ulong CxBA = C & ~B & A;
      ulong xCBxA = ~C & B & ~A;
      ulong CBxA = C & B & ~A;
      ulong CBA = C & B & A;
      ulong CxBxA = C & ~B & ~A;

      fixed (void* bitmapsR = &board.OurPawns)
      {
        ulong* bitmaps = (ulong*)bitmapsR;

        if (alreadyFromCorrectPerspective)
        {
          bitmaps[6] = D & xCA; // black pawn (en passant or not enpassant)
          bitmaps[0] = ~D & xCA;// white pawn (en passant or not enpassant)

          bitmaps[7] = D & CxBA; // black knight
          bitmaps[1] = ~D & CxBA; // white knight

          bitmaps[8] = D & xCBxA; // black bishop
          bitmaps[2] = ~D & xCBxA; // white bishop

          bitmaps[9] = D & CxBxA; // black rook
          bitmaps[3] = ~D & CxBxA; // white rook

          bitmaps[10] = D & CBxA; // black queen
          bitmaps[4] = ~D & CBxA; // white queen

          bitmaps[11] = D & CBA; // black king
          bitmaps[5] = ~D & CBA; // white king
        }
        else
        {
          bitmaps[0] = D & xCA; // black pawn (en passant or not enpassant)
          bitmaps[6] = ~D & xCA;// white pawn (en passant or not enpassant)

          bitmaps[1] = D & CxBA; // black knight
          bitmaps[7] = ~D & CxBA; // white knight

          bitmaps[2] = D & xCBxA; // black bishop
          bitmaps[8] = ~D & xCBxA; // white bishop

          bitmaps[3] = D & CxBxA; // black rook
          bitmaps[9] = ~D & CxBxA; // white rook

          bitmaps[4] = D & CBxA; // black queen
          bitmaps[10] = ~D & CBxA; // white queen

          bitmaps[5] = D & CBA; // black king
          bitmaps[11] = ~D & CBA; // white king

        }

        bitmaps[12] = isRepetition ? 0xFFFF_FFFF_FFFF_FFFF : 0;
      }
    }

    public static EncodedPositionBoard GetBoard(in MGPosition mgPos, SideType desiredFromSidePerspective, bool isRepetition)
    {
      // PositionCompressed are always stored from perspective of white
      bool alreadyFromCorrectPerspective = desiredFromSidePerspective == SideType.White;

      Span<BitVector64> bitmaps = stackalloc BitVector64[16];

      ulong A = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.A) : BitVector64.Mirror(BitVector64.Reverse(mgPos.A));
      ulong B = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.B) : BitVector64.Mirror(BitVector64.Reverse(mgPos.B));
      ulong C = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.C) : BitVector64.Mirror(BitVector64.Reverse(mgPos.C));
      ulong D = alreadyFromCorrectPerspective ? BitVector64.Mirror(mgPos.D) : BitVector64.Mirror(BitVector64.Reverse(mgPos.D));

      ulong xCA = ~C & A       & ~B; // DJE added ~B
      bitmaps[9] = new BitVector64(D & xCA); // black pawn (en passant or not enpassant)
      bitmaps[1] = new BitVector64(~D & xCA);// white pawn (en passant or not enpassant)

      ulong CxBA = C & ~B & A;
      bitmaps[10] = new BitVector64(D & CxBA); // black knight
      bitmaps[2] = new BitVector64(~D & CxBA); // white knight

      ulong xCBxA = ~C & B & ~A;
      bitmaps[11] = new BitVector64(D & xCBxA); // black bishop
      bitmaps[3] = new BitVector64(~D & xCBxA); // white bishop

      ulong CxBxA = C & ~B & ~A;
      bitmaps[12] = new BitVector64(D & CxBxA); // black rook
      bitmaps[4] = new BitVector64(~D & CxBxA); // white rook

      ulong CBxA = C & B & ~A;
      bitmaps[13] = new BitVector64(D & CBxA); // black queen
      bitmaps[5] = new BitVector64(~D & CBxA); // white queen

      ulong CBA = C & B & A;
      bitmaps[14] = new BitVector64(D & CBA); // black king
      bitmaps[6] = new BitVector64(~D & CBA); // white king

      EncodedPositionBoard board = GetBoard(bitmaps, desiredFromSidePerspective, isRepetition);
      return board;
    }

    static EncodedPositionBoard GetBoard(Span<BitVector64> bitmaps, SideType desiredFromSidePerspective, bool isRepetition)
    {
      // PositionCompressed are always stored from perspective of white
      bool alreadyFromCorrectPerspective = desiredFromSidePerspective == SideType.White;

      EncodedPositionBoard planes;
      const int WHITE = (int)SideType.White << 3;
      const int BLACK = (int)SideType.Black << 3;

      // (unused, P, N, B, R, Q, K, unused, unused, p, n, b, r, q, k, unused)
      // these need to be mapped into "our" followed by "their" pieces (in the same order)

      if (alreadyFromCorrectPerspective)
      {
        planes = new EncodedPositionBoard(bitmaps.Slice(WHITE + 1, 6), bitmaps.Slice(BLACK + 1, 6), isRepetition);

#if NOT
        planesDJE = new LZBoard(bitmaps[WHITE + (int)PieceType.Pawn], bitmaps[WHITE + (int)PieceType.Knight], bitmaps[WHITE + (int)PieceType.Bishop],
                                bitmaps[WHITE + (int)PieceType.Rook], bitmaps[WHITE + (int)PieceType.Queen], bitmaps[WHITE + (int)PieceType.King],

                                bitmaps[BLACK + (int)PieceType.Pawn], bitmaps[BLACK + (int)PieceType.Knight], bitmaps[BLACK + (int)PieceType.Bishop],
                                bitmaps[BLACK + (int)PieceType.Rook], bitmaps[BLACK + (int)PieceType.Queen], bitmaps[BLACK + (int)PieceType.King], isRepetition);
#endif
      }
      else
      {
        planes = new EncodedPositionBoard(bitmaps.Slice(BLACK + 1, 6), bitmaps.Slice(WHITE + 1, 6), isRepetition);
#if NOT
        planesDJE = new LZBoard(bitmaps[BLACK + (int)PieceType.Pawn], bitmaps[BLACK + (int)PieceType.Knight], bitmaps[BLACK + (int)PieceType.Bishop],
                                bitmaps[BLACK + (int)PieceType.Rook], bitmaps[BLACK + (int)PieceType.Queen], bitmaps[BLACK + (int)PieceType.King],

                                bitmaps[WHITE + (int)PieceType.Pawn], bitmaps[WHITE + (int)PieceType.Knight], bitmaps[WHITE + (int)PieceType.Bishop],
                                bitmaps[WHITE + (int)PieceType.Rook], bitmaps[WHITE + (int)PieceType.Queen], bitmaps[WHITE + (int)PieceType.King], isRepetition);
#endif
      }

      return planes;
    }


  }

}
