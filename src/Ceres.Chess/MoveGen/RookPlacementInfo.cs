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

using System.Runtime.InteropServices;
#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Efficiently represents initial rook placement information used in Chess960.
  /// Each of the 4 fields supports values in the range [0..15].
  /// 
  /// The internal representation is constructed such that a default value of 
  /// rookInitPlacementBits = 0 corresponds to:
  ///   WhiteKRInitPlacement = 0, 
  ///   WhiteQRInitPlacement = 7,
  ///   BlackKRInitPlacement = 0, 
  ///   BlackQRInitPlacement = 7.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public record struct RookPlacementInfo
  {
    /// <summary>
    /// Raw data representation.
    /// </summary>
    private ushort rookInitPlacementBits;

    /// <summary>
    /// Underlying raw storage value.
    /// </summary>
    public ushort RawValue => rookInitPlacementBits;

    // Encoding logic to pack the 4 fields into a single ushort.
    private static byte EncodeValue(byte value, byte defaultValue) => (byte)((value - defaultValue) & 0x0F);

    // Decoding logic to unpack the 4 fields from a single ushort.
    private static byte DecodeValue(byte rawValue, byte defaultValue) => (byte)((rawValue + defaultValue) & 0x0F);

    // Constants to define the default value mapping for each field
    private const byte WhiteKRDefault = 0;
    private const byte WhiteQRDefault = 7;
    private const byte BlackKRDefault = 0;
    private const byte BlackQRDefault = 7;

    public byte WhiteKRInitPlacement
    {
      get => DecodeValue((byte)((rookInitPlacementBits >> 12) & 0x0F), WhiteKRDefault); // Extract and decode bits 12-15
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~(0x0F << 12)) | (EncodeValue(value, WhiteKRDefault) << 12)); // Encode and set bits 12-15
    }

    public byte WhiteQRInitPlacement
    {
      get => DecodeValue((byte)((rookInitPlacementBits >> 8) & 0x0F), WhiteQRDefault); // Extract and decode bits 8-11
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~(0x0F << 8)) | (EncodeValue(value, WhiteQRDefault) << 8)); // Encode and set bits 8-11
    }

    public byte BlackKRInitPlacement
    {
      get => DecodeValue((byte)((rookInitPlacementBits >> 4) & 0x0F), BlackKRDefault); // Extract and decode bits 4-7
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~(0x0F << 4)) | (EncodeValue(value, BlackKRDefault) << 4)); // Encode and set bits 4-7
    }

    public byte BlackQRInitPlacement
    {
      get => DecodeValue((byte)(rookInitPlacementBits & 0x0F), BlackQRDefault); // Extract and decode bits 0-3
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~0x0F) | EncodeValue(value, BlackQRDefault)); // Encode and set bits 0-3
    }


    /// <summary>
    /// Derives a RookPlacementInfo from board state by scanning rank 1 (white) and
    /// rank 8 (black) for the outermost same-color rook on each side of its king.
    /// Castling-rights flags are passed by reference and cleared if no qualifying rook
    /// is found (or the king is not on its back rank), so the result is internally
    /// consistent: every active castling-rights slot maps to a real rook square.
    ///
    /// Files are stored H1-indexed (h=0..a=7), matching FENParser's convention (it writes
    /// rook.SquareIndexStartH1, which equals 7 - Square.File on rank 0 / rank 7). Standard
    /// chess positions (king on e, rooks on a and h) reduce to RookInfo defaults.
    /// </summary>
    /// <param name="pos">Position with piece placement already populated; MiscInfo is not consulted.</param>
    /// <param name="whiteCanOO">In: caller-claimed white kingside right. Out: cleared if no qualifying rook found.</param>
    /// <param name="whiteCanOOO">In/Out: white queenside.</param>
    /// <param name="blackCanOO">In/Out: black kingside.</param>
    /// <param name="blackCanOOO">In/Out: black queenside.</param>
    public static RookPlacementInfo DeriveFromBoard(in Position pos,
                                                    ref bool whiteCanOO, ref bool whiteCanOOO,
                                                    ref bool blackCanOO, ref bool blackCanOOO)
    {
      RookPlacementInfo info = default;
      if (!(whiteCanOO || whiteCanOOO || blackCanOO || blackCanOOO))
      {
        return info;
      }

      // Find king files on rank 1 (white) and rank 8 (black). Castling is only possible
      // when the king sits on its back rank; clear the rights otherwise.
      int wKingFile = -1, bKingFile = -1;
      foreach (PieceOnSquare ps in pos.PiecesEnumeration)
      {
        if (ps.Piece.Type != PieceType.King) continue;
        if (ps.Piece.Side == SideType.White && ps.Square.Rank == 0) wKingFile = ps.Square.File;
        else if (ps.Piece.Side == SideType.Black && ps.Square.Rank == 7) bKingFile = ps.Square.File;
      }
      if (wKingFile < 0) { whiteCanOO = whiteCanOOO = false; }
      if (bKingFile < 0) { blackCanOO = blackCanOOO = false; }

      if (whiteCanOO)
      {
        int f = ScanForOutermostRook(in pos, fromFile: 7, towardKingFile: wKingFile, rank: 0, SideType.White);
        if (f >= 0) info.WhiteKRInitPlacement = (byte)(7 - f);
        else whiteCanOO = false;
      }
      if (whiteCanOOO)
      {
        int f = ScanForOutermostRook(in pos, fromFile: 0, towardKingFile: wKingFile, rank: 0, SideType.White);
        if (f >= 0) info.WhiteQRInitPlacement = (byte)(7 - f);
        else whiteCanOOO = false;
      }
      if (blackCanOO)
      {
        int f = ScanForOutermostRook(in pos, fromFile: 7, towardKingFile: bKingFile, rank: 7, SideType.Black);
        if (f >= 0) info.BlackKRInitPlacement = (byte)(7 - f);
        else blackCanOO = false;
      }
      if (blackCanOOO)
      {
        int f = ScanForOutermostRook(in pos, fromFile: 0, towardKingFile: bKingFile, rank: 7, SideType.Black);
        if (f >= 0) info.BlackQRInitPlacement = (byte)(7 - f);
        else blackCanOOO = false;
      }

      return info;
    }


    /// <summary>
    /// Scans rank from a corner (fromFile = 0 or 7) inward toward the king (exclusive)
    /// and returns the file of the first same-color rook found, or -1 if none.
    /// The first rook found is the outermost — i.e. closest to the corner — which is
    /// the most-likely castling rook in FRC (and the only candidate in standard chess).
    /// </summary>
    private static int ScanForOutermostRook(in Position pos, int fromFile, int towardKingFile, int rank, SideType side)
    {
      Piece target = new Piece(side, PieceType.Rook);
      int step = fromFile > towardKingFile ? -1 : 1;
      for (int f = fromFile; f != towardKingFile; f += step)
      {
        if (pos.PieceOnSquare(Square.FromFileAndRank(f, rank)) == target)
        {
          return f;
        }
      }
      return -1;
    }
  }
}
