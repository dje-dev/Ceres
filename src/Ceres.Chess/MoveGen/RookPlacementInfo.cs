using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

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
  }
}
