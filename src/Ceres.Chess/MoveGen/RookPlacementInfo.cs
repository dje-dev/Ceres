using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// Efficiently represents initial rook palcement information used in Chess960.
  /// Each of the 4 fields supports values in the range [0..15].
  /// 
  /// Initialized such that values returned by default object will return as 15.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public record struct RookPlacementInfo
  {
    // Data representation. Encoded such that the default C#
    // value of 0 will map to our default value of 15.
    private ushort rookInitPlacementBits;

    private static byte EncodeValue(byte value) => (byte)(((value + 1) & 0x0F) % 16);
    private static byte DecodeValue(byte value) => (byte)(((value - 1) & 0x0F) % 16);

    public byte WhiteKRInitPlacement
    {
      get => DecodeValue((byte)((rookInitPlacementBits >> 12) & 0x0F)); // Extract bits 12-15
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~(0x0F << 12)) | (EncodeValue(value) << 12)); // Set bits 12-15
    }

    public byte WhiteQRInitPlacement
    {
      get => DecodeValue((byte)((rookInitPlacementBits >> 8) & 0x0F)); // Extract bits 8-11
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~(0x0F << 8)) | (EncodeValue(value) << 8)); // Set bits 8-11
    }

    public byte BlackKRInitPlacement
    {
      get => DecodeValue((byte)((rookInitPlacementBits >> 4) & 0x0F)); // Extract bits 4-7
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~(0x0F << 4)) | (EncodeValue(value) << 4)); // Set bits 4-7
    }

    public byte BlackQRInitPlacement
    {
      get => DecodeValue((byte)(rookInitPlacementBits & 0x0F)); // Extract bits 0-3
      set => rookInitPlacementBits = (ushort)((rookInitPlacementBits & ~0x0F) | EncodeValue(value)); // Set bits 0-3
    }
  }
}
