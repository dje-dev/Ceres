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
using System.Text;

#endregion

namespace Ceres.Chess.NNBackends.LC0EX
{
  /// <summary>
  /// Minimal forward-only reader of the protobuf wire format over a region of a byte array.
  ///
  /// Used instead of generated classes because:
  ///   - the lc0ex protos and the Lc0 net.proto share the "pblczero" package
  ///     (which would collide with the existing generated Pblczero classes), and
  ///   - large payloads (several hundred MB of raw weights) can be returned as
  ///     zero-copy (offset, length) slices into the source array.
  /// </summary>
  internal struct LC0EXProtoReader
  {
    public const int WIRE_VARINT = 0;
    public const int WIRE_FIXED64 = 1;
    public const int WIRE_LENGTH_DELIMITED = 2;
    public const int WIRE_FIXED32 = 5;

    readonly byte[] buffer;
    readonly int end;
    int pos;

    /// <summary>
    /// The underlying buffer (slices returned by ReadBytesSlice refer into it).
    /// </summary>
    public byte[] Buffer => buffer;


    public LC0EXProtoReader(byte[] buffer, int start, int length)
    {
      if (start < 0 || length < 0 || (long)start + length > buffer.Length)
      {
        throw new ArgumentOutOfRangeException(nameof(length), "Protobuf region outside buffer.");
      }

      this.buffer = buffer;
      pos = start;
      end = start + length;
    }


    public LC0EXProtoReader(byte[] buffer) : this(buffer, 0, buffer.Length)
    {
    }


    /// <summary>
    /// Advances to the next field; returns false at end of region.
    /// </summary>
    public bool Next(out int fieldNumber, out int wireType)
    {
      if (pos >= end)
      {
        fieldNumber = 0;
        wireType = 0;
        return false;
      }

      ulong tag = ReadVarint();
      fieldNumber = (int)(tag >> 3);
      wireType = (int)(tag & 7);
      if (fieldNumber == 0)
      {
        throw new FormatException("Invalid protobuf field number 0.");
      }
      return true;
    }


    public ulong ReadVarint()
    {
      ulong result = 0;
      int shift = 0;
      while (true)
      {
        if (pos >= end)
        {
          throw new FormatException("Truncated protobuf varint.");
        }

        byte b = buffer[pos++];
        result |= (ulong)(b & 0x7F) << shift;
        if ((b & 0x80) == 0)
        {
          return result;
        }

        shift += 7;
        if (shift > 63)
        {
          throw new FormatException("Malformed protobuf varint.");
        }
      }
    }


    public uint ReadFixed32()
    {
      if (pos + 4 > end)
      {
        throw new FormatException("Truncated protobuf fixed32.");
      }
      uint v = BitConverter.ToUInt32(buffer, pos);
      pos += 4;
      return v;
    }


    public ulong ReadFixed64()
    {
      if (pos + 8 > end)
      {
        throw new FormatException("Truncated protobuf fixed64.");
      }
      ulong v = BitConverter.ToUInt64(buffer, pos);
      pos += 8;
      return v;
    }


    /// <summary>
    /// Reads a length-delimited field, returning its (offset, length) within Buffer.
    /// </summary>
    public (int offset, int length) ReadBytesSlice()
    {
      ulong len = ReadVarint();
      if (len > (ulong)(end - pos))
      {
        throw new FormatException("Truncated protobuf length-delimited field.");
      }

      int offset = pos;
      pos += (int)len;
      return (offset, (int)len);
    }


    /// <summary>
    /// Reads a length-delimited field and returns a reader over its contents (a sub-message).
    /// </summary>
    public LC0EXProtoReader ReadMessage()
    {
      (int offset, int length) = ReadBytesSlice();
      return new LC0EXProtoReader(buffer, offset, length);
    }


    public byte[] ReadBytesCopy()
    {
      (int offset, int length) = ReadBytesSlice();
      byte[] ret = new byte[length];
      Array.Copy(buffer, offset, ret, 0, length);
      return ret;
    }


    public string ReadString()
    {
      (int offset, int length) = ReadBytesSlice();
      return Encoding.UTF8.GetString(buffer, offset, length);
    }


    /// <summary>
    /// Reads one or more varint values of a repeated scalar field,
    /// accepting both packed (length-delimited) and unpacked encodings.
    /// </summary>
    public void ReadRepeatedVarint(int wireType, List<ulong> values)
    {
      if (wireType == WIRE_VARINT)
      {
        values.Add(ReadVarint());
      }
      else if (wireType == WIRE_LENGTH_DELIMITED)
      {
        LC0EXProtoReader packed = ReadMessage();
        while (packed.pos < packed.end)
        {
          values.Add(packed.ReadVarint());
        }
      }
      else
      {
        throw new FormatException($"Unexpected wire type {wireType} for repeated varint field.");
      }
    }


    public void Skip(int wireType)
    {
      switch (wireType)
      {
        case WIRE_VARINT:
          ReadVarint();
          break;
        case WIRE_FIXED64:
          if (pos + 8 > end) throw new FormatException("Truncated protobuf fixed64.");
          pos += 8;
          break;
        case WIRE_LENGTH_DELIMITED:
          ReadBytesSlice();
          break;
        case WIRE_FIXED32:
          if (pos + 4 > end) throw new FormatException("Truncated protobuf fixed32.");
          pos += 4;
          break;
        default:
          throw new FormatException($"Unsupported protobuf wire type {wireType}.");
      }
    }
  }
}
