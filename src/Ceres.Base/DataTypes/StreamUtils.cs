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
using System.IO;
using System.Runtime.InteropServices;
using Zstandard.Net;

#endregion

namespace Ceres.Base.DataType
{
  /// <summary>
  /// Static helper methods relating to Streams.
  /// </summary>
  public static class StreamUtils
  {
    public static unsafe void WriteSpanToStream<T>(Stream stream, Span<T> items) where T : unmanaged
    {
      fixed (T* pointer = items)
      {
        Span<byte> span = new Span<byte>(pointer, sizeof(T) * items.Length);
        stream.Write(span);
      }
    }

    public static unsafe void ReadStreamIntoSpan<T>(Stream stream, Span<T> items, int numItems) where T : unmanaged
    {
      if (items.Length < numItems)
      {
        throw new ArgumentOutOfRangeException(nameof(items), "Span too small to receive items");
      }

      fixed (T* pointer = items)
      {
        Span<byte> byteSpan = new Span<byte>(pointer, sizeof(T) * numItems);
        stream.Read(byteSpan);
      }
    }


    public static void WriteSpanToFile<T>(string FN, Span<T> span) where T : unmanaged
    {
      FileStream stream = File.Create(FN);
      try
      {
        WriteSpanToStream<T>(stream, span);
      }
      finally
      {
        stream.Close();
      }
    }


    /// <summary>
    /// Reads a specified number of structs from a file with a specified name.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="stream"></param>
    /// <param name="rawBuffer"></param>
    /// <param name="buffer"></param>
    /// <param name="numStructsToRead"></param>
    /// <returns></returns>
    public static int ReadFromStream<T>(Stream stream, byte[] rawBuffer, ref T[] buffer, int numStructsToRead) where T : unmanaged
    {
      // Read bytes until we have as many as requested (or end of stream).
      int bytesRead = 0;
      int bytesToRead = numStructsToRead * Marshal.SizeOf<T>();
      int thisBytes;
      do
      {
        thisBytes = stream.Read(rawBuffer, 0, bytesToRead);
        bytesRead += thisBytes;
      } while (thisBytes > 0 && bytesRead < bytesToRead);

      return bytesRead == 0 ? 0 : SerializationUtils.DeSerializeArrayIntoBuffer(rawBuffer, bytesRead, ref buffer);
    }


    public static long ReadFileIntoSpan<T>(string FN, Span<T> span) where T : unmanaged
    {
      if (!File.Exists(FN))
      {
        throw new Exception("file not found " + FN);
      }

      long numBytes = new FileInfo(FN).Length;
      if (numBytes == 0)
      {
        throw new Exception("File is empty " + FN);
      }

      long leftoverBytes = numBytes % Marshal.SizeOf<T>();
      if (leftoverBytes != 0)
      {
        throw new Exception("File size inconsistent with integral number of items");
      }

      long numItems = numBytes / Marshal.SizeOf<T>();
      FileStream stream = new FileStream(FN, FileMode.Open);
      try
      {
        ReadStreamIntoSpan(stream, span, (int)numItems);
      }
      finally
      {
        stream.Close();
      }

      return numItems;
    }


    /// <summary>
    /// Compresses array of structs to a stream, using ZStandard compression
    /// (unless stream is null, in which case a new MemoryStream is created and returned).
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="dataBuffer"></param>
    /// <param name="numItems"></param>
    /// <param name="stream"></param>
    /// <param name="compressionLevel"></param>
    /// <returns></returns>
    public unsafe static Stream ZStandardCompressStructArrayToStream<T>(T[] dataBuffer, int numItems,
                                                               Stream stream, int compressionLevel) where T : unmanaged
    {
      // Compress byte array
      stream = stream ?? new MemoryStream();
      using (ZstandardStream compressionStream = new(stream, compressionLevel))
      {
        fixed (T* ptr = &dataBuffer[0])
        {
          compressionStream.Write(new ReadOnlySpan<byte>((byte*)ptr, numItems * Marshal.SizeOf(typeof(T))));
        }
        compressionStream.Close();
      }

      return stream;
    }

  }

}
