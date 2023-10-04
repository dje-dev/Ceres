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
