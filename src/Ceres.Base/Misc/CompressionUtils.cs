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
using System.IO.Compression;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods for compression/decompression (via GZIP).
  /// </summary>
  public static class CompressionUtils
  {
    /// <summary>
    /// TODO: make this in a central place
    /// </summary>
    /// <param name="fn"></param>
    /// <returns></returns>
    public static byte[] GetDecompressedBytes(string fn)
    {
      byte[] allBytes = new byte[new FileInfo(fn).Length * 2];
      int bytesRead = 0;

      using (FileStream fileReader = System.IO.File.OpenRead(fn))
      using (GZipStream compressionStream = new GZipStream(fileReader, CompressionMode.Decompress))
      {
        // Decompresses and reads data from stream to file
        int readlength = 0;
        byte[] buffer = new byte[65536];
        do
        {
          readlength = compressionStream.Read(buffer, 0, buffer.Length);
          Array.Copy(buffer, 0, allBytes, bytesRead, readlength);
          bytesRead += readlength;
        } while (readlength > 0);
      }

      Array.Resize(ref allBytes, bytesRead);
      return allBytes;
    }


  }
}
