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
    /// Returns decompressed bytes from a file with specified name encoded using Gzip.
    /// </summary>
    /// <param name="fn"></param>
    /// <returns></returns>
    public static byte[] GetDecompressedBytes(string fn)
    {
        using (FileStream inStream = File.OpenRead(fn))
        {
        using (GZipStream gzipStream = new GZipStream(inStream, CompressionMode.Decompress))
        {
          using (MemoryStream outStream = new MemoryStream())
          {
            gzipStream.CopyTo(outStream);
            return outStream.ToArray();
          }
        }
      }
    }


  }
}
