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

using System.IO;
using Ceres.Base.DataType;
using System.Runtime.InteropServices;
using Ceres.Base.OperatingSystem;
using System.Collections.Generic;
using System.Linq;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods relating to files and directories.
  /// </summary>
  public static class FileUtils
  {
    /// <summary>
    /// Returns if a specified filename is valid on this platform.
    /// </summary>
    /// <param name="filename"></param>
    /// <returns></returns>
    public static bool IsValidFilename(string filename)
    {
      if (string.IsNullOrEmpty(filename))
        return false;

      HashSet<char> invalidChars = new HashSet<char>(Path.GetInvalidFileNameChars());
      return !filename.Any(c => invalidChars.Contains(c));
    }


    /// <summary>
    /// Returns if all of the semicolon/colon separated paths exist.
    /// </summary>
    /// <param name="paths"></param>
    /// <returns></returns>
    public static bool PathsListAllExist(string paths)
    {
      if (paths == null)
      {
        return true;
      }

      // Verify each of the parts exists.
      string[] parts = paths.Split(SoftwareManager.IsLinux ? ":" : ";");
      foreach (string part in parts)
      {
        if (!Directory.Exists(part))
        {
          return false;
        }
      }

      return true;
    }


    /// <summary>
    /// Returns if file with specified name appears to be a 
    /// file compressed using ZIP format.
    /// </summary>
    public static bool IsZippedFile(string fn)
    {
      const byte ZIP_MAGIC_BYTE_0 = 0x1F;
      const byte ZIP_MAGIC_BYTE_1 = 0x8B;

      FileStream file = File.OpenRead(fn);
      bool isZipped = file.Length > 2
                   && file.ReadByte() == ZIP_MAGIC_BYTE_0
                   && file.ReadByte() == ZIP_MAGIC_BYTE_1;
      file.Close();
      return isZipped;
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

  }
}
