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
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Allows iterations over all positions/games in a single TPG
  /// (training position generator) file containing a sequence 
  /// of raw v6 LC0 training records (in GZIP format)
  /// </summary>
  public class EncodedTrainingPositionReaderTPG : IEncodedTrainingPositionReader, IDisposable
  {
    /// <summary>
    /// Name of TPG file containing data.
    /// </summary>
    public readonly string FileName;

    const int BUFFER_POSITIONS_PER_BUFFER = 1024;

    FileStream rf;
    Stream gu;

    byte[] buffer;


    /// <summary>
    /// Constructor to iterate over all positions in a specified file.
    /// </summary>
    public EncodedTrainingPositionReaderTPG(string fileName)
    {
      if (!File.Exists(fileName))
      {
        throw new ArgumentException($"{fileName} does not exist");
      }

      Debug.Assert(EncodedTrainingPosition.V6_LEN == Marshal.SizeOf<EncodedTrainingPosition>());

      FileName = fileName;

      rf = new FileStream(FileName, FileMode.Open, FileAccess.Read);
      gu = new GZipStream(rf, CompressionMode.Decompress);

      buffer = new byte[BUFFER_POSITIONS_PER_BUFFER * EncodedTrainingPosition.V6_LEN];
    }


    /// <summary>
    /// Enumerates over all positions.
    /// </summary>
    public IEnumerable<EncodedTrainingPosition> EnumeratePositions()
    {
      while (true)
      {
        EncodedTrainingPosition[] block = Read(BUFFER_POSITIONS_PER_BUFFER).ToArray();
        if (block != null)
        {
          foreach (EncodedTrainingPosition position in block)
          {
            yield return position;
          }
        }
        else
        {
          yield break;
        }
      }

    }


    /// <summary>
    /// Reads a specified number of positions from the file at the current position.
    /// </summary>
    public ReadOnlySpan<EncodedTrainingPosition> Read(int numPositions)
    {
      if (buffer == null || buffer.Length < numPositions)
      {
        buffer = new byte[numPositions * Marshal.SizeOf<EncodedTrainingPosition>()];
      }

      int numBytes;
      checked
      {
        numBytes = numPositions * EncodedTrainingPosition.V6_LEN;
      }
      int numBytesRead = gu.Read(buffer, 0, numBytes);
      if (numBytesRead == 0) return null;

      Span<byte> bufferSpan = buffer.AsSpan().Slice(0, numBytesRead);
      ReadOnlySpan<EncodedTrainingPosition> bufferAsPositions = MemoryMarshal.Cast<byte, EncodedTrainingPosition>(bufferSpan);
      return bufferAsPositions;
    }


    public void Dispose()
    {
      gu.Dispose();
      rf.Dispose();
    }

  }
}