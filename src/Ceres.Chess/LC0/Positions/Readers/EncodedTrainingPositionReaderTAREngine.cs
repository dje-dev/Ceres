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
// TODO: restore this class

using Ceres.Chess;
using SharpCompress.Common;
using SharpCompress.Readers;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using Ceres.Base.DataType;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Static internal helper class for enumerating all the raw positions
  /// contained in a Leela Chess Zero raw training file (packed with TAR).
  /// </summary>
  public static class EncodedTrainingPositionReaderTAREngine
  {
    [Flags]
    public enum ReaderOptions {  None, FillInMoveNum };

    public static IEnumerable<(EncodedTrainingPosition[], int)> EnumerateRawPos(string trainingTARFileName,
                                                                              Predicate<string> processFilePredicate = null,
                                                                              ReaderOptions options = ReaderOptions.None,
                                                                              int maxGames = int.MaxValue,
                                                                              int maxPositions = int.MaxValue)
    {
      if (!trainingTARFileName.ToUpper().EndsWith("TAR")) throw new Exception("expected TAR");

      byte[] buffer = new byte[10 * 1000 * 1000];

      const int MAX_POS_PER_GAME = 2000;
      EncodedTrainingPosition[] rawPosBuffer = new EncodedTrainingPosition[MAX_POS_PER_GAME];

      int numGamesProcessed = 0;
      int numPositionsProcessed = 0;
      using (Stream stream = System.IO.File.OpenRead(trainingTARFileName))
      {
        IReader reader = ReaderFactory.Open(stream);
        while (reader.MoveToNextEntry())
        {
          if (numPositionsProcessed >= maxPositions)
          {
            yield break;
          }

          if (!reader.Entry.IsDirectory && reader.Entry.Key.ToLower().EndsWith("gz"))
          {
            // Skip if this file does not match our filter
            if (processFilePredicate != null && !processFilePredicate(reader.Entry.Key.ToUpper()))
            {
              continue;
            }

            using (EntryStream es = reader.OpenEntryStream())
            {
              numGamesProcessed++;
              //if (numGamesProcessed % 1000 == 0) Console.WriteLine("  games read " + numGamesProcessed + " " + (bytesWritten / 1_000_000_000.0) + " GB");

              // Process all GZIP files within
              using (GZipStream decompressionStream = new GZipStream(es, CompressionMode.Decompress))
              {
                if (numGamesProcessed >= maxGames) yield break;

                // Uncompressed read
                int numRead = ReadFromStream(decompressionStream, buffer, ref rawPosBuffer);

                if (options.HasFlag(ReaderOptions.FillInMoveNum))
                {
                  for (int moveNum = 0; moveNum < numRead; moveNum++)
                  {
                    // TO DO: Find a more elegant way of setting value; this reflection does not work due to copies being made
                    // FieldInfo fieldMoveCount = typeof(LZPositionMiscInfo).GetField("MoveCount");
                    //fieldMoveCount.SetValue(rawPosBuffer[moveNum].MiscInfo, moveCountValue);

                    // We start counting at 1, and truncate at 255 due to size being byte
                    // WARNING: this potential truncation causes these values to be not always correct
                    //byte moveCountValue = moveNum >= 254 ? (byte)255 : (byte)(moveNum + 1);
                    //SetMoveNum(rawPosBuffer, moveNum, moveCountValue);

                    throw new NotImplementedException();
                  }
                }

                numPositionsProcessed += numRead;

                yield return (rawPosBuffer, numRead);
              }
            }
          }
        }
      }
    }


    public static int ReadFromStream(Stream stream, byte[] rawBuffer, ref EncodedTrainingPosition[] buffer)
    {
      // Read decompressed bytes
      int bytesRead = stream.Read(rawBuffer, 0, rawBuffer.Length);

      return bytesRead == 0
          ? throw new Exception(" trying to read " + rawBuffer.Length)
          : SerializationUtils.DeSerializeArrayIntoBuffer(rawBuffer, bytesRead, ref buffer);
    }


  }
}
