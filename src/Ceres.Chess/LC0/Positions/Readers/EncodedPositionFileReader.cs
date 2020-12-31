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
  /// Static helper method for enumerating all the raw positions
  /// contained in a Leela Chess Zero raw training file (packed withTAR).
  /// </summary>
  public static class EncodedPositionFileReader
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
          if (numPositionsProcessed >= maxPositions) yield break;

          if (!reader.Entry.IsDirectory)
          {
            // Skip if this file does not match our filter
            if (processFilePredicate != null && !processFilePredicate(reader.Entry.Key.ToUpper()))
              continue;

            using (EntryStream es = reader.OpenEntryStream())
            {
              numGamesProcessed++;
              //if (numGamesProcessed % 1000 == 0) Console.WriteLine("  games read " + numGamesProcessed + " " + (bytesWritten / 1_000_000_000.0) + " GB");

              // Process all GZIP files within
              using (GZipStream decompressionStream = new GZipStream(es, CompressionMode.Decompress))
              {
                if (numGamesProcessed >= maxGames) yield break;

                // Uncompressed read
                const bool MIRROR_PLANES = true; // The board convention differs between this file storage format and what is sent to the neural network input
                int numRead = ReadFromStream(decompressionStream, buffer, ref rawPosBuffer, MIRROR_PLANES);

                if (options.HasFlag(ReaderOptions.FillInMoveNum))
                {
                  for (int moveNum = 0; moveNum < numRead; moveNum++)
                  {
                    // TO DO: Find a more elegant way of setting value; this reflection does not work due to copies being made
                    // FieldInfo fieldMoveCount = typeof(LZPositionMiscInfo).GetField("MoveCount");
                    //fieldMoveCount.SetValue(rawPosBuffer[moveNum].MiscInfo, moveCountValue);

                    // We start counting at 1, and truncate at 255 due to size being byte
                    // WARNING: this potential truncation causes these values to be not always correct
                    byte moveCountValue = moveNum >= 254 ? (byte)255 : (byte)(moveNum + 1);
                    SetMoveNum(rawPosBuffer, moveNum, moveCountValue);
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


    private unsafe static void SetMoveNum(EncodedTrainingPosition[] rawPos, int index, byte value)
    {
      fixed (byte *moveCountPtr = &rawPos[index].Position.MiscInfo.InfoPosition.MoveCount)
        *moveCountPtr = value;
    }


    public static int ReadFromStream(Stream stream, byte[] rawBuffer, ref EncodedTrainingPosition[] buffer, bool mirrorBoardBitmaps)
    {
      // Read decompressed bytes
      int bytesRead = stream.Read(rawBuffer, 0, rawBuffer.Length);
      if (bytesRead == 0) throw new Exception(" trying to read " + rawBuffer.Length);

      // If this is legacy V3 data, remap it on the fly to look like V2
      bool isV3 = rawBuffer[0] == 3;
      if (isV3)
      {
        if (bytesRead % EncodedTrainingPosition.V3_LEN != 0) throw new Exception("data not of expected length");
        int numPos = bytesRead / EncodedTrainingPosition.V3_LEN;

        // Spread out the V3 data into the V4 structures
        byte[] expandedBytes = new byte[numPos * EncodedTrainingPosition.V4_LEN];
        for (int i = 0; i < numPos; i++)
          Array.Copy(rawBuffer, i * EncodedTrainingPosition.V3_LEN, expandedBytes, i * EncodedTrainingPosition.V4_LEN, EncodedTrainingPosition.V3_LEN);

        rawBuffer = expandedBytes;
        bytesRead = numPos * EncodedTrainingPosition.V4_LEN;
      }

      // Convert to LeelaTrainingEncodedPos
      int numDeserialized = SerializationUtils.DeSerializeArrayIntoBuffer<EncodedTrainingPosition>(rawBuffer, bytesRead, ref buffer);

#if NOT
      if (!isV3)
      {
        // V4 data has some probabilities NaNs (indicates illegal move)
        // But we zero that out
        for (int i = 0; i < numDeserialized; i++)
          for (int j = 0; j < EncodedPolicyVector.POLICY_VECTOR_LENGTH; j++)
            if (float.IsNaN(buffer[i].Probabilities[j]))
              buffer[i].Probabilities[j] = 0f;
      }
#endif

      if (mirrorBoardBitmaps)
      {
        for (int i = 0; i < numDeserialized; i++)
          buffer[i].Position.BoardsHistory.MirrorBoardsInPlace();
      }
      return numDeserialized;
    }


  }
}
