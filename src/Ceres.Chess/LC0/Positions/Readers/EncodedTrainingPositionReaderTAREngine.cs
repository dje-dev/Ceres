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

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

using System.Runtime.InteropServices;
using System.Threading;

using SharpCompress.Common;
using SharpCompress.Readers;
using SharpCompress.Readers.Tar;

using Zstandard.Net;

using Ceres.Base.Misc;
using Ceres.Base.Benchmarking;

#endregion


namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Static internal helper class for enumerating all the raw positions
  /// contained in a Leela Chess Zero raw training file (packed with TAR).
  /// 
  /// Alternately, any entries compressed with Zstandard are assumed to have been
  /// compressed to EncodedTrainingPositionCompressed format, and are decompressed and returned.
  /// </summary>
  public static class EncodedTrainingPositionReaderTAREngine
  {
    public const int MAX_POSITIONS_PER_STREAM = 100 * 300; // assume 100 games, worst case 300 ply on average

    [Flags]
    public enum ReaderOptions {  None, FillInMoveNum };


    /// <summary>
    /// Enumerate all the games in a TAR file.
    /// </summary>
    /// <param name="trainingTARFileName"></param>
    /// <param name="processFilePredicate"></param>
    /// <param name="options"></param>
    /// <param name="maxGames"></param>
    /// <param name="maxPositions"></param>
    /// <param name="filterOutFRCGames"></param>
    /// <param name="skipCount"></param>
    /// <param name="maxPosToEvaluate"></param>
    /// <returns></returns>
    public static IEnumerable<(Memory<EncodedTrainingPosition> gamePositions, int indexInGame)> 
      EnumeratedPositions(string trainingTARFileName,
                          Predicate<string> processFilePredicate = null,
                          ReaderOptions options = ReaderOptions.None,
                          int maxGames = int.MaxValue,
                          int maxPositions = int.MaxValue,
                          bool filterOutFRCGames = true,
                          int skipCount = 1, 
                          int maxPosToEvaluate = int.MaxValue)
    {
      var reader = EnumerateGames(trainingTARFileName, processFilePredicate, options, maxGames, maxPositions, filterOutFRCGames);

      int numSeen = 0;
      int numUsed = 0;
      int skipCountBase = Environment.TickCount % skipCount; // for randomization

      // For every game in TAR....
      foreach (Memory<EncodedTrainingPosition> gamePositionsBuffer in reader)
      {
        if (numUsed >= maxPosToEvaluate)
        {
          break;
        }

        // For every position.....
        for (int i = 0; i < gamePositionsBuffer.Length; i++)
        {
          // Skip most positions so we get sampling across many games.
          if ((skipCountBase + numSeen++) % skipCount != skipCount - 1)
          {
            continue;
          }
          else if (numUsed >= maxPosToEvaluate)
          {
            break;
          }

          yield return (gamePositionsBuffer, i); // TODO: could we return a ref instead of struct itself? Or use a static buffer.
          numUsed++;
        }
      }
    }

    public static IEnumerable<Memory<EncodedTrainingPosition>> EnumerateGames(string trainingTARFileName,
                                                                          Predicate<string> processFilePredicate = null,
                                                                          ReaderOptions options = ReaderOptions.None,
                                                                          int maxGames = int.MaxValue,
                                                                          int maxPositions = int.MaxValue,
                                                                          bool filterOutFRCGames = true)
    {
      int numGamesProcessed = 0;
      int numPositionsProcessed = 0;

      foreach (Memory<EncodedTrainingPosition> gamePositions in EnumerateGamesCore(trainingTARFileName, processFilePredicate))
      {
        // Possibly stop early if reached max positions or games.
        if (numPositionsProcessed >= maxPositions
          || numGamesProcessed >= maxGames)
        {
          yield break;
        }

        if (options.HasFlag(ReaderOptions.FillInMoveNum))
        {
          for (int moveNum = 0; moveNum < gamePositions.Length; moveNum++)
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

        if (filterOutFRCGames)
        {
          EncodedTrainingPosition firstPos = gamePositions.Span[0];
          firstPos.MirrorInPlace();

          if (firstPos.FinalPosition != Position.StartPosition)
          {
            continue;
          }
        }

        // Immediately unmirror positions, which are stored in TAR files in mirrored form.
        EncodedTrainingPosition.MirrorPositions(gamePositions.Span, gamePositions.Length);

        yield return gamePositions;

        numGamesProcessed++;
        numPositionsProcessed += gamePositions.Length;
      }
    }


    /// <summary>
    /// Core enumerator which opens TAR file and iterates over all entries matching specified name filter.
    /// </summary>
    /// <param name="trainingTARFileName"></param>
    /// <param name="processFilePredicate"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    static IEnumerable<Memory<EncodedTrainingPosition>> EnumerateGamesCore(string trainingTARFileName,
                                                                           Predicate<string> processFilePredicate = null)
    {      
      EncodedTrainingPosition[] positionsBuffer = new EncodedTrainingPosition[MAX_POSITIONS_PER_STREAM];
      byte[] streamByteBuffer = new byte[Marshal.SizeOf<EncodedTrainingPositionCompressed>() * MAX_POSITIONS_PER_STREAM];


      CheckBuffersAllocated();
      if (!trainingTARFileName.ToUpper().EndsWith("TAR"))
      {
        throw new Exception("expected TAR");
      }


      using (Stream stream = File.OpenRead(trainingTARFileName))
      {
        IReader reader = TarReader.Open(stream);
        while (reader.MoveToNextEntry())
        {
          if (reader.Entry.IsDirectory)
          {
            continue; 
          }

          // Skip if this file does not match our filter.
          if (processFilePredicate != null && !processFilePredicate(reader.Entry.Key.ToUpper()))
          {
            continue;
          }

          string fileName = reader.Entry.Key.ToLower();
          if (fileName.EndsWith("gz") || fileName.EndsWith("zst"))
          {
            // We assume that any files compressed with ZStandard (ZST) are also compressed as EncodedTrainingPositionCompressed
            // so we need to first decompress using ZStandard, then convert the format of the positions
            // to the full EncodedTrainingPosition (uncompressed) structures.
            bool areCompressedPositions = fileName.EndsWith("zst");

            using (EntryStream es = reader.OpenEntryStream())
            {
              // Process all files within
              bool isZST = fileName.ToUpper().EndsWith(".ZST");
              bool isGZ = fileName.ToUpper().EndsWith(".GZ");

              if (!isZST && !isGZ)
              {
                throw new Exception("TAR entry name was expected to be a ZST or GZ file, saw instead " + fileName);
              }

              Stream decompressionStream = isGZ ? new GZipStream(es, CompressionMode.Decompress)
                                                : new ZstandardStream(es, CompressionMode.Decompress);
              using (decompressionStream)
              {
                // Uncompressed read
                int numRead = ReadFromStream(decompressionStream, streamByteBuffer, ref positionsBuffer, areCompressedPositions);

                if (numRead == 0)
                {
                  throw new Exception("Stream contained no data, expected EncodedTrainingPosition");
                }

                bool isPackedGames = positionsBuffer[0].PositionWithBoards.MiscInfo.InfoTraining.Unused1 == EncodedTrainingPositionCompressedConverter.SENTINEL_MARK_FIRST_MOVE_IN_GAME_IN_UNUSED1;
                if (!isPackedGames)
                {
                  // Single game, not packed with multiple.
                  yield return positionsBuffer.AsMemory(0, numRead);
                }
                else
                {
                  // Set of packed games.
                  int curIndex = 0;
                  while (curIndex < numRead)
                  {
                    // Find the start of the next game to know how long this game is.
                    int nextStartIndex = curIndex + 1;
                    while (nextStartIndex < numRead 
                        && positionsBuffer[nextStartIndex].PositionWithBoards.MiscInfo.InfoTraining.Unused1 != EncodedTrainingPositionCompressedConverter.SENTINEL_MARK_FIRST_MOVE_IN_GAME_IN_UNUSED1)
                    {
                      nextStartIndex++;
                    }

                    int lengthThisGame = nextStartIndex - curIndex;
                    Memory<EncodedTrainingPosition> thisGame = new Memory<EncodedTrainingPosition>(positionsBuffer, curIndex, lengthThisGame);
                    thisGame.Span[0].PositionWithBoards.MiscInfo.InfoTraining.SetUnused1(0); // reset the sentinel
                    yield return thisGame;

                    curIndex = nextStartIndex;
                  }
                }
              }
            }
          }
        }
      }
    }



    [ThreadStatic]
    static EncodedTrainingPositionCompressed[] compressedPositionBuffer;

    static void CheckBuffersAllocated()
    {
      if (compressedPositionBuffer == null)
      {
        compressedPositionBuffer = new EncodedTrainingPositionCompressed[MAX_POSITIONS_PER_STREAM];
      }
    }

    static long totalBytesRead = 0;
    static unsafe int ReadFromStream(Stream stream, byte[] rawBuffer, ref EncodedTrainingPosition[] buffer, bool sourceAreEncodedTrainingPositionCompressed)
    {
      CheckBuffersAllocated();

      int bytesRead = 0;
      
      // Read decompressed bytes.
      int thisBytes = 0;
      do
      {
        thisBytes = stream.Read(rawBuffer, bytesRead, rawBuffer.Length - bytesRead);
        bytesRead += thisBytes;
      } while (thisBytes > 0);
      

      Interlocked.Add(ref totalBytesRead, bytesRead);

      int numItems;
      if (sourceAreEncodedTrainingPositionCompressed)
      {
        numItems = bytesRead / Marshal.SizeOf(typeof(EncodedTrainingPositionCompressed));
        int leftover = bytesRead % Marshal.SizeOf(typeof(EncodedTrainingPositionCompressed));
        if (leftover != 0)
        {
          throw new Exception("Number of bytes read not a multiple of EncodedPolicyVectorCompressed size");
        }

        ObjUtils.CopyBytesIntoStructArray(rawBuffer, compressedPositionBuffer, bytesRead); 
        EncodedTrainingPositionCompressedConverter.Decompress(compressedPositionBuffer, buffer, numItems);
      }
      else
      {
        numItems = ObjUtils.CopyBytesIntoStructArray(rawBuffer, buffer, bytesRead);
      }

      return numItems;
    }

  }
}
