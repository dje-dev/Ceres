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
using System.IO;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Allows iterations over all positions/games in a single TAR file 
  /// containing v6 LC0 training data games.
  /// </summary>
  public class EncodedTrainingPositionReaderTAR : IEncodedTrainingPositionReader
  {
    public readonly string FileName;

    public EncodedTrainingPositionReaderTAR(string fileName)
    {
      if (!File.Exists(fileName))
      {
        throw new ArgumentException($"{fileName} does not exist");
      }

      FileName = fileName;
    }


    /// <summary>
    /// Enumerates all games in the TAR.
    /// </summary>
    public IEnumerable<Memory<EncodedTrainingPosition>> EnumerateGames()
    {
      var reader = EncodedTrainingPositionReaderTAREngine.EnumerateRawPos(FileName, s => true, default);

      foreach ((EncodedTrainingPosition[] gamePositionsBuffer, int numPosThisBuffer) in reader)
      {
        yield return new Memory<EncodedTrainingPosition>(gamePositionsBuffer).Slice(0, numPosThisBuffer);
      }
    }


    /// <summary>
    /// Enumerates all positions across all games in the TAR.
    /// </summary>
    public IEnumerable<EncodedTrainingPosition> EnumeratePositions()
    {
      var reader = EncodedTrainingPositionReaderTAREngine.EnumerateRawPos(FileName, s => true, default);

      foreach ((EncodedTrainingPosition[] gamePositionsBuffer, int numPosThisBuffer) in reader)
      {
        for (int i = 0; i < numPosThisBuffer; i++)
        {
          yield return gamePositionsBuffer[i];
        }
      }
    }

  }

}
