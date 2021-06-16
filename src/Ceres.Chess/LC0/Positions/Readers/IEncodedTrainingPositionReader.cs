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

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Interface for an object capable of producing an
  /// IEnumerable of EncodedTrainingPosition.
  /// </summary>
  public interface IEncodedTrainingPositionReader
  {
    /// <summary>
    /// Enumerates all positions sequentiall.
    /// </summary>
    /// <returns></returns>
    IEnumerable<EncodedTrainingPosition> EnumeratePositions();


    /// <summary>
    /// Enumerates as a sequence of batches of specified maximum size.
    /// </summary>
    /// <param name="maxPositionsPerBatch"></param>
    /// <param name="positionSkipCount">difference in sequential position index between selected positions, 1 for all positions</param>
    /// <param name="returnFinalPartialBatch">if any final batch which would be less than requested batch size should be returned</param>
    /// <returns></returns>
    IEnumerable<Memory<EncodedTrainingPosition>> EnumerateBatches(int maxPositionsPerBatch, 
                                                                  int positionSkipCount = 1, 
                                                                  bool returnFinalPartialBatch = true)
    {
      EncodedTrainingPosition[] buffer = new EncodedTrainingPosition[maxPositionsPerBatch];

      int countRead = 0;
      int countWrittenThisBatch = 0;
      foreach (EncodedTrainingPosition pos in EnumeratePositions())
      {
        if (countRead++ % positionSkipCount != 0)
        {
          continue;
        }

        buffer[countWrittenThisBatch++] = pos;

        if (countWrittenThisBatch >= maxPositionsPerBatch)
        {
          yield return buffer;
          countWrittenThisBatch = 0;
        }
      }

      // Return any residual positions not filling batch.
      if (returnFinalPartialBatch && countWrittenThisBatch > 0)
      {
        yield return new Memory<EncodedTrainingPosition>(buffer).Slice(0, countWrittenThisBatch);
      }
    }

  }

}
