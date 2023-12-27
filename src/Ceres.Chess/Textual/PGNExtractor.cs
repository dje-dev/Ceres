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

using Ceres.Chess.Games;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Chess.Textual
{
  public static class PGNExtractor
  {
    /// <summary>
    /// Extracts a subset of positions for an input file (PGN or EPD)
    /// which match a specified predicate and writes them to an output file (PGN or EPD).
    /// 
    /// In the case of PGN, the game is scanned up until the point if/where the predicate is satisfied
    /// and the game is truncated at this first satisfying position.
    /// </summary>
    /// <param name="sourcePGN"></param>
    /// <param name="outputFN"></param>
    /// <param name="truncateAtPos"></param>
    /// <param name="maxPositions"></param>
    /// <param name="idString"></param>
    /// <exception cref="Exception"></exception>
    public static void ExtractFromPGN(string sourcePGN, string outputFN,
                                      Predicate<Position> truncateAtPos = null,
                                      int maxPositions = int.MaxValue,
                                      string idString = null)
    {
      bool outputPGN = outputFN.ToUpper().EndsWith("PGN");
      bool outputEPD = outputFN.ToUpper().EndsWith("EPD");
      if (!outputPGN && !outputEPD)
      {
        throw new Exception("Invalid outputFN type, expect extension to be PGN or EPD: " + outputFN);
      }

      if (File.Exists(outputFN))
      {
        throw new Exception("Output file already exists: " + outputFN);
      }


      using TextWriter textWriter = (TextWriter)new StreamWriter(new FileStream(outputFN, FileMode.Create));

      PositionsWithHistory allPos = PositionsWithHistory.FromEPDOrPGNFile(sourcePGN, maxPositions, truncateAtPos);
      foreach (PositionWithHistory thisPos in allPos)
      {
        PGNWriter pgnWriter = outputPGN ? new PGNWriter(idString ?? "Extracted", "Player1", "Player2") : null;

        if (thisPos.FinalPosition.CalcTerminalStatus() == GameResult.Unknown) // Do not output terminal positions
        {
          if (outputPGN)
          {
            pgnWriter.WriteMoveSequence(thisPos);
            pgnWriter.WriteResultUnknown();
            textWriter.WriteLine(pgnWriter.GetText());
          }
          else
          {
            textWriter.WriteLine(thisPos.FinalPosition.FEN);
          }
        }
      }
    }

  }
}

