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
using Ceres.Base.DataType;
using Ceres.Chess;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.LC0.Batches;

#endregion

namespace Chess.Ceres.NNEvaluators.TensorRT
{
  /// <summary>
  /// Writes out binary data files containing positions to be 
  /// supplied to TensorRT during Int8 calibration.
  /// </summary>
  public static class TensorRTCalibDataWriter
  {

    public static void WriteCalibFilesNEW()
    {
      const string DIR = @"z:\chess\data\epd\";
      WriteCalibFilesNEW(
        (DIR + @"dje.epd", 1024 * 64)
//        (DIR + @"mea_allsets_nodupes_from_pohl.epd", 1024),
//        (DIR + @"ECE3.epd", 1024)
        );
    }


    public static void WriteCalibFilesNEW(params (string fileName, int count)[] epdFiles)
    {
      List<string> fens = new List<string>();

      int numWritten = 0;
      foreach (var (fileName, count) in epdFiles)
      {
        List<EPDEntry> epds = EPDEntry.EPDEntriesInEPDFile(fileName);
        if (epds.Count < count) throw new Exception($"Insufficient positions in {fileName}");

        for (int i = 0; i < count; i++)
          fens.Add(epds[i].FEN);

        // For debugging purposes force the first two positions to be a well-known position (start position)
        fens[0] = Position.StartPosition.FEN;
        fens[1] = Position.StartPosition.FEN;

        throw new Exception("This needs rewriting to include history planes");

        EncodedPositionBatchFlat batch = null;// new EncodedPositionBatchFlat(EncodedPositionType.PositionOnly, fens.Count);
#if NOT
        for (int i = 0; i < count; i++)
        {
          batch.Add(position);
        }
#endif
        bool append = numWritten > 0;
        WriteCalibFilesNEW(batch,batch.NumPos, append);

        numWritten += count;
      }

      Console.WriteLine($"Count of positions across EPD files {numWritten}");
    }


    public static void WriteCalibFilesNEW(EncodedPositionBatchFlat batch, int batchSize, bool append)
    {
      if (batchSize != batch.NumPos) throw new NotImplementedException(); // NOTE: if we expand this, modify numFiles below as well

      string fn1 = $"d:\\temp\\calib_batch_bitmaps_{0}.dat";
      Console.WriteLine($"writing {fn1}");
      string fn2 = $"d:\\temp\\calib_batch_values_{0}.dat";
      Console.WriteLine($"writing {fn2}");

      // Write plane bitmaps
      WriteAllBytes(fn1, SerializationUtils.SerializeArray(batch.PosPlaneBitmaps), append);
      WriteAllBytes(fn2, SerializationUtils.SerializeArray(batch.PosPlaneValues), append);

      // Write expanded planes (with binary data ready for direct loading into neural net)
      string fn = @"d:\temp\calib_batch_flat.dat";
      Console.WriteLine($"writing {fn} ");
      float[] valuesFlat = batch.ValuesFlatFromPlanes();
      WriteAllBytes(fn, SerializationUtils.SerializeArray(valuesFlat), append);
    }


    static void WriteAllBytes(string path, byte[] bytes, bool append)
    {
      if (!append) System.IO.File.Delete(path);
      using (FileStream stream = new FileStream(path, FileMode.Append))
        stream.Write(bytes, 0, bytes.Length);
    }
  }

}


#if OLD
    // --------------------------------------------------------------------------------------------
    public static void WriteCalibFilesNEW(List<Position> positions, int batchSize)
    {
      // TO DO: don't use this code. Instead use the more modern batch code
      LZTrainingPositionServerBatch batch = LZTrainingPositionServerBatch.GenBatchFromPositions(positions);

      float[] planeValues = batch.PosPlaneValues;

      byte[] bytesBitmaps = ArrayMisc.SerializeArray(batch.PosPlaneBitmaps);
      byte[] bytesValues = ArrayMisc.SerializeArray(planeValues);

      int bytesPerPosBitmaps = bytesBitmaps.Length / positions.Count;
      int bytesPerPosValues = bytesValues.Length / positions.Count;

      byte[] batchBytesBitmaps = new byte[batchSize * bytesPerPosBitmaps];
      byte[] batchBytesValues = new byte[batchSize * bytesPerPosValues];

      int numFiles = positions.Count / batchSize;
      for (int i = 0; i < numFiles; i++)
      {
        string fn1 = $"c:\\temp\\calib_batch_bitmaps_{i}.dat";
        Console.WriteLine($"writing {fn1}");
        string fn2 = $"c:\\temp\\calib_batch_values_{i}.dat";
        Console.WriteLine($"writing {fn2}");

        int offsetStartBitmaps = i * batchSize * bytesPerPosBitmaps;
        Array.Copy(bytesBitmaps, offsetStartBitmaps, batchBytesBitmaps, 0, batchBytesBitmaps.Length);

        int offsetStartValues = i * batchSize * bytesPerPosValues;
        Array.Copy(bytesBitmaps, offsetStartValues, batchBytesValues, 0, batchBytesValues.Length);

        // Write plane bitmaps
        using (FileStream outFile = new FileStream(fn1, FileMode.Create, FileAccess.Write))
          outFile.Write(batchBytesBitmaps);

        // Write plane values
        using (FileStream outFile = new FileStream(fn2, FileMode.Create, FileAccess.Write))
          outFile.Write(batchBytesValues);

        // Write expanded planes
        string fn = @"c:\temp\calib_batch_flat.dat";
        Console.WriteLine($"writing {fn} ");
        using (FileStream outFile = new FileStream(fn, FileMode.Create, FileAccess.Write))
        {
          float[] flatValues = batch.ValuesFlatFromPlanes(LZTrainingPositionServerBatch.FlatValuesEncodingType.Direct);
          outFile.Write(ArrayMisc.SerializeArray(flatValues));          
        }
      }



      Console.WriteLine("done write calibration");
    }
#endif
