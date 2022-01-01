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

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Manages the parsing of batch size file configuration files which specify:
  ///   - maximum batch size (first line), and
  ///   - one more optional subsequent explicit partitions
  /// 
  /// Example file:
  /// 192
  /// 160 = 64,96
  /// 176 = 84,92
  /// 
  /// </summary>
  internal static class BatchSizeFileParser
  {
    /// <summary>
    /// Returns contents of parsed file with specified name.
    /// </summary>
    public static (int maxBatchSize, List<(int, int[])> predefinedPartitions) ParseFromFile(string fn)
    {
      if (!File.Exists(fn))
      {
        throw new Exception($"Specified file used for device specification string does not exist {fn}");
      }

      void ThrowBadLineFormat(string line) => throw new Exception($"{fn}: Batch size lines must be of form such as 160=80,80, saw: {line}");

      string[] lines = File.ReadAllLines(fn);
      if (lines.Length < 2)
      {
        throw new Exception($"{fn}: Batch size file must consist of first line with max batch size followed by one or more lines such as 160=80,80");

      }

      int maxBatchSize = int.Parse(lines[0]);

      List<(int, int[])> partitions = new();
      for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++)
      {
        string[] split = lines[lineIndex].Split("=");
        if (split.Length != 2)
        {
          ThrowBadLineFormat(lines[lineIndex]);
        }
        int size = int.Parse(split[0]);
        
        string[] sizes = split[1].Split(",");
        List<int> sizeList = new();
        foreach (string thisSizeStr in sizes)
        {
          if (!int.TryParse(thisSizeStr, out int thisSize))
          {
            throw new Exception($"{fn}: Requested batch size is invalid number in line {lines[lineIndex]}");
          }

          if (thisSize > maxBatchSize)
          {
            throw new Exception($"{fn}: Requested batch size in line {lines[lineIndex]} exceeds maximum of {maxBatchSize}");
          }
          sizeList.Add(int.Parse(thisSizeStr));
        }
        partitions.Add((size, sizeList.ToArray()));
      }

      return (maxBatchSize, partitions);
    }
  }
}
