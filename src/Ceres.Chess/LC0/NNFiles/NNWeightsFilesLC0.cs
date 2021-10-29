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

using Ceres.Chess.NNFiles;
using System.IO;

#endregion

namespace Ceres.Chess.LC0.NNFiles
{
  /// <summary>
  /// Global static manager of access to NN weights files 
  /// from an LC0 network file (weights protobuf).
  /// </summary>
  public static class NNWeightsFilesLC0
  {
    /// <summary>
    /// Registers all weights files within a specified directory
    /// (matching an optional file name search pattern).
    /// </summary>
    /// <param name="directoryName"></param>
    /// <param name="searchPattern"></param>
    public static void RegisterDirectory(string directoryName, string searchPattern = "*")
    {
      NNWeightsFiles.RegisterDirectory(directoryName, searchPattern, (string id, string fileName) =>
      {
        if (new FileInfo(id).Exists)
        {
          // Full filename directly specified, just directly use it.
          return new NNWeightsFileLC0(id, id);
        }
        else
        {
          // Check if the filename seems to correspond to the requested ID,
          // after stipping off common prefixes and suffixes.
          string cleanedFileName = Path.GetFileName(fileName).ToLower();
          if (cleanedFileName.StartsWith("weights_run1_")) cleanedFileName = cleanedFileName.Replace("weights_run1_", "");
          if (cleanedFileName.StartsWith("weights_run2_")) cleanedFileName = cleanedFileName.Replace("weights_run2_", "");
          if (cleanedFileName.StartsWith("weights_run3_")) cleanedFileName = cleanedFileName.Replace("weights_run3_", "");
          cleanedFileName = cleanedFileName.Replace(".gz", "").Replace(".pb", "");

          if (cleanedFileName == id.ToLower())
          {
            return new NNWeightsFileLC0(id, fileName);
          }
          else
          {
            return null;
          }
        }
      });

    }

  }
}

