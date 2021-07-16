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
using System.Collections.Concurrent;
using Ceres.Chess.LC0.NNFiles;

#endregion

namespace Ceres.Chess.NNFiles
{
  /// <summary>
  /// Global static manager of access to NN weights files.
  /// </summary>
  public static class NNWeightsFiles
  {
    /// <summary>
    /// Set of files which are registered as sources of NN files.
    /// </summary>
    public static ConcurrentDictionary<string, INNWeightsFileInfo> RegisteredFiles = new ConcurrentDictionary<string, INNWeightsFileInfo>();

    /// <summary>
    /// Set of directories which are registered as sources of NN files.
    /// </summary>
    public static ConcurrentBag<NNWeightsFileDirectory> RegisteredDirectories = new ConcurrentBag<NNWeightsFileDirectory>();

    /// <summary>
    /// Registers an individual NN file.
    /// </summary>
    /// <param name="netWeightsID"></param>
    /// <param name="weightsFile"></param>
    public static void RegisterFile(string netWeightsID, INNWeightsFileInfo weightsFile) => RegisteredFiles[netWeightsID] = weightsFile;

    /// <summary>
    /// Registers a directory 
    /// </summary>
    /// <param name="directory"></param>
    /// <param name="searchPattern"></param>
    /// <param name="filenameToWeightsFileFunc"></param>
    public static void RegisterDirectory(string directory, string searchPattern, 
                                         Func<string, string, INNWeightsFileInfo> filenameToWeightsFileFunc)
    {
      if (!Directory.Exists(directory))
      {
        throw new ArgumentException(nameof(directory), $"Specified directory does not exist {directory}");
      }

      RegisteredDirectories.Add(new NNWeightsFileDirectory(directory, filenameToWeightsFileFunc, searchPattern));
    }


    /// <summary>
    /// Attempts to locate and return information about the NN file containing a network with a specified ID.
    /// </summary>
    /// <param name="netWeightsID"></param>
    /// <returns></returns>
    public static INNWeightsFileInfo LookupNetworkFile(string netWeightsID, bool throwExceptionIfMissing = true)
    {
      if (netWeightsID.Contains(":") && File.Exists(netWeightsID.Substring(netWeightsID.IndexOf(":") + 1)))
      {
        // Direct file name with LC0: prefix.
        return NNWeightsFileLC0.LookupOrDownload(netWeightsID.Substring(netWeightsID.IndexOf(":") + 1));
      }
      else if (File.Exists(netWeightsID))
      {
        // Direct file name with no LC0: prefix.
        return new NNWeightsFileLC0(netWeightsID, netWeightsID);
      }
      else if (RegisteredFiles.TryGetValue(netWeightsID, out INNWeightsFileInfo weightsFile))
      {
        // Prefer if found in set of explicitly registered files
        return weightsFile;
      }
      else
      {
        // Next check each file in each of the directories registered
        foreach (NNWeightsFileDirectory directory in RegisteredDirectories)
        {
          foreach (string fileName in Directory.EnumerateFiles(directory.DirectoryName, directory.SearchPattern))
          {
            // Check if the registration source accepts this file and creates an INNWeightsFile
            INNWeightsFileInfo file = directory.FilenameToNetworkIDFunc(netWeightsID, fileName);
            if (file != null)
            {
              RegisteredFiles[netWeightsID] = file;
              return file;
            }
          }
        }
      }

      return !throwExceptionIfMissing
          ? null
          : throw new Exception($"Network {netWeightsID} not registered via Register or discoverable via directories specified via NNWeightsFilesLC0.RegisterDirectory method.");
    }

    
  }
}
