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

#endregion

namespace Ceres.Chess.NNFiles
{
  /// <summary>
  /// Represents a directory on the file system which contains NN weights files.
  /// </summary>
  public class NNWeightsFileDirectory
  {
    /// <summary>
    /// Full path to directory.
    /// </summary>
    public readonly string DirectoryName;

    /// <summary>
    /// Optional filename search pattern to filter set of recognized files.
    /// </summary>
    public readonly string SearchPattern;

    /// <summary>
    /// Function which maps full fullname to corresponding short ID (network ID).
    /// </summary>
    public readonly Func<string, string, INNWeightsFileInfo> FilenameToNetworkIDFunc;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="directoryName">full directory path</param>
    /// <param name="filenameToWeightsFileFunc">function which maps full fullname to corresponding short ID (network ID)</param>
    /// <param name="searchPattern">optional filename search pattern to filter set of recognized files</param>
    public NNWeightsFileDirectory(string directoryName, Func<string, string, INNWeightsFileInfo> filenameToNetworkIDFunc, string searchPattern = "*")
    {
      DirectoryName = directoryName;
      SearchPattern = searchPattern;
      FilenameToNetworkIDFunc = filenameToNetworkIDFunc;
    }
  }
}
