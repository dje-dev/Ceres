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

#endregion

namespace Ceres.Chess.UserSettings
{
  /// <summary>
  /// Set of Ceres configuration settings specified by user
  /// and persisted in cofiguration file.
  /// </summary>
  public record CeresUserSettings
  {
    /// <summary>
    /// Directory in which LC0 binaries (executable and/or library) are located.
    /// </summary>
    public string DirLC0Binaries { get; set; } = ".";

    /// <summary>
    /// Directory in which LC0 network weights files are located
    /// </summary>
    public string DirLC0Networks { get; set; } = ".";

    /// <summary>
    /// Directory in which EPD files are located.
    /// </summary>
    public string DirEPD { get; set; } = ".";

    /// <summary>
    /// Directory in which PGN files are located.
    /// </summary>
    public string DirPGN { get; set; } = ".";

    /// <summary>
    /// Directory in which output files from Ceres (e.g. PGN) are located.
    /// </summary>
    public string DirCeresOutput { get; set; } = ".";

    /// <summary>
    /// Directory in which external UCI engines (executables) are found.
    /// </summary>
    public string DirExternalEngines { get; set; } = ".";

    /// <summary>
    /// Directory in which Syzygy tablebases are located, or empty if none.
    /// </summary>
    public string DirTablebases { get; set; } = "";

    /// <summary>
    /// Default value for Ceres network specification string (for network evaluation).
    /// </summary>
    public string DefaultNetworkSpecString { get; set; }

    /// <summary>
    /// Defalut value for Ceres device specification string (for network evaluation).
    /// </summary>
    public string DefaultDeviceSpecString { get; set; }

    /// <summary>
    /// Base URL to download location of LC0 weights files (e.g. http://training.lczero.org/networks).
    /// </summary>
    public string URLLC0Networks { get; set; } = @"http://training.lczero.org/networks";

    #region Metrics and Logging

    /// <summary>
    /// If the .NET metrics monitoring window should be launced at startup.
    /// </summary>
    public bool LaunchMonitor { get; set; } = false;

    /// <summary>
    /// If logging messages of Info severity should be output.
    /// </summary>
    public bool LogInfo { get; set; } = false;

    /// <summary>
    /// If logging messages of Warn severity should be output.
    /// </summary>
    public bool LogWarn { get; set; } = false;

    #endregion
  }

}

