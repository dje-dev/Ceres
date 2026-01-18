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
using System.Runtime.InteropServices;
using Ceres.Base.Misc;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.APIExamples;

/// <summary>
/// Helper methods for Ceres API usage.
/// </summary>
public static class APIHelpers
{
  /// <summary>
  /// Loads Ceres.json from default locations.
  /// Checks current directory first, then home directory.
  /// Throws if not found.
  /// Outputs summary line of context information to console.
  /// </summary>
  public static void InitializeLoadCeres(string description = null)
  {
    string localPath = Path.Combine(Environment.CurrentDirectory, "Ceres.json");
    string homePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Ceres.json");

    string loadedFrom;
    if (File.Exists(localPath))
    {
      CeresUserSettingsManager.LoadFromFile(localPath);
      loadedFrom = localPath;
    }
    else if (File.Exists(homePath))
    {
      CeresUserSettingsManager.LoadFromFile(homePath);
      loadedFrom = homePath;
    }
    else
    {
      throw new FileNotFoundException(
        $"Ceres.json not found. Checked:{Environment.NewLine}" +
        $"  {localPath}{Environment.NewLine}" +
        $"  {homePath}");
    }

    ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"{System.Environment.MachineName}: {Environment.CurrentDirectory}/{description} | {RuntimeInformation.FrameworkDescription} | Ceres.json: {loadedFrom}");
  }
  
}
