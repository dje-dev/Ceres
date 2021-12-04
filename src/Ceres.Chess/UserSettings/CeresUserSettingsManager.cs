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
using System.Reflection;
using System.Text.Json;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.LC0.NNFiles;

#endregion

namespace Ceres.Chess.UserSettings
{
  /// <summary>
  /// Manages the reading and writing of Ceres user settings files.
  /// </summary>
  public static partial class CeresUserSettingsManager
  {
    /// <summary>
    /// Default name of file to which Ceres user settings are serialized (in JSON).
    /// The file is always sourced from the same directory where the Ceres executable runs.
    /// </summary>

    public static string DefaultCeresConfigFileName
    {
      get
      {
        string baseName;
        if (SoftwareManager.IsWSL2 && File.Exists(DEFAULT_SETTINGS_WSL_FN))
        {
          // Optional override for WSL
          baseName = DEFAULT_SETTINGS_WSL_FN;
        }
        else
        {
          baseName = DEFAULT_SETTINGS_FN;
        }

        string directory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        return Path.Combine(directory, baseName);
      }

    }

    const string DEFAULT_SETTINGS_FN = "Ceres.json";
    const string DEFAULT_SETTINGS_WSL_FN = "Ceres.wsl.json";


    /// <summary>
    /// Returns if a default Ceres configuration file exists.
    /// </summary>
    public static bool DefaultConfigFileExists => File.Exists(DefaultCeresConfigFileName);


    /// <summary>
    /// Current Ceres user settings.
    /// </summary>
    public static CeresUserSettings Settings
    {
      get
      {
        if (settings == null && File.Exists(DefaultCeresConfigFileName))
        {
          LoadFromDefaultFile();
        }

        if (settings == null)
        {
          Console.WriteLine();
          Console.WriteLine("ERROR: Ceres requires a settings file typically named Ceres.json.");
          Console.WriteLine("         - this file will be loaded from the working directory automatically");
          Console.WriteLine("         - alternately call CeresUserSettingsManager.LoadFromFile() to load a specific settings file");
          Console.WriteLine("         - running Ceres with the setup command (\"Ceres setup\") assists in initializing a settings file");
          Environment.Exit(1);
          return null; // never reached
        }
        else
        {
          return settings;
        }
      }

      set
      {
        if (value == null)
        {
          throw new Exception("Cannot assign null to Settings.");
        }

        settings = value;
      }
    }

    /// <summary>
    /// Reads settings from the default settings file.
    /// </summary>
    public static void LoadFromDefaultFile() => LoadFromFile(DefaultCeresConfigFileName);

    /// <summary>
    /// Writes current settings to the default settings file.
    /// </summary>
    public static void SaveToDefaultFile() => SaveToFile(DefaultCeresConfigFileName);


    static CeresUserSettings settings = null;

    /// <summary>
    /// Reads settings from specified settings file.
    /// The DirLC0Networks directory (if present) will be registered with NNWeightsFilesLC0.
    /// </summary>
    public static void LoadFromFile(string settingsFileName)
    {
      if (!File.Exists(settingsFileName))
      {
        throw new ArgumentException($"No such file: {settingsFileName}");
      }

      string jsonString = File.ReadAllText(settingsFileName);
      JsonSerializerOptions options = new JsonSerializerOptions() { AllowTrailingCommas = true };
      settings = JsonSerializer.Deserialize<CeresUserSettings>(jsonString, options);

      if (settings.DirLC0Networks != null & settings.DirLC0Networks != "")
      {
        NNWeightsFilesLC0.RegisterDirectory(settings.DirLC0Networks);
      }

      Console.WriteLine($"Ceres user settings loaded from file {settingsFileName}");
    }


    /// <summary>
    /// Writes current settings to specified settings file.
    /// </summary>
    public static void SaveToFile(string settingsFileName)
    {
      string ceresConfigJSON = JsonSerializer.Serialize(Settings, new JsonSerializerOptions() { WriteIndented = true });
      File.WriteAllText(settingsFileName, ceresConfigJSON);
    }

    #region Helpers

    private static void ThrowErrorWithMessage(string nameMissingProperty, string extraInfo)
    {
      throw new Exception($"Ceres user setting {nameMissingProperty} not defined in {DefaultCeresConfigFileName}{extraInfo}");
    }

    public static string URLLC0NetworksValidated
    {
      get 
      {
        if (Settings.URLLC0Networks == null)
        {
          ThrowErrorWithMessage("URLLC0Networks", 
                               @", set to base URL from which LC0 networks available for downloading such as " + 
                               " http://training.lczero.org/networks");
        }

        return Settings.URLLC0Networks;
      }
    }

    public static string GetLC0ExecutableFileName() => GetLC0Executable(true);
   

    internal static string GetLC0Executable(bool executable)
    {
      string dir = Settings.DirLC0Binaries;
      if (dir == null)
      {
        throw new Exception("Required setting DirLC0Binaries not defined in user settings.");
      }
      else if (!Directory.Exists(dir))
      {
        throw new Exception($"Directory {dir} specified by the user setting DirLC0Binaries does not exist. ");
      }

      string extension;
      if (executable)
      {
        extension = OperatingSystem.IsLinux() ? "" : ".exe";
      }
      else
      {
        extension = OperatingSystem.IsLinux() ? ".so" : ".dll";
      }

      string fn = Path.Combine(dir, "lc0" + extension);

      if (!System.IO.File.Exists(fn))
      {
        if (executable)
          {
          Console.WriteLine($"lc0 executable {fn} not found  (using path referenced in DirLC0Binaries user setting.");
        }
        else
        {
          Console.WriteLine($"lc0 library file {fn} not found. This file is a modified Leela Chess Zero binary, uisng a patch provided by the Ceres project.");
        }

        throw new Exception("Missing LC0 binary");
      }

      return fn;
    }


    #endregion
  }

}
