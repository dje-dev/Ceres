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
using Ceres.Base.Misc;
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


    /// <summary>
    /// Loads Ceres.json settings file with the following preference order:
    /// 1. Path specified by CERES_JSON environment variable
    /// 2. Current working directory
    /// 3. User's home directory
    /// Throws an exception with detailed information about searched locations if not found.
    /// This method is thread-safe and will only load the settings file once.
    /// </summary>
    public static void LoadCeresJSON()
    {
      if (ceresJSONLoaded)
      {
        return;
      }

      lock (loadCeresJSONLock)
      {
        if (ceresJSONLoaded)
        {
          return;
        }

        const string CERES_JSON_ENV_VAR = "CERES_JSON";
        const string CERES_JSON_FILENAME = "Ceres.json";

        // 1. Check environment variable
        string envPath = Environment.GetEnvironmentVariable(CERES_JSON_ENV_VAR);
        if (!string.IsNullOrWhiteSpace(envPath))
        {
          if (File.Exists(envPath))
          {
            LoadFromFile(envPath);
            ceresJSONLoaded = true;
            return;
          }
        }

        // 2. Check current working directory
        string currentDirPath = Path.Combine(Directory.GetCurrentDirectory(), CERES_JSON_FILENAME);
        if (File.Exists(currentDirPath))
        {
          LoadFromFile(currentDirPath);
          ceresJSONLoaded = true;
          return;
        }

        // 3. Check user's home directory
        string homeDirPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), CERES_JSON_FILENAME);
        if (File.Exists(homeDirPath))
        {
          LoadFromFile(homeDirPath);
          ceresJSONLoaded = true;
          return;
        }

        // File not found in any location - output message and create default file
        const string DEFAULT_CERES_JSON = """
{
  "SyzygyPath": null,
  "DirCeresNetworks": ".",
  "DirLC0Networks": ".",
  "device": "GPU:0"
}
""";

        Console.WriteLine();
        Console.WriteLine($"Could not locate {CERES_JSON_FILENAME}. Searched locations:");

        if (!string.IsNullOrWhiteSpace(envPath))
        {
          Console.WriteLine($"  1. Environment variable {CERES_JSON_ENV_VAR}: '{envPath}' - NOT FOUND");
        }
        else
        {
          Console.WriteLine($"  1. Environment variable {CERES_JSON_ENV_VAR}: not set");
        }

        Console.WriteLine($"  2. Current working directory: '{currentDirPath}' - NOT FOUND");
        Console.WriteLine($"  3. User's home directory: '{homeDirPath}' - NOT FOUND");

        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"*** NOTE: Configuration file {CERES_JSON_FILENAME} not found.");
        Console.WriteLine($"A new {CERES_JSON_FILENAME} will be created in the current working directory with default settings:");
        Console.WriteLine("To proceed, either use UCI setoption commands for network and device or complete this Ceres.json and restart.");
        Console.WriteLine(DEFAULT_CERES_JSON);
        Console.WriteLine();

        File.WriteAllText(currentDirPath, DEFAULT_CERES_JSON);
        LoadFromFile(currentDirPath);
        ceresJSONLoaded = true;
      }
    }


    static CeresUserSettings settings = null;
    static readonly object loadCeresJSONLock = new object();
    static bool ceresJSONLoaded = false;

    /// <summary>
    /// Reads settings from specified settings file.
    /// The DirLC0Networks directory (if present) will be registered with NNWeightsFilesLC0.
    /// Supports property aliases:
    ///   - "network" for "DefaultNetworkSpecString"
    ///   - "device" for "DefaultDeviceSpecString"
    /// </summary>
    public static void LoadFromFile(string settingsFileName)
    {
      if (!File.Exists(settingsFileName))
      {
        throw new ArgumentException($"No such file: {settingsFileName}");
      }

      string jsonString = File.ReadAllText(settingsFileName);

      // Preprocess JSON to map property aliases to canonical names
      jsonString = MapPropertyAliases(jsonString);

      JsonSerializerOptions options = new JsonSerializerOptions() { AllowTrailingCommas = true };
      settings = JsonSerializer.Deserialize<CeresUserSettings>(jsonString, options);

      if (settings.DirLC0Networks != null & settings.DirLC0Networks != "")
      {
        NNWeightsFilesLC0.RegisterDirectory(settings.DirLC0Networks);
      }

      FileInfo ceresFileInfo = new FileInfo(settingsFileName);
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Ceres user settings loaded from {ceresFileInfo.FullName}");
    }


    /// <summary>
    /// Maps property aliases to their canonical names in JSON string.
    /// Supports:
    ///   - "network" --> "DefaultNetworkSpecString"
    ///   - "device" --> "DefaultDeviceSpecString"
    /// </summary>
    private static string MapPropertyAliases(string jsonString)
    {
      using JsonDocument doc = JsonDocument.Parse(jsonString, new JsonDocumentOptions { AllowTrailingCommas = true });

      using MemoryStream stream = new MemoryStream();
      using (Utf8JsonWriter writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
      {
        writer.WriteStartObject();

        foreach (JsonProperty property in doc.RootElement.EnumerateObject())
        {
          // Map aliases to canonical property names
          string propertyName = property.Name switch
          {
            "network" => "DefaultNetworkSpecString",
            "device" => "DefaultDeviceSpecString",
            _ => property.Name
          };

          writer.WritePropertyName(propertyName);
          property.Value.WriteTo(writer);
        }

        writer.WriteEndObject();
      }

      return System.Text.Encoding.UTF8.GetString(stream.ToArray());
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

      string baseEXEName = Settings.LC0ExeName;

      string fn = Path.Combine(dir, baseEXEName + extension);

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
