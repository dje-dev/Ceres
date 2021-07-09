#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;
using System.IO;
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.UserSettings;
using Ceres.Features.UCI;

#endregion

namespace Ceres.Commands
{
  public static class DispatchCommands
  {
    static (string part1, string part2) SplitLeft(string str)
    {
      int index = str.IndexOf(" ");
      if (index == -1)
        return (str, "");
      else
      {
        string left = str.Substring(0, index);
        string right = str.Substring(index + 1);
        if (left.Contains("="))
        {
          // The leftmost token contains "=" so we interpret
          // this as the left part (feature) having been omitted
          return (null, str);
        }
        else
        {
          return (left, right);
        }
      }
    }


    internal static void ShowErrorExit(string errorString)
    {
      ConsoleColor priorColor = Console.ForegroundColor;
      Console.ForegroundColor = ConsoleColor.Red;
      Console.Write("ERROR: ");
      Console.ForegroundColor = priorColor;
      Console.WriteLine(errorString);
      Console.WriteLine();
      System.Environment.Exit(-1);
    }


    public static void ProcessCommand(string cmd)
    {
      cmd = StringUtils.WhitespaceRemoved(cmd).TrimEnd();
      string[] parts = cmd.Split(" ");

      if (cmd == "")
      {
        // No arguments at all
        LaunchUCI("");
        Environment.Exit(0);
      }
      else if (parts.Length > 0 && parts[0].ToUpper() == "UCI")
      {
        // First argument explicit UCI
        LaunchUCI(cmd.Substring(cmd.IndexOf("UCI ") + 4));
        Environment.Exit(0);
      }
      else if (parts.Length > 0 && parts[0].Contains("="))
      {
        // No command, just some immediate key-value pairs
        LaunchUCI(cmd);
        Environment.Exit(0);
      }


      if (cmd.ToLower().EndsWith("-h") || cmd.ToLower().EndsWith("help"))
        HelpCommands.ProcessHelpCommand(cmd);

      // Command consists command name, then options (sequence of key=value pairs), sometimes followed by a FEN
      (string featureName, string args) = SplitLeft(cmd);
      string keyValueArgs = "";
      string fen = null;

      // Default feature (if not specified) is UCI
      if (featureName == null) featureName = "UCI";

      featureName = featureName.ToUpper();

      // Separate key/value pairs and FEN (if any)
      if ((featureName == "ANALYZE" || featureName == "TOURN") && args != "")
      {
        int posLastEquals = args.LastIndexOf("=");
        if (posLastEquals == -1)
          fen = args;
        else
        {
          int indexEndValue = args.Substring(posLastEquals).IndexOf(" ");
          if (indexEndValue != -1)
          {
            fen = args.Substring(1 + posLastEquals + indexEndValue);
            keyValueArgs = args.Substring(0, posLastEquals + indexEndValue);
          }
          else
            keyValueArgs = args;
        }
      }
      else
        keyValueArgs = args;

      if (fen != null && fen.ToLower() == "startpos") fen = Position.StartPosition.FEN;

      // Extract the feature name

      // Extract and parse options string as sequence of key=value pairs
      //string options = keyValueArgs.Contains(" ") ? keyValueArgs.Substring(keyValueArgs.IndexOf(" ") + 1) : "";
      //KeyValueSetParsed keyValues =  options == "" ? null : new KeyValueSetParsed(options);

      if (featureName == "SUITE")
      {
        FeatureSuiteParams suiteParams = FeatureSuiteParams.ParseSuiteCommand(fen, keyValueArgs);
        FeatureSuiteParams.RunSuiteTest(suiteParams);

      }
      else if (featureName == "SETOPT")
      {
        if (!keyValueArgs.Contains("=")) SetoptError();

        KeyValueSetParsed keyValues = new KeyValueSetParsed(keyValueArgs, null);

        foreach ((string key, string value) in keyValues.KeyValuePairs)
        {
          string keyLower = key.ToLower();
          switch (keyLower)
          {
            case "network":
              CeresUserSettingsManager.Settings.DefaultNetworkSpecString = value;
              break;
            case "device":
              CeresUserSettingsManager.Settings.DefaultDeviceSpecString = value;
              break;
            case "dir-epd":
              CeresUserSettingsManager.Settings.DirEPD = value;
              break;
            case "dir-pgn":
              CeresUserSettingsManager.Settings.DirPGN = value;
              break;
            case "dir-lc0-networks":
              CeresUserSettingsManager.Settings.DirLC0Networks = value;
              break;
            case "dir-tablebases":
              CeresUserSettingsManager.Settings.SyzygyPath = value;
              CeresUserSettingsManager.Settings.DirTablebases = null;
              break;
            case "launch-monitor":
              CeresUserSettingsManager.Settings.LaunchMonitor = bool.Parse(value);
              break;
            case "log-info":
              CeresUserSettingsManager.Settings.LogInfo = bool.Parse(value);
              break;
            case "log-warn":
              CeresUserSettingsManager.Settings.LogWarn = bool.Parse(value);
              break;
            default:
              SetoptError();
              break;
          }

          Console.WriteLine($"Set {key} to {value}");
        }

        Console.WriteLine($"Updating default Ceres settings file {CeresUserSettingsManager.DefaultCeresConfigFileName}");
        CeresUserSettingsManager.SaveToDefaultFile();
      }
      else if (featureName == "TOURN")
      {
        FeatureTournParams tournParams = FeatureTournParams.ParseTournCommand(fen, keyValueArgs);
        FeatureTournParams.RunTournament(tournParams, fen);
        Console.WriteLine(tournParams.ToString());
      }
      else if (featureName == "ANALYZE")
      {
        FeatureAnalyzeParams analyzeParams = FeatureAnalyzeParams.ParseAnalyzeCommand(fen, keyValueArgs);
        analyzeParams.Execute(fen);
      }
      else if (featureName == "SETUP")
      {
        if (File.Exists(CeresUserSettingsManager.DefaultCeresConfigFileName))
        {
          Console.WriteLine();
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, "WARNING: This action will overwrite the Ceres.json file in the current directory.");
          Console.WriteLine();
        }

        CeresUserSettingsManager.DoSetupInitialize();
      }
      else if (featureName == "SYSBENCH")
      {
        FeatureBenchmark.DumpBenchmark();
      }
      else if (featureName == "BACKENDBENCH")
      {
        FeatureBenchmarkBackend backendBench = new FeatureBenchmarkBackend();
        backendBench.ParseFields(keyValueArgs);
        backendBench.Execute();
      }
      else if (featureName == "BENCHMARK")
      {
        FeatureBenchmarkSearch analyzeParams = FeatureBenchmarkSearch.ParseBenchmarkCommand(keyValueArgs);
        analyzeParams.Execute();
      }
      else
        ShowErrorExit("Expected argument to begin with one of the features " + 
                       "UCI, ANALYZE, SUITE, TOURN, SYSBENCH, BACKENDBENCH, BENCHMARK or SETOPT");

    }

    static void SetoptError()
    {
      ShowErrorExit("Expected key=value pairs with keys: { network, device, dir-pgn, dir-epd, dir-lc0networks\r\n"
                   +"                                             dir-tablebases, launch-monitor, log-info, log-warn }");
    }

    private static void LaunchUCI(string keyValueArgs)
    {
      FeatureUCIParams uciParams = FeatureUCIParams.ParseUCICommand(keyValueArgs);

      UCIManager ux = new UCIManager(uciParams.NetworkSpec, uciParams.DeviceSpec, null, null, null,
                                     uciParams.Pruning == false,
                                     CeresUserSettingsManager.Settings.UCILogFile,
                                     CeresUserSettingsManager.Settings.SearchLogFile);


      Console.WriteLine();
      Console.WriteLine("Entering UCI command processing mode.");
      ux.PlayUCI();
    }
  }
}
