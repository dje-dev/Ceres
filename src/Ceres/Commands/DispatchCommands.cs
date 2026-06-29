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
using System.Collections.Generic;

using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Chess.NNEvaluators;

using Ceres.MCTS.Params;

using Ceres.Base.OperatingSystem;
using Ceres.Features.NetPublishing;

#endregion

namespace Ceres.Commands
{
  public static class DispatchCommands
  {
    static (string part1, string part2) SplitLeft(string str)
    {
      int index = str.IndexOf(" ");
      if (index == -1)
      {
        return (str, "");
      }
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


    public static void ProcessCommand(string cmd, Action<ParamsSearch> searchModifier, Action<ParamsSelect> selectModifier)
    {
      cmd = StringUtils.WhitespaceRemoved(cmd).TrimEnd();
      string[] parts = cmd.Split(" ");

      if (cmd == "")
      {
        // No arguments at all
        LaunchUCI("", searchModifier, selectModifier);
        Environment.Exit(0);
      }
      else if (parts.Length > 0 && parts[0].ToUpper() == "UCI")
      {
        // First argument explicit UCI
        LaunchUCI(cmd.Substring(cmd.IndexOf("UCI ", StringComparison.OrdinalIgnoreCase) + 4), searchModifier, selectModifier);
        Environment.Exit(0);
      }
      else if (parts.Length > 0 && parts[0].Contains("="))
      {
        // No command, just some immediate key-value pairs
        LaunchUCI(cmd, searchModifier, selectModifier);
        Environment.Exit(0);
      }


      if (cmd.ToLower().EndsWith("-h") || cmd.ToLower().EndsWith("help"))
      {
        HelpCommands.ProcessHelpCommand(cmd);
      }

      // Command consists command name, then options (sequence of key=value pairs), sometimes followed by a FEN
      (string featureName, string args) = SplitLeft(cmd);
      string keyValueArgs = "";
      string fen = null;

      // Default feature (if not specified) is UCI
      if (featureName == null)
      {
        featureName = "UCI";
      }

      featureName = featureName.ToUpper();

      // Separate key/value pairs and FEN (if any)
      if ((featureName == "ANALYZE" || featureName == "TOURN") && args != "")
      {
        int posLastEquals = args.LastIndexOf("=");
        if (posLastEquals == -1)
        {
          fen = args;
        }
        else
        {
          int indexEndValue = args.Substring(posLastEquals).IndexOf(" ");
          if (indexEndValue != -1)
          {
            fen = args.Substring(1 + posLastEquals + indexEndValue);
            keyValueArgs = args.Substring(0, posLastEquals + indexEndValue);
          }
          else
          {
            keyValueArgs = args;
          }
        }
      }
      else
      {
        keyValueArgs = args;
      }

      if (fen != null && fen.ToLower() == "startpos")
      {
        fen = Position.StartPosition.FEN;
      }

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
        backendBench.ExecuteBenchmark(null, null);
      }
      else if (featureName == "BACKENDCOMPARE")
      {
        FeatureBenchmarkBackend backendBench = new FeatureBenchmarkBackend();
        backendBench.ParseFields(keyValueArgs);
        backendBench.ExecuteComparisonTest();
      }
      else if (featureName == "BENCHMARK")
      {
        FeatureBenchmarkSearch analyzeParams = FeatureBenchmarkSearch.ParseBenchmarkCommand(keyValueArgs);
        analyzeParams.Execute();
      }

      else if (featureName == "PERFT")
      {
        FeatureBenchmarkPerft.Execute(keyValueArgs);
      }

      else if (featureName == "SERVER")
      {
        FeatureServerParams.ParseAndExecute(keyValueArgs);
      }

      else if (featureName == "TEST_REMOTE")
      {
        FeatureTestRemote.ParseAndExecute(keyValueArgs);
      }

      else if (featureName == "TEST_REMOTE_CLIENT")
      {
        FeatureTestRemoteClient.ParseAndExecute(keyValueArgs);
      }

      else if (featureName == "GRAPH")
      {
        KeyValueSetParsed keys = new KeyValueSetParsed(args, null);
        string options = keys.GetValue("Options");
        InterprocessCommandManager.EnqueueCommand("graph", options);
      }

      else if (featureName == "PUBLISHNET")
      {
        KeyValueSetParsed keys = new KeyValueSetParsed(keyValueArgs, null);
        string configPath = keys.GetRequiredValue("Config", "PUBLISHNET requires Config=<path to JSON config file>");
        CeresNetGitHubUploader.Run(configPath);
        Environment.Exit(0);
      }
      else if (featureName == "GAME-ANALYZE" || featureName == "GAME-ANALYZE-LC0")
      {
        // Positional args: <pgn file> <move number> <time>.
        // Key=value args (e.g. network=, device=) may appear in any position; they are
        // distinguished from positionals by the presence of "=".
        bool isLC0 = featureName == "GAME-ANALYZE-LC0";
        string cmdName = featureName.ToLower();

        List<string> positionals = new();
        List<string> keyVals = new();
        foreach (string part in args.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
          if (part.Contains("="))
          {
            // Accept "net=" as a convenience alias for "network=".
            keyVals.Add(part.StartsWith("net=", StringComparison.OrdinalIgnoreCase)
                          ? "network=" + part.Substring(4)
                          : part);
          }
          else
          {
            positionals.Add(part);
          }
        }

        if (positionals.Count < 3)
        {
          ShowErrorExit($"{featureName} requires three arguments: <pgn file> <move number> <time>\r\n"
                      + $"  Example: {cmdName} game.pgn 105 10s [network=<net> device=<device>]\r\n"
                      + "  Append \"..\" to the move number to start with Black to move (e.g. 105..).");
        }

        string startup = $"{positionals[0]} {positionals[1]} {positionals[2]}";
        if (isLC0)
        {
          LaunchUCI(string.Join(" ", keyVals), searchModifier, selectModifier, gameAnalyzeLC0Startup: startup);
        }
        else
        {
          LaunchUCI(string.Join(" ", keyVals), searchModifier, selectModifier, gameAnalyzeStartup: startup);
        }
        Environment.Exit(0);
      }
      else
      {
        ShowErrorExit("Expected argument to begin with one of the features " +
                       "UCI, ANALYZE, SUITE, TOURN, SYSBENCH, BACKENDBENCH, BACKENDCOMPARE, BENCHMARK, PERFT, GRAPH, GAME-ANALYZE, GAME-ANALYZE-LC0, PUBLISHNET or SETOPT");
      }
    }

    static void SetoptError()
    {
      ShowErrorExit("Expected key=value pairs with keys: { network, device, dir-pgn, dir-epd, dir-lc0networks\r\n"
                   + "                                      dir-tablebases, launch-monitor, log-info, log-warn }");
    }


    private static void LaunchUCI(string keyValueArgs, Action<ParamsSearch> searchModifier, Action<ParamsSelect> selectModifier,
                                  string gameAnalyzeStartup = null, string gameAnalyzeLC0Startup = null)
    {
      FeatureUCIParams uciParams = FeatureUCIParams.ParseUCICommand(keyValueArgs);

      Action<NNEvaluatorDef, int> searchBenchmarkAction = delegate (NNEvaluatorDef evalDef, int secondsPerMove)
      {
        FeatureBenchmarkSearch.Benchmark(evalDef, SearchLimit.SecondsPerMove(secondsPerMove), false, int.MaxValue);
        Console.WriteLine();
      };

      if (CeresEngineConfig.IsMCGS)
      {
        Action<NNEvaluatorDef, NNEvaluator, int, int, int, int> backendBenchMCGS =
          delegate (NNEvaluatorDef evalDef, NNEvaluator evaluator, int minSize, int maxSize, int stepSize, int numPositions)
          {
            FeatureBenchmarkBackend backendBench = new();
            backendBench.ExecuteBenchmark(evalDef, evaluator);
            Console.WriteLine();
          };

        Ceres.MCGS.UCI.UCIManagerMCGS ux = new Ceres.MCGS.UCI.UCIManagerMCGS(
          uciParams.NetworkSpec, uciParams.DeviceSpec,
          null, null, null, null, null,
          uciParams.Pruning == false,
          CeresUserSettingsManager.Settings.UCILogFile,
          CeresUserSettingsManager.Settings.SearchLogFile,
          backendBenchMCGS,
          searchBenchmarkAction);

        // Inject the live TCEC monitor handler (lives in Ceres.Features, which references
        // Ceres.MCGS, so it cannot be referenced from inside the MCGS UCI manager directly).
        ux.TCECMonitorHandler = engine => Ceres.Features.TCEC.TCECMonitor.Run(engine);

        // Inject the Lc0 analysis handler for "game-analyze-lc0" (also lives in Ceres.Features).
        ux.LC0AnalyzeHandler = a => Ceres.Features.GameEngines.GameAnalyzeLC0Runner.Run(
                                      a.evaluatorDef, a.fenAndMoves, a.movetimeMs, a.outWriter);

        Console.WriteLine();
        Console.WriteLine("Entering UCI command processing mode (MCGS v2).");
        ux.PlayUCI(gameAnalyzeStartup, gameAnalyzeLC0Startup);
      }
      else
      {
        Action<NNEvaluatorDef, NNEvaluator, int, int, int?> backendBenchMCTS =
          delegate (NNEvaluatorDef evalDef, NNEvaluator evaluator, int minSize, int maxSize, int? stepSize)
          {
            FeatureBenchmarkBackend backendBench = new();
            backendBench.ExecuteBenchmark(evalDef, evaluator);
            Console.WriteLine();
          };

        Ceres.MCTS.UCI.UCIManager ux = new Ceres.MCTS.UCI.UCIManager(
          uciParams.NetworkSpec, uciParams.DeviceSpec,
          searchModifier, selectModifier, null, null, null,
          uciParams.Pruning == false,
          CeresUserSettingsManager.Settings.UCILogFile,
          CeresUserSettingsManager.Settings.SearchLogFile,
          backendBenchMCTS,
          searchBenchmarkAction);

        if (gameAnalyzeStartup != null || gameAnalyzeLC0Startup != null)
        {
          Console.WriteLine("WARNING: the game-analyze / game-analyze-lc0 features require MCGS (v2) mode; ignoring startup analysis request.");
        }

        Console.WriteLine();
        Console.WriteLine("Entering UCI command processing mode (MCTS v1).");
        ux.PlayUCI();
      }
    }
  }
}
