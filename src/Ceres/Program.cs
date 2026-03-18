#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres.  If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives 

using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;

using Microsoft.Extensions.Logging;

using Ceres.Base.Misc;
using Ceres.Base.Environment;
using Ceres.Base.OperatingSystem;
using Ceres.Base.CUDA;
using Ceres.Chess;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Params;
using Ceres.MCTS.Environment;

using Ceres.APIExamples;
using Ceres.Commands;
using Ceres.Features;

#endregion

namespace Ceres
{
  public static class Program
  {
    /// <summary>
    /// Startup method for Ceres UCI chess engine and supplemental features.
    /// </summary>
    /// <param name="args"></param>
    static void Main(string[] args)
    {
      // Scan for /v1 flag to select legacy MCTS engine.
      List<string> filteredArgs = new List<string>();
      if (args != null)
      {
        foreach (string arg in args)
        {
          if (arg.Equals("/v1", StringComparison.OrdinalIgnoreCase))
          {
            CeresEngineConfig.ActiveEngine = CeresEngineVersion.V1MCTS;
          }
          else
          {
            filteredArgs.Add(arg);
          }
        }
      }

      LaunchUCI(filteredArgs.ToArray());
    }


    /// <summary>
    /// Perform engine initialization and enters into UCI processing loop.
    /// </summary>
    /// <param name="args"></param>
    /// <param name="searchModifier"></param>
    /// <param name="selectModifier"></param>
    /// <param name="enableLogging"></param>
    public static void LaunchUCI(string[] args, 
                                 Action<ParamsSearch> searchModifier = null, 
                                 Action<ParamsSelect> selectModifier = null,
                                 bool enableLogging = false)
    {    
#if DEBUG
      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "*** WARNING: Ceres binaries built in Debug mode and will run much more slowly than Release");
#endif

      OutputBanner();
      CheckRecursiveOverflow();
      HardwareManager.VerifyHardwareSoftwareCompatability();

      // Load (or cause to be created) a settings file.
      CeresUserSettingsManager.LoadCeresJSON();

      // Configure logging level
      CeresEnvironment.MONITORING_EVENTS = enableLogging;
      LogLevel logLevel = enableLogging ? LogLevel.Information : LogLevel.Critical;
      LoggerTypes loggerTypes = LoggerTypes.WinDebugLogger | LoggerTypes.ConsoleLogger;
      CeresEnvironment.Initialize(loggerTypes, logLevel);

      CeresEnvironment.MONITORING_METRICS = !CommandLineWorkerSpecification.IsWorker
                                           && CeresUserSettingsManager.Settings.LaunchMonitor;

      //      if (CeresUserSettingsManager.Settings.DirLC0Networks != null)
      //        NNWeightsFilesLC0.RegisterDirectory(CeresUserSettingsManager.Settings.DirLC0Networks);

      // Perform low-level hardware initialization.
      MCTSEngineInitialization.BaseInitialize(CeresEnvironment.MONITORING_METRICS, CeresUserSettingsManager.Settings.NUMANode);

      Console.WriteLine();

      //Features.BatchAnalysis.BatchAnalyzer.Test();      return;

      if (args != null && args.Length > 0 && (args[0].ToUpper() == "CUSTOM" || args[0].StartsWith("WORKER")))
      {
        TournamentTest.Test();
        //TournamentTest.TestSFLeela(0, true); return;
        //        SuiteTest.RunSuiteTest(); return;
      }

#if DEBUG
      CheckDebugAllowed();
#endif

      StringBuilder allArgs = new StringBuilder();
      if (args != null)
      {
        for (int i = 0; i < args.Length; i++)
        {
          allArgs.Append(args[i] + " ");
        }
      }

      string allArgsString = allArgs.ToString();

      DispatchCommands.ProcessCommand(allArgsString, searchModifier, selectModifier);


      //  Win32.WriteCrashdumpFile(@"d:\temp\dump.dmp");
    }


    /// <summary>
    /// Because Ceres runs much more slowly under Debug mode (at least 30%)
    /// this check verifies a debug bulid will not run unless explicitly
    /// requested in the options file or environment variables.
    /// </summary>
    private static void CheckDebugAllowed()
    {
      if (!CeresUserSettingsManager.Settings.DebugAllowed
        && Environment.GetEnvironmentVariable("CERES_DEBUG") == null)
      {
        const string MSG = "ERROR: Ceres was compiled in Debug mode and will only run\r\n"
                         + "if the the DebugAllowed option is set to true\r\n"
                         + "or the operating system environment variable CERES_DEBUG is defined.";
        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, MSG);
        System.Environment.Exit(-1);
      }
    }


    const string BannerString =
    @"
|=========================================================|
| Ceres - {ENGINE}                                        |
|                                                         |
| (c) 2020- David Elliott and the Ceres Authors           |
|   Use help to list available commands.                  |
|                                                         |
|  Version {VER}                                       |
|  Runtime {VER}                                       |
|=========================================================|
";

    static void OutputBanner()
    {
      string dotnetVersion = RuntimeInformation.FrameworkDescription;
      (int majorCUDAVersion, int minorCUDAVersion) = CUDADevice.GetCUDAVersion();

      string cudaVersion = $"{majorCUDAVersion}.{minorCUDAVersion}";     

      string engineLabel = CeresEngineConfig.IsMCGS
        ? "Monte Carlo Graph Search Engine (MCGS v2)"
        : "Monte Carlo Tree Search Engine (MCTS v1)";

      string[] bannerLines = BannerString.Split(Environment.NewLine);
      foreach (string line in bannerLines)
      {
        if (line.StartsWith("| Ceres"))
        {
          string ceresLine = $"| Ceres - {engineLabel}";
          int spaceLeft = line.Length - ceresLine.Length;
          string padding = new string(' ', Math.Max(0, spaceLeft - 1));

          ConsoleColor defaultColor = Console.ForegroundColor;
          Console.Write("|");
          Console.ForegroundColor = ConsoleColor.Magenta;
          Console.Write($" Ceres - {engineLabel}{padding}");
          Console.ForegroundColor = defaultColor;
          Console.WriteLine("|");
        }

        else if (line.StartsWith("|  Version"))
        {
          string version = $"|  Version {Chess.CeresVersion.VersionString} [git:{GitInfo.LastCommitSHA}]";
          int spaceLeft = line.Length - version.Length;
          string empty = new string(' ', 3 + spaceLeft - 1);
          Console.WriteLine($"{version}{empty}|");
        }
        else if (line.StartsWith("|  Runtime"))
        {
          string runtime = $"|  Runtime {dotnetVersion} and CUDA {cudaVersion}";
          int spaceLeft = line.Length - runtime.Length;
          string empty = new string(' ', 3 + spaceLeft - 1);
          Console.WriteLine($"{runtime}{empty}|");
        }
        else
        {
          Console.WriteLine(line);
        }
      }
    }



    /// <summary>
    /// Shuts down process if too many Ceres executables are running.
    /// This prevents situation where computer becomes unresponsive
    /// due to infinite cascade of Ceres processes (due to a coding error).
    /// </summary>
    static void CheckRecursiveOverflow()
    {
      int countCeres = 0;
      foreach (Process p in Process.GetProcesses())
      {
        if (p.ProcessName.ToUpper().Contains("CERES"))
          countCeres++;
      }

      const int MAX_CERES_EXECUTABLE = 20;
      if (countCeres > MAX_CERES_EXECUTABLE)
      {
        Console.WriteLine("Shutting down, possible infinite process recursion, too many Ceres executables running running");
        System.Environment.Exit(3);
      }
    }

  }
}
