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
using System.Diagnostics;
using System.Text;

using Microsoft.Extensions.Logging;

using Ceres.Base.Misc;
using Ceres.Base.Environment;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Environment;
using Ceres.APIExamples;
using Ceres.Commands;
using Ceres.Features;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Ceres.Base.CUDA;
using ManagedCuda;

#endregion

namespace Ceres
{
  static class Program
  {
    /// <summary>
    /// Startup method for Ceres UCI chess engine and supplemental features.
    /// </summary>
    /// <param name="args"></param>
    static void Main(string[] args)
    {
#if DEBUG
      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "*** WARNING: Ceres binaries built in Debug mode and will run much more slowly than Release");
#endif

      OutputBanner();
      CheckRecursiveOverflow();
      HardwareManager.VerifyHardwareSoftwareCompatability();

      // Load (or cause to be created) a settings file.
      if (!CeresUserSettingsManager.DefaultConfigFileExists)
      {
        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"*** NOTE: Configuration file {CeresUserSettingsManager.DefaultCeresConfigFileName} not found in working directory.");
        Console.WriteLine();
        Console.WriteLine($"Prompting for 4 configuration values to be written to Ceres.json:");
        CeresUserSettingsManager.DoSetupInitialize();
      }

      // Configure logging level
      const bool LOG = false;
      CeresEnvironment.MONITORING_EVENTS = LOG;
      LogLevel logLevel = LOG ? LogLevel.Information : LogLevel.Critical;
      LoggerTypes loggerTypes = LoggerTypes.WinDebugLogger | LoggerTypes.ConsoleLogger;
      CeresEnvironment.Initialize(loggerTypes, logLevel);

      CeresEnvironment.MONITORING_METRICS = !CommandLineWorkerSpecification.IsWorker
                                           && CeresUserSettingsManager.Settings.LaunchMonitor;

      //      if (CeresUserSettingsManager.Settings.DirLC0Networks != null)
      //        NNWeightsFilesLC0.RegisterDirectory(CeresUserSettingsManager.Settings.DirLC0Networks);

      MCTSEngineInitialization.BaseInitialize(CeresEnvironment.MONITORING_METRICS, 1);


      Console.WriteLine();

      //Features.BatchAnalysis.BatchAnalyzer.Test();      return;

      if (args.Length > 0 && (args[0].ToUpper() == "CUSTOM" || args[0].StartsWith("WORKER")))
      {
        TournamentTest.Test();
        //TournamentTest.TestSFLeela(0, true); return;
        //        SuiteTest.RunSuiteTest(); return;
      }

#if DEBUG
      CheckDebugAllowed();
#endif

      StringBuilder allArgs = new StringBuilder();
      for (int i = 0; i < args.Length; i++)
      {
        allArgs.Append(args[i] + " ");
      }

      string allArgsString = allArgs.ToString();

      DispatchCommands.ProcessCommand(allArgsString);


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
| Ceres - A Monte Carlo Tree Search Chess Engine          |
|                                                         |
| (c) 2020- David Elliott and the Ceres Authors           |
|   With network backend code from Leela Chess Zero.      |
|   Use help to list available commands.                  |
|                                                         |
|  Version {VER}                                       |
|  Runtime {VER}                                       |
|=========================================================|
";

    static void OutputBanner()
    {
      string dotnetVersion = RuntimeInformation.FrameworkDescription;
      CudaDeviceProperties deviceProperties = CUDADevice.GetContext(0).Context.GetDeviceInfo();
      string cudaVersion = $"{deviceProperties.DriverVersion.Major}.{deviceProperties.DriverVersion.Minor}";
      bool isDotNet5 = dotnetVersion.Contains(".NET 5");
      bool isPGO = !isDotNet5 && !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTNET_TieredPGO"));
      string pgoString = isPGO ? "PGO" : "NA";

      string[] bannerLines = BannerString.Split(Environment.NewLine);
      foreach (string line in bannerLines)
      {
        if (line.StartsWith("| Ceres"))
        {
          ConsoleColor defaultColor = Console.ForegroundColor;
          Console.Write("|");
          Console.ForegroundColor = ConsoleColor.Magenta;
          Console.Write(line.Substring(1, line.Length - 2));
          Console.ForegroundColor = defaultColor;
          Console.WriteLine("|");
        }

        else if (line.StartsWith("|  Version"))
        {
          string version = $"|  Version {CeresVersion.VersionString} with PGO: {pgoString}";
          int spaceLeft = line.Length - version.Length;
          string empty = new string(' ', 3 + spaceLeft - 1);
          Console.WriteLine($"{version}{empty}|");
        }
        else if (line.StartsWith("|  Runtime"))
        {
          string runtime = $"|  Runtime {dotnetVersion} and Cuda {cudaVersion}";
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
