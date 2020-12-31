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
        Console.WriteLine($"Prompting to for required values to initialize:");
        CeresUserSettingsManager.DoSetupInitialize();
      }

      // Configure logging level
      const bool LOG = false;
      CeresEnvironment.MONITORING_EVENTS = LOG;
      LogLevel logLevel = LOG ? LogLevel.Information : LogLevel.Critical;
      LoggerTypes loggerTypes = LoggerTypes.WinDebugLogger | LoggerTypes.ConsoleLogger;
      CeresEnvironment.Initialize(loggerTypes, logLevel);

      CeresEnvironment.MONITORING_METRICS = CeresUserSettingsManager.Settings.LaunchMonitor;

//      if (CeresUserSettingsManager.Settings.DirLC0Networks != null)
//        NNWeightsFilesLC0.RegisterDirectory(CeresUserSettingsManager.Settings.DirLC0Networks);

      MCTSEngineInitialization.BaseInitialize();

      if (args.Length > 0 && args[0].ToUpper() == "CUSTOM")
      {
        TournamentTest.Test(); return;
//        SuiteTest.RunSuiteTest(); return;
      }

      StringBuilder allArgs = new StringBuilder();
      for (int i = 0; i < args.Length; i++)
        allArgs.Append(args[i] + " ");
      string allArgsString = allArgs.ToString();

      DispatchCommands.ProcessCommand(allArgsString);


      //  Win32.WriteCrashdumpFile(@"d:\temp\dump.dmp");
   }

    const string BannerString =
@"
|=====================================================|
| Ceres - A Monte Carlo Tree Search Chess Engine      |
|                                                     |
| (c) 2020- David Elliott and the Ceres Authors       |
|   With network backend code from Leela Chess Zero.  |
|                                                     |
|  Version 0.80. Use help to list available commands. |
|  {git}
|=====================================================|
";

    static void OutputBanner()
    {
      string[] bannerLines = BannerString.Split("\r\n");
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
        else
        {
          // DISABLED git functionality until crash on a user's computer is better understood
          if (false && line.Contains("{git}"))
          {
            string outString = line.Replace("{git}", GitInfo.VersionString).TrimEnd();
            outString = outString.PadRight(bannerLines[1].Length - 1);
            outString = outString + "|";
            Console.WriteLine(outString);
          }
          else
          {
            Console.WriteLine(line);
          }
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
