#region Using directives

using System.IO;
using System.IO.Pipes;
using Ceres.Base.Misc;


#endregion

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
using System.Threading;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.MCGS.Environment;

public class MCGSLaunch
{
  /// <summary>
  /// Bootstrap method to launch the application, either in regular mode or logger mode
  /// (based on presence of LOGGER keyword as first command line argument).
  /// </summary>
  /// <param name="args"></param>
  public static void Launch(string[] args)
  {
    string ceresJsonPath = System.Environment.GetEnvironmentVariable("CERES_JSON");
    if (ceresJsonPath is not null)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Loading Ceres settings from CERES_JSON environment variable: {ceresJsonPath}");
      CeresUserSettingsManager.LoadFromFile(ceresJsonPath);
    }
    if (args?.Length == 2 && args[0] == "LOGGER")
    {
      Console.WriteLine("Launch as logger");
      RunInLoggerMode(args[1]);
      return;
    }
    else
    {
      Console.WriteLine("Launch regular");
    }
  }


  static void RunInLoggerMode(string pid)
  {
    MCGSEnvironment.RUNNING_AS_LOGGER = true;

    string pipeName = $"{pid}";
    string mutexName = $"Global\\CeresLoggerInstance_{pid}";

    bool createdNew;

    using Mutex singleInstanceMutex = new Mutex(initiallyOwned: true, name: mutexName, createdNew: out createdNew);

    if (!createdNew)
    {
      Console.WriteLine($"[LOGGER] Logger already running for PID {pid}. Aborting.");
      System.Environment.Exit(4); // distinct exit code for "already running"
    }

    Console.Title = $"Log Window for PID {pid}";

    using NamedPipeServerStream pipe = new NamedPipeServerStream(pipeName, PipeDirection.In);
    using StreamReader reader = new StreamReader(pipe);

    Console.WriteLine($"[LOGGER] Waiting for connection on pipe: {pipeName}");
    pipe.WaitForConnection();
    Console.WriteLine("[LOGGER] Connected!");

    string line;
    while ((line = reader.ReadLine()) != null)
    {
      ConsoleColor consoleColor = line.Contains("[Error]") ? ConsoleColor.Red :
                           line.Contains("[Warning]") ? ConsoleColor.Yellow :
                           line.Contains("[Debug]") ? ConsoleColor.Gray :
                           line.Contains("[Information]") ? ConsoleColor.Yellow :
                           ConsoleColor.White;
      ConsoleUtils.WriteLineColored(consoleColor, line, endLine: true);
    }

    Console.WriteLine("[LOGGER] Pipe closed. Exiting.");
    System.Environment.Exit(0);
  }
}
