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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Ceres.Base.OperatingSystem.Windows;

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Debug;

#endregion

namespace Ceres.Base.Environment
{
  [Flags]
  public enum LoggerTypes
  {
    ConsoleLogger,
    DebugLogger,
    WinDebugLogger,
    EventSourceLogger,
    EventLog
  }

  public static class CeresEnvironment
  {
    public static bool MONITORING_METRICS = true; // overhead seems very low (about 2%)
    public static bool MONITORING_EVENTS = false;  // large performance impact (circa 50%)

    /// <summary>
    /// Flag used in testing to indicate current thread is running in "test mode."
    /// </summary>
    [ThreadStatic]
    public static bool TEST_MODE;

    static ILogger logger;
    static LoggerTypes loggers;

    public static ILogger Logger
    {
      get
      {
        if (logger == null) throw new Exception("CeresEnviroment.Initialize was not called");
        return logger;
      }
    }

    public static void Initialize(LoggerTypes loggers, LogLevel loggingLevel)
    {
      CeresEnvironment.loggers = loggers;
      CreateLogger(loggers, loggingLevel);
    }


    static string FmtMessage(string category, string id, int instanceID, string message, 
                             string file = null, string member = null)
    {
      string fileMemberString = $" at {file} {member}";
      if (instanceID == -1)
        return $"[{category}:{id}]: {message} {fileMemberString}\r\n";
      else
        return $"[{category}:{id}:{instanceID}]: {message} {fileMemberString}\r\n";
    }

    static string ExceptionMessage(Exception exception)
    {
      StringBuilder sb = new StringBuilder();
      int depth = 0;

      Exception curException = exception.InnerException;
      while (curException != null)
      {
        sb.Append($"Exception[depth {depth}] '{curException.Message}' at: \r\n{curException.StackTrace.ToString()}");
        curException = curException.InnerException;
        depth++;
      }

      return sb.ToString();
    }

    static readonly object logLockObj = new();

    // Can use DebugView utility from Microsoft to monitor output from OutputDebugString

    
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, BestFitMapping = true)]
    public static extern void OutputDebugString(String message);

    public static void LogInfo(string category, string id, string message, int instanceID = -1,
                              [CallerFilePath] string file = "",
                              [CallerMemberName] string member = "")
    {
      if (CeresEnvironment.MONITORING_EVENTS)
      {
        lock (logLockObj)
        {
          if (loggers.HasFlag(LoggerTypes.WinDebugLogger))
          {
            OutputDebugString(FmtMessage(category, id, instanceID, message));
          }
          
          logger.LogInformation(FmtMessage(category, id, instanceID, message));
        }
      }
    }

    public static void LogWarn(string category, string id, string message, int instanceID = -1) 
    { 
      if (CeresEnvironment.MONITORING_EVENTS )
        logger.LogWarning(FmtMessage(category, id, instanceID, message));
    }

    public static void LogError(string category, string id, string message, int instanceID = -1)
    {
      // We always log these to Console
      ConsoleColor saveColor = Console.ForegroundColor;
      Console.ForegroundColor = ConsoleColor.Yellow;
      Console.Write($"[ERROR] ");
      Console.ForegroundColor = saveColor;
      Console.WriteLine($" {DateTime.Now}   {category}:{id} {message} {(instanceID == -1 ? "" : instanceID)}");

      if (CeresEnvironment.MONITORING_EVENTS)
        logger.LogError(FmtMessage(category, id, instanceID, message));
    }


    public static void LogCritical(string category, string id, string message, int instanceID = -1)
    {
      // We always log these to Console
      ConsoleColor saveColor = Console.ForegroundColor;
      Console.ForegroundColor = ConsoleColor.Red;
      Console.Write($"[CRITICAL] ");
      Console.ForegroundColor = saveColor;
      Console.WriteLine($"{DateTime.Now}   {category}:{id} {message} {(instanceID==-1 ?"":instanceID)}");

      if (CeresEnvironment.MONITORING_EVENTS)
        logger.LogCritical(FmtMessage(category, id, instanceID, message));
    }


    static void CreateLogger(LoggerTypes loggerTypes, LogLevel minLogLevel)
    {
      ILoggerFactory loggerFactory = LoggerFactory.Create(builder =>
      {
        builder = builder.AddFilter((LogLevel l) => l >= minLogLevel);

        if (loggerTypes.HasFlag(LoggerTypes.ConsoleLogger))
        {
          builder.AddConsole
          (
            cLogger =>
            {
              cLogger.LogToStandardErrorThreshold = minLogLevel;
              cLogger.IncludeScopes = false;
            });
        }

        if (loggerTypes.HasFlag(LoggerTypes.ConsoleLogger))
        {
          builder.AddEventSourceLogger();
        }

        if (loggerTypes.HasFlag(LoggerTypes.DebugLogger))
        {
          builder.AddProvider(new DebugLoggerProvider());
        }

        if (loggerTypes.HasFlag(LoggerTypes.EventSourceLogger))
        {
          builder .AddEventSourceLogger();
        }

        if (loggerTypes.HasFlag(LoggerTypes.EventLog))
        {
          throw new NotImplementedException("LoggerTypes.EventLog not yet supported");
        }
      });
      logger = loggerFactory.CreateLogger<CeresApp>();

      
      logger.LogInformation("CeresEnvironment initialized");
    }

  }

  internal class CeresApp
  {

  }
}


