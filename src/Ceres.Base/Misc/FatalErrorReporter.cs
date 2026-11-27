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
using System.Threading;

#endregion

namespace Ceres.Base.Misc;

/// <summary>
/// Last-resort reporting of fatal errors (exceptions which are about to terminate the process).
///
/// When the engine is driven by a GUI or tournament manager typically only standard output is
/// captured, whereas the .NET runtime reports an unhandled exception only on standard error.
/// An exception on a thread which has no handler of its own (for example a search worker thread)
/// would therefore make the process disappear without leaving any usable diagnostic.
///
/// The methods here write the full exception text to standard output, to standard error,
/// and append it to a crash log file in the current directory (falling back to the temporary directory).
///
/// The handler installed by InstallUnhandledExceptionHandler only reports: the runtime still
/// terminates the process in its usual way afterwards (abort, with a crash dump if dumps are enabled).
/// </summary>
public static class FatalErrorReporter
{
  /// <summary>
  /// Name of the file (in the current working directory) to which fatal errors are appended.
  /// </summary>
  public const string CRASH_LOG_FILE_NAME = "Ceres_crash.log";

  /// <summary>
  /// Maximum number of reports emitted by a process (several worker threads may fail at once).
  /// </summary>
  const int MAX_REPORTS = 4;

  static int numReports;
  static int handlerInstalled;
  static readonly Lock writeLock = new();


  /// <summary>
  /// Reports a fatal error to standard output (optionally), standard error and the crash log file.
  /// Never throws.
  /// </summary>
  /// <param name="context">short description of where the error was caught</param>
  /// <param name="exception">the exception (may be null)</param>
  /// <param name="writeToStandardOutput">if the text should also be written to standard output</param>
  public static void Report(string context, Exception exception, bool writeToStandardOutput = true)
    => DoReport("fatal error", context, exception, writeToStandardOutput);


  /// <summary>
  /// Reports (in the same way as Report) an error from which the engine was able to recover.
  /// Never throws.
  /// </summary>
  /// <param name="context">short description of where the error was caught and how it was handled</param>
  /// <param name="exception">the exception (may be null)</param>
  public static void ReportRecovered(string context, Exception exception)
    => DoReport("recovered error", context, exception, true);


  /// <summary>
  /// Reports (in the same way as Report) a condition from which the engine was able to recover,
  /// described by text rather than by an exception. Never throws.
  /// </summary>
  /// <param name="context">short description of what was detected and how it was handled</param>
  /// <param name="details">descriptive details</param>
  public static void ReportRecovered(string context, string details)
    => DoReport("recovered error", context, null, true, details);


  static void DoReport(string severity, string context, Exception exception, bool writeToStandardOutput, string details = null)
  {
    try
    {
      if (Interlocked.Increment(ref numReports) > MAX_REPORTS)
      {
        return;
      }

      string text = $"Ceres {severity} ({context}) at {DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} "
                  + $"on thread {System.Environment.CurrentManagedThreadId}:"
                  + System.Environment.NewLine
                  + (exception != null ? exception.ToString() : (details ?? "(no exception information available)"));

      lock (writeLock)
      {
        if (writeToStandardOutput)
        {
          try
          {
            Console.Out.WriteLine(text);
            Console.Out.Flush();
          }
          catch
          {
          }
        }

        try
        {
          Console.Error.WriteLine(text);
          Console.Error.Flush();
        }
        catch
        {
        }

        AppendToCrashLogFile(text);
      }
    }
    catch
    {
      // Reporting is best effort only.
    }
  }


  /// <summary>
  /// Reports a fatal error (see Report) and then terminates the process with the specified exit code.
  /// </summary>
  /// <param name="context"></param>
  /// <param name="exception"></param>
  /// <param name="exitCode"></param>
  public static void ReportAndExit(string context, Exception exception, int exitCode = 3)
  {
    Report(context, exception);
    System.Environment.Exit(exitCode);
  }


  /// <summary>
  /// Installs (once per process) a handler which reports any exception that is unhandled
  /// on any thread, before the runtime terminates the process.
  /// </summary>
  public static void InstallUnhandledExceptionHandler()
  {
    if (Interlocked.Exchange(ref handlerInstalled, 1) != 0)
    {
      return;
    }

    AppDomain.CurrentDomain.UnhandledException += (_, args) =>
    {
      Report("unhandled exception", args.ExceptionObject as Exception);
    };
  }


  /// <summary>
  /// Appends the specified text to the crash log file. Never throws.
  /// </summary>
  /// <param name="text"></param>
  static void AppendToCrashLogFile(string text)
  {
    string entry = text + System.Environment.NewLine + System.Environment.NewLine;

    try
    {
      File.AppendAllText(Path.Combine(System.Environment.CurrentDirectory, CRASH_LOG_FILE_NAME), entry);
      return;
    }
    catch
    {
    }

    try
    {
      File.AppendAllText(Path.Combine(Path.GetTempPath(), CRASH_LOG_FILE_NAME), entry);
    }
    catch
    {
    }
  }
}
