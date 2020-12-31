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
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Microsoft.Extensions.Logging;

#endregion

namespace Ceres.Base.OperatingSystem.Windows
{
  /// <summary>
  /// A logger that writes messages to the Windows OutputDebugString API.
  /// </summary>
  [SupportedOSPlatform("windows")]
  public class WinDebugLogger : ILogger
  {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, BestFitMapping = true)]
    public static extern void OutputDebugString(String message);


    private readonly Func<string, LogLevel, bool> filter;
    private readonly string loggerID;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name">The name of the logger.</param>
    public WinDebugLogger(string name) : this(name, filter: null)
    {
    }

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name">Identifying name of the logger.</param>
    /// <param name="filter">Filtering function.</param>
    public WinDebugLogger(string name, Func<string, LogLevel, bool> filter)
    {
      loggerID = string.IsNullOrEmpty(name) ? nameof(WinDebugLogger) : name;
      this.filter = filter;
    }

    public bool IsEnabled(LogLevel logLevel)
    {
      return logLevel != LogLevel.None 
          && (filter == null || filter(loggerID, logLevel));
    }

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
    {
      if (!IsEnabled(logLevel)) return;      
      if (formatter == null) throw new ArgumentNullException(nameof(formatter));

      string debugMessage = formatter(state, exception);
      if (string.IsNullOrEmpty(debugMessage))  return;
      
      string message = $"{ logLevel }: {debugMessage}";

      if (exception != null)      
        message += System.Environment.NewLine + exception.ToString();

      OutputDebugString(loggerID + " : " + message);
    }

    public IDisposable BeginScope<TState>(TState state)
    {
      return new NullDisposable(); 
    }

    private class NullDisposable : IDisposable
    {
      public void Dispose()
      {
      }
    }
  }
}