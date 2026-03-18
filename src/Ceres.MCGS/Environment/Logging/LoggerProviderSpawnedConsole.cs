#region Using directives

using System;
using System.IO;
using System.IO.Pipes;
using Microsoft.Extensions.Logging;


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

namespace Ceres.MCGS.Environment;

public sealed class LoggerProviderSpawnedConsole : ILoggerProvider
{
  private readonly StreamWriter _writer;
  private bool echoInline;

  public LoggerProviderSpawnedConsole(string pipeName, bool echoInline)
  {
    this.echoInline = echoInline;

    NamedPipeClientStream pipe = new NamedPipeClientStream(".", pipeName, PipeDirection.Out);
    Console.WriteLine("Connecting to " + pipeName);
    try
    {
      pipe.Connect(2000);
    }
    catch (Exception)
    {
      Console.WriteLine("Timeout, unable to connect.");
      System.Environment.Exit(3);
    }

    _writer = new StreamWriter(pipe)
    {
      AutoFlush = true
    };
  }

  public ILogger CreateLogger(string categoryName)
  {
    return new LoggerSpawnedConsole(categoryName, _writer, echoInline);
  }

  public void Dispose()
  {
    _writer?.Dispose();
  }
}
