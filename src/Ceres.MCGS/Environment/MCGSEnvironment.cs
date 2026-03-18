#region Using directives

using System.Diagnostics;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.Json;
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

#region Using directives

using System;
using System.Threading;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Environment;

public static class MCGSEnvironment
{
  public static bool RUNNING_AS_LOGGER = false;

  private static readonly Lazy<ILoggerFactory> lazyFactory = new(() =>
  {
    IConfiguration config = new ConfigurationBuilder()
        .AddJsonFile("appsettings.json", optional: false, reloadOnChange: false)
        .Build();

    bool enableSecondary = config.GetValue<bool>("Logging:SecondaryConsole:Enabled");
    int delay = config.GetValue<int>("Logging:SecondaryConsole:StartupDelayMs", 500);

    string pipeName = $"LogPipe_{Process.GetCurrentProcess().Id}";

    if (enableSecondary)
    {
      if (RUNNING_AS_LOGGER)
      {
        throw new Exception("Application running in logger mode attempted to span another logger. Abort.");
      }
      string exePath = System.Environment.ProcessPath
                       ?? Process.GetCurrentProcess().MainModule?.FileName
                       ?? throw new InvalidOperationException("Cannot locate process path.");

      ProcessStartInfo startInfo = new ProcessStartInfo
      {
        FileName = exePath,
        Arguments = $"LOGGER {pipeName}",
        UseShellExecute = true,
        WindowStyle = ProcessWindowStyle.Minimized
      };

      Process.Start(startInfo);

      Thread.Sleep(delay);
    }

    return LoggerFactory.Create(builder =>
    {
      if (enableSecondary)
      {
        builder.AddProvider(new LoggerProviderSpawnedConsole(pipeName, MCGSParamsFixed.LOG_ECHO_INLINE));
      }
      else
      {
        builder.AddConsole();
      }
    });
  });


  public static ILoggerFactory Factory => lazyFactory.Value;


  public static ILogger<T> CreateLogger<T>()
  {
    return Factory.CreateLogger<T>();
  }
}

