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
using System.Runtime.Versioning;
using Microsoft.Extensions.Logging;

#endregion

namespace Ceres.Base.OperatingSystem.Windows
{
  /// <summary>
  /// Provider for a logger that outputs messages to the OutputDebugString API.
  /// </summary>
  [SupportedOSPlatform("windows")]
  public class WinDebugLoggerProvider : ILoggerProvider, IDisposable
  {
    public WinDebugLoggerProvider()
    {
    }
    public ILogger CreateLogger(string name) => new WinDebugLogger(name);

    public void Dispose()
    { 
    }

  }
}