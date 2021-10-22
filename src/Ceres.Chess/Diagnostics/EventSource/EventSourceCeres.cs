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
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;

#endregion

namespace Ceres.Chess.Diagnostics
{
  /// <summary>
  /// Interfaces with .NET event logging mechanism.
  /// Usage:
  ///   -- at startup, call LaunchConsoleMonitor to begin monitoring
  ///   -- to add a counter:
  ///        - define a field myEvent of type EventCounter
  ///        - initialize as myEvent = new EventCounter("event name", this) where this is the EventSourceCeres
  ///        - call like myEvent.WriteMetric(123);

  /// 
  /// See: https://github.com/microsoft/dotnet-samples/blob/master/Microsoft.Diagnostics.Tracing/EventSource/docs/EventSource.md
  /// INSTALL/UPDATE:
  ///   dotnet tool uninstall --global dotnet-counters
  ///   dotnet tool install --global dotnet-counters 
  ///   dotnet tool update --global dotnet-counters 
  /// COMMAND LINE MONITOR
  ///   dotnet-counters monitor --process-id 13404 Ceres System.Runtime
  /// PERFVIEW MONITOR  
  ///   \t4\PerfView /onlyProviders= *Ceres*:Informational:EventCounterIntervalSec= 1 collect
  /// 
  /// </summary>
  [EventSource(Name = "Ceres")]
  public sealed class EventSourceCeres : EventSource
  {
    private static Dictionary<string, int> events = new Dictionary<string, int>();
    private static Dictionary<string, EventCounter> counters = new Dictionary<string, EventCounter>();

    public static EventSourceCeres Log = new EventSourceCeres();

    // TODO: remove
    //       This is old style logging, inefficient, not thread-safe
    public static bool ENABLED = false; // !System.Diagnostics.Debugger.IsAttached; // annoying Exceptions if under debugger

    EventSourceCeres()
    {
    }

    public static void LaunchConsoleMonitor(string extraSourceNames = "")
    {
      if (ENABLED)
      {
        int processID = Process.GetCurrentProcess().Id;
        ProcessStartInfo ps = new ProcessStartInfo()
        {
          FileName = "dotnet-counters",
          Arguments = $"monitor --process-id {processID} Ceres System.Runtime " + extraSourceNames,
          UseShellExecute = true,
          WindowStyle  = ProcessWindowStyle.Maximized           
        };
        Process.Start(ps);
      }
    }

    static int nextEventNum = 0;

    public void WriteEvent(string eventID, string description)
    {
      if (ENABLED)
      {
        if (!events.TryGetValue(eventID, out int eventIDNum))
        {
          eventIDNum = nextEventNum++;
          events.Add(eventID, eventIDNum);
        }

        base.WriteEvent(eventIDNum, description);
      }
    }


    public void WriteMetric(string metricID, float value)
    {
      if (ENABLED)
      {
        if (!counters.TryGetValue(metricID, out EventCounter thisCounter))
        {
          thisCounter = new EventCounter(metricID, this) { DisplayName = metricID };
          counters.Add(metricID, thisCounter);
        }
        thisCounter.WriteMetric(value);
      }
    }

  }
}
