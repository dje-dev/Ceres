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
using System.Diagnostics;

#endregion

namespace Ceres.Base.Benchmarking
{
  public class TimingBlock : IDisposable
  {
    public enum LoggingType { None, Console, ConsoleWithMemoryTracking };

    #region Internal State

    /// <summary>
    /// If the TC is enabled (otherwise will not track statistics)
    /// </summary>
    public readonly bool Enabled;

    /// <summary>
    /// String description of task being timed
    /// </summary>
    public readonly string Description;

    /// <summary>
    /// Target location for outputs messages
    /// </summary>
    public readonly LoggingType Target = LoggingType.Console;

    Stopwatch timingStopwatch = new Stopwatch();
    TimingStats outStats = null;

    TimeSpan timeStart;
    long memUsageStart;
    static Process process = Process.GetCurrentProcess();

    #endregion

    #region Constructors

    void Startup()
    {
      if (Enabled)
      {
        const bool DO_GARAGE_COLLECTIONS = false;
        memUsageStart = Target == LoggingType.ConsoleWithMemoryTracking ? System.GC.GetTotalMemory(DO_GARAGE_COLLECTIONS) : 0;

        timeStart = process.TotalProcessorTime;
        timingStopwatch.Start();
      }
    }

    /// <summary>
    /// Constructor.
    /// </summary>
    public TimingBlock(bool enabled = true)
    {
      Enabled = enabled;
      Startup();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="outStats"></param>
    /// <param name="target"></param>
    /// <param name="enabled"></param>
    public TimingBlock(TimingStats outStats, TimingBlock.LoggingType target = LoggingType.Console, bool enabled = true)
      : this((string)null, outStats, target, enabled)
    {

    }

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="description"></param>
    /// <param name="outStats"></param>
    /// <param name="target"></param>
    /// <param name="enabled"></param>
    public TimingBlock(string description, TimingStats outStats, TimingBlock.LoggingType target = LoggingType.Console, bool enabled = true)
    {
      Description = description;
      this.outStats = outStats;
      Enabled = enabled;
      Target = target;
      Startup();
    }

    // --------------------------------------------------------------------------------------------
    /// <summary>
    /// Constructor (with description and target).
    /// </summary>
    /// <param name="description"></param>
    /// <param name="target">where to send logging output messages</param>
    public TimingBlock(string description, LoggingType target = LoggingType.Console, bool enabled = true) : this(enabled)
    {
      Description = description;
      Target = target;
      Startup();
    }


    #endregion

    #region ShowStats/Dispose

    /// <summary>
    /// Summary string
    /// </summary>
    public string StatsMessage(long memUsage, double elapsedTimeSecs, double cpuTimeSecs)
    {
      const double MILLION = 1024 * 1024;
      const double BILLION = 1024 * MILLION;
      string ticksStr = (elapsedTimeSecs > 0.1 ? "" : " (" + timingStopwatch.ElapsedTicks + "),");


      double timeValue = elapsedTimeSecs > 0.01 ? elapsedTimeSecs : elapsedTimeSecs * 1000.0f;
      string timeValueUnits = elapsedTimeSecs > 0.01 ? "sec" : "msec";

      long gcMemory = (Target != LoggingType.ConsoleWithMemoryTracking) ? 0 : GC.GetTotalMemory(false);// too expensive to do if not logging
      return $"{timeValue,6:F3} {timeValueUnits} TC: {Description } {ticksStr} mdiff:{memUsage / MILLION,6:F2} memTot:{gcMemory / MILLION,6:F2} "
        + $" CPU:{cpuTimeSecs,6:F2} secs, GC {GC.CollectionCount(0)} {GC.CollectionCount(1)} {GC.CollectionCount(2)} "
        + $" WS:{Process.GetCurrentProcess().WorkingSet64 / BILLION,6:F2} WSPeak: {Process.GetCurrentProcess().PeakWorkingSet64/BILLION,6:F2}";
    }

    /// <summary>
    /// Dispose method, triggering output of statistcs.
    /// </summary>
    public void Dispose()
    {
      if (Enabled)
      {
        timingStopwatch.Stop();

        // Collect stats
        double cpuTimeSec = (process.TotalProcessorTime - timeStart).TotalSeconds;
        const bool DO_GARAGE_COLLECTIONS = true;
        long memUsage = Target == LoggingType.ConsoleWithMemoryTracking ? System.GC.GetTotalMemory(DO_GARAGE_COLLECTIONS) - memUsageStart : 0;

        double elapsedTimeSecs = (double)timingStopwatch.ElapsedTicks / (double)Stopwatch.Frequency;

        if (Target != LoggingType.None)
          Console.WriteLine(StatsMessage(memUsage, elapsedTimeSecs, cpuTimeSec));

        // Possibly return stats
        if (outStats != null)
        {
          outStats.ElapsedTimeSecs = elapsedTimeSecs;
          outStats.ElapsedTimeTicks = timingStopwatch.ElapsedTicks;
          outStats.MemUsedBytes = memUsage;
          outStats.CPUTimeSecs = cpuTimeSec;
        }
      }
    }

    #endregion

    #region Static helpers

    public static void RunActionTimingTest(string description, int numTests, int numPerTest, Action action)
    {
      for (int test = 0; test < numTests; test++)
      {
        using (new TimingBlock(description))
        {
          for (int i = 0; i < numPerTest; i++)
            action();
        }
      }

    }


    #endregion

  }
}



