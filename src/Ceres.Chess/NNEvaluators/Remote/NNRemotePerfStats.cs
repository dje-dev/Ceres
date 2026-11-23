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
using System.Text;

#endregion

namespace Ceres.Chess.NNEvaluators.Remote
{
  /// <summary>
  /// Optional lightweight per-phase timing of the remote evaluation path
  /// (client, server and balanced split), for performance diagnosis.
  ///
  /// Enabled by setting the environment variable CERES_REMOTE_PERF=1 (off by default, in which case
  /// no instances are created and the cost is a null check per batch). Every CERES_REMOTE_PERF_SEC
  /// seconds (default 5) each instance writes one line to stderr with the mean time per batch spent
  /// in each phase, the bytes on the wire and the garbage collections in that window.
  ///
  /// An instance is used by a single thread (one per connection or evaluator), so no locking is needed.
  /// </summary>
  public sealed class NNRemotePerfStats
  {
    /// <summary>
    /// If performance statistics collection was requested.
    /// </summary>
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("CERES_REMOTE_PERF") is "1" or "true";

    static readonly double IntervalSec =
      double.TryParse(Environment.GetEnvironmentVariable("CERES_REMOTE_PERF_SEC"), out double s) && s > 0 ? s : 5;

    readonly string name;
    readonly string[] phaseNames;
    readonly long[] phaseTicks;

    /// <summary>
    /// Optional additional text appended to each statistics line (evaluated when the line is written).
    /// </summary>
    public Func<string> ExtraInfo;

    long batches;
    long positions;
    long rawBytesOut, wireBytesOut, rawBytesIn, wireBytesIn;
    long windowStart;
    int gc0, gc2;
    TimeSpan gcPause;


    /// <summary>
    /// Returns a new instance if statistics are enabled, otherwise null.
    /// </summary>
    public static NNRemotePerfStats CreateIfEnabled(string name, params string[] phaseNames)
      => Enabled ? new NNRemotePerfStats(name, phaseNames) : null;


    NNRemotePerfStats(string name, string[] phaseNames)
    {
      this.name = name;
      this.phaseNames = phaseNames;
      phaseTicks = new long[phaseNames.Length];
      ResetWindow();
    }


    void ResetWindow()
    {
      Array.Clear(phaseTicks);
      batches = positions = rawBytesOut = wireBytesOut = rawBytesIn = wireBytesIn = 0;
      windowStart = Stopwatch.GetTimestamp();
      gc0 = GC.CollectionCount(0);
      gc2 = GC.CollectionCount(2);
      gcPause = GC.GetTotalPauseDuration();
    }


    /// <summary>
    /// Adds elapsed Stopwatch ticks to a phase.
    /// </summary>
    public void Add(int phase, long ticks) => phaseTicks[phase] += ticks;


    /// <summary>
    /// Records the payload sizes (before and after compression) of one sent and one received message.
    /// </summary>
    public void AddBytes(int rawOut, int wireOut, int rawIn, int wireIn)
    {
      rawBytesOut += rawOut;
      wireBytesOut += wireOut;
      rawBytesIn += rawIn;
      wireBytesIn += wireIn;
    }


    /// <summary>
    /// Marks the end of one batch, writing the statistics line if the window has elapsed.
    /// </summary>
    public void EndBatch(int numPositions)
    {
      batches++;
      positions += numPositions;
      if (Stopwatch.GetElapsedTime(windowStart).TotalSeconds >= IntervalSec)
      {
        Dump();
      }
    }


    /// <summary>
    /// Writes the statistics line for the current window (if any batches were seen) and starts a new window.
    /// </summary>
    public void Dump()
    {
      if (batches > 0)
      {
        double sec = Stopwatch.GetElapsedTime(windowStart).TotalSeconds;
        double usPerTick = 1_000_000.0 / Stopwatch.Frequency;
        StringBuilder sb = new();
        sb.Append($"[PERF] {name}: {sec:F1}s {batches} batches ({batches / sec:F0}/s) avgN {(double)positions / batches:F0} {positions / sec:N0} pos/s | us/batch:");
        for (int i = 0; i < phaseNames.Length; i++)
        {
          double us = phaseTicks[i] * usPerTick / batches;
          double pct = 100.0 * phaseTicks[i] / Stopwatch.Frequency / sec;
          sb.Append($" {phaseNames[i]} {us:F0} ({pct:F0}%)");
        }
        if (rawBytesOut + rawBytesIn > 0)
        {
          sb.Append($" | B/pos out {(double)wireBytesOut / positions:F0} (raw {(double)rawBytesOut / positions:F0})"
                  + $" in {(double)wireBytesIn / positions:F0} (raw {(double)rawBytesIn / positions:F0})");
        }
        sb.Append($" | GC g0 {GC.CollectionCount(0) - gc0} g2 {GC.CollectionCount(2) - gc2} pause {(GC.GetTotalPauseDuration() - gcPause).TotalMilliseconds:F0}ms");
        if (ExtraInfo != null)
        {
          sb.Append(" | ").Append(ExtraInfo());
        }
        Console.Error.WriteLine(sb.ToString());
      }
      ResetWindow();
    }
  }
}
