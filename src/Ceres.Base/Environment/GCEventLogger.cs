#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Globalization;

#endregion

namespace Ceres.Base.Environment;

/// <summary>
/// Provides an event listener that logs .NET garbage collection (GC) events, 
/// including GC start, end, pause durations, and heap statistics, 
/// to the console for diagnostic and performance analysis purposes.
/// </summary>
/// <remarks>
/// GCEventLogger attaches to the .NET runtime event source and outputs a formatted summary
/// of each GC occurrence, including generation, reason, type, pause time, and memory statistics. 
/// Optionally, it can track allocation ticks between GCs if enabled via the constructor. 
/// </remarks>
public sealed class GCEventLogger : EventListener, IDisposable
{
  private const long MEGABYTE = 1024 * 1024;

  private readonly object eventLock = new object();
  private readonly Dictionary<uint, GcInfo> inflight = new Dictionary<uint, GcInfo>();
  private readonly bool trackAllocTicks;

  // We compute a "pause % since previous GC" if you want it later.
  private double lastGcEndTimeMs = 0.0;
  private double lastGcStartTimeMs = 0.0;


  /// <summary>
  /// Only used when _trackAllocTicks == true.
  /// </summary>
  private ulong allocSinceLastGCBytes = 0;

  /// <summary>
  /// GC count for which we are currently paused, for attributing RestartEEEnd.
  /// </summary>
  private uint lastSuspendCount = 0;

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="trackAllocationTicks"></param>
  public GCEventLogger(bool trackAllocationTicks = false)
  {
    trackAllocTicks = trackAllocationTicks;
    PrintHeader();
  }


  /// <summary>
  /// Called upon creation of the event source.
  /// </summary>
  /// <param name="eventSource"></param>
  protected override void OnEventSourceCreated(EventSource eventSource)
  {
    // This provider name is used cross-platform for EventPipe
    // (not actually Windows-specific).
    if (eventSource?.Name == "Microsoft-Windows-DotNETRuntime")
    {
      EventKeywords keywords = (EventKeywords)0x1; // GCKeyword
      EventLevel level = trackAllocTicks ? EventLevel.Verbose : EventLevel.Informational;

      // When tracking allocations we need Verbose to receive GCAllocationTick_V3 events.
      EnableEvents(eventSource, level, keywords);
    }
  }


  /// <summary>
  /// Event called for each EventSource event.
  /// </summary>
  /// <param name="e"></param>
  protected override void OnEventWritten(EventWrittenEventArgs e)
  {
    if (e.EventName == null)
    {
      return;
    }

    double nowMs = NowMs();

    switch (e.EventName)
    {
      case "GCStart_V2":
        {
          uint count = ReadU32(e, "Count");
          uint depth = ReadU32(e, "Depth");
          uint reason = ReadU32(e, "Reason");
          uint type = ReadU32(e, "Type");

          lock (eventLock)
          {
            GcInfo info = new GcInfo();
            info.Count = count;
            info.Generation = depth;
            info.Reason = reason;
            info.Type = type;
            info.StartTimeMs = nowMs;
            inflight[count] = info;

            // Reset between-GC allocation accumulator at start of a GC.
            if (trackAllocTicks)
            {
              allocSinceLastGCBytes = 0;
            }
          }
          break;
        }

      case "GCSuspendEEBegin_V1":
        {
          uint count = ReadU32(e, "Count");
          lock (eventLock)
          {
            if (inflight.TryGetValue(count, out GcInfo info))
            {
              info.PauseStartMs = nowMs;
              info.InPause = true;
              inflight[count] = info;
              lastSuspendCount = count;
            }
            else
            {
              // If we haven't seen GCStart yet (rare ordering), remember count anyway.
              lastSuspendCount = count;
            }
          }
          break;
        }

      case "GCRestartEEEnd_V1":
        {
          // No payload; attribute the pause to the last suspend’s GC count.
          lock (eventLock)
          {
            if (lastSuspendCount != 0 && inflight.TryGetValue(lastSuspendCount, out GcInfo info) && info.InPause)
            {
              double delta = System.Math.Max(0, nowMs - info.PauseStartMs);
              info.TotalPauseMs += delta;
              info.InPause = false;
              inflight[lastSuspendCount] = info;
            }
          }
          break;
        }

      case "GCHeapStats_V2":
        {
          // Heap stats are raised at the end of each GC; record promotions/survivors.
          uint count = ReadU32(e, "Count", defaultValue: 0); // Some runtimes omit Count here; tolerate missing.
          lock (eventLock)
          {
            // If Count isn't present, try to attach to the latest inflight GC.
            if (count == 0 && inflight.Count > 0)
            {
              foreach (KeyValuePair<uint, GcInfo> kv in inflight)
              {
                if (!kv.Value.HeapStatsSeen)
                {
                  count = kv.Key;
                  break;
                }
              }
            }

            if (count != 0 && inflight.TryGetValue(count, out GcInfo info))
            {
              info.Promoted0 = ReadU64Safe(e, "TotalPromotedSize0");
              info.Promoted1 = ReadU64Safe(e, "TotalPromotedSize1");
              info.SurvivedGen2 = ReadU64Safe(e, "TotalPromotedSize2");
              info.SurvivedLoh = ReadU64Safe(e, "TotalPromotedSize3");
              info.SurvivedPoh = ReadU64Safe(e, "TotalPromotedSize4");
              info.HeapStatsSeen = true;
              inflight[count] = info;
            }
          }
          break;
        }

      case "GCEnd_V1":
        {
          uint count = ReadU32(e, "Count");
          uint depth = ReadU32(e, "Depth");

          lock (eventLock)
          {
            if (inflight.TryGetValue(count, out GcInfo info))
            {
              info.EndTimeMs = nowMs;
              info.EndSeen = true;

              // Attach allocation since last GC (optional, off by default).
              if (trackAllocTicks)
              {
                info.AllocSinceLastGcBytes = allocSinceLastGCBytes;
              }

              // When both End and HeapStats are seen, we can print the line.
              if (info.HeapStatsSeen)
              {
                PrintRow(info);

                // Update “since previous GC” timestamp for optional % calculations.
                lastGcEndTimeMs = info.EndTimeMs;

                inflight.Remove(count);
              }
              else
              {
                inflight[count] = info; // wait for heap stats
              }
            }
          }
          break;
        }

      case "GCAllocationTick_V3":
        {
          if (trackAllocTicks)
          {
            // Approximate SOH allocations since last GC.
            uint allocKind = ReadU32(e, "AllocationKind");
            ulong amount64 = ReadU64Safe(e, "AllocationAmount64");
            if (allocKind == 0) // 0 == small object heap
            {
              lock (eventLock)
              {
                allocSinceLastGCBytes += amount64;
              }
            }
          }
          break;
        }
    }
  }

  /// <summary>
  /// Prints header line for GC log.
  /// </summary>
  private static void PrintHeader()
  {
    Console.WriteLine("GC#   Reason        Gen   Type         Pause(ms)   Pause%   Promoted(MB)   SurvivedGen2+LOH+POH(MB)   AllocSinceLastGC(MB)");
  }


  /// <summary>
  /// Prints row for a completed GC.
  /// </summary>
  /// <param name="i"></param>
  private void PrintRow(GcInfo i)
  {
    double promotedMb = (i.Promoted0 + i.Promoted1) / (double)MEGABYTE;
    double survivedBigMb = (i.SurvivedGen2 + i.SurvivedLoh + i.SurvivedPoh) / (double)MEGABYTE;

    string reason = ReasonToString(i.Reason);
    string type = TypeToString(i.Type);
    string gen = i.Generation.ToString(CultureInfo.InvariantCulture);

    string allocSince = trackAllocTicks
        ? (i.AllocSinceLastGcBytes / (double)MEGABYTE).ToString("N1", CultureInfo.InvariantCulture)
        : "-";

    // Calculate pause % since previous GC
    double pausePercent = 0.0;
    if (lastGcStartTimeMs > 0.0)
    {
      double timeSincePrevGc = i.StartTimeMs - lastGcStartTimeMs;
      if (timeSincePrevGc > 0)
      {
        pausePercent = (i.TotalPauseMs / timeSincePrevGc) * 100.0;
      }
    }
    string pausePct = pausePercent > 0 ? pausePercent.ToString("N2", CultureInfo.InvariantCulture) : "-";

    Console.WriteLine(
        $"{i.Count,4}  {reason,-12}  {gen,3}   {type,-12}  {i.TotalPauseMs,9:N2}   {pausePct,6}   {promotedMb,12:N2}   {survivedBigMb,24:N2}   {allocSince,20}");

    // Update for next GC
    lastGcStartTimeMs = i.StartTimeMs;
  }


  /// <summary>
  /// Converts GC type to string.
  /// </summary>
  /// <param name="type"></param>
  /// <returns></returns>
  private static string TypeToString(uint type)
  {
    // 0 = blocking outside BGC, 1 = Background GC, 2 = blocking during BGC
    return type switch
    {
      0 => "Blocking",
      1 => "Background",
      2 => "Block(whileBGC)",
      _ => type.ToString(CultureInfo.InvariantCulture)
    };
  }


  /// <summary>
  /// Converts GC reason to string.
  /// </summary>
  /// <param name="reason"></param>
  /// <returns></returns>
  private static string ReasonToString(uint reason)
  {
    // From official docs for GCStart_V2.Reason
    // 0: AllocSmall, 1: Induced, 2: LowMemory, 3: Empty,
    // 4: AllocLarge, 5: OutOfSpaceSOH, 6: OutOfSpaceLOH, 7: InducedNotBlocking
    return reason switch
    {
      0 => "AllocSmall",
      1 => "Induced",
      2 => "LowMemory",
      3 => "Empty",
      4 => "AllocLarge",
      5 => "OOS SOH",
      6 => "OOS LOH",
      7 => "InducedNB",
      _ => reason.ToString(CultureInfo.InvariantCulture)
    };
  }

  /// <summary>
  /// Returns current time in milliseconds.
  /// </summary>
  /// <returns></returns>
  private static double NowMs()
  {
    // High-resolution monotonic time
    long ticks = Stopwatch.GetTimestamp();
    return ticks * 1000.0 / Stopwatch.Frequency;
  }

  /// <summary>
  /// Reads a uint32 payload value from an event, with optional default.
  /// </summary>
  /// <param name="e"></param>
  /// <param name="name"></param>
  /// <param name="defaultValue"></param>
  /// <returns></returns>
  private static uint ReadU32(EventWrittenEventArgs e, string name, uint defaultValue = 0)
  {
    int idx = e.PayloadNames.IndexOf(name);
    if (idx < 0) { return defaultValue; }
    object o = e.Payload[idx];
    return Convert.ToUInt32(o, CultureInfo.InvariantCulture);
  }


  /// <summary>
  /// Reads a uint64 payload value from an event, returning 0 if not present.
  /// </summary>
  /// <param name="e"></param>
  /// <param name="name"></param>
  /// <returns></returns>
  private static ulong ReadU64Safe(EventWrittenEventArgs e, string name)
  {
    int idx = e.PayloadNames.IndexOf(name);
    if (idx < 0) { return 0UL; }
    object o = e.Payload[idx];
    return Convert.ToUInt64(o, CultureInfo.InvariantCulture);
  }


  /// <summary>
  /// Struct representing in-flight GC information.
  /// </summary>
  private struct GcInfo
  {
    public uint Count;
    public uint Generation;
    public uint Reason;
    public uint Type;

    public double StartTimeMs;
    public double EndTimeMs;

    public bool InPause;
    public double PauseStartMs;
    public double TotalPauseMs;

    public bool HeapStatsSeen;
    public bool EndSeen;

    public ulong Promoted0;
    public ulong Promoted1;
    public ulong SurvivedGen2;
    public ulong SurvivedLoh;
    public ulong SurvivedPoh;

    public ulong AllocSinceLastGcBytes;
  }
}
