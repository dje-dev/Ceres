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
using System.Diagnostics;
using System.Runtime;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.MCGS.UCI;

/// <summary>
/// Helpers for emitting a "dump-info-block": a self-contained diagnostics block consisting of
/// a header (process/GC/machine statistics) followed by the standard dump-info content, all
/// bracketed by unambiguous begin/end marker lines.
///
/// The markers allow a consuming process (e.g. the tournament manager driving an external Ceres
/// engine over UCI) to cleanly demarcate this block from any other UCI output that may be
/// interleaved on the stream, capture it, and strip the markers before display.
/// </summary>
public static class DiagnosticsBlock
{
  /// <summary>
  /// Line emitted at the very start of a dump-info-block. Consumers should match this exactly
  /// (after trimming) to know where the block begins.
  /// </summary>
  public const string BEGIN_MARKER = "===CERES-DUMP-INFO-BLOCK-BEGIN===";

  /// <summary>
  /// Line emitted at the very end of a dump-info-block. Consumers should match this exactly
  /// (after trimming) to know where the block ends.
  /// </summary>
  public const string END_MARKER = "===CERES-DUMP-INFO-BLOCK-END===";

  /// <summary>
  /// Shared lock used to serialize diagnostics output to the console across all engines (in-process
  /// MCGS engines and the wrappers that capture/display dumps from external UCI engines) so their
  /// output never interleaves.
  /// </summary>
  public static readonly object ConsoleLock = new();


  /// <summary>
  /// Writes a complete dump-info-block to the specified writer: the BEGIN marker, the system
  /// information header, the caller-supplied body, and finally the END marker.
  ///
  /// Callers that need the block to appear atomically (uninterrupted by other console output)
  /// should hold the relevant output lock while invoking this method.
  /// </summary>
  /// <param name="writer">destination for the block</param>
  /// <param name="writeBody">action that writes the dump-info content (between header and END marker)</param>
  public static void WriteBlock(TextWriter writer, Action<TextWriter> writeBody)
  {
    writer.WriteLine(BEGIN_MARKER);
    WriteSystemInfoHeader(writer);
    try
    {
      writeBody?.Invoke(writer);
    }
    catch (Exception exc)
    {
      writer.WriteLine("dump-info body failed: " + exc.Message);
    }
    writer.WriteLine(END_MARKER);
  }


  /// <summary>
  /// Writes a header block of process, garbage collector, and machine/operating-system statistics.
  /// Each statistic is gathered defensively so that an unavailable value never aborts the dump.
  /// </summary>
  /// <param name="writer">destination for the header</param>
  public static void WriteSystemInfoHeader(TextWriter writer)
  {
    writer.WriteLine("----- System Information -----");

    // Machine and operating system.
    TryWrite(writer, "Machine", () => System.Environment.MachineName);
    TryWrite(writer, "OS", () => RuntimeInformation.OSDescription + " (" + RuntimeInformation.OSArchitecture + ")");
    TryWrite(writer, "Runtime", () => RuntimeInformation.FrameworkDescription);
    TryWrite(writer, "Processors", () => System.Environment.ProcessorCount.ToString());
    TryWrite(writer, "User/Host", () => System.Environment.UserName + " @ " + System.Environment.UserDomainName);

    // Process timing and memory.
    try
    {
      Process proc = Process.GetCurrentProcess();
      TimeSpan wall = DateTime.Now - proc.StartTime;
      writer.WriteLine($"Process PID            : {proc.Id}");
      writer.WriteLine($"Process wall time      : {FormatTimeSpan(wall)}");
      writer.WriteLine($"Process CPU time       : {FormatTimeSpan(proc.TotalProcessorTime)}");
      writer.WriteLine($"Working set            : {FormatBytes(proc.WorkingSet64)}");
      writer.WriteLine($"Peak working set       : {FormatBytes(proc.PeakWorkingSet64)}");
      writer.WriteLine($"Private memory         : {FormatBytes(proc.PrivateMemorySize64)}");
      writer.WriteLine($"Thread count           : {proc.Threads.Count}");
    }
    catch (Exception exc)
    {
      writer.WriteLine("Process stats           : (unavailable: " + exc.Message + ")");
    }

    // Garbage collector statistics.
    try
    {
      writer.WriteLine($"GC managed heap        : {FormatBytes(GC.GetTotalMemory(false))}");
      writer.WriteLine($"GC collections         : gen0={GC.CollectionCount(0)} gen1={GC.CollectionCount(1)} gen2={GC.CollectionCount(2)}");
      writer.WriteLine($"GC mode                : ServerGC={GCSettings.IsServerGC} Latency={GCSettings.LatencyMode}");
    }
    catch (Exception exc)
    {
      writer.WriteLine("GC stats               : (unavailable: " + exc.Message + ")");
    }

    try
    {
      writer.WriteLine($"GC total pause time    : {FormatTimeSpan(GC.GetTotalPauseDuration())}");
    }
    catch
    {
      // GetTotalPauseDuration not available on this runtime; skip silently.
    }

    // Per-generation / large-object-heap sizes from the most recent GC.
    try
    {
      GCMemoryInfo info = GC.GetGCMemoryInfo();
      writer.WriteLine($"GC heap (last GC)      : {FormatBytes(info.HeapSizeBytes)} (committed {FormatBytes(info.TotalCommittedBytes)})");

      // GenerationInfo indices: 0=gen0, 1=gen1, 2=gen2, 3=LOH (large object heap), 4=POH (pinned object heap).
      string[] genNames = { "gen0", "gen1", "gen2", "LOH", "POH" };
      ReadOnlySpan<GCGenerationInfo> gens = info.GenerationInfo;
      for (int i = 0; i < gens.Length && i < genNames.Length; i++)
      {
        writer.WriteLine($"  {genNames[i],-4} size after GC      : {FormatBytes(gens[i].SizeAfterBytes)}");
      }
    }
    catch (Exception exc)
    {
      writer.WriteLine("GC generation sizes    : (unavailable: " + exc.Message + ")");
    }

    writer.WriteLine("-----------------------------");
  }


  /// <summary>
  /// Writes a "label : value" line, substituting an error note if the value getter throws.
  /// </summary>
  static void TryWrite(TextWriter writer, string label, Func<string> valueGetter)
  {
    string value;
    try
    {
      value = valueGetter();
    }
    catch (Exception exc)
    {
      value = "(unavailable: " + exc.Message + ")";
    }
    writer.WriteLine($"{label,-22}: {value}");
  }


  /// <summary>
  /// Formats a byte count as a human-readable size (bytes, KB, MB, or GB).
  /// </summary>
  static string FormatBytes(long bytes)
  {
    const double KB = 1024.0;
    const double MB = KB * 1024.0;
    const double GB = MB * 1024.0;
    if (bytes >= GB) return $"{bytes / GB:F2} GB";
    if (bytes >= MB) return $"{bytes / MB:F2} MB";
    if (bytes >= KB) return $"{bytes / KB:F2} KB";
    return $"{bytes} bytes";
  }


  /// <summary>
  /// Formats a TimeSpan compactly (e.g. "1:02:03.456" or "5.123s").
  /// </summary>
  static string FormatTimeSpan(TimeSpan ts)
  {
    if (ts.TotalHours >= 1)
    {
      return $"{(int)ts.TotalHours}:{ts.Minutes:D2}:{ts.Seconds:D2}.{ts.Milliseconds:D3}";
    }
    if (ts.TotalMinutes >= 1)
    {
      return $"{ts.Minutes}:{ts.Seconds:D2}.{ts.Milliseconds:D3}";
    }
    return $"{ts.TotalSeconds:F3}s";
  }
}
