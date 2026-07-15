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
using System.Text;

using Ceres.Base.Misc;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// Shared console-dump formatting helpers used by both the FPU/RPO select-phase dump
/// (FPUDumpDiagnostics, this namespace) and the TPS backup dump
/// (Ceres.MCGS.Search.TPS.TPSDumpDiagnostics).  Both emit the same "Label  v0 v1 v2 | v3 ..."
/// per-child row layout, so the row formatter, the fixed-width value formatters, the
/// significance-gated header line, and the shared console lock live here to keep the two
/// dumps column-aligned and mutually non-interleaved.
/// </summary>
internal static class SearchDumpFormatting
{
  /// <summary>
  /// Width of the leading label column.
  /// </summary>
  private const int LABEL_WIDTH = 22;


  /// <summary>
  /// When true, dumps are emitted ONLY for calculation invocations where the
  /// corresponding threshold was exceeded (i.e. the header would have been
  /// red).  Useful for surfacing just the cases where the TPS / RPO machinery
  /// actually diverges from the standard algorithm.  Default false.
  /// </summary>
  public const bool ONLY_SHOW_SIGNIFICANT = false;


  /// <summary>
  /// Shared lock that serializes an entire multi-line dump for a single invocation,
  /// preventing rows from concurrent calc threads (or from an FPU dump and a TPS dump)
  /// from interleaving on the console.
  /// </summary>
  public static readonly object ConsoleLock = new object();


  /// <summary>
  /// Formats a single row of per-child values with a '|' separator inserted
  /// after the expanded/unexpanded boundary (i.e. after index boundaryIndex-1).
  /// If boundaryIndex is &lt;= 0 or &gt;= count, no separator is inserted.
  /// </summary>
  public static string FormatRow(string label, int count, int boundaryIndex, Func<int, string> valueFunc)
  {
    StringBuilder sb = new();
    sb.Append(label.PadRight(LABEL_WIDTH));
    for (int i = 0; i < count; i++)
    {
      if (i > 0)
      {
        sb.Append(' ');
      }
      sb.Append(valueFunc(i));
      if (i == boundaryIndex - 1 && boundaryIndex > 0 && boundaryIndex < count)
      {
        sb.Append(" |");
      }
    }
    return sb.ToString();
  }


  /// <summary>
  /// Formats a double with a fixed 6-char field width (sign + 0.000).
  /// </summary>
  public static string FmtQ(double v)
  {
    if (double.IsNaN(v))
    {
      return "  NaN ";
    }
    return v.ToString("+0.000;-0.000; 0.000").PadLeft(6);
  }


  /// <summary>
  /// Formats a non-negative double (policy / pi) in [0,1] with 6 chars.
  /// </summary>
  public static string FmtP(double v)
  {
    if (double.IsNaN(v))
    {
      return "  NaN ";
    }
    return v.ToString("0.000").PadLeft(6);
  }


  /// <summary>
  /// Formats an integer-valued visit count (edge.N / child.N) with no decimal point,
  /// 6-char field to stay column-aligned with the other rows.
  /// </summary>
  public static string FmtNInt(double v)
  {
    return v.ToString("0").PadLeft(6);
  }


  /// <summary>
  /// Writes a dump header line.  If significant is true, the header is rendered
  /// in red to draw the eye; otherwise default color is used.
  /// </summary>
  public static void WriteHeaderLine(string header, bool significant)
  {
    if (significant)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, header);
    }
    else
    {
      Console.WriteLine(header);
    }
  }
}
