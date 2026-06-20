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
using System.IO;
using System.Linq;
using System.Text;

#endregion

namespace Ceres.Features.Suites
{
  /// <summary>
  /// Renders the multiengine suite statistics as a single block which is refreshed in place
  /// as the suite runs (rather than appending one row per position).
  ///
  /// The block contains, for each engine (one column per engine, baseline marked with '*'):
  ///   Q difference                  - mean |dQ| vs the baseline    (baseline / zero blank, lower=better)
  ///   Policy difference             - mean symmetric KLD vs baseline (baseline / unavailable blank, lower=better)
  ///                                   (blank separator line)
  ///   Search time (average sec)     - baseline shows absolute secs; others show % delta vs baseline (lower=better)
  ///   Evaluation intensity (EPS)    - baseline shows absolute EPS; others show % delta vs baseline (higher=better)
  ///                                   (blank separator line)
  ///   Solve score                   - average solve score percent    (true value, higher=better)
  ///   Solve correct move visits (3%)- average % of root visits on correct move(s) (higher=better)
  ///   Graded solution-quality difference - z-score of the paired graded-score difference vs the
  ///                                   baseline (baseline blank; positive=this engine better, higher=better)
  ///
  /// On each refresh the "best" cell of every row is colored green and the "worst" red (where
  /// for the two difference rows the engine closest to the baseline is considered best).
  ///
  /// When the output is an interactive console the block is rewritten in place (the cursor is
  /// repositioned to the top of the block); otherwise (output redirected to a file, NO_COLOR
  /// set, etc.) coloring and in-place refresh are skipped and the block is written as plain
  /// text once at the end of the run (with occasional progress lines while running).
  /// </summary>
  internal sealed class MultiEngineLiveDisplay
  {
    const int LABEL_WIDTH = 36;
    const int COL_WIDTH = 12;

    readonly TextWriter output;
    readonly bool interactive;
    readonly string[] columnHeaders;
    readonly int totalPositions;

    readonly ConsoleColor originalColor;

    // Elapsed wall-clock time of the run (started at display construction, just before the
    // position loop) so the header can show a live elapsed-seconds counter.
    readonly Stopwatch stopwatch = Stopwatch.StartNew();

    int lineCount;
    bool firstRender = true;
    int lastProgressMilestone = -1;

    /// <summary>The header progress label, e.g. "Positions 197/200  (72.3 seconds)".</summary>
    private string ProgressLabel(int positionsDone)
      => $"Positions {positionsDone}/{totalPositions}  ({stopwatch.Elapsed.TotalSeconds:F1} seconds)";

    /// <summary>One displayed statistics row.</summary>
    private sealed class RowDef
    {
      public string Label;
      public bool IsSeparator;
      public Func<MultiEngineEngineResult, float> Selector;  // returns NaN when the cell is blank
      public string Format;
      public bool LowerIsBetter;   // true for the difference rows
      public float BlankIfAbsBelow;  // values with |v| below this render blank (e.g. exact-zero differences)

      // When true, the baseline cell shows the absolute value (Format) while every other cell
      // shows its delta from the baseline (DeltaFormat). Best/worst coloring still uses the
      // underlying absolute values. Used by the search-time and EPS rows.
      public bool BaselineRelativeDelta;
      public string DeltaFormat;

      // When true (and BaselineRelativeDelta), the non-baseline delta is a percentage of the
      // baseline value ((v - base)/base) rather than an absolute difference. DeltaFormat should
      // then be a percent format (the '%' specifier scales the fraction by 100), e.g. "+0.0%;-0.0%;0.0%".
      public bool DeltaAsPercent;

      public static RowDef Separator() => new RowDef() { IsSeparator = true };
    }

    readonly RowDef[] rows;


    public MultiEngineLiveDisplay(TextWriter output, MultiEngineEntry[] entries, int totalPositions)
    {
      this.output = output;
      this.totalPositions = totalPositions;

      columnHeaders = entries.Select(e => e.ID + (e.IsBaseline ? "*" : "")).ToArray();

      bool noColor = Environment.GetEnvironmentVariable("NO_COLOR") != null;
      interactive = ReferenceEquals(output, Console.Out) && !Console.IsOutputRedirected && !noColor;
      originalColor = interactive ? Console.ForegroundColor : ConsoleColor.Gray;

      rows = new[]
      {
        new RowDef { Label = "Q difference",                   Selector = r => r.AvgAbsQDiffVsBaseline,  Format = "0.000",  LowerIsBetter = true,  BlankIfAbsBelow = 0.0005f },
        new RowDef { Label = "Policy difference",              Selector = r => r.AvgPolicyKLDVsBaseline, Format = "0.0000", LowerIsBetter = true,  BlankIfAbsBelow = 0.00005f },
        RowDef.Separator(),
        new RowDef { Label = "Search time (average sec)",      Selector = r => r.AvgTimeSecs,            Format = "0.00",   LowerIsBetter = true,  BaselineRelativeDelta = true, DeltaAsPercent = true, DeltaFormat = "+0.0%;-0.0%;0.0%" },
        new RowDef { Label = "Evaluation intensity (EPS)",     Selector = r => r.AvgEPS,                 Format = "N0",     LowerIsBetter = false, BaselineRelativeDelta = true, DeltaAsPercent = true, DeltaFormat = "+0.0%;-0.0%;0.0%" },
        new RowDef { Label = "Backend busy fraction",          Selector = r => r.BackendBusyFraction,    Format = "0.000",  LowerIsBetter = false },
        RowDef.Separator(),
        new RowDef { Label = "Solve score",                    Selector = r => r.AvgSolveScorePct,       Format = "0.0",    LowerIsBetter = false },
        new RowDef { Label = "Solve correct move visits (3%)", Selector = r => r.AvgCorrectMoveVisitPct, Format = "0.0",    LowerIsBetter = false },
        new RowDef { Label = "Graded solution-quality difference", Selector = r => r.GradedScoreDiffZVsBaseline, Format = "0.0", LowerIsBetter = false },
      };
    }


    /// <summary>
    /// Refreshes the block with the latest per-engine aggregates. Safe to call from the suite
    /// worker threads provided the caller holds the suite lock (so renders do not interleave).
    /// </summary>
    public void Refresh(MultiEngineEngineResult[] engines, int positionsDone)
    {
      if (interactive)
      {
        RenderInPlace(engines, positionsDone);
      }
      else
      {
        // Non-interactive (redirected / NO_COLOR): emit a sparse progress line so long runs
        // show life in logs; the full block is written by RenderFinal at the end.
        int milestoneStep = Math.Max(1, totalPositions / 20);
        int milestone = positionsDone / milestoneStep;
        if (milestone != lastProgressMilestone)
        {
          lastProgressMilestone = milestone;
          output.WriteLine($"  ... {ProgressLabel(positionsDone)}");
        }
      }
    }


    /// <summary>
    /// Writes the final (frozen) block. In interactive mode this is the last in-place render
    /// followed by a newline so subsequent output starts below the block; in non-interactive
    /// mode the block is written once here as plain text.
    /// </summary>
    public void RenderFinal(MultiEngineEngineResult[] engines, int positionsDone)
    {
      if (interactive)
      {
        RenderInPlace(engines, positionsDone);
        Console.ForegroundColor = originalColor;
        Console.WriteLine();
      }
      else
      {
        foreach (string line in BuildPlainLines(engines, positionsDone))
        {
          output.WriteLine(line);
        }
      }
    }


    #region Interactive (in-place, colored) rendering

    private void RenderInPlace(MultiEngineEngineResult[] engines, int positionsDone)
    {
      try
      {
        if (!firstRender)
        {
          Console.SetCursorPosition(0, Math.Max(0, Console.CursorTop - lineCount));
        }
      }
      catch
      {
        // Cursor repositioning unavailable (e.g. buffer too small) - fall back to appending.
      }

      int lines = 0;

      WriteFixed(ProgressLabel(positionsDone), LABEL_WIDTH + columnHeaders.Length * COL_WIDTH);
      Console.WriteLine();
      lines++;

      // Header row (labels blank, then engine ids).
      WriteFixed("", LABEL_WIDTH);
      foreach (string h in columnHeaders)
      {
        WriteCell(h, originalColor);
      }
      Console.WriteLine();
      lines++;

      // Divider.
      WriteFixed(new string('-', LABEL_WIDTH - 1), LABEL_WIDTH);
      foreach (string _ in columnHeaders)
      {
        WriteCell(new string('-', COL_WIDTH - 2), originalColor);
      }
      Console.WriteLine();
      lines++;

      foreach (RowDef row in rows)
      {
        if (row.IsSeparator)
        {
          WriteFixed("", LABEL_WIDTH + columnHeaders.Length * COL_WIDTH);
          Console.WriteLine();
          lines++;
          continue;
        }

        WriteFixed(row.Label, LABEL_WIDTH);

        (string text, bool isBlank)[] cells = BuildRowCells(row, engines, out int bestIdx, out int worstIdx);
        for (int i = 0; i < cells.Length; i++)
        {
          ConsoleColor color = originalColor;
          if (!cells[i].isBlank && bestIdx != worstIdx)
          {
            if (i == bestIdx)
            {
              color = ConsoleColor.Green;
            }
            else if (i == worstIdx)
            {
              color = ConsoleColor.Red;
            }
          }
          WriteCell(cells[i].text, color);
        }
        Console.WriteLine();
        lines++;
      }

      Console.ForegroundColor = originalColor;
      lineCount = lines;
      firstRender = false;
    }


    /// <summary>Writes a right-aligned, fixed-width cell in the specified color.</summary>
    private void WriteCell(string text, ConsoleColor color)
    {
      Console.ForegroundColor = color;
      Console.Write(Pad(text, COL_WIDTH));
      Console.ForegroundColor = originalColor;
    }

    /// <summary>Writes a left-aligned, fixed-width field (no color change).</summary>
    private void WriteFixed(string text, int width)
    {
      if (text.Length > width)
      {
        text = text.Substring(0, width);
      }
      Console.Write(text.PadRight(width));
    }

    #endregion


    #region Shared cell computation

    /// <summary>
    /// Builds the displayed text for every engine cell of a row, and identifies which cell is
    /// best (green) and worst (red). Blank cells (NaN, or magnitudes below the row's threshold)
    /// do not participate in the best/worst selection. If fewer than two distinct values are
    /// present, bestIdx == worstIdx (signaling "no coloring").
    /// </summary>
    private (string text, bool isBlank)[] BuildRowCells(RowDef row, MultiEngineEngineResult[] engines,
                                                        out int bestIdx, out int worstIdx)
    {
      (string text, bool isBlank)[] cells = new (string, bool)[engines.Length];

      bestIdx = -1;
      worstIdx = -1;
      float bestVal = 0, worstVal = 0;

      // For a baseline-relative-delta row, the baseline cell shows the absolute value while the
      // others show their signed delta from it (only possible if the baseline value is present).
      float baselineVal = float.NaN;
      if (row.BaselineRelativeDelta)
      {
        for (int i = 0; i < engines.Length; i++)
        {
          if (engines[i].IsBaseline)
          {
            baselineVal = row.Selector(engines[i]);
            break;
          }
        }
      }

      for (int i = 0; i < engines.Length; i++)
      {
        float v = row.Selector(engines[i]);
        bool blank = float.IsNaN(v) || Math.Abs(v) < row.BlankIfAbsBelow;

        string text;
        if (blank)
        {
          text = "";
        }
        else if (row.BaselineRelativeDelta && !engines[i].IsBaseline && !float.IsNaN(baselineVal)
              && (!row.DeltaAsPercent || baselineVal != 0))
        {
          // Non-baseline cell: signed delta from baseline, as a percentage of the baseline value
          // (the '%' in DeltaFormat scales the fraction by 100) or as an absolute difference.
          text = row.DeltaAsPercent
               ? ((v - baselineVal) / baselineVal).ToString(row.DeltaFormat)
               : (v - baselineVal).ToString(row.DeltaFormat);
        }
        else
        {
          text = v.ToString(row.Format);
        }
        cells[i] = (text, blank);

        if (blank)
        {
          continue;
        }

        // Best/worst tracking. For the difference rows lower is better (closer to baseline).
        bool better = bestIdx < 0 || (row.LowerIsBetter ? v < bestVal : v > bestVal);
        if (better)
        {
          bestIdx = i;
          bestVal = v;
        }
        bool worse = worstIdx < 0 || (row.LowerIsBetter ? v > worstVal : v < worstVal);
        if (worse)
        {
          worstIdx = i;
          worstVal = v;
        }
      }

      // If all displayed values are equal, do not color anything.
      if (bestIdx >= 0 && bestVal == worstVal)
      {
        bestIdx = worstIdx = -1;
      }

      return cells;
    }


    private string[] BuildPlainLines(MultiEngineEngineResult[] engines, int positionsDone)
    {
      var lines = new System.Collections.Generic.List<string>();

      lines.Add(ProgressLabel(positionsDone));

      StringBuilder header = new StringBuilder();
      header.Append("".PadRight(LABEL_WIDTH));
      foreach (string h in columnHeaders)
      {
        header.Append(Pad(h, COL_WIDTH));
      }
      lines.Add(header.ToString());

      StringBuilder divider = new StringBuilder();
      divider.Append(new string('-', LABEL_WIDTH - 1).PadRight(LABEL_WIDTH));
      foreach (string _ in columnHeaders)
      {
        divider.Append(Pad(new string('-', COL_WIDTH - 2), COL_WIDTH));
      }
      lines.Add(divider.ToString());

      foreach (RowDef row in rows)
      {
        if (row.IsSeparator)
        {
          lines.Add("");
          continue;
        }

        StringBuilder sb = new StringBuilder();
        sb.Append(row.Label.PadRight(LABEL_WIDTH));
        (string text, bool isBlank)[] cells = BuildRowCells(row, engines, out _, out _);
        foreach ((string text, bool _) in cells)
        {
          sb.Append(Pad(text, COL_WIDTH));
        }
        lines.Add(sb.ToString());
      }

      return lines.ToArray();
    }


    private static string Pad(string s, int width)
    {
      if (s.Length > width)
      {
        s = s.Substring(0, width);
      }
      return s.PadLeft(width);
    }

    #endregion
  }
}
