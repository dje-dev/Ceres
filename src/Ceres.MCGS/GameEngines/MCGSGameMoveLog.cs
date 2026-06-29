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
using System.Globalization;
using System.IO;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;

using Ceres.MCGS.UCI;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.GameEngines;

/// <summary>
/// Writer for a per-tournament diagnostic "move log" text file emitted by a
/// GameEngineCeresMCGSInProcess. The file is intended primarily for consumption by
/// post-processing tools, with brief parser-friendly labels.
///
/// File structure (single file for the whole tournament):
///   - a one-time header (timestamp, host/system info, engine configuration dumps),
///   - per game: a separator line, then one line per move (flushed after each),
///   - per game: a result footer block supplied by the caller.
///
/// This class owns only the underlying file I/O (it holds no graph/search state) and
/// guards all writes with an internal lock so it remains safe even if shared across
/// engine instances.
/// </summary>
public sealed class MCGSGameMoveLog : IDisposable
{
  /// <summary>
  /// Order of the scalar tokens emitted on each move line (documented in the header legend).
  /// </summary>
  public const string BODY_LEGEND =
    "LEGEND (per move line): FEN, RootN, StoreN, NNEvals, TimeRem, LimInit, Elapsed, "
    + "BudgetFrac%, NPS, EPS, BackendBusy, Depth, SelDepth | candidate moves as (SAN, visit%, Q) "
    + "sorted by visits descending, '*' prefixes the played move, Q from side-to-move perspective. "
    + "LimInit is the per-move allocated budget (seconds for time limits, nodes for node limits); "
    + "BudgetFrac is the percent of that budget used this move (elapsed/LimInit for time limits, "
    + "nodes-this-search/LimInit for node limits). "
    + "Note: RootN/visit% are cumulative across tree reuse, so per-move visit% may sum to under 100.";

  readonly string fileName;
  readonly StreamWriter writer;
  readonly object lockObj = new();

  bool headerWritten;
  bool closed;

  /// <summary>
  /// Name of the underlying file being written.
  /// </summary>
  public string FileName => fileName;


  /// <summary>
  /// Constructor. Creates (or truncates) the target file.
  /// </summary>
  /// <param name="fileName">full path of the file to write</param>
  public MCGSGameMoveLog(string fileName)
  {
    this.fileName = fileName ?? throw new ArgumentNullException(nameof(fileName));
    writer = new StreamWriter(fileName, append: false) { AutoFlush = false };
  }


  /// <summary>
  /// Writes the one-time header section (idempotent; subsequent calls are ignored).
  /// Includes timestamp, host/system information, engine identity and evaluator,
  /// the assigned search limit, and a full dump of the search and select parameters.
  /// </summary>
  public void WriteHeader(string engineID, NNEvaluatorDef evaluatorDef, SearchLimit assignedSearchLimit,
                          ParamsSearch searchParams, ParamsSelect selectParams)
  {
    lock (lockObj)
    {
      if (closed || headerWritten)
      {
        return;
      }
      headerWritten = true;

      writer.WriteLine("=== CERES MCGS GAME MOVE LOG ===");
      writer.WriteLine("Timestamp: " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));

      // Host / OS / runtime / process information (reuse existing diagnostics helper).
      DiagnosticsBlock.WriteSystemInfoHeader(writer);

      writer.WriteLine("Engine    : " + engineID);
      writer.WriteLine("Evaluator : " + (evaluatorDef == null ? "(none)" : evaluatorDef.ToString()));
      writer.WriteLine("AssignedSearchLimit: " + (assignedSearchLimit == null ? "(none)" : assignedSearchLimit.ToString()));

      if (searchParams != null)
      {
        writer.WriteLine();
        writer.WriteLine("--- ParamsSearch (all properties) ---");
        writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearch>(searchParams, new ParamsSearch(), false));

        writer.WriteLine("--- ParamsSearchExecution (all properties) ---");
        writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(searchParams.Execution, new ParamsSearchExecution(), false));
      }

      if (selectParams != null)
      {
        writer.WriteLine("--- ParamsSelect (all properties) ---");
        writer.Write(ObjUtils.FieldValuesDumpString<ParamsSelect>(selectParams, new ParamsSelect(), false));
      }

      writer.WriteLine();
      writer.WriteLine(BODY_LEGEND);
      writer.WriteLine();
      writer.Flush();
    }
  }


  /// <summary>
  /// Writes a separator marking the start of a new game.
  /// </summary>
  public void WriteNewGameSeparator(string gameID)
  {
    lock (lockObj)
    {
      if (closed)
      {
        return;
      }
      writer.WriteLine();
      writer.WriteLine("=== NEW GAME: " + (gameID ?? "(unnamed)") + " ===");
      writer.Flush();
    }
  }


  /// <summary>
  /// Writes a single (already formatted) per-move body line and flushes,
  /// so the most recent move survives an abnormal termination.
  /// </summary>
  public void WriteMoveLine(string line)
  {
    lock (lockObj)
    {
      if (closed)
      {
        return;
      }
      writer.WriteLine(line);
      writer.Flush();
    }
  }


  /// <summary>
  /// Appends a (preformatted) per-game result footer block and flushes.
  /// </summary>
  public void AppendGameResultFooter(string footerText)
  {
    lock (lockObj)
    {
      if (closed)
      {
        return;
      }
      writer.WriteLine(footerText);
      writer.Flush();

      // Regenerate the standalone HTML rendering alongside the log (same directory) at the end of
      // each game, so an up-to-date view is always available. Never let this disrupt the game.
      try
      {
        MCGSGameMoveLogHtmlFormatter.WriteHtmlFile(fileName, fileName + ".html");
      }
      catch (Exception)
      {
        // Ignore HTML generation failures.
      }
    }
  }


  /// <summary>
  /// Appends a (preformatted) blunder-diagnostics block, delimited by markers so post-processors can
  /// locate it, and flushes. Emitted inline after the move that triggered the blunder confirmation.
  /// </summary>
  public void AppendBlunderSection(string blunderText)
  {
    lock (lockObj)
    {
      if (closed)
      {
        return;
      }
      writer.WriteLine("=== BLUNDER ===");
      writer.WriteLine(blunderText);
      writer.WriteLine("=== END BLUNDER ===");
      writer.Flush();
    }
  }


  /// <summary>
  /// Flushes and closes the underlying file (idempotent).
  /// </summary>
  public void Close()
  {
    lock (lockObj)
    {
      if (closed)
      {
        return;
      }
      closed = true;
      writer.Flush();
      writer.Dispose();
    }
  }


  void IDisposable.Dispose() => Close();
}
