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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;

using Ceres.Chess;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.MCGS.UCI;

/// <summary>
/// Implementation of the "game-analyze" feature, which locates a specified position (or range of
/// positions) within a PGN file (by move number / side to move), feeds it to the engine with full
/// move history, runs a fixed-time search, and dumps detailed search information.
///
/// A move specification may be a single move ("105", "105.." / "..105" for Black to move) or a
/// range "&lt;start&gt;-&lt;end&gt;" (e.g. "1-9999" for effectively all moves, or "..7-..12" for
/// Black's 7th through Black's 12th move). For a range, every position arising between the two
/// endpoints (inclusive) is analyzed in sequence; consecutive searches reuse the prior search graph
/// (each position is one ply deeper than the last), so the search continues rather than restarting.
///
/// Interactive controls while a (range) analysis is running on an interactive console:
///   Ctrl-C  ends the analysis (and returns to the UCI prompt).
///   Ctrl-P  toggles "pause" mode, in which the analysis waits for &lt;Enter&gt; after displaying each
///           position's dump before showing the next. The next position is computed in the background
///           while paused, so it is ready (or nearly so) when the user advances.
/// </summary>
public partial class UCIManagerMCGS
{
  #region Interactive state (game-analyze)

  /// <summary>When non-null, UCIWriteLine output is captured here instead of the console (used to
  /// render a position's output in the background so it does not interleave with the displayed one).</summary>
  private volatile TextWriter gameAnalyzeCaptureWriter;

  /// <summary>Set when the user requests the analysis to end (Ctrl-C).</summary>
  private volatile bool gameAnalyzeCancelRequested;

  /// <summary>Whether pause-after-each-position mode is currently enabled (toggled by Ctrl-P).</summary>
  private volatile bool gameAnalyzePauseEnabled;

  /// <summary>Signals the key-monitor thread to stop.</summary>
  private volatile bool gameAnalyzeStopKeyMonitor;

  /// <summary>Signaled (by the key monitor) when the user presses Enter to advance past a pause.</summary>
  private AutoResetEvent gameAnalyzeEnterEvent;

  /// <summary>Signaled when the analysis is cancelled (wakes a paused consumer).</summary>
  private ManualResetEventSlim gameAnalyzeCancelEvent;

  #endregion


  /// <summary>
  /// Entry point from the UCI command loop (or the command-line verb startup hook).
  /// Parses any arguments already present on the command line and interactively prompts
  /// for any that are missing, then runs the analysis.
  /// </summary>
  /// <param name="command">the full command line, e.g. "game-analyze /path/game.pgn 105 10s"</param>
  internal void ProcessGameAnalyzeInteractive(string command)
  {
    if (TryParseGameAnalyzeArgs(command, "game-analyze", out string pgnFile, out string moveSpec, out string timeSpec))
    {
      try
      {
        ProcessGameAnalyze(pgnFile, moveSpec, timeSpec);
      }
      catch (Exception exc)
      {
        UCIWriteLine($"info string game-analyze failed: {exc.Message}");
      }
    }
  }


  /// <summary>
  /// Parses the three game-analyze arguments from the command line, interactively prompting
  /// for any that are missing. Returns false (with an error message) if any remain empty.
  /// </summary>
  internal bool TryParseGameAnalyzeArgs(string command, string commandName,
                                        out string pgnFile, out string moveSpec, out string timeSpec)
  {
    string[] parts = command.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    pgnFile = parts.Length > 1 ? parts[1]
                               : PromptForGameAnalyzeArg("Enter path to PGN file:");
    moveSpec = parts.Length > 2 ? parts[2]
                                : PromptForGameAnalyzeArg("Enter move number or range (e.g. 105, 105.., 1-9999, ..7-..12):");
    timeSpec = parts.Length > 3 ? parts[3]
                                : PromptForGameAnalyzeArg("Enter search time (e.g. 10s):");

    if (string.IsNullOrWhiteSpace(pgnFile) || string.IsNullOrWhiteSpace(moveSpec) || string.IsNullOrWhiteSpace(timeSpec))
    {
      UCIWriteLine($"info string {commandName} requires three arguments: <pgn file> <move number or range> <time> (e.g. {commandName} game.pgn 105 10s)");
      return false;
    }

    return true;
  }


  /// <summary>
  /// Writes a prompt and reads a line of input (used to interactively gather missing arguments).
  /// </summary>
  private string PromptForGameAnalyzeArg(string prompt)
  {
    UCIWriteLine(prompt);
    return InStream.ReadLine();
  }


  /// <summary>
  /// Loads the first game from the specified PGN file. Returns false (with an error message) on failure.
  /// </summary>
  internal bool TryLoadGameAnalyzeGame(string pgnFile, string commandName, out Game game)
  {
    game = null;

    if (!System.IO.File.Exists(pgnFile))
    {
      UCIWriteLine($"info string {commandName}: PGN file not found: {pgnFile}");
      return false;
    }

    game = Game.FromPGN(pgnFile).FirstOrDefault();
    if (game == null)
    {
      UCIWriteLine($"info string {commandName}: no games found in {pgnFile}");
      return false;
    }

    return true;
  }


  /// <summary>
  /// Parses a single move endpoint ("105", "105..", "..105"; dots anywhere indicate Black to move).
  /// On success returns the full-move number, side-to-move flag, and the encoded ply value used by
  /// PositionMiscInfo.MoveNum (fullmove * 2, +1 if Black to move).
  /// </summary>
  internal static bool TryParseMoveEndpoint(string spec, out int moveNumber, out bool blackToMove, out int targetMoveNum)
  {
    moveNumber = 0;
    blackToMove = false;
    targetMoveNum = 0;

    if (string.IsNullOrWhiteSpace(spec))
    {
      return false;
    }

    spec = spec.Trim();
    blackToMove = spec.Contains('.');     // dots anywhere (leading or trailing) indicate Black to move
    string numberPart = spec.Trim('.');   // strip leading and trailing dots
    if (!int.TryParse(numberPart, NumberStyles.Integer, CultureInfo.InvariantCulture, out moveNumber) || moveNumber < 1)
    {
      return false;
    }

    targetMoveNum = moveNumber * 2 + (blackToMove ? 1 : 0);
    return true;
  }


  /// <summary>
  /// Resolves a move specification (single move or "&lt;start&gt;-&lt;end&gt;" range) to the list of
  /// ply indices (positions) within the game to analyze, in ascending order. Returns false (with an
  /// error message) if the specification is invalid or matches no positions.
  /// </summary>
  internal bool TryResolveGameAnalyzePlies(Game game, string moveSpec, string commandName, out List<int> plyIndices)
  {
    plyIndices = new List<int>();
    string spec = moveSpec.Trim();

    int dashPos = spec.IndexOf('-');
    if (dashPos >= 0)
    {
      // Range: <start>-<end>. Each endpoint maps to an encoded MoveNum (ply) value; every position
      // whose MoveNum falls in [start, end] is included (MoveNum increases by one each ply).
      string startSpec = spec.Substring(0, dashPos);
      string endSpec = spec.Substring(dashPos + 1);
      if (!TryParseMoveEndpoint(startSpec, out _, out _, out int startMoveNum)
       || !TryParseMoveEndpoint(endSpec, out _, out _, out int endMoveNum))
      {
        UCIWriteLine($"info string {commandName}: invalid move range \"{moveSpec}\" (expected e.g. 1-9999 or ..7-..12)");
        return false;
      }

      if (startMoveNum > endMoveNum)
      {
        UCIWriteLine($"info string {commandName}: empty move range \"{moveSpec}\" (start is after end)");
        return false;
      }

      int plyIndex = 0;
      foreach (Position pos in game.Positions)
      {
        int mn = pos.MiscInfo.MoveNum;
        if (mn >= startMoveNum && mn <= endMoveNum)
        {
          plyIndices.Add(plyIndex);
        }
        plyIndex++;
      }

      if (plyIndices.Count == 0)
      {
        UCIWriteLine($"info string {commandName}: no positions in range \"{moveSpec}\" found in game ({game.Moves.Count} half-moves)");
        return false;
      }

      return true;
    }
    else
    {
      // Single move.
      if (!TryParseMoveEndpoint(spec, out int moveNumber, out bool blackToMove, out int targetMoveNum))
      {
        UCIWriteLine($"info string {commandName}: invalid move number \"{moveSpec}\" (expected e.g. 105 or 105..)");
        return false;
      }

      SideType targetSide = blackToMove ? SideType.Black : SideType.White;
      game.FirstMatchingPosition(pos => pos.MiscInfo.MoveNum == targetMoveNum
                                     && pos.MiscInfo.SideToMove == targetSide, out int moveIndex);
      if (moveIndex == -1)
      {
        UCIWriteLine($"info string {commandName}: position at move {moveNumber}{(blackToMove ? ".." : "")} "
                   + $"({targetSide} to move) not found in game (game has {game.Moves.Count} half-moves)");
        return false;
      }

      plyIndices.Add(moveIndex);
      return true;
    }
  }


  /// <summary>
  /// Builds the position (with full move history) after the specified number of plies, and the
  /// corresponding UCI "position fen ... moves ..." command.
  /// </summary>
  internal void BuildGameAnalyzePositionForPly(Game game, int plyIndex,
                                               out PositionWithHistory positionWithHistory, out string positionCommand)
  {
    Game truncatedGame = game.TruncatedAtMove(plyIndex);
    positionWithHistory = truncatedGame.FinalPositionWithHistory;
    positionCommand = "position fen " + positionWithHistory.GetFENAndMovesString(IsChess960OptionSet);
  }


  /// <summary>
  /// Returns a short description of a position's move number and side to move (e.g. "move 105.. (Black to move)").
  /// </summary>
  internal static string DescribeGameAnalyzePosition(Position pos)
  {
    int fullMove = pos.MiscInfo.MoveNum / 2;
    bool black = pos.MiscInfo.SideToMove == SideType.Black;
    return $"move {fullMove}{(black ? ".." : "")} ({pos.MiscInfo.SideToMove} to move)";
  }


  /// <summary>
  /// Loads the PGN, navigates to the position specified by the (single) move spec (preserving full move
  /// history), and builds the corresponding UCI "position fen ... moves ..." command. Echoes the
  /// constructed position and final FEN to the console. Returns false (with an error message) on failure.
  /// Used by the "game-analyze-lc0" feature (which analyzes a single position only).
  /// </summary>
  internal bool TryBuildGameAnalyzePosition(string pgnFile, string moveSpec, string commandName,
                                            out PositionWithHistory positionWithHistory, out string positionCommand)
  {
    positionWithHistory = null;
    positionCommand = null;

    if (!TryLoadGameAnalyzeGame(pgnFile, commandName, out Game game))
    {
      return false;
    }

    if (!TryResolveGameAnalyzePlies(game, moveSpec, commandName, out List<int> plyIndices))
    {
      return false;
    }

    if (plyIndices.Count != 1)
    {
      UCIWriteLine($"info string {commandName}: a move range is not supported here; please specify a single move.");
      return false;
    }

    BuildGameAnalyzePositionForPly(game, plyIndices[0], out positionWithHistory, out positionCommand);

    UCIWriteLine();
    UCIWriteLine($"{commandName}: {game.PlayerWhite} vs {game.PlayerBlack}, {DescribeGameAnalyzePosition(positionWithHistory.FinalPosition)}");
    UCIWriteLine(positionCommand);
    UCIWriteLine("Final FEN: " + positionWithHistory.FinalPosition.FEN);
    UCIWriteLine();

    return true;
  }


  /// <summary>
  /// Core of the game-analyze feature: loads the PGN, resolves the move spec (single move or range)
  /// to a sequence of positions, and analyzes each (in game order, preserving full move history),
  /// reusing the search graph from one position to the next. Leaves the last analyzed position current.
  /// </summary>
  /// <param name="pgnFile">path to the PGN file</param>
  /// <param name="moveSpec">single move ("105", "105..") or range ("1-9999", "..7-..12")</param>
  /// <param name="timeSpec">search time, e.g. "10s", "500ms", "1m" (bare number interpreted as seconds)</param>
  internal void ProcessGameAnalyze(string pgnFile, string moveSpec, string timeSpec)
  {
    // Validate / parse the time first (cheap, fail fast).
    int? searchMs = ParseAnalyzeTimeMs(timeSpec);
    if (searchMs == null || searchMs <= 0)
    {
      UCIWriteLine($"info string game-analyze: invalid time \"{timeSpec}\" (expected e.g. 10s, 500ms, 1m)");
      return;
    }

    if (!TryLoadGameAnalyzeGame(pgnFile, "game-analyze", out Game game))
    {
      return;
    }

    if (!TryResolveGameAnalyzePlies(game, moveSpec, "game-analyze", out List<int> plyIndices))
    {
      return;
    }

    if (!InitializeEngineIfNeeded())
    {
      return;
    }

    if (plyIndices.Count == 1)
    {
      AnalyzeGameAnalyzeSingle(game, plyIndices[0], searchMs.Value);
    }
    else
    {
      AnalyzeGameAnalyzeRange(game, plyIndices, searchMs.Value);
    }
  }


  /// <summary>
  /// Analyzes a single position (with full move history): runs a fixed-time search and dumps the
  /// search information. Ctrl-C ends the search early (the partial result is still dumped).
  /// </summary>
  private void AnalyzeGameAnalyzeSingle(Game game, int plyIndex, int searchMs)
  {
    gameAnalyzeCancelRequested = false;

    ConsoleCancelEventHandler ctrlCHandler = (s, e) =>
    {
      e.Cancel = true;
      gameAnalyzeCancelRequested = true;
      RequestGameAnalyzeStopCurrentSearch();
      OutStream.WriteLine();
      OutStream.WriteLine("game-analyze: Ctrl-C — ending analysis.");
    };
    Console.CancelKeyPress += ctrlCHandler;

    try
    {
      BuildGameAnalyzePositionForPly(game, plyIndex, out PositionWithHistory pwh, out string positionCommand);

      UCIWriteLine();
      UCIWriteLine($"game-analyze: {game.PlayerWhite} vs {game.PlayerBlack}, {DescribeGameAnalyzePosition(pwh.FinalPosition)}");
      UCIWriteLine(positionCommand);
      UCIWriteLine("Final FEN: " + pwh.FinalPosition.FEN);
      UCIWriteLine();

      ProcessPosition(positionCommand);
      taskSearchCurrentlyExecuting = ProcessGo($"go movetime {searchMs}");
      taskSearchCurrentlyExecuting.Wait();

      if (CeresEngine?.Search?.Manager != null)
      {
        CeresEngine.Search.Manager.DumpFullInfo(lastSearchResult, Console.Out, "game-analyze");
      }
    }
    finally
    {
      Console.CancelKeyPress -= ctrlCHandler;
    }
  }


  /// <summary>
  /// Analyzes a sequence of positions, reusing the search graph from one to the next. Each position's
  /// output is rendered (computed) on a background producer thread and displayed on this thread;
  /// in pause mode the display waits for &lt;Enter&gt; after each position (while the next is being
  /// computed). Ctrl-C ends the analysis. Ctrl-P toggles pause mode (interactive consoles only).
  /// </summary>
  private void AnalyzeGameAnalyzeRange(Game game, List<int> plyIndices, int searchMs)
  {
    int total = plyIndices.Count;

    gameAnalyzeCancelRequested = false;
    gameAnalyzePauseEnabled = false;
    gameAnalyzeStopKeyMonitor = false;
    gameAnalyzeEnterEvent = new AutoResetEvent(false);
    gameAnalyzeCancelEvent = new ManualResetEventSlim(false);

    bool interactive = !Console.IsInputRedirected;

    UCIWriteLine();
    UCIWriteLine($"game-analyze: {game.PlayerWhite} vs {game.PlayerBlack} — analyzing {total} positions for {searchMs}ms each (graph reused across positions).");
    if (interactive)
    {
      UCIWriteLine("game-analyze: press Ctrl-C to end the analysis, Ctrl-P to toggle pause-after-each-position mode.");
    }

    ConsoleCancelEventHandler ctrlCHandler = (s, e) =>
    {
      e.Cancel = true;
      RequestGameAnalyzeCancel();
    };
    Console.CancelKeyPress += ctrlCHandler;

    Thread keyThread = null;
    if (interactive)
    {
      keyThread = new Thread(GameAnalyzeKeyMonitorLoop) { IsBackground = true, Name = "GameAnalyzeKeyMonitor" };
      keyThread.Start();
    }

    BlockingCollection<(string output, string positionCommand)> queue
      = new BlockingCollection<(string, string)>(boundedCapacity: 1);

    // Producer: compute each position sequentially (graph reuse), render its output to a string, and
    // enqueue it. The bounded queue limits look-ahead to one position so we do not race far ahead.
    Thread producer = new Thread(() =>
    {
      try
      {
        for (int i = 0; i < total && !gameAnalyzeCancelRequested; i++)
        {
          string output = RenderGameAnalyzePosition(game, i, total, plyIndices[i], searchMs, out string positionCommand);
          try
          {
            queue.Add((output, positionCommand));
          }
          catch (InvalidOperationException)
          {
            break;   // queue marked complete (consumer ended)
          }
        }
      }
      catch (Exception exc)
      {
        try { queue.Add(($"info string game-analyze error: {exc.Message}{System.Environment.NewLine}", null)); } catch { }
      }
      finally
      {
        queue.CompleteAdding();
      }
    }) { IsBackground = true, Name = "GameAnalyzeProducer" };
    producer.Start();

    // Consumer (this thread): display each rendered position; pause after each (except the last) in pause mode.
    int displayed = 0;
    try
    {
      foreach ((string output, string positionCommand) in queue.GetConsumingEnumerable())
      {
        OutStream.Write(output);
        OutStream.Flush();
        displayed++;

        if (gameAnalyzeCancelRequested)
        {
          break;
        }

        bool isLast = displayed >= total;
        if (gameAnalyzePauseEnabled && !isLast)
        {
          OutStream.WriteLine();
          OutStream.WriteLine($"[game-analyze paused after position {displayed}/{total} — press <Enter> for next, Ctrl-C to end]");
          OutStream.Flush();

          int sig = WaitHandle.WaitAny(new WaitHandle[] { gameAnalyzeEnterEvent, gameAnalyzeCancelEvent.WaitHandle });
          if (sig == 1 || gameAnalyzeCancelRequested)
          {
            break;
          }
        }
      }
    }
    finally
    {
      // Ensure the producer stops and is not left blocked on a full queue (drain anything pending).
      gameAnalyzeCancelRequested = true;
      try { foreach (var _ in queue.GetConsumingEnumerable()) { } } catch { }
      producer.Join(3000);

      gameAnalyzeStopKeyMonitor = true;
      keyThread?.Join(500);

      Console.CancelKeyPress -= ctrlCHandler;
      queue.Dispose();
      gameAnalyzeEnterEvent.Dispose();
      gameAnalyzeCancelEvent.Dispose();
      gameAnalyzeEnterEvent = null;
      gameAnalyzeCancelEvent = null;
    }

    if (displayed < total)
    {
      UCIWriteLine();
      UCIWriteLine($"game-analyze: ended after {displayed} of {total} positions.");
    }
  }


  /// <summary>
  /// Computes the analysis for a single position in a range and renders all of its output (header,
  /// search progress, and dump) into a string (rather than printing it directly), so it can be
  /// displayed atomically by the consumer when the user is ready for it.
  /// </summary>
  private string RenderGameAnalyzePosition(Game game, int index, int total, int plyIndex, int searchMs, out string positionCommand)
  {
    positionCommand = null;

    StringWriter raw = new StringWriter();
    TextWriter captureWriter = TextWriter.Synchronized(raw);
    gameAnalyzeCaptureWriter = captureWriter;
    try
    {
      BuildGameAnalyzePositionForPly(game, plyIndex, out PositionWithHistory pwh, out positionCommand);
      Position finalPos = pwh.FinalPosition;

      UCIWriteLine();
      UCIWriteLine($"=== game-analyze [{index + 1}/{total}] {game.PlayerWhite} vs {game.PlayerBlack}, {DescribeGameAnalyzePosition(finalPos)} ===");
      UCIWriteLine(positionCommand);
      UCIWriteLine("Final FEN: " + finalPos.FEN);
      UCIWriteLine();

      ProcessPosition(positionCommand);
      taskSearchCurrentlyExecuting = ProcessGo($"go movetime {searchMs}");
      taskSearchCurrentlyExecuting.Wait();

      if (CeresEngine?.Search?.Manager != null)
      {
        CeresEngine.Search.Manager.DumpFullInfo(lastSearchResult, captureWriter, "game-analyze");
      }
    }
    finally
    {
      gameAnalyzeCaptureWriter = null;
    }

    return raw.ToString();
  }


  /// <summary>
  /// Requests the analysis to end: sets the cancel flag, wakes any paused consumer, and stops the
  /// search currently in progress.
  /// </summary>
  private void RequestGameAnalyzeCancel()
  {
    gameAnalyzeCancelRequested = true;
    gameAnalyzeCancelEvent?.Set();
    RequestGameAnalyzeStopCurrentSearch();
  }


  /// <summary>
  /// Asks the currently-executing search (if any) to stop as soon as possible.
  /// </summary>
  private void RequestGameAnalyzeStopCurrentSearch()
  {
    try
    {
      var manager = CeresEngine?.Search?.Manager;
      if (manager != null)
      {
        manager.ExternalStopRequested = true;
      }
    }
    catch
    {
      // Best effort; the cancel flag will stop the loop before the next position regardless.
    }
  }


  /// <summary>
  /// Background loop (interactive consoles only) that watches for Ctrl-P (toggle pause mode),
  /// Enter (advance past a pause), and Ctrl-C (end analysis, as a backup to Console.CancelKeyPress).
  /// Polls Console.KeyAvailable so it can be stopped promptly without blocking on ReadKey.
  /// </summary>
  private void GameAnalyzeKeyMonitorLoop()
  {
    while (!gameAnalyzeStopKeyMonitor)
    {
      try
      {
        if (!Console.KeyAvailable)
        {
          Thread.Sleep(40);
          continue;
        }

        ConsoleKeyInfo k = Console.ReadKey(intercept: true);
        bool ctrl = (k.Modifiers & ConsoleModifiers.Control) != 0;
        // Detect by key+modifier, with a fallback on the raw control character because some
        // terminals (notably on Linux) deliver the control char without setting the modifier.
        bool isCtrlP = (ctrl && k.Key == ConsoleKey.P) || k.KeyChar == '\u0010';
        bool isCtrlC = (ctrl && k.Key == ConsoleKey.C) || k.KeyChar == '\u0003';
        bool isEnter = k.Key == ConsoleKey.Enter || k.KeyChar == '\r' || k.KeyChar == '\n';

        if (isCtrlP)
        {
          bool nowEnabled = !gameAnalyzePauseEnabled;
          gameAnalyzePauseEnabled = nowEnabled;
          OutStream.WriteLine();
          OutStream.WriteLine(nowEnabled
            ? "[game-analyze: PAUSE mode ON — will wait for <Enter> after each position]"
            : "[game-analyze: PAUSE mode OFF]");
          OutStream.Flush();
          if (!nowEnabled)
          {
            gameAnalyzeEnterEvent?.Set();   // release any pending pause wait so analysis resumes
          }
        }
        else if (isCtrlC)
        {
          RequestGameAnalyzeCancel();
        }
        else if (isEnter)
        {
          gameAnalyzeEnterEvent?.Set();
        }
      }
      catch (Exception)
      {
        // A console quirk (e.g. an input-mode change) -- back off briefly and continue.
        Thread.Sleep(200);
      }
    }
  }


  /// <summary>
  /// Parses a human-friendly time specification into milliseconds.
  /// Supported suffixes: "ms" (milliseconds), "s" (seconds), "m" (minutes).
  /// A bare number is interpreted as seconds. Returns null if unparseable.
  /// </summary>
  internal static int? ParseAnalyzeTimeMs(string spec)
  {
    if (string.IsNullOrWhiteSpace(spec))
    {
      return null;
    }

    spec = spec.Trim().ToLowerInvariant();

    double multiplierToMs;
    string numberPart;
    if (spec.EndsWith("ms"))
    {
      multiplierToMs = 1;
      numberPart = spec[..^2];
    }
    else if (spec.EndsWith("s"))
    {
      multiplierToMs = 1000;
      numberPart = spec[..^1];
    }
    else if (spec.EndsWith("m"))
    {
      multiplierToMs = 60_000;
      numberPart = spec[..^1];
    }
    else
    {
      // Bare number: interpret as seconds.
      multiplierToMs = 1000;
      numberPart = spec;
    }

    if (!double.TryParse(numberPart, NumberStyles.Any, CultureInfo.InvariantCulture, out double value) || value <= 0)
    {
      return null;
    }

    return (int)Math.Round(value * multiplierToMs);
  }
}
