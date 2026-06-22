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
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;

using Ceres.MCGS.Analysis;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Utils;
using Ceres.MCGS.Visualization.AnalysisGraph;

#endregion

namespace Ceres.Features.TCEC;

/// <summary>
/// Live monitor that follows the current game on the TCEC broadcast feed and runs a
/// continuous Ceres MCGS analysis on it.
///
///   engine == null : feed-only mode. Fetches the latest TCEC position and prints a
///                     one-shot header + move summary, then returns.
///
///   engine != null : analysis mode. Searches the current TCEC position indefinitely,
///                     refreshing the console every ~1 seconds. A background poller
///                     watches the feed; when a new move is played the analysis advances
///                     to the new position reusing the existing search graph (continuation)
///                     rather than restarting. To keep the header at the top of the screen
///                     the continual display is compact and has two modes (toggled live):
///                       m = candidate moves off the root      p = principal variation
///                     Two on-demand commands run an analysis then hold until Esc:
///                       g = analysis graph (prompts for a detail level)   r = revalue-root
///                     q or Esc ends the monitor and returns to the caller (UCI prompt).
/// </summary>
public static class TCECMonitor
{
  const double DUMP_INTERVAL_SECONDS = 5.0;
  const int FEED_POLL_SECONDS = 2;
  const int MAX_PV_PLIES = 30;
  const int EVAL_LABEL_WIDTH = 54;

  enum DisplayMode { Moves = 0, PrincipalVariation = 1, Info = 2 }
  enum PendingAction { None = 0, Graph = 1, Revalue = 2 }
  enum MonitorKey { None, Quit, ModeMoves, ModePV, ModeInfo, Graph, Revalue }

  /// <summary>
  /// Entry point. Invoked from the MCGS "tcec" UCI command and from MCGSTest's "TCEC" argument.
  /// </summary>
  public static void Run(GameEngineCeresMCGSInProcess engine = null)
  {
    if (engine == null)
    {
      RunFeedOnly();
    }
    else
    {
      RunWithEngine(engine);
    }
  }


  // ===========================================================================
  //  Shared cross-thread control state (poller thread <-> search thread <-> driver)
  // ===========================================================================

  sealed class Shared
  {
    public volatile bool QuitRequested;
    public volatile bool UpdateAvailable;
    public volatile int Mode = (int)DisplayMode.Moves;
    public volatile int Action = (int)PendingAction.None;

    readonly object gate = new();
    TCECLiveSnapshot pending;
    bool pendingGameChange;
    string currentGameKey = "";
    int currentPly = -1;

    public void SetCurrent(TCECLiveSnapshot s)
    {
      lock (gate)
      {
        currentGameKey = s.GameKey ?? "";
        currentPly = s.Moves?.Count ?? 0;
      }
    }

    public (string gameKey, int ply) GetCurrent()
    {
      lock (gate)
      {
        return (currentGameKey, currentPly);
      }
    }

    public void SetPending(TCECLiveSnapshot s, bool gameChange)
    {
      lock (gate)
      {
        pending = s;
        pendingGameChange = gameChange;
      }
      UpdateAvailable = true;
    }

    public bool TryTakePending(out TCECLiveSnapshot s, out bool gameChange)
    {
      lock (gate)
      {
        if (!UpdateAvailable)
        {
          s = default;
          gameChange = false;
          return false;
        }
        s = pending;
        gameChange = pendingGameChange;
        UpdateAvailable = false;
        return true;
      }
    }
  }


  // ===========================================================================
  //  Engine analysis mode
  // ===========================================================================

  static void RunWithEngine(GameEngineCeresMCGSInProcess engine)
  {
    bool savedChess960 = MGPositionConstants.IsChess960;
    Shared shared = new();
    using CancellationTokenSource pollerCts = new();
    Task pollerTask = null;

    try
    {
      Console.Clear();
      PrintInstructions();

      engine.ResetGame("TCEC");   // clean graph for this monitoring session

      // --- obtain the first searchable position ---
      TCECLiveSnapshot current = default;
      PositionWithHistory curPos = null;
      DateTime lastMoveSeenAt = DateTime.Now;
      bool haveInitial = false;
      while (!haveInitial && !shared.QuitRequested)
      {
        if (PollKey() == MonitorKey.Quit)
        {
          shared.QuitRequested = true;
          break;
        }
        Console.WriteLine("Fetching latest TCEC position ...");
        TCECLiveSnapshot? maybe = TCECLiveFeed.FetchOnce(pollerCts.Token, onWarn: ConsoleWarn);
        if (maybe != null)
        {
          PositionWithHistory p = ResolvePosition(maybe.Value);
          if (p != null)
          {
            current = maybe.Value;
            curPos = p;
            haveInitial = true;
            break;
          }
        }
        WaitWithQuit(FEED_POLL_SECONDS, shared);
      }
      if (!haveInitial)
      {
        return;
      }

      shared.SetCurrent(current);
      pollerTask = Task.Run(() => PollLoop(shared, pollerCts.Token));

      // --- main search / advance loop ---
      while (!shared.QuitRequested)
      {
        ApplyChess960(current);

        DateTime searchStartedAt = DateTime.Now;
        TCECLiveSnapshot snapForCb = current;
        DateTime lastMoveSeen = lastMoveSeenAt;
        DateTime lastDump = DateTime.MinValue;   // force an immediate render once nodes exist

        void Callback(object ctx)
        {
          MCGSManager mgr = (MCGSManager)ctx;

          if (shared.QuitRequested || shared.UpdateAvailable
              || shared.Action != (int)PendingAction.None)
          {
            mgr.ExternalStopRequested = true;
            return;
          }

          switch (PollKey())
          {
            case MonitorKey.Quit:
              shared.QuitRequested = true;
              mgr.ExternalStopRequested = true;
              return;
            case MonitorKey.Graph:
              shared.Action = (int)PendingAction.Graph;
              mgr.ExternalStopRequested = true;
              return;
            case MonitorKey.Revalue:
              shared.Action = (int)PendingAction.Revalue;
              mgr.ExternalStopRequested = true;
              return;
            case MonitorKey.ModeMoves:
              shared.Mode = (int)DisplayMode.Moves;
              lastDump = DateTime.MinValue;   // re-render promptly
              break;
            case MonitorKey.ModePV:
              shared.Mode = (int)DisplayMode.PrincipalVariation;
              lastDump = DateTime.MinValue;
              break;
            case MonitorKey.ModeInfo:
              shared.Mode = (int)DisplayMode.Info;
              lastDump = DateTime.MinValue;
              break;
            default:
              break;
          }

          long n = SafeRootN(mgr);
          DateTime now = DateTime.Now;
          if (n > 0 && (now - lastDump).TotalSeconds >= DUMP_INTERVAL_SECONDS)
          {
            lastDump = now;
            RenderScreen(mgr, in snapForCb, lastMoveSeen, searchStartedAt, (DisplayMode)shared.Mode);
          }
        }

        try
        {
          // Continuation: no ResetGame between same-game positions, so the engine
          // reuses the existing graph/subtree (GraphReuseManager.TryReuseGraph).
          engine.SearchCeres(curPos,
                             SearchLimit.NodesPerMove(GraphStore.MAX_NODES),
                             callback: Callback);
        }
        catch (Exception ex)
        {
          ConsoleWarn("search error: " + ex.Message);
        }

        if (shared.QuitRequested)
        {
          break;
        }

        // On-demand analyses run on the now-stopped tree, then hold until Esc.
        PendingAction action = (PendingAction)shared.Action;
        shared.Action = (int)PendingAction.None;
        if (action == PendingAction.Graph)
        {
          RunGraphCommand(engine);
          WaitForEsc(shared);
          continue;   // resume search on the same position (graph reused)
        }
        if (action == PendingAction.Revalue)
        {
          RunRevalueCommand(engine);
          WaitForEsc(shared);
          continue;
        }

        // Otherwise advance to a newly played move if one arrived.
        if (shared.TryTakePending(out TCECLiveSnapshot newSnap, out bool gameChange))
        {
          if (gameChange)
          {
            engine.ResetGame("TCEC");   // different game: cannot reuse the graph
          }
          if ((newSnap.Moves?.Count ?? 0) != (current.Moves?.Count ?? 0))
          {
            lastMoveSeenAt = DateTime.Now;
          }
          PositionWithHistory p = ResolvePosition(newSnap);
          if (p != null)
          {
            current = newSnap;
            curPos = p;
            shared.SetCurrent(current);
          }
        }
      }
    }
    finally
    {
      pollerCts.Cancel();
      try
      {
        pollerTask?.Wait(2000);
      }
      catch
      {
        // ignore poller shutdown errors
      }
      MGPositionConstants.IsChess960 = savedChess960;
      Console.ResetColor();
      Console.WriteLine();
      Console.WriteLine("TCEC live analysis ended.");
    }
  }


  /// <summary>
  /// Background loop: polls the TCEC feed and signals the driver when the live game
  /// advances to a new move (or switches to a different game).
  /// </summary>
  static void PollLoop(Shared shared, CancellationToken ct)
  {
    try
    {
      while (!ct.IsCancellationRequested && !shared.QuitRequested)
      {
        if (ct.WaitHandle.WaitOne(TimeSpan.FromSeconds(FEED_POLL_SECONDS)))
        {
          break;   // cancellation signaled
        }

        TCECLiveSnapshot? maybe = TCECLiveFeed.FetchOnce(ct);
        if (maybe == null)
        {
          continue;
        }

        TCECLiveSnapshot s = maybe.Value;
        (string curKey, int curPly) = shared.GetCurrent();
        int newPly = s.Moves?.Count ?? 0;
        bool gameChange = !string.Equals(s.GameKey ?? "", curKey, StringComparison.Ordinal);

        if (gameChange)
        {
          shared.SetPending(s, true);
        }
        else if (newPly > curPly)
        {
          shared.SetPending(s, false);
        }
      }
    }
    catch (OperationCanceledException)
    {
      // expected on shutdown
    }
    catch (Exception ex)
    {
      ConsoleWarn("poll error: " + ex.Message);
    }
  }


  static PositionWithHistory ResolvePosition(in TCECLiveSnapshot snap)
  {
    if (snap.History != null)
    {
      return snap.History;
    }
    if (snap.Moves != null && snap.Moves.Count > 0)
    {
      string lastFen = snap.Moves[snap.Moves.Count - 1].FENAfterRaw;
      if (!string.IsNullOrEmpty(lastFen))
      {
        try
        {
          return PositionWithHistory.FromFENAndMovesUCI(lastFen);
        }
        catch
        {
          // fall through
        }
      }
    }
    return null;
  }


  static void ApplyChess960(in TCECLiveSnapshot snap)
  {
    MGPositionConstants.IsChess960 = string.Equals(snap.Variant, "fischerandom",
                                                   StringComparison.OrdinalIgnoreCase);
  }


  static long SafeRootN(MCGSManager mgr)
  {
    try
    {
      return mgr.Engine.SearchRootNode.NodeRef.N;
    }
    catch
    {
      return 0;
    }
  }


  // ===========================================================================
  //  On-demand analyses (graph / revalue-root): run on the stopped tree, hold until Esc
  // ===========================================================================

  static void RunGraphCommand(GameEngineCeresMCGSInProcess engine)
  {
    ClearConsole();
    Console.Write("Analysis graph detail level (e.g. 4, blank = default): ");
    string line = SafeReadLine();
    string optionsStr = string.IsNullOrWhiteSpace(line) ? null : line.Trim();
    try
    {
      AnalysisGraphOptions options = AnalysisGraphOptions.FromString(optionsStr);
      AnalysisGraphGenerator generator = new(engine.Search, options);
      string path = generator.Write(launchWithBrowser: true);
      Console.WriteLine("Analysis graph written: " + path);
    }
    catch (Exception ex)
    {
      ConsoleWarn("graph command failed: " + ex.Message);
    }
    Console.WriteLine();
    PrintResumeHint();
  }


  static void RunRevalueCommand(GameEngineCeresMCGSInProcess engine)
  {
    ClearConsole();
    Console.WriteLine("Running revalue-root analysis (deep rollouts from the visit frontier)...");
    Console.WriteLine();
    try
    {
      MCGSManager mgr = engine.Search.Manager;
      const int ROUNDS_PER_STAGE = 20;
      PrincipalRevaluationResult result = PrincipalRevaluation.Run(mgr, ROUNDS_PER_STAGE);
      PrincipalRevaluationDumper.DumpToConsole(result, engine.Search.BestMove);
    }
    catch (Exception ex)
    {
      ConsoleWarn("revalue-root command failed: " + ex.Message);
    }
    Console.WriteLine();
    PrintResumeHint();
  }


  static void PrintResumeHint()
  {
    ConsoleColor prev = Console.ForegroundColor;
    try
    {
      Console.ForegroundColor = ConsoleColor.Yellow;
      Console.WriteLine(">>> press Esc to resume live analysis (q to quit) <<<");
    }
    finally
    {
      Console.ForegroundColor = prev;
    }
  }


  /// <summary>
  /// Blocks (without clearing/refreshing) until the user presses Esc to resume, or q to quit.
  /// </summary>
  static void WaitForEsc(Shared shared)
  {
    while (!shared.QuitRequested)
    {
      try
      {
        if (Console.KeyAvailable)
        {
          ConsoleKey k = Console.ReadKey(intercept: true).Key;
          if (k == ConsoleKey.Escape)
          {
            return;
          }
          if (k == ConsoleKey.Q)
          {
            shared.QuitRequested = true;
            return;
          }
        }
        else
        {
          Thread.Sleep(50);
        }
      }
      catch (InvalidOperationException)
      {
        return;   // console input redirected; cannot wait for a key
      }
    }
  }


  // ===========================================================================
  //  Screen rendering (clear -> instructions -> header -> evals -> mode content)
  // ===========================================================================

  static void RenderScreen(MCGSManager mgr, in TCECLiveSnapshot snap,
                           DateTime lastMoveSeenAt, DateTime searchStartedAt, DisplayMode mode)
  {
    ClearConsole();
    PrintInstructions();

    BestMoveInfoMCGS bmi = null;
    try
    {
      bmi = mgr.GetBestMove(out _, out _, out _, isFinalBestMoveCalc: false);
    }
    catch
    {
      bmi = null;
    }

    long nodes = SafeRootN(mgr);
    PrintHeader(in snap, nodes, lastMoveSeenAt, searchStartedAt);
    PrintReuseStatus(mgr);
    PrintEvalPerspectives(in snap, mgr, bmi);

    // The continual section reuses MCGSManager.DumpFullInfo's exact format, restricted to the
    // single section selected by the current mode (m = moves, p = PV, i = info).
    string modeLabel = mode switch
    {
      DisplayMode.Moves => "candidate moves off root [m]",
      DisplayMode.PrincipalVariation => "principal variation [p]",
      _ => "search info [i]",
    };
    ConsoleColor prevSep = Console.ForegroundColor;
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    Console.WriteLine("  ---- " + modeLabel + "   (switch: m / p / i) ----");
    Console.ForegroundColor = prevSep;

    if (bmi != null)
    {
      DumpFullInfoSections section = mode switch
      {
        DisplayMode.Moves => DumpFullInfoSections.Moves,
        DisplayMode.PrincipalVariation => DumpFullInfoSections.PrincipalVariation,
        _ => DumpFullInfoSections.Info,
      };
      try
      {
        mgr.DumpFullInfo(bmi, mgr.Engine.SearchRootNode, default, Console.Out, null, section);
      }
      catch (Exception ex)
      {
        Console.WriteLine("(dump unavailable: " + ex.Message + ")");
      }
    }
    else
    {
      Console.WriteLine("    (search warming up...)");
    }
  }


  static void PrintInstructions()
  {
    ConsoleColor prev = Console.ForegroundColor;
    try
    {
      Console.ForegroundColor = ConsoleColor.Yellow;
      Console.WriteLine("================================ TCEC LIVE ANALYSIS ================================");
      Console.WriteLine("  Ceres analyzes the live TCEC game and AUTO-FOLLOWS each move (graph reused). 5s refresh.");
      Console.WriteLine("  View:  [m] candidate moves   [p] principal variation   [i] search info");
      Console.WriteLine("  Run :  [g] analysis graph (prompts level)    [r] revalue-root   (then Esc to resume)");
      Console.WriteLine("  [q] or [Esc] = quit monitor (return to UCI prompt)");
      Console.WriteLine("===================================================================================");
      Console.WriteLine();
    }
    finally
    {
      Console.ForegroundColor = prev;
    }
  }


  /// <summary>
  /// Direct evidence of whether the engine reused its prior search graph for this position.
  /// RootNWhenSearchStarted is the visit count the (continuation) search root already had on entry
  /// (&gt; 0 means a subtree was inherited from the previous move's search); the search-root depth below
  /// the graph root grows by one each successful continuation.
  /// </summary>
  static void PrintReuseStatus(MCGSManager mgr)
  {
    try
    {
      long inherited = mgr.RootNWhenSearchStarted;
      int pliesBelow = mgr.Engine.SearchRootPathFromGraphRoot?.Length ?? 0;
      ConsoleColor prev = Console.ForegroundColor;
      Console.ForegroundColor = inherited > 0 ? ConsoleColor.Green : ConsoleColor.DarkGray;
      Console.WriteLine("  Graph reuse: " + (inherited > 0
        ? FormatBig(inherited) + " nodes inherited (search root is +" + pliesBelow + " plies below graph root)"
        : "fresh graph this move (no subtree inherited)"));
      Console.ForegroundColor = prev;
    }
    catch
    {
      // best-effort indicator only
    }
  }


  static void PrintHeader(in TCECLiveSnapshot snap, long nodes,
                          DateTime lastMoveSeenAt, DateTime searchStartedAt)
  {
    int plies = snap.Moves?.Count ?? 0;
    bool whiteToMove = (plies % 2) == 0;
    int moveNumber = (plies / 2) + 1;

    string lastMoveDesc = "(none)";
    if (plies > 0)
    {
      TCECMoveInfo last = snap.Moves[plies - 1];
      string num = last.IsWhite
        ? last.MoveNumber.ToString(CultureInfo.InvariantCulture) + "."
        : last.MoveNumber.ToString(CultureInfo.InvariantCulture) + "...";
      lastMoveDesc = num + " " + Safe(last.SAN) + "  eval " + FormatEval(last.EvalPawns) + " (White POV)"
                     + (last.IsBookMove ? "  (book)" : "");
    }

    double sinceLastMove = (DateTime.Now - lastMoveSeenAt).TotalSeconds;
    double searchElapsed = (DateTime.Now - searchStartedAt).TotalSeconds;
    long whiteClock = ClockForColor(in snap, white: true);
    long blackClock = ClockForColor(in snap, white: false);

    ConsoleColor prev = Console.ForegroundColor;
    try
    {
      Console.WriteLine("  " + Safe(snap.Event) + "   Round " + Safe(snap.Round)
                        + "   " + Safe(snap.Variant) + "   TC " + Safe(snap.TimeControl));
      Console.WriteLine("  White: " + PadName(snap.White.Name) + " (" + snap.White.Elo + ")  clock "
                        + FormatClock(whiteClock) + (whiteToMove ? "  <= to move" : ""));
      Console.WriteLine("  Black: " + PadName(snap.Black.Name) + " (" + snap.Black.Elo + ")  clock "
                        + FormatClock(blackClock) + (whiteToMove ? "" : "  <= to move"));

      // Most prominent line: how long since the last move appeared on the feed.
      Console.ForegroundColor = ConsoleColor.Cyan;
      Console.WriteLine("  >>> Move " + moveNumber + ", " + (whiteToMove ? "WHITE" : "BLACK")
                        + " to move    SINCE LAST MOVE: " + FormatSeconds(sinceLastMove)
                        + "    (analyzing " + FormatSeconds(searchElapsed) + ", " + FormatBig(nodes) + " nodes)");
      Console.ResetColor();

      Console.WriteLine("  TCEC last move: " + lastMoveDesc);
    }
    finally
    {
      Console.ForegroundColor = prev;
    }
  }


  /// <summary>
  /// Prints evals for the current position from the perspectives the TCEC feed makes available
  /// (the engine that just moved) plus the local Ceres search. NOTE: the side-to-move engine's
  /// own live eval is NOT broadcast in live.json, so that perspective is necessarily omitted.
  /// </summary>
  static void PrintEvalPerspectives(in TCECLiveSnapshot snap, MCGSManager mgr, BestMoveInfoMCGS bmi)
  {
    int plies = snap.Moves?.Count ?? 0;
    bool whiteToMove = (plies % 2) == 0;
    string stmName = TruncName(whiteToMove ? snap.White.Name : snap.Black.Name);

    ConsoleColor prev = Console.ForegroundColor;
    try
    {
      Console.ForegroundColor = ConsoleColor.Magenta;
      Console.WriteLine();
      Console.WriteLine("  Evals for current position (perspective: side to move = " + stmName + ")");
      Console.ResetColor();

      // All three PVs are trimmed to begin at the CURRENT position so they line up: a TCEC move's PV
      // starts with that move plus any plies already played since, so we skip (plies - itsIndex) moves.

      // 1. Opponent (the engine that just moved): fresh eval + PV of this position.
      if (plies > 0)
      {
        TCECMoveInfo last = snap.Moves[plies - 1];
        if (!double.IsNaN(last.EvalPawns))
        {
          string label = "(TCEC " + TruncName(last.IsWhite ? snap.White.Name : snap.Black.Name) + ", just moved)";
          EvalLine(ToStmCp(last.EvalPawns, whiteToMove), label,
                   CapPVList(last.PVMoves, MAX_PV_PLIES, plies - (plies - 1)));
        }
      }

      // 2. Local Ceres search: fresh eval + PV (rendered in SAN, the notation TCEC uses).
      if (mgr != null && bmi != null)
      {
        Position? rootPos = RootPosition(in snap);
        string pv = rootPos.HasValue ? CeresPVStringSAN(mgr, rootPos.Value, MAX_PV_PLIES) : "";
        EvalLine(EncodedEvalLogistic.WinLossToCentipawn(bmi.QOfBest), "(Ceres local)", pv);
      }

      // 3. On-move engine's OWN most-recent eval (stale, from its previous move) - shown gray.
      int staleIdx = LastMoveIndexForColor(in snap, whiteToMove);
      if (staleIdx >= 0 && !double.IsNaN(snap.Moves[staleIdx].EvalPawns))
      {
        TCECMoveInfo stale = snap.Moves[staleIdx];
        Console.ForegroundColor = ConsoleColor.DarkGray;
        EvalLine(ToStmCp(stale.EvalPawns, whiteToMove),
                 "(" + stmName + ", stale from move " + stale.MoveNumber + ")",
                 CapPVList(stale.PVMoves, MAX_PV_PLIES, plies - staleIdx));
        Console.ResetColor();
      }
    }
    finally
    {
      Console.ForegroundColor = prev;
    }
  }


  static void EvalLine(float cp, string label, string pv)
  {
    Console.WriteLine("    " + PadLeft(FormatCp(cp), 6) + " cp  " + PadRight(label, EVAL_LABEL_WIDTH) + pv);
  }


  /// <summary>Converts a TCEC eval (pawns, always White POV) to centipawns from the side-to-move POV.</summary>
  static float ToStmCp(double wvWhitePawns, bool whiteToMove)
    => (float)((whiteToMove ? wvWhitePawns : -wvWhitePawns) * 100.0);


  /// <summary>Index in snap.Moves of the most recent move played by the given color, or -1.</summary>
  static int LastMoveIndexForColor(in TCECLiveSnapshot snap, bool white)
  {
    if (snap.Moves == null)
    {
      return -1;
    }
    for (int i = snap.Moves.Count - 1; i >= 0; i--)
    {
      if (snap.Moves[i].IsWhite == white)
      {
        return i;
      }
    }
    return -1;
  }


  /// <summary>
  /// The local Ceres principal variation rendered in SAN (the notation TCEC uses), built by replaying
  /// the PV moves from the given root position. Capped at maxPlies.
  /// </summary>
  static string CeresPVStringSAN(MCGSManager mgr, Position rootPos, int maxPlies)
  {
    try
    {
      GNode root = mgr.Engine.SearchRootNode;
      SearchPrincipalVariationMCGS pv = new(mgr, root, default, startFromRoot: true, minN: 1);
      Position pos = rootPos;
      List<string> sans = new();
      bool truncated = false;
      foreach (GNodeAndOptionalEdge ne in pv.Nodes)
      {
        if (!ne.HasEdge)
        {
          continue;
        }
        if (sans.Count >= maxPlies)
        {
          truncated = true;
          break;
        }
        Move m = MGMoveConverter.ToMove(ne.Edge.MoveMG);
        sans.Add(m.ToSAN(in pos));
        pos = pos.AfterMove(m);
      }
      return string.Join(' ', sans) + (truncated ? " ..." : "");
    }
    catch
    {
      return "";
    }
  }


  static long ClockForColor(in TCECLiveSnapshot snap, bool white)
  {
    if (snap.Moves == null)
    {
      return -1;
    }
    for (int i = snap.Moves.Count - 1; i >= 0; i--)
    {
      if (snap.Moves[i].IsWhite == white)
      {
        return snap.Moves[i].TimeLeftMs;
      }
    }
    return -1;
  }


  // ===========================================================================
  //  Feed-only mode (engine == null)
  // ===========================================================================

  static void RunFeedOnly()
  {
    bool savedChess960 = MGPositionConstants.IsChess960;
    using CancellationTokenSource cts = new();
    try
    {
      Console.WriteLine("Fetching latest TCEC position from " + TCECLiveFeed.DEFAULT_URL + " ...");
      TCECLiveSnapshot? maybe = TCECLiveFeed.FetchLatestSnapshot(cts.Token, onWarn: ConsoleWarn);
      if (maybe == null)
      {
        Console.WriteLine("No TCEC snapshot retrieved.");
        return;
      }

      TCECLiveSnapshot snap = maybe.Value;
      PrintInstructions();
      PrintHeader(in snap, 0, snap.FetchedAtUtc.ToLocalTime(), DateTime.Now);
      PrintEvalPerspectives(in snap, null, null);
      DumpFeedSummary(in snap);
    }
    finally
    {
      MGPositionConstants.IsChess960 = savedChess960;
      Console.ResetColor();
    }
  }


  static void DumpFeedSummary(in TCECLiveSnapshot snap)
  {
    int n = snap.Moves?.Count ?? 0;
    Console.WriteLine("  Moves played: " + n);
    if (n == 0)
    {
      return;
    }

    void DumpPly(TCECMoveInfo m)
    {
      string num = m.IsWhite
        ? string.Format(CultureInfo.InvariantCulture, "{0,3}.   ", m.MoveNumber)
        : string.Format(CultureInfo.InvariantCulture, "{0,3}... ", m.MoveNumber);
      string book = m.IsBookMove ? "  (book)" : "";
      Console.WriteLine("    " + num + (m.IsWhite ? "W" : "B") + "  "
                        + PadRight(Safe(m.SAN), 8) + " " + FormatEval(m.EvalPawns)
                        + "   nodes " + FormatBig(m.Nodes) + book);
    }

    int head = Math.Min(5, n);
    for (int i = 0; i < head; i++)
    {
      DumpPly(snap.Moves[i]);
    }
    if (n > 10)
    {
      Console.WriteLine("    ...");
    }
    for (int i = Math.Max(head, n - 5); i < n; i++)
    {
      DumpPly(snap.Moves[i]);
    }
  }


  // ===========================================================================
  //  Console / keyboard helpers
  // ===========================================================================

  /// <summary>
  /// Drains pending keystrokes and returns the last recognized control key.
  /// </summary>
  static MonitorKey PollKey()
  {
    try
    {
      MonitorKey result = MonitorKey.None;
      while (Console.KeyAvailable)
      {
        switch (Console.ReadKey(intercept: true).Key)
        {
          case ConsoleKey.Q:
          case ConsoleKey.Escape:
            result = MonitorKey.Quit;
            break;
          case ConsoleKey.M:
            result = MonitorKey.ModeMoves;
            break;
          case ConsoleKey.P:
            result = MonitorKey.ModePV;
            break;
          case ConsoleKey.I:
            result = MonitorKey.ModeInfo;
            break;
          case ConsoleKey.G:
            result = MonitorKey.Graph;
            break;
          case ConsoleKey.R:
            result = MonitorKey.Revalue;
            break;
        }
      }
      return result;
    }
    catch (InvalidOperationException)
    {
      return MonitorKey.None;   // console input redirected
    }
  }


  static string SafeReadLine()
  {
    try
    {
      return Console.ReadLine();
    }
    catch
    {
      return null;
    }
  }


  static void WaitWithQuit(int seconds, Shared shared)
  {
    int elapsedMs = 0;
    while (elapsedMs < seconds * 1000)
    {
      if (shared.QuitRequested || PollKey() == MonitorKey.Quit)
      {
        shared.QuitRequested = true;
        return;
      }
      Thread.Sleep(100);
      elapsedMs += 100;
    }
  }


  static void ClearConsole()
  {
    try
    {
      Console.Clear();
    }
    catch (IOException)
    {
      Console.WriteLine();
      Console.WriteLine("-----------------------------------------------------------------------------------");
    }
  }


  static void ConsoleWarn(string msg)
  {
    ConsoleColor prev = Console.ForegroundColor;
    try
    {
      Console.ForegroundColor = ConsoleColor.DarkYellow;
      Console.WriteLine("  [warn] " + msg);
    }
    finally
    {
      Console.ForegroundColor = prev;
    }
  }


  // ===========================================================================
  //  Formatting helpers
  // ===========================================================================

  static string Safe(string s) => string.IsNullOrEmpty(s) ? "?" : s;

  static string TruncName(string n)
  {
    string s = string.IsNullOrEmpty(n) ? "?" : n;
    return s.Length > 30 ? s.Substring(0, 30) : s;
  }

  static string PadName(string n) => TruncName(n).PadRight(30);

  static string PadRight(string s, int width)
  {
    if (string.IsNullOrEmpty(s))
    {
      return new string(' ', width);
    }
    return s.Length >= width ? s : s.PadRight(width);
  }

  static string PadLeft(string s, int width)
  {
    s ??= "";
    return s.Length >= width ? s : s.PadLeft(width);
  }

  /// <summary>Caps a move list to at most maxPlies, skipping the first <paramref name="skip"/> moves.</summary>
  static string CapPVList(IReadOnlyList<string> moves, int maxPlies, int skip)
  {
    if (moves == null || moves.Count == 0)
    {
      return "";
    }
    int start = Math.Max(0, Math.Min(skip, moves.Count));
    List<string> outMoves = new();
    for (int i = start; i < moves.Count && outMoves.Count < maxPlies; i++)
    {
      outMoves.Add(moves[i]);
    }
    string joined = string.Join(' ', outMoves);
    return (moves.Count - start) > maxPlies ? joined + " ..." : joined;
  }


  /// <summary>The current (search-root) position: the snapshot's full history if available, else the
  /// FEN after the last played move.</summary>
  static Position? RootPosition(in TCECLiveSnapshot snap)
  {
    if (snap.History != null)
    {
      return snap.History.FinalPosition;
    }
    if (snap.Moves != null && snap.Moves.Count > 0)
    {
      string fen = snap.Moves[snap.Moves.Count - 1].FENAfterRaw;
      if (!string.IsNullOrEmpty(fen))
      {
        try
        {
          return Position.FromFEN(fen);
        }
        catch
        {
          // fall through
        }
      }
    }
    return null;
  }

  static string FormatEval(double v)
  {
    if (double.IsNaN(v))
    {
      return "  --  ";
    }
    string sign = v > 0 ? "+" : (v < 0 ? "-" : " ");
    return sign + Math.Abs(v).ToString("F2", CultureInfo.InvariantCulture);
  }

  static string FormatCp(float cp)
  {
    if (float.IsNaN(cp))
    {
      return "--";
    }
    string sign = cp > 0 ? "+" : (cp < 0 ? "-" : " ");
    return sign + Math.Abs(cp).ToString("F0", CultureInfo.InvariantCulture);
  }

  static string FormatSeconds(double secs)
  {
    if (secs < 0)
    {
      secs = 0;
    }
    if (secs < 60)
    {
      return secs.ToString("F0", CultureInfo.InvariantCulture) + "s";
    }
    long total = (long)secs;
    return string.Format(CultureInfo.InvariantCulture, "{0}m{1:D2}s", total / 60, total % 60);
  }

  static string FormatClock(long ms)
  {
    if (ms < 0)
    {
      return "--:--";
    }
    long total = ms / 1000;
    long h = total / 3600;
    long m = (total % 3600) / 60;
    long s = total % 60;
    if (h > 0)
    {
      return string.Format(CultureInfo.InvariantCulture, "{0}:{1:D2}:{2:D2}", h, m, s);
    }
    return string.Format(CultureInfo.InvariantCulture, "{0}:{1:D2}", m, s);
  }

  static string FormatBig(long num)
  {
    if (num < 0)
    {
      return "-" + FormatBig(-num);
    }
    if (num >= 1_000_000_000L)
    {
      return (num / 1_000_000_000.0).ToString("F2", CultureInfo.InvariantCulture) + "G";
    }
    if (num >= 1_000_000L)
    {
      return (num / 1_000_000.0).ToString("F2", CultureInfo.InvariantCulture) + "M";
    }
    if (num >= 10_000L)
    {
      return (num / 1_000.0).ToString("F1", CultureInfo.InvariantCulture) + "k";
    }
    return num.ToString(CultureInfo.InvariantCulture);
  }
}
