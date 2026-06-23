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
using System.Threading;
using System.Threading.Tasks;

using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.SearchResultVerboseMoveInfo;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Helper that runs a single fixed-time Lc0 analysis on a position and streams the Lc0 UCI
  /// output to a writer, then shuts the Lc0 engine down. Used to implement the "game-analyze-lc0"
  /// command/verb in the MCGS UCI manager.
  ///
  /// This lives in Ceres.Features (which references Ceres.MCGS) and is injected into the MCGS
  /// UCI manager via a handler delegate (mirroring the TCEC monitor handler), since the MCGS
  /// UCI manager cannot reference the Lc0 engine classes directly.
  /// </summary>
  public static class GameAnalyzeLC0Runner
  {
    /// <summary>
    /// Launches an Lc0 engine configured from the supplied evaluator definition, analyzes the
    /// given position (FEN + moves) for the specified movetime, streams Lc0's UCI output to
    /// <paramref name="outWriter"/>, then shuts the engine down.
    /// </summary>
    /// <param name="evaluatorDef">network/device configuration (must reference a single LC0-type network)</param>
    /// <param name="fenAndMoves">position as "&lt;FEN&gt; moves m1 m2 ..." (no "position" prefix)</param>
    /// <param name="movetimeMs">analysis time in milliseconds</param>
    /// <param name="outWriter">writer to which UCI output is streamed</param>
    public static void Run(NNEvaluatorDef evaluatorDef, string fenAndMoves, int movetimeMs, TextWriter outWriter)
    {
      if (evaluatorDef == null)
      {
        outWriter.WriteLine("info string game-analyze-lc0: no network/device configured");
        return;
      }

      // Create the Lc0 engine from the current configuration (uses the default Lc0 option set:
      // UCI_ShowWDL, UCI_ShowEPS, nodes-as-playouts, backend/threads, etc.). verbose:true requests
      // per-move statistics, matching how the ANALYZE feature runs an Lc0 opponent.
      GameEngineLC0 engine;
      try
      {
        GameEngineDefLC0 def = new("LC0-analyze", evaluatorDef, forceDisableSmartPruning: false, verbose: true);
        engine = (GameEngineLC0)def.CreateEngine();
      }
      catch (Exception exc)
      {
        outWriter.WriteLine($"info string game-analyze-lc0: unable to launch Lc0 ({exc.Message}). "
                          + "Ensure the configured network is an LC0-type network and DirLC0Binaries/LC0ExeName are set.");
        return;
      }

      try
      {
        engine.LC0Engine.DoSearchPrepare();
        outWriter.WriteLine($"game-analyze-lc0: launched Lc0 (pid {engine.LC0Engine.ProcessID}); analyzing for {movetimeMs}ms ...");
        outWriter.WriteLine();

        // Run the (blocking) analysis on a background task so we can stream output as it arrives.
        SearchLimit limit = SearchLimit.SecondsPerMove(movetimeMs / 1000.0f);
        Task<VerboseMoveStats> analyzeTask = Task.Run(() => engine.LC0Engine.AnalyzePositionFromFENAndMoves(fenAndMoves, limit));

        // Stream the latest UCI info (PV) line as it updates.
        string lastEchoed = null;
        while (!analyzeTask.IsCompleted)
        {
          Thread.Sleep(300);
          string info = engine.LC0Engine.Runner.LastInfoString;
          if (info != null && info != lastEchoed)
          {
            outWriter.WriteLine(info);
            lastEchoed = info;
          }
        }
        analyzeTask.Wait();

        // Final info line, full per-move statistics, and best move.
        string finalInfo = engine.LC0Engine.Runner.LastInfoString;
        if (finalInfo != null && finalInfo != lastEchoed)
        {
          outWriter.WriteLine(finalInfo);
        }

        outWriter.WriteLine();
        try
        {
          analyzeTask.Result?.Dump();
        }
        catch (Exception)
        {
          // Per-move stats dump is best-effort; ignore parsing issues.
        }

        string bestMove = engine.LC0Engine.Runner.LastBestMove;
        if (bestMove != null)
        {
          outWriter.WriteLine(bestMove);
        }
        outWriter.WriteLine();
      }
      finally
      {
        // Close the Lc0 engine (UCI quit + process shutdown).
        engine.Dispose();
      }
    }
  }
}
