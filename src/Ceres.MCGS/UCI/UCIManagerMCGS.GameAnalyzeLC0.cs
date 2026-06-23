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

using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.MCGS.UCI;

/// <summary>
/// Implementation of the "game-analyze-lc0" feature: like "game-analyze", but the fixed-time
/// analysis is run by an Lc0 engine (configured from the current network/device), streaming
/// Lc0's UCI output to the console. When complete the Lc0 engine is closed and the position is
/// loaded into Ceres for further manual analysis — but NO automatic Ceres search is run.
///
/// The Lc0 engine classes live in Ceres.Features (which references Ceres.MCGS), so the actual
/// Lc0 run is performed by an injected handler (mirroring TCECMonitorHandler) rather than being
/// referenced directly here.
/// </summary>
public partial class UCIManagerMCGS
{
  /// <summary>
  /// Optional handler that runs a single fixed-time Lc0 analysis and streams its UCI output.
  /// Injected by the host (DispatchCommands) from Ceres.Features. The tuple carries the evaluator
  /// definition (network/device), the position as "&lt;FEN&gt; moves ...", the movetime in
  /// milliseconds, and the writer to stream Lc0 output to.
  /// </summary>
  public Action<(NNEvaluatorDef evaluatorDef, string fenAndMoves, int movetimeMs, TextWriter outWriter)> LC0AnalyzeHandler;


  /// <summary>
  /// Entry point from the UCI command loop (or the command-line verb startup hook) for
  /// "game-analyze-lc0". Parses any arguments present and interactively prompts for any missing.
  /// </summary>
  /// <param name="command">the full command line, e.g. "game-analyze-lc0 /path/game.pgn 105 10s"</param>
  internal void ProcessGameAnalyzeLC0Interactive(string command)
  {
    if (TryParseGameAnalyzeArgs(command, "game-analyze-lc0", out string pgnFile, out string moveSpec, out string timeSpec))
    {
      try
      {
        ProcessGameAnalyzeLC0(pgnFile, moveSpec, timeSpec);
      }
      catch (Exception exc)
      {
        UCIWriteLine($"info string game-analyze-lc0 failed: {exc.Message}");
      }
    }
  }


  /// <summary>
  /// Core of the game-analyze-lc0 feature: loads the PGN, navigates to the requested position
  /// (full move history), echoes the constructed position, runs a fixed-time Lc0 analysis
  /// (streaming Lc0 UCI output), closes Lc0, then loads the position into Ceres WITHOUT running a search.
  /// </summary>
  internal void ProcessGameAnalyzeLC0(string pgnFile, string moveSpec, string timeSpec)
  {
    // Validate / parse the time first (cheap, fail fast).
    int? searchMs = ParseAnalyzeTimeMs(timeSpec);
    if (searchMs == null || searchMs <= 0)
    {
      UCIWriteLine($"info string game-analyze-lc0: invalid time \"{timeSpec}\" (expected e.g. 10s, 500ms, 1m)");
      return;
    }

    if (!TryBuildGameAnalyzePosition(pgnFile, moveSpec, "game-analyze-lc0", out PositionWithHistory pwh, out string positionCommand))
    {
      return;
    }

    if (LC0AnalyzeHandler == null)
    {
      UCIWriteLine("info string game-analyze-lc0: Lc0 analysis handler not available in this host.");
      return;
    }

    // Ensure an evaluator definition exists (built from the current network/device configuration).
    if (EvaluatorDef == null)
    {
      try
      {
        CreateEvaluator();
      }
      catch (Exception e)
      {
        UCIWriteLine($"info string game-analyze-lc0: cannot configure network/device: {e.Message}");
        return;
      }
    }

    // Run the Lc0 analysis (streams output to the console, then closes the Lc0 engine).
    string fenAndMoves = pwh.GetFENAndMovesString(IsChess960OptionSet);
    LC0AnalyzeHandler((EvaluatorDef, fenAndMoves, searchMs.Value, OutStream));

    // Load the position into Ceres for further manual analysis, but do NOT auto-run a Ceres search.
    ProcessPosition(positionCommand);
    UCIWriteLine();
    UCIWriteLine("game-analyze-lc0: Lc0 analysis complete; position loaded in Ceres (no search run). Type 'go ...' to analyze with Ceres.");
  }
}
