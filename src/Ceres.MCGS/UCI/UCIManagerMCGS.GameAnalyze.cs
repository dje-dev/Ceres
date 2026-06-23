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
using System.Linq;

using Ceres.Chess;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.MCGS.UCI;

/// <summary>
/// Implementation of the "game-analyze" feature, which locates a specified position
/// within a PGN file (by move number / side to move), feeds it to the engine with full
/// move history, runs a fixed-time search, and dumps detailed search information.
///
/// Available both as an interactive UCI command ("game-analyze", which prompts for the
/// three arguments) and as a command-line verb (see DispatchCommands). After running,
/// the analyzed position remains current so the user can continue analysis interactively.
/// </summary>
public partial class UCIManagerMCGS
{
  /// <summary>
  /// Entry point from the UCI command loop (or the command-line verb startup hook).
  /// Parses any arguments already present on the command line and interactively prompts
  /// for any that are missing, then runs the analysis.
  /// </summary>
  /// <param name="command">the full command line, e.g. "game-analyze /path/game.pgn 105 10s"</param>
  internal void ProcessGameAnalyzeInteractive(string command)
  {
    string[] parts = command.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    string pgnFile = parts.Length > 1 ? parts[1]
                                      : PromptForGameAnalyzeArg("Enter path to PGN file:");
    string moveSpec = parts.Length > 2 ? parts[2]
                                       : PromptForGameAnalyzeArg("Enter move number (e.g. 105 = move 105 White to move, 105.. = Black to move):");
    string timeSpec = parts.Length > 3 ? parts[3]
                                       : PromptForGameAnalyzeArg("Enter search time (e.g. 10s):");

    if (string.IsNullOrWhiteSpace(pgnFile) || string.IsNullOrWhiteSpace(moveSpec) || string.IsNullOrWhiteSpace(timeSpec))
    {
      UCIWriteLine("info string game-analyze requires three arguments: <pgn file> <move number> <time> (e.g. game-analyze game.pgn 105 10s)");
      return;
    }

    try
    {
      ProcessGameAnalyze(pgnFile, moveSpec, timeSpec);
    }
    catch (Exception exc)
    {
      UCIWriteLine($"info string game-analyze failed: {exc.Message}");
    }
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
  /// Core of the game-analyze feature: loads the PGN, navigates to the requested position
  /// (preserving full move history), echoes the constructed position, runs a fixed-time
  /// search, and dumps the search information. Leaves the analyzed position current.
  /// </summary>
  /// <param name="pgnFile">path to the PGN file</param>
  /// <param name="moveSpec">move number, optionally suffixed with dots to indicate Black to move (e.g. "105" or "105..")</param>
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

    // Parse the move specification (a trailing dot indicates Black to move).
    string trimmedMoveSpec = moveSpec.Trim();
    bool blackToMove = trimmedMoveSpec.EndsWith(".");
    string numberPart = trimmedMoveSpec.TrimEnd('.');
    if (!int.TryParse(numberPart, NumberStyles.Integer, CultureInfo.InvariantCulture, out int moveNumber) || moveNumber < 1)
    {
      UCIWriteLine($"info string game-analyze: invalid move number \"{moveSpec}\" (expected e.g. 105 or 105..)");
      return;
    }

    // Load the game from the PGN.
    if (!System.IO.File.Exists(pgnFile))
    {
      UCIWriteLine($"info string game-analyze: PGN file not found: {pgnFile}");
      return;
    }

    Game game = Game.FromPGN(pgnFile).FirstOrDefault();
    if (game == null)
    {
      UCIWriteLine($"info string game-analyze: no games found in {pgnFile}");
      return;
    }

    // PositionMiscInfo.MoveNum is stored as a ply count (fullmove * 2, +1 if Black to move).
    SideType targetSide = blackToMove ? SideType.Black : SideType.White;
    int targetMoveNum = moveNumber * 2 + (blackToMove ? 1 : 0);

    game.FirstMatchingPosition(pos => pos.MiscInfo.MoveNum == targetMoveNum
                                   && pos.MiscInfo.SideToMove == targetSide, out int moveIndex);
    if (moveIndex == -1)
    {
      UCIWriteLine($"info string game-analyze: position at move {moveNumber}{(blackToMove ? ".." : "")} "
                 + $"({targetSide} to move) not found in game (game has {game.Moves.Count} half-moves)");
      return;
    }

    // Build the position (with full move history) up to and including the target position.
    Game truncatedGame = game.TruncatedAtMove(moveIndex);
    PositionWithHistory pwh = truncatedGame.FinalPositionWithHistory;
    string positionCommand = "position fen " + pwh.GetFENAndMovesString(IsChess960OptionSet);

    UCIWriteLine();
    UCIWriteLine($"game-analyze: {game.PlayerWhite} vs {game.PlayerBlack}, "
               + $"analyzing move {moveNumber}{(blackToMove ? ".." : "")} ({targetSide} to move) for {searchMs.Value}ms");
    UCIWriteLine(positionCommand);
    UCIWriteLine("Final FEN: " + pwh.FinalPosition.FEN);
    UCIWriteLine();

    // Set up the position exactly as if the user had typed the position command.
    ProcessPosition(positionCommand);

    // Run the search (reuses the standard "go movetime" path, blocking until complete).
    if (InitializeEngineIfNeeded())
    {
      taskSearchCurrentlyExecuting = ProcessGo($"go movetime {searchMs.Value}");
      taskSearchCurrentlyExecuting.Wait();
    }

    // Dump full search information (same as the "dump-info" command).
    if (CeresEngine?.Search?.Manager != null)
    {
      CeresEngine.Search.Manager.DumpFullInfo(lastSearchResult, Console.Out, "game-analyze");
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
