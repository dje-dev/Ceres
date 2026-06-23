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
                                : PromptForGameAnalyzeArg("Enter move number (e.g. 105 = move 105 White to move, 105.. = Black to move):");
    timeSpec = parts.Length > 3 ? parts[3]
                                : PromptForGameAnalyzeArg("Enter search time (e.g. 10s):");

    if (string.IsNullOrWhiteSpace(pgnFile) || string.IsNullOrWhiteSpace(moveSpec) || string.IsNullOrWhiteSpace(timeSpec))
    {
      UCIWriteLine($"info string {commandName} requires three arguments: <pgn file> <move number> <time> (e.g. {commandName} game.pgn 105 10s)");
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
  /// Loads the PGN, navigates to the position specified by the move spec (preserving full move
  /// history), and builds the corresponding UCI "position fen ... moves ..." command. Echoes the
  /// constructed position and final FEN to the console. Returns false (with an error message) on failure.
  /// Shared by both the "game-analyze" and "game-analyze-lc0" features.
  /// </summary>
  /// <param name="pgnFile">path to the PGN file</param>
  /// <param name="moveSpec">move number, optionally suffixed with dots to indicate Black to move (e.g. "105" or "105..")</param>
  /// <param name="commandName">name of the invoking command (used in messages)</param>
  /// <param name="positionWithHistory">resulting position with full move history</param>
  /// <param name="positionCommand">resulting UCI "position fen ... moves ..." command string</param>
  internal bool TryBuildGameAnalyzePosition(string pgnFile, string moveSpec, string commandName,
                                            out PositionWithHistory positionWithHistory, out string positionCommand)
  {
    positionWithHistory = null;
    positionCommand = null;

    // Parse the move specification (a trailing dot indicates Black to move).
    string trimmedMoveSpec = moveSpec.Trim();
    bool blackToMove = trimmedMoveSpec.EndsWith(".");
    string numberPart = trimmedMoveSpec.TrimEnd('.');
    if (!int.TryParse(numberPart, NumberStyles.Integer, CultureInfo.InvariantCulture, out int moveNumber) || moveNumber < 1)
    {
      UCIWriteLine($"info string {commandName}: invalid move number \"{moveSpec}\" (expected e.g. 105 or 105..)");
      return false;
    }

    // Load the game from the PGN.
    if (!System.IO.File.Exists(pgnFile))
    {
      UCIWriteLine($"info string {commandName}: PGN file not found: {pgnFile}");
      return false;
    }

    Game game = Game.FromPGN(pgnFile).FirstOrDefault();
    if (game == null)
    {
      UCIWriteLine($"info string {commandName}: no games found in {pgnFile}");
      return false;
    }

    // PositionMiscInfo.MoveNum is stored as a ply count (fullmove * 2, +1 if Black to move).
    SideType targetSide = blackToMove ? SideType.Black : SideType.White;
    int targetMoveNum = moveNumber * 2 + (blackToMove ? 1 : 0);

    game.FirstMatchingPosition(pos => pos.MiscInfo.MoveNum == targetMoveNum
                                   && pos.MiscInfo.SideToMove == targetSide, out int moveIndex);
    if (moveIndex == -1)
    {
      UCIWriteLine($"info string {commandName}: position at move {moveNumber}{(blackToMove ? ".." : "")} "
                 + $"({targetSide} to move) not found in game (game has {game.Moves.Count} half-moves)");
      return false;
    }

    // Build the position (with full move history) up to and including the target position.
    Game truncatedGame = game.TruncatedAtMove(moveIndex);
    positionWithHistory = truncatedGame.FinalPositionWithHistory;
    positionCommand = "position fen " + positionWithHistory.GetFENAndMovesString(IsChess960OptionSet);

    UCIWriteLine();
    UCIWriteLine($"{commandName}: {game.PlayerWhite} vs {game.PlayerBlack}, "
               + $"move {moveNumber}{(blackToMove ? ".." : "")} ({targetSide} to move)");
    UCIWriteLine(positionCommand);
    UCIWriteLine("Final FEN: " + positionWithHistory.FinalPosition.FEN);
    UCIWriteLine();

    return true;
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

    if (!TryBuildGameAnalyzePosition(pgnFile, moveSpec, "game-analyze", out _, out string positionCommand))
    {
      return;
    }

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
