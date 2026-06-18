#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System.Collections.Generic;

namespace Ceres.Features.Tournaments.Streaming
{
  /// <summary>
  /// Plain data-transfer objects defining the CELT/1 wire schema used to stream a
  /// live Ceres tournament to external consumers over NDJSON (one JSON object per line).
  /// These are deliberately simple (public auto-properties only) and contain no Ceres
  /// internal types, so that System.Text.Json can serialize them directly and a non-.NET
  /// consumer could also parse them. Units are documented explicitly on each field.
  /// </summary>
  public abstract class WireMsg
  {
    /// <summary>Message type discriminator (e.g. "move", "gameStart").</summary>
    public string Type { get; set; }

    /// <summary>Monotonic sequence number per thread (0 for tournament-global control frames).</summary>
    public long Seq { get; set; }

    /// <summary>Stable game-thread id (-1 for tournament-global frames).</summary>
    public int ThreadId { get; set; } = -1;
  }


  public sealed class PlayerDTO
  {
    /// <summary>Engine/player id (used as the display name).</summary>
    public string Name { get; set; }

    /// <summary>Longer engine description (e.g. network / configuration).</summary>
    public string Description { get; set; }
  }


  public sealed class TimeControlDTO
  {
    /// <summary>One of: secondsForAllMoves, secondsPerMove, nodesPerMove, nodesForAllMoves, other.</summary>
    public string Kind { get; set; }

    /// <summary>Base time in milliseconds (time-based limits only).</summary>
    public int ValueMs { get; set; }

    /// <summary>Per-move increment in milliseconds.</summary>
    public int IncrementMs { get; set; }

    /// <summary>Node budget (node-based limits only).</summary>
    public int Nodes { get; set; }

    /// <summary>Moves-to-go (0 = none).</summary>
    public int MovesToGo { get; set; }
  }


  /// <summary>type = "tournamentInfo" (tournament-global).</summary>
  public sealed class TournamentMetaDTO : WireMsg
  {
    public string Id { get; set; }
    public string Name { get; set; }
    public string MachineName { get; set; }
    public string Mode { get; set; }
    public int NumGamePairs { get; set; }
    public int ThreadCount { get; set; }
    public List<PlayerDTO> Players { get; set; } = new();
    public TimeControlDTO TimeControl { get; set; }
    public string CeresVersion { get; set; }
  }


  public sealed class OpeningMoveDTO
  {
    /// <summary>Move in UCI / long algebraic notation.</summary>
    public string Lan { get; set; }
  }


  /// <summary>type = "gameStart" (per thread).</summary>
  public sealed class GameStartDTO : WireMsg
  {
    public int GameSequenceNum { get; set; }
    public int OpeningIndex { get; set; }
    public int RoundNumber { get; set; }
    public string WhiteName { get; set; }
    public string BlackName { get; set; }
    public string WhiteDescription { get; set; }
    public string BlackDescription { get; set; }

    /// <summary>FEN before any opening moves.</summary>
    public string InitialFEN { get; set; }

    /// <summary>FEN after the opening moves, from which live play begins.</summary>
    public string StartFEN { get; set; }

    /// <summary>Space-separated UCI opening moves (may be empty).</summary>
    public string OpeningUci { get; set; }

    public string OpeningName { get; set; }
    public bool WhiteToMove { get; set; }

    /// <summary>Initial clock for white in milliseconds (0 if not time-based).</summary>
    public int WhiteTimeMs { get; set; }

    /// <summary>Initial clock for black in milliseconds (0 if not time-based).</summary>
    public int BlackTimeMs { get; set; }
  }


  public sealed class WdlDTO
  {
    public float W { get; set; }
    public float D { get; set; }
    public float L { get; set; }
  }


  public sealed class TopMoveDTO
  {
    /// <summary>Move in UCI / long algebraic notation.</summary>
    public string Lan { get; set; }

    /// <summary>Visit count (N) for this child.</summary>
    public long N { get; set; }

    /// <summary>Policy prior (percent, 0-100 as reported by the engine).</summary>
    public float P { get; set; }

    /// <summary>Q value in [-1, 1] (logistic).</summary>
    public float Q { get; set; }

    /// <summary>Win-minus-loss probability.</summary>
    public float Wl { get; set; }

    /// <summary>Draw probability.</summary>
    public float D { get; set; }
  }


  /// <summary>type = "move" (per thread). Carries the played move plus the engine's final search stats and top moves.</summary>
  public sealed class MoveDTO : WireMsg
  {
    public int Ply { get; set; }

    /// <summary>"w" or "b" (side that made this move).</summary>
    public string Side { get; set; }

    /// <summary>Move played, UCI / long algebraic.</summary>
    public string Lan { get; set; }

    /// <summary>FEN after the move.</summary>
    public string Fen { get; set; }

    /// <summary>Evaluation in CENTIPAWNS (consumer divides by 100). Null if unavailable.</summary>
    public int? EvalCp { get; set; }

    /// <summary>Q value of best move in [-1, 1].</summary>
    public float ScoreQ { get; set; }

    /// <summary>Time used for this move, milliseconds.</summary>
    public int MoveTimeMs { get; set; }

    /// <summary>Clock remaining for the mover after the move, milliseconds (0 if not time-based).</summary>
    public int TimeLeftMs { get; set; }

    public long Nodes { get; set; }
    public int Nps { get; set; }
    public int Eps { get; set; }
    public int Depth { get; set; }

    /// <summary>Estimated moves left (M head).</summary>
    public float MAvg { get; set; }

    public int PiecesLeft { get; set; }

    /// <summary>True if the move was an instamove / "obvious move" (search continuation).</summary>
    public bool Instamove { get; set; }

    public WdlDTO Wdl { get; set; }

    /// <summary>Top moves (TopQ), ordered best-first.</summary>
    public List<TopMoveDTO> Top { get; set; }
  }


  /// <summary>type = "gameEnd" (per thread) and "gameResult" (tournament-global). Same payload.</summary>
  public sealed class GameEndDTO : WireMsg
  {
    public string WhiteName { get; set; }
    public string BlackName { get; set; }

    /// <summary>"1-0", "0-1" or "1/2-1/2".</summary>
    public string Result { get; set; }

    /// <summary>Result reason as a short code (CM, SM, TB, AM, AE, R3, FL, XX, ...) per the streaming protocol.</summary>
    public string Reason { get; set; }

    /// <summary>Number of moves (not plies).</summary>
    public int Moves { get; set; }

    public int PlyCount { get; set; }

    /// <summary>Total game time in milliseconds.</summary>
    public long GameTimeMs { get; set; }
  }


  /// <summary>type = "tournamentEnd" (tournament-global).</summary>
  public sealed class TournamentEndDTO : WireMsg
  {
    public string Name { get; set; }
    public string Reason { get; set; }
  }


  public sealed class DirThreadDTO
  {
    public int ThreadId { get; set; }
    public string White { get; set; }
    public string Black { get; set; }
    public int GameNr { get; set; }

    /// <summary>"playing", "betweenGames" or "finished".</summary>
    public string State { get; set; }

    public long LatestSeq { get; set; }
  }


  /// <summary>type = "directoryResponse" (response to a directory query).</summary>
  public sealed class DirectoryDTO : WireMsg
  {
    public string TournamentName { get; set; }
    public List<DirThreadDTO> Threads { get; set; } = new();
  }


  /// <summary>type = "hello" (sent by server immediately on connect).</summary>
  public sealed class HelloDTO : WireMsg
  {
    public string Protocol { get; set; }
    public List<string> ProtocolsSupported { get; set; } = new();
    public int ThreadCount { get; set; }
  }


  /// <summary>type = "subscribed" (server acknowledgement of a subscribe request).</summary>
  public sealed class SubscribedDTO : WireMsg
  {
    public string Scope { get; set; }
    public string Protocol { get; set; }
    public long CurrentSeq { get; set; }
    public bool SnapshotFollows { get; set; }
  }


  /// <summary>Incoming client request (subscribe / directory). Deserialized from the first client line.</summary>
  public sealed class SubscribeRequest
  {
    public string Type { get; set; }

    /// <summary>"global" or "thread".</summary>
    public string Scope { get; set; }

    public int ThreadId { get; set; }
    public string Protocol { get; set; }
  }
}
