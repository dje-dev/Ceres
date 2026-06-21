#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.Json;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;

namespace Ceres.Features.TCEC;

// Immutable snapshot of one TCEC live.json payload.
// Encapsulates the parsed game state using Ceres.Chess primitives
// (Position, PositionWithHistory, Move) plus all engine-analysis fields
// surfaced by the TCEC API. Reusable beyond the live-stream feature
// (e.g. logging, replay, PGN export).

public readonly record struct TCECEngineInfo(string Name, int Elo);

public readonly record struct TCECMoveInfo
{
  public int Ply { get; init; }
  public int MoveNumber { get; init; }
  public bool IsWhite { get; init; }
  public string SAN { get; init; }
  public string FromSquare { get; init; }
  public string ToSquare { get; init; }
  public Move Move { get; init; }
  public bool MoveParseOk { get; init; }
  public Position PositionAfter { get; init; }
  public string FENAfterRaw { get; init; }
  public bool IsBookMove { get; init; }
  public double EvalPawns { get; init; }
  public int Depth { get; init; }
  public int SelDepth { get; init; }
  public long Nodes { get; init; }
  public long NodesPerSecond { get; init; }
  public long MoveTimeMs { get; init; }
  public long TimeLeftMs { get; init; }
  public int TablebaseHits { get; init; }
  public double HashUsagePct { get; init; }
  public string PredictedReply { get; init; }
  public string PVSan { get; init; }
  public IReadOnlyList<string> PVMoves { get; init; }
}

public readonly record struct TCECLiveSnapshot
{
  public string Event { get; init; }
  public string Round { get; init; }
  public string Date { get; init; }
  public string Site { get; init; }
  public string Variant { get; init; }
  public string TimeControl { get; init; }
  public TCECEngineInfo White { get; init; }
  public TCECEngineInfo Black { get; init; }
  public string Result { get; init; }
  public Game.GameResult ResultEnum { get; init; }
  public string Termination { get; init; }
  public Position StartingPosition { get; init; }
  public PositionWithHistory History { get; init; }
  public IReadOnlyList<TCECMoveInfo> Moves { get; init; }
  public string GameKey { get; init; }
  public DateTime FetchedAtUtc { get; init; }
  public DateTime GameDate { get; init; }

  // Parse a live.json payload into a snapshot.
  // Throws JsonException only when the JSON itself is malformed.
  // Per-move SAN parse failures are recorded on TCECMoveInfo.MoveParseOk
  // and do not abort the snapshot.
  public static TCECLiveSnapshot FromJson(string json)
  {
    using JsonDocument doc = JsonDocument.Parse(json);
    JsonElement root = doc.RootElement;

    JsonElement headers = root.TryGetProperty("Headers", out JsonElement h)
                          ? h : default;

    string eventName = SafeStr(headers, "Event");
    string round = SafeStr(headers, "Round");
    string dateStr = SafeStr(headers, "Date");
    string site = SafeStr(headers, "Site");
    string variant = SafeStr(headers, "Variant");
    string timeControl = SafeStr(headers, "TimeControl");
    string whiteName = SafeStr(headers, "White");
    string blackName = SafeStr(headers, "Black");
    int whiteElo = SafeInt(headers, "WhiteElo", 0);
    int blackElo = SafeInt(headers, "BlackElo", 0);
    string result = SafeStr(headers, "Result");
    string termination = SafeStr(headers, "Termination");

    // Apply the variant-specific Chess960 flag BEFORE any move parsing.
    // The TCECMonitor loop is responsible for restoring the original
    // value when the feature exits.
    MGPositionConstants.IsChess960 = string.Equals(variant, "fischerandom",
                                                    StringComparison.OrdinalIgnoreCase);

    // The starting FEN lives in Headers.FEN. (Older descriptions placed it
    // at the JSON root; not the case in current TCEC payloads.)
    string startFen = SafeStr(headers, "FEN");
    if (string.IsNullOrEmpty(startFen))
    {
      startFen = SafeStr(root, "FEN");
    }
    // Standard TCEC games carry no FEN header (only Chess960 / custom-start games do). Default to the
    // normal initial position so the move history and principal variations reconstruct correctly (and
    // so successive snapshots share a common prefix, enabling engine graph reuse across moves).
    if (string.IsNullOrEmpty(startFen))
    {
      startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    }
    Position startingPos = default;
    bool startParseOk = false;
    if (!string.IsNullOrEmpty(startFen))
    {
      try
      {
        startingPos = Position.FromFEN(startFen);
        startParseOk = true;
      }
      catch
      {
        startParseOk = false;
      }
    }

    List<TCECMoveInfo> moveInfos = new List<TCECMoveInfo>();

    // Robustly reconstructed full-game move list (as MGMoves). Built from each move's parsed SAN,
    // falling back to its from/to coordinates, so a single SAN-parse quirk cannot silently drop a move
    // and corrupt the sequence. historyContiguous goes false the instant a move cannot be resolved, so
    // we never build a partial/misaligned history (which would defeat the engine's graph reuse).
    List<MGMove> historyMoves = new List<MGMove>();
    bool historyContiguous = startParseOk;

    if (root.TryGetProperty("Moves", out JsonElement movesArr)
        && movesArr.ValueKind == JsonValueKind.Array)
    {
      Position runningPos = startingPos;
      bool runningOk = startParseOk;
      int ply = 0;
      foreach (JsonElement me in movesArr.EnumerateArray())
      {
        ply++;
        Position posBefore = runningPos;
        bool posBeforeOk = runningOk;
        bool isWhite = (ply % 2) == 1;
        int moveNumber = (ply + 1) / 2;
        string san = SafeStr(me, "m");
        string from = SafeStr(me, "from");
        string to = SafeStr(me, "to");
        string fenAfter = SafeStr(me, "fen");

        Move parsedMove = default;
        bool parseOk = false;
        Position posAfter = default;

        if (runningOk && !string.IsNullOrEmpty(san))
        {
          try
          {
            parsedMove = runningPos.MoveSAN(san);
            posAfter = runningPos.AfterMove(parsedMove);
            parseOk = true;
          }
          catch
          {
            parseOk = false;
          }
        }

        // Always recover the position-after via the raw FEN if available;
        // this keeps the running position aligned even when SAN parsing
        // fails (rare, but possible on edge cases).
        if (!parseOk && !string.IsNullOrEmpty(fenAfter))
        {
          try
          {
            posAfter = Position.FromFEN(fenAfter);
            runningOk = true;
          }
          catch
          {
            runningOk = false;
          }
        }

        runningPos = posAfter;

        // Resolve this move as an MGMove for the reusable game history (SAN first, else from/to UCI).
        if (historyContiguous && posBeforeOk)
        {
          MGMove histMove = default;
          bool gotHistMove = false;
          if (parseOk)
          {
            try
            {
              histMove = Ceres.Chess.MoveGen.Converters.MGMoveConverter.MGMoveFromPosAndMove(in posBefore, parsedMove);
              gotHistMove = true;
            }
            catch
            {
              gotHistMove = false;
            }
          }
          if (!gotHistMove && !string.IsNullOrEmpty(from) && !string.IsNullOrEmpty(to))
          {
            try
            {
              Move uciMove = Move.FromUCI(in posBefore, from + to + PromoFromSAN(san));
              histMove = Ceres.Chess.MoveGen.Converters.MGMoveConverter.MGMoveFromPosAndMove(in posBefore, uciMove);
              gotHistMove = true;
            }
            catch
            {
              gotHistMove = false;
            }
          }
          if (gotHistMove)
          {
            historyMoves.Add(histMove);
          }
          else
          {
            historyContiguous = false;
          }
        }
        else
        {
          historyContiguous = false;
        }

        // Read PV.
        string pvSan = "";
        IReadOnlyList<string> pvMoves = Array.Empty<string>();
        if (me.TryGetProperty("pv", out JsonElement pvEl))
        {
          if (pvEl.ValueKind == JsonValueKind.Object)
          {
            pvSan = SafeStr(pvEl, "San");
            if (pvEl.TryGetProperty("Moves", out JsonElement pvMovesEl)
                && pvMovesEl.ValueKind == JsonValueKind.Array)
            {
              List<string> mlist = new List<string>(pvMovesEl.GetArrayLength());
              foreach (JsonElement pm in pvMovesEl.EnumerateArray())
              {
                if (pm.ValueKind == JsonValueKind.String)
                {
                  mlist.Add(pm.GetString());
                }
                else if (pm.ValueKind == JsonValueKind.Object)
                {
                  string pmStr = SafeStr(pm, "m");
                  if (!string.IsNullOrEmpty(pmStr))
                  {
                    mlist.Add(pmStr);
                  }
                }
              }
              pvMoves = mlist;
            }
          }
          else if (pvEl.ValueKind == JsonValueKind.String)
          {
            pvSan = pvEl.GetString();
          }
        }

        moveInfos.Add(new TCECMoveInfo
        {
          Ply = ply,
          MoveNumber = moveNumber,
          IsWhite = isWhite,
          SAN = san,
          FromSquare = from,
          ToSquare = to,
          Move = parsedMove,
          MoveParseOk = parseOk,
          PositionAfter = posAfter,
          FENAfterRaw = fenAfter,
          IsBookMove = SafeBool(me, "book", false),
          EvalPawns = SafeDouble(me, "wv", double.NaN),
          Depth = SafeInt(me, "d", 0),
          SelDepth = SafeInt(me, "sd", 0),
          Nodes = SafeLong(me, "n", 0),
          NodesPerSecond = SafeLong(me, "s", 0),
          MoveTimeMs = SafeLong(me, "mt", 0),
          TimeLeftMs = SafeLong(me, "tl", 0),
          TablebaseHits = SafeInt(me, "tb", 0),
          HashUsagePct = SafeDouble(me, "h", 0.0),
          PredictedReply = SafeStr(me, "pd"),
          PVSan = pvSan,
          PVMoves = pvMoves,
        });
      }
    }

    // Build the full-game PositionWithHistory ONLY when every move resolved into a contiguous sequence,
    // and verify its final position matches the feed's latest position (piece placement / side to move /
    // castling). A complete, verified history is what lets the MCGS engine reuse its graph across moves
    // (GraphReuseManager matches it as a prefix and shifts the search root down). If anything is off we
    // leave History null so the caller searches the correct position from a FEN (no reuse, but correct).
    PositionWithHistory history = null;
    if (startParseOk && historyContiguous && historyMoves.Count == moveInfos.Count)
    {
      try
      {
        PositionWithHistory hist = new PositionWithHistory(in startingPos);
        foreach (MGMove mg in historyMoves)
        {
          hist.AppendMove(mg);
        }
        string lastFen = moveInfos.Count > 0 ? moveInfos[moveInfos.Count - 1].FENAfterRaw : null;
        if (string.IsNullOrEmpty(lastFen) || FenCoreMatches(hist.FinalPosition.FEN, lastFen))
        {
          history = hist;
        }
      }
      catch
      {
        history = null;
      }
    }

    DateTime gameDate = default;
    if (!string.IsNullOrEmpty(dateStr))
    {
      DateTime parsed;
      if (DateTime.TryParseExact(dateStr, "yyyy.MM.dd",
            CultureInfo.InvariantCulture, DateTimeStyles.None, out parsed))
      {
        gameDate = parsed;
      }
    }

    Game.GameResult resultEnum = Game.GameResult.Unterminated;
    if (result != null)
    {
      if (result.Contains("1/2-1/2"))
      {
        resultEnum = Game.GameResult.Draw;
      }
      else if (result == "1-0")
      {
        resultEnum = Game.GameResult.WhiteWins;
      }
      else if (result == "0-1")
      {
        resultEnum = Game.GameResult.BlackWins;
      }
    }

    string gameKey = string.Join("|",
      eventName ?? "", round ?? "", whiteName ?? "", blackName ?? "",
      dateStr ?? "");

    return new TCECLiveSnapshot
    {
      Event = eventName,
      Round = round,
      Date = dateStr,
      Site = site,
      Variant = variant,
      TimeControl = timeControl,
      White = new TCECEngineInfo(whiteName, whiteElo),
      Black = new TCECEngineInfo(blackName, blackElo),
      Result = result,
      ResultEnum = resultEnum,
      Termination = termination,
      StartingPosition = startingPos,
      History = history,
      Moves = moveInfos,
      GameKey = gameKey,
      FetchedAtUtc = DateTime.UtcNow,
      GameDate = gameDate,
    };
  }

  // Promotion piece (lowercase) extracted from a SAN move (e.g. "e8=Q" -> "q"), or "" if none.
  static string PromoFromSAN(string san)
  {
    if (string.IsNullOrEmpty(san))
    {
      return "";
    }
    int eq = san.IndexOf('=');
    if (eq >= 0 && eq + 1 < san.Length)
    {
      return char.ToLowerInvariant(san[eq + 1]).ToString();
    }
    return "";
  }

  // Confirms two FENs describe the same board by comparing piece placement and side to move only.
  // Castling (FRC vs standard notation), en-passant square, and move clocks differ by convention
  // between Ceres and TCEC and are not corruption, so they are intentionally ignored.
  static bool FenCoreMatches(string a, string b)
  {
    if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b))
    {
      return false;
    }
    string[] pa = a.Split(' ');
    string[] pb = b.Split(' ');
    if (pa.Length < 2 || pb.Length < 2)
    {
      return string.Equals(a, b, StringComparison.Ordinal);
    }
    return pa[0] == pb[0] && pa[1] == pb[1];
  }

  // -------- defensive JSON readers (handle string-vs-number variation) --------

  static string SafeStr(JsonElement parent, string name)
  {
    if (parent.ValueKind != JsonValueKind.Object)
    {
      return "";
    }
    if (!parent.TryGetProperty(name, out JsonElement el))
    {
      return "";
    }
    if (el.ValueKind == JsonValueKind.String)
    {
      string s = el.GetString();
      return s ?? "";
    }
    if (el.ValueKind == JsonValueKind.Number)
    {
      return el.GetRawText();
    }
    if (el.ValueKind == JsonValueKind.True)
    {
      return "true";
    }
    if (el.ValueKind == JsonValueKind.False)
    {
      return "false";
    }
    return "";
  }

  static int SafeInt(JsonElement parent, string name, int dflt)
  {
    if (parent.ValueKind != JsonValueKind.Object
        || !parent.TryGetProperty(name, out JsonElement el))
    {
      return dflt;
    }
    if (el.ValueKind == JsonValueKind.Number)
    {
      int n;
      if (el.TryGetInt32(out n))
      {
        return n;
      }
      double d;
      if (el.TryGetDouble(out d))
      {
        return (int)d;
      }
      return dflt;
    }
    if (el.ValueKind == JsonValueKind.String)
    {
      string s = el.GetString();
      int n;
      if (int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out n))
      {
        return n;
      }
      double d;
      if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out d))
      {
        return (int)d;
      }
    }
    return dflt;
  }

  static long SafeLong(JsonElement parent, string name, long dflt)
  {
    if (parent.ValueKind != JsonValueKind.Object
        || !parent.TryGetProperty(name, out JsonElement el))
    {
      return dflt;
    }
    if (el.ValueKind == JsonValueKind.Number)
    {
      long n;
      if (el.TryGetInt64(out n))
      {
        return n;
      }
      double d;
      if (el.TryGetDouble(out d))
      {
        return (long)d;
      }
      return dflt;
    }
    if (el.ValueKind == JsonValueKind.String)
    {
      string s = el.GetString();
      long n;
      if (long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out n))
      {
        return n;
      }
      double d;
      if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out d))
      {
        return (long)d;
      }
    }
    return dflt;
  }

  static double SafeDouble(JsonElement parent, string name, double dflt)
  {
    if (parent.ValueKind != JsonValueKind.Object
        || !parent.TryGetProperty(name, out JsonElement el))
    {
      return dflt;
    }
    if (el.ValueKind == JsonValueKind.Number)
    {
      double d;
      if (el.TryGetDouble(out d))
      {
        return d;
      }
      return dflt;
    }
    if (el.ValueKind == JsonValueKind.String)
    {
      string s = el.GetString();
      double d;
      if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out d))
      {
        return d;
      }
    }
    return dflt;
  }

  static bool SafeBool(JsonElement parent, string name, bool dflt)
  {
    if (parent.ValueKind != JsonValueKind.Object
        || !parent.TryGetProperty(name, out JsonElement el))
    {
      return dflt;
    }
    if (el.ValueKind == JsonValueKind.True)
    {
      return true;
    }
    if (el.ValueKind == JsonValueKind.False)
    {
      return false;
    }
    if (el.ValueKind == JsonValueKind.String)
    {
      string s = el.GetString();
      bool b;
      if (bool.TryParse(s, out b))
      {
        return b;
      }
      if (string.Equals(s, "1", StringComparison.Ordinal))
      {
        return true;
      }
      if (string.Equals(s, "0", StringComparison.Ordinal))
      {
        return false;
      }
    }
    if (el.ValueKind == JsonValueKind.Number)
    {
      int n;
      if (el.TryGetInt32(out n))
      {
        return n != 0;
      }
    }
    return dflt;
  }
}
