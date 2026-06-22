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
using System.Linq;

using Ceres.Base.Environment;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;

#endregion

namespace Ceres.Features.Tournaments.Streaming
{
  /// <summary>
  /// Pure converters from Ceres internal tournament/search objects into the plain
  /// wire DTOs. This is the only place that knows how Ceres fields map onto the
  /// CELT/1 schema; the publisher and the tournament core never touch DTO internals.
  /// </summary>
  public static class DtoMappers
  {
    const int MAX_TOP_MOVES = 8;

    /// <summary>
    /// Coerces a float to a finite value. Engines can occasionally report NaN/Infinity for a stat
    /// (e.g. moves-left or a verbose-move-stat value); System.Text.Json throws on non-finite floats,
    /// which would otherwise cause the whole move frame to be dropped, so sanitize at the boundary.
    /// </summary>
    static float Fin(float x) => float.IsFinite(x) ? x : 0f;


    /// <summary>Valid, non-"node" verbose move stats ordered best-first by visit count.</summary>
    static List<VerboseMoveStat> OrderedValidStats(IEnumerable<VerboseMoveStat> stats)
      => stats.Where(v => v != null && v.Valid && v.MoveString != "node")
              .OrderByDescending(v => v.VisitCount)
              .ToList();


    /// <summary>Maps the (already ordered) top verbose move stats into TopQ DTOs.</summary>
    static List<TopMoveDTO> BuildTopMoves(List<VerboseMoveStat> ordered)
      => ordered.Take(MAX_TOP_MOVES)
                .Select(v => new TopMoveDTO
                {
                  Lan = v.MoveString,
                  N = v.VisitCount,
                  P = Fin(v.P),
                  Q = Fin((float)v.Q.LogisticValue),
                  Wl = Fin(v.WL),
                  D = Fin(v.D)
                }).ToList();


    /// <summary>Derives a win/draw/loss split (fractions) from a single verbose move stat.</summary>
    static WdlDTO WdlFromStat(VerboseMoveStat v)
    {
      float win = (1f - v.D + v.WL) / 2f;
      float loss = (1f - v.D - v.WL) / 2f;
      return new WdlDTO { W = Fin(win), D = Fin(v.D), L = Fin(loss) };
    }


    /// <summary>
    /// Builds a transient mid-search interim snapshot from the live search manager (MCTS or MCGS).
    /// Returns null if the manager type is unrecognized (e.g. a UCI/external engine) or the search
    /// has not yet produced any visits. Reads only; safe to call from the search progress callback.
    /// Eval/Q are from the mover's (thinking side's) perspective, matching the move frame; the
    /// consumer applies the white-perspective negation. Any failure yields null (the next tick retries).
    /// </summary>
    public static InterimDTO ToInterim(object managerContext, int ply, bool sideToMoveIsWhite)
    {
      try
      {
        if (managerContext is Ceres.MCTS.Iteration.MCTSManager mcts)
        {
          return ToInterimMCTS(mcts, ply, sideToMoveIsWhite);
        }
        if (managerContext is Ceres.MCGS.Search.Coordination.MCGSManager mcgs)
        {
          return ToInterimMCGS(mcgs, ply, sideToMoveIsWhite);
        }
      }
      catch { }
      return null;
    }


    static InterimDTO ToInterimMCTS(Ceres.MCTS.Iteration.MCTSManager m, int ply, bool isWhite)
    {
      var root = m.Root;
      if (root.N == 0)
      {
        return null;
      }

      float qBest = root.BestMoveInfo(false).QOfBest;
      int cp = (int)Math.Round(Ceres.Chess.LC0.Positions.EncodedEvalLogistic.WinLossToCentipawn(qBest));

      float elapsedSec = (float)(DateTime.Now - m.StartTimeThisSearch).TotalSeconds;
      long nodes = root.N;
      int nps = elapsedSec > 0 ? (int)Math.Round(nodes / elapsedSec) : 0;
      int depth = (int)Math.Round(m.Context.AvgDepth);

      List<TopMoveDTO> top = null;
      WdlDTO wdl = null;
      try
      {
        List<VerboseMoveStat> ordered = OrderedValidStats(Ceres.MCTS.UCI.VerboseMoveStatsFromMCTSNode.BuildStats(root));
        if (ordered.Count > 0)
        {
          top = BuildTopMoves(ordered);
          wdl = WdlFromStat(ordered[0]);
        }
      }
      catch { }

      return new InterimDTO
      {
        Type = "interim",
        Ply = ply,
        Side = isWhite ? "w" : "b",
        EvalCp = cp,
        ScoreQ = Fin(qBest),
        Nodes = nodes,
        Nps = nps,
        Eps = 0,
        Depth = depth,
        SelDepth = depth,
        MAvg = Fin(root.MAvg),
        MoveTimeMs = (int)Math.Round(elapsedSec * 1000),
        Wdl = wdl,
        Top = top
      };
    }


    static InterimDTO ToInterimMCGS(Ceres.MCGS.Search.Coordination.MCGSManager m, int ply, bool isWhite)
    {
      long rootN = (long)m.RootNWhenSearchStarted + m.NumNodesVisitedThisSearch;
      if (rootN <= 0)
      {
        return null;
      }

      // false = non-final best-move calc (a read-only mid-search query, like the live UCI info line).
      var best = m.GetBestMove(out _, out _, out _, false);
      float qBest = best.QOfBest;
      int cp = (int)Math.Round(Ceres.Chess.LC0.Positions.EncodedEvalLogistic.WinLossToCentipawn(qBest));

      float elapsedSec = (float)(DateTime.Now - m.StartTimeThisSearch).TotalSeconds;
      int nodesThisSearch = m.NumNodesVisitedThisSearch;
      int nps = elapsedSec > 0 ? (int)Math.Round(nodesThisSearch / elapsedSec) : 0;
      int eps = elapsedSec > 0 ? (int)Math.Round(m.NumEvalsThisSearch / elapsedSec) : 0;

      List<TopMoveDTO> top = null;
      WdlDTO wdl = null;
      try
      {
        List<VerboseMoveStat> ordered = OrderedValidStats(Ceres.MCGS.UCI.VerboseMoveStatsFromMCGSNode.BuildStats(m, best));
        if (ordered.Count > 0)
        {
          top = BuildTopMoves(ordered);
          wdl = WdlFromStat(ordered[0]);
        }
      }
      catch { }

      return new InterimDTO
      {
        Type = "interim",
        Ply = ply,
        Side = isWhite ? "w" : "b",
        EvalCp = cp,
        ScoreQ = Fin(qBest),
        Nodes = rootN,
        Nps = nps,
        Eps = eps,
        Depth = (int)Math.Round(m.AvgDepth),
        SelDepth = m.MaxDepth,
        MAvg = 0f,
        MoveTimeMs = (int)Math.Round(elapsedSec * 1000),
        Wdl = wdl,
        Top = top
      };
    }

    public static TournamentMetaDTO ToTournamentMeta(TournamentDef def, int threadCount)
    {
      List<PlayerDTO> players = new();
      SearchLimit tc;

      if (def.Engines != null && def.Engines.Length > 0)
      {
        foreach (EnginePlayerDef e in def.Engines)
        {
          players.Add(new PlayerDTO { Name = e.ID, Description = e.EngineDef?.ToString() });
        }
        tc = def.Engines[0].SearchLimit;
      }
      else
      {
        players.Add(new PlayerDTO { Name = def.Player1Def.ID, Description = def.Player1Def.EngineDef?.ToString() });
        players.Add(new PlayerDTO { Name = def.Player2Def.ID, Description = def.Player2Def.EngineDef?.ToString() });
        tc = def.Player1Def.SearchLimit;
      }

      string ceresVersion = null;
      try { ceresVersion = CeresVersion.VersionString; } catch { }

      return new TournamentMetaDTO
      {
        Type = "tournamentInfo",
        Seq = 0,
        ThreadId = -1,
        Id = def.ID,
        Name = def.ID,
        MachineName = Environment.MachineName,
        Mode = string.IsNullOrEmpty(def.ReferenceEngineId) ? "RR" : "Gauntlet",
        NumGamePairs = def.NumGamePairs ?? 0,
        ThreadCount = threadCount,
        Players = players,
        TimeControl = ToTimeControl(tc),
        CeresVersion = ceresVersion
      };
    }


    public static TimeControlDTO ToTimeControl(SearchLimit limit)
    {
      if (limit == null)
      {
        return new TimeControlDTO { Kind = "other" };
      }

      TimeControlDTO dto = new() { MovesToGo = limit.MaxMovesToGo ?? 0 };
      switch (limit.Type)
      {
        case SearchLimitType.SecondsForAllMoves:
          dto.Kind = "secondsForAllMoves";
          dto.ValueMs = (int)Math.Round(limit.Value * 1000);
          dto.IncrementMs = (int)Math.Round(limit.ValueIncrement * 1000);
          break;
        case SearchLimitType.SecondsPerMove:
          dto.Kind = "secondsPerMove";
          dto.ValueMs = (int)Math.Round(limit.Value * 1000);
          break;
        case SearchLimitType.NodesPerMove:
        case SearchLimitType.NodesPerTree:
          dto.Kind = "nodesPerMove";
          dto.Nodes = (int)limit.Value;
          break;
        case SearchLimitType.NodesForAllMoves:
          dto.Kind = "nodesForAllMoves";
          dto.Nodes = (int)limit.Value;
          break;
        default:
          dto.Kind = "other";
          break;
      }
      return dto;
    }


    static int InitialTimeMs(SearchLimit limit)
    {
      if (limit == null)
      {
        return 0;
      }
      if (limit.Type == SearchLimitType.SecondsForAllMoves || limit.Type == SearchLimitType.SecondsPerMove)
      {
        return (int)Math.Round(limit.Value * 1000);
      }
      return 0;
    }


    public static GameStartDTO ToGameStart(int gameSequenceNum, int openingIndex,
                                           GameEngine engine1, GameEngine engine2, bool engine2IsWhite,
                                           SearchLimit searchLimitEngine1, SearchLimit searchLimitEngine2,
                                           string startFEN, PositionWithHistory opening, string openingName)
    {
      GameEngine whiteEngine = engine2IsWhite ? engine2 : engine1;
      GameEngine blackEngine = engine2IsWhite ? engine1 : engine2;
      SearchLimit whiteLimit = engine2IsWhite ? searchLimitEngine2 : searchLimitEngine1;
      SearchLimit blackLimit = engine2IsWhite ? searchLimitEngine1 : searchLimitEngine2;

      bool whiteToMove = opening.FinalPosition.MiscInfo.SideToMove == SideType.White;

      return new GameStartDTO
      {
        Type = "gameStart",
        GameSequenceNum = gameSequenceNum,
        OpeningIndex = openingIndex,
        RoundNumber = openingIndex + 1,
        WhiteName = whiteEngine.ID,
        BlackName = blackEngine.ID,
        WhiteDescription = whiteEngine.ID,
        BlackDescription = blackEngine.ID,
        InitialFEN = opening.InitialPosition.FEN,
        StartFEN = startFEN,
        OpeningUci = opening.MovesStr,
        OpeningName = openingName,
        WhiteToMove = whiteToMove,
        WhiteTimeMs = InitialTimeMs(whiteLimit),
        BlackTimeMs = InitialTimeMs(blackLimit)
      };
    }


    public static MoveDTO ToMove(int ply, bool isWhite, string moveStr, string fenAfter,
                                 float scoreCentipawns, float scoreQ, float mAvg, int depth,
                                 GameEngineSearchResult engineMove, float engineTimeSec,
                                 SearchLimit preMoveLimit, int piecesLeft)
    {
      List<TopMoveDTO> top = null;
      WdlDTO wdl = null;
      if (engineMove?.VerboseMoveStats != null && engineMove.VerboseMoveStats.Count > 0)
      {
        top = BuildTopMoves(OrderedValidStats(engineMove.VerboseMoveStats));

        VerboseMoveStat played = engineMove.VerboseMoveStats.FirstOrDefault(v => v != null && v.MoveString == moveStr);
        if (played != null)
        {
          wdl = WdlFromStat(played);
        }
      }

      int timeLeftMs = 0;
      if (preMoveLimit != null
          && (preMoveLimit.Type == SearchLimitType.SecondsForAllMoves || preMoveLimit.Type == SearchLimitType.SecondsPerMove))
      {
        timeLeftMs = (int)Math.Round(Math.Max(0, preMoveLimit.Value - engineTimeSec) * 1000);
      }

      // Depth: some engines (e.g. MCGS) leave the integer Depth at 0 but populate the AvgDepth/MaxDepth
      // float fields, so fall back to those. SelDepth shows the max depth where available.
      int effDepth = depth > 0 ? depth : (engineMove != null ? (int)Math.Round(engineMove.AvgDepth) : 0);
      int effSelDepth = (engineMove != null && engineMove.MaxDepth > 0) ? (int)Math.Round(engineMove.MaxDepth) : effDepth;

      return new MoveDTO
      {
        Type = "move",
        Ply = ply,
        Side = isWhite ? "w" : "b",
        Lan = moveStr,
        Fen = fenAfter,
        EvalCp = float.IsFinite(scoreCentipawns) ? (int)Math.Round(scoreCentipawns) : 0,
        ScoreQ = Fin(scoreQ),
        MoveTimeMs = (int)Math.Round(engineTimeSec * 1000),
        TimeLeftMs = timeLeftMs,
        Nodes = engineMove?.FinalN ?? 0,
        Nps = engineMove?.NPS ?? 0,
        Eps = engineMove?.EPS ?? 0,
        Depth = effDepth,
        SelDepth = effSelDepth,
        MAvg = Fin(mAvg),
        PiecesLeft = piecesLeft,
        Instamove = (engineMove?.CountSearchContinuations ?? 0) > 0,
        Wdl = wdl,
        Top = top
      };
    }


    public static GameEndDTO ToGameEnd(TournamentGameInfo r)
    {
      // Result is relative to engine1 (matches WritePGNResult): Win => engine1 won, Loss => engine2 won.
      // Engine2IsWhite tells which color engine2 played, so we can resolve the white/black result string.
      string resultStr = r.Result switch
      {
        TournamentGameResult.Draw => "1/2-1/2",
        TournamentGameResult.Win => r.Engine2IsWhite ? "0-1" : "1-0",
        TournamentGameResult.Loss => r.Engine2IsWhite ? "1-0" : "0-1",
        _ => "*"
      };

      return new GameEndDTO
      {
        Type = "gameEnd",
        WhiteName = r.PlayerWhite,
        BlackName = r.PlayerBlack,
        Result = resultStr,
        Reason = MapReason(r.ResultReason),
        Moves = (r.PlyCount + 1) / 2,
        PlyCount = r.PlyCount,
        GameTimeMs = (long)Math.Round((r.TotalTimeEngine1 + r.TotalTimeEngine2) * 1000),
        OpeningIndex = r.OpeningIndex
      };
    }


    static string MapReason(TournamentGameResultReason reason) => reason switch
    {
      TournamentGameResultReason.Checkmate => "CM",
      TournamentGameResultReason.Stalemate => "SM",
      TournamentGameResultReason.AdjudicateTB => "TB",
      TournamentGameResultReason.AdjudicateMaterial => "AM",
      TournamentGameResultReason.ExcessiveMoves => "XX",
      TournamentGameResultReason.Repetition => "R3",
      TournamentGameResultReason.AdjudicatedEvaluation => "AE",
      TournamentGameResultReason.ForfeitLimits => "FL",
      _ => "XX"
    };
  }
}
