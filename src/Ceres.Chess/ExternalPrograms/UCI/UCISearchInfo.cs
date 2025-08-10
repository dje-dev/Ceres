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
using System.Text;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.SearchResultVerboseMoveInfo;

#endregion

namespace Ceres.Chess.ExternalPrograms.UCI
{
  /// <summary>
  /// Statistics related to a search conducted via an UCI engine.
  /// </summary>
  public record UCISearchInfo
  {
    /// <summary>
    /// The string as directly output by the UCI search engine.
    /// </summary>
    public string RawString;

    /// <summary>
    /// Primary search depth.
    /// </summary>
    public int Depth;

    /// <summary>
    /// Maximum search depth.
    /// </summary>
    public int SelDepth;

    /// <summary>
    /// Position evaluation score.
    /// </summary>
    public int ScoreCentipawns;

    /// <summary>
    /// Number of search positions visited.
    /// </summary>
    public ulong Nodes;

    /// <summary>
    /// Search speed in nodes per second.
    /// </summary>
    public int NPS;

    /// <summary>
    /// Search time reported by engine (in milliseconds).
    /// </summary>
    public int EngineReportedSearchTime;

    /// <summary>
    /// Moves to mate (if any).
    /// </summary>
    public int Mate;

    /// <summary>
    /// Selected best move by engine.
    /// </summary>
    public string BestMove;

    /// <summary>
    /// Move proposed for ponder by engine.
    /// </summary>
    public string Ponder;

    /// <summary>
    /// Win/draw/loss percentages (if available).
    /// </summary>
    public (float W, float D, float L) WDL = (float.NaN, float.NaN, float.NaN);

    /// <summary>
    /// Returns Q search evaluation (if available).
    /// </summary>
    public float Q => WDL.W - WDL.L;

    /// <summary>
    /// Win percentage
    /// </summary>
    public float W = float.NaN;

    /// <summary>
    /// Sequence of UCI detail lines emitted by engine 
    /// with search progress updates.
    /// </summary>
    public List<string> Infos = new List<string>();


    /// <summary>
    /// Optional extra infromation.
    /// </summary>
    public object ExtraInfo;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="rawString"></param>
    /// <param name="bestMove">optional, otherwise taken from pv in rawString</param>
    /// <param name="infos"></param>
    public UCISearchInfo(string rawString, string bestMove = null, List<string> infos = null)
    {
      RawString = rawString;

      if (infos != null)
      {
        foreach (string info in infos) Infos.Add(info);
        infos.Clear();
      }

      if (bestMove != null)
      {
        string[] parts = bestMove.Split(' ');

        if (parts.Length >= 2)
        {
          BestMove = parts[1];
        }
        if (parts.Length >= 4)
        {
          Ponder = parts[3];
        }
      }

      Depth = -1;
      SelDepth = -1;
      ScoreCentipawns = -1;
      Nodes = 0;
      NPS = -1;
      EngineReportedSearchTime = -1;
      Mate = 0;

      if (rawString == null) return; // no info returned before bestmove
      string[] tokens = rawString.Split(' ');

      void Check(int index, string str, ref int save) { if (tokens[index - 1] == str) int.TryParse(tokens[index], out save); }
      void CheckU(int index, string str, ref ulong save) { if (tokens[index - 1] == str) ulong.TryParse(tokens[index], out save); }

      for (int i = 1; i < tokens.Length; i++)
      {
        // info depth 12 seldepth 22 multipv 1 score cp 7 upperbound nodes 48421 nps 896685 tbhits 0 time 54 pv f1e1 e8e7
        //score mate -8
        Check(i, "depth", ref Depth);
        Check(i, "seldepth", ref SelDepth);
        Check(i, "cp", ref ScoreCentipawns);
        Check(i, "mate", ref Mate);
        CheckU(i, "nodes", ref Nodes);
        Check(i, "nps", ref NPS);
        Check(i, "time", ref EngineReportedSearchTime);

        if (tokens[i].ToUpper() == "WDL")
        {
          int.TryParse(tokens[i + 1], out int w);
          int.TryParse(tokens[i + 2], out int d);
          int.TryParse(tokens[i + 3], out int l);
          WDL = (w / 1000f, d / 1000f, l / 1000f);
        }

        if (tokens.Contains("mate"))
        {
          if (Mate <= 0) ScoreCentipawns = -2000 + Mate * 5;
          if (Mate > 0) ScoreCentipawns = 2000 - Mate * 5;
        }
      }
    }



    /// <summary>
    /// Returns the principal variation, if any.
    /// </summary>
    public string PVString
    {
      get
      {
        if (RawString == null) return null;
        int indexPV = RawString.IndexOf(" pv ");
        string stripped = RawString[(indexPV + 4)..];
        if (stripped.Contains(" string "))
        {
          stripped = stripped.Substring(0, stripped.IndexOf(" string"));
        }
        return stripped;

      }
    }

    /// <summary>
    /// Returns a string with sequence of moves in the principal variation in SAN format.
    /// </summary>
    /// <param name="position"></param>
    /// <returns></returns>
    public string PVMovesSAN(Position position)
    {
      return ParsedPVString(position, PVString).pvStringSAN;
    }

    /// <summary>
    /// Returns a List of Move containing the principal variation moves.
    /// </summary>
    /// <param name="position"></param>
    /// <returns></returns>
    public List<Move> PVMoves(Position position)
    {
      return ParsedPVString(position, PVString).moves;
    }


    /// <summary>
    /// Returns the score as a logistic (winning percentage).
    /// </summary>
    public float ScoreLogistic
    {
      get
      {
        // TODO: make more elegant via subclassing as noted above
        // TODO: consider switching to the LC0 2019 variant formula
        return ExtraInfo is VerboseMoveStats
            ? ((VerboseMoveStats)ExtraInfo).ScoreCentipawns
            : EncodedEvalLogistic.CentipawnToLogistic(ScoreCentipawns);
      }
    }


    /// <summary>
    /// Dumps all lines of info to Console.
    /// </summary>
    public void Dump()
    {
      // TODO: make more elegant via subclassing as noted above
      if (ExtraInfo is VerboseMoveStats)
      {
        ((VerboseMoveStats)ExtraInfo).Dump();
      }
    }

    #region Static helpers

    /// <summary>
    /// Parses the moves in principal variation in an UCI move output line.
    /// TODO: handle multipv lines?
    /// </summary>
    static (List<Move> moves, string pvStringSAN) ParsedPVString(Position position, string pvString)
    {
      // Split off starting at the PV moves.
      string[] movesSplit = pvString.Split(" ");

      int count = 0;
      MGPosition curPos = position.ToMGPosition;
      List<Move> movesList = new();
      StringBuilder allMovesStr = new();

      foreach (string thisMoveStr in movesSplit)
      {
        try
        {
          MGMove mgMove = MGMoveFromString.ParseMove(in curPos, thisMoveStr);
          string moveNumStr = curPos.BlackToMove ? (count == 0 ? " .. " : " ") : $" {1 + curPos.MoveNumber / 2}. ";
          Move move = MGMoveConverter.ToMove(mgMove);
          movesList.Add(move);
          string moveSAN = move.ToSAN(curPos.ToPosition);
          string moveStr = moveNumStr + moveSAN;
          allMovesStr.Append(moveStr);
          curPos.MakeMove(mgMove);
          count++;
        }
        catch (Exception e)
        {
          // Silently ignore errors in parsing
          break;
        }
      }

      return (movesList, allMovesStr.ToString());
    }

    #endregion
  }

}
