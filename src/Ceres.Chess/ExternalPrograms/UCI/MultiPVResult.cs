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

using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.ExternalPrograms.UCI
{
  /// <summary>
  /// A single principal variation line from a MultiPV UCI search.
  /// Scores are expressed from the perspective of the side to move in the searched position
  /// (i.e. positive means good for the side to move), exactly as emitted by the engine.
  /// </summary>
  /// <param name="MultiPVIndex">1-based rank of this line (1 = best).</param>
  /// <param name="Move">First move of the line (the candidate move being scored).</param>
  /// <param name="MoveUCI">UCI string of the first move.</param>
  /// <param name="ScoreCentipawns">Evaluation in centipawns (mates encoded as large values).</param>
  /// <param name="MateInMoves">Non-zero if this is a forced mate (positive = mate for side to move).</param>
  /// <param name="Depth">Search depth at which this line was reported.</param>
  public record MultiPVLine(int MultiPVIndex, Move Move, string MoveUCI,
                            int ScoreCentipawns, int MateInMoves, int Depth)
  {
    public bool IsMate => MateInMoves != 0;
  }


  /// <summary>
  /// Parsed result of a MultiPV UCI search: the ordered set of best lines plus
  /// the depth at which the best move stabilized (a proxy for position difficulty).
  ///
  /// Produced by parsing the per-multipv "info" lines retained by <see cref="UCIGameRunner"/>
  /// during the most recent search.
  /// </summary>
  /// <param name="Lines">Lines ordered by MultiPV index ascending (best first).</param>
  /// <param name="StabilizationDepth">Smallest depth from which the best move never changed again
  /// (lower = easier to find; equal to MaxDepth if the best move was unstable to the end).</param>
  /// <param name="MaxDepth">Maximum reported search depth.</param>
  /// <param name="StabilizationNodes">Node count reported at the depth where the best move last
  /// changed to its final value (i.e. how many nodes it took the engine to settle). In a single-PV
  /// search this is a far more faithful difficulty signal than under MultiPV.</param>
  /// <param name="EvalJumpCp">Signed change in evaluation (centipawns) across the final best-move
  /// switch: eval at the stabilization depth minus eval at the depth immediately before it. Large
  /// magnitude means the engine's assessment changed sharply when it found the move (a critical
  /// resource). Zero if the best move never changed.</param>
  public record MultiPVResult(IReadOnlyList<MultiPVLine> Lines, int StabilizationDepth, int MaxDepth,
                              int StabilizationNodes = 0, int EvalJumpCp = 0)
  {
    /// <summary>
    /// The best (top ranked) line, or null if none were parsed.
    /// </summary>
    public MultiPVLine Best => Lines.Count > 0 ? Lines[0] : null;

    /// <summary>
    /// Centipawn margin between the best and second-best move (the discrimination "delta").
    /// Returns int.MaxValue if fewer than two lines are available.
    /// </summary>
    public int MarginToSecondCp => Lines.Count >= 2 ? Lines[0].ScoreCentipawns - Lines[1].ScoreCentipawns : int.MaxValue;


    /// <summary>
    /// Parses the per-multipv info lines (and best-line history) captured during a search
    /// into an ordered MultiPVResult.
    /// </summary>
    /// <param name="linesByMultiPV">Most recent info line for each multipv index.</param>
    /// <param name="bestLineHistory">Sequence of multipv-1 info lines, in arrival order.</param>
    /// <param name="pos">The position that was searched (used to decode UCI moves).</param>
    public static MultiPVResult Parse(IReadOnlyDictionary<int, string> linesByMultiPV,
                                      IEnumerable<string> bestLineHistory,
                                      Position pos)
    {
      List<MultiPVLine> lines = new();

      if (linesByMultiPV != null)
      {
        foreach (KeyValuePair<int, string> kv in linesByMultiPV.OrderBy(k => k.Key))
        {
          MultiPVLine line = ParseLine(kv.Key, kv.Value, pos);
          if (line != null)
          {
            lines.Add(line);
          }
        }
      }

      int maxDepth = lines.Count > 0 ? lines.Max(l => l.Depth) : 0;

      // Reconstruct the per-depth history of the best (multipv 1) move with its node count and eval.
      List<(int depth, string mv, int cp, int nodes)> history = new();
      if (bestLineHistory != null)
      {
        foreach (string h in bestLineHistory)
        {
          string mv = UCIInfoParse.FirstPVMoveUCI(h);
          if (mv == null)
          {
            continue;
          }
          int depth = UCIInfoParse.IntAfter(h, "depth");
          int nodes = UCIInfoParse.IntAfter(h, "nodes");
          history.Add((depth == int.MinValue ? 0 : depth, mv, UCIInfoParse.ScoreCp(h), nodes == int.MinValue ? 0 : nodes));
        }
      }

      string finalBestMove = lines.Count > 0 ? lines[0].MoveUCI : (history.Count > 0 ? history[^1].mv : null);

      // Find the start of the trailing run during which the best move stayed at its final value.
      int stabilizationDepth = maxDepth;
      int stabilizationNodes = 0;
      int evalJumpCp = 0;
      if (finalBestMove != null && history.Count > 0)
      {
        int stabIdx = history.Count - 1;
        for (int i = history.Count - 1; i >= 0; i--)
        {
          if (history[i].mv == finalBestMove)
          {
            stabIdx = i;
          }
          else
          {
            break;
          }
        }

        stabilizationDepth = history[stabIdx].depth;
        stabilizationNodes = history[stabIdx].nodes;

        // Eval change across the switch: eval once settled vs eval of the (different) move just before.
        int evalAtStab = history[stabIdx].cp;
        int evalBeforeSwitch = stabIdx > 0 ? history[stabIdx - 1].cp : evalAtStab;
        evalJumpCp = evalAtStab - evalBeforeSwitch;
      }

      return new MultiPVResult(lines, stabilizationDepth, maxDepth, stabilizationNodes, evalJumpCp);
    }


    static MultiPVLine ParseLine(int multiPVIndex, string line, Position pos)
    {
      if (line == null)
      {
        return null;
      }

      string moveUCI = UCIInfoParse.FirstPVMoveUCI(line);
      if (moveUCI == null)
      {
        return null;
      }

      Move move;
      try
      {
        move = Move.FromUCI(pos, moveUCI);
      }
      catch
      {
        // Unparsable move (e.g. malformed/partial line); skip.
        return null;
      }

      int depth = UCIInfoParse.IntAfter(line, "depth");
      if (depth == int.MinValue)
      {
        depth = 0;
      }

      int mate = UCIInfoParse.IntAfter(line, "mate");
      if (mate == int.MinValue)
      {
        mate = 0;
      }
      int cp = UCIInfoParse.ScoreCp(line);

      return new MultiPVLine(multiPVIndex, move, moveUCI, cp, mate, depth);
    }
  }


  /// <summary>
  /// Lightweight token-based helpers for parsing UCI "info" lines.
  /// </summary>
  internal static class UCIInfoParse
  {
    /// <summary>
    /// Returns the multipv index reported on a line, defaulting to 1 if not present
    /// (engines searching with MultiPV=1 may omit the token).
    /// </summary>
    public static int ExtractMultiPVIndex(string line)
    {
      int idx = IntAfter(line, "multipv");
      return idx == int.MinValue ? 1 : idx;
    }


    /// <summary>
    /// Returns the integer immediately following the first occurrence of <paramref name="token"/>,
    /// or int.MinValue if the token (followed by an integer) is not found.
    /// </summary>
    public static int IntAfter(string line, string token)
    {
      if (line == null)
      {
        return int.MinValue;
      }

      string[] tokens = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
      for (int i = 0; i < tokens.Length - 1; i++)
      {
        if (tokens[i] == token && int.TryParse(tokens[i + 1], out int value))
        {
          return value;
        }
      }

      return int.MinValue;
    }


    /// <summary>
    /// Returns the evaluation of an info line in centipawns, encoding "score mate N" as a large
    /// centipawn value (same convention as UCISearchInfo). Returns 0 if no score is present.
    /// </summary>
    public static int ScoreCp(string line)
    {
      int mate = IntAfter(line, "mate");
      if (mate != int.MinValue)
      {
        return mate > 0 ? 2000 - mate * 5 : -2000 + mate * 5;
      }
      int cp = IntAfter(line, "cp");
      return cp == int.MinValue ? 0 : cp;
    }


    /// <summary>
    /// Returns the UCI string of the first move in the principal variation of an info line,
    /// or null if the line contains no " pv " segment.
    /// </summary>
    public static string FirstPVMoveUCI(string line)
    {
      if (line == null)
      {
        return null;
      }

      int pvPos = line.IndexOf(" pv ");
      if (pvPos < 0)
      {
        return null;
      }

      string rest = line.Substring(pvPos + 4).Trim();
      int spacePos = rest.IndexOf(' ');
      return spacePos < 0 ? rest : rest.Substring(0, spacePos);
    }
  }
}
