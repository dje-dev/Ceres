#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Features.Visualization.AnalysisGraph
{
  /// <summary>
  /// Generates SVG files (via Graphviz) containing graphical comparison 
  /// of "split points" in games where play deviated in a set of two or more games.
  /// </summary>
  public class GameCompareGraphGenerator
  {
    const string BASE_FN = "CeresGameCompGraph";

    /// <summary>
    /// The temporary directory in which all output files are placed.
    /// </summary>
    public readonly string TempDir;

    /// <summary>
    /// StringBuilder that accumulates generated text.
    /// </summary>
    StringBuilder dot = new();

    List<PGNGame> games;
    Predicate<PGNGame> gameFilter;
    Func<PGNGame, IComparable> gameGroupBy;

    public GameCompareGraphGenerator(List<PGNGame> games, Predicate<PGNGame> gameFilter, Func<PGNGame, IComparable> gampGroupBy)
    {
      TempDir = GraphvizUtils.CreateUniqueTempDirectory("Ceres_GameComp_Graph_");
      this.games = games;
      this.gameFilter = gameFilter;
      this.gameGroupBy = gampGroupBy;
    }

    /// <summary>
    /// Gererates SVG with game comparison details, returning FN of generated file.
    /// </summary>
    /// <param name="launchWithBrowser"></param>
    /// <returns></returns>
    public string Write(bool launchWithBrowser)
    {
      List<PGNGame> gamesMatching = games.Where(g => gameFilter(g)).ToList();
      if (gamesMatching.Count == 0)
      {
        Console.WriteLine("GameCompareGraphGenerator: no matching games");
        return null;
      }
      else
      {
        EmitHeader();
        foreach (IGrouping<IComparable, PGNGame> group in gamesMatching.GroupBy(gameGroupBy))
        {
          List<PGNGame> theseGames = group.ToList();
          if (theseGames.Count > 0)
          {
            GenSubgraph(theseGames);
          }
        }

        EmitFooter();

        // Covvert to SVG.
        GraphvizUtils.Convert(dot.ToString(), BASE_FN, TempDir);
        string fn = Path.Combine(TempDir, $"{BASE_FN}.svg");

        if (launchWithBrowser)
        {
          StringUtils.LaunchBrowserWithURL(fn);
        }
        return fn;
      }
    }

    void EmitHeader()
    {
      dot.AppendLine($"digraph Games" + "\r\n{");
      dot.AppendLine("rankdir=LR;fontsize=\"18\";");
    }

    void EmitFooter()
    {
      dot.AppendLine("}");
    }


    int graphIndex = 0;

    void GenSubgraph(List<PGNGame> games)
    {
      dot.AppendLine($"subgraph Games_{graphIndex++}" + "\r\n{");
      dot.AppendLine("rankdir=TB;fontsize=\"18\";");

      int lastDepth = -1;
      Stack<PGNCommonMoves.PGNCommonMoveSequence> stack = new();

      foreach (PGNCommonMoves.PGNCommonMoveSequence thisSet in PGNCommonMoves.PGNCommonMoveSequences(null, games, 0))
      {
        while (stack.Count() > 0 && stack.Peek().depth >= thisSet.depth)
        {
          // End this set.
          stack.Pop();
          WriteIndent(thisSet.depth);
          dot.AppendLine("}");
        }

        if (stack.Count > 0)
        {
          // Start a new set.
          WriteIndent(thisSet.depth);
          var lastSet = stack.Peek();
          dot.AppendLine(ToLabel(lastSet, lastSet.startMoveNum + lastSet.numMoves - 1) + "->" + ToLabel(thisSet, thisSet.startMoveNum));
        }

        stack.Push(thisSet);
        WriteIndent(thisSet.depth);
        dot.AppendLine($" subgraph cluster_{thisSet.id}");
        WriteIndent(thisSet.depth);
        dot.AppendLine("{");
        WriteIndent(thisSet.depth);

        int moveNum = 1 + (thisSet.startMoveNum / 2);
        string moveStr = moveNum.ToString();
        bool whiteToMove = thisSet.newGames[0].Moves.Moves[thisSet.startMoveNum].WhiteToMove;
        string label = GetPlayerLabels(thisSet.newGames, whiteToMove, moveStr, out string bgColorStr);

        dot.AppendLine($"label=\"{ label}\" bgcolor=\"{bgColorStr}\"");
        int lastMoveIndexInSequence = thisSet.startMoveNum + thisSet.numMoves - 1;
        bool haveWrittenInnerSet = false;

        string lastLabel = "";
        for (int i = thisSet.startMoveNum; i <= lastMoveIndexInSequence; i++)
        {
          bool isInner = i != thisSet.startMoveNum
                      && i != lastMoveIndexInSequence;
          string thisLabel = ToLabel(thisSet, i);
          if (!isInner)
          {
            WriteIndent(thisSet.depth);
            dot.AppendLine(thisLabel);
            PositionWithHistory moves = thisSet.newGames[0].Moves;
            MGMove priorMove = default;
            if (i > 0)
            {
              priorMove = moves.Moves[i - 1];
            }
            
            Position thisPos = moves.GetPositions()[i];
            string[] descriptions = new string[thisSet.newGames.Count];
            for (int engineNum = 0; engineNum < descriptions.Length; engineNum++)
            {
              PGNGame thisGame = thisSet.newGames[engineNum];
              float evalCP = i == 0 ? float.NaN : thisGame.MovePlayerEvalCP(i - 1);
              float evalTime = i == 0 ? float.NaN : thisGame.MoveTimeSeconds(i - 1);
              ulong evalNodes = i == 0 ? 0 : thisGame.MoveNodes(i - 1);
              if (!float.IsNaN(evalCP) || !float.IsNaN(evalTime))
              {
                StringBuilder moveInfo = new();
                moveInfo.Append((thisPos.MiscInfo.SideToMove == SideType.White ? thisGame.BlackPlayer : thisGame.WhitePlayer) + "|");
                moveInfo.Append($" {evalTime:F2} s |");
                moveInfo.Append(evalNodes == 0 ? "" : $" {NodesStr(evalNodes):N0} |");
                moveInfo.Append($" {evalCP:F0} cp ");
                descriptions[engineNum] = moveInfo.ToString();
              }
            }

            string firstLine = null;
            if (i > 0)
            {
              Position priorPos = moves.GetPositions()[i - 1];
              string moveSAN = MGMoveConverter.ToMove(priorMove).ToSAN(priorPos);
              int showMoveNum = (1 + priorPos.MiscInfo.MoveNum) / 2;
              firstLine = "after " + showMoveNum + (thisPos.MiscInfo.SideToMove == SideType.White ? ".." : ". ") + moveSAN + " played by";
            }
            dot.Append(PosNodeInfo(thisPos, priorMove, firstLine, descriptions));

            if (i > thisSet.startMoveNum)
            {
              WriteIndent(thisSet.depth);
              dot.AppendLine(lastLabel + "->" + thisLabel);
            }
            lastLabel = thisLabel;
          }
          else if (!haveWrittenInnerSet)
          {
            WriteIndent(thisSet.depth);
            thisLabel = "Moves_" + thisSet.id + "_" + (thisSet.startMoveNum + i) + "_to_" + (lastMoveIndexInSequence - 1);
            int numMoves = lastMoveIndexInSequence - thisSet.startMoveNum;
            dot.AppendLine(thisLabel + $" [label=\"{numMoves} ply\", fontsize=\"24\"]");

            dot.AppendLine(lastLabel + "->" + thisLabel);
            haveWrittenInnerSet = true;
            lastLabel = thisLabel;
          }
        }

        dot.AppendLine();
        lastDepth = thisSet.depth;
      }

      while (stack.Count > 0)
      {
        var thisSet = stack.Pop();
        for (int i = 0; i < 2 + thisSet.depth * 2; i++) dot.Append(" ");
        dot.AppendLine("}");
      }

      dot.AppendLine("}");

    }



    #region Helper methods

    /// <summary>
    /// Returns compact string representing (possibly approximate) number of nodes.
    /// </summary>
    static string NodesStr(ulong nodes)
    {
      if (nodes < 1_000_000)
      {
        return nodes.ToString();
      }
      else if (nodes < (1_000_000 * 1000))
      {
        return $"{ nodes / (float)1_000_000f:F2}m";
      }
      else
      {
        return $"{ nodes / ((float)1_000_000f * 1000):F2}b";
      }
    }

    static int indexWrittenPos = 0;


    string PosNodeInfo(Ceres.Chess.Position pos, MGMove mgMove, string firstLine, string[] extraLines)
    {

      StringBuilder ret = new();
      ret.Append($"[shape=none,");
      ret.Append($"href=\"https://lichess.org/editor/{pos.FEN}\"");

      Ceres.Chess.Move move = MGMoveConverter.ToMove(mgMove);
      string posFN = GraphvizUtils.WritePosToSVGFile(in pos,
                                                     pos.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98),
                                                     indexWrittenPos++, move, Path.GetTempPath());

      //sb.AppendLine($"tooltip=\"{tooltip}]\"");

      ret.Append($" fontsize=\"18\"");

      string colorAttr = pos.MiscInfo.SideToMove == SideType.White ? "bgcolor=\"white\"" : "bgcolor=\"silver\"";
      ret.AppendLine("label= <<TABLE cellspacing=\"0\">");
      ret.AppendLine($"<TR><TD border=\"0\" {colorAttr} ><IMG SRC=\"{posFN}\"/></TD></TR>");
      string bgColorScore = GraphvizUtils.ColorStr(0);//(float)node.Q);// TO DO

      if (firstLine != null)
      {
        ret.AppendLine($"<TR><TD>{firstLine}</TD></TR>");
      }

      if (extraLines != null)
      {
        for (int i = 0; i < extraLines.Length; i++)
        {
          if (extraLines[i] != null)
          {
            ret.Append($"<TR>");
            foreach (string col in extraLines[i].Split("|"))
            {
              ret.Append($"<TD>{col}</TD>");
            }
            ret.Append("</TR>");
            ret.AppendLine();
          }
        }
      }
      ret.AppendLine("</TABLE>>");


      ret.Append("]");
      return ret.ToString();
    }

    void WriteIndent(int depth)
    {
      for (int i = 0; i < 2 + depth * 2; i++)
      {
        dot.Append(" ");
      }
    }

    static string ToLabel(PGNCommonMoves.PGNCommonMoveSequence set, int index) => "Pos" + set.id + "_" + index;

    static string TransResult(PGNGame.GameResult result)
    {
      if (result == PGNGame.GameResult.WhiteWins)
        return "1-0";
      else if (result == PGNGame.GameResult.BlackWins)
        return "0-1";
      else if (result == PGNGame.GameResult.Draw)
        return "=";
      else
        return "?";
    }


    static string GetPlayerLabels(List<PGNGame> games, bool isWhite, string moveNumString, out string bgColor)
    {
      StringBuilder ret = new StringBuilder();
      int count = 0;
      int countWhiteWin = 0;
      int countWhiteLose = 0;
      foreach (var game in games)
      {
        if (count++ > 0)
        {
          ret.Append("\n");
        }

        // Determine which player deviated first
        string whiteDevChar = isWhite ? "" : " (DEV)";
        string blackDevChar = isWhite ? " (DEV)" : "";
        string roundStr = game.Round == 0 ? "" : $"Round {game.Round} ";
        ret.Append($"{roundStr}Game {game.GameIndex + 1,4} move {moveNumString} { game.WhitePlayer,15}{whiteDevChar,5}"
                  + " / " + $"{ game.BlackPlayer,15}{blackDevChar,5} ({TransResult(game.Result)})");

        if (game.Result == PGNGame.GameResult.WhiteWins)
        {
          countWhiteWin++;
        }
        else if (game.Result == PGNGame.GameResult.BlackWins)
        {
          countWhiteLose++;
        }
      }


      if (countWhiteWin == games.Count)
      {
        bgColor = GraphvizUtils.ColorStr(0.10f);
      }
      else if (countWhiteLose == games.Count)
      {
        bgColor = GraphvizUtils.ColorStr(-0.10f);
      }
      else
      {
        bgColor = "whitesmoke";
      }

      return ret.ToString();
    }

    #endregion

  }

}
