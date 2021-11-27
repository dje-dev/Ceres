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
using System.IO;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes.Analysis
{
  /// <summary>
  /// Static helper methods for dumping contents of tree to Console.
  /// </summary>
  public static class MCTSPosTreeNodeDumper
  {
    static void DumpWithColor(float v, string s, float thresholdRed, float thresholdGreen, TextWriter writer)
    {
      if (writer != Console.Out)
      {
        // No point in using color if not the Console
        writer.Write(s);
      }
      else
      {
        ConsoleColor defaultColor = Console.ForegroundColor;

        try
        {
          if (v > thresholdGreen)
          {
            Console.ForegroundColor = ConsoleColor.Green;
          }
          else if (v < thresholdRed)
          {
            Console.ForegroundColor = ConsoleColor.Red;
          }

          writer.Write(s);
        }
        finally
        {
          Console.ForegroundColor = defaultColor;
        }
      }
    }


    static void DumpNodeStr(MCTSNode searchRootNode, MCTSNode node,
                            int countTimesSeen, bool fullDetail, TextWriter writer = null)
    {
      int depth = node.Depth - searchRootNode.Depth;

      writer = writer ?? Console.Out;

      node.Tree.Annotate(node);

      char extraChar = ' ';
      switch (node.Terminal)
      {
        case GameResult.Checkmate:
          extraChar = 'C';
          break;
        case GameResult.Draw:
          extraChar = 'D';
          break;
        default:
          if (countTimesSeen > 1)
          {
            extraChar = countTimesSeen > 9 ? '9' : countTimesSeen.ToString()[0];
          }

          break;
      }

      float multiplier = 1; // currently alternate perspective

      float pctOfVisits = node.IsRoot ? 100.0f : (100.0f * node.N / node.Parent.N);

      MCTSNode bestMove = default;
      // TODO: someday show this too      MCTSNode nextBestMove = null;
      if (!node.IsRoot)
      {
        MCTSNode[] parentsChildrenSortedQ = node.ChildrenSorted(innerNode => -multiplier * (float)innerNode.Q);
        if (parentsChildrenSortedQ.Length > 0)
        {
          bestMove = parentsChildrenSortedQ[0];
        }
        //        if (parentsChildrenSortedQ.Length > 1) nextBestMove = parentsChildrenSortedQ[1];
      }

      // Depth, move
      writer.Write($"{depth,3}. ");
      writer.Write(extraChar);
      writer.Write($" {node.NumPolicyMoves,3} ");

      writer.Write($"{node.Index,13:N0}");

      string san = node.IsRoot ? "" : MGMoveConverter.ToMove(node.Annotation.PriorMoveMG).ToSAN(in node.Parent.Annotation.Pos);
      //      string sanNextBest = node.IsRoot ? "" : MGMoveConverter.ToMove(nextBestMove.Annotation.PriorMoveMG).ToSAN(in node.Parent.Annotation.Pos);
      if (node.Annotation.Pos.MiscInfo.SideToMove == SideType.White)
      {
        writer.Write($"      ");
        writer.Write($"{san,6}");

      }
      else
      {
        writer.Write($"{san,6}");
        writer.Write($"      ");
      }

      //      float diffBestNextBestQ = 0;
      //      if (nextBestMove != null) diffBestNextBestQ = (float)(bestMove.Q - nextBestMove.Q);
      //      writer.Write($"{  (nextBestMove?.Annotation == null ? "" : nextBestMove.Annotation.PriorMoveMG.ToString()),8}");
      //      writer.Write($"{diffBestNextBestQ,8:F2}");


      writer.Write($"{node.N,13:N0} ");
      writer.Write($" {pctOfVisits,5:F0}%");
      string secondaryChar = node.StructRef.SecondaryNN ? "s" : " ";
      writer.Write($"   {100.0 * node.P,6:F2}%{secondaryChar} ");
      DumpWithColor(multiplier * node.V, $" {multiplier * node.V,6:F3}  ", -0.2f, 0.2f, writer);
      //      DumpWithColor(multiplier * node.VSecondary, $" {multiplier * node.VSecondary,6:F3} ", -0.2f, 0.2f);
      double q = multiplier * node.Q;
      DumpWithColor((float)q, $" {q,6:F3} ", -0.2f, 0.2f, writer);

      //      float qStdDev = MathF.Sqrt(node.Ref.VVariance);
      //      if (float.IsNaN(qStdDev))
      //        writer.WriteLine("found negative var");
      //      writer.Write($" +/-{qStdDev,5:F2}  ");

      bool invert = multiplier == -1;
      writer.Write($" {(invert ? node.LossP : node.WinP),5:F2}/{node.DrawP,5:F2}/{(invert ? node.WinP : node.LossP),5:F2}  ");

      writer.Write($" {(invert ? node.LAvg : node.WAvg),5:F2}/{node.DAvg,5:F2}/{(invert ? node.WAvg : node.LAvg),5:F2}  ");

      //      writer.Write($"   {node.Ref.QUpdatesWtdAvg,5:F2}  ");
      //      writer.Write($" +/-:{MathF.Sqrt(node.Ref.QUpdatesWtdVariance),5:F2}  ");
      //      writer.Write($" {node.Ref.TrendBonusToP,5:F2}  ");

      writer.Write($" {node.MPosition,3:F0} ");
      writer.Write($" {node.MAvg,3:F0}  ");

      if (fullDetail)
      {
        int numPieces = node.Annotation.Pos.PieceCount;

        //        writer.Write($" {PosStr(node.Annotation.Pos)} ");
        writer.Write($" {node.Annotation.Pos.FEN}");
      }


      writer.WriteLine();
    }

    private static void WriteHeaders(bool fullDetail, TextWriter writer)
    {
      // Write headers
      writer.Write(" Dep  T #M    FirstVisit   MvWh  MvBl           N  Visits    Policy     V         Q     WPos  DPos  LPos   WTree DTree LTree  MPos MTree");
      if (fullDetail)
      {
        writer.WriteLine("  FEN");
      }
      else
      {
        writer.WriteLine();
      }

      if (fullDetail)
      {
        writer.WriteLine("----  - --  ------------  -----  ---- -----------  ------  --------  -------    -----   ----  ----  ----   -----  ---- -----  ---- -----");
      }
      else
      {
        writer.WriteLine();
      }
    }


    static string ToStr(char pieceChar, int count, int maxCount)
    {
      string ret = "";

      if (count > maxCount)
      {
        // Rare case of more than expected number of pieces (due to promotion)
        ret += pieceChar + count.ToString();
        for (int i = 0; i < maxCount - 2; i++)
        {
          ret += " ";
        }
      }
      else
      {
        for (int i = 0; i < maxCount; i++)
        {
          if (count > i)
          {
            ret += pieceChar;
          }
          else
          {
            ret += " ";
          }
        }
      }
      return ret;
    }


    /// <summary>
    /// Returns a compact string describing the number of each type of piece on
    /// on the board in the position. Example: "K R BBN PPPPPPP   k r bbn ppppppp"
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    static string PosStr(Position pos)
    {
      string str = "K";

      str += ToStr('Q', pos.PieceCountOfType(new Piece(SideType.White, PieceType.Queen)), 1);
      str += ToStr('R', pos.PieceCountOfType(new Piece(SideType.White, PieceType.Rook)), 2);
      str += ToStr('B', pos.PieceCountOfType(new Piece(SideType.White, PieceType.Bishop)), 2);
      str += ToStr('N', pos.PieceCountOfType(new Piece(SideType.White, PieceType.Knight)), 2);
      str += ToStr('P', pos.PieceCountOfType(new Piece(SideType.White, PieceType.Pawn)), 8);

      str += "  k";
      str += ToStr('q', pos.PieceCountOfType(new Piece(SideType.Black, PieceType.Queen)), 1);
      str += ToStr('r', pos.PieceCountOfType(new Piece(SideType.Black, PieceType.Rook)), 2);
      str += ToStr('b', pos.PieceCountOfType(new Piece(SideType.Black, PieceType.Bishop)), 2);
      str += ToStr('n', pos.PieceCountOfType(new Piece(SideType.Black, PieceType.Knight)), 2);
      str += ToStr('p', pos.PieceCountOfType(new Piece(SideType.Black, PieceType.Pawn)), 8);

      return str;
    }


    public static void DumpAllNodes(MCTSIterator context, ref MCTSNodeStruct node,
                                    Base.DataType.Trees.TreeTraversalType type = Base.DataType.Trees.TreeTraversalType.BreadthFirst,
                                    bool childDetail = false)
    {
      int index = 1;

      // Visit all nodes and verify various conditions are true
      node.Traverse(context.Tree.Store,
                          (ref MCTSNodeStruct node) =>
                          {
                            Console.WriteLine(index + " " + node);
                            if (childDetail)
                            {
                              int childIndex = 0;
                              foreach (MCTSNodeStructChild childInfo in node.Children)
                                Console.WriteLine($"  {childIndex++,3} {childInfo}");
                            }
                            index++;
                            return true;
                          }, type);
    }


    static MCTSNode DescendMovesToNode(MCTSNode rootNode, List<MGMove> moves)
    {
      MCTSNode node = rootNode;

      foreach (MGMove move in moves)
      {
        int? childIndex = node.StructRef.ChildIndexWithMove(move);
        if (childIndex is null)
        {
          throw new Exception($"Move  {move} not found at node {node}");
        }
        else
        {
          (node, _, _) = node.ChildAtIndexInfo(childIndex.Value);
        }
      }
      return node;
    }


    public static void DumpPV(MCTSNode node, bool fullDetail, TextWriter writer = null)
    {
      try
      {
        using (new SearchContextExecutionBlock(node.Context))
        {
          writer = writer ?? Console.Out;

          writer.WriteLine();
          WriteHeaders(fullDetail, writer);

          List<Position> seenPositions = new List<Position>();

          int CountDuplicatePos(Position pos)
          {
            int count = 0;
            foreach (Position priorPos in seenPositions)
            {
              if (pos.EqualAsRepetition(in priorPos))
              {
                count++;
              }
            }

            return count;
          }

          int depth = 0;
          MCTSNode searchRootNode = node;
          while (true)
          {
            node.Tree.Annotate(node);
            seenPositions.Add(node.Annotation.Pos);
            int countSeen = CountDuplicatePos(node.Annotation.Pos);

            DumpNodeStr(searchRootNode, node, countSeen, fullDetail, writer);

            if (node.NumChildrenVisited == 0)
            {
              return;
            }

            node = node.BestMove(false);

            depth++;
          }
        }
      }
      catch (Exception e)
      {
        Console.WriteLine($"Dump failed with message: {e.Message}");
        writer.Write($"Dump failed with message: {e.Message}");
        Console.WriteLine($"Stacktrace: {e.StackTrace}");
        writer.Write($"Stacktrace: {e.StackTrace}");
        //Console.WriteLine($"Inner exception: {e.InnerException.Message}");
        //writer.Write($"Inner exception: {e.InnerException.Message}");
        //throw;
      }
    }

  }

}
