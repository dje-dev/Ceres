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
using System.Text;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Struct;
using SharpCompress.Compressors.Xz;

#endregion

namespace Ceres.MCTS.MTCSNodes.Analysis
{
  /// <summary>
  /// Static helper methods for dumping contents of tree to Console.
  /// </summary>
  public static class MCTSPosTreeNodeDumper
  {
    static void DumpWithColor(float v, string s, float thresholdRed, float thresholdGreen)
    {
      ConsoleColor defaultColor = Console.ForegroundColor;

      try
      {
        if (v > thresholdGreen)
          Console.ForegroundColor = ConsoleColor.Green;
        else if (v < thresholdRed)
          Console.ForegroundColor = ConsoleColor.Red;

        Console.Write(s);
      }
      finally
      {
        Console.ForegroundColor = defaultColor;
      }
    }


    static void DumpNodeStr(PositionWithHistory priorMoves, MCTSNode node, int depth, int countTimesSeen, bool fullDetail)
    {
      node.Context.Tree.Annotate(node);

      char extraChar = ' ';
      if (node.Terminal == GameResult.Checkmate)
        extraChar = 'C';
      else if (node.Terminal == GameResult.Draw)
        extraChar = 'D';
      else if (countTimesSeen > 1)
        extraChar = countTimesSeen > 9 ? '9' : countTimesSeen.ToString()[0];

      float multiplier = depth % 2 == 0 ? 1.0f : -1.0f;

      float pctOfVisits = node.IsRoot ? 100.0f : (100.0f * node.N / node.Parent.N);

      MCTSNode bestMove = null;
// TODO: someday show this too      MCTSNode nextBestMove = null;
      if (!node.IsRoot)
      {
        MCTSNode[] parentsChildrenSortedQ = node.ChildrenSorted(innerNode => -multiplier * (float)innerNode.Q);
        if (parentsChildrenSortedQ.Length > 0) bestMove = parentsChildrenSortedQ[0];
//        if (parentsChildrenSortedQ.Length > 1) nextBestMove = parentsChildrenSortedQ[1];
      }

      // Depth, move
      Console.Write($"{depth,3}. ");
      Console.Write(extraChar);
      Console.Write($" {node.NumPolicyMoves,3} ");

      Console.Write($"{node.Index,13:N0}");

      string san         = node.IsRoot ? "" : MGMoveConverter.ToMove(node        .Annotation.PriorMoveMG).ToSAN(in node.Parent.Annotation.Pos);
//      string sanNextBest = node.IsRoot ? "" : MGMoveConverter.ToMove(nextBestMove.Annotation.PriorMoveMG).ToSAN(in node.Parent.Annotation.Pos);
      if (node.Annotation.Pos.MiscInfo.SideToMove == SideType.White)
      {
        Console.Write($"      ");
        Console.Write($"{san,6}");

      }
      else
      {
        Console.Write($"{san,6}");
        Console.Write($"      ");
      }

//      float diffBestNextBestQ = 0;
//      if (nextBestMove != null) diffBestNextBestQ = (float)(bestMove.Q - nextBestMove.Q);
//      Console.Write($"{  (nextBestMove?.Annotation == null ? "" : nextBestMove.Annotation.PriorMoveMG.ToString()),8}");
//      Console.Write($"{diffBestNextBestQ,8:F2}");


      Console.Write($"{node.N,13:N0} ");
      Console.Write($" {pctOfVisits,5:F0}%");
      Console.Write($"   {100.0 * node.P,6:F2}%  ");
      DumpWithColor(multiplier * node.V, $" {multiplier * node.V,6:F3}  ", -0.2f, 0.2f);
//      DumpWithColor(multiplier * node.VSecondary, $" {multiplier * node.VSecondary,6:F3} ", -0.2f, 0.2f);
      double q = multiplier * node.Q;
      DumpWithColor((float)q, $" {q,6:F3} ", -0.2f, 0.2f);

      //      float qStdDev = MathF.Sqrt(node.Ref.VVariance);
      //      if (float.IsNaN(qStdDev))
      //        Console.WriteLine("found negative var");
      //      Console.Write($" +/-{qStdDev,5:F2}  ");

      Console.Write($" {node.WinP,5:F2}/{node.DrawP,5:F2}/{node.LossP,5:F2}  ");

      Console.Write($" {node.WAvg,5:F2}/{node.DAvg,5:F2}/{node.LAvg,5:F2}  ");

      //      Console.Write($"   {node.Ref.QUpdatesWtdAvg,5:F2}  ");
      //      Console.Write($" +/-:{MathF.Sqrt(node.Ref.QUpdatesWtdVariance),5:F2}  ");
      //      Console.Write($" {node.Ref.TrendBonusToP,5:F2}  ");

      Console.Write($" {node.MPosition,3:F0} ");
      Console.Write($" {node.MAvg,3:F0}  ");

      if (fullDetail)
      {
        int numPieces = node.Annotation.Pos.PieceCount;

//        Console.Write($" {PosStr(node.Annotation.Pos)} ");
        Console.Write($" {node.Annotation.Pos.FEN}");
      }


      Console.WriteLine();
    }

    private static void WriteHeaders(bool fullDetail)
    {
      // Write headers
      Console.Write(" Dep  T #M    FirstVisit   MvWh  MvBl           N  Visits    Policy     V         Q     WPos  DPos  LPos   WTree DTree LTree  MPos MTree");
      if (fullDetail)
        Console.WriteLine("  FEN");
      else
        Console.WriteLine();

      if (fullDetail)
        Console.WriteLine("----  - --  ------------  -----  ---- -----------  ------  --------  -------    -----   ----  ----  ----   -----  ---- -----  ---- -----");
      else
        Console.WriteLine();
    }


    static string ToStr(char pieceChar, int count, int maxCount)
    {
      string ret = "";

      if (count > maxCount)
      {
        // Rare case of more than expected number of pieces (due to promotion)
        ret += pieceChar + count.ToString();
        for (int i = 0; i < maxCount - 2; i++) ret += " ";
      }
      else
      {
        for (int i = 0; i < maxCount; i++)
        {
          if (count > i)
            ret += pieceChar;
          else
            ret += " ";
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
        int? childIndex = node.Ref.ChildIndexWithMove(move);
        if (childIndex is null) 
          throw new Exception($"Move  {move} not found at node {node}");
        else
          (node, _, _) = node.ChildAtIndexInfo(childIndex.Value);
      }
      return node;
    }


    public static void DumpPV(PositionWithHistory priorMoves, MCTSNode node, 
                               bool fullDetail, List<MGMove> subvariation = null)
    {
      if (subvariation != null)
      {
        List<MGMove> allMoves = new List<MGMove>();
        allMoves.AddRange(priorMoves.Moves);
        allMoves.AddRange(subvariation);
        DumpPV(new PositionWithHistory(priorMoves.InitialPosMG, allMoves),
                                  DescendMovesToNode(node, subvariation), fullDetail);
      }

      Console.WriteLine();
      WriteHeaders(fullDetail);


      List<Position> seenPositions = new List<Position>();

      int CountDuplicatePos(Position pos)
      {
        int count = 0;
        foreach (Position priorPos in seenPositions)
          if (pos.EqualAsRepetition(priorPos))
            count++;

        return count;
      }

      int depth = 0;
      while (true)
      {
        node.Context.Tree.Annotate(node);
        seenPositions.Add(node.Annotation.Pos);
        int countSeen = CountDuplicatePos(node.Annotation.Pos);

        DumpNodeStr(priorMoves, node, depth, countSeen, fullDetail);

        if (node.NumChildrenVisited == 0)
          return;
        node = node.BestMove(false);

        depth++;
      }

    }

  }

}
