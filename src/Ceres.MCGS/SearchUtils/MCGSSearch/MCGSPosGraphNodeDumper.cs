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
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Static helper methods for dumping contents of graph to Console.
/// </summary>
public static class MCGSPosGraphNodeDumper
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


  internal static void DumpNodeStr(MCGSManager manager,
                                   int depth,
                                   GNode searchRootNode,
                                   GNode parentNode,
                                   GNode node,
                                   GEdge edge,
                                   int countTimesSeen,
                                   bool fullDetail,
                                   TextWriter writer = null)
  {
    int indexInParent = parentNode.IsNull ? -1 : parentNode.IndexOfChildInChildEdges(node.Index);

    writer ??= Console.Out;

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

    float pctOfVisits = node.IsSearchRoot ? 100.0f : (100.0f * node.N / parentNode.N);

    if (!node.IsSearchRoot)
    {
      foreach (GEdge edgeTest in parentNode.ChildEdgesExpanded)
      {
        if (edgeTest.ChildNode == node)
        {
          edge = edgeTest;
          break;
        }
      }
    }


    // Print extra characters for nodes with special characteristics
    char extraFlag = ' ';
    if (!node.IsSearchRoot && parentNode.IsSearchRoot
      && manager.RootMovesPruningStatus != null
      && manager.RootMovesPruningStatus[indexInParent] != MCGSFutilityPruningStatus.NotPruned)
    {
      MCGSFutilityPruningStatus status = manager.RootMovesPruningStatus[indexInParent];
      extraFlag = status switch
      {
        MCGSFutilityPruningStatus.PrunedDueToFutility => 'F',
        MCGSFutilityPruningStatus.PrunedDueToTablebaseNotWinning => 'T',
        MCGSFutilityPruningStatus.PrunedDueToSearchMoves => 'S',
        MCGSFutilityPruningStatus.NotPruned => ' ',
        _ => throw new NotImplementedException()
      };
    }

    if (node.Terminal == GameResult.Draw)
    {
      extraFlag = 'D';
    }
    else if (node.Terminal == GameResult.Checkmate)
    {
      extraFlag = 'C';
    }

    // TODO:

    // Depth, move
    writer.Write($"{depth,3}. ");
    writer.Write($" {node.NumPolicyMoves,3} ");

    writer.Write($"#{node.Index.Index,13:N0}");
    writer.Write($"{extraFlag,3}");
    Position pos = node.IsSearchRoot ? manager.Engine.SearchRootNode.CalcPosition().ToPosition : edge.CalcChildPosition().ToPosition;

    string san = node.IsSearchRoot ? "" : MGMoveConverter.ToMove(edge.MoveMG).ToSAN(parentNode.CalcPosition().ToPosition);
    //      string sanNextBest = node.IsRoot ? "" : MGMoveConverter.ToMove(nextBestMove.Annotation.PriorMoveMG).ToSAN(in node.Parent.Annotation.Pos);
    if (pos.MiscInfo.SideToMove == SideType.White)
    {
      //        writer.Write($"      ");
      writer.Write($"{san,6}");
    }
    else
    {
      writer.Write($"{san,6}");
      //        writer.Write($"      ");
    }


    // TODO: ActionV if present
    float actionV = 0;

    float movePriorProb = 1.0f;
    int parentEdgeN;
    if (node.IsSearchRoot)
    {
       parentEdgeN = node.N;
    }
    else
    {
      // If not root, use parent node's policy
      GEdge edgeParent = parentNode.ChildFromGPosition(node);
      movePriorProb = edgeParent.P;
      parentEdgeN = edgeParent.N;
    }

    writer.Write($"  {100.0 * movePriorProb,6:F2}% ");
    writer.Write($" {pctOfVisits,6:F2}% ");

    writer.Write($"  VIS= {parentEdgeN,13:N0} ");
    writer.Write($"  N= {node.N,13:N0} ");
    writer.Write(extraChar);
    DumpWithColor(multiplier * -actionV, $" {multiplier * -actionV,6:F3}  ", -0.2f, 0.2f, writer);
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

    writer.Write($" {(invert ? node.L : node.W),5:F2}/{node.D,5:F2}/{(invert ? node.W : node.L),5:F2}  ");

    //      writer.Write($"   {node.Ref.QUpdatesWtdAvg,5:F2}  ");
    //      writer.Write($" +/-:{MathF.Sqrt(node.Ref.QUpdatesWtdVariance),5:F2}  ");
    //      writer.Write($" {node.Ref.TrendBonusToP,5:F2}  ");

    if (false)
    {
      writer.Write($" {node.NodeRef.StdDevEstimate.RunningStdDev,3:F0} ");
    }
    writer.Write($" {100 * node.UncertaintyValue,3:F0} ");
    writer.Write($" {100 * node.UncertaintyPolicy,3:F0}  ");

    int? plyUntilIrreversible = node.PlyUntilPVIsIrreversibleMove();
    writer.Write(plyUntilIrreversible.HasValue ? $" {plyUntilIrreversible.Value,3} " : "   - ");

    writer.Write($" {100 * MathF.Abs(node.V - (float)node.Q),3:F0}  ");

    //writer.Write($" {node.MPosition,3:F0} ");
    //writer.Write($" {node.MAvg,3:F0}  ");

    if (fullDetail)
    {
      Position position = node.IsSearchRoot ? manager.Engine.SearchRootNode.CalcPosition().ToPosition : edge.CalcChildPosition().ToPosition;
      int numPieces = position.PieceCount;

      //        writer.Write($" {PosStr(node.Annotation.Pos)} ");
      writer.Write($" {position.FEN}");
    }


    writer.WriteLine();
  }


  internal static void WriteHeaders(bool fullDetail, TextWriter writer)
  {
    // Write headers
//    writer.Write(" Dep  T #M    FirstVisit   MvWh  MvBl              N  Visits    Policy    Act       V        Q      WPos  DPos  LPos   WTree DTree LTree  VUnc PUnc PIrr VErr");
    writer.Write(" Dep  T #M    FirstVisit   MvWh  MvBl            VIS             N  Visits    Policy    Act       V        Q      WPos  DPos  LPos   WTree DTree LTree  VUnc PUnc PIrr VErr");

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
      writer.WriteLine("----  - --  ------------  -----  ----    -----------  ------  --------   ------  -------   ------   ----  ----  ----   -----  ---- -----  ---- ----- -----");
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



  public static void DumpPVFromRoot(MCGSManager manager, GNode node, bool fullDetail, TextWriter writer = null)
  {
    HashSet<int> visitNodes = NodesToRootSet(node);

    DumpPV(manager, manager.Engine.SearchRootNode, fullDetail, writer, visitNodes);
  }

  public static HashSet<int> NodesToRootSet(GNode node)
  {
    // Build set of nodes ascending to search root via tree-parent edges.
    // The tree-parent invariant guarantees position 0 in each node's parent list
    // is always on a cycle-free path to the root.
    HashSet<int> visitNodes = [];
    while (!node.IsNull && !node.IsSearchRoot && !node.IsGraphRoot)
    {
      visitNodes.Add(node.Index.Index);
      node = node.Graph[node.TreeParentNodeIndex];
    }

    // Add the root itself (search root or graph root, whichever we stopped at).
    if (!node.IsNull)
    {
      visitNodes.Add(node.Index.Index);
    }

    return visitNodes;
  }


  public static void DumpPV(MCGSManager manager, GNode node, bool fullDetail,
                            TextWriter writer = null,
                            HashSet<int> mustVisitNodes = null,
                            int minN = 0)
  {
    try
    {
      writer ??= Console.Out;

      writer.WriteLine();
      WriteHeaders(fullDetail, writer);

      List<Position> seenPositions = [];

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
      GNode priorNode = default;
      int numOutput = 0;
      while (numOutput++ < 256)
      {
        GEdge edge = default;
        Position pos = node.CalcPosition().ToPosition;
        seenPositions.Add(pos);
        int countSeen = CountDuplicatePos(pos);

        DumpNodeStr(manager, depth, node, priorNode, node, edge, countSeen, fullDetail, writer);

        // Stop dumping of N becomes too small
        if (node.N < minN)
        {
          return;
        }


        GEdge mustVisitEdge = default;
        if (mustVisitNodes != null)
        {
          foreach (GEdge childEdge in node.ChildEdgesExpanded)
          {
            if (childEdge.Type == GEdgeStruct.EdgeType.ChildEdge
             && mustVisitNodes.Contains(childEdge.ChildNode.Index.Index))
            {
              mustVisitEdge = childEdge;
            }
          }
        }

        // Advance to next child. Use "must visit" child if found, otherwise best move.
        priorNode = node;

        ManagerChooseBestMoveMCGS bm = new(manager, node, false, default, false);

        edge = !mustVisitEdge.IsNull ? mustVisitEdge : bm.BestMoveCalc.BestMoveEdge;
        node = edge.ChildNode;
        if (node.IsNull)
        {
          if (edge.Type.IsTerminal())
          {
            writer.WriteLine("  TERMINAL " + edge);
          }
          return;
        }
        depth++;
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




