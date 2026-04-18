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
using Ceres.MCGS.Graphs.GEdgeHeaders;
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


  // Column widths (keep WriteHeaders and DumpNodeStr in sync).
  private const int W_DEP = 3, W_NUM_MOVES = 3, W_INDEX = 10, W_FLG = 3, W_MOVE = 6;
  private const int W_POLICY = 7, W_VISITPCT = 7, W_PAR_N = 10, W_N = 10, W_REP = 3;
  private const int W_ACTV = 7, W_V = 7, W_Q = 7, W_WDL = 17;
  private const int W_UNC = 4, W_PIRR = 4, W_VERR = 4;


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
    writer ??= Console.Out;

    bool hasParent = !parentNode.IsNull;
    int indexInParent = hasParent ? parentNode.IndexOfChildInChildEdges(node.Index) : -1;

    // Resolve edge for non-root nodes (caller may or may not have passed one).
    if (!node.IsSearchRoot && hasParent && edge.IsNull)
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

    // Flag column: pruning status (F/T/S) overridden by terminal (C/D).
    char flag = ' ';
    if (!node.IsSearchRoot && hasParent && parentNode.IsSearchRoot
     && manager.RootMovesPruningStatus != null && indexInParent >= 0
     && manager.RootMovesPruningStatus[indexInParent] != MCGSFutilityPruningStatus.NotPruned)
    {
      flag = manager.RootMovesPruningStatus[indexInParent] switch
      {
        MCGSFutilityPruningStatus.PrunedDueToFutility => 'F',
        MCGSFutilityPruningStatus.PrunedDueToTablebaseNotWinning => 'T',
        MCGSFutilityPruningStatus.PrunedDueToSearchMoves => 'S',
        _ => ' '
      };
    }
    if (node.Terminal == GameResult.Checkmate) flag = 'C';
    else if (node.Terminal == GameResult.Draw) flag = 'D';

    // Rep column: nonblank only when the position repeated earlier in the walk.
    char repChar = ' ';
    if (countTimesSeen > 1)
    {
      repChar = countTimesSeen > 9 ? '9' : countTimesSeen.ToString()[0];
    }

    // Policy / parent-edge N (blank for root since there is no incoming edge).
    string policyStr;
    int parentEdgeN;
    if (node.IsSearchRoot || !hasParent || edge.IsNull)
    {
      policyStr = "";
      parentEdgeN = node.N;
    }
    else
    {
      policyStr = $"{100.0f * (float)edge.P,6:F2}%";
      parentEdgeN = edge.N;
    }

    float pctOfVisits = node.IsSearchRoot || !hasParent ? 100.0f : (100.0f * node.N / parentNode.N);

    // Action V: read from the edge header (action data lives there, not on the edge struct).
    // Blank if NN evaluator doesn't expose action head, no parent edge, or value is NaN.
    // Negate so value is shown from parent's (mover's) perspective.
    float actionVDisplay = float.NaN;
    if (manager.NNEvaluator0.HasAction && hasParent && indexInParent >= 0)
    {
      Span<GEdgeHeaderStruct> headers = parentNode.EdgeHeadersSpan;
      if (indexInParent < headers.Length)
      {
        float a = (float)headers[indexInParent].ActionV;
        if (!float.IsNaN(a))
        {
          actionVDisplay = -a;
        }
      }
    }

    // SAN of the move leading into this node (blank for root).
    string san = node.IsSearchRoot || !hasParent || edge.IsNull
      ? ""
      : MGMoveConverter.ToMove(edge.MoveMG).ToSAN(parentNode.CalcPosition().ToPosition);

    double q = node.Q;
    int? plyUntilIrreversible = node.PlyUntilPVIsIrreversibleMove();

    // Column writes - widths must match WriteHeaders.
    writer.Write($"{depth,W_DEP} ");
    writer.Write($"{node.NumPolicyMoves,W_NUM_MOVES} ");
    writer.Write($"{node.Index.Index,W_INDEX:N0} ");
    writer.Write($"{flag,W_FLG} ");
    writer.Write($"{san,W_MOVE} ");
    writer.Write($"{policyStr,W_POLICY} ");
    writer.Write($"{pctOfVisits,6:F2}% ");                          // W_VISITPCT=7
    writer.Write($"{parentEdgeN,W_PAR_N:N0} ");
    writer.Write($"{node.N,W_N:N0} ");
    writer.Write($"{repChar,W_REP} ");

    if (float.IsNaN(actionVDisplay))
    {
      writer.Write($"{"",W_ACTV} ");
    }
    else
    {
      DumpWithColor(actionVDisplay, $"{actionVDisplay,W_ACTV:F3} ", -0.2f, 0.2f, writer);
    }

    DumpWithColor(node.V, $"{node.V,W_V:F3} ", -0.2f, 0.2f, writer);
    DumpWithColor((float)q, $"{q,W_Q:F3} ", -0.2f, 0.2f, writer);

    writer.Write($"{node.WinP,5:F2}/{node.DrawP,5:F2}/{node.LossP,5:F2} ");      // W_WDL = 17
    writer.Write($"{node.W,5:F2}/{node.D,5:F2}/{node.L,5:F2} ");

    writer.Write($"{100 * node.UncertaintyValue,W_UNC:F0} ");
    writer.Write($"{100 * node.UncertaintyPolicy,W_UNC:F0} ");
    writer.Write(plyUntilIrreversible.HasValue
      ? $"{plyUntilIrreversible.Value,W_PIRR} "
      : $"{"-",W_PIRR} ");
    writer.Write($"{100 * MathF.Abs(node.V - (float)q),W_VERR:F0}");

    if (fullDetail)
    {
      Position position = node.IsSearchRoot
        ? manager.Engine.SearchRootNode.CalcPosition().ToPosition
        : edge.CalcChildPosition().ToPosition;
      writer.Write($" {position.FEN}");
    }

    writer.WriteLine();
  }


  internal static void WriteHeaders(bool fullDetail, TextWriter writer)
  {
    // Column widths must match DumpNodeStr.
    writer.Write($"{"Dep",W_DEP} ");
    writer.Write($"{"#M",W_NUM_MOVES} ");
    writer.Write($"{"Index",W_INDEX} ");
    writer.Write($"{"Flg",W_FLG} ");
    writer.Write($"{"Move",W_MOVE} ");
    writer.Write($"{"Policy",W_POLICY} ");
    writer.Write($"{"Visit%",W_VISITPCT} ");
    writer.Write($"{"ParN",W_PAR_N} ");
    writer.Write($"{"N",W_N} ");
    writer.Write($"{"Rep",W_REP} ");
    writer.Write($"{"ActV",W_ACTV} ");
    writer.Write($"{"V",W_V} ");
    writer.Write($"{"Q",W_Q} ");
    writer.Write($"{"W/D/L(Pos)",W_WDL} ");
    writer.Write($"{"W/D/L(Tree)",W_WDL} ");
    writer.Write($"{"VUnc",W_UNC} ");
    writer.Write($"{"PUnc",W_UNC} ");
    writer.Write($"{"PIrr",W_PIRR} ");
    writer.Write($"{"VErr",W_VERR}");
    if (fullDetail) writer.Write("  FEN");
    writer.WriteLine();

    // Separator line.
    writer.Write($"{new string('-', W_DEP),W_DEP} ");
    writer.Write($"{new string('-', W_NUM_MOVES),W_NUM_MOVES} ");
    writer.Write($"{new string('-', W_INDEX),W_INDEX} ");
    writer.Write($"{new string('-', W_FLG),W_FLG} ");
    writer.Write($"{new string('-', W_MOVE),W_MOVE} ");
    writer.Write($"{new string('-', W_POLICY),W_POLICY} ");
    writer.Write($"{new string('-', W_VISITPCT),W_VISITPCT} ");
    writer.Write($"{new string('-', W_PAR_N),W_PAR_N} ");
    writer.Write($"{new string('-', W_N),W_N} ");
    writer.Write($"{new string('-', W_REP),W_REP} ");
    writer.Write($"{new string('-', W_ACTV),W_ACTV} ");
    writer.Write($"{new string('-', W_V),W_V} ");
    writer.Write($"{new string('-', W_Q),W_Q} ");
    writer.Write($"{new string('-', W_WDL),W_WDL} ");
    writer.Write($"{new string('-', W_WDL),W_WDL} ");
    writer.Write($"{new string('-', W_UNC),W_UNC} ");
    writer.Write($"{new string('-', W_UNC),W_UNC} ");
    writer.Write($"{new string('-', W_PIRR),W_PIRR} ");
    writer.Write($"{new string('-', W_VERR),W_VERR}");
    if (fullDetail) writer.Write("  ---");
    writer.WriteLine();
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

        // Stop if position has been seen before (cycle detected, draw by repetition).
        // Note: CountDuplicatePos runs after seenPositions.Add, so the current position
        // is always counted once; a genuine repetition corresponds to countSeen > 1.
        if (countSeen > 1)
        {
          writer.WriteLine("  (draw by repetition - cycle detected)");
          return;
        }

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




