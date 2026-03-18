#region Using directives

using System.Collections.Generic;
using System;
using System.Linq;

using Spectre.Console;

using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Analysis;

/// <summary>
/// Static helper class for dumping PrincipalPositionSet information to the console.
/// </summary>
public static class PrincipalPosSetDumper
{
  /// <summary>
  /// Dumps a PrincipalPositionSet to the console, grouped by first move from root.
  /// </summary>
  /// <param name="set">The PrincipalPositionSet to dump</param>
  /// <param name="chosenMove">The move that was chosen (for highlighting)</param>
  /// <param name="customColumnFunc">Optional custom column (name, calculation function)</param>
  public static void DumpToConsole(PrincipalPosSet set, MGMove chosenMove, (string colName, Func<PrincipalPos, double> calcFunc)? customColumnFunc = null)
  {
    if (set == null || set.Members.Count == 0)
    {
      Console.WriteLine("No principal positions found.");
      return;
    }

    Console.WriteLine($"Principal Position Set Summary:");
    Console.WriteLine($"  Start Position: {set.StartPosition.ToPosition.FEN}");

    // Build visits summary line
    string visitsSummary = $"Min {set.MinVisits:N0} visits from {set.StartSearchNode.N:N0}";
    if (set.MaxAbsQDiff < float.MaxValue)
    {
      visitsSummary += $" with maximum Q deviation {set.MaxAbsQDiff:F3}";
    }
    Console.WriteLine($"  Visits: {visitsSummary}");

    Console.WriteLine($"  Total Positions: {set.Members.Count}");
    Console.WriteLine();

    // Get root position piece count and Q value
    int rootPieceCount = set.StartPosition.PieceCount;
    double rootQ = set.StartSearchNode.Q;

    // Group by first move
    Dictionary<string, (List<PrincipalPos> positions, MGMove move)> groupedByFirstMove = new();

    foreach (PrincipalPos principalPos in set.Members)
    {
      if (principalPos.PathFromRoot.Count < 2)
      {
        // No first move (only root), skip or handle separately
        continue;
      }

      // Get the first move (from root to first child)
      GNode rootNode = principalPos.PathFromRoot[0].Node;
      GNode firstChildNode = principalPos.PathFromRoot[1].Node;
      MGPosition firstPosition = principalPos.PathFromRoot[0].Position;

      // Find the edge from root to first child
      string firstMoveStr = "Unknown";
      MGMove firstMove = default;
      foreach (GEdge edge in rootNode.ChildEdgesExpanded)
      {
        if (edge.ChildNode.Index == firstChildNode.Index)
        {
          firstMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(edge.Move, in firstPosition);
          firstMoveStr = firstMove.MoveStr(MGMoveNotationStyle.LongAlgebraic);
          break;
        }
      }

      if (!groupedByFirstMove.ContainsKey(firstMoveStr))
      {
        groupedByFirstMove[firstMoveStr] = (new List<PrincipalPos>(), firstMove);
      }
      groupedByFirstMove[firstMoveStr].positions.Add(principalPos);
    }

    // Calculate average Q for each move and sort by avg Q descending
    var sortedMoves = groupedByFirstMove
      .Select(kvp => {
        double sumQDiff = 0;
        int count = kvp.Value.positions.Count;
        foreach (PrincipalPos principalPos in kvp.Value.positions)
        {
          int depth = principalPos.PathFromRoot.Count;
          bool isRootPlayerPerspective = (depth % 2) == 1;
          double qFromRootPerspective = isRootPlayerPerspective ? principalPos.LeafNode.Q : -principalPos.LeafNode.Q;
          double qDiff = qFromRootPerspective - rootQ;
          sumQDiff += qDiff;
        }
        double avgQDiff = sumQDiff / count;
        return new { MoveStr = kvp.Key, Data = kvp.Value, AvgQDiff = avgQDiff };
      })
      .OrderByDescending(x => x.AvgQDiff)
      .ToList();

    // Display grouped results
    foreach (var moveGroup in sortedMoves)
    {
      (string MoveStr, (List<PrincipalPos> positions, MGMove move) Data) kvp = (moveGroup.MoveStr, moveGroup.Data);

      // Calculate Q statistics for this move group
      double minQDiff = double.MaxValue;
      double maxQDiff = double.MinValue;
      double sumQDiff = 0;
      int count = kvp.Data.positions.Count;

      foreach (PrincipalPos principalPos in kvp.Data.positions)
      {
        int depth = principalPos.PathFromRoot.Count;
        bool isRootPlayerPerspective = (depth % 2) == 1;
        double qFromRootPerspective = isRootPlayerPerspective ? principalPos.LeafNode.Q : -principalPos.LeafNode.Q;
        double qDiff = qFromRootPerspective - rootQ;

        if (qDiff < minQDiff) minQDiff = qDiff;
        if (qDiff > maxQDiff) maxQDiff = qDiff;
        sumQDiff += qDiff;
      }
      double avgQDiff = sumQDiff / count;

      // Check if this is the chosen move
      bool isChosenMove = !chosenMove.IsNull && kvp.Data.move.FromSquareIndex == chosenMove.FromSquareIndex && kvp.Data.move.ToSquareIndex == chosenMove.ToSquareIndex;
      string bestMovePrefix = isChosenMove ? "(BEST MOVE) " : "";

      Console.WriteLine($"{bestMovePrefix} QDiff avg={avgQDiff:F3} {kvp.MoveStr} ({count} positions) min={minQDiff:F3} max={maxQDiff:F3}");

      foreach (PrincipalPos principalPos in kvp.Data.positions)
      {
        Console.Write("  ");

        // Build the move sequence
        List<string> moveSequence = new();
        for (int i = 0; i < principalPos.PathFromRoot.Count - 1; i++)
        {
          GNode currentNode = principalPos.PathFromRoot[i].Node;
          GNode nextNode = principalPos.PathFromRoot[i + 1].Node;
          MGPosition currentPosition = principalPos.PathFromRoot[i].Position;

          foreach (GEdge edge in currentNode.ChildEdgesExpanded)
          {
            if (edge.ChildNode.Index == nextNode.Index)
            {
              MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(edge.Move, in currentPosition);
              moveSequence.Add(mgMove.MoveStr(MGMoveNotationStyle.LongAlgebraic));
              break;
            }
          }
        }

        // Calculate piece count difference
        int leafPieceCount = principalPos.LeafPosition.PieceCount;
        int pieceDiff = leafPieceCount - rootPieceCount;
        string pieceDiffStr = pieceDiff == 0 ? "" : $" ({pieceDiff:+#;-#;0} pieces)";

        // Calculate pawn differences
        MGPosition startPos = set.StartPosition;
        MGPosition leafPos = principalPos.LeafPosition;
        int pawnDiffs = CountPawnDifferences(in startPos, in leafPos);

        // Convert Q to root player's perspective
        int depth = principalPos.PathFromRoot.Count;
        bool isRootPlayerPerspective = (depth % 2) == 1;
        double qFromRootPerspective = isRootPlayerPerspective ? principalPos.LeafNode.Q : -principalPos.LeafNode.Q;
        double qDiff = qFromRootPerspective - rootQ;

        // Add custom column if provided
        string customColumnValue = "";
        if (customColumnFunc.HasValue)
        {
          double customValue = customColumnFunc.Value.calcFunc(principalPos);
          customColumnValue = $", {customColumnFunc.Value.colName}={customValue:F3}";
        }

        Console.Write(string.Join(" ", moveSequence));
        string numTrStr = principalPos.NumOccurrences > 1 ? $", #Tr={principalPos.NumOccurrences - 1}" : "";
        Console.WriteLine($" (N={principalPos.LeafNode.N}, QDiff={qDiff:F3}{customColumnValue}, Depth={principalPos.PathFromRoot.Count}{numTrStr}{pieceDiffStr})");
      }

      Console.WriteLine();
    }
  }



  /// <summary>
  /// Dumps a PrincipalPositionSet to the console using Spectre.Console for enhanced visualization.
  /// </summary>
  /// <param name="set">The PrincipalPositionSet to dump</param>
  /// <param name="chosenMove">The move that was chosen (for highlighting)</param>
  /// <param name="customColumnFunc">Optional custom column (name, calculation function)</param>
  public static void DumpToConsoleGraphical(PrincipalPosSet set, MGMove chosenMove, (string colName, Func<PrincipalPos, string> calcFunc) customColumnFunc = default)
  {
    if (set == null || set.Members.Count == 0)
    {
      AnsiConsole.MarkupLine("[red]No principal positions found.[/]");
      return;
    }

    // Display header
    Rule rule = new("[yellow]Principal Position Set Analysis[/]");
    AnsiConsole.Write(rule);
    AnsiConsole.WriteLine();

    // Display summary panel
    Table summaryTable = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);

    summaryTable.AddColumn("Property");
    summaryTable.AddColumn("Value");
    summaryTable.AddRow("Start Position", set.StartPosition.ToPosition.FEN);
    summaryTable.AddRow("Start Q", set.StartSearchNode.Q.ToString("F3"));

    // Build visits summary line
    string visitsSummary = $"Min N= {set.MinVisits:N0} ";
    if (set.MaxAbsQDiff < float.MaxValue)
    {
      visitsSummary += $" Min QDev= {set.MaxAbsQDiff:F3}";
    }
    visitsSummary += $" from {set.StartSearchNode.N:N0}";
    summaryTable.AddRow("Visits", visitsSummary);

    summaryTable.AddRow("Total Positions", set.Members.Count.ToString());

    AnsiConsole.Write(summaryTable);
    AnsiConsole.WriteLine();

    // Get root position piece count and Q value
    int rootPieceCount = set.StartPosition.PieceCount;
    double rootQ = set.StartSearchNode.Q;

    // Detect shared initial move prefix (only if there are multiple positions)
    int sharedPrefixLength = 0;
    List<string> sharedMoves = new();
    MGPosition displayRootPosition = set.StartPosition;

    if (set.Members.Count > 1)
    {
      (sharedPrefixLength, sharedMoves, displayRootPosition) = DetectSharedMovePrefix(set);

      // Display shared initial moves if found
      if (sharedPrefixLength >= 2)
      {
        string sharedMovesStr = string.Join(" ", sharedMoves);
        string displayRootFEN = displayRootPosition.ToPosition.FEN;
        AnsiConsole.MarkupLine($"[blue]Shared initial moves:[/] {sharedMovesStr}");
        AnsiConsole.MarkupLine($"[blue]Position after shared moves:[/] {displayRootFEN}");
        AnsiConsole.WriteLine();
      }
    }

    // Group by first move after the shared prefix
    Dictionary<string, (List<PrincipalPos> positions, MGMove move)> groupedByFirstMove = new();

    foreach (PrincipalPos principalPos in set.Members)
    {
      if (principalPos.PathFromRoot.Count < sharedPrefixLength + 2)
      {
        // Path is too short to have a move after the shared prefix
        continue;
      }

      // Get the first move after the shared prefix
      GNode displayRootNode = principalPos.PathFromRoot[sharedPrefixLength].Node;
      GNode firstChildNode = principalPos.PathFromRoot[sharedPrefixLength + 1].Node;
      MGPosition displayRootPos = principalPos.PathFromRoot[sharedPrefixLength].Position;

      string firstMoveStr = "Unknown";
      MGMove firstMove = default;
      foreach (GEdge edge in displayRootNode.ChildEdgesExpanded)
      {
        if (edge.ChildNode.Index == firstChildNode.Index)
        {
          firstMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(edge.Move, in displayRootPos);
          firstMoveStr = firstMove.MoveStr(MGMoveNotationStyle.LongAlgebraic);
          break;
        }
      }

      if (!groupedByFirstMove.ContainsKey(firstMoveStr))
      {
        groupedByFirstMove[firstMoveStr] = (new List<PrincipalPos>(), firstMove);
      }
      groupedByFirstMove[firstMoveStr].positions.Add(principalPos);
    }

    // Calculate average Q for each move and sort by avg Q descending
    var sortedMoves = groupedByFirstMove
      .Select(kvp => {
        double sumQDiff = 0;
        int count = kvp.Value.positions.Count;
        foreach (PrincipalPos principalPos in kvp.Value.positions)
        {
          int depth = principalPos.PathFromRoot.Count;
          bool isRootPlayerPerspective = (depth % 2) == 1;
          double qFromRootPerspective = isRootPlayerPerspective ? principalPos.LeafNode.Q : -principalPos.LeafNode.Q;
          double qDiff = qFromRootPerspective - rootQ;
          sumQDiff += qDiff;
        }
        double avgQDiff = sumQDiff / count;
        return new { MoveStr = kvp.Key, Data = kvp.Value, AvgQDiff = avgQDiff };
      })
      .OrderByDescending(x => x.AvgQDiff)
      .ToList();

    // Display each move group in a separate table
    foreach (var moveGroup in sortedMoves)
    {
      (string MoveStr, (List<PrincipalPos> positions, MGMove move) Data) kvp = (moveGroup.MoveStr, moveGroup.Data);

      // Calculate Q statistics for this move group
      double minQDiff = double.MaxValue;
      double maxQDiff = double.MinValue;
      double sumQDiff = 0;
      int count = kvp.Data.positions.Count;

      foreach (PrincipalPos principalPos in kvp.Data.positions)
      {
        int depth = principalPos.PathFromRoot.Count;
        bool isRootPlayerPerspective = (depth % 2) == 1;
        double qFromRootPerspective = isRootPlayerPerspective ? principalPos.LeafNode.Q : -principalPos.LeafNode.Q;
        double qDiff = qFromRootPerspective - rootQ;

        if (qDiff < minQDiff) minQDiff = qDiff;
        if (qDiff > maxQDiff) maxQDiff = qDiff;
        sumQDiff += qDiff;
      }
      double avgQDiff = sumQDiff / count;

      // Sort positions by visit count descending
      List<PrincipalPos> sortedPositions = kvp.Data.positions.OrderByDescending(p => p.LeafNode.N).ToList();

      // Create table for this move
      Table moveTable = new Table().Border(TableBorder.Rounded).BorderColor(Color.Blue);

      // Check if this is the chosen move
      bool isChosenMove = !chosenMove.IsNull && kvp.Data.move.FromSquareIndex == chosenMove.FromSquareIndex && kvp.Data.move.ToSquareIndex == chosenMove.ToSquareIndex;
      string bestMovePrefix = isChosenMove ? "[bold red](BEST MOVE)[/] " : "";

      // Add title with Q statistics - QDiff avg first, then move, then min/max
      moveTable.Title = new TableTitle($"{bestMovePrefix}[grey]QDiff avg=[/]{PrincipalPosSetDumper.FormatQValue(avgQDiff)} " +
                                       $"[bold cyan]{kvp.MoveStr}[/] ({count} positions) " +
                                       $"[grey]min=[/]{FormatQValue(minQDiff)} " +
                                       $"[grey]max=[/]{FormatQValue(maxQDiff)}");

      moveTable.AddColumn(new TableColumn("Visits").RightAligned());
      moveTable.AddColumn(new TableColumn("QDiff").RightAligned());
      moveTable.AddColumn(new TableColumn("Depth").RightAligned());
      moveTable.AddColumn(new TableColumn("#Tr").RightAligned());
      moveTable.AddColumn(new TableColumn("#PcD").RightAligned());
      moveTable.AddColumn(new TableColumn("#PwD").RightAligned());

      // Add custom column if provided
      if (customColumnFunc != default)
      {
        moveTable.AddColumn(new TableColumn(customColumnFunc.colName).RightAligned());
      }

      moveTable.AddColumn(new TableColumn("Move Sequence").LeftAligned());
      moveTable.AddColumn(new TableColumn("FEN").LeftAligned());

      foreach (PrincipalPos principalPos in sortedPositions)
      {
        // Build the move sequence starting from display root (after shared prefix)
        List<string> moveSequence = new();
        for (int i = sharedPrefixLength; i < principalPos.PathFromRoot.Count - 1; i++)
        {
          GNode currentNode = principalPos.PathFromRoot[i].Node;
          GNode nextNode = principalPos.PathFromRoot[i + 1].Node;
          MGPosition currentPosition = principalPos.PathFromRoot[i].Position;

          foreach (GEdge edge in currentNode.ChildEdgesExpanded)
          {
            if (edge.ChildNode.Index == nextNode.Index)
            {
              MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(edge.Move, in currentPosition);
              moveSequence.Add(mgMove.MoveStr(MGMoveNotationStyle.LongAlgebraic));
              break;
            }
          }
        }

        // Calculate piece count difference
        int leafPieceCount = principalPos.LeafPosition.PieceCount;
        int pieceDiff = leafPieceCount - rootPieceCount;

        // Calculate pawn differences
        MGPosition startPos = set.StartPosition;
        MGPosition leafPos = principalPos.LeafPosition;
        int pawnDiffs = CountPawnDifferences(in startPos, in leafPos);

        // Convert Q to root player's perspective
        int depth = principalPos.PathFromRoot.Count;
        bool isRootPlayerPerspective = (depth % 2) == 1;
        double qFromRootPerspective = isRootPlayerPerspective ? principalPos.LeafNode.Q : -principalPos.LeafNode.Q;

        // Calculate Q difference from root
        double qDiff = qFromRootPerspective - rootQ;

        // Get FEN of leaf position
        string fen = principalPos.LeafPosition.ToPosition.FEN;

        string visitsText = principalPos.LeafNode.N.ToString("N0");
        string qDiffText = FormatQValue(qDiff);
        string depthText = depth.ToString();  // Keep original depth from true root
        string numTrText = principalPos.NumOccurrences > 1 ? (principalPos.NumOccurrences - 1).ToString() : "";
        string pieceDiffText = pieceDiff == 0 ? "" :
                               pieceDiff > 0 ? $"[green]+{pieceDiff}[/]" :
                               $"[red]{pieceDiff}[/]";
        string pawnDiffsText = pawnDiffs == 0 ? "" : pawnDiffs.ToString();
        string movesText = string.Join(" ", moveSequence);
        string fenText = fen;

        // Add row with or without custom column
        if (customColumnFunc != default)
        {
          string customValueText = customColumnFunc.calcFunc(principalPos);
          moveTable.AddRow(visitsText, qDiffText, depthText, numTrText, pieceDiffText, pawnDiffsText, customValueText, movesText, fenText);
        }
        else
        {
          moveTable.AddRow(visitsText, qDiffText, depthText, numTrText, pieceDiffText, pawnDiffsText, movesText, fenText);
        }
      }

      AnsiConsole.Write(moveTable);
      AnsiConsole.WriteLine();
    }

    // Display legend
    AnsiConsole.WriteLine();
    AnsiConsole.WriteLine();

    AnsiConsole.MarkupLine("[yellow]Visits[/]        - Number of times this leaf node was visited during search");
    AnsiConsole.MarkupLine("[yellow]QDiff[/]         - Q value difference from root (positive = better for root player)");
    AnsiConsole.MarkupLine("[yellow]Depth[/]         - Number of plies from root to this position");
    AnsiConsole.MarkupLine("[yellow]#Tr[/]           - Number of transpositions (times same position reached via different paths)");
    AnsiConsole.MarkupLine("[yellow]#PcD[/]          - Piece count difference from root position");
    AnsiConsole.MarkupLine("[yellow]#PwD[/]          - Pawn difference count from root position");
    AnsiConsole.MarkupLine("[yellow]Move Sequence[/] - Complete move path from root to this position");
    AnsiConsole.MarkupLine("[yellow]FEN[/]           - Forsyth-Edwards Notation of the leaf position");
    AnsiConsole.WriteLine();
  }


  /// <summary>
  /// Detects the common prefix of moves shared by all principal positions.
  /// Returns the length of the shared prefix, the list of shared moves, and the position after the shared moves.
  /// </summary>
  internal static (int prefixLength, List<string> sharedMoves, MGPosition displayRootPosition) DetectSharedMovePrefix(PrincipalPosSet set)
  {
    if (set.Members.Count == 0)
    {
      return (0, new List<string>(), set.StartPosition);
    }

    // Find the minimum path length
    int minPathLength = int.MaxValue;
    foreach (PrincipalPos principalPos in set.Members)
    {
      if (principalPos.PathFromRoot.Count < minPathLength)
      {
        minPathLength = principalPos.PathFromRoot.Count;
      }
    }

    // Find the longest common prefix
    int sharedPrefixLength = 0;
    List<string> sharedMoves = new();
    MGPosition displayRootPosition = set.StartPosition;

    // Start from index 0 (root) and check each position in the path
    for (int pathIndex = 0; pathIndex < minPathLength - 1; pathIndex++)
    {
      // Get the node index at this path position from the first member
      PrincipalPos firstPos = set.Members[0];
      NodeIndex expectedNodeIndex = firstPos.PathFromRoot[pathIndex + 1].Node.Index;

      // Check if all members have the same node at this path position
      bool allMatch = true;
      foreach (PrincipalPos principalPos in set.Members)
      {
        if (principalPos.PathFromRoot[pathIndex + 1].Node.Index != expectedNodeIndex)
        {
          allMatch = false;
          break;
        }
      }

      if (!allMatch)
      {
        break;
      }

      // All match, so this is part of the shared prefix
      GNode currentNode = firstPos.PathFromRoot[pathIndex].Node;
      GNode nextNode = firstPos.PathFromRoot[pathIndex + 1].Node;
      MGPosition currentPosition = firstPos.PathFromRoot[pathIndex].Position;

      // Find the move from current to next
      foreach (GEdge edge in currentNode.ChildEdgesExpanded)
      {
        if (edge.ChildNode.Index == nextNode.Index)
        {
          MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(edge.Move, in currentPosition);
          sharedMoves.Add(mgMove.MoveStr(MGMoveNotationStyle.LongAlgebraic));
          break;
        }
      }

      sharedPrefixLength = pathIndex + 1;
      displayRootPosition = firstPos.PathFromRoot[pathIndex + 1].Position;
    }

    // Only return the shared prefix if it has at least 2 moves
    if (sharedMoves.Count >= 2)
    {
      return (sharedPrefixLength, sharedMoves, displayRootPosition);
    }

    return (0, new List<string>(), set.StartPosition);
  }


  /// <summary>
  /// Counts the number of squares where pawn occupancy (including color) changed between two positions.
  /// Counts pawn moves as a single change (not double-counting source and destination).
  /// </summary>
  internal static int CountPawnDifferences(in MGPosition pos1, in MGPosition pos2)
  {
    int pawnsDisappeared = 0;
    int pawnsAppeared = 0;

    // Check all 64 squares
    for (int squareIndex = 0; squareIndex < 64; squareIndex++)
    {
      ulong mask = 1UL << squareIndex;

      // Get piece codes for both positions at this square
      int piece1 = GetPieceCodeAtSquare(in pos1, mask);
      int piece2 = GetPieceCodeAtSquare(in pos2, mask);

      // Check if either piece is a pawn (white pawn = 1, black pawn = 9)
      bool isPawn1 = (piece1 == 1 || piece1 == 9);
      bool isPawn2 = (piece2 == 1 || piece2 == 9);

      // Count appearances and disappearances
      if (isPawn1 && !isPawn2)
      {
        // Pawn disappeared from this square
        pawnsDisappeared++;
      }
      else if (!isPawn1 && isPawn2)
      {
        // Pawn appeared on this square
        pawnsAppeared++;
      }
      else if (isPawn1 && isPawn2 && piece1 != piece2)
      {
        // Pawn changed color (count as both disappearance and appearance)
        pawnsDisappeared++;
        pawnsAppeared++;
      }
    }

    // Return the maximum of appearances or disappearances
    // This accounts for moves (1 pawn), captures (difference), and promotions
    return Math.Max(pawnsAppeared, pawnsDisappeared);
  }


  /// <summary>
  /// Gets the piece code at a specific square using bitboard mask.
  /// </summary>
  private static int GetPieceCodeAtSquare(in MGPosition pos, ulong mask)
  {
    int code = 0;
    if ((pos.D & mask) != 0) code |= 8;
    if ((pos.C & mask) != 0) code |= 4;
    if ((pos.B & mask) != 0) code |= 2;
    if ((pos.A & mask) != 0) code |= 1;
    return code;
  }


  /// <summary>
  /// Formats a Q value with color coding (green for positive, red for negative, yellow for near-zero).
  /// </summary>
  internal static string FormatQValue(double q)
  {
    string formatted = $"{q:F3}";

    return q switch
    {
      > 0.05 => $"[green]{formatted}[/]",
      < -0.05 => $"[red]{formatted}[/]",
      _ => $"[yellow]{formatted}[/]"
    };
  }

}
