#region Using directives

using System.Collections.Generic;
using System;
using System.Linq;
using System.IO;
using System.Diagnostics;
using System.Text;

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

public static class PrincipalPosSetDumperHTML
{
  /// <summary>
  /// Dumps a PrincipalPositionSet to an interactive HTML visualization and opens it in a browser.
  /// </summary>
  /// <param name="set">The PrincipalPositionSet to dump</param>
  /// <param name="chosenMove">The move that was chosen (for highlighting)</param>
  /// <param name="customColumnFunc">Optional custom column (name, calculation function)</param>
  public static void DumpToGraphHTML(PrincipalPosSet set, MGMove chosenMove, (string colName, Func<PrincipalPos, string> calcFunc) customColumnFunc = default)
  {
    if (set == null || set.Members.Count == 0)
    {
      Console.WriteLine("No principal positions found.");
      return;
    }

    // Create temporary directory
    string tempDir = Path.Combine(Path.GetTempPath(), $"CeresGraph_{DateTime.Now:yyyyMMdd_HHmmss}");
    Directory.CreateDirectory(tempDir);
    string htmlPath = Path.Combine(tempDir, "graph.html");

    // Generate HTML content
    string htmlContent = GenerateTableHTML(set, chosenMove, customColumnFunc);

    // Write to file
    File.WriteAllText(htmlPath, htmlContent);

    Console.WriteLine($"Generated HTML file: {htmlPath}");
    Console.WriteLine($"HTML file size: {new FileInfo(htmlPath).Length:N0} bytes");

    // Launch in browser (Windows)
    try
    {
      Process.Start(new ProcessStartInfo
      {
        FileName = htmlPath,
        UseShellExecute = true
      });
      Console.WriteLine($"Graph visualization opened in browser: {htmlPath}");
    }
    catch (Exception ex)
    {
      Console.WriteLine($"Failed to open browser: {ex.Message}");
      Console.WriteLine($"Please open manually: {htmlPath}");
    }
  }


  /// <summary>
  /// Generates the complete HTML content for the table-based visualization.
  /// </summary>
  private static string GenerateTableHTML(PrincipalPosSet set, MGMove chosenMove, (string colName, Func<PrincipalPos, string> calcFunc) customColumnFunc)
  {
    StringBuilder html = new();

    html.AppendLine("<!DOCTYPE html>");
    html.AppendLine("<html lang='en'>");
    html.AppendLine("<head>");
    html.AppendLine("  <meta charset='UTF-8'>");
    html.AppendLine("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>");
    html.AppendLine("  <title>Ceres Principal Position Analysis</title>");
    html.AppendLine("  <style>");
    html.AppendLine(GetTableCSS());
    html.AppendLine("  </style>");
    html.AppendLine("</head>");
    html.AppendLine("<body>");

    // Header
    html.AppendLine("  <div class='header'>");
    html.AppendLine("    <h1>Principal Position Set Analysis</h1>");
    html.AppendLine("    <div class='summary-grid'>");
    html.AppendLine($"      <div class='summary-item'><span class='label'>Start Position:</span> {set.StartPosition.ToPosition.FEN}</div>");
    html.AppendLine($"      <div class='summary-item'><span class='label'>Start Q:</span> {set.StartSearchNode.Q:F3}</div>");

    // Build visits summary line
    string visitsSummary = $"Min N= {set.MinVisits:N0} ";
    if (set.MaxAbsQDiff < float.MaxValue)
    {
      visitsSummary += $" Min QDev= {set.MaxAbsQDiff:F3}";
    }
    visitsSummary += $" from {set.StartSearchNode.N:N0}";
    html.AppendLine($"      <div class='summary-item'><span class='label'>Visits:</span> {visitsSummary}</div>");
    html.AppendLine($"      <div class='summary-item'><span class='label'>Total Positions:</span> {set.Members.Count}</div>");
    html.AppendLine("    </div>");
    html.AppendLine("  </div>");

    // Get root position piece count and Q value
    int rootPieceCount = set.StartPosition.PieceCount;
    double rootQ = set.StartSearchNode.Q;

    // Detect shared initial move prefix (only if there are multiple positions)
    int sharedPrefixLength = 0;
    List<string> sharedMoves = new();
    MGPosition displayRootPosition = set.StartPosition;

    if (set.Members.Count > 1)
    {
      (sharedPrefixLength, sharedMoves, displayRootPosition) = PrincipalPosSetDumper.DetectSharedMovePrefix(set);

      // Display shared initial moves if found
      if (sharedPrefixLength >= 2)
      {
        string sharedMovesStr = string.Join(" ", sharedMoves);
        string displayRootFEN = displayRootPosition.ToPosition.FEN;
        html.AppendLine("  <div class='shared-moves'>");
        html.AppendLine($"    <div><span class='label'>Shared initial moves:</span> {sharedMovesStr}</div>");
        html.AppendLine($"    <div><span class='label'>Position after shared moves:</span> {displayRootFEN}</div>");
        html.AppendLine("  </div>");
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

        if (qDiff < minQDiff)
        {
          minQDiff = qDiff;
        }
        if (qDiff > maxQDiff)
        {
          maxQDiff = qDiff;
        }
        sumQDiff += qDiff;
      }
      double avgQDiff = sumQDiff / count;

      // Sort positions by visit count descending
      List<PrincipalPos> sortedPositions = kvp.Data.positions.OrderByDescending(p => p.LeafNode.N).ToList();

      // Check if this is the chosen move
      bool isChosenMove = !chosenMove.IsNull && kvp.Data.move.FromSquareIndex == chosenMove.FromSquareIndex && kvp.Data.move.ToSquareIndex == chosenMove.ToSquareIndex;
      string bestMoveClass = isChosenMove ? " best-move" : "";

      html.AppendLine($"  <div class='move-group{bestMoveClass}'>");
      html.AppendLine("    <div class='move-header'>");
      string bestMovePrefix = isChosenMove ? "<span class='best-move-badge'>BEST MOVE</span> " : "";
      html.AppendLine($"      <h2>{bestMovePrefix}QDiff avg={FormatQValueHTML(avgQDiff)} <span class='move-name'>{kvp.MoveStr}</span> ({count} positions) min={FormatQValueHTML(minQDiff)} max={FormatQValueHTML(maxQDiff)}</h2>");
      html.AppendLine("    </div>");

      html.AppendLine("    <table class='positions-table'>");
      html.AppendLine("      <thead>");
      html.AppendLine("        <tr>");
      html.AppendLine("          <th>Board</th>");
      html.AppendLine("          <th>Visits</th>");
      html.AppendLine("          <th>QDiff</th>");
      html.AppendLine("          <th>Depth</th>");
      html.AppendLine("          <th>#Tr</th>");
      html.AppendLine("          <th>#PcD</th>");
      html.AppendLine("          <th>#PwD</th>");

      // Add custom column if provided
      if (customColumnFunc != default)
      {
        html.AppendLine($"          <th>{customColumnFunc.colName}</th>");
      }

      html.AppendLine("          <th>Move Sequence</th>");
      html.AppendLine("        </tr>");
      html.AppendLine("      </thead>");
      html.AppendLine("      <tbody>");

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
        string pieceDiffStr = pieceDiff == 0 ? "" : $" ({pieceDiff:+#;-#;0} pieces)";

        // Calculate pawn differences
        MGPosition startPos = set.StartPosition;
        MGPosition leafPos = principalPos.LeafPosition;
        int pawnDiffs = PrincipalPosSetDumper.CountPawnDifferences(in startPos, in leafPos);

        // Convert Q to root player's perspective
        int depth = principalPos.PathFromRoot.Count;
        bool isRootPlayerPerspective = (depth % 2) == 1;
        double qFromRootPerspective = isRootPlayerPerspective ? principalPos.LeafNode.Q : -principalPos.LeafNode.Q;
        double qDiff = qFromRootPerspective - rootQ;

        // Add custom column if provided
        string customColumnValue = "";
        if (customColumnFunc != default)
        {
          string customValue = customColumnFunc.calcFunc(principalPos);
          customColumnValue = $", {customColumnFunc.colName}={customValue}";
        }

        // Render chess board
        string boardSVG = RenderChessBoard(principalPos.LeafPosition, 300);

        string visitsText = principalPos.LeafNode.N.ToString("N0");
        string qDiffText = FormatQValueHTML(qDiff);
        string depthText = depth.ToString();
        string numTrText = principalPos.NumOccurrences > 1 ? (principalPos.NumOccurrences - 1).ToString() : "";
        string pawnDiffsText = pawnDiffs == 0 ? "" : pawnDiffs.ToString();
        string movesText = string.Join(" ", moveSequence);

        html.AppendLine("        <tr>");
        html.AppendLine($"          <td><svg class='board' width='300' height='300' viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'>{boardSVG}</svg></td>");
        html.AppendLine($"          <td class='number'>{visitsText}</td>");
        html.AppendLine($"          <td class='number'>{qDiffText}</td>");
        html.AppendLine($"          <td class='number'>{depthText}</td>");
        html.AppendLine($"          <td class='number'>{numTrText}</td>");
        html.AppendLine($"          <td class='number'>{pieceDiffStr}</td>");
        html.AppendLine($"          <td class='number'>{pawnDiffsText}</td>");

        // Add custom column if provided
        if (customColumnFunc != default)
        {
          string customValueText = customColumnFunc.calcFunc(principalPos);
          html.AppendLine($"          <td class='number'>{customValueText}</td>");
        }

        html.AppendLine($"          <td class='moves'>{movesText}</td>");
        html.AppendLine("        </tr>");
      }

      html.AppendLine("      </tbody>");
      html.AppendLine("    </table>");
      html.AppendLine("  </div>");
    }

    // Display legend
    html.AppendLine("  <div class='legend'>");
    html.AppendLine("    <h3>Legend</h3>");
    html.AppendLine("    <div class='legend-grid'>");
    html.AppendLine("      <div><span class='label'>Visits:</span> Number of times this leaf node was visited during search</div>");
    html.AppendLine("      <div><span class='label'>QDiff:</span> Q value difference from root (positive = better for root player)</div>");
    html.AppendLine("      <div><span class='label'>Depth:</span> Number of plies from root to this position</div>");
    html.AppendLine("      <div><span class='label'>#Tr:</span> Number of transpositions (times same position reached via different paths)</div>");
    html.AppendLine("      <div><span class='label'>#PcD:</span> Piece count difference from root position</div>");
    html.AppendLine("      <div><span class='label'>#PwD:</span> Pawn difference count from root position</div>");
    html.AppendLine("      <div><span class='label'>Move Sequence:</span> Complete move path from root to this position</div>");
    html.AppendLine("    </div>");
    html.AppendLine("  </div>");

    // Add hidden SVG with piece definitions that all boards can reference
    html.AppendLine("  <svg width='0' height='0' style='position: absolute;'>");
    html.AppendLine(GetSVGPieceDefs());
    html.AppendLine("  </svg>");

    html.AppendLine("</body>");
    html.AppendLine("</html>");

    return html.ToString();
  }



  /// <summary>
  /// Formats a Q value with HTML color coding (green for positive, red for negative, orange for near-zero).
  /// </summary>
  private static string FormatQValueHTML(double q)
  {
    string formatted = $"{q:F3}";
    string cssClass = q switch
    {
      > 0.05 => "positive",
      < -0.05 => "negative",
      _ => "neutral"
    };
    return $"<span class='{cssClass}'>{formatted}</span>";
  }


  /// <summary>
  /// Gets the CSS styles for the table-based HTML page.
  /// </summary>
  private static string GetTableCSS()
  {
    return @"
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #1a1a2e;
        min-height: 100vh;
        padding: 20px;
      }
      
      .header {
        background: #2d2d44;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
      }
      
      .header h1 {
        color: #e0e0e0;
        margin-bottom: 15px;
        font-size: 28px;
      }
      
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 10px;
        margin-top: 15px;
      }
      
      .summary-item {
        color: #b0b0b0;
        font-size: 14px;
        padding: 8px;
        background: #3a3a52;
        border-radius: 5px;
      }
      
      .shared-moves {
        background: #2a3f5f;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #4a90e2;
      }
      
      .shared-moves > div {
        margin: 5px 0;
        color: #6db3f2;
        font-size: 14px;
      }
      
      .label {
        font-weight: bold;
        margin-right: 5px;
      }
      
      .move-group {
        background: #2d2d44;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        overflow: hidden;
      }
      
      .move-group.best-move {
        border: 3px solid #e74c3c;
      }
      
      .move-header {
        background: linear-gradient(135deg, #5b6ee1 0%, #6b5b95 100%);
        color: white;
        padding: 15px 20px;
      }
      
      .move-group.best-move .move-header {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
      }
      
      .move-header h2 {
        font-size: 20px;
        font-weight: 600;
      }
      
      .best-move-badge {
        background: #fff;
        color: #e74c3c;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        margin-right: 8px;
      }
      
      .move-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
      }
      
      .positions-table {
        width: 100%;
        border-collapse: collapse;
      }
      
      .positions-table thead {
        background: #3a3a52;
        border-bottom: 2px solid #4a4a62;
      }
      
      .positions-table th {
        padding: 12px;
        text-align: left;
        font-weight: 600;
        color: #e0e0e0;
        font-size: 16px;
      }
      
      .positions-table td {
        padding: 12px;
        border-bottom: 1px solid #3a3a52;
        vertical-align: middle;
        background: #25253a;
      }
      
      .positions-table tbody tr:hover {
        background: #2f2f48;
      }
      
      .positions-table .number {
        text-align: right;
        font-family: 'Courier New', monospace;
        font-size: 20px;
        color: #d0d0d0;
        font-weight: 500;
      }
      
      .positions-table .moves {
        font-family: 'Courier New', monospace;
        font-size: 20px;
        color: #d0d0d0;
        font-weight: 500;
      }
      
      .board {
        border: 2px solid #5a5a6a;
        border-radius: 4px;
        background: white;
      }
      
      .positive {
        color: #2ecc71;
        font-weight: bold;
      }
      
      .negative {
        color: #e74c3c;
        font-weight: bold;
      }
      
      .neutral {
        color: #f39c12;
        font-weight: bold;
      }
      
      .legend {
        background: #2d2d44;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-top: 20px;
      }
      
      .legend h3 {
        margin-bottom: 15px;
        color: #e0e0e0;
        font-size: 20px;
      }
      
      .legend-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 10px;
      }
      
      .legend-grid > div {
        padding: 8px;
        background: #3a3a52;
        border-radius: 5px;
        font-size: 14px;
        color: #b0b0b0;
      }
    ";
  }


  /// <summary>
  /// Renders a chess board as SVG using the PositionToSVG approach.
  /// </summary>
  private static string RenderChessBoard(MGPosition position, int size)
  {
    StringBuilder svg = new();

    // Use the PositionToSVG approach: 400x400 viewBox, pieces defined in defs and referenced
    Ceres.Chess.Position chessPos = position.ToPosition;

    // First, render all the squares (using the BODY template coordinates)
    for (int rank = 0; rank < 8; rank++)
    {
      for (int file = 0; file < 8; file++)
      {
        // Calculate positions (20px offset, 45px squares)
        int x = 20 + file * 45;
        int y = 20 + (7 - rank) * 45;  // Flip Y for display

        // Determine square color
        bool isLightSquare = (rank + file) % 2 == 1;
        string color = isLightSquare ? "#ffce9e" : "#d18b47";

        svg.Append($"<rect x=\"{x}\" y=\"{y}\" width=\"45\" height=\"45\" fill=\"{color}\" stroke=\"none\" />");
      }
    }

    // Add coordinate labels
    for (int file = 0; file < 8; file++)
    {
      char fileChar = (char)('a' + file);
      int x = 20 + file * 45 + 22;  // Center of square
      svg.Append($"<text alignment-baseline=\"middle\" font-size=\"14\" text-anchor=\"middle\" x=\"{x}\" y=\"10\">{fileChar}</text>");
      svg.Append($"<text alignment-baseline=\"middle\" font-size=\"14\" text-anchor=\"middle\" x=\"{x}\" y=\"390\">{fileChar}</text>");
    }

    for (int rank = 0; rank < 8; rank++)
    {
      int rankNum = rank + 1;
      int y = 20 + (7 - rank) * 45 + 22;  // Center of square
      svg.Append($"<text alignment-baseline=\"middle\" font-size=\"14\" text-anchor=\"middle\" x=\"10\" y=\"{y}\">{rankNum}</text>");
      svg.Append($"<text alignment-baseline=\"middle\" font-size=\"14\" text-anchor=\"middle\" x=\"390\" y=\"{y}\">{rankNum}</text>");
    }

    // Now render pieces using <use> references
    for (int rank = 0; rank < 8; rank++)
    {
      for (int file = 0; file < 8; file++)
      {
        int chessRank = rank;
        int chessFile = file;

        Ceres.Chess.Square square = Ceres.Chess.Square.FromFileAndRank(chessFile, chessRank);
        Ceres.Chess.Piece piece = chessPos.PieceOnSquare(square);

        if (piece.Type != Ceres.Chess.PieceType.None)
        {
          string pieceId = GetPieceSVGId(piece);

          // Calculate position for the piece (translate to square position)
          int x = 20 + file * 45;
          int y = 20 + (7 - rank) * 45;  // Flip Y for display

          svg.Append($"<use transform=\"translate({x}, {y})\" xlink:href=\"#{pieceId}\" />");
        }
      }
    }

    return svg.ToString();
  }


  /// <summary>
  /// Gets the SVG piece ID for referencing in the defs section.
  /// </summary>
  private static string GetPieceSVGId(Ceres.Chess.Piece piece)
  {
    string color = piece.Side == Ceres.Chess.SideType.White ? "white" : "black";
    string pieceType = piece.Type switch
    {
      Ceres.Chess.PieceType.Pawn => "pawn",
      Ceres.Chess.PieceType.Knight => "knight",
      Ceres.Chess.PieceType.Bishop => "bishop",
      Ceres.Chess.PieceType.Rook => "rook",
      Ceres.Chess.PieceType.Queen => "queen",
      Ceres.Chess.PieceType.King => "king",
      _ => ""
    };
    return $"{color}-{pieceType}";
  }


  /// <summary>
  /// Gets the SVG piece definitions (defs section) using the PositionToSVG approach.
  /// This is the complete defs section with all piece SVG paths.
  /// </summary>
  private static string GetSVGPieceDefs()
  {
    // Return the complete piece definitions from PositionToSVG
    // Using the exact same SVG paths to ensure consistency and quality
    return @"<defs><g class=""white pawn"" id=""white-pawn""><path d=""M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"" fill=""#fff"" stroke=""#000"" stroke-linecap=""round"" stroke-width=""1.5"" /></g><g class=""white knight"" fill=""none"" fill-rule=""evenodd"" id=""white-knight"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18"" style=""fill:#ffffff; stroke:#000000;"" /><path d=""M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10"" style=""fill:#ffffff; stroke:#000000;"" /><path d=""M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z"" style=""fill:#000000; stroke:#000000;"" /><path d=""M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z"" style=""fill:#000000; stroke:#000000;"" transform=""matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)"" /></g><g class=""white bishop"" fill=""none"" fill-rule=""evenodd"" id=""white-bishop"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><g fill=""#fff"" stroke-linecap=""butt""><path d=""M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zM15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"" /></g><path d=""M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5"" stroke-linejoin=""miter"" /></g><g class=""white rook"" fill=""#fff"" fill-rule=""evenodd"" id=""white-rook"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5"" stroke-linecap=""butt"" /><path d=""M34 14l-3 3H14l-3-3"" /><path d=""M31 17v12.5H14V17"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M31 29.5l1.5 2.5h-20l1.5-2.5"" /><path d=""M11 14h23"" fill=""none"" stroke-linejoin=""miter"" /></g><g class=""white queen"" fill=""#fff"" fill-rule=""evenodd"" id=""white-queen"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM16 8.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM33 9a2 2 0 1 1-4 0 2 2 0 1 1 4 0z"" /><path d=""M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14l2 12zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"" stroke-linecap=""butt"" /><path d=""M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0"" fill=""none"" /></g><g class=""white king"" fill=""none"" fill-rule=""evenodd"" id=""white-king"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M22.5 11.63V6"" stroke-linejoin=""miter"" /><path d=""M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"" fill=""#fff"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z"" fill=""#fff"" /><path d=""M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"" /></g><g class=""black pawn"" id=""black-pawn""><path d=""M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"" stroke=""#000"" stroke-linecap=""round"" stroke-width=""1.5"" /></g><g class=""black knight"" fill=""none"" fill-rule=""evenodd"" id=""black-knight"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18"" style=""fill:#000000; stroke:#000000;"" /><path d=""M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10"" style=""fill:#000000; stroke:#000000;"" /><path d=""M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z"" style=""fill:#ececec; stroke:#ececec;"" /><path d=""M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z"" style=""fill:#ececec; stroke:#ececec;"" transform=""matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)"" /><path d=""M 24.55,10.4 L 24.1,11.85 L 24.6,12 C 27.75,13 30.25,14.49 32.5,18.75 C 34.75,23.01 35.75,29.06 35.25,39 L 35.2,39.5 L 37.45,39.5 L 37.5,39 C 38,28.94 36.62,22.15 34.25,17.66 C 31.88,13.17 28.46,11.02 25.06,10.5 L 24.55,10.4 z "" style=""fill:#ececec; stroke:none;"" /></g><g class=""black bishop"" fill=""none"" fill-rule=""evenodd"" id=""black-bishop"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zm6-4c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"" fill=""#000"" stroke-linecap=""butt"" /><path d=""M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5"" stroke=""#fff"" stroke-linejoin=""miter"" /></g><g class=""black rook"" fill=""#000"" fill-rule=""evenodd"" id=""black-rook"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M9 39h27v-3H9v3zM12.5 32l1.5-2.5h17l1.5 2.5h-20zM12 36v-4h21v4H12z"" stroke-linecap=""butt"" /><path d=""M14 29.5v-13h17v13H14z"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M14 16.5L11 14h23l-3 2.5H14zM11 14V9h4v2h5V9h5v2h5V9h4v5H11z"" stroke-linecap=""butt"" /><path d=""M12 35.5h21M13 31.5h19M14 29.5h17M14 16.5h17M11 14h23"" fill=""none"" stroke=""#fff"" stroke-linejoin=""miter"" stroke-width=""1"" /></g><g class=""black queen"" fill=""#000"" fill-rule=""evenodd"" id=""black-queen"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><g fill=""#000"" stroke=""none""><circle cx=""6"" cy=""12"" r=""2.75"" /><circle cx=""14"" cy=""9"" r=""2.75"" /><circle cx=""22.5"" cy=""8"" r=""2.75"" /><circle cx=""31"" cy=""9"" r=""2.75"" /><circle cx=""39"" cy=""12"" r=""2.75"" /></g><path d=""M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5 9 26zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"" stroke-linecap=""butt"" /><path d=""M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0"" fill=""none"" /></g><g class=""black king"" fill=""none"" fill-rule=""evenodd"" id=""black-king"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M22.5 11.63V6"" stroke-linejoin=""miter"" /><path d=""M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"" fill=""#000"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z"" fill=""#000"" /><path d=""M20 8h5"" stroke-linejoin=""miter"" /><path d=""M32 29.5s8.5-4 6.03-9.65C34.15 14 25 18 22.5 24.5l.01 2.1-.01-2.1C20 18 9.906 14 6.997 19.85c-2.497 5.65 4.853 9 4.853 9M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"" stroke=""#fff"" /></g></defs>";
  }

}
