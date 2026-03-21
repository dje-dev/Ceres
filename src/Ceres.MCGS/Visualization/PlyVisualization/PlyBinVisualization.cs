using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Ceres.Chess;
using Ceres.Chess.NetEvaluation.Batch;

namespace Ceres.MCGS.Visualization.PlyVisualization;

record PlyBinEntry(string Fen, string Title, Half[,] MovementProbs, Half[,] CaptureProbs,
                   string EvalSummary = null, bool HighlightTop4 = true,
                   Half[] PunimSelfProbs = null, Half[] PunimOpponentProbs = null,
                   List<PVCandidate> ProjectedPVs = null, float MLH = 0);

static class PlyBinVisualization
{
  // Bin midpoints for expected ply calculation.
  static readonly double[] BIN_MIDPOINTS = [1.5, 3.5, 7.5, 16.5, 31.5, 53.0, 80.0, 300.0];

  // Bin widths (number of plies each bin spans), used for density normalization.
  // "never" treated as width 100 so it gets low density weight.
  static readonly double[] BIN_WIDTHS = [2, 2, 6, 12, 18, 25, 34, 100];

  // Bin labels for the legend.
  static readonly string[] BIN_LABELS = ["1-2", "3-4", "5-10", "11-22", "23-40", "41-65", "66+", "never"];

  // Bin colors (warm -> cool -> gray).
  static readonly string[] BIN_COLORS =
  [
    "#d73027", "#e66101", "#f48c42", "#fdd49e",
    "#b2df8a", "#abd9e9", "#4575b4", "#808080"
  ];

  // Heatmap gradient stops (5-stop: red -> orange -> yellow -> light blue -> dark blue).
  static readonly (double t, int r, int g, int b)[] GRADIENT_STOPS =
  [
    (0.0, 0xd7, 0x30, 0x27),
    (0.25, 0xfd, 0xae, 0x61),
    (0.5, 0xff, 0xff, 0xbf),
    (0.75, 0xab, 0xd9, 0xe9),
    (1.0, 0x45, 0x75, 0xb4)
  ];

  const bool USE_ABSOLUTE_COLOR_BARS = true;

  const int SQUARE_SIZE = 45;
  const int BOARD_OFFSET = 20;
  const int BOARD_SVG_SIZE = 400;


  /// <summary>
  /// Converts flat Half[] arrays from NNEvaluatorResult into [64,8] arrays,
  /// remapping network squares to absolute board squares (flipping ranks for black).
  /// </summary>
  public static (Half[,] movement, Half[,] capture) ConvertProbs(NNEvaluatorResult result, bool isBlack)
  {
    Half[,] movementProbs = new Half[64, 8];
    Half[,] captureProbs = new Half[64, 8];

    for (int sq = 0; sq < 64; sq++)
    {
      int file = sq % 8;
      int rank = sq / 8;
      int networkSq = isBlack ? file + (7 - rank) * 8 : sq;
      for (int b = 0; b < 8; b++)
      {
        movementProbs[sq, b] = result.PlyBinMoveProbs[networkSq * 8 + b];
        captureProbs[sq, b] = result.PlyBinCaptureProbs[networkSq * 8 + b];
      }
    }

    return (movementProbs, captureProbs);
  }


  /// <summary>
  /// Returns path of generated HTML file for a single position.
  /// Square ordering: index = file + 8*rank (A1=0, B1=1, ..., H8=63).
  /// Bin ordering: [0]=1-2 ply, [1]=3-4, [2]=5-10, [3]=11-22, [4]=23-40, [5]=41-65, [6]=66+, [7]=never.
  /// </summary>
  public static string Generate(string fen, Half[,] movementProbs, Half[,] captureProbs,
                                string titleString, string outputPath = null)
  {
    List<PlyBinEntry> entries = [new PlyBinEntry(fen, titleString, movementProbs, captureProbs)];
    return GenerateMulti(entries, titleString, outputPath);
  }


  /// <summary>
  /// Generates a single HTML file with multiple positions shown sequentially.
  /// </summary>
  public static string GenerateMulti(List<PlyBinEntry> entries, string pageTitle,
                                     string outputPath = null)
  {
    foreach (PlyBinEntry entry in entries)
    {
      if (entry.MovementProbs.GetLength(0) != 64 || entry.MovementProbs.GetLength(1) != 8)
      {
        throw new ArgumentException("movementProbs must be [64, 8]");
      }
      if (entry.CaptureProbs.GetLength(0) != 64 || entry.CaptureProbs.GetLength(1) != 8)
      {
        throw new ArgumentException("captureProbs must be [64, 8]");
      }
    }

    string filePath = outputPath ?? Path.Combine(Directory.GetCurrentDirectory(),
                        $"plybin_{DateTime.Now:yyyyMMdd_HHmmss}.html");

    StringBuilder html = new StringBuilder();
    AppendHtmlHeader(html, pageTitle);

    for (int i = 0; i < entries.Count; i++)
    {
      PlyBinEntry entry = entries[i];
      AppendPositionSections(html, entry.Fen, entry.MovementProbs, entry.CaptureProbs,
                             entry.Title, $"pos{i}", entry.EvalSummary, entry.HighlightTop4,
                             entry.PunimSelfProbs, entry.PunimOpponentProbs,
                             entry.ProjectedPVs, entry.MLH);
    }

    html.AppendLine("</body></html>");
    File.WriteAllText(filePath, html.ToString());

    return filePath;
  }


  static void AppendHtmlHeader(StringBuilder html, string pageTitle)
  {
    html.AppendLine("<!DOCTYPE html>");
    html.AppendLine("<html><head><meta charset=\"utf-8\">");
    html.AppendLine($"<title>{HtmlEncode(pageTitle)}</title>");
    html.AppendLine("<style>");
    html.AppendLine("body { font-family: 'Segoe UI', Arial, sans-serif; background: #dcdcdc; margin: 20px; }");
    html.AppendLine(".section { background: #eae8e4; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }");
    html.AppendLine(".section-title { font-size: 20px; font-weight: bold; margin-bottom: 12px; color: #333; }");
    html.AppendLine(".panel-container { display: flex; gap: 30px; align-items: flex-start; }");
    html.AppendLine(".legend { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; align-items: center; font-size: 12px; }");
    html.AppendLine(".legend-item { display: flex; align-items: center; gap: 3px; }");
    html.AppendLine(".legend-swatch { width: 14px; height: 14px; border: 1px solid #ccc; border-radius: 2px; }");
    html.AppendLine("h1 { color: #222; margin-bottom: 4px; }");
    html.AppendLine("h2 { color: #333; margin-top: 30px; margin-bottom: 4px; border-top: 2px solid #bbb; padding-top: 16px; }");
    html.AppendLine(".fen { font-family: monospace; color: #666; margin-bottom: 16px; font-size: 13px; }");
    html.AppendLine("</style>");
    html.AppendLine("</head><body>");
    html.AppendLine($"<h1>{HtmlEncode(pageTitle)}</h1>");
  }


  static void AppendPositionSections(StringBuilder html, string fen, Half[,] movementProbs,
                                     Half[,] captureProbs, string title, string idPrefix,
                                     string evalSummary = null, bool highlightTop4 = true,
                                     Half[] punimSelfProbs = null, Half[] punimOpponentProbs = null,
                                     List<PVCandidate> projectedPVs = null, float mlh = 0)
  {
    Position pos = Position.FromFEN(fen);

    html.AppendLine($"<h2>{HtmlEncode(title)}</h2>");
    html.AppendLine($"<div class=\"fen\">FEN: {HtmlEncode(fen)}</div>");
    if (evalSummary != null)
    {
      html.AppendLine($"<div class=\"fen\">{HtmlEncode(evalSummary)}</div>");
    }

    // Derive value estimate from king capture probabilities.
    {
      int ourKingSq = -1, theirKingSq = -1;
      SideType sideToMove = pos.SideToMove;
      for (int sq = 0; sq < 64; sq++)
      {
        Square square = Square.FromFileAndRank(sq % 8, sq / 8);
        Piece piece = pos.PieceOnSquare(square);
        if (piece.Type == PieceType.King)
        {
          if (piece.Side == sideToMove)
          {
            ourKingSq = sq;
          }
          else
          {
            theirKingSq = sq;
          }
        }
      }
      if (ourKingSq >= 0 && theirKingSq >= 0)
      {
        double pNeverOurKing = (double)captureProbs[ourKingSq, 7];
        double pNeverTheirKing = (double)captureProbs[theirKingSq, 7];
        double selfCaptureProb = 1.0 - pNeverTheirKing;
        double opponentCaptureProb = 1.0 - pNeverOurKing;
        double vCapture = selfCaptureProb - opponentCaptureProb;
        html.AppendLine($"<div class=\"fen\">From PieceCapture: V = {vCapture:F2} ({selfCaptureProb:F2} - {opponentCaptureProb:F2})</div>");
      }
    }

    // Projected PV lines.
    if (projectedPVs != null && projectedPVs.Count > 0)
    {
      for (int i = 0; i < projectedPVs.Count; i++)
      {
        PVCandidate pv = projectedPVs[i];
        string sanLine = PVExtractor.FormatPVasSAN(pv, pos);
        string label = i == 0 ? "Projected PV:" : $"Alt PV {i}:    ";
        string scoreStr = $"(score: {pv.LogScore:F2})";
        html.AppendLine($"<div style=\"font-family: monospace; color: #444; margin-bottom: {(i == projectedPVs.Count - 1 ? 12 : 2)}px; font-size: 13px;\">");
        html.AppendLine($"  {HtmlEncode(label)} {HtmlEncode(sanLine)} <span style=\"color:#999\">{HtmlEncode(scoreStr)}</span>");
        html.AppendLine("</div>");
      }
    }

    // PUNIM section (first panel).
    if (punimSelfProbs != null || punimOpponentProbs != null)
    {
      string mlhText = mlh > 0 ? $" <span style=\"font-weight:normal; font-size:16px; color:#555;\">MLH = {Math.Round(mlh)} ply</span>" : "";
      html.AppendLine("<div class=\"section\">");
      html.AppendLine($"<div class=\"section-title\">Ply Until Next Irreversible Move (PUNIM){mlhText}</div>");
      html.AppendLine("<div class=\"panel-container\">");
      AppendPunimPanel(html, pos, punimSelfProbs, punimOpponentProbs, mlh);
      html.AppendLine("</div>");
      html.AppendLine("</div>");
    }

    // Occupancy change section.
    html.AppendLine("<div class=\"section\">");
    html.AppendLine("<div class=\"section-title\">Square Occupancy Change (ply until square changes occupancy)</div>");
    html.AppendLine("<div class=\"panel-container\">");
    AppendBoardAndHistogram(html, pos, movementProbs, $"{idPrefix}_movement", showAllSquares: true, highlightTop4: highlightTop4);
    html.AppendLine("</div>");
    html.AppendLine("</div>");

    // Capture section.
    html.AppendLine("<div class=\"section\">");
    html.AppendLine("<div class=\"section-title\">Piece Capture (ply until piece captured)</div>");
    html.AppendLine("<div class=\"panel-container\">");
    AppendBoardAndHistogram(html, pos, captureProbs, $"{idPrefix}_capture", showAllSquares: false, highlightTop4: highlightTop4);
    html.AppendLine("</div>");
    html.AppendLine("</div>");
  }


  static void AppendPunimPanel(StringBuilder html, Position pos, Half[] selfProbs, Half[] opponentProbs, float mlh = 0)
  {
    string sideLabel = pos.SideToMove == SideType.White ? "White" : "Black";
    string oppLabel = pos.SideToMove == SideType.White ? "Black" : "White";

    // Compute stats for a distribution: expected ply (excl. never), mode bin, never%.
    static (double expected, double modeMid, double neverPct) ComputeStats(Half[] probs)
    {
      if (probs == null)
      {
        return (0, 0, 0);
      }

      double pActive = 0;
      int modeBin = 0;
      double maxProb = -1;
      for (int b = 0; b < 7; b++)
      {
        double p = (double)probs[b];
        pActive += p;
        if (p > maxProb)
        {
          maxProb = p;
          modeBin = b;
        }
      }
      double neverPct = (double)probs[7] * 100.0;
      double expected = 0;
      if (pActive > 0.01)
      {
        for (int b = 0; b < 7; b++)
        {
          expected += (double)probs[b] * BIN_MIDPOINTS[b];
        }
        expected /= pActive;
      }
      return (expected, BIN_MIDPOINTS[modeBin], neverPct);
    }

    (double selfExp, double selfMode, double selfNever) = ComputeStats(selfProbs);
    (double oppExp, double oppMode, double oppNever) = ComputeStats(opponentProbs);

    // Absolute density bar layout (same as histogram bars).
    int barWidth = 300;
    int labelWidth = 120;
    int statsWidth = 220;
    int totalWidth = labelWidth + barWidth + statsWidth;
    int rowHeight = 28;
    int topPadding = 5;
    int rows = (selfProbs != null ? 1 : 0) + (opponentProbs != null ? 1 : 0);
    int totalHeight = topPadding + rows * rowHeight + 5;

    double extraBinPlyWidth = 65.0 * 0.20;
    int totalSegments = 65 + (int)(2 * extraBinPlyWidth);
    double segWidth = (double)barWidth / totalSegments;
    double extraBinDisplayWidth = extraBinPlyWidth * segWidth;

    // Compute max density across both distributions.
    double maxDensity = 0;
    foreach (Half[] probs in new[] { selfProbs, opponentProbs })
    {
      if (probs == null)
      {
        continue;
      }
      for (int ply = 1; ply <= 65; ply++)
      {
        int bin = PlyToBinIndex(ply);
        double density = (double)probs[bin] / BIN_WIDTHS[bin];
        if (density > maxDensity)
        {
          maxDensity = density;
        }
      }
      double d6 = (double)probs[6] / extraBinPlyWidth;
      double d7 = (double)probs[7] / extraBinPlyWidth;
      if (d6 > maxDensity) maxDensity = d6;
      if (d7 > maxDensity) maxDensity = d7;
    }
    if (maxDensity < 1e-12) maxDensity = 1e-12;

    // MLH marker x-position (fixed for both bars).
    double mlhMarkerX = -1;
    if (mlh > 0)
    {
      if (mlh <= 65)
      {
        mlhMarkerX = labelWidth + (mlh - 1) * segWidth;
      }
      else if (mlh <= 90)
      {
        double block66Start = labelWidth + 65 * segWidth;
        double frac = (mlh - 65.0) / 25.0;
        mlhMarkerX = block66Start + frac * extraBinDisplayWidth;
      }
      else
      {
        mlhMarkerX = labelWidth + barWidth;
      }
    }

    html.AppendLine($"<svg width=\"{totalWidth}\" height=\"{totalHeight}\" viewBox=\"0 0 {totalWidth} {totalHeight}\" xmlns=\"http://www.w3.org/2000/svg\">");

    int row = 0;

    // Render a single PUNIM density bar row.
    void RenderRow(string label, Half[] probs, double expected, double modeMid, double neverPct)
    {
      if (probs == null)
      {
        return;
      }
      int y = topPadding + row * rowHeight;
      int barHeight = rowHeight - 8;

      // Label.
      html.AppendLine($"  <text x=\"{labelWidth - 5}\" y=\"{y + 18}\" text-anchor=\"end\" font-size=\"13\" font-family=\"'Segoe UI', Arial, sans-serif\" font-weight=\"bold\" fill=\"#333\">{HtmlEncode(label)}</text>");

      // Density segments for plies 1-65.
      double xOff = labelWidth;
      for (int ply = 1; ply <= 65; ply++)
      {
        int bin = PlyToBinIndex(ply);
        double density = (double)probs[bin] / BIN_WIDTHS[bin];
        string color = DensityColor(density, maxDensity);
        html.AppendLine($"  <rect x=\"{xOff:F2}\" y=\"{y + 4}\" width=\"{segWidth:F2}\" height=\"{barHeight}\" fill=\"{color}\" />");
        xOff += segWidth;
      }

      // 66+ block.
      double density6 = (double)probs[6] / extraBinPlyWidth;
      html.AppendLine($"  <rect x=\"{xOff:F2}\" y=\"{y + 4}\" width=\"{extraBinDisplayWidth:F2}\" height=\"{barHeight}\" fill=\"{DensityColor(density6, maxDensity)}\" />");
      xOff += extraBinDisplayWidth;

      // Never block.
      double density7 = (double)probs[7] / extraBinPlyWidth;
      html.AppendLine($"  <rect x=\"{xOff:F2}\" y=\"{y + 4}\" width=\"{extraBinDisplayWidth:F2}\" height=\"{barHeight}\" fill=\"{DensityColor(density7, maxDensity)}\" />");

      // Bar outline.
      html.AppendLine($"  <rect x=\"{labelWidth}\" y=\"{y + 4}\" width=\"{barWidth}\" height=\"{barHeight}\" fill=\"none\" stroke=\"#ccc\" stroke-width=\"0.5\" rx=\"2\" />");

      // MLH vertical marker line (same position for both bars).
      if (mlhMarkerX >= 0)
      {
        html.AppendLine($"  <line x1=\"{mlhMarkerX:F1}\" y1=\"{y + 2}\" x2=\"{mlhMarkerX:F1}\" y2=\"{y + rowHeight - 2}\" stroke=\"#000\" stroke-width=\"1.5\" />");
      }

      // Stats text.
      string statsText = $"avg ~{expected:F0} ply, mode {modeMid:F0} (never: {neverPct:F0}%)";
      html.AppendLine($"  <text x=\"{labelWidth + barWidth + 8}\" y=\"{y + 18}\" font-size=\"12\" font-family=\"monospace\" fill=\"#555\">{statsText}</text>");
      row++;
    }

    RenderRow($"{sideLabel} (self)", selfProbs, selfExp, selfMode, selfNever);
    RenderRow($"{oppLabel} (opponent)", opponentProbs, oppExp, oppMode, oppNever);

    html.AppendLine("</svg>");

    // Density legend below (matching histogram legend).
    AppendDensityBarLegend(html);
  }


  static void AppendBoardAndHistogram(StringBuilder html, Position pos, Half[,] probs, string idPrefix,
                                      bool showAllSquares, bool highlightTop4 = false)
  {
    // Compute expected ply (for histogram sorting) and density-weighted
    // color values (for board heatmap) for each square.
    double[] expectedPly = new double[64];
    for (int sq = 0; sq < 64; sq++)
    {
      double ePly = 0;
      for (int b = 0; b < 8; b++)
      {
        ePly += (double)probs[sq, b] * BIN_MIDPOINTS[b];
      }
      expectedPly[sq] = ePly;
    }

    // Compute top-4 highlighted squares (top 2 = strong, next 2 = light).
    HashSet<int> highlighted = new();
    HashSet<int> highlightedStrong = new();
    if (highlightTop4)
    {
      // Build (sqIdx, metric) for candidate squares.
      List<(int sqIdx, double metric)> candidates = new();
      for (int sq = 0; sq < 64; sq++)
      {
        if (showAllSquares)
        {
          // Occupancy change: sum of probs in two smallest bins (1-2 + 3-4 ply).
          double metric = (double)probs[sq, 0] + (double)probs[sq, 1];
          candidates.Add((sq, metric));
        }
        else
        {
          // Capture: only occupied squares; sum of probs in two smallest bins.
          Square square = Square.FromFileAndRank(sq % 8, sq / 8);
          if (pos.PieceOnSquare(square).Type != PieceType.None)
          {
            double metric = (double)probs[sq, 0] + (double)probs[sq, 1];
            candidates.Add((sq, metric));
          }
        }
      }
      candidates.Sort((a, b) => b.metric.CompareTo(a.metric));
      int count = Math.Min(4, candidates.Count);
      for (int i = 0; i < count; i++)
      {
        highlighted.Add(candidates[i].sqIdx);
      }
      // Top 2 get full-width border, next 2 get half-width.
      for (int i = 0; i < Math.Min(2, candidates.Count); i++)
      {
        highlightedStrong.Add(candidates[i].sqIdx);
      }
    }

    // Build board SVG.
    html.AppendLine("<div>");
    AppendBoardSVG(html, pos, probs, idPrefix, showAllSquares, highlighted, highlightedStrong);
    AppendHeatmapLegend(html);
    html.AppendLine("</div>");

    // Build histogram SVG.
    html.AppendLine("<div>");
    AppendHistogramSVG(html, pos, probs, expectedPly, idPrefix, showAllSquares);
    if (USE_ABSOLUTE_COLOR_BARS)
    {
      AppendDensityBarLegend(html);
    }
    else
    {
      AppendBinLegend(html);
    }
    html.AppendLine("</div>");
  }


  /// <summary>
  /// Computes density-weighted average ply for a square using bin midpoints.
  /// BIN_MIDPOINTS uses 100 for 66+ and 300 for never.
  /// </summary>
  static double ComputeWeightedAvgPly(Half[,] probs, int sq)
  {
    double avg = 0;
    for (int b = 0; b < 8; b++)
    {
      avg += (double)probs[sq, b] * BIN_MIDPOINTS[b];
    }
    return avg;
  }


  /// <summary>
  /// Maps a weighted-average ply value to a color on a dark red (ply=1) to pale yellow (ply>=200) gradient.
  /// </summary>
  static string PlyToColor(double avgPly)
  {
    // Clamp to [1, 200] range, then normalize to [0, 1].
    double t = Math.Clamp((avgPly - 1.0) / (200.0 - 1.0), 0.0, 1.0);
    // Dark red (#a01020) at t=0 -> orange (#e07030) at t=0.3 -> pale yellow (#fdfae8) at t=1.
    int r, g, b;
    if (t < 0.3)
    {
      double f = t / 0.3;
      r = (int)(0xa0 + f * (0xe0 - 0xa0));
      g = (int)(0x10 + f * (0x70 - 0x10));
      b = (int)(0x20 + f * (0x30 - 0x20));
    }
    else
    {
      double f = (t - 0.3) / 0.7;
      r = (int)(0xe0 + f * (0xfd - 0xe0));
      g = (int)(0x70 + f * (0xfa - 0x70));
      b = (int)(0x30 + f * (0xe8 - 0x30));
    }
    return $"#{r:x2}{g:x2}{b:x2}";
  }


  static void AppendBoardSVG(StringBuilder html, Position pos, Half[,] probs,
                             string idPrefix, bool showAllSquares,
                             HashSet<int> highlighted = null, HashSet<int> highlightedStrong = null)
  {
    html.AppendLine($"<svg width=\"{BOARD_SVG_SIZE}\" height=\"{BOARD_SVG_SIZE}\" viewBox=\"0 0 {BOARD_SVG_SIZE} {BOARD_SVG_SIZE}\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">");

    // Piece defs.
    html.AppendLine(PIECE_DEFS);

    // Draw squares.
    for (int rank = 0; rank < 8; rank++)
    {
      for (int file = 0; file < 8; file++)
      {
        int x = BOARD_OFFSET + file * SQUARE_SIZE;
        int y = 335 - rank * SQUARE_SIZE;
        int sqIdx = file + 8 * rank;

        Square square = Square.FromFileAndRank(file, rank);
        Piece piece = pos.PieceOnSquare(square);
        bool occupied = piece.Type != PieceType.None;

        string fillColor;
        if (occupied || showAllSquares)
        {
          double avgPly = ComputeWeightedAvgPly(probs, sqIdx);
          fillColor = PlyToColor(avgPly);
        }
        else
        {
          fillColor = "#fdfae8";
        }

        html.AppendLine($"  <rect x=\"{x}\" y=\"{y}\" width=\"{SQUARE_SIZE}\" height=\"{SQUARE_SIZE}\" fill=\"{fillColor}\" stroke=\"none\" />");

        // Place piece glyph.
        if (occupied)
        {
          string pieceName = PieceSvgId(piece);
          html.AppendLine($"  <use transform=\"translate({x}, {y})\" xlink:href=\"#{pieceName}\" />");
        }

        // Highlight top-4 squares with a border (strong = 3px, light = 1.5px).
        if (highlighted != null && highlighted.Contains(sqIdx))
        {
          int inset = 2;
          bool strong = highlightedStrong != null && highlightedStrong.Contains(sqIdx);
          string sw = strong ? "3" : "1.5";
          html.AppendLine($"  <rect x=\"{x + inset}\" y=\"{y + inset}\" width=\"{SQUARE_SIZE - 2 * inset}\" height=\"{SQUARE_SIZE - 2 * inset}\" fill=\"none\" stroke=\"#2266cc\" stroke-width=\"{sw}\" />");
        }
      }
    }

    // File labels.
    string fileChars = "abcdefgh";
    for (int f = 0; f < 8; f++)
    {
      int cx = BOARD_OFFSET + f * SQUARE_SIZE + SQUARE_SIZE / 2;
      html.AppendLine($"  <text x=\"{cx}\" y=\"10\" text-anchor=\"middle\" alignment-baseline=\"middle\" font-size=\"14\">{fileChars[f]}</text>");
      html.AppendLine($"  <text x=\"{cx}\" y=\"390\" text-anchor=\"middle\" alignment-baseline=\"middle\" font-size=\"14\">{fileChars[f]}</text>");
    }

    // Rank labels.
    for (int r = 0; r < 8; r++)
    {
      int cy = 335 - r * SQUARE_SIZE + SQUARE_SIZE / 2;
      html.AppendLine($"  <text x=\"10\" y=\"{cy}\" text-anchor=\"middle\" alignment-baseline=\"middle\" font-size=\"14\">{r + 1}</text>");
      html.AppendLine($"  <text x=\"390\" y=\"{cy}\" text-anchor=\"middle\" alignment-baseline=\"middle\" font-size=\"14\">{r + 1}</text>");
    }

    html.AppendLine("</svg>");
  }


  static void AppendHistogramSVG(StringBuilder html, Position pos, Half[,] probs, double[] expectedPly, string idPrefix, bool showAllSquares)
  {
    SideType sideToMove = pos.SideToMove;

    // Collect squares.
    List<(int sqIdx, string label, double modeMid, double stddev, double neverPct, bool sideToMovePiece, bool isKing)> entries = new();
    for (int rank = 0; rank < 8; rank++)
    {
      for (int file = 0; file < 8; file++)
      {
        int sqIdx = file + 8 * rank;
        Square square = Square.FromFileAndRank(file, rank);
        Piece piece = pos.PieceOnSquare(square);
        bool occupied = piece.Type != PieceType.None;

        if (!occupied && !showAllSquares)
        {
          continue;
        }

        string label = occupied
          ? PieceLabel(piece, file, rank)
          : $".{"abcdefgh"[file]}{"12345678"[rank]}";

        bool isSideToMovePiece = occupied && piece.Side == sideToMove;
        bool isKing = occupied && piece.Type == PieceType.King;

        // Compute mode, mean, stddev over bins 0-6 (excluding never).
        double pActive = 0;
        int modeBin = 0;
        double maxProb = -1;
        for (int b = 0; b < 7; b++)
        {
          double p = (double)probs[sqIdx, b];
          pActive += p;
          if (p > maxProb)
          {
            maxProb = p;
            modeBin = b;
          }
        }

        double neverPct = (double)probs[sqIdx, 7] * 100.0;
        double modeMid = BIN_MIDPOINTS[modeBin];
        double stddev = 0;

        if (pActive > 0.01)
        {
          double mean = 0;
          for (int b = 0; b < 7; b++)
          {
            mean += (double)probs[sqIdx, b] * BIN_MIDPOINTS[b];
          }
          mean /= pActive;
          double variance = 0;
          for (int b = 0; b < 7; b++)
          {
            double diff = BIN_MIDPOINTS[b] - mean;
            variance += (double)probs[sqIdx, b] * diff * diff;
          }
          variance /= pActive;
          stddev = Math.Sqrt(variance);
        }

        entries.Add((sqIdx, label, modeMid, stddev, neverPct, isSideToMovePiece, isKing));
      }
    }

    // Sort: ascending mode + half stddev, then side-to-move pieces first on ties.
    entries.Sort((a, b) =>
    {
      double aKey = a.modeMid + 0.5 * a.stddev;
      double bKey = b.modeMid + 0.5 * b.stddev;
      int cmp = aKey.CompareTo(bKey);
      if (cmp != 0)
      {
        return cmp;
      }
      // Prefer side-to-move pieces (true sorts before false).
      return b.sideToMovePiece.CompareTo(a.sideToMovePiece);
    });

    // Limit to top N for occupancy change view.
    int maxEntries = showAllSquares ? 20 : entries.Count;
    int displayCount = Math.Min(entries.Count, maxEntries);

    int rowHeight = 22;
    int labelWidth = 60;
    int topPadding = 5;

    if (USE_ABSOLUTE_COLOR_BARS)
    {
      // 66+ and never each get a virtual ply-width of 20% of 65 = 13 plies.
      // Total segments: 65 + 13 + 13 = 91 segment-widths.
      double extraBinPlyWidth = 65.0 * 0.20;
      int totalSegments = 65 + (int)(2 * extraBinPlyWidth);
      int barWidth = 300;
      double segWidth = (double)barWidth / totalSegments;
      double extraBinDisplayWidth = extraBinPlyWidth * segWidth;

      // Compute global max density across all displayed squares (including 66+ and never).
      double maxDensity = 0;
      for (int i = 0; i < displayCount; i++)
      {
        int sqIdx = entries[i].sqIdx;
        for (int ply = 1; ply <= 65; ply++)
        {
          int bin = PlyToBinIndex(ply);
          double density = (double)probs[sqIdx, bin] / BIN_WIDTHS[bin];
          if (density > maxDensity)
          {
            maxDensity = density;
          }
        }
        double d6 = (double)probs[sqIdx, 6] / extraBinPlyWidth;
        double d7 = (double)probs[sqIdx, 7] / extraBinPlyWidth;
        if (d6 > maxDensity) maxDensity = d6;
        if (d7 > maxDensity) maxDensity = d7;
      }
      if (maxDensity < 1e-12)
      {
        maxDensity = 1e-12;
      }

      int valueWidth = 180;
      int totalWidth = labelWidth + barWidth + valueWidth;
      int totalHeight = topPadding + displayCount * rowHeight + 5;

      html.AppendLine($"<svg width=\"{totalWidth}\" height=\"{totalHeight}\" viewBox=\"0 0 {totalWidth} {totalHeight}\" xmlns=\"http://www.w3.org/2000/svg\">");
      html.AppendLine("  <defs><pattern id=\"king-hatch\" width=\"4\" height=\"4\" patternUnits=\"userSpaceOnUse\" patternTransform=\"rotate(45)\">");
      html.AppendLine("    <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"4\" stroke=\"rgba(0,0,0,0.45)\" stroke-width=\"1.5\" />");
      html.AppendLine("  </pattern></defs>");

      for (int i = 0; i < displayCount; i++)
      {
        (int sqIdx, string label, double modeMid, double stddev, double neverPct, bool _, bool isKing) = entries[i];
        int y = topPadding + i * rowHeight;
        int barHeight = rowHeight - 6;

        // Piece label (red for kings in capture mode).
        string labelFill = (!showAllSquares && isKing) ? "#cc2222" : "#000";
        html.AppendLine($"  <text x=\"{labelWidth - 5}\" y=\"{y + 15}\" text-anchor=\"end\" font-size=\"13\" font-family=\"monospace\" fill=\"{labelFill}\" font-weight=\"{(!showAllSquares && isKing ? "bold" : "normal")}\">{HtmlEncode(label)}</text>");

        // 65-ply density segments.
        double xOff = labelWidth;
        for (int ply = 1; ply <= 65; ply++)
        {
          int bin = PlyToBinIndex(ply);
          double density = (double)probs[sqIdx, bin] / BIN_WIDTHS[bin];
          string color = DensityColor(density, maxDensity);
          html.AppendLine($"  <rect x=\"{xOff:F2}\" y=\"{y + 3}\" width=\"{segWidth:F2}\" height=\"{barHeight}\" fill=\"{color}\" />");
          xOff += segWidth;
        }

        // Bin 6 (66+) block.
        double density6 = (double)probs[sqIdx, 6] / extraBinPlyWidth;
        html.AppendLine($"  <rect x=\"{xOff:F2}\" y=\"{y + 3}\" width=\"{extraBinDisplayWidth:F2}\" height=\"{barHeight}\" fill=\"{DensityColor(density6, maxDensity)}\" />");
        xOff += extraBinDisplayWidth;

        // Bin 7 (never) block.
        double density7 = (double)probs[sqIdx, 7] / extraBinPlyWidth;
        html.AppendLine($"  <rect x=\"{xOff:F2}\" y=\"{y + 3}\" width=\"{extraBinDisplayWidth:F2}\" height=\"{barHeight}\" fill=\"{DensityColor(density7, maxDensity)}\" />");
        if (!showAllSquares && isKing)
        {
          html.AppendLine($"  <rect x=\"{xOff:F2}\" y=\"{y + 3}\" width=\"{extraBinDisplayWidth:F2}\" height=\"{barHeight}\" fill=\"url(#king-hatch)\" />");
        }

        // Bar outline.
        html.AppendLine($"  <rect x=\"{labelWidth}\" y=\"{y + 3}\" width=\"{barWidth}\" height=\"{barHeight}\" fill=\"none\" stroke=\"#ccc\" stroke-width=\"0.5\" />");

        // Mode +/- stddev (never: X%).
        string valueText = $"{modeMid:F0} +/- {stddev:F0} (never: {neverPct:F0}%)";
        html.AppendLine($"  <text x=\"{labelWidth + barWidth + 5}\" y=\"{y + 15}\" font-size=\"12\" font-family=\"monospace\" fill=\"#555\">{valueText}</text>");
      }

      html.AppendLine("</svg>");
    }
    else
    {
      int barWidth = 300;
      int valueWidth = 180;
      int totalWidth = labelWidth + barWidth + valueWidth;
      int totalHeight = topPadding + displayCount * rowHeight + 5;

      html.AppendLine($"<svg width=\"{totalWidth}\" height=\"{totalHeight}\" viewBox=\"0 0 {totalWidth} {totalHeight}\" xmlns=\"http://www.w3.org/2000/svg\">");
      html.AppendLine("  <defs><pattern id=\"king-hatch2\" width=\"4\" height=\"4\" patternUnits=\"userSpaceOnUse\" patternTransform=\"rotate(45)\">");
      html.AppendLine("    <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"4\" stroke=\"rgba(0,0,0,0.45)\" stroke-width=\"1.5\" />");
      html.AppendLine("  </pattern></defs>");

      for (int i = 0; i < displayCount; i++)
      {
        (int sqIdx, string label, double modeMid, double stddev, double neverPct, bool _, bool isKing) = entries[i];
        int y = topPadding + i * rowHeight;

        // Piece label (red for kings in capture mode).
        string labelFill = (!showAllSquares && isKing) ? "#cc2222" : "#000";
        html.AppendLine($"  <text x=\"{labelWidth - 5}\" y=\"{y + 15}\" text-anchor=\"end\" font-size=\"13\" font-family=\"monospace\" fill=\"{labelFill}\" font-weight=\"{(!showAllSquares && isKing ? "bold" : "normal")}\">{HtmlEncode(label)}</text>");

        // Stacked bar.
        double xOff = labelWidth;
        double neverX = 0, neverW = 0;
        for (int b = 0; b < 8; b++)
        {
          double prob = (double)probs[sqIdx, b];
          double w = prob * barWidth;
          if (b == 7)
          {
            neverX = xOff;
            neverW = w;
          }
          if (w < 0.5)
          {
            continue;
          }
          html.AppendLine($"  <rect x=\"{xOff:F1}\" y=\"{y + 3}\" width=\"{w:F1}\" height=\"{rowHeight - 6}\" fill=\"{BIN_COLORS[b]}\" />");
          xOff += w;
        }
        if (!showAllSquares && isKing && neverW >= 0.5)
        {
          html.AppendLine($"  <rect x=\"{neverX:F1}\" y=\"{y + 3}\" width=\"{neverW:F1}\" height=\"{rowHeight - 6}\" fill=\"url(#king-hatch2)\" />");
        }

        // Bar outline.
        html.AppendLine($"  <rect x=\"{labelWidth}\" y=\"{y + 3}\" width=\"{barWidth}\" height=\"{rowHeight - 6}\" fill=\"none\" stroke=\"#ccc\" stroke-width=\"0.5\" />");

        // Mode +/- stddev (never: X%).
        string valueText = $"{modeMid:F0} +/- {stddev:F0} (never: {neverPct:F0}%)";
        html.AppendLine($"  <text x=\"{labelWidth + barWidth + 5}\" y=\"{y + 15}\" font-size=\"12\" font-family=\"monospace\" fill=\"#555\">{valueText}</text>");
      }

      html.AppendLine("</svg>");
    }
  }


  static void AppendHeatmapLegend(StringBuilder html)
  {
    html.AppendLine("<div class=\"legend\">");
    html.AppendLine("<span style=\"font-weight:bold;\">Avg ply:</span>");
    html.AppendLine("<span>1</span>");

    int steps = 20;
    html.Append("<span style=\"display:flex;\">");
    for (int i = 0; i < steps; i++)
    {
      double ply = 1.0 + (double)i / (steps - 1) * 199.0;
      string color = PlyToColor(ply);
      html.Append($"<span style=\"width:8px;height:14px;background:{color};\"></span>");
    }
    html.Append("</span>");

    html.AppendLine("<span>200+</span>");
    html.AppendLine("</div>");
  }


  static void AppendBinLegend(StringBuilder html)
  {
    html.AppendLine("<div class=\"legend\">");
    html.AppendLine("<span style=\"font-weight:bold;\">Bins:</span>");
    for (int b = 0; b < 8; b++)
    {
      html.AppendLine($"<span class=\"legend-item\"><span class=\"legend-swatch\" style=\"background:{BIN_COLORS[b]};\"></span>{BIN_LABELS[b]}</span>");
    }
    html.AppendLine("</div>");
  }


  static void AppendDensityBarLegend(StringBuilder html)
  {
    html.AppendLine("<div class=\"legend\">");
    html.AppendLine("<span style=\"font-weight:bold;\">Density:</span>");
    html.AppendLine("<span>low</span>");

    int steps = 12;
    html.Append("<span style=\"display:flex;\">");
    for (int i = 0; i < steps; i++)
    {
      double t = (double)i / (steps - 1);
      string color = DensityColor(t, 1.0);
      html.Append($"<span style=\"width:6px;height:14px;background:{color};\"></span>");
    }
    html.Append("</span>");

    html.AppendLine("<span>high</span>");
    html.AppendLine("<span style=\"margin-left:12px;color:#888;\">Bar: plies 1-65 then 66+ and never</span>");
    html.AppendLine("</div>");
  }


  static int PlyToBinIndex(int ply) => ply switch
  {
    <= 2  => 0,
    <= 4  => 1,
    <= 10 => 2,
    <= 22 => 3,
    <= 40 => 4,
    _     => 5
  };


  static string DensityColor(double density, double maxDensity)
  {
    double t = maxDensity > 1e-12 ? Math.Clamp(density / maxDensity, 0, 1) : 0;
    // Pale yellow (#fffde4) -> orange (#fd8d3c) -> red (#d73027).
    int r, g, b;
    if (t < 0.5)
    {
      double f = t / 0.5;
      r = (int)(0xff + f * (0xfd - 0xff));
      g = (int)(0xfd + f * (0x8d - 0xfd));
      b = (int)(0xe4 + f * (0x3c - 0xe4));
    }
    else
    {
      double f = (t - 0.5) / 0.5;
      r = (int)(0xfd + f * (0xd7 - 0xfd));
      g = (int)(0x8d + f * (0x30 - 0x8d));
      b = (int)(0x3c + f * (0x27 - 0x3c));
    }
    return $"#{r:x2}{g:x2}{b:x2}";
  }


  static string HeatmapColor(double expectedPly)
  {
    double t = Math.Log(Math.Max(1.0, expectedPly)) / Math.Log(100.0);
    t = Math.Clamp(t, 0.0, 1.0);

    // Interpolate through gradient stops.
    for (int i = 0; i < GRADIENT_STOPS.Length - 1; i++)
    {
      (double t0, int r0, int g0, int b0) = GRADIENT_STOPS[i];
      (double t1, int r1, int g1, int b1) = GRADIENT_STOPS[i + 1];
      if (t <= t1 || i == GRADIENT_STOPS.Length - 2)
      {
        double f = (t1 > t0) ? (t - t0) / (t1 - t0) : 0;
        f = Math.Clamp(f, 0.0, 1.0);
        int r = (int)(r0 + f * (r1 - r0));
        int g = (int)(g0 + f * (g1 - g0));
        int b = (int)(b0 + f * (b1 - b0));
        return $"#{r:x2}{g:x2}{b:x2}";
      }
    }

    return "#808080";
  }


  static string PieceSvgId(Piece piece)
  {
    string color = piece.Side == SideType.White ? "white" : "black";
    string name = piece.Type switch
    {
      PieceType.Pawn   => "pawn",
      PieceType.Knight => "knight",
      PieceType.Bishop => "bishop",
      PieceType.Rook   => "rook",
      PieceType.Queen  => "queen",
      PieceType.King   => "king",
      _                => throw new ArgumentException($"Unknown piece type: {piece.Type}")
    };
    return $"{color}-{name}";
  }


  static string PieceLabel(Piece piece, int file, int rank)
  {
    char pieceChar = piece.Type switch
    {
      PieceType.Pawn   => 'P',
      PieceType.Knight => 'N',
      PieceType.Bishop => 'B',
      PieceType.Rook   => 'R',
      PieceType.Queen  => 'Q',
      PieceType.King   => 'K',
      _                => '?'
    };

    string sideIndicator = piece.Side == SideType.White ? "" : "*";
    string squareName = $"{"abcdefgh"[file]}{"12345678"[rank]}";
    string piecePrefix = pieceChar == 'P' ? "" : pieceChar.ToString();
    return $"{sideIndicator}{piecePrefix}{squareName}";
  }


  static string HtmlEncode(string text)
  {
    return text.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");
  }


  // SVG piece definitions extracted from PositionToSVG.
  const string PIECE_DEFS = @"<defs>
<g class=""white pawn"" id=""white-pawn""><path d=""M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"" fill=""#fff"" stroke=""#000"" stroke-linecap=""round"" stroke-width=""1.5"" /></g>
<g class=""white knight"" fill=""none"" fill-rule=""evenodd"" id=""white-knight"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18"" style=""fill:#ffffff; stroke:#000000;"" /><path d=""M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10"" style=""fill:#ffffff; stroke:#000000;"" /><path d=""M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z"" style=""fill:#000000; stroke:#000000;"" /><path d=""M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z"" style=""fill:#000000; stroke:#000000;"" transform=""matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)"" /></g>
<g class=""white bishop"" fill=""none"" fill-rule=""evenodd"" id=""white-bishop"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><g fill=""#fff"" stroke-linecap=""butt""><path d=""M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zM15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"" /></g><path d=""M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5"" stroke-linejoin=""miter"" /></g>
<g class=""white rook"" fill=""#fff"" fill-rule=""evenodd"" id=""white-rook"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5"" stroke-linecap=""butt"" /><path d=""M34 14l-3 3H14l-3-3"" /><path d=""M31 17v12.5H14V17"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M31 29.5l1.5 2.5h-20l1.5-2.5"" /><path d=""M11 14h23"" fill=""none"" stroke-linejoin=""miter"" /></g>
<g class=""white queen"" fill=""#fff"" fill-rule=""evenodd"" id=""white-queen"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM16 8.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM33 9a2 2 0 1 1-4 0 2 2 0 1 1 4 0z"" /><path d=""M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14l2 12zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"" stroke-linecap=""butt"" /><path d=""M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0"" fill=""none"" /></g>
<g class=""white king"" fill=""none"" fill-rule=""evenodd"" id=""white-king"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M22.5 11.63V6M20 8h5"" stroke-linejoin=""miter"" /><path d=""M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"" fill=""#fff"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z"" fill=""#fff"" /><path d=""M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"" /></g>
<g class=""black pawn"" id=""black-pawn""><path d=""M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"" stroke=""#000"" stroke-linecap=""round"" stroke-width=""1.5"" /></g>
<g class=""black knight"" fill=""none"" fill-rule=""evenodd"" id=""black-knight"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18"" style=""fill:#000000; stroke:#000000;"" /><path d=""M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10"" style=""fill:#000000; stroke:#000000;"" /><path d=""M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z"" style=""fill:#ececec; stroke:#ececec;"" /><path d=""M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z"" style=""fill:#ececec; stroke:#ececec;"" transform=""matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)"" /><path d=""M 24.55,10.4 L 24.1,11.85 L 24.6,12 C 27.75,13 30.25,14.49 32.5,18.75 C 34.75,23.01 35.75,29.06 35.25,39 L 35.2,39.5 L 37.45,39.5 L 37.5,39 C 38,28.94 36.62,22.15 34.25,17.66 C 31.88,13.17 28.46,11.02 25.06,10.5 L 24.55,10.4 z "" style=""fill:#ececec; stroke:none;"" /></g>
<g class=""black bishop"" fill=""none"" fill-rule=""evenodd"" id=""black-bishop"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zm6-4c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"" fill=""#000"" stroke-linecap=""butt"" /><path d=""M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5"" stroke=""#fff"" stroke-linejoin=""miter"" /></g>
<g class=""black rook"" fill=""#000"" fill-rule=""evenodd"" id=""black-rook"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M9 39h27v-3H9v3zM12.5 32l1.5-2.5h17l1.5 2.5h-20zM12 36v-4h21v4H12z"" stroke-linecap=""butt"" /><path d=""M14 29.5v-13h17v13H14z"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M14 16.5L11 14h23l-3 2.5H14zM11 14V9h4v2h5V9h5v2h5V9h4v5H11z"" stroke-linecap=""butt"" /><path d=""M12 35.5h21M13 31.5h19M14 29.5h17M14 16.5h17M11 14h23"" fill=""none"" stroke=""#fff"" stroke-linejoin=""miter"" stroke-width=""1"" /></g>
<g class=""black queen"" fill=""#000"" fill-rule=""evenodd"" id=""black-queen"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><g fill=""#000"" stroke=""none""><circle cx=""6"" cy=""12"" r=""2.75"" /><circle cx=""14"" cy=""9"" r=""2.75"" /><circle cx=""22.5"" cy=""8"" r=""2.75"" /><circle cx=""31"" cy=""9"" r=""2.75"" /><circle cx=""39"" cy=""12"" r=""2.75"" /></g><path d=""M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5 9 26zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"" stroke-linecap=""butt"" /><path d=""M11 38.5a35 35 1 0 0 23 0"" fill=""none"" stroke-linecap=""butt"" /><path d=""M11 29a35 35 1 0 1 23 0M12.5 31.5h20M11.5 34.5a35 35 1 0 0 22 0M10.5 37.5a35 35 1 0 0 24 0"" fill=""none"" stroke=""#fff"" /></g>
<g class=""black king"" fill=""none"" fill-rule=""evenodd"" id=""black-king"" stroke=""#000"" stroke-linecap=""round"" stroke-linejoin=""round"" stroke-width=""1.5""><path d=""M22.5 11.63V6"" stroke-linejoin=""miter"" /><path d=""M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"" fill=""#000"" stroke-linecap=""butt"" stroke-linejoin=""miter"" /><path d=""M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z"" fill=""#000"" /><path d=""M20 8h5"" stroke-linejoin=""miter"" /><path d=""M32 29.5s8.5-4 6.03-9.65C34.15 14 25 18 22.5 24.5l.01 2.1-.01-2.1C20 18 9.906 14 6.997 19.85c-2.497 5.65 4.853 9 4.853 9M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"" stroke=""#fff"" /></g>
</defs>";
}
