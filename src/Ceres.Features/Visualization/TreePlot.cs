#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;
using System.Linq;
using System.Drawing;
using System.Diagnostics;
using System.Globalization;
using Ceres.MCTS.MTCSNodes.Struct;
using System.IO;

namespace Ceres.Features.Visualization.TreePlot
{
  /// <summary>
  /// Plot of the search tree that can be saved to disk.
  /// </summary>
  public class TreePlot : IDisposable
  {
    readonly DrawTreeNode root;
    readonly MCTSNodeStruct rawRoot;

    //Drawing is not able to handle line width 0.5, so we draw everything in x2 and scale down.
    readonly int superSample = 2;
    readonly int canvasWidth = 1920;
    readonly int canvasHeight = 1080;
    float scaleX;
    float scaleY;
    readonly float pointRadius = 2.0f;
    readonly float edgeWidth = 0.5f;
    int leftMargin = 10;
    readonly int rightMargin = 10;
    int topMargin = 10;
    readonly int bottomMargin = 10;

    readonly int rightHistogramWidth = 200;
    readonly int bottomHistogramHeight = 200;
    // Spacing between treeplot and the labels on left and right.
    readonly int horisontalSpacing = 5;
    // Spacing between treeplot and bottom histogram.
    readonly int verticalSpacing = 5;
    // Spacing between plot title and tree plot.
    readonly int titleMargin = 20;

    int plotAreaWidth;
    int plotAreaHeight;

    readonly int histogramTitleFontSize = 15;
    readonly int tickFontSize = 12;
    readonly Bitmap image;

    readonly Color rootNodeColor = Color.FromArgb(255, 255, 0, 0);
    readonly Color evenNodeColor = Color.FromArgb(255, 255, 127, 14);
    readonly Color oddNodeColor = Color.FromArgb(255, 31, 119, 180);
    readonly Color edgeColor = Color.FromArgb(191, 0, 0, 0);
    readonly Color gridLineColor = Color.FromArgb(63, 127, 127, 127);
    readonly Color fontColor = Color.FromArgb(255, 127, 127, 127);

    readonly DrawTreeInfo treeInfo;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="node"></param>
    public TreePlot(MCTSNodeStruct node)
    {
      rawRoot = node;
      //Stopwatch sw = Stopwatch.StartNew();
      (root, treeInfo) = DrawTreeNode.Layout(node);
      //Console.WriteLine("Layout time in ms:");
      //Console.WriteLine(sw.ElapsedMilliseconds);
      canvasHeight *= superSample;
      canvasWidth *= superSample;
      pointRadius *= superSample;
      edgeWidth *= superSample;
      leftMargin *= superSample;
      rightMargin *= superSample;
      topMargin *= superSample;
      bottomMargin *= superSample;
      rightHistogramWidth *= superSample;
      bottomHistogramHeight *= superSample;
      horisontalSpacing *= superSample;
      verticalSpacing *= superSample;
      titleMargin *= superSample;
      tickFontSize *= superSample;
      histogramTitleFontSize *= superSample;
      plotAreaHeight = canvasHeight - topMargin - bottomMargin - bottomHistogramHeight - verticalSpacing - titleMargin;
      plotAreaWidth = canvasWidth - leftMargin - rightMargin - rightHistogramWidth - 2 * horisontalSpacing;
      image = new Bitmap(canvasWidth / superSample, canvasHeight / superSample);
      Plot();
    }

    internal void Plot()
    {
      using (var bmp = new Bitmap(canvasWidth, canvasHeight))
      using (var gfx = Graphics.FromImage(bmp))
      using (var gfxFinal = Graphics.FromImage(image))
      using (var penEdge = new Pen(edgeColor, edgeWidth))
      using (var brushRoot = new SolidBrush(rootNodeColor))
      using (var brushEven = new SolidBrush(evenNodeColor))
      using (var brushOdd = new SolidBrush(oddNodeColor))
      {
        gfx.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        gfx.Clear(Color.White);
        gfxFinal.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

        //Adjust left margin as it needs to contains depth axis' title and tick labels.
        using (var font = new Font("Arial", tickFontSize))
        using (var titleFont = new Font("Arial", histogramTitleFontSize))
        {
          SizeF titleSize = gfx.MeasureString("Depth", titleFont);
          SizeF size = gfx.MeasureString(treeInfo.MaxDepth.ToString(), font);

          // Update space reserved for left margin.
          // Title height is used here since text is rotated by 90 degrees.
          leftMargin += (int)(titleSize.Height + size.Width);
          plotAreaWidth = canvasWidth - leftMargin - rightMargin - rightHistogramWidth - 2 * horisontalSpacing;
        }

        //Write tree statistics into top margin.
        TreeStats(gfx);

        // Multipiers used for transforming node's x,y-coordinates to canvas coordinates.
        scaleX = plotAreaWidth / treeInfo.MaxX;
        scaleY = plotAreaHeight / treeInfo.MaxDepth;

        // Draw grid lines.
        GridLines(gfx);

        // Draw right hand side histogram.
        NodesPerDepthHistogram(gfx);

        //Draw bottom histogram.
        VisitDistributionHistogram(gfx);

        //Draw the tree.
        //Stopwatch sw = Stopwatch.StartNew();
        DrawTree(root, gfx, penEdge, brushRoot, brushEven, brushOdd);
        //Console.WriteLine("Draw time in ms:");
        //Console.WriteLine(sw.ElapsedMilliseconds);

        // Scale down to final image size.
        gfxFinal.DrawImage(bmp, 0, 0, canvasWidth / superSample, canvasHeight / superSample);
      }
    }

    /// <summary>
    /// Plots the search tree for given root node and saves the image as png-file.
    /// </summary>
    /// <param name="rawNode"></param>
    public static void Save(MCTSNodeStruct rawNode, string fileName)
    {
      TreePlot treePlot = new TreePlot(rawNode);
      treePlot.Save(fileName);
      treePlot.Dispose();
    }

    internal void Save(string fileName)
    {
      fileName += fileName.EndsWith(".png") ? "" : ".png";
      image.Save(fileName);
    }

    /// <summary>
    /// Plots the search tree and opens it in image viewer.
    /// </summary>
    /// <param name="rawNode"></param>
    public static void Show(MCTSNodeStruct rawNode)
    {
      string file = Path.GetTempFileName() + ".png";
      Save(rawNode, file);
      new Process
      {
        StartInfo = new ProcessStartInfo(file)
        {
          UseShellExecute = true
        }
      }.Start();
    }

    /// <summary>
    /// Write tree statistics into top margin.
    /// </summary>
    /// <param name="gfx"></param>
    internal void TreeStats(Graphics gfx)
    {
      string text = "";
      text += "Node count " + treeInfo.NrNodes.ToString();
      //average branching factor = total nr of child nodes / total nr of nodes with children.
      double branchingFactor = Math.Round((treeInfo.NrNodes - 1) / ((float)(treeInfo.NrNodes - treeInfo.NrLeafNodes)), 3);
      text += ", Branching factor " + branchingFactor.ToString(CultureInfo.GetCultureInfo("en-GB"));
      double leafShare = Math.Round(100.0f * treeInfo.NrLeafNodes / treeInfo.NrNodes, 2);
      text += ", Leaf nodes " + leafShare.ToString(CultureInfo.GetCultureInfo("en-GB")) + "%";
      using (var brush = new SolidBrush(fontColor))
      using (var font = new Font("Arial", histogramTitleFontSize))
      {
        SizeF size = gfx.MeasureString(text, font);
        gfx.DrawString(text, font, brush, leftMargin + horisontalSpacing + plotAreaWidth / 2 - size.Width / 2, topMargin);
        topMargin += (int)size.Height + titleMargin;
        plotAreaHeight = canvasHeight - topMargin - bottomMargin - bottomHistogramHeight - verticalSpacing;
      }

    }

    /// <summary>
    /// Draw horizontal grid lines.
    /// </summary>
    /// <param name="gfx"></param>
    internal void GridLines(Graphics gfx)
    {
      float x0;
      float y;
      float x1;
      string label;
      string title = "Depth";

      using (var pen = new Pen(gridLineColor, edgeWidth))
      using (var brush = new SolidBrush(fontColor))
      using (var font = new Font("Arial", tickFontSize))
      using (var titleFont = new Font("Arial", histogramTitleFontSize))
      {
        SizeF titleSize = gfx.MeasureString(title, titleFont);
        SizeF size = gfx.MeasureString(treeInfo.MaxDepth.ToString(), font);
        gfx.RotateTransform(-90);
        gfx.DrawString(title, titleFont, brush, -(topMargin + plotAreaHeight / 2.0f + titleSize.Width / 2), leftMargin - titleSize.Height - size.Width - 1);
        gfx.ResetTransform();
        for (int i = 0; i <= treeInfo.MaxDepth; i++)
        {
          (x0, y) = CanvasXY(0, i);
          x1 = CanvasX(treeInfo.MaxX);
          gfx.DrawLine(pen, x0, y, x1, y);
          label = i.ToString();
          gfx.DrawString(label, font, brush, x0 - size.Width - horisontalSpacing, y + 1 - size.Height / 2);
        }
      }

    }

    /// <summary>
    /// Draw histogram visualizing number of nodes per depth.
    /// </summary>
    /// <param name="gfx"></param>
    internal void NodesPerDepthHistogram(Graphics gfx)
    {
      float x;
      float y;
      string label;

      string title = "Nodes per depth";

      int maxNodesPerDepth = treeInfo.NodesPerDepth.Max();
      float width = 0.75f * plotAreaHeight / treeInfo.MaxDepth;

      using (var pen = new Pen(oddNodeColor, width))
      using (var font = new Font("Arial", tickFontSize))
      using (var titleFormat = new StringFormat())
      using (var titleFont = new Font("Arial", histogramTitleFontSize))
      using (var brush = new SolidBrush(fontColor))
      {
        titleFormat.FormatFlags = StringFormatFlags.DirectionVertical;
        SizeF titleSize = gfx.MeasureString(title, titleFont);
        SizeF maxLabelSize = gfx.MeasureString(treeInfo.NodesPerDepth.Max().ToString(), font);
        float scale = (rightHistogramWidth - maxLabelSize.Width - horisontalSpacing - titleSize.Height) / maxNodesPerDepth;
        gfx.DrawString(title, titleFont, brush, canvasWidth - rightMargin - titleSize.Height, topMargin + plotAreaHeight / 2.0f - titleSize.Width / 2, titleFormat);
        for (int i = 0; i <= treeInfo.MaxDepth; i++)
        {
          label = treeInfo.NodesPerDepth[i].ToString();
          (x, y) = CanvasXY(treeInfo.MaxX, i);
          x += horisontalSpacing;
          gfx.DrawString(label, font, brush, x, y + 1 - maxLabelSize.Height / 2);
          gfx.DrawLine(pen, x + maxLabelSize.Width, y, x + maxLabelSize.Width + treeInfo.NodesPerDepth[i] * scale, y);
        }
      }
    }

    /// <summary>
    /// Draw histogram visualizing visit distribution per move canditate.
    /// </summary>
    /// <param name="gfx"></param>
    internal void VisitDistributionHistogram(Graphics gfx)
    {
      float x0;
      float y0;
      float x1;
      float y1;
      string label;
      float fontHeight;
      float titleFontHeight;
      float scale;

      string title = "Visit distribution";

      int maxVisits = (from ind in Enumerable.Range(0, rawRoot.NumChildrenExpanded) select rawRoot.ChildAtIndexRef(ind).N).Max();

      int numBranches = rawRoot.NumChildrenExpanded;

      float barWidth = 0.75f * plotAreaWidth / numBranches;

      using (var pen = new Pen(oddNodeColor, barWidth))
      using (var font = new Font("Arial", tickFontSize))
      using (var titleFont = new Font("Arial", histogramTitleFontSize))
      using (var brush = new SolidBrush(fontColor))
      {
        fontHeight = (from ind in Enumerable.Range(0, rawRoot.NumChildrenExpanded) select gfx.MeasureString(rawRoot.ChildAtIndexRef(ind).PriorMove.ToString() + "\r\n" + "100%", font).Height).Max();
        titleFontHeight = gfx.MeasureString(title, titleFont).Height;
        SizeF titleSize = gfx.MeasureString(title, titleFont);
        gfx.DrawString(title, titleFont, brush, leftMargin + horisontalSpacing + plotAreaWidth / 2 - titleSize.Width / 2, canvasHeight - bottomMargin - titleSize.Height);

        scale = (bottomHistogramHeight - verticalSpacing - fontHeight - titleFontHeight) / (float)maxVisits;

        int i = 0;
        foreach (MCTSNodeStruct child in (from ind in Enumerable.Range(0, rawRoot.NumChildrenExpanded) select rawRoot.ChildAtIndexRef(ind)).OrderBy(c => -c.N))
        {
          x0 = (leftMargin + horisontalSpacing + barWidth / 2) + i * ((plotAreaWidth - barWidth) / (float)(numBranches - 1));
          y0 = canvasHeight - bottomMargin - bottomHistogramHeight + verticalSpacing;
          y1 = y0 + fontHeight + scale * child.N;

          label = child.PriorMove.ToString() + "\r\n" + Math.Round(100.0f * child.N / rawRoot.N, 1).ToString(CultureInfo.GetCultureInfo("en-GB")) + "%";

          gfx.DrawString(label, font, brush, x0 - gfx.MeasureString(label, font).Width / 2, y0);
          gfx.DrawLine(pen, x0, y0 + fontHeight, x0, y1);
          i++;
        }
      }
    }

    /// <summary>
    /// Convert node coordinates into canvas coordinates.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    internal (float, float) CanvasXY(float x, float y)
    {
      float canvasX = CanvasX(x);
      float canvasY = CanvasY(y);
      return (canvasX, canvasY);
    }

    internal float CanvasX(float x)
    {
      // Without 0.0f check we hit Overflow error, it is a mystery how 0.0f * scaleX can overflow.
      float canvasX = leftMargin + horisontalSpacing + (x != 0.0f ? x * scaleX : 0.0f);
      return canvasX;
    }

    internal float CanvasY(float y)
    {
      float canvasY = topMargin + (y != 0.0f ? y * scaleY : 0.0f);
      return canvasY;
    }

    /// <summary>
    /// Draw edges and nodes of the search tree.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="gfx"></param>
    /// <param name="penEdge"></param>
    /// <param name="brushRoot"></param>
    /// <param name="brushEven"></param>
    /// <param name="brushOdd"></param>
    internal void DrawTree(DrawTreeNode node, Graphics gfx, Pen penEdge, SolidBrush brushRoot, SolidBrush brushEven, SolidBrush brushOdd)
    {
      float x;
      float y;
      SolidBrush brush = node.BranchIndex == -1 ? brushRoot : node.BranchIndex % 2 == 0 ? brushEven : brushOdd;
      (x, y) = CanvasXY(node.X, node.Y);
      if (!(node.Parent is null))
      {
        float xParent;
        float yParent;
        (xParent, yParent) = CanvasXY(node.Parent.X, node.Parent.Y);
        gfx.DrawLine(penEdge, x, y, xParent, yParent);
      }
      foreach (DrawTreeNode child in node.Children)
      {
        DrawTree(child, gfx, penEdge, brushRoot, brushEven, brushOdd);
      }
      DrawNode(x, y, gfx, brush);
    }

    internal void DrawNode(float x, float y, Graphics gfx, SolidBrush brush)
    {
      gfx.FillEllipse(brush, x - pointRadius, y - pointRadius, pointRadius * 2, pointRadius * 2);
    }

    /// <summary>
    /// Dispose method which disposes the draw objects.
    /// </summary>
    public void Dispose()
    {
      image.Dispose();
    }
  }
}