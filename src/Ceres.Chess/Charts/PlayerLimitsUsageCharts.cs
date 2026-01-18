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

#endregion

namespace Ceres.Chess.Charts
{
  /// <summary>
  /// Static helper methods that generate Plotly Chart objects
  /// depicting limit usage for both players in a game.
  /// </summary>
  public static class PlayerLimitsUsageCharts
  {
#if DISABLED_NO_PLOTLY
    /// <summary>
    /// Returns a chart with lines for cumulative usage by both players
    /// (and optionally the allotted time as of each move according to the limits control).
    /// </summary>
    public static PlotlyChart LineChart(string title, string labelPlayer1, string labelPlayer2,
                                        float[] perMoveUsagePlayer1, float[] perMoveUsagePlayer2,
                                        float? baseAllotment, float? incrementalAllotment)
    {
      // Create ply numbers
      int[] elements = Enumerable.Range(0, perMoveUsagePlayer1.Length).ToArray();

      int maxNumPly = Math.Max(perMoveUsagePlayer1.Length, perMoveUsagePlayer2.Length);
      float[] perMoveUsagePlayer1Cumulative = new float[maxNumPly];
      float[] perMoveUsagePlayer2Cumulative = new float[maxNumPly];

      // Compute cumulative values.
      float acc1 = 0;
      float acc2 = 0;
      for (int i = 0; i < maxNumPly; i++)
      {
        acc1 += i < perMoveUsagePlayer1.Length ? perMoveUsagePlayer1[i] : 0;
        acc2 += i < perMoveUsagePlayer2.Length ? perMoveUsagePlayer2[i] : 0;
        perMoveUsagePlayer1Cumulative[i] = acc1;
        perMoveUsagePlayer2Cumulative[i] = acc2;
      }

      // Create the allotted time series, and compute cumulatives
      float[] timeAllot = null;
      if (baseAllotment.HasValue)
      {
        bool haveSeenNonzero = false;

        timeAllot = new float[perMoveUsagePlayer1.Length];
        timeAllot[0] = baseAllotment.Value;
        for (int i = 0; i < perMoveUsagePlayer1.Length; i++)
        {
          if (i == 0)
          {
            timeAllot[i] = baseAllotment.Value;
          }
          else
          {
            float increment = haveSeenNonzero ? (incrementalAllotment.HasValue ? incrementalAllotment.Value : 0) : 0; // book move
            timeAllot[i] = timeAllot[i - 1] + increment;
          }
          haveSeenNonzero |= perMoveUsagePlayer1[i] != 0;
        }
      }

      Scatter TimeAllotLine = new Scatter()
      {

        x = elements,
        y = timeAllot,
        name = "Allotment "
      };

      Scatter Player1Line = new Scatter()
      {
        x = elements,
        y = perMoveUsagePlayer1Cumulative,
        name = "White " + labelPlayer1
      };

      Scatter Player2Line = new Scatter()
      {
        x = elements,
        y = perMoveUsagePlayer2Cumulative,
        name = "Black " + labelPlayer2
      };

      List<Scatter> plots = new();
      plots.Add(Player1Line);
      plots.Add(Player2Line);
      if (timeAllot != null)
      {
        plots.Add(TimeAllotLine);
      }

      PlotlyChart chart = Chart.Plot(plots);
      Layout.Layout layout = new Layout.Layout() { barmode = "group", title = $"Cumulative Usage by Player, {title}" };
      chart.WithLayout(layout);
      chart.WithXTitle("Ply");
      chart.WithYTitle("Cumulative Used");
      chart.WithLegend(true);
      chart.Width = 700;
      chart.Height = 400;

      return chart;
    }

    public static PlotData LineChartData(string title, string labelPlayer1, string labelPlayer2,
                                        double[] perMoveUsagePlayer1, double[] perMoveUsagePlayer2,
                                        float? baseAllotment, float? incrementalAllotment)
    {
      // Create ply numbers
      int[] elements = Enumerable.Range(0, perMoveUsagePlayer1.Length).ToArray();

      int maxNumPly = Math.Max(perMoveUsagePlayer1.Length, perMoveUsagePlayer2.Length);
      double[] perMoveUsagePlayer1Cumulative = new double[maxNumPly];
      double[] perMoveUsagePlayer2Cumulative = new double[maxNumPly];

      // Compute cumulative values.
      double acc1 = 0;
      double acc2 = 0;
      for (int i = 0; i < maxNumPly; i++)
      {
        acc1 += i < perMoveUsagePlayer1.Length ? perMoveUsagePlayer1[i] : 0;
        acc2 += i < perMoveUsagePlayer2.Length ? perMoveUsagePlayer2[i] : 0;
        perMoveUsagePlayer1Cumulative[i] = acc1;
        perMoveUsagePlayer2Cumulative[i] = acc2;
      }

      // Create the allotted time series, and compute cumulatives
      double[] timeAllot = null;
      if (baseAllotment.HasValue)
      {
        bool haveSeenNonzero = false;

        timeAllot = new double[perMoveUsagePlayer1.Length];
        timeAllot[0] = baseAllotment.Value;
        for (int i = 0; i < perMoveUsagePlayer1.Length; i++)
        {
          if (i == 0)
          {
            timeAllot[i] = baseAllotment.Value;
          }
          else
          {
            float increment = haveSeenNonzero ? (incrementalAllotment.HasValue ? incrementalAllotment.Value : 0) : 0; // book move
            timeAllot[i] = timeAllot[i - 1] + increment;
          }
          haveSeenNonzero |= perMoveUsagePlayer1[i] != 0;
        }
      }

      Scatter TimeAllotLine = new Scatter()
      {

        x = elements,
        y = timeAllot,
        name = "Allotment "
      };

      Scatter Player1Line = new Scatter()
      {
        x = elements,
        y = perMoveUsagePlayer1Cumulative,
        name = "White " + labelPlayer1
      };

      Scatter Player2Line = new Scatter()
      {
        x = elements,
        y = perMoveUsagePlayer2Cumulative,
        name = "Black " + labelPlayer2
      };

      List<Scatter> plots = new();
      plots.Add(Player1Line);
      plots.Add(Player2Line);
      if (timeAllot != null)
      {
        plots.Add(TimeAllotLine);
      }

      PlotlyChart chart = Chart.Plot(plots);
      Layout.Layout layout = new Layout.Layout() { barmode = "group", title = $"Cumulative Usage by Player, {title}" };
      chart.WithLayout(layout);
      chart.WithXTitle("Ply");
      chart.WithYTitle("Cumulative Used");
      chart.WithLegend(true);
      chart.Width = 700;
      chart.Height = 400;

      var plot = new PlotData()
      {
        Title = title,
        TimeAllotData = timeAllot,
        PlayerBlack = "Black " + labelPlayer2,
        PlayerWhite = "White " + labelPlayer1,
        PlayerBlackData = perMoveUsagePlayer2Cumulative,
        PlayerWhiteData = perMoveUsagePlayer1Cumulative,
        MoveElements = elements,
      };

      return plot;
    }

    /// <summary>
    /// Returns a chart with bars for per-move usage by both players.
    /// </summary>
    public static PlotlyChart BarChart(string title, string labelPlayer1, string labelPlayer2, float[] perMoveUsagePlayer1, float[] perMoveUsagePlayer2)
    {
      // Create ply numbers
      int[] elements = Enumerable.Range(0, perMoveUsagePlayer1.Length).ToArray();

      Bar Player1Line = new Bar
      {
        x = elements,
        y = perMoveUsagePlayer1,
        name = "White " + labelPlayer1
      };

      Bar Player2Line = new Bar()
      {
        x = elements,
        y = perMoveUsagePlayer2,
        name = "Black " + labelPlayer2
      };

      PlotlyChart chart = Chart.Plot(new[] { Player1Line, Player2Line });
      Layout.Layout layout = new Layout.Layout() { barmode = "group", title = $"Per Move Usage by Player, {title}" };
      chart.WithLayout(layout);
      chart.WithXTitle("Move");
      chart.WithYTitle("Used");
      chart.WithLegend(true);
      chart.Width = 700;
      chart.Height = 400;

      return chart;
    }

    public static PlotData BarChartData(string title, string labelPlayer1, string labelPlayer2, double[] perMoveUsagePlayer1, double[] perMoveUsagePlayer2)
    {
      // Create ply numbers
      int[] elements = Enumerable.Range(0, perMoveUsagePlayer1.Length).ToArray();

      Bar Player1Line = new Bar
      {
        x = elements,
        y = perMoveUsagePlayer1,
        name = "White " + labelPlayer1
      };

      Bar Player2Line = new Bar()
      {
        x = elements,
        y = perMoveUsagePlayer2,
        name = "Black " + labelPlayer2
      };

      PlotlyChart chart = Chart.Plot(new[] { Player1Line, Player2Line });
      Layout.Layout layout = new Layout.Layout() { barmode = "group", title = $"Per Move Usage by Player, {title}" };
      chart.WithLayout(layout);
      chart.WithXTitle("Move");
      chart.WithYTitle("Used");
      chart.WithLegend(true);
      chart.Width = 700;
      chart.Height = 400;

      var plot = new PlotData()
      {
        Title = title,
        MoveElements = elements,
        PlayerBlack = "Black " + labelPlayer2,
        PlayerWhite = "White " + labelPlayer1,
        PlayerBlackData = perMoveUsagePlayer2,
        PlayerWhiteData = perMoveUsagePlayer1,
      };

      return plot;
    }
#endif
  }


  public class PlotData
  {
    public string Title { get; set; }
    public string PlayerWhite { get; set; }
    public string PlayerBlack { get; set; }
    public double[] PlayerWhiteData { get; set; }
    public double[] PlayerBlackData { get; set; }
    public double[] TimeAllotData { get; set; }
    public int[] MoveElements { get; set; }
  }
}
