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

using Ceres.Base.Misc;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.RPO;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// Optional diagnostic dump for the PolicyImputedRPO FPU calculation path
/// (PUCTSelector.ApplyRPOImputedFPU).  Gated by DEBUG_DUMP_FPU_CALCS below; when the
/// gate is false the helper is unreachable and the JIT removes it (and its call-site
/// guard) entirely.
///
/// Output format:
///   - One header line (default color, or red when significant) summarizing inputs.
///   - One row per per-child input vector (default color).
///   - One row per per-child output vector (yellow), so the result stands out.
///
/// Rows render as "Label                v0 v1 v2 | v3 v4 ..." where the '|'
/// separator marks the boundary between expanded (visited) children and
/// unexpanded (FPU-imputed) children.  Shared row/value formatting lives in
/// SearchDumpFormatting (also used by the TPS backup dump).
/// </summary>
internal static class FPUDumpDiagnostics
{
  /// <summary>
  /// Gate for PUCTSelector.ApplyRPOImputedFPU dumps.
  /// </summary>
  public const bool DEBUG_DUMP_FPU_CALCS = false;


  /// <summary>
  /// Threshold (absolute Q) for flagging an FPU dump as a significant
  /// divergence from the prior (non-RPO) FPU algorithm.  Header turns red when
  /// any unvisited child's imputed Q differs from the scalar default FPU by
  /// more than this amount.
  /// </summary>
  public const double FPU_DIFF_THRESHOLD = 0.1;


  /// <summary>
  /// Dump for PUCTSelector.ApplyRPOImputedFPU.
  ///
  /// Includes a "PUCT_FPU" comparison row showing what the prior non-RPO FPU
  /// algorithm would have produced for each child: a single scalar defaultFPU
  /// for every unvisited child, and the actual -W/N for visited children
  /// (which both algorithms agree on).  The header is red-flagged when any
  /// unvisited child's RPO-imputed Q differs from defaultFPU by more than
  /// FPU_DIFF_THRESHOLD.
  /// </summary>
  public static void DumpFPURPO(GNode node,
                                ReadOnlySpan<double> pSpan, ReadOnlySpan<double> nSpan, ReadOnlySpan<double> wSpan,
                                ReadOnlySpan<double> qOut,
                                int numToProcess, int numExpanded,
                                double lambda, RPORegularization regularization, RPOAnchor anchor,
                                double defaultFPU)
  {
    if (!DEBUG_DUMP_FPU_CALCS)
    {
      return;
    }

    string anchorStr = anchor.Mode == RPOAnchorMode.None
      ? "None"
      : (anchor.Mode == RPOAnchorMode.MatchChild
          ? $"MatchChild(idx={anchor.Index},val={anchor.Value:F3})"
          : $"MatchValue(val={anchor.Value:F3})");

    // Copy spans into local arrays so the row-formatter lambdas (which cannot
    // capture refs/spans) can read them.
    double[] p = new double[numToProcess];
    double[] q = new double[numToProcess];
    double[] qO = new double[numToProcess];
    double[] qPrior = new double[numToProcess];
    double maxFPUDelta = 0.0;
    for (int i = 0; i < numToProcess; i++)
    {
      p[i] = pSpan[i];
      bool visited = i < numExpanded && nSpan[i] > 0;
      q[i] = visited ? -wSpan[i] / nSpan[i] : double.NaN;
      qO[i] = qOut[i];
      // Prior (non-RPO) algorithm: visited children get their actual q; every
      // unvisited child gets the same scalar defaultFPU.
      qPrior[i] = visited ? q[i] : defaultFPU;
      if (!visited)
      {
        double delta = Math.Abs(qO[i] - defaultFPU);
        if (delta > maxFPUDelta)
        {
          maxFPUDelta = delta;
        }
      }
    }
    double nodeQ = node.Q;
    bool significant = maxFPUDelta > FPU_DIFF_THRESHOLD;
    if (SearchDumpFormatting.ONLY_SHOW_SIGNIFICANT && !significant)
    {
      return;
    }

    lock (SearchDumpFormatting.ConsoleLock)
    {
      Console.WriteLine();
      SearchDumpFormatting.WriteHeaderLine(
        $"[FPU/RPO] NumEdgesExpanded={numExpanded}/{numToProcess} Q={nodeQ:F3} " +
        $"tau={lambda:F3} reg={regularization} anchor={anchorStr} " +
        $"defaultFPU={defaultFPU:F3} maxFPUDelta={maxFPUDelta:F3}", significant);
      Console.WriteLine(SearchDumpFormatting.FormatRow("Q_in (parent persp):", numToProcess, numExpanded, i => SearchDumpFormatting.FmtQ(q[i])));
      Console.WriteLine(SearchDumpFormatting.FormatRow("P:", numToProcess, numExpanded, i => SearchDumpFormatting.FmtP(p[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        SearchDumpFormatting.FormatRow("Q_out (imputed):", numToProcess, numExpanded, i => SearchDumpFormatting.FmtQ(qO[i])));
      Console.WriteLine(SearchDumpFormatting.FormatRow("PUCT_FPU (prior):", numToProcess, numExpanded, i => SearchDumpFormatting.FmtQ(qPrior[i])));
    }
  }
}
