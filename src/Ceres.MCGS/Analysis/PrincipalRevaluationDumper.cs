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

using System.Linq;

using Spectre.Console;

using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.MCGS.Analysis;

/// <summary>
/// Static helper for dumping a PrincipalRevaluationResult to the console
/// (headline root estimates followed by per-root-move estimates).
/// </summary>
public static class PrincipalRevaluationDumper
{
  /// <summary>
  /// Dumps the result of a revaluation pass as Spectre.Console tables.
  /// </summary>
  /// <param name="r">The revaluation result.</param>
  /// <param name="chosenMove">The move chosen by the search (marked in the per-move table).</param>
  public static void DumpToConsole(PrincipalRevaluationResult r, MGMove chosenMove)
  {
    AnsiConsole.Write(new Rule("[yellow]Root Revaluation Analysis[/]"));
    AnsiConsole.WriteLine();

    static string PM(double q, double sigma) => $"{q:F3} ±{sigma:F3}";

    Table summary = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
    summary.AddColumn("Property");
    summary.AddColumn("Value");
    summary.AddRow("Root Q (pre-rollout)", r.RootQOriginal.ToString("F3"));
    summary.AddRow("Root Q (post-rollout)", r.RootQPostRollout.ToString("F3"));
    summary.AddRow("Reval Q (average)", PM(r.RootQAvg, r.RootSigmaAvg));
    summary.AddRow("Reval Q (negamax)", PM(r.RootQNegamax, r.RootSigmaNegamax));
    summary.AddRow($"Reval Q (soft p={PrincipalRevaluation.SOFTMAX_P:0.#})", PM(r.RootQSoft, r.RootSigmaSoft));
    summary.AddRow("Reval Q (frontier-extrapolated)", double.IsNaN(r.RootQFrontierExtrapolated)
                     ? "" : PM(r.RootQFrontierExtrapolated, r.RootQFrontierExtrapolatedSigma));
    summary.AddRow("Root DRQ (raw rollout)", double.IsNaN(r.RootDRQ)
                     ? "" : PM(r.RootDRQ, r.RootDRQSigma));
    summary.AddRow("Avg-op baseline (diagnostic)", r.RootQAvgBaseline.ToString("F3"));
    summary.AddRow("First-order ΔQ", r.FirstOrderDeltaQ.ToString("+0.000;-0.000;0.000"));
    summary.AddRow("Coverage", r.Coverage.ToString("F2"));
    summary.AddRow("Root class", r.RootClass?.ToString() ?? "MIXED");
    summary.AddRow("Frontier / exact leaves", $"{r.NumFrontier} / {r.NumExactLeaves}");
    summary.AddRow("Rollout visits", r.NumRolloutVisits.ToString("N0"));
    summary.AddRow("Distinct rollout lines", r.TotalDistinctPaths.ToString("N0"));
    summary.AddRow("Rollout depth avg / max", double.IsNaN(r.AvgRolloutDepth)
                                                ? "" : $"{r.AvgRolloutDepth:F1} / {r.MaxRolloutDepth}");
    summary.AddRow("Elapsed sec", r.ElapsedSecs.ToString("F2"));
    if (r.Aborted)
    {
      summary.AddRow("[red]Aborted[/]", "true (deadline/stop reached; results partial)");
    }
    AnsiConsole.Write(summary);

    if (r.Assessment != RootQAssessment.Unknown)
    {
      string color = r.Assessment switch
      {
        RootQAssessment.Suspect => "red",
        RootQAssessment.Drifting => "yellow",
        _ => "green"
      };
      AnsiConsole.MarkupLine($"[bold {color}]Top-level Q assessment: {r.Assessment.ToString().ToUpperInvariant()}[/]"
                           + $" — raw rollout evidence {r.RootDRQ:F3} vs search {r.RootQOriginal:F3}"
                           + $" (Δ {r.RootQRawDelta:+0.000;-0.000}, {r.RootQRawZ:F1}σ);"
                           + $" frontier-extrapolated {r.RootQFrontierExtrapolated:F3}");
    }

    if (r.ClassMass.Count > 0)
    {
      string massStr = string.Join("  ", r.ClassMass.OrderByDescending(kv => kv.Value)
                                                    .Select(kv => $"{kv.Key}={kv.Value:F2}"));
      AnsiConsole.MarkupLine($"[blue]Class mass:[/] {massStr}");
    }

    if (r.StageSummaries.Count > 0)
    {
      string stagesStr = string.Join("  |  ", r.StageSummaries.Select(s =>
        $"ε={s.Epsilon}: {s.Visits} visits over {s.Nodes} nodes"
        + (s.StoppedDry > 0 ? $", {s.StoppedDry} dried up" : "")
        + (s.StoppedTerminal > 0 ? $", {s.StoppedTerminal} hit terminal" : "")));
      AnsiConsole.MarkupLine($"[blue]Stages:[/] {stagesStr}");
      AnsiConsole.MarkupLine("[grey](requested rollouts are an upper bound: a node stops early once consecutive"
                           + " rounds retrace known lines (dry-up) or, at ε=0, once a terminal is reached)[/]");
    }
    AnsiConsole.WriteLine();

    Table moves = new Table().Border(TableBorder.Rounded).BorderColor(Color.Blue);
    moves.AddColumn(new TableColumn("Move").LeftAligned());
    moves.AddColumn(new TableColumn("N").RightAligned());
    moves.AddColumn(new TableColumn("QOrig").RightAligned());
    moves.AddColumn(new TableColumn("QAvg").RightAligned());
    moves.AddColumn(new TableColumn("QNegamax").RightAligned());
    moves.AddColumn(new TableColumn("QSoft").RightAligned());
    moves.AddColumn(new TableColumn("QSoftCP").RightAligned());
    moves.AddColumn(new TableColumn("QXtr").RightAligned());
    moves.AddColumn(new TableColumn("DRQ").RightAligned());
    moves.AddColumn(new TableColumn("Cov").RightAligned());
    moves.AddColumn(new TableColumn("±σ").RightAligned());
    moves.AddColumn(new TableColumn("#Fr").RightAligned());
    moves.AddColumn(new TableColumn("#DR").RightAligned());
    moves.AddColumn(new TableColumn("DRDepth").RightAligned());
    moves.AddColumn(new TableColumn("DRMax").RightAligned());
    moves.AddColumn(new TableColumn("Vol").RightAligned());
    moves.AddColumn(new TableColumn("Draw").RightAligned());

    // Show only moves reasonably close to the best (by pre-rollout Q); the long tail of
    // clearly inferior moves carries no analysis value and clutters the table.
    // (Threshold lives with the other tunables in PrincipalRevaluation.)
    double bestQOrig = r.RootMoves.Count > 0 ? r.RootMoves.Max(rm => rm.QOrig) : 0;
    int numHidden = 0;

    foreach (RootMoveReval rm in r.RootMoves.OrderByDescending(rm => rm.EdgeN))
    {
      bool isChosen = !chosenMove.IsNull
                   && rm.Move.FromSquareIndex == chosenMove.FromSquareIndex
                   && rm.Move.ToSquareIndex == chosenMove.ToSquareIndex;

      if (!isChosen && rm.QOrig < bestQOrig - PrincipalRevaluation.DUMP_MAX_Q_GAP_FROM_BEST)
      {
        numHidden++;
        continue;
      }
      string moveStr = rm.Move.MoveStr(MGMoveNotationStyle.LongAlgebraic);
      if (isChosen)
      {
        moveStr = $"[bold red]{moveStr}*[/]";
      }

      moves.AddRow(moveStr,
                   rm.EdgeN.ToString("N0"),
                   PrincipalPosSetDumper.FormatQValue(rm.QOrig),
                   rm.InRegion ? PrincipalPosSetDumper.FormatQValue(rm.QAvg) : "",
                   rm.InRegion ? PrincipalPosSetDumper.FormatQValue(rm.QNegamax) : "",
                   PrincipalPosSetDumper.FormatQValue(rm.QSoft),
                   EncodedEvalLogistic.LogisticToCentipawn((float)rm.QSoft).ToString("F0"),
                   double.IsNaN(rm.QExtrapolated) ? "" : PrincipalPosSetDumper.FormatQValue(rm.QExtrapolated),
                   double.IsNaN(rm.DRQ) ? "" : PrincipalPosSetDumper.FormatQValue(rm.DRQ),
                   rm.FrontierCoverage > 0 ? rm.FrontierCoverage.ToString("F2") : "",
                   rm.SigmaSoft.ToString("F3"),
                   rm.NumFrontier > 0 ? rm.NumFrontier.ToString() : "",
                   rm.DistinctPaths > 0 ? rm.DistinctPaths.ToString() : "",
                   double.IsNaN(rm.AvgRolloutDepth) ? "" : rm.AvgRolloutDepth.ToString("0.#"),
                   rm.MaxRolloutDepth > 0 ? rm.MaxRolloutDepth.ToString() : "",
                   rm.VolatileMass > 0 ? rm.VolatileMass.ToString("F2") : "",
                   rm.DrawMass > 0 ? rm.DrawMass.ToString("F2") : "");
    }

    AnsiConsole.Write(moves);
    if (numHidden > 0)
    {
      AnsiConsole.MarkupLine($"[grey]({numHidden} move(s) with Q more than {PrincipalRevaluation.DUMP_MAX_Q_GAP_FROM_BEST:F2} below best omitted)[/]");
    }
    AnsiConsole.WriteLine();

    AnsiConsole.MarkupLine("[yellow]QOrig[/]    - move Q before rollouts (root player's perspective; * = chosen move)");
    AnsiConsole.MarkupLine("[yellow]QAvg[/]     - revalued Q, visit-weighted average backup (engine-consistent)");
    AnsiConsole.MarkupLine("[yellow]QNegamax[/] - revalued Q, negamax over visit-supported children");
    AnsiConsole.MarkupLine("[yellow]QSoft[/]    - revalued Q, soft-minimax power mean (headline estimate)");
    AnsiConsole.MarkupLine("[yellow]QSoftCP[/]  - QSoft converted to centipawns (standard Ceres/Lc0 convention)");
    AnsiConsole.MarkupLine("[yellow]QXtr[/]     - coverage-extrapolated move Q: frontier correction projected over the move's uncovered mass (aggressive estimate)");
    AnsiConsole.MarkupLine("[yellow]DRQ[/]      - pure rollout evidence: rollout-weighted mean leaf Q over this move's frontier (root perspective, no anchor blend)");
    AnsiConsole.MarkupLine("[yellow]Cov[/]      - fraction of this move's value mass explained by the frontier cut");
    AnsiConsole.MarkupLine("[yellow]#Fr/#DR[/]  - frontier positions / distinct rollout lines in this move's subtree (transposed positions count toward every move reaching them)");
    AnsiConsole.MarkupLine("[yellow]DRDepth[/]  - distinct-line-weighted average of per-position median rollout depths; [yellow]DRMax[/] - deepest descent");
    AnsiConsole.MarkupLine("[yellow]Vol/Draw[/] - influence mass of move's frontier classified Volatile / DeepDraw");
    AnsiConsole.WriteLine();
  }
}
