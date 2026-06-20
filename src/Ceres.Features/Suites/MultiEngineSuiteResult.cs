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

using Ceres.Chess;

#endregion

namespace Ceres.Features.Suites
{
  /// <summary>
  /// Kind of engine that participated in a multiengine suite test, which governs how many
  /// statistics could be extracted (the in-process Ceres engines expose the full set; an
  /// external engine exposes only a subset).
  /// </summary>
  public enum MultiEngineKind
  {
    /// <summary>Standard in-process Ceres MCTS engine (GameEngineCeresInProcess).</summary>
    CeresMCTS,

    /// <summary>Standard in-process Ceres MCGS engine (GameEngineCeresMCGSInProcess).</summary>
    CeresMCGS,

    /// <summary>External engine (UCI/LC0), for which only a subset of statistics is available.</summary>
    External
  }


  /// <summary>
  /// Comprehensive aggregated statistics for a single engine across a multiengine suite test.
  ///
  /// Statistics that an engine could not provide (e.g. policy divergence or correct-move-visit
  /// fraction for an external engine, or a difference-vs-baseline statistic for the baseline
  /// engine itself) are represented as float.NaN.
  /// </summary>
  public sealed class MultiEngineEngineResult
  {
    /// <summary>Short display id (matches the column header in the live block).</summary>
    public string ID;

    /// <summary>If this engine is the baseline (the reference for the difference statistics).</summary>
    public bool IsBaseline;

    /// <summary>Kind of engine (governs the breadth of available statistics).</summary>
    public MultiEngineKind Kind;

    /// <summary>Search limit used by this engine.</summary>
    public SearchLimit SearchLimit;

    /// <summary>Number of positions over which this engine was evaluated.</summary>
    public int NumPositions;

    // --- "True value" statistics (always available, higher is better) ---

    /// <summary>Average solve score as a percentage (fraction of positions solved * 100).</summary>
    public float AvgSolveScorePct;

    /// <summary>Average engine-reported evaluations per second.</summary>
    public float AvgEPS;

    /// <summary>Average search time per position (seconds).</summary>
    public float AvgTimeSecs;

    /// <summary>Average root Q value.</summary>
    public float AvgQ;

    // --- Difference-vs-baseline statistics (NaN for the baseline engine / when unavailable) ---

    /// <summary>Mean absolute difference of this engine's per-position root Q vs the baseline's.</summary>
    public float AvgAbsQDiffVsBaseline = float.NaN;

    /// <summary>
    /// Mean symmetric KL divergence between this engine's and the baseline's empirical root-move
    /// visit distributions (only when both engines exposed per-move visit statistics).
    /// </summary>
    public float AvgPolicyKLDVsBaseline = float.NaN;

    /// <summary>
    /// Z-score of the paired per-position graded solution-quality difference (this engine minus the
    /// baseline, in 0-10 graded points). Positive means this engine scores better than the baseline;
    /// |z| >~ 2 is significant. NaN for the baseline engine itself (and when fewer than two positions
    /// were compared). This is the per-column analogue of the headline two-engine z-score.
    /// </summary>
    public float GradedScoreDiffZVsBaseline = float.NaN;

    // --- Additional "true value" statistic available only for engines exposing root visit stats ---

    /// <summary>Average percentage of total root visits placed on the correct move(s) (NaN if unavailable).</summary>
    public float AvgCorrectMoveVisitPct = float.NaN;

    // --- Totals / additional aggregates (where available) ---

    /// <summary>Total nodes searched across all positions.</summary>
    public long TotalNodes;

    /// <summary>Total search time (seconds) across all positions.</summary>
    public float TotalTimeSecs;

    /// <summary>Total neural network position evaluations (0 for external engines).</summary>
    public long TotalNNEvals;

    /// <summary>Total tablebase hits (0 for external engines).</summary>
    public long TotalTablebaseHits;

    /// <summary>Average search tree average-depth (NaN if unavailable).</summary>
    public float AvgDepth = float.NaN;

    /// <summary>Average visit-distribution entropy (NaN if unavailable, e.g. MCTS / external).</summary>
    public float AvgVisitEntropy = float.NaN;

    /// <summary>
    /// Fraction of search-loop wall-clock during which the device backend ("in C++ interop") was
    /// busy, aggregated across all positions where the metric is supported. Approaches 1.0 as
    /// searches become GPU-bound. NaN if unsupported (NNEvaluatorTensorRT / NNEvaluatorCUDA report it).
    /// </summary>
    public float BackendBusyFraction = float.NaN;

    public override string ToString()
      => $"<MultiEngineEngineResult {ID}{(IsBaseline ? "*" : "")} {Kind} solve={AvgSolveScorePct:F1}% EPS={AvgEPS:N0} Q={AvgQ:F3}>";
  }


  /// <summary>
  /// Comprehensive result of a multiengine suite test, holding aggregated statistics for every
  /// participating engine (see MultiEngineEngineResult), together with suite identity metadata.
  /// </summary>
  public sealed class MultiEngineSuiteResult
  {
    /// <summary>The suite test definition that produced this result.</summary>
    public readonly SuiteTestDef Def;

    /// <summary>Per-engine aggregated statistics, in the order the engines were specified.</summary>
    public MultiEngineEngineResult[] Engines;

    /// <summary>Index (into Engines) of the baseline engine.</summary>
    public int BaselineIndex;

    /// <summary>The suite id.</summary>
    public string ID;

    /// <summary>The EPD file name used as the source of test positions.</summary>
    public string EPDFileName;

    /// <summary>Number of positions actually run (after filtering/slicing).</summary>
    public int NumPositionsTested;

    /// <summary>Machine on which the suite was run.</summary>
    public string MachineName;

    /// <summary>Date/time the suite result was produced.</summary>
    public DateTime RunDateTime;

    public MultiEngineSuiteResult(SuiteTestDef def)
    {
      Def = def;
    }

    /// <summary>The baseline engine's aggregated result.</summary>
    public MultiEngineEngineResult Baseline
      => (Engines != null && BaselineIndex >= 0 && BaselineIndex < Engines.Length) ? Engines[BaselineIndex] : null;
  }
}
