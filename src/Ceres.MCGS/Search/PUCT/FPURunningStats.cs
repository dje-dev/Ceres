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
using System.Threading;

using Ceres.Base.Math;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// Simple running mean accumulator with both a sliding "window" mean
/// (reset on each dump) and a lifetime mean (never reset).
/// </summary>
internal sealed class RunningCorrelation
{
  public int WindowCount;
  public double WindowSum;

  public long TotalCount;
  public double TotalSum;

  public void Add(double value)
  {
    if (double.IsNaN(value))
    {
      return;
    }
    WindowSum += value;
    WindowCount++;
    TotalSum += value;
    TotalCount++;
  }

  public void Snapshot(out int windowN, out double windowAvg,
                       out long totalN, out double totalAvg)
  {
    windowN = WindowCount;
    windowAvg = WindowCount > 0 ? WindowSum / WindowCount : double.NaN;
    totalN = TotalCount;
    totalAvg = TotalCount > 0 ? TotalSum / TotalCount : double.NaN;
  }

  public void ResetWindow()
  {
    WindowCount = 0;
    WindowSum = 0;
  }
}


/// <summary>
/// Per-ParamsSearch.ID running statistics for the FPU correlation diagnostic.
/// Tracks two parallel series:
///   - "All":    correlation over all considered children.
///   - "GT1Pct": correlation restricted to children with policy P >= 0.01.
/// The "All" series drives the dump cadence (every DUMP_INTERVAL samples).
/// </summary>
internal sealed class FPURunningStatsDetail
{
  public const int DUMP_INTERVAL = 3000;

  /// <summary>Policy threshold used to define the GT1Pct subset.</summary>
  public const double GT1PCT_THRESHOLD = 0.01;

  public readonly string ID;
  public readonly RunningCorrelation All = new();
  public readonly RunningCorrelation GT1Pct = new();
  public readonly RunningCorrelation AllMAD = new();
  public readonly RunningCorrelation GT1PctMAD = new();
  public readonly RunningCorrelation AllBias = new();
  public readonly RunningCorrelation GT1PctBias = new();

  public FPURunningStatsDetail(string id)
  {
    ID = id;
  }
}


/// <summary>
/// Tracks correlation between the FPU-imputed Q vector (qWhenNoChildrenComposite,
/// in parent perspective) and the true child V values (negated to parent
/// perspective) for each distinct ParamsSearch.ID.
///
/// Two series are tracked side-by-side:
///   - "All":    every child included in numToProcess.
///   - "GT1Pct": only children whose policy P >= 0.01.
///
/// Every <see cref="FPURunningStats.DUMP_INTERVAL"/> "All" samples for a
/// given ID, both series for that ID are dumped to the console in yellow.
/// </summary>
internal static class FPURunningStats
{
  internal const bool DEBUG_DUMP_FPU_CORRELATION_STATS = false;

  private static readonly Dictionary<string, FPURunningStatsDetail> statsById = new();
  private static readonly Lock statsLock = new();

  #region Shared NN child-V evaluator

  // Shared NN evaluator used to obtain V values for not-yet-expanded children
  // (used by MCGSStrategyPUCT for action-head diagnostics and by the FPU
  // correlation diagnostic below). A single dedicated instance is kept here
  // to avoid locking problems with the main search evaluator.
  private static NNEvaluator actualVEvaluator;
  private static readonly Lock actualVEvaluatorLock = new();
  private static NNEvaluatorDef evaluatorDef;

  /// <summary>
  /// Registers the NNEvaluatorDef used to lazily create the shared
  /// NN evaluator returned by <see cref="GetChildV"/>. Safe to call
  /// multiple times; only the first non-null definition is retained.
  /// </summary>
  public static void EnsureEvaluatorDef(NNEvaluatorDef def)
  {
    if (def != null && evaluatorDef == null)
    {
      evaluatorDef = def;
    }
  }

  /// <summary>
  /// Returns the V value (child's own perspective) for the i-th child of
  /// <paramref name="node"/>. If the child is already expanded the cached
  /// node V is returned; otherwise the position is evaluated on demand using
  /// the shared NN evaluator registered via <see cref="EnsureEvaluatorDef"/>.
  /// </summary>
  public static double GetChildV(GNode node, int childIndex)
  {
    if (childIndex < node.NumEdgesExpanded)
    {
      return node.ChildEdgeAtIndex(childIndex).ChildNode.V;
    }

    lock (actualVEvaluatorLock)
    {
      if (actualVEvaluator == null)
      {
        if (evaluatorDef == null)
        {
          throw new InvalidOperationException(
            "FPUCorrelationTracker.EnsureEvaluatorDef must be called before GetChildV " +
            "can evaluate an unexpanded child.");
        }
        // Dedicated NN evaluator to avoid locking problems with the main search.
        actualVEvaluator = evaluatorDef.ToEvaluator();
      }

      MGPosition pos = node.CalcPosition();
      MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(node.ChildEdgeHeaderAtIndex(childIndex).Move, pos);
      pos.MakeMove(mgMove);
      NNEvaluatorResult childEval = actualVEvaluator.Evaluate(new PositionWithHistory(pos.ToPosition));
      return childEval.V;
    }
  }

  #endregion


  /// <summary>
  /// Computes the parent-perspective correlation between qWhenNoChildrenComposite
  /// and the children's true V values (overall and on the P >= 1% subset),
  /// accumulates into the running statistics for paramsSearch.ID, and writes a
  /// yellow console line every <see cref="FPURunningStats.DUMP_INTERVAL"/>
  /// samples for that ID.
  /// </summary>
  public static void Record(GNode node,
                            ParamsSearch paramsSearch,
                            ParamsSelect paramsSelect,
                            double[] qWhenNoChildrenComposite,
                            ReadOnlySpan<double> policy,
                            int numToProcess)
  {
    if (qWhenNoChildrenComposite == null || numToProcess < 2)
    {
      return;
    }

    // Build parent-perspective arrays for the full set, then derive the
    // GT1Pct subset (children with P >= 1%) into separate buffers.
    Span<double> predictedAll = stackalloc double[numToProcess];
    Span<double> actualAll = stackalloc double[numToProcess];
    Span<double> predictedSubset = stackalloc double[numToProcess];
    Span<double> actualSubset = stackalloc double[numToProcess];

    int subsetCount = 0;
    double absDevSumAll = 0;
    double absDevSumSubset = 0;
    double devSumAll = 0;
    double devSumSubset = 0;
    for (int i = 0; i < numToProcess; i++)
    {
      double pred = qWhenNoChildrenComposite[i];
      // qWhenNoChildrenComposite is parent perspective; child V is child
      // perspective, so negate to express both from the parent's side.
      double actual = -GetChildV(node, i);

      predictedAll[i] = pred;
      actualAll[i] = actual;
      double dev = pred - actual;
      double absDev = Math.Abs(dev);
      absDevSumAll += absDev;
      devSumAll += dev;

      if (policy[i] >= FPURunningStatsDetail.GT1PCT_THRESHOLD)
      {
        predictedSubset[subsetCount] = pred;
        actualSubset[subsetCount] = actual;
        absDevSumSubset += absDev;
        devSumSubset += dev;
        subsetCount++;
      }
    }

    double corrAll = StatUtils.Correlation(predictedAll, actualAll);
    double corrSubset = subsetCount >= 2
      ? StatUtils.Correlation(predictedSubset[..subsetCount], actualSubset[..subsetCount])
      : double.NaN;
    double madAll = numToProcess > 0 ? absDevSumAll / numToProcess : double.NaN;
    double madSubset = subsetCount > 0 ? absDevSumSubset / subsetCount : double.NaN;
    double biasAll = numToProcess > 0 ? devSumAll / numToProcess : double.NaN;
    double biasSubset = subsetCount > 0 ? devSumSubset / subsetCount : double.NaN;

    if (double.IsNaN(corrAll))
    {
      // Without a valid "All" sample we have nothing meaningful to record;
      // skip to avoid skewing the GT1Pct dump cadence relative to "All".
      return;
    }

    string id = paramsSearch.ID ?? "";
    ParamsSelect.FPUType fpuMode = paramsSelect.GetFPUMode(node.IsSearchRoot);

    bool shouldDump;
    int allWinN = 0, gtWinN = 0, allMadWinN = 0, gtMadWinN = 0, allBiasWinN = 0, gtBiasWinN = 0;
    double allWinAvg = 0, gtWinAvg = 0, allMadWinAvg = 0, gtMadWinAvg = 0, allBiasWinAvg = 0, gtBiasWinAvg = 0;
    long allTotN = 0, gtTotN = 0, allMadTotN = 0, gtMadTotN = 0, allBiasTotN = 0, gtBiasTotN = 0;
    double allTotAvg = 0, gtTotAvg = 0, allMadTotAvg = 0, gtMadTotAvg = 0, allBiasTotAvg = 0, gtBiasTotAvg = 0;

    lock (statsLock)
    {
      if (!statsById.TryGetValue(id, out FPURunningStatsDetail stats))
      {
        stats = new FPURunningStatsDetail(id);
        statsById[id] = stats;
      }

      stats.All.Add(corrAll);
      stats.GT1Pct.Add(corrSubset); // safely ignores NaN
      stats.AllMAD.Add(madAll);
      stats.GT1PctMAD.Add(madSubset);
      stats.AllBias.Add(biasAll);
      stats.GT1PctBias.Add(biasSubset);

      shouldDump = stats.All.WindowCount >= FPURunningStatsDetail.DUMP_INTERVAL;
      if (shouldDump)
      {
        stats.All.Snapshot(out allWinN, out allWinAvg, out allTotN, out allTotAvg);
        stats.GT1Pct.Snapshot(out gtWinN, out gtWinAvg, out gtTotN, out gtTotAvg);
        stats.AllMAD.Snapshot(out allMadWinN, out allMadWinAvg, out allMadTotN, out allMadTotAvg);
        stats.GT1PctMAD.Snapshot(out gtMadWinN, out gtMadWinAvg, out gtMadTotN, out gtMadTotAvg);
        stats.AllBias.Snapshot(out allBiasWinN, out allBiasWinAvg, out allBiasTotN, out allBiasTotAvg);
        stats.GT1PctBias.Snapshot(out gtBiasWinN, out gtBiasWinAvg, out gtBiasTotN, out gtBiasTotAvg);
        stats.All.ResetWindow();
        stats.GT1Pct.ResetWindow();
        stats.AllMAD.ResetWindow();
        stats.GT1PctMAD.ResetWindow();
        stats.AllBias.ResetWindow();
        stats.GT1PctBias.ResetWindow();
      }
    }

    if (shouldDump)
    {
      Dump(id, fpuMode,
           allWinN, allWinAvg, allTotN, allTotAvg,
           gtWinN, gtWinAvg, gtTotN, gtTotAvg,
           allMadWinN, allMadWinAvg, allMadTotN, allMadTotAvg,
           gtMadWinN, gtMadWinAvg, gtMadTotN, gtMadTotAvg,
           allBiasWinN, allBiasWinAvg, allBiasTotN, allBiasTotAvg,
           gtBiasWinN, gtBiasWinAvg, gtBiasTotN, gtBiasTotAvg);
    }
  }

  private static void Dump(string id,
                           ParamsSelect.FPUType fpuMode,
                           int allWinN, double allWinAvg, long allTotN, double allTotAvg,
                           int gtWinN, double gtWinAvg, long gtTotN, double gtTotAvg,
                           int allMadWinN, double allMadWinAvg, long allMadTotN, double allMadTotAvg,
                           int gtMadWinN, double gtMadWinAvg, long gtMadTotN, double gtMadTotAvg,
                           int allBiasWinN, double allBiasWinAvg, long allBiasTotN, double allBiasTotAvg,
                           int gtBiasWinN, double gtBiasWinAvg, long gtBiasTotN, double gtBiasTotAvg)
  {
    ConsoleColor prev = Console.ForegroundColor;
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine(
      $"[FPU correlation] ID={id,-12} FPUMode={fpuMode,-14} " +
      $"All:    last {allWinN,5} avg={FormatAvg(allWinAvg)}  lifetime N={allTotN,-10} avg={FormatAvg(allTotAvg)}  | " +
      $"GT1Pct: last {gtWinN,5} avg={FormatAvg(gtWinAvg)}  lifetime N={gtTotN,-10} avg={FormatAvg(gtTotAvg)}");
    Console.WriteLine(
      $"[FPU mean-abs-dev] ID={id,-12} FPUMode={fpuMode,-14} " +
      $"All:    last {allMadWinN,5} avg={FormatAvg(allMadWinAvg)}  lifetime N={allMadTotN,-10} avg={FormatAvg(allMadTotAvg)}  | " +
      $"GT1Pct: last {gtMadWinN,5} avg={FormatAvg(gtMadWinAvg)}  lifetime N={gtMadTotN,-10} avg={FormatAvg(gtMadTotAvg)}");
    Console.WriteLine(
      $"[FPU bias       ] ID={id,-12} FPUMode={fpuMode,-14} " +
      $"All:    last {allBiasWinN,5} avg={FormatAvg(allBiasWinAvg)}  lifetime N={allBiasTotN,-10} avg={FormatAvg(allBiasTotAvg)}  | " +
      $"GT1Pct: last {gtBiasWinN,5} avg={FormatAvg(gtBiasWinAvg)}  lifetime N={gtBiasTotN,-10} avg={FormatAvg(gtBiasTotAvg)}");
    Console.ForegroundColor = prev;
  }

  private static string FormatAvg(double v) => double.IsNaN(v) ? "    N/A" : $"{v,7:F4}";
}
