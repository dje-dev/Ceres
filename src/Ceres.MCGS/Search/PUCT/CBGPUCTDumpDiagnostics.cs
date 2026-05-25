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
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;

using Ceres.Base.Misc;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.RPO;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// Optional diagnostic dump helpers for the PUCT / CB-GPUCT calculation paths.
/// Each dump is independently gated by a public const bool below; when all
/// gates are false the helpers are unreachable and the JIT removes them
/// (and any call site guards) entirely.
///
/// Output format per dump:
///   - One header line (default color) summarizing inputs to the calculation.
///   - One row per per-child input vector (default color).
///   - One row per per-child output vector (yellow), so the result of the
///     calculation stands out visually in a noisy console stream.
///
/// Where useful, the dump also includes the value that the prior non-CBGPUCT
/// (standard PUCT / standard backup) path would have produced from the same
/// inputs, so the effect of the regularizer is immediately visible.
///
/// Rows render as "Label                v0 v1 v2 | v3 v4 ..." where the '|'
/// separator marks the boundary between expanded (visited) children and
/// unexpanded (FPU-imputed) children.
///
/// All output goes through a static lock so that multi-line dumps from
/// concurrent search threads are not interleaved on the console.
/// </summary>
internal static class CBGPUCTDumpDiagnostics
{
  /// <summary>
  /// Gate for PUCTSelector.ApplyRPOImputedFPU dumps.
  /// </summary>
  public const bool DEBUG_DUMP_FPU_CALCS = false;

  /// <summary>
  /// Gate for CBGPUCTScoreCalc.ScoreCalc (visit-target selection) dumps.
  /// </summary>
  public const bool DEBUG_DUMP_CBGPUCT_SELECT_CALCS = false;

  /// <summary>
  /// Gate for CBGPUCTScoreCalc.ComputeVBar (regularized backup) dumps.
  /// </summary>
  public const bool DEBUG_DUMP_CBGPUCT_BACKUP_CALCS = false;


  /// <summary>
  /// When true, dumps are emitted ONLY for calculation invocations where the
  /// corresponding threshold was exceeded (i.e. the header would have been
  /// red).  Useful for surfacing just the cases where the CB-GPUCT / RPO
  /// machinery actually diverges from the prior algorithm.  Default false.
  /// </summary>
  public const bool ONLY_SHOW_SIGNIFICANT = false;


  /// <summary>
  /// Threshold (in visits) for flagging a select dump as a significant
  /// divergence from vanilla PUCT.  The metric is the number of visits CB-GPUCT
  /// allocated to children that vanilla PUCT would not have visited at all
  /// this batch (equivalently, half the L1 distance between the two visit
  /// vectors).  Header turns red when redirected &gt;= threshold.
  /// </summary>
  public const int SELECT_REDIRECT_THRESHOLD = 1;

  /// <summary>
  /// Threshold (absolute Q) for flagging a backup dump as a significant
  /// divergence from the standard non-CBGPUCT backup.  Header turns red when
  /// |V_bar - PUCT_Q| &gt; threshold.
  /// </summary>
  public const double BACKUP_DIFF_THRESHOLD = 0.1;

  /// <summary>
  /// Threshold (absolute Q) for flagging an FPU dump as a significant
  /// divergence from the prior (non-RPO) FPU algorithm.  Header turns red when
  /// any unvisited child's imputed Q differs from the scalar default FPU by
  /// more than this amount.
  /// </summary>
  public const double FPU_DIFF_THRESHOLD = 0.1;


  /// <summary>
  /// Width of the leading label column.
  /// </summary>
  private const int LABEL_WIDTH = 22;


  /// <summary>
  /// Shared lock that serializes the entire multi-line dump for a single
  /// invocation, preventing rows from concurrent calc threads from interleaving.
  /// </summary>
  private static readonly object consoleLock = new object();


  /// <summary>
  /// Per-search cumulative counts used by the select diagnostic header.  Counts are
  /// bucketed by sumN at the time of the select calculation, partitioned by the
  /// cutpoints below.  For each bucket we track:
  ///   TotalSelects[r]     = number of CB-GPUCT select calculations seen in bucket r.
  ///   MoreExploratory[r]  = subset whose visit distribution was "more rightward
  ///                         leaning" than vanilla PUCT (CB-GPUCT placed more visits
  ///                         on lower-P children, pWeightedShift &lt; 0).
  ///   LessExploratory[r]  = subset that was less exploratory than vanilla PUCT
  ///                         (CB-GPUCT placed more visits on higher-P children,
  ///                         pWeightedShift &gt; 0).
  /// The residual TotalSelects[r] - MoreExploratory[r] - LessExploratory[r] is the
  /// number of calculations in bucket r where the two allocations agreed.
  ///
  /// Keyed by ParamsSelect reference (each ParamsSearch instance has its own
  /// ParamsSelect, so this effectively gives per-ParamsSearch tracking).  Uses
  /// ConditionalWeakTable so counters are reclaimed when the search ends.
  /// </summary>
  private sealed class SelectStatsCounter
  {
    public readonly long[] TotalSelects = new long[NumSumNBuckets];
    public readonly long[] MoreExploratory = new long[NumSumNBuckets];
    public readonly long[] LessExploratory = new long[NumSumNBuckets];
  }

  /// <summary>
  /// sumN cutpoints used to bucket the per-search exploratory statistics.  Bucket r
  /// covers [sumNBucketCutpoints[r-1], sumNBucketCutpoints[r]) for r = 1..NumSumNBuckets-2,
  /// bucket 0 covers [0, sumNBucketCutpoints[0]), and the last bucket covers
  /// [sumNBucketCutpoints[NumSumNBuckets-2], +inf).
  /// </summary>
  private static readonly double[] sumNBucketCutpoints = { 500.0, 5000.0, 50000.0 };
  private const int NumSumNBuckets = 4;
  private static readonly string[] sumNBucketLabels = { "<500", "500-5k", "5k-50k", ">=50k" };

  private static readonly ConditionalWeakTable<ParamsSelect, SelectStatsCounter> selectStatsByParamsSelect = new();


  /// <summary>
  /// Returns the bucket index for a given sumN.  Boundary points belong to the
  /// higher bucket (i.e. sumN==500 maps to bucket 1, sumN==5000 maps to bucket 2).
  /// </summary>
  private static int SumNBucketIndex(double sumN)
  {
    for (int i = 0; i < sumNBucketCutpoints.Length; i++)
    {
      if (sumN < sumNBucketCutpoints[i])
      {
        return i;
      }
    }
    return sumNBucketCutpoints.Length;
  }


  /// <summary>
  /// Formats a single row of per-child values with a '|' separator inserted
  /// after the expanded/unexpanded boundary (i.e. after index boundaryIndex-1).
  /// If boundaryIndex is &lt;= 0 or &gt;= count, no separator is inserted.
  /// </summary>
  public static string FormatRow(string label, int count, int boundaryIndex, Func<int, string> valueFunc)
  {
    StringBuilder sb = new();
    sb.Append(label.PadRight(LABEL_WIDTH));
    for (int i = 0; i < count; i++)
    {
      if (i > 0)
      {
        sb.Append(' ');
      }
      sb.Append(valueFunc(i));
      if (i == boundaryIndex - 1 && boundaryIndex > 0 && boundaryIndex < count)
      {
        sb.Append(" |");
      }
    }
    return sb.ToString();
  }


  /// <summary>
  /// Formats a double with a fixed 6-char field width (sign + 0.000).
  /// </summary>
  private static string FmtQ(double v)
  {
    if (double.IsNaN(v))
    {
      return "  NaN ";
    }
    return v.ToString("+0.000;-0.000; 0.000").PadLeft(6);
  }


  /// <summary>
  /// Formats a non-negative double (policy / pi_bar) in [0,1] with 6 chars.
  /// </summary>
  private static string FmtP(double v)
  {
    if (double.IsNaN(v))
    {
      return "  NaN ";
    }
    return v.ToString("0.000").PadLeft(6);
  }


  /// <summary>
  /// Formats a visit count (integer-ish, may include fractional in-flight).
  /// </summary>
  private static string FmtN(double v)
  {
    return v.ToString("0.0").PadLeft(6);
  }


  /// <summary>
  /// Simulates a standard (non-CBGPUCT) PUCT visit allocation from the same
  /// per-child inputs used by CB-GPUCT.  Uses the canonical PUCT formula
  ///   score(i) = q(i) + cpuct * P(i) * sqrt(sumN) / (1 + n(i))
  /// with greedy sequential allocation over the visit budget.  Cpuct comes
  /// from paramsSelect.CalcCPUCT (cpuctMultiplier assumed = 1.0; this is for
  /// visual comparison only, not production allocation).
  /// </summary>
  private static void SimulateVanillaPUCT(ParamsSelect paramsSelect, bool parentIsRoot, int parentN,
                                          ReadOnlySpan<double> mu, ReadOnlySpan<double> qIn,
                                          ReadOnlySpan<double> currentN,
                                          int numChildren, int budget,
                                          Span<int> visitsOut)
  {
    double cpuct = paramsSelect.CalcCPUCT(parentIsRoot, parentN);
    double sumN = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      sumN += currentN[i];
      visitsOut[i] = 0;
    }

    for (int v = 0; v < budget; v++)
    {
      double sqrtSum = Math.Sqrt(Math.Max(sumN, 1.0));
      int bestIdx = 0;
      double bestScore = double.NegativeInfinity;
      for (int i = 0; i < numChildren; i++)
      {
        double n = currentN[i] + visitsOut[i];
        double q = qIn[i];
        if (double.IsNaN(q))
        {
          q = 0.0;
        }
        double score = q + cpuct * mu[i] * sqrtSum / (1.0 + n);
        if (score > bestScore)
        {
          bestScore = score;
          bestIdx = i;
        }
      }
      visitsOut[bestIdx]++;
      sumN += 1.0;
    }
  }


  /// <summary>
  /// Computes the value the standard (non-CBGPUCT) backup would produce:
  /// the visit-weighted child Q blended with the node's own network value V.
  /// Mirrors the V_bar blend formula but uses empirical edge.N weights instead
  /// of pi_bar.
  /// </summary>
  private static double ComputeStandardBackupQ(ReadOnlySpan<double> qRaw, ReadOnlySpan<double> edgeN,
                                               int numChildren, double selfV, int totalN, double nodeQ)
  {
    double weightedSum = 0.0;
    double sumW = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      double n = edgeN[i];
      if (n > 0 && !double.IsNaN(qRaw[i]))
      {
        weightedSum += n * qRaw[i];
        sumW += n;
      }
    }

    double childAvg = sumW > 0.0 ? weightedSum / sumW : nodeQ;
    if (totalN <= 0)
    {
      return selfV;
    }
    return (childAvg * (totalN - 1) + selfV) / totalN;
  }


  /// <summary>
  /// Writes a dump header line.  If significant is true, the header is rendered
  /// in red to draw the eye; otherwise default color is used.
  /// </summary>
  private static void WriteHeaderLine(string header, bool significant)
  {
    if (significant)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, header);
    }
    else
    {
      Console.WriteLine(header);
    }
  }


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
    if (ONLY_SHOW_SIGNIFICANT && !significant)
    {
      return;
    }

    lock (consoleLock)
    {
      Console.WriteLine();
      WriteHeaderLine($"[FPU/RPO] NumEdgesExpanded={numExpanded}/{numToProcess} Q={nodeQ:F3} " +
                      $"tau={lambda:F3} reg={regularization} anchor={anchorStr} " +
                      $"defaultFPU={defaultFPU:F3} maxFPUDelta={maxFPUDelta:F3}", significant);
      Console.WriteLine(FormatRow("Q_in (parent persp):", numToProcess, numExpanded, i => FmtQ(q[i])));
      Console.WriteLine(FormatRow("P:", numToProcess, numExpanded, i => FmtP(p[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        FormatRow("Q_out (imputed):", numToProcess, numExpanded, i => FmtQ(qO[i])));
      Console.WriteLine(FormatRow("PUCT_FPU (prior):", numToProcess, numExpanded, i => FmtQ(qPrior[i])));
    }
  }


  /// <summary>
  /// Dump for CBGPUCTScoreCalc.ScoreCalc (visit-target selection).
  /// Also computes (for visual comparison) the visit allocation that vanilla
  /// PUCT would have produced from the same per-child inputs.
  /// </summary>
  public static void DumpCBGPUCTSelect(ParamsSelect paramsSelect,
                                       GNode parentNode,
                                       ReadOnlySpan<double> pSpan,
                                       ReadOnlySpan<double> qInOriginal,
                                       ReadOnlySpan<double> qIn,
                                       ReadOnlySpan<double> piBar,
                                       ReadOnlySpan<double> firstStepDeficits,
                                       ReadOnlySpan<double> currentN,
                                       ReadOnlySpan<double> nEdge,
                                       ReadOnlySpan<double> nInFlight,
                                       ReadOnlySpan<short> visitsAdded,
                                       int numChildren, int numVisitsToCompute,
                                       double lambdaN, double sumNStart, double vStar,
                                       RPORegularization regularization)
  {
    if (!DEBUG_DUMP_CBGPUCT_SELECT_CALCS)
    {
      return;
    }

    int numExpanded = parentNode.NumEdgesExpanded;
    bool parentIsRoot = parentNode.IsSearchRoot;
    int parentN = parentNode.NodeRef.N;

    double[] p = new double[numChildren];
    double[] qOrig = new double[numChildren];
    double[] q = new double[numChildren];
    double[] pi = new double[numChildren];
    double[] dfc = new double[numChildren];
    double[] cn = new double[numChildren];
    double[] ne = new double[numChildren];
    double[] nf = new double[numChildren];
    int[] va = new int[numChildren];
    bool anyQDelta = false;
    for (int i = 0; i < numChildren; i++)
    {
      p[i] = pSpan[i];
      qOrig[i] = qInOriginal[i];
      q[i] = qIn[i];
      if (qOrig[i] != q[i])
      {
        anyQDelta = true;
      }
      pi[i] = piBar[i];
      dfc[i] = firstStepDeficits[i];
      cn[i] = currentN[i];
      ne[i] = nEdge[i];
      nf[i] = nInFlight[i];
      va[i] = visitsAdded[i];
    }

    // Simulate what vanilla PUCT would have allocated this batch.
    int[] vanilla = new int[numChildren];
    if (numVisitsToCompute > 0)
    {
      SimulateVanillaPUCT(paramsSelect, parentIsRoot, parentN, p, q, cn,
                          numChildren, numVisitsToCompute, vanilla);
    }

    // Count visits CB-GPUCT placed that vanilla PUCT did not: equivalent to
    // half the L1 distance between the two allocation vectors (each redirected
    // visit shows up once as a positive delta on one child and once as a
    // negative delta on another, so summing the positive deltas gives the
    // number of visits actually relocated).
    int redirected = 0;
    for (int i = 0; i < numChildren; i++)
    {
      int delta = va[i] - vanilla[i];
      if (delta > 0)
      {
        redirected += delta;
      }
    }

    // Exploration-shift detector: P-weighted shift of visits between the two
    // allocations.  Sign tells us which direction CB-GPUCT moved visits relative
    // to vanilla PUCT:
    //   pWeightedShift < 0 : visits onto lower-P children   (more exploratory)
    //   pWeightedShift > 0 : visits onto higher-P children  (less exploratory)
    //   pWeightedShift == 0: allocations agreed             (neither bucket)
    // Independent of child ordering in the array.
    double pWeightedShift = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      pWeightedShift += p[i] * (va[i] - vanilla[i]);
    }
    bool moreExploratory = pWeightedShift < 0.0;
    bool lessExploratory = pWeightedShift > 0.0;

    // Update the cumulative per-search counters in the appropriate sumN bucket and
    // read back the running totals for display.  Interlocked increments keep updates
    // lock-free; the reads below may momentarily disagree under heavy concurrency but
    // the displayed ratios are at most one increment off, which is fine for debug output.
    int bucket = SumNBucketIndex(sumNStart);
    SelectStatsCounter stats = selectStatsByParamsSelect.GetValue(paramsSelect, _ => new SelectStatsCounter());
    Interlocked.Increment(ref stats.TotalSelects[bucket]);
    if (moreExploratory)
    {
      Interlocked.Increment(ref stats.MoreExploratory[bucket]);
    }
    else if (lessExploratory)
    {
      Interlocked.Increment(ref stats.LessExploratory[bucket]);
    }

    StringBuilder bucketsSb = new();
    bucketsSb.Append("cum more/less explor by sumN:");
    for (int r = 0; r < NumSumNBuckets; r++)
    {
      long t = Volatile.Read(ref stats.TotalSelects[r]);
      long m = Volatile.Read(ref stats.MoreExploratory[r]);
      long l = Volatile.Read(ref stats.LessExploratory[r]);
      double mPct = t > 0 ? 100.0 * m / t : 0.0;
      double lPct = t > 0 ? 100.0 * l / t : 0.0;
      bucketsSb.Append($" {sumNBucketLabels[r]}={m}/{l}({mPct:F0}%/{lPct:F0}%)");
    }

    bool significant = redirected >= SELECT_REDIRECT_THRESHOLD;
    if (ONLY_SHOW_SIGNIFICANT && !significant)
    {
      return;
    }

    lock (consoleLock)
    {
      Console.WriteLine();
      WriteHeaderLine($"[CBGPUCT-SEL] root={parentIsRoot} numChildren={numChildren} " +
                      $"sumN={sumNStart:F1} lambda_N={lambdaN:F4} v*={vStar:F3} " +
                      $"budget={numVisitsToCompute} reg={regularization} " +
                      $"redirected={redirected}    " +
                      bucketsSb.ToString(),
                      significant);
      // "Q_in (pre-shrink)" shows the raw Q values (-W/N for visited, original FPU for
      // unvisited) before any Bayesian shrinkage or fixed-point iteration adjustment.
      // Only shown when the two vectors actually differ - typically when shrinkage is
      // active (CBGPUCT_QShrinkageSelectFractionAtN1 > 0) or fixed-point iteration is
      // active (RPOSelectFixedPointIterations > 0).
      if (anyQDelta)
      {
        Console.WriteLine(FormatRow("Q_in (pre-shrink):", numChildren, numExpanded, i => FmtQ(qOrig[i])));
      }
      Console.WriteLine(FormatRow("Q_in (post-shrink):", numChildren, numExpanded, i => FmtQ(q[i])));
      Console.WriteLine(FormatRow("nEdge:", numChildren, numExpanded, i => FmtN(ne[i])));
      Console.WriteLine(FormatRow("nInFlight:", numChildren, numExpanded, i => FmtN(nf[i])));
      Console.WriteLine(FormatRow("currentN:", numChildren, numExpanded, i => FmtN(cn[i])));
      Console.WriteLine(FormatRow("P:", numChildren, numExpanded, i => FmtP(p[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        FormatRow("pi_bar:", numChildren, numExpanded, i => FmtP(pi[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        FormatRow("first_step_deficit:", numChildren, numExpanded, i => FmtQ(dfc[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        FormatRow("visits_added:", numChildren, numExpanded, i => va[i].ToString().PadLeft(6)));
      Console.WriteLine(FormatRow("PUCT visits_added:", numChildren, numExpanded, i => vanilla[i].ToString().PadLeft(6)));
    }
  }


  /// <summary>
  /// Dump for CBGPUCTScoreCalc.ComputeVBar (regularized backup).
  /// Also computes (for visual comparison) the parent-Q the standard
  /// non-CBGPUCT backup would have produced from the same inputs.
  ///
  /// numExpanded marks the boundary between expanded children (i &lt; numExpanded) and
  /// extended-coverage unexpanded children (numExpanded &lt;= i &lt; numChildren).  A '|'
  /// separator is drawn after the boundary so the two sub-blocks are visually distinct.
  ///
  /// qFill is Solve's NaN-imputed q vector; it lets the Q_raw row show the imputed value
  /// for unvisited slots (so the row is never NaN), and lets the contribution row exactly
  /// match the V_bar dot product (which uses qFill[i] for NaN-qRaw slots and qRaw[i] for
  /// visited slots).
  /// </summary>
  public static void DumpCBGPUCTBackup(GNode node,
                                       ReadOnlySpan<double> mu,
                                       ReadOnlySpan<double> qRaw,
                                       ReadOnlySpan<double> qShrunk,
                                       ReadOnlySpan<double> qFill,
                                       ReadOnlySpan<double> piBarPreShrink,
                                       ReadOnlySpan<double> piBar,
                                       ReadOnlySpan<double> edgeN,
                                       int numChildren, int numExpanded, double sumN, double lambdaN,
                                       double childContribution, double vBar,
                                       RPORegularization regularization)
  {
    if (!DEBUG_DUMP_CBGPUCT_BACKUP_CALCS)
    {
      return;
    }

    double nodeQ = node.Q;
    double selfV = node.NodeRef.V;
    int totalN = node.NodeRef.N;

    double[] m = new double[numChildren];
    double[] qrDisplay = new double[numChildren];
    double[] qs = new double[numChildren];
    double[] qf = new double[numChildren];
    double[] piPre = new double[numChildren];
    double[] piPost = new double[numChildren];
    double[] contrib = new double[numChildren];
    double[] en = new double[numChildren];
    for (int i = 0; i < numChildren; i++)
    {
      m[i] = mu[i];
      // Q_raw row shows the actual value the dot product used: raw observation for
      // visited slots, Solve's imputation (qFill) for unvisited slots.  Combined with
      // the edgeN row (0 indicates imputed), the reader can still distinguish them.
      qrDisplay[i] = double.IsNaN(qRaw[i]) ? qFill[i] : qRaw[i];
      qs[i] = qShrunk[i];
      qf[i] = qFill[i];
      piPre[i] = piBarPreShrink[i];
      piPost[i] = piBar[i];
      en[i] = edgeN[i];
      // Contribution uses post-shrinkage piBar (what the dot product actually consumed).
      double qForAvg = double.IsNaN(qRaw[i]) ? qFill[i] : qRaw[i];
      contrib[i] = piBar[i] * qForAvg;
    }

    // PUCT_Q comparison uses the truly-raw qRaw (with NaN intact for unvisited) so that
    // unvisited slots are correctly excluded from the visit-weighted average; this keeps
    // the side-by-side comparison fair to the legacy backup, which never imputes.
    double puctQ = ComputeStandardBackupQ(qRaw, edgeN, numChildren, selfV, totalN, nodeQ);
    double backupDelta = Math.Abs(vBar - puctQ);
    bool significant = backupDelta > BACKUP_DIFF_THRESHOLD;
    if (ONLY_SHOW_SIGNIFICANT && !significant)
    {
      return;
    }

    // Boundary is the index where the expanded segment ends and the unexpanded
    // extended-coverage segment begins.  If numChildren == numExpanded (no extension)
    // or numExpanded == 0, no separator is drawn (FormatRow handles those cases).
    int boundary = numExpanded;

    lock (consoleLock)
    {
      Console.WriteLine();
      WriteHeaderLine($"[CBGPUCT-BAK] numChildren={numChildren} (expanded={numExpanded}) " +
                      $"sumN={sumN:F1} lambda_N={lambdaN:F4} " +
                      $"nodeQ={nodeQ:F3} selfV={selfV:F3} N={totalN} reg={regularization} " +
                      $"delta={backupDelta:F3}", significant);
      Console.WriteLine(FormatRow("edgeN:", numChildren, boundary, i => FmtN(en[i])));
      Console.WriteLine(FormatRow("Q_raw:", numChildren, boundary, i => FmtQ(qrDisplay[i])));
      Console.WriteLine(FormatRow("Q_shrunk:", numChildren, boundary, i => FmtQ(qs[i])));
      Console.WriteLine(FormatRow("Q_fill:", numChildren, boundary, i => FmtQ(qf[i])));
      Console.WriteLine(FormatRow("P:", numChildren, boundary, i => FmtP(m[i])));
      // pi_bar = unshrunk (straight from Solve); pi_bar_shrunk = post all shrinkage, used in dot product.
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        FormatRow("pi_bar:", numChildren, boundary, i => FmtP(piPre[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        FormatRow("pi_bar_shrunk:", numChildren, boundary, i => FmtP(piPost[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        FormatRow("contribution:", numChildren, boundary, i => FmtQ(contrib[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        $"V_bar={vBar:F4} (childContrib={childContribution:F4}, totalN={totalN})    PUCT_Q={puctQ:F4}");
    }
  }
}
