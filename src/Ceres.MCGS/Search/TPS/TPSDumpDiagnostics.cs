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

using Ceres.Base.Misc;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PUCT;

#endregion

namespace Ceres.MCGS.Search.TPS;

/// <summary>
/// Optional diagnostic dump for the TPS (Tempered Posterior Search) backup path
/// (TPSScoreCalc.ComputeVBar).  Gated by DEBUG_DUMP_TPS_BACKUP_CALCS below; when the
/// gate is false the helper (and its windowed running-stats machinery) is unreachable
/// and the JIT removes it entirely.
///
/// Output format per dump:
///   - One header line (default color, or red when significant) summarizing inputs.
///   - One row per per-child input vector (default color).
///   - One row per per-child output vector (yellow), so the result stands out.
///
/// Rows render as "Label                v0 v1 v2 | v3 v4 ..." where the '|'
/// separator marks the boundary between expanded (visited) children and
/// unexpanded (FPU-imputed) children.  Shared row/value formatting lives in
/// SearchDumpFormatting (also used by the FPU/RPO dump).
///
/// Where useful, the dump also includes the value the standard (visit-weighted)
/// backup would have produced from the same inputs, so the effect of the tempered
/// posterior is immediately visible.
/// </summary>
internal static class TPSDumpDiagnostics
{
  /// <summary>
  /// Gate for TPSScoreCalc.ComputeVBar (tempered-posterior backup) dumps.
  /// </summary>
  public const bool DEBUG_DUMP_TPS_BACKUP_CALCS = false;


  /// <summary>
  /// Threshold (absolute Q) for flagging a backup dump as a significant
  /// divergence from the standard visit-weighted backup.  Header turns red when
  /// |V - PUCT_Q| &gt; threshold.
  /// </summary>
  public const double BACKUP_DIFF_THRESHOLD = 0.1;


  /// <summary>
  /// Per-search running stats for the TPS backup, bucketed by parent totalN
  /// (node.NodeRef.N at the time of the backup).  Accumulated for every backup
  /// when DEBUG_DUMP_TPS_BACKUP_CALCS is true (regardless of ONLY_SHOW_SIGNIFICANT),
  /// then flushed every BACKUP_STATS_FLUSH_INTERVAL_SECONDS as one line per non-empty
  /// bucket.  Each flush snapshots-and-zeros the counters so each dump reflects the
  /// prior interval only, not cumulative history.
  ///
  /// Keyed by ParamsSelect via ConditionalWeakTable, so counters are reclaimed
  /// automatically when the search ends.
  /// </summary>
  private sealed class BackupStatsCounter
  {
    public readonly object Lock = new();
    public DateTime WindowStartUtc = DateTime.UtcNow;
    public readonly long[] Count = new long[NumBackupNBuckets];
    public readonly double[] SumDiff = new double[NumBackupNBuckets];      // signed (V - PUCT_Q)
    public readonly double[] SumSqDiff = new double[NumBackupNBuckets];    // squared diff for std dev
    public readonly double[] SumAbsDiff = new double[NumBackupNBuckets];   // |V - PUCT_Q|
    public readonly double[] SumTau = new double[NumBackupNBuckets];       // per-call tau_backup
  }

  /// <summary>
  /// parent-totalN cutpoints for the backup running-stats buckets.  Finer at the low end
  /// (where the temperature is warmest and V is most likely to diverge from the
  /// visit-weighted backup) and coarser at the high end where convergence has set in.
  /// </summary>
  private static readonly double[] backupNBucketCutpoints =
    { 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0 };
  private const int NumBackupNBuckets = 8;
  private static readonly string[] backupNBucketLabels =
    { "<10", "10-30", "30-100", "100-300", "300-1k", "1k-3k", "3k-10k", ">=10k" };

  private static readonly ConditionalWeakTable<ParamsSelect, BackupStatsCounter>
    backupStatsByParamsSelect = new();

  /// <summary>
  /// Window length for the backup stats flush.  Each DumpTPSBackup call does a
  /// cheap unlocked elapsed check; the first thread to observe the window has closed
  /// claims the flush under stats.Lock, snapshots-and-zeros the counters, then prints.
  /// </summary>
  private const double BACKUP_STATS_FLUSH_INTERVAL_SECONDS = 60.0;


  /// <summary>
  /// Returns the bucket index for a given parent totalN.  Cutpoints are
  /// backupNBucketCutpoints; boundary points belong to the higher bucket
  /// (e.g. totalN == 10 maps to bucket 1, totalN == 30 to bucket 2).
  /// </summary>
  private static int BackupNBucketIndex(double totalN)
  {
    for (int i = 0; i < backupNBucketCutpoints.Length; i++)
    {
      if (totalN < backupNBucketCutpoints[i])
      {
        return i;
      }
    }
    return backupNBucketCutpoints.Length;
  }


  /// <summary>
  /// Snapshots the bucket counters under stats.Lock (re-checks elapsed time so only
  /// one thread per window flushes), zeros them and updates WindowStartUtc, then prints
  /// one line per non-empty bucket under the shared console lock.  Reports per bucket:
  ///   bias          = mean signed (V - PUCT_Q)             -- direction of TPS bias
  ///   mean_abs_diff = mean |V - PUCT_Q|                    -- typical gap size
  ///   std_dev       = sqrt(max(0, sumSqDiff/n - bias^2))   -- dispersion (clamp for FP safety)
  ///   mean_tau      = mean tau_backup over the window      -- average temperature
  /// </summary>
  private static void FlushBackupStats(BackupStatsCounter stats, DateTime nowUtc)
  {
    long[] count = new long[NumBackupNBuckets];
    double[] sumDiff = new double[NumBackupNBuckets];
    double[] sumSqDiff = new double[NumBackupNBuckets];
    double[] sumAbsDiff = new double[NumBackupNBuckets];
    double[] sumTau = new double[NumBackupNBuckets];
    double windowSeconds;

    lock (stats.Lock)
    {
      windowSeconds = (nowUtc - stats.WindowStartUtc).TotalSeconds;
      if (windowSeconds < BACKUP_STATS_FLUSH_INTERVAL_SECONDS)
      {
        // Another thread already flushed; nothing to do.
        return;
      }
      for (int r = 0; r < NumBackupNBuckets; r++)
      {
        count[r] = stats.Count[r];
        sumDiff[r] = stats.SumDiff[r];
        sumSqDiff[r] = stats.SumSqDiff[r];
        sumAbsDiff[r] = stats.SumAbsDiff[r];
        sumTau[r] = stats.SumTau[r];
        stats.Count[r] = 0;
        stats.SumDiff[r] = 0.0;
        stats.SumSqDiff[r] = 0.0;
        stats.SumAbsDiff[r] = 0.0;
        stats.SumTau[r] = 0.0;
      }
      stats.WindowStartUtc = nowUtc;
    }

    long total = 0;
    int nonEmpty = 0;
    for (int r = 0; r < NumBackupNBuckets; r++)
    {
      if (count[r] > 0)
      {
        total += count[r];
        nonEmpty++;
      }
    }

    lock (SearchDumpFormatting.ConsoleLock)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Cyan,
        $"[TPS-BAK-STATS] last {windowSeconds:F1}s ({nonEmpty} bucket(s), {total} backups)");
      for (int r = 0; r < NumBackupNBuckets; r++)
      {
        long n = count[r];
        if (n == 0)
        {
          continue;
        }
        double bias = sumDiff[r] / n;
        double meanAbsDiff = sumAbsDiff[r] / n;
        double variance = sumSqDiff[r] / n - bias * bias;
        if (variance < 0.0)
        {
          variance = 0.0;
        }
        double stdDev = Math.Sqrt(variance);
        double meanTau = sumTau[r] / n;
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan,
          $"  bucket={backupNBucketLabels[r],-8} n={n,8} "
          + $"bias={bias,+8:+0.0000;-0.0000} "
          + $"mean_abs_diff={meanAbsDiff,7:F4} "
          + $"std_dev={stdDev,7:F4} "
          + $"mean_tau={meanTau,7:F4}");
      }
    }
  }


  /// <summary>
  /// Computes the value the standard (non-TPS) backup would produce:
  /// the visit-weighted child Q blended with the node's own network value V.
  /// Mirrors the TPS blend formula but uses empirical edge.N weights instead
  /// of the tempered posterior.
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
  /// Dump for TPSScoreCalc.ComputeVBar (tempered-posterior backup).
  ///
  /// Row order tells the story of the calculation: raw per-child support (edgeN,
  /// childN) and observations (Q_obs; imputed value shown for unvisited slots so the
  /// row is never NaN), the policy-implied shrinkage targets (Q_fpu), the robust
  /// values actually aggregated (Q_tilde), the policy (P), the tempered posterior
  /// (pi_tilde, yellow), the per-child contribution pi_tilde * Q_tilde (yellow;
  /// sums to childContrib), and finally V with the standard visit-weighted PUCT_Q
  /// alongside for comparison.  Also accumulates the windowed per-bucket
  /// V-vs-PUCT_Q running stats (see FlushBackupStats).
  /// </summary>
  public static void DumpTPSBackup(GNode node,
                                   ParamsSelect paramsSelect,
                                   ReadOnlySpan<double> mu,
                                   ReadOnlySpan<double> qObs,
                                   ReadOnlySpan<double> qFpu,
                                   ReadOnlySpan<double> qTilde,
                                   ReadOnlySpan<double> piTilde,
                                   ReadOnlySpan<double> edgeN,
                                   ReadOnlySpan<double> nSupport,
                                   int numChildren, int numExpanded,
                                   double sigmaBar, double tauBackup, double meanShrinkW,
                                   double childContribution, double consensusQ, double vBar)
  {
    if (!DEBUG_DUMP_TPS_BACKUP_CALCS)
    {
      return;
    }

    double nodeQ = node.Q;
    double selfV = node.NodeRef.V;
    int totalN = node.NodeRef.N;

    double[] m = new double[numChildren];
    double[] qoDisplay = new double[numChildren];
    double[] qfpu = new double[numChildren];
    double[] qt = new double[numChildren];
    double[] pi = new double[numChildren];
    double[] contrib = new double[numChildren];
    double[] en = new double[numChildren];
    double[] ns = new double[numChildren];
    for (int i = 0; i < numChildren; i++)
    {
      m[i] = mu[i];
      // Q_obs row shows the raw observation for visited slots and the imputed target
      // for unvisited slots (edgeN row = 0 marks the imputed ones), so the shrinkage
      // effect (Q_obs vs Q_tilde) is visible.
      qoDisplay[i] = double.IsNaN(qObs[i]) ? qFpu[i] : qObs[i];
      qfpu[i] = qFpu[i];
      qt[i] = qTilde[i];
      pi[i] = piTilde[i];
      en[i] = edgeN[i];
      ns[i] = nSupport[i];
      contrib[i] = piTilde[i] * qTilde[i];
    }

    // PUCT_Q comparison uses the truly-raw qObs (with NaN intact for unvisited) so that
    // unvisited slots are correctly excluded from the visit-weighted average; this keeps
    // the side-by-side comparison fair to the legacy backup, which never imputes.
    double puctQ = ComputeStandardBackupQ(qObs, edgeN, numChildren, selfV, totalN, nodeQ);
    double backupDelta = Math.Abs(vBar - puctQ);

    // Accumulate running stats by parent totalN bucket BEFORE the ONLY_SHOW_SIGNIFICANT
    // gate, so the windowed summary reflects every backup - not just the ones loud
    // enough to print individually.
    BackupStatsCounter stats = backupStatsByParamsSelect.GetValue(
      paramsSelect, _ => new BackupStatsCounter());
    int backupBucket = BackupNBucketIndex(totalN);
    double signedDiff = vBar - puctQ;
    lock (stats.Lock)
    {
      stats.Count[backupBucket]++;
      stats.SumDiff[backupBucket] += signedDiff;
      stats.SumSqDiff[backupBucket] += signedDiff * signedDiff;
      stats.SumAbsDiff[backupBucket] += backupDelta;
      stats.SumTau[backupBucket] += tauBackup;
    }
    DateTime nowUtc = DateTime.UtcNow;
    if ((nowUtc - stats.WindowStartUtc).TotalSeconds >= BACKUP_STATS_FLUSH_INTERVAL_SECONDS)
    {
      FlushBackupStats(stats, nowUtc);
    }

    bool significant = backupDelta > BACKUP_DIFF_THRESHOLD;
    if (SearchDumpFormatting.ONLY_SHOW_SIGNIFICANT && !significant)
    {
      return;
    }

    int boundary = numExpanded;

    lock (SearchDumpFormatting.ConsoleLock)
    {
      Console.WriteLine();
      SearchDumpFormatting.WriteHeaderLine(
        $"[TPS-BAK] numChildren={numChildren} (expanded={numExpanded}) " +
        $"N={totalN} sigmaBar={sigmaBar:F4} tau={tauBackup:F4} meanW={meanShrinkW:F3} " +
        $"nodeQ={nodeQ:F3} selfV={selfV:F3} consensusQ={consensusQ:F3} " +
        $"delta={backupDelta:F3}", significant);
      Console.WriteLine(SearchDumpFormatting.FormatRow("edgeN:", numChildren, boundary, i => SearchDumpFormatting.FmtNInt(en[i])));
      Console.WriteLine(SearchDumpFormatting.FormatRow("childN:", numChildren, boundary, i => SearchDumpFormatting.FmtNInt(ns[i])));
      Console.WriteLine(SearchDumpFormatting.FormatRow("Q_obs:", numChildren, boundary, i => SearchDumpFormatting.FmtQ(qoDisplay[i])));
      Console.WriteLine(SearchDumpFormatting.FormatRow("Q_fpu (target):", numChildren, boundary, i => SearchDumpFormatting.FmtQ(qfpu[i])));
      Console.WriteLine(SearchDumpFormatting.FormatRow("Q_tilde (robust):", numChildren, boundary, i => SearchDumpFormatting.FmtQ(qt[i])));
      Console.WriteLine(SearchDumpFormatting.FormatRow("P:", numChildren, boundary, i => SearchDumpFormatting.FmtP(m[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        SearchDumpFormatting.FormatRow("pi_tilde:", numChildren, boundary, i => SearchDumpFormatting.FmtP(pi[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        SearchDumpFormatting.FormatRow("contribution:", numChildren, boundary, i => SearchDumpFormatting.FmtQ(contrib[i])));
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        $"V={vBar:F4} (childContrib={childContribution:F4}, totalN={totalN})    PUCT_Q={puctQ:F4}");
    }
  }
}
