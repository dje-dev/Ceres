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
using System.IO;
using System.Threading;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PUCT;

#endregion

namespace Ceres.MCGS.Search.TPS;

/// <summary>
/// Read-side source for the MUTABLE node/edge statistics consumed by
/// TPSScoreCalc.ComputeVBar, allowing the identical kernel to be evaluated against
/// hypothetical ("shadow") statistics without touching the graph.
///
/// The engine's live backup passes null (all statistics read directly from the graph).
/// The QProbe shadow replay (Ceres.MCGS.TestSuite, ShadowBackup) implements this
/// interface over its private overlay so that hypothetical TPS backup values are
/// computed by exactly the same arithmetic as the engine's own backup - the same
/// guarantee the standard visit-weighted shadow backup provides (QProbe test T2).
///
/// Only statistics the backup phase can change are routed through the overlay:
/// node N, node leaf-value volatility, and per-edge (N, diluted Q). Statistics that
/// are immutable during backup (policy P, network V, edge headers, expansion counts)
/// are always read live.
/// </summary>
internal interface ITPSStatsOverlay
{
  /// <summary>Overlay-adjusted total visit count of a node (GNodeStruct.N analog).</summary>
  int NodeN(GNode node);

  /// <summary>
  /// Overlay-adjusted debiased leaf-value volatility of a node
  /// (GNode.LeafValueVolatilityDebiased analog; the debiasing must use the
  /// overlay-adjusted N).
  /// </summary>
  double NodeVolatilityDebiased(GNode node);

  /// <summary>
  /// Overlay-adjusted (N, diluted Q in child perspective) of the expanded edge at
  /// childSlot of parent (GEdge.N / GEdge.Q analogs).
  /// </summary>
  (int N, double Q) EdgeNAndQ(GNode parent, int childSlot, GEdge edge);
}


/// <summary>
/// TPS (Tempered Posterior Search) BACKUP: a closed-form, solver-free replacement for
/// the visit-weighted backup, active when ParamsSelect.TPS_Mode = BackupOnly.
///
/// At every backed-up node the value is the expectation of a tempered posterior over a
/// robust value vector:
///   V = sum_i pi_i * q_tilde_i,   pi_i proportional to p_i * exp(q_tilde_i / tau)
/// where q_tilde is each child's search Q shrunk toward its policy-imputed prior in
/// proportion to its estimated standard error sigma_hat_i = s_i / sqrt(n_i + 1) (s_i =
/// the child's MEASURED leaf-value volatility - the load-bearing ingredient per the
/// 2026-07-13 frozen-s ablation), and the temperature is a winner's-curse-referenced
/// multiple of the node's measured noise scale:
///   tau = TPS_BackupTemperatureK * sigma_bar * sqrt(2 ln k_visited).
/// The posterior stays policy-PROPORTIONAL at every N (beta = 1: the proposal's
/// beta(N) prior-anchor decay measurably hurts backup), and V blends the node's own
/// network value as one visit, exactly like the standard backup.
///
/// See TPS_PROPOSAL.md and TPS_CAMPAIGN_RESULTS.md (IntegrationTests, harness repo)
/// for the design and the tuning/ablation evidence.  The TPS SELECT designs tested in
/// that campaign were removed (a completely different SELECT algorithm is forthcoming);
/// selection under BackupOnly is standard Ceres PUCT.
///
/// Reuses RPOImputation.ComputePolicyImpliedQ (the PolicyImputedRPO FPU machinery) for
/// the per-child shrinkage targets; requires ParamsSearch.TrackLeafValueVolatility
/// (enforced in ParamsSelect.ValidateAgainst).
/// </summary>
internal static class TPSScoreCalc
{
  #region Constants (deliberately not exposed as knobs; see TPS_PROPOSAL.md section 2)

  /// <summary>
  /// Clamp bounds on the per-child standard error sigma_hat so that 2-byte volatility
  /// quantization / early-sample artifacts can neither zero the temperature (one-hot
  /// posterior at a node that merely LOOKS settled) nor blow up a shrinkage.
  /// </summary>
  private const double SIGMA_HAT_MIN = 0.005;
  private const double SIGMA_HAT_MAX = 0.5;

  /// <summary>
  /// Children with fewer than this many support visits use the parent's volatility
  /// instead of their own (borrowed strength; own estimate too unreliable).
  /// </summary>
  private const int MIN_CHILD_N_FOR_OWN_SIGMA = 4;

  /// <summary>
  /// Volatility assumed when neither the child nor the parent has a warmed-up
  /// estimator (cold start: RunningStdDevShort decodes to 0 until samples arrive).
  /// </summary>
  private const double DEFAULT_PRIOR_VOLATILITY = 0.20;

  /// <summary>
  /// Floor on the temperature: keeps exp(q/tau) finite at fully settled nodes
  /// (log-space computation handles the rest).
  /// </summary>
  private const double TAU_MIN = 1e-3;

  /// <summary>
  /// Minimum raw policy P for an unexpanded child to enter the backup action set
  /// (the imputed-tail pi mass is monitored via the stats sampler).
  /// </summary>
  private const float MIN_P_FOR_Q_IF_UNVISITED = 0.05f;

  /// <summary>
  /// Consensus weight for the q_bar anchor of the policy-imputed prior:
  /// w_a = mu_a * n_a / (n_a + K) (policy x saturating-support gate).
  /// </summary>
  private const double CONSENSUS_RELIABILITY_K = 3.0;

  #endregion

  #region Empirical-variance ablation (env CERES_TPS_FREEZE_S)

  // Research ablation: when CERES_TPS_FREEZE_S is set to a positive value, EVERY child's
  // (and the parent-fallback) leaf-value volatility s is replaced by that constant, so
  // the shrinkage weights collapse to pure count-based James-Stein
  // (w = (n+1)/((n+1)+(s0/sigma0)^2)) and the temperature to a pure count schedule.
  // The 2026-07-13 ablation on HOP @15k showed this collapses TPS to the count-based
  // failure level (-9 -> -35 Elo, blunder asymmetry flips) - i.e. MEASURING the
  // variance is the transfer ingredient.  0 (unset) = normal live measurement.
  private static readonly double FrozenS = ReadFrozenS();

  private static double ReadFrozenS()
  {
    string s = System.Environment.GetEnvironmentVariable("CERES_TPS_FREEZE_S");
    if (!string.IsNullOrEmpty(s)
        && double.TryParse(s, System.Globalization.NumberStyles.Float,
                           System.Globalization.CultureInfo.InvariantCulture, out double v)
        && v > 0)
    {
      Console.WriteLine($"[TPS] ABLATION: empirical variance FROZEN at s = {v} (CERES_TPS_FREEZE_S)");
      return v;
    }
    return 0;
  }

  #endregion

  #region Thread-local scratch buffers

  // Per-thread scratch reused across calls; each buffer is allocated once (lazily, per
  // thread) to PUCTScoreCalcVector.MAX_CHILDREN and sliced to the live numChildren.
  // Every element in [0, numChildren) is fully written before read on each call.
  [ThreadStatic] private static double[] bufferMu;
  [ThreadStatic] private static double[] bufferQObs;
  [ThreadStatic] private static double[] bufferEdgeN;
  [ThreadStatic] private static double[] bufferNSupport;
  [ThreadStatic] private static double[] bufferSChild;
  [ThreadStatic] private static double[] bufferQFpu;
  [ThreadStatic] private static double[] bufferQTilde;
  [ThreadStatic] private static double[] bufferPiTilde;

  #endregion

  #region Kernel

  /// <summary>
  /// STEP 1 of TPS: the robust value vector and the node's noise scale.
  ///
  /// Computes (all in the node's own perspective; caller performs the -edge.Q
  /// negation): the consensus anchor q_bar over visited children, the per-child
  /// policy-imputed targets q^FPU (via the PolicyImputedRPO machinery), the
  /// inverse-variance-shrunk robust values
  ///   q_tilde_i = w_i q_i + (1-w_i) q^FPU_i,   w_i = sigma0^2/(sigma0^2 + sigma_hat_i^2)
  /// (unvisited children: q_tilde = q^FPU exactly, the w=0 limit), and the node noise
  /// scale sigma_bar = support-weighted median of sigma_hat over visited children.
  ///
  /// qObs entries must be NaN for unvisited children and clamped to [-1, 1] otherwise.
  /// </summary>
  private static void ComputeRobustQ(ParamsSelect paramsSelect,
                                     ReadOnlySpan<double> mu, ReadOnlySpan<double> qObs,
                                     ReadOnlySpan<double> nSupport, ReadOnlySpan<double> sChild,
                                     double parentVolatility, double consensusFallbackQ,
                                     int numChildren,
                                     Span<double> qFpuOut, Span<double> qTildeOut,
                                     out double sigmaBar, out int kVisited,
                                     out double meanShrinkW, out double consensusQ)
  {
    // Consensus q_bar over visited children (weights mu * saturating support gate).
    double sumW = 0.0;
    double sumWQ = 0.0;
    kVisited = 0;
    for (int i = 0; i < numChildren; i++)
    {
      if (!double.IsNaN(qObs[i]))
      {
        kVisited++;
        double gate = nSupport[i] > 0.0 ? nSupport[i] / (nSupport[i] + CONSENSUS_RELIABILITY_K) : 0.0;
        double w = mu[i] * gate;
        sumW += w;
        sumWQ += w * qObs[i];
      }
    }
    consensusQ = sumW > 0.0 ? sumWQ / sumW : consensusFallbackQ;

    // Policy-shaped per-child prior anchored at the consensus (E_mu[q^FPU] = q_bar).
    RPOImputation.ComputePolicyImpliedQ(paramsSelect, mu, consensusQ, qFpuOut, numChildren);

    // Borrowed strength for children whose own volatility estimate is unreliable:
    // the parent's debiased volatility, with a global default when the parent's own
    // estimator is also cold (decode == 0).
    double sParent = parentVolatility > 0.0 ? parentVolatility : DEFAULT_PRIOR_VOLATILITY;
    if (FrozenS > 0.0)
    {
      sParent = FrozenS;   // ablation: no measurement anywhere (see FrozenS)
    }

    double sigma0 = paramsSelect.TPS_ShrinkageSigma0;
    double sigma0Sq = sigma0 * sigma0;

    // Per-child shrinkage + accumulation of (sigma_hat, weight) pairs for the median.
    Span<double> medSigma = stackalloc double[numChildren];
    Span<double> medWeight = stackalloc double[numChildren];
    int medCount = 0;
    double sumShrinkW = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      if (double.IsNaN(qObs[i]))
      {
        qTildeOut[i] = qFpuOut[i];   // the n=0 / w=0 limit: pure FPU imputation
        continue;
      }
      double n = nSupport[i];
      double s = FrozenS > 0.0
        ? FrozenS
        : ((n >= MIN_CHILD_N_FOR_OWN_SIGMA && sChild[i] > 0.0) ? sChild[i] : sParent);
      double sigmaHat = Math.Clamp(s / Math.Sqrt(n + 1.0), SIGMA_HAT_MIN, SIGMA_HAT_MAX);
      double w = sigma0Sq / (sigma0Sq + sigmaHat * sigmaHat);
      qTildeOut[i] = w * qObs[i] + (1.0 - w) * qFpuOut[i];
      sumShrinkW += w;

      medSigma[medCount] = sigmaHat;
      medWeight[medCount] = Math.Max(1.0, n);
      medCount++;
    }
    meanShrinkW = kVisited > 0 ? sumShrinkW / kVisited : 0.0;

    sigmaBar = medCount == 0
      ? Math.Clamp(sParent, SIGMA_HAT_MIN, SIGMA_HAT_MAX)
      : WeightedMedian(medSigma, medWeight, medCount);
  }


  /// <summary>
  /// Weighted median of values[0..count) with the given nonnegative weights
  /// (in-place insertion sort; count is at most MAX_CHILDREN).
  /// </summary>
  private static double WeightedMedian(Span<double> values, Span<double> weights, int count)
  {
    for (int i = 1; i < count; i++)
    {
      double v = values[i];
      double w = weights[i];
      int j = i - 1;
      while (j >= 0 && values[j] > v)
      {
        values[j + 1] = values[j];
        weights[j + 1] = weights[j];
        j--;
      }
      values[j + 1] = v;
      weights[j + 1] = w;
    }
    double totalWeight = 0.0;
    for (int i = 0; i < count; i++)
    {
      totalWeight += weights[i];
    }
    double half = totalWeight * 0.5;
    double cum = 0.0;
    for (int i = 0; i < count; i++)
    {
      cum += weights[i];
      if (cum >= half)
      {
        return values[i];
      }
    }
    return values[count - 1];
  }


  /// <summary>
  /// STEP 2 of TPS: the tempered posterior pi_i proportional to mu_i * exp(q_tilde_i / tau)
  /// (policy-proportional at every N - the beta(N) prior-anchor decay of the original
  /// proposal measurably hurt backup and was removed), computed in log-space with
  /// max-subtraction.  Entries with mu &lt;= 0 receive zero mass; if the whole vector
  /// degenerates (cannot happen with the engine's minimum-policy floor, but guarded
  /// anyway) falls back to the normalized prior.
  /// </summary>
  private static void ComputeTemperedPosterior(ReadOnlySpan<double> mu, ReadOnlySpan<double> qTilde,
                                               double tau, int numChildren,
                                               Span<double> piOut)
  {
    double maxLogit = double.NegativeInfinity;
    for (int i = 0; i < numChildren; i++)
    {
      double logit = mu[i] > 0.0
        ? Math.Log(mu[i]) + qTilde[i] / tau
        : double.NegativeInfinity;
      piOut[i] = logit;
      if (logit > maxLogit)
      {
        maxLogit = logit;
      }
    }

    double z = 0.0;
    if (!double.IsNegativeInfinity(maxLogit))
    {
      for (int i = 0; i < numChildren; i++)
      {
        double e = double.IsNegativeInfinity(piOut[i]) ? 0.0 : Math.Exp(piOut[i] - maxLogit);
        piOut[i] = e;
        z += e;
      }
    }

    if (z > 0.0)
    {
      for (int i = 0; i < numChildren; i++)
      {
        piOut[i] /= z;
      }
    }
    else
    {
      // Degenerate: fall back to the normalized prior (uniform if that too is zero).
      double sumMu = 0.0;
      for (int i = 0; i < numChildren; i++)
      {
        sumMu += mu[i];
      }
      for (int i = 0; i < numChildren; i++)
      {
        piOut[i] = sumMu > 0.0 ? mu[i] / sumMu : 1.0 / numChildren;
      }
    }
  }

  #endregion

  #region Backup (V computation)

  /// <summary>
  /// TPS backup value: V = sum_i pi_i(tau) * q_tilde_i over the action set (all
  /// expanded children plus unexpanded children with header.P at least
  /// MIN_P_FOR_Q_IF_UNVISITED), blended with the node's own network value as one
  /// visit.  tau is winner's-curse referenced: k_b * sigma_bar * sqrt(2 ln k_visited).
  /// Closed form; no solver (the single small Solve inside ComputePolicyImpliedQ
  /// produces the shrinkage targets, not the aggregation).
  /// </summary>
  internal static double ComputeVBar(GNode node, ParamsSelect paramsSelect)
    => ComputeVBar(node, paramsSelect, null);


  /// <summary>
  /// Overlay-capable variant of <see cref="ComputeVBar(GNode, ParamsSelect)"/>.
  /// When overlay is non-null, all backup-mutable statistics (node N, node volatility,
  /// edge N / diluted Q) are read through it instead of the live graph, and the
  /// diagnostic/statistics side channels are suppressed (they describe the live search,
  /// not hypothetical replays). Used by the QProbe shadow backup to compute hypothetical
  /// TPS backup values with bit-identical arithmetic.
  /// </summary>
  internal static double ComputeVBar(GNode node, ParamsSelect paramsSelect, ITPSStatsOverlay overlay)
  {
    int numExpanded = node.NumEdgesExpanded;
    if (numExpanded == 0)
    {
      return node.NodeRef.V;
    }

    // Action set: all expanded children plus unexpanded children with sufficient P
    // (edges are (usually) sorted strictly descending by P; break at first below).
    int numChildren = numExpanded;
    int numPolicyMoves = node.NumPolicyMoves;
    Span<GEdgeHeaderStruct> headersSpan = node.EdgeHeadersSpan;
    if (numPolicyMoves > numExpanded)
    {
      int maxKeptIndex = numExpanded - 1;
      for (int i = numExpanded; i < numPolicyMoves; i++)
      {
        if ((float)headersSpan[i].P < MIN_P_FOR_Q_IF_UNVISITED)
        {
          break;
        }
        maxKeptIndex = i;
      }
      numChildren = maxKeptIndex + 1;
    }

    int cap = PUCTScoreCalcVector.MAX_CHILDREN;
    numChildren = Math.Min(numChildren, cap);
    Span<double> mu = (bufferMu ??= new double[cap]).AsSpan(0, numChildren);
    Span<double> qObs = (bufferQObs ??= new double[cap]).AsSpan(0, numChildren);
    Span<double> edgeNSpan = (bufferEdgeN ??= new double[cap]).AsSpan(0, numChildren);
    Span<double> nSupport = (bufferNSupport ??= new double[cap]).AsSpan(0, numChildren);
    Span<double> sChild = (bufferSChild ??= new double[cap]).AsSpan(0, numChildren);

    double sumEdgeN = 0.0;   // for the diagnostic vanilla visit-weighted V (sampler only)
    double sumEdgeW = 0.0;
    for (int i = 0; i < numChildren; i++)
    {
      if (i < numExpanded)
      {
        GEdge edge = node.ChildEdgeAtIndex(i);
        mu[i] = edge.P;
        int edgeN;
        double edgeQ;
        if (overlay == null)
        {
          edgeN = edge.N;
          edgeQ = edge.Q;
        }
        else
        {
          (edgeN, edgeQ) = overlay.EdgeNAndQ(node, i, edge);
        }
        edgeNSpan[i] = edgeN;
        // Child perspective -> node's own perspective; clamp to the legal value range
        // (defuses proven-value sentinels).
        qObs[i] = edgeN == 0 ? double.NaN : Math.Clamp(-edgeQ, -1.0, 1.0);
        if (edgeN > 0)
        {
          sumEdgeN += edgeN;
          sumEdgeW += -edgeQ * edgeN;
        }
        if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
        {
          GNode childNode = edge.ChildNode;
          // True statistical support (cross-parent).
          nSupport[i] = overlay == null ? childNode.NodeRef.N : overlay.NodeN(childNode);
          sChild[i] = overlay == null ? childNode.LeafValueVolatilityDebiased
                                      : overlay.NodeVolatilityDebiased(childNode);
        }
        else
        {
          nSupport[i] = edgeN;                 // terminal edges: Q exact
          sChild[i] = 0.0;
        }
      }
      else
      {
        mu[i] = (double)headersSpan[i].P;
        edgeNSpan[i] = 0.0;
        qObs[i] = double.NaN;
        nSupport[i] = 0.0;
        sChild[i] = 0.0;
      }
    }

    Span<double> qFpu = (bufferQFpu ??= new double[cap]).AsSpan(0, numChildren);
    Span<double> qTilde = (bufferQTilde ??= new double[cap]).AsSpan(0, numChildren);
    ComputeRobustQ(paramsSelect, mu, qObs, nSupport, sChild,
                   parentVolatility: overlay == null ? node.LeafValueVolatilityDebiased
                                                     : overlay.NodeVolatilityDebiased(node),
                   consensusFallbackQ: node.NodeRef.V,
                   numChildren, qFpu, qTilde,
                   out double sigmaBar, out int kVisited, out double meanShrinkW,
                   out double consensusQ);

    if (kVisited == 0)
    {
      // No search evidence among children: aggregating pure imputations at low
      // temperature would just be winner's curse on the prior.  Use the network value.
      return node.NodeRef.V;
    }

    int totalN = overlay == null ? node.NodeRef.N : overlay.NodeN(node);
    int kEff = Math.Max(2, kVisited);   // sqrt(2 ln 1) = 0 would zero the temperature
    double tauBackup = Math.Max(TAU_MIN,
        paramsSelect.TPS_BackupTemperatureK * sigmaBar * Math.Sqrt(2.0 * Math.Log(kEff)));

    Span<double> piTilde = (bufferPiTilde ??= new double[cap]).AsSpan(0, numChildren);
    ComputeTemperedPosterior(mu, qTilde, tauBackup, numChildren, piTilde);

    double childContribution = System.Numerics.Tensors.TensorPrimitives.Dot<double>(piTilde, qTilde);

    // Self-V blend: the node's own network value counts as exactly one visit.
    double vBar = totalN <= 0
      ? node.NodeRef.V
      : (childContribution * (totalN - 1) + node.NodeRef.V) / totalN;

    if (TPSDumpDiagnostics.DEBUG_DUMP_TPS_BACKUP_CALCS && overlay == null)
    {
      TPSDumpDiagnostics.DumpTPSBackup(node, paramsSelect, mu, qObs, qFpu, qTilde, piTilde,
                                       edgeNSpan, nSupport, numChildren, numExpanded,
                                       sigmaBar, tauBackup, meanShrinkW,
                                       childContribution, consensusQ, vBar);
    }

    if (StatsWriter != null && overlay == null && (Interlocked.Increment(ref statsCounter) & 1023) == 0)
    {
      // Diagnostic reference: the vanilla visit-weighted backup value on the same node
      // (children weighted by edge.N plus self-V as one visit), so the sampler exposes
      // the systematic bias vTPS - vVanilla by node size.
      double vVanilla = (sumEdgeW + node.NodeRef.V) / (sumEdgeN + 1.0);
      WriteStatsSample(totalN, numChildren, kVisited, sigmaBar, tauBackup,
                       meanShrinkW, piTilde, qTilde, qObs, vBar, vVanilla);
    }

    return vBar;
  }

  #endregion

  #region Statistics instrumentation (env CERES_TPS_STATS)

  // Research instrumentation: when the CERES_TPS_STATS environment variable names a CSV
  // path, roughly 1 of every 1024 backups appends one sample row.  Zero overhead when
  // the variable is unset (single null check).
  private static readonly StreamWriter StatsWriter = CreateStatsWriter();
  private static int statsCounter;

  private static StreamWriter CreateStatsWriter()
  {
    string path = System.Environment.GetEnvironmentVariable("CERES_TPS_STATS");
    if (string.IsNullOrEmpty(path))
    {
      return null;
    }
    StreamWriter writer = new StreamWriter(path, append: true) { AutoFlush = true };
    lock (writer)
    {
      writer.WriteLine("N,k,kVisited,sigmaBar,tau,meanW,piEntropy,imputedMass,q1,gap12,vTPS,vVanilla");
    }
    return writer;
  }

  /// <summary>
  /// Writes one CSV sample: the node's noise scale and temperature (the quantities every
  /// TPS decision is denominated in - logging their percentiles per run is the health
  /// check that tuning transfers across nets), the posterior's entropy, the pi mass on
  /// FPU-imputed (unvisited) children, the top-2 robust-Q separation, and the TPS value
  /// against the vanilla visit-weighted value (systematic-bias measurement).
  /// </summary>
  private static void WriteStatsSample(double n, int numChildren, int kVisited,
                                       double sigmaBar, double tau, double meanW,
                                       ReadOnlySpan<double> piTilde, ReadOnlySpan<double> qTilde,
                                       ReadOnlySpan<double> qObs,
                                       double vTPS, double vVanilla)
  {
    double entropy = 0.0;
    double imputedMass = 0.0;
    double q1 = double.NegativeInfinity, q2 = double.NegativeInfinity;
    for (int i = 0; i < numChildren; i++)
    {
      double p = piTilde[i];
      if (p > 0.0)
      {
        entropy -= p * Math.Log(p);
      }
      if (double.IsNaN(qObs[i]))
      {
        imputedMass += p;
      }
      else
      {
        double q = qTilde[i];
        if (q > q1) { q2 = q1; q1 = q; }
        else if (q > q2) { q2 = q; }
      }
    }
    double gap12 = (kVisited >= 2) ? (q1 - q2) : 0.0;

    string line = string.Create(System.Globalization.CultureInfo.InvariantCulture,
        $"{n:F0},{numChildren},{kVisited},{sigmaBar:F5},{tau:F5},{meanW:F4},{entropy:F4},{imputedMass:F4},{q1:F4},{gap12:F4},{vTPS:F4},{vVanilla:F4}");
    lock (StatsWriter)
    {
      StatsWriter.WriteLine(line);
    }
  }

  #endregion
}
