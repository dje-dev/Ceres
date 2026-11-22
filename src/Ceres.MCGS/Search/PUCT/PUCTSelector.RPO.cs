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

using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.RPO;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// The policy-imputed first play urgency (FPU) computation used by PUCT selection.
///
/// Two implementations live here: a closed form covering the default configuration, and the
/// general path which routes through the shared RegularizedPolicyOptimum solver for every other
/// configuration. They must agree - see the notes on ApplyRPOImputedFPUClosedForm - so any change
/// to the clamping or the tail capping in one belongs in the other as well.
/// </summary>
public static partial class PUCTSelector
{
  /// <summary>
  /// Thread-local double buffer receiving the RPO solver's output before it is narrowed into
  /// qWhenNoChildrenBuffer (the solver and its shared helpers work in double).
  /// </summary>
  [ThreadStatic] static double[] rpoResultBuffer;


  /// <summary>
  /// Hard floor applied to raw priors before taking their log, matching the solver's
  /// NormalizeMu (which floors before normalizing, so the floor applies to the raw prior).
  /// </summary>
  private const float RPO_LOG_FLOOR = 1e-10f;


  /// <summary>
  /// Per-child FPU computed via the unified RegularizedPolicyOptimum primitive.
  /// Imputes parent-perspective Q for every child from the policy prior and any
  /// observed q's, using the KL direction selected by ParamsSelect.RPOFPURegularization
  /// (default ForwardKLSoftmax, matching legacy Boltzmann behavior).
  ///
  /// Anchor selection (forward-KL only; reverse-KL ignores the anchor mode):
  ///   - If the top-policy child (index 0) is visited:
  ///       MatchChild anchor with index 0, value = node.Q.
  ///       Note: this preserves a legacy quirk where the anchor index is taken to
  ///       be 0 (the top-policy child) regardless of which visited child has the
  ///       most informative Q.  See earlier dead code computing 'bestIndex' for
  ///       context.  Bug-for-bug preserved by request.
  ///   - Otherwise (top-policy child unvisited):
  ///       MatchValue anchor with value = node.Q, so E_mu[q_fill] = node.Q.
  /// </summary>
  private static float[] ApplyRPOImputedFPU(ParamsSelect paramsSelect, GNode node, GatheredChildStats stats, int numToProcess)
  {
    // The default configuration admits a closed form (see ApplyRPOImputedFPUClosedForm).
    // Anchors other than the parent-derived ones need the visited children's -W/N to pick their
    // value, and reverse-KL has an entirely different fallback, so those keep the general solver.
    // Both diagnostic hooks read every lane (including visited children, whose values the closed
    // form does not reproduce), so they force the general path too; both are compile-time false.
    if (paramsSelect.RPOFPURegularization == RPORegularization.ForwardKLSoftmax
     && (paramsSelect.FPU_QAnchorType == ParamsSelect.FPUQAnchorType.ParentQ
      || paramsSelect.FPU_QAnchorType == ParamsSelect.FPUQAnchorType.ParentV)
     && !FPUDumpDiagnostics.DEBUG_DUMP_FPU_CALCS
     && !FPURunningStats.DEBUG_DUMP_FPU_CORRELATION_STATS)
    {
      return ApplyRPOImputedFPUClosedForm(paramsSelect, node, stats, numToProcess);
    }

    ReadOnlySpan<float> pSpan = stats.P.Span;
    ReadOnlySpan<float> nSpan = stats.N.Span;
    ReadOnlySpan<float> wSpan = stats.W.Span;

    int numExpanded = node.NumEdgesExpanded;

    // Build mu (normalization happens inside Solve), and q with NaN for unvisited children.
    Span<double> mu = stackalloc double[numToProcess];
    Span<double> qIn = stackalloc double[numToProcess];
    for (int i = 0; i < numToProcess; i++)
    {
      mu[i] = pSpan[i];
      qIn[i] = (i < numExpanded && nSpan[i] > 0) ? -wSpan[i] / nSpan[i] : double.NaN;
    }

    // Anchor VALUE is dispatched by FPU_QAnchorType (default ParentQ = node.Q,
    // matching legacy behavior).  Anchor MODE selection (MatchChild vs MatchValue) is
    // independent of the value and stays based on whether child 0 is visited - this
    // affects only the q calibration formula's intercept, not the value being matched.
    // The reverse-KL path ignores the anchor entirely (must be None there).
    RPORegularization regularization = paramsSelect.RPOFPURegularization;
    double anchorValue = RPOImputation.ComputeImputationAnchor(paramsSelect.FPU_QAnchorType, node, qIn, numToProcess);
    RPOAnchor anchor = regularization == RPORegularization.ReverseKL
      ? RPOAnchor.None
      : (nSpan[0] > 0
          ? new RPOAnchor(RPOAnchorMode.MatchChild, 0, anchorValue)
          : new RPOAnchor(RPOAnchorMode.MatchValue, -1, anchorValue));

    double lambda = paramsSelect.PolicyImputationTau;
    RPOOptions opts = new(bisectionIterations: 12,
                          bisectionResidualTol: 1e-6,
                          clampQ: true,
                          minPriorProbability: 0.0);

    // The solver works in double (shared with the CB-GPUCT prior); its output is narrowed into
    // the float buffer the score kernel loads from.
    double[] solved = rpoResultBuffer ??= new double[PUCTScoreCalcVector.MAX_CHILDREN];
    Span<double> resultSpan = solved.AsSpan(0, numToProcess);

    RegularizedPolicyOptimum.Solve(mu, qIn, lambda, anchor, regularization,
                                   yOut: default,
                                   qFillOut: resultSpan,
                                   out double _,
                                   options: opts,
                                   nanFallbackQ: node.Q);

    // Cap Q values for unexpanded children to not exceed defaultFPU + 0.20.
    double defaultFPU = paramsSelect.CalcQWhenNoChildren(node.IsSearchRoot, node.Q, stats.SumPVisited);
    double maxQ = 0.20 + defaultFPU;
    for (int i = numExpanded; i < numToProcess; i++)
    {
      double thisResult = solved[i] + paramsSelect.RPOFPUValue;
      thisResult = Math.Clamp(thisResult, -1, 1);
      solved[i] = thisResult > maxQ ? maxQ : thisResult;
    }

    float[] result = qWhenNoChildrenBuffer ??= new float[PUCTScoreCalcVector.MAX_CHILDREN];
    for (int i = 0; i < numToProcess; i++)
    {
      result[i] = (float)solved[i];
    }

    if (FPUDumpDiagnostics.DEBUG_DUMP_FPU_CALCS)
    {
      FPUDumpDiagnostics.DumpFPURPO(node, pSpan, nSpan, wSpan, resultSpan,
                                    numToProcess, numExpanded,
                                    lambda, regularization, anchor, defaultFPU);
    }

    return result;
  }


  /// <summary>
  /// Computes the per-child imputed FPU in closed form, for the default configuration only.
  ///
  /// The solver's forward-KL branch is itself closed form (q_i = lambda*log(mu_i) + C, with the
  /// softmax skipped whenever the caller wants only q_fill). What this avoids is the machinery
  /// around it: building double mu/qIn spans, normalizing mu, the general NaN/anchor handling,
  /// and narrowing double results back to float. Writing mu_i = v_i / S with
  /// v_i = max(p_i, LOG_FLOOR) - NormalizeMu floors the RAW prior before normalizing - gives
  ///
  ///   MatchChild(0):  C = anchorQ - lambda*log(mu_0)  =>  q_i = anchorQ + lambda*(log v_i - log v_0)
  ///   MatchValue:     C = anchorQ + lambda*H(mu)      =>  q_i = anchorQ + lambda*(log v_i - E_mu[log v])
  ///
  /// in which log S cancels, so the normalization never has to be performed at all.
  ///
  /// Every lane is written, including visited children, for whom the solver would instead have
  /// preserved clamp(-W/N). Nothing reads those: the score kernel consults the per-child FPU
  /// only where N + NInFlight == 0 (see ComputeScoresSIMD), which implies N == 0.
  ///
  /// Agreement with the solver is to ~3e-7 (float log vs the double path); it is NOT
  /// bit-identical, so an exact tie in the score kernel could in principle break differently.
  /// </summary>
  private static float[] ApplyRPOImputedFPUClosedForm(ParamsSelect paramsSelect, GNode node,
                                                      GatheredChildStats stats, int numToProcess)
  {
    ReadOnlySpan<float> pSpan = stats.P.Span;
    ReadOnlySpan<float> nSpan = stats.N.Span;

    float lambda = paramsSelect.PolicyImputationTau;
    float anchorQ = paramsSelect.FPU_QAnchorType == ParamsSelect.FPUQAnchorType.ParentV
                  ? (float)node.NodeRef.V
                  : (float)node.Q;

    float[] result = qWhenNoChildrenBuffer ??= new float[PUCTScoreCalcVector.MAX_CHILDREN];
    Span<float> q = result.AsSpan(0, numToProcess);

    // Pass 1: log of the floored priors, accumulating the mass and the mu-weighted log needed
    // by the MatchValue anchor (both are skipped implicitly when MatchChild applies).
    float sumV = 0;
    float dotVLogV = 0;
    for (int i = 0; i < numToProcess; i++)
    {
      float v = MathF.Max(pSpan[i], RPO_LOG_FLOOR);
      float logV = MathF.Log(v);
      q[i] = logV;
      sumV += v;
      dotVLogV += v * logV;
    }

    // Anchor MODE follows the general path: MatchChild(0) when child 0 is visited, else MatchValue.
    float reference = nSpan[0] > 0 ? q[0] : dotVLogV / sumV;
    float intercept = anchorQ - lambda * reference;

    // Pass 2: q_i = anchorQ + lambda*(log v_i - reference), clamped as the solver does for
    // imputed values (deliberately [-1, 1], not the wider proven-result clamp).
    for (int i = 0; i < numToProcess; i++)
    {
      q[i] = Math.Clamp(lambda * q[i] + intercept, -1f, 1f);
    }

    // Cap Q values for unexpanded children to not exceed defaultFPU + 0.20 (as the general path).
    int numExpanded = node.NumEdgesExpanded;
    if (numExpanded < numToProcess)
    {
      float defaultFPU = (float)paramsSelect.CalcQWhenNoChildren(node.IsSearchRoot, node.Q, stats.SumPVisited);
      float maxQ = 0.20f + defaultFPU;
      float fpuOffset = paramsSelect.RPOFPUValue;

      for (int i = numExpanded; i < numToProcess; i++)
      {
        float thisResult = Math.Clamp(q[i] + fpuOffset, -1f, 1f);
        q[i] = thisResult > maxQ ? maxQ : thisResult;
      }
    }

    return result;
  }

}
