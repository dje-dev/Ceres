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
/// Shared policy-imputation helpers built on the RegularizedPolicyOptimum primitive.
/// Used by the FPU path (PUCTSelector.ApplyRPOImputedFPU, FPUType.PolicyImputedRPO)
/// and by the TPS backup (TPSScoreCalc.ComputeRobustQ shrinkage targets).
/// (Formerly part of CBGPUCTScoreCalc; rehomed when the CBGPUCT machinery was
/// removed in the 2026-07 TPS consolidation.)
/// </summary>
internal static class RPOImputation
{
  // Per-thread scratch for ComputePolicyImpliedQ (reached from both select-phase FPU
  // and backup-phase TPS shrinkage, so it keeps its own buffer).
  [ThreadStatic] private static double[] bufferPolicyImpliedQNaN;

  private static readonly RPOOptions SolveOptionsPolicyImputation =
    new(bisectionIterations: 20, bisectionResidualTol: 1e-7,
        clampQ: true, minPriorProbability: 0.0);


  /// <summary>
  /// Computes the base anchor value for per-child Q imputation, dispatched by anchor
  /// type.  Both PUCTSelector.ApplyRPOImputedFPU (FPU) and the TPS backup call this
  /// helper so the anchor semantics stay aligned across the imputation sites.
  ///
  /// qRaw is the per-child observed Q in PARENT perspective (i.e. -edge.Q for visited
  /// children, NaN for unvisited).  Callers that have qRaw in a different form should
  /// build it once and pass the same span here.
  /// </summary>
  internal static double ComputeImputationAnchor(ParamsSelect.FPUQAnchorType type,
                                                 GNode node,
                                                 ReadOnlySpan<double> qRaw, int numChildren)
  {
    switch (type)
    {
      case ParamsSelect.FPUQAnchorType.ParentV:
        return node.NodeRef.V;

      case ParamsSelect.FPUQAnchorType.ParentQ:
        return node.Q;

      case ParamsSelect.FPUQAnchorType.FirstChildElseParentQ:
        return (numChildren > 0 && !double.IsNaN(qRaw[0])) ? qRaw[0] : node.Q;

      case ParamsSelect.FPUQAnchorType.BestChildElseParentQ:
      case ParamsSelect.FPUQAnchorType.BlendBestChildParentQ:
        {
          double best = double.NegativeInfinity;
          for (int i = 0; i < numChildren; i++)
          {
            double q = qRaw[i];
            if (!double.IsNaN(q) && q > best)
            {
              best = q;
            }
          }
          if (double.IsNegativeInfinity(best))
          {
            return node.Q;
          }
          return type == ParamsSelect.FPUQAnchorType.BestChildElseParentQ
            ? best
            : 0.5 * (best + node.Q);
        }

      default:
        return node.Q;
    }
  }


  /// <summary>
  /// Computes the per-child "policy-implied Q" by running the FPU-style RPO imputation
  /// with all q inputs treated as unknown.  Uses the same machinery as PolicyImputedRPO
  /// FPU (RPOFPURegularization, PolicyImputationTau, MatchValue anchor at referenceQ)
  /// but forces every entry to NaN so the result reflects only the policy prior, not
  /// any observed q.  Used as the per-child shrinkage target for the TPS robust-Q
  /// shrinkage: for a 0.2%-policy move, the implied Q is low (the network thinks it's
  /// bad), so noisy observed q's get pulled toward that pessimistic prior rather than
  /// toward the parent's average Q.
  /// </summary>
  internal static void ComputePolicyImpliedQ(ParamsSelect paramsSelect,
                                             ReadOnlySpan<double> mu, double referenceQ,
                                             Span<double> output, int n)
  {
    Span<double> qNaN = (bufferPolicyImpliedQNaN ??= new double[PUCTScoreCalcVector.MAX_CHILDREN]).AsSpan(0, n);
    qNaN.Fill(double.NaN);

    RPORegularization regularization = paramsSelect.RPOFPURegularization;
    // ReverseKL uses no anchor (level set via nanFallbackQ); forward-KL family uses
    // MatchValue anchor so E_y[q_fill] = referenceQ holds.
    RPOAnchor anchor = regularization == RPORegularization.ReverseKL
      ? RPOAnchor.None
      : new RPOAnchor(RPOAnchorMode.MatchValue, -1, referenceQ);

    double lambda = paramsSelect.PolicyImputationTau;

    RegularizedPolicyOptimum.Solve(mu, qNaN, lambda, anchor, regularization,
                                   yOut: default,
                                   qFillOut: output,
                                   out double _,
                                   options: SolveOptionsPolicyImputation,
                                   nanFallbackQ: referenceQ);
  }
}
