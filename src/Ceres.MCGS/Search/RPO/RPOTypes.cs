#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Search.RPO;

/// <summary>
/// Choice of divergence D in the regularized policy optimization objective
///   y*(a; mu, q, lambda) = argmax_y { sum_a y(a) q(a)  -  lambda * D(mu, y) }
/// from Grill et al. "Monte-Carlo Tree Search as Regularized Policy Optimization" (2020).
/// </summary>
public enum RPORegularization
{
  /// <summary>
  /// Reverse KL: D = KL(mu || y).  Closed form  y(a) = lambda * mu(a) / (alpha - q(a))
  /// with alpha the Lagrange multiplier solved by bisection.  This is the form used
  /// in Grill et al. and matches MCTS visit-count behavior.
  /// </summary>
  ReverseKL,

  /// <summary>
  /// Forward KL / softmax: D = KL(y || mu).  Closed form  y(a) proportional to
  /// mu(a) * exp(q(a) / lambda).  Inverse identity  q(a) = lambda * log(mu(a)) + C(s)
  /// makes this useful as a Boltzmann Q-imputation when q is partly unknown.
  /// </summary>
  ForwardKLSoftmax
}


/// <summary>
/// How an anchor pins the otherwise-free intercept C(s) of the inverse closed form
/// (only meaningful when q contains NaN entries that must be imputed).  Reverse KL
/// has no analogous freedom and rejects any non-None anchor.
/// </summary>
public enum RPOAnchorMode
{
  /// <summary>No anchor; valid only when all q are finite (or for reverse KL).</summary>
  None,

  /// <summary>
  /// Choose C(s) so that the expectation E_mu[q_fill] equals Anchor.Value.
  /// Matches the entropy-anchored Boltzmann form  q_i = v + lambda * (log mu_i + H(mu)).
  /// </summary>
  MatchValue,

  /// <summary>
  /// Choose C(s) so that q_fill[Anchor.Index] = Anchor.Value.
  /// Matches  q_i = anchorQ + lambda * (log mu_i - log mu_anchor).
  /// </summary>
  MatchChild
}


/// <summary>
/// Anchor specification for the inverse problem.  Index is unused (and should be -1)
/// for modes other than MatchChild.
/// </summary>
public readonly record struct RPOAnchor(RPOAnchorMode Mode, int Index, double Value)
{
  public static RPOAnchor None => new(RPOAnchorMode.None, -1, 0.0);
}


/// <summary>
/// Tuning knobs for the RegularizedPolicyOptimum solver.  Default values aim at
/// the selection-time use case (cheap and approximate); raise BisectionIterations
/// and tighten BisectionResidualTol for backup-time uses.
/// </summary>
public readonly struct RPOOptions
{
  public RPOOptions(int bisectionIterations = 12,
                    double bisectionResidualTol = 1e-6,
                    bool clampQ = true,
                    double minPriorProbability = 0.0)
  {
    BisectionIterations = bisectionIterations;
    BisectionResidualTol = bisectionResidualTol;
    ClampQ = clampQ;
    MinPriorProbability = minPriorProbability;
  }

  /// <summary>Upper bound on bisection iterations when solving for alpha (reverse KL).</summary>
  public int BisectionIterations { get; init; }

  /// <summary>Residual tolerance |sum(coeff/(alpha - q)) - 1| at which bisection terminates early.</summary>
  public double BisectionResidualTol { get; init; }

  /// <summary>
  /// If true, q values are clamped before the closed form is applied.  Reverse KL clamps
  /// finite q's AND the NaN-fallback to [-1.2, +1.2] (preserving proven-win/loss encodings
  /// slightly above |1|); forward KL clamps to [-1, +1] (imputed guesses must not claim
  /// proven-result magnitudes).  See RegularizedPolicyOptimum.CLAMP_Q_MIN/MAX.
  /// </summary>
  public bool ClampQ { get; init; }

  /// <summary>Optional floor applied to mu before renormalization (epsilon-smoothing of the prior).</summary>
  public double MinPriorProbability { get; init; }

  /// <summary>Conventional defaults intended for the selection path.</summary>
  public static RPOOptions Default => new(12, 1e-6, true, 0.0);
}


/// <summary>
/// Choice of apportionment algorithm in RPOVisitAllocator.Allocate.
/// </summary>
public enum RPOAllocationAlgorithm
{
  /// <summary>
  /// One-at-a-time greedy: at each step pick argmax_i (pi_bar_i * (sumN+k) - N_i).
  /// Equivalent to Hamilton's largest-remainders for fixed pi_bar.  Matches the
  /// historical CBGPUCT inner loop exactly.
  /// </summary>
  IterativeLargestDeficit,

  /// <summary>
  /// Closed-form Hamilton (largest-remainders) apportionment: O(n log n) regardless
  /// of budget.  Cheaper than iterative for large budgets; otherwise behaviorally
  /// identical for fixed pi_bar.
  /// </summary>
  HamiltonClosedForm
}


/// <summary>
/// Tuning knobs for the RPOVisitAllocator.
/// </summary>
public readonly struct RPOAllocationOptions
{
  public RPOAllocationOptions(RPOAllocationAlgorithm algorithm = RPOAllocationAlgorithm.IterativeLargestDeficit,
                              bool stopWhenAllOverQuota = true)
  {
    Algorithm = algorithm;
    StopWhenAllOverQuota = stopWhenAllOverQuota;
  }

  /// <summary>Apportionment algorithm to use.</summary>
  public RPOAllocationAlgorithm Algorithm { get; init; }

  /// <summary>
  /// If true, allocation halts early once all children are above their pi_bar target.
  /// Matches CBGPUCT's graph-aware-deficit early-break.
  /// </summary>
  public bool StopWhenAllOverQuota { get; init; }

  public static RPOAllocationOptions Default => new(RPOAllocationAlgorithm.IterativeLargestDeficit, true);
}
