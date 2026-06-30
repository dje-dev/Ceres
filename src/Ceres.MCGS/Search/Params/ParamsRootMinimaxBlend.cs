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

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Parameters for the opt-in root tiebreak that disambiguates root moves which are
/// nearly equal in averaged Q by using a static, depth-bounded soft-minimax blend
/// over the existing search graph.
///
/// Motivation: Q is the mean over visits, but the opponent plays the max-for-them
/// reply; averaging therefore systematically underestimates danger when one sharp
/// refutation hides among many bland alternatives. When several root moves are within
/// QGapThreshold in Q (and are already well supported by visits), this recomputes a
/// partial-minimax-blended value for each over a bounded subgraph and prefers the more
/// robust move.
///
/// The pass is READ-ONLY over the final tree (no NN evaluations, no graph mutation),
/// so unlike the deep-rollout revaluation it cannot contaminate the search or the tree
/// reused for the following move. It reuses the proven backup operators in
/// PrincipalRevaluation.ComputeOperators (A = visit-weighted mean, which reproduces the
/// engine's backed-up Q; C = visit-weighted soft-minimax power mean).
///
/// Disabled by default; intended for A/B Elo evaluation.
/// </summary>
[Serializable]
public record ParamsRootMinimaxBlend
{
  /// <summary>
  /// Constructor.
  /// </summary>
  public ParamsRootMinimaxBlend()
  {
  }

  /// <summary>
  /// Operating mode for the feature.
  /// </summary>
  public enum ModeType
  {
    /// <summary>Disabled: the tiebreak is never computed (default; a true no-op).</summary>
    Disabled,

    /// <summary>
    /// Shadow: the tiebreak is computed and logs what it would have done, but the move actually
    /// played is left unchanged (for measuring firing rate/quality before trusting it to affect play).
    /// </summary>
    Shadow,

    /// <summary>Active: the tiebreak is computed and overrides the chosen move when indicated.</summary>
    Active
  };

  /// <summary>
  /// Operating mode: Disabled (default), Shadow (compute and log only), or Active (apply to the move).
  /// </summary>
  public ModeType Mode = ModeType.Disabled;

  /// <summary>
  /// Look-ahead depth in plies measured from the root (the candidate move is ply 1).
  /// The deepest recomputed nodes lie at this ply, i.e. Depth-1 reply plies below the
  /// candidate are re-evaluated. A value below 2 reduces to the engine's averaged Q
  /// (there is no opponent reply to minimax over), so Depth &gt;= 2 is required for effect.
  /// </summary>
  public int Depth = 4;

  /// <summary>
  /// Blend weight (lambda) between the averaged operator A and the soft-minimax operator C:
  ///   Vtilde = (1 - Intensity) * A + Intensity * C.
  /// 0 = pure averaging (an exact no-op, since A reproduces the engine's Q);
  /// 1 = pure soft-minimax. The primary "lean toward minimax" intensity dial.
  /// </summary>
  public float Intensity = 0.5f;

  /// <summary>
  /// Exponent (p) of the visit-weighted power-mean soft-minimax operator C
  /// (1 = averaging, larger values approach hard negamax). 
  /// Secondary sharpness knob. 
  /// At p=8, even sharp refutations only pull the value ~halfway to minimax.
  /// </summary>
  public float SoftmaxP = 8;

  /// <summary>
  /// The tiebreak engages only among root moves whose (opponent-perspective) Q is within
  /// this absolute band of the baseline chosen move. About 22% of positions exhibit such a
  /// near-equal-Q cluster at the default 0.01.
  /// </summary>
  public float QGapThreshold = 0.01f;

  /// <summary>
  /// A move is eligible as a tiebreak candidate only if its visit count N is at least this
  /// fraction of the most-visited move's N - i.e. it is "well supported, not very small relative
  /// to the largest N". This is the firing control: lower values consider more alternatives.
  ///
  /// Default matches the chooser's absolute visit floor ManagerChooseBestMoveMCGS.MIN_FRAC_N_REQUIRED_MIN
  /// (0.18, default mode; the strict-mode floor is 0.325). Note the chooser's per-candidate
  /// MinFractionNToUseQ gate is deliberately NOT reused here: for near-equal Q it demands near-full
  /// visit parity (~0.95+) and so would rarely admit a second candidate.
  /// </summary>
  public float MinCandidateFractionN = 0.3f; // somewhat conservative since minimax is noisy

  /// <summary>
  /// A challenger replaces the baseline move only if its blended value exceeds the baseline
  /// move's blended value by at least this margin (0 = strictly greater).
  /// </summary>
  public float SwitchMargin = 0.0f;

  /// <summary>
  /// Minimum root N for the tiebreak to run (the recomputation is meaningless for tiny searches).
  /// </summary>
  public int MinRootN = 10_000;

  /// <summary>
  /// A child is descended through in the recomputation only if its N is at least
  /// max(NCutAbs, CutFraction * rootN); otherwise it is consumed at its edge Q.
  /// Mirrors the frontier visit cut used by PrincipalRevaluation.CollectRegion.
  /// </summary>
  public float CutFraction = 0.01f;

  /// <summary>
  /// Absolute minimum-visit floor for the descent cut (see CutFraction).
  /// </summary>
  public int NCutAbs = 32;

  /// <summary>
  /// Validates settings for self-consistency.
  /// </summary>
  public void Validate()
  {
    if (Intensity < 0 || Intensity > 1)
    {
      throw new Exception("ParamsRootMinimaxBlend.Intensity must be in [0, 1]");
    }

    if (SoftmaxP < 1)
    {
      throw new Exception("ParamsRootMinimaxBlend.SoftmaxP must be at least 1");
    }

    if (Depth < 1)
    {
      throw new Exception("ParamsRootMinimaxBlend.Depth must be at least 1");
    }

    if (CutFraction <= 0 || CutFraction > 0.5f)
    {
      throw new Exception("ParamsRootMinimaxBlend.CutFraction must be in (0, 0.5]");
    }

    if (QGapThreshold < 0)
    {
      throw new Exception("ParamsRootMinimaxBlend.QGapThreshold must be non-negative");
    }

    if (MinCandidateFractionN <= 0 || MinCandidateFractionN > 1)
    {
      throw new Exception("ParamsRootMinimaxBlend.MinCandidateFractionN must be in (0, 1]");
    }

    if (NCutAbs < 1)
    {
      throw new Exception("ParamsRootMinimaxBlend.NCutAbs must be at least 1");
    }
  }
}
