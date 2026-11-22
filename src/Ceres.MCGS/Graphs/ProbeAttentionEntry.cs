#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Graphs;

/// <summary>
/// One "attention" directive for child selection (ADVOCATE experiment, 2026-08): the edge from
/// ParentNodeIndex to ChildNodeIndex receives a fading additive selection-score bonus
/// epsilon * W * K / (childN + K), where epsilon is ParamsSelect.AttentionBonusEpsilon,
/// W in [0,1] is the directive's weight (typically scaled by an external probe's margin) and
/// K > 0 is its pseudo-visit mass (typically kappa * childN at directive time, so the bonus
/// fades as the child accumulates visits, exactly as probe stamps fade).
///
/// The child is identified by node index rather than child slot because edge slots can be
/// permuted between searches (move ordering phases); the slot is resolved against the parent's
/// expanded edges at consumption time.
/// 
/// TODO: this "edge slot permutation" defensiveness may be unnecessary, it can only happen
///       upon ceratain reorderin goperations which are believed to be disabled by default.
/// </summary>
public readonly record struct ProbeAttentionEntry(int ParentNodeIndex, int ChildNodeIndex, float W, float K);
