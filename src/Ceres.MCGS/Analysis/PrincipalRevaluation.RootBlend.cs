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

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Analysis;

/// <summary>
/// Static (rollout-free) depth-bounded soft-minimax re-evaluation of root moves,
/// used by the opt-in near-equal-Q tiebreak (see ParamsRootMinimaxBlend and
/// ManagerChooseBestMoveMCGS.TryOverrideWithRootMinimaxBlend).
///
/// This reuses the backup operators of the analysis-only deep-rollout revaluation
/// (ComputeOperators: A = visit-weighted mean == engine Q, B = negamax, C = visit-weighted
/// soft-minimax power mean) but performs a pure READ-ONLY pass over the existing graph Q
/// values: no NN rollouts, no graph mutation, so it cannot contaminate the search or the
/// reused tree. Frontier nodes (depth reached, unexpanded, or below the visit cut) contribute
/// their current graph Q. Values are negamax (side-to-move relative); the sign is flipped once
/// per ply and repetition-draw dilution is respected, exactly as in Reval.
/// </summary>
public static partial class PrincipalRevaluation
{
  /// <summary>
  /// Returns, for each candidate root child edge, the root-perspective blended value of playing
  /// that move: Vtilde = -((1-lambda)*A + lambda*C) * dilution, where A and C are the depth-bounded
  /// static operators evaluated at the child node. Terminal edges return -edge.Q directly.
  ///
  /// A single memo is shared across candidates so shared (transposed) subgraphs are evaluated once;
  /// it is keyed by (node index, remaining depth) to remain sound under the ply bound. Cycles are
  /// consumed at the edge's diluted Q (the same approximation Reval makes).
  /// </summary>
  /// <param name="candidates">Candidate root move edges (read-only).</param>
  /// <param name="depthFromRoot">Look-ahead plies measured from the root (candidate = ply 1).</param>
  /// <param name="lambda">Blend weight between operator A (averaging) and operator C (soft-minimax).</param>
  /// <param name="softmaxP">Power-mean exponent for operator C.</param>
  /// <param name="cutFraction">Descent visit cut as a fraction of rootN.</param>
  /// <param name="nCutAbs">Absolute floor for the descent visit cut.</param>
  /// <param name="rootN">Root visit count (for the descent cut).</param>
  /// <returns>Per-candidate root-perspective blended values (higher = better for the side to move).</returns>
  internal static double[] BlendedRootChildValues(IReadOnlyList<GEdge> candidates, int depthFromRoot,
                                                  double lambda, double softmaxP,
                                                  double cutFraction, int nCutAbs, int rootN)
  {
    int nCut = Math.Max(nCutAbs, (int)(cutFraction * rootN));
    Dictionary<(int, int), OpVal> memo = new();
    double[] result = new double[candidates.Count];

    for (int i = 0; i < candidates.Count; i++)
    {
      GEdge e = candidates[i];

      if (e.Type != GEdgeStruct.EdgeType.ChildEdge || e.ChildNodeIndex.IsNull)
      {
        // Terminal edge carries its own exact (child-perspective) Q.
        result[i] = -e.Q;
        continue;
      }

      HashSet<int> onStack = new();
      OpVal cv = RevalStatic(e.ChildNode, depthFromRoot - 1, softmaxP, nCut, memo, onStack);

      double blendChild = (1.0 - lambda) * cv.A + lambda * cv.C;
      double dilution = e.N == 0 ? 1.0 : (double)(e.N - e.NDrawByRepetition) / e.N;
      result[i] = -blendChild * dilution;
    }

    return result;
  }


  /// <summary>
  /// Depth-bounded, rollout-free clone of Reval: computes the three backup operators
  /// (A=avg, B=negamax, C=soft-minimax) over the bounded subgraph rooted at the given node,
  /// using each frontier node's current graph Q. Cycle-safe (onStack) and transposition-safe
  /// (memo keyed by node index and remaining depth).
  /// </summary>
  static OpVal RevalStatic(GNode node, int remainingDepth, double softmaxP, int nCut,
                           Dictionary<(int, int), OpVal> memo, HashSet<int> onStack)
  {
    int idx = node.Index.Index;
    (int, int) key = (idx, remainingDepth);
    if (memo.TryGetValue(key, out OpVal cached))
    {
      return cached;
    }

    // Frontier: depth exhausted, unexpanded, or below the visit cut -> use the engine's own Q.
    if (remainingDepth <= 0 || node.NumEdgesExpanded == 0 || node.N < nCut)
    {
      double q = node.Q;
      OpVal leaf = new() { A = q, B = q, C = q, SA = 0, SB = 0, SC = 0 };
      memo[key] = leaf;
      return leaf;
    }

    onStack.Add(idx);

    int numExpanded = node.NumEdgesExpanded;
    List<(double QA, double QB, double QC, double SA, double SB, double SC, double W)> children = new(numExpanded);

    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge e = node.ChildEdgeAtIndex(slot);
      if (e.N == 0)
      {
        continue;
      }

      double w = e.N;
      double qa, qc;

      if (e.Type != GEdgeStruct.EdgeType.ChildEdge || e.ChildNodeIndex.IsNull)
      {
        // Terminal edge: exact child-perspective Q, no recursion possible.
        qa = qc = -e.Q;
      }
      else
      {
        int childIdx = e.ChildNodeIndex.Index;
        GNode child = e.ChildNode;

        if (child.N >= nCut && !onStack.Contains(childIdx))
        {
          OpVal cv = RevalStatic(child, remainingDepth - 1, softmaxP, nCut, memo, onStack);
          double dilution = (double)(e.N - e.NDrawByRepetition) / e.N;
          qa = -cv.A * dilution; // negamax: flip child perspective to this node's, then dilute by repetition
          qc = -cv.C * dilution;
        }
        else
        {
          // Below-cut child, or a cycle back to an ancestor: consume at the edge's own
          // (dilution-adjusted) cached Q, exactly as the engine's gather does.
          qa = qc = -e.Q;
        }
      }

      if (double.IsNaN(qa))
      {
        continue; // defensively skip edges with undefined values
      }

      // The negamax operator B is unused by the blend; pass qc as a harmless placeholder.
      children.Add((qa, qc, qc, 0, 0, 0, w));
    }

    onStack.Remove(idx);

    OpVal opVal = ComputeOperators(node, children, softmaxP);
    memo[key] = opVal;
    return opVal;
  }
}
