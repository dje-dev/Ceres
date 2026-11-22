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

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Immutable snapshot of one probe-worthy interior node taken during a target sweep:
/// its identity, visit statistics, dominant child, and the move path from the search root
/// discovered by the sweep's walk (an arbitrary DAG node has no canonical path, but the
/// walk's path is a valid one and is what the probe analyzes). Round-tripped to the probe
/// result as the request's opaque RequesterToken.
/// </summary>
public sealed class ProbeGraftTargetSnapshot
{
  /// <summary>Graph index of the target node.</summary>
  public int NodeIndex;

  /// <summary>Node visit count N at snapshot time.</summary>
  public int N;

  /// <summary>Node Q (own side-to-move perspective) at snapshot time.</summary>
  public double Q;

  /// <summary>Plies below the search root at which the walk found this node.</summary>
  public int Depth;

  /// <summary>Index (within the node's edges) of the child with the most visits.</summary>
  public int DominantChildIndex;

  /// <summary>Fraction of the node's N held by the dominant child.</summary>
  public float DominantVisitFraction;

  /// <summary>The dominant child's move (default when the node has no visited children).</summary>
  public MGMove DominantMove;

  /// <summary>If the dominant-fraction refutation gate held at snapshot time.</summary>
  public bool DominantFractionGateHeld;

  /// <summary>Move path from the search root to this node, as discovered by the walk.</summary>
  public MGMove[] MovesFromRoot;

  /// <summary>
  /// Sweep priority, highest first. In VisitCount mode this is N when the dominant-fraction gate
  /// holds and 0 otherwise; in RootFlipDistance mode it is the reciprocal of RequiredDelta.
  /// </summary>
  public double Priority;

  /// <summary>
  /// RootFlipDistance mode: estimated value change required at this node to reorder the root's
  /// move choice (NaN in VisitCount mode). Smaller is more decision-relevant.
  /// </summary>
  public double RequiredDelta = double.NaN;

  /// <summary>Probe stamps: this request is the confirmation probe of an earlier flag.</summary>
  public bool IsConfirm;

  /// <summary>Probe stamps: the first probe's disagreement d = A - Q (set on the confirm request).</summary>
  public double FirstD;

  /// <summary>Probe stamps: node budget of this probe (ratio-scaled; x multiplier for confirms).</summary>
  public int ProbeNodes;
}


/// <summary>
/// Target selection for PICKET M1 refutation grafting: a periodic depth-bounded region walk
/// from the search root (in the style of PrincipalRevaluation's region descent: visit cut
/// nCut = max(TargetNCutAbs, TargetCutFraction * rootN), onStack cycle guard, visited-set
/// DAG dedup) that collects probe-worthy interior nodes together with their root move paths.
///
/// Eligibility mirrors the DoSearchInnerNodes "rollable" check: evaluated, N &gt; 0,
/// non-terminal, NumPolicyMoves &gt; 0, and (unless TargetIncludeSearchRoot) not the search root.
/// Nodes probed too recently are skipped via a per-node ledger, re-probed only after N grows by
/// ReprobeNGrowthFactor - or by the smaller ReprobeNGrowthFactorShallow at or above
/// ReprobeShallowMaxDepth, since the geometric rule otherwise allows only O(log N) probes per node
/// and saturates coverage of the few high-leverage nodes near the root.
///
/// Priority follows TargetPriorityMode: either N gated on the dominant-child visit fraction
/// (VisitCount, the v1 rule), or proximity to changing the root's move (RootFlipDistance), which
/// ranks by the value change a candidate would need in order to reorder the root's children.
/// SEAM (proposal P6): targeting enrichment via QUnc / GNode.LeafValueVolatility signals
/// belongs here (a volatility term in the priority score), as a later, separately-tested
/// upgrade.
///
/// Runs only on the pump thread at the post-batch quiescent point (read-only graph access).
/// </summary>
internal sealed class ProbeGraftTargeter
{
  readonly ParamsProbeGraft Params;

  /// <summary>
  /// The per-node ledger "node N as of the last accepted probe submission" lives in the node
  /// itself (GNodeStruct.ProbedN, log-coded) so that it survives across moves under graph reuse;
  /// this counts the nodes first probed during this search.
  /// </summary>
  int numDistinctNodesProbed;

  /// <summary>Paths rejected by the incremental collector because their visit chain was not root-anchored/consistent.</summary>
  internal int NumPathsInconsistent;

  /// <summary>Graph of the most recent sweep/collect (for ledger writes by node index).</summary>
  Graph graph;

  /// <summary>Scratch collections reused across sweeps.</summary>
  readonly List<ProbeGraftTargetSnapshot> candidates = new();
  readonly HashSet<int> visited = new();
  readonly HashSet<int> onStack = new();
  readonly List<MGMove> pathScratch = new();

  /// <summary>Scratch for the incremental (per-batch path) collection: one path's visits, leaf to root.</summary>
  readonly List<(int NodeIndex, GEdge Edge, MGPosition Pos)> pathVisitsScratch = new();


  /// <summary>Constructor (one per coordinator/search; parms are the owning search's ProbeGraft params).</summary>
  internal ProbeGraftTargeter(ParamsProbeGraft parms)
  {
    Params = parms;
  }



  #region Root flip distance

  /// <summary>Guards against division by zero for a candidate that is already at the flip point.</summary>
  const double REQUIRED_DELTA_EPSILON = 0.005;

  /// <summary>Priority discount applied when the dominant-visit gate does not hold.</summary>
  const double NON_DOMINANT_PRIORITY_DISCOUNT = 0.25;

  /// <summary>Per-slot gap (in Q units) between each root child and the best alternative root child.</summary>
  readonly List<double> rootChildGaps = new();


  /// <summary>
  /// Computes, for each expanded child slot of the search root, the Q gap that a refutation below
  /// it would have to overcome for the root to choose differently. Edge Q is from the child's
  /// perspective, so the root prefers the SMALLEST edge Q: for the current best child the relevant
  /// gap is to the runner-up (how much worse it would have to become), and for every other child
  /// it is the distance to the best child (how much better it would have to become). Both are
  /// non-negative.
  /// </summary>
  /// <param name="root"></param>
  void ComputeRootChildGaps(GNode root)
  {
    rootChildGaps.Clear();

    int numExpanded = root.NumEdgesExpanded;
    double bestQ = double.MaxValue;
    double secondQ = double.MaxValue;
    int bestSlot = -1;

    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = root.ChildEdgeAtIndex(slot);
      rootChildGaps.Add(0);

      if (edge.N <= 0)
      {
        continue;
      }

      double q = edge.Q;
      if (q < bestQ)
      {
        secondQ = bestQ;
        bestQ = q;
        bestSlot = slot;
      }
      else if (q < secondQ)
      {
        secondQ = q;
      }
    }

    if (bestSlot < 0)
    {
      return;
    }

    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = root.ChildEdgeAtIndex(slot);
      if (edge.N <= 0)
      {
        // Unvisited root children have no meaningful Q yet; treat them as maximally distant
        // rather than maximally urgent.
        rootChildGaps[slot] = double.MaxValue;
        continue;
      }

      rootChildGaps[slot] = slot == bestSlot
                          ? (secondQ == double.MaxValue ? double.MaxValue : secondQ - bestQ)
                          : edge.Q - bestQ;
    }
  }


  /// <summary>
  /// Gap for one root child slot (MaxValue when unknown, which suppresses targeting below it).
  /// </summary>
  /// <param name="slot"></param>
  double RootChildGap(int slot) => slot < rootChildGaps.Count ? rootChildGaps[slot] : double.MaxValue;


  /// <summary>
  /// Estimates the value change required at a candidate node for the root to choose a different
  /// move. A change of d at a node propagates to its root-child ancestor damped by that node's
  /// share of the ancestor's visits, so closing a gap g needs d = g * N(rootChild) / N(node).
  /// The search root itself is the decision, so it requires nothing.
  /// </summary>
  /// <param name="nodeN"></param>
  /// <param name="depth"></param>
  /// <param name="rootChildN"></param>
  /// <param name="rootChildGap"></param>
  static double RequiredDeltaToFlipRoot(int nodeN, int depth, int rootChildN, double rootChildGap)
  {
    if (depth == 0)
    {
      return 0;
    }

    if (rootChildGap == double.MaxValue || nodeN <= 0 || rootChildN <= 0)
    {
      return double.MaxValue;
    }

    return rootChildGap * rootChildN / Math.Max(1, nodeN);
  }

  #endregion

  /// <summary>
  /// Performs one target sweep over the current graph, returning the selected snapshots
  /// (top TargetMaxNodesPerSweep by priority, shallower depth breaking ties).
  /// </summary>
  /// <param name="engine"></param>
  /// <returns></returns>
  internal List<ProbeGraftTargetSnapshot> Sweep(MCGSEngine engine)
  {
    candidates.Clear();
    visited.Clear();
    onStack.Clear();
    pathScratch.Clear();

    graph = engine.Graph;
    GNode root = engine.SearchRootNode;
    int rootN = root.N;
    int nCut = Math.Max(Params.TargetNCutAbs, (int)(Params.TargetCutFraction * rootN));

    ComputeRootChildGaps(root);

    visited.Add(root.Index.Index);
    Walk(root, engine.SearchRootPosMG, 0, nCut, rootChildN: 0, rootChildGap: 0);

    candidates.Sort((a, b) => a.Priority != b.Priority ? b.Priority.CompareTo(a.Priority)
                                                       : a.Depth.CompareTo(b.Depth));
    if (candidates.Count > Params.TargetMaxNodesPerSweep)
    {
      candidates.RemoveRange(Params.TargetMaxNodesPerSweep, candidates.Count - Params.TargetMaxNodesPerSweep);
    }

    return candidates;
  }


  /// <summary>
  /// Records that a probe for the node was accepted by the source
  /// (arms the re-probe backoff at the node's current N).
  /// </summary>
  /// <param name="snapshot"></param>
  internal void RecordProbeSubmitted(ProbeGraftTargetSnapshot snapshot)
  {
    GNode node = graph[snapshot.NodeIndex];
    ref GNodeStruct nodeRef = ref node.NodeRef;
    if (nodeRef.ProbedN == 0)
    {
      numDistinctNodesProbed++;
    }
    nodeRef.ProbedN = ProbeStamps.EncodeN0(snapshot.N);
  }


  /// <summary>
  /// Number of distinct graph nodes first probed during this search (nodes carrying a ProbedN
  /// record from an earlier move's search are not counted again).
  /// </summary>
  internal int NumDistinctNodesProbed => numDistinctNodesProbed;


  /// <summary>
  /// Re-probe backoff against the node's persistent ledger: true when the node was probed before
  /// and its N has not yet grown by the required factor since.
  /// </summary>
  bool ProbedTooRecently(in GNodeStruct nodeRef, int depth)
  {
    byte code = nodeRef.ProbedN;
    if (code == 0)
    {
      return false;
    }
    float growthFactorRequired = depth <= Params.ReprobeShallowMaxDepth
                               ? Math.Min(Params.ReprobeNGrowthFactorShallow, Params.ReprobeNGrowthFactor)
                               : Params.ReprobeNGrowthFactor;
    return nodeRef.N < (long)(ProbeStamps.DecodeN0(code) * growthFactorRequired);
  }


  /// <summary>
  /// Incremental stamp-mode targeting (ParamsProbeGraft.ProbeStampIncrementalTargeting): instead of
  /// a periodic walk of the whole graph, examines only the nodes on the paths the iterator just
  /// backed up (the batch's PathsSet, intact at the quiescent pump) and collects those that have
  /// crossed the probe threshold: N in [nCut, ProbeStampMaxN], never probed or grown by the
  /// re-probe factor since (GNodeStruct.ProbedN), evaluated, non-terminal, not the search root.
  /// A node crosses nCut exactly on a visit that lies on one of these paths, so nothing is missed
  /// that the periodic sweep would have found; a node never visited again is never probed, which
  /// is the intended economy (attention follows the search). Each node is taken once per call;
  /// the move path is the path's own (a valid path from the search root). Candidates are returned
  /// sorted by N descending (larger subtrees first), as the sweep does.
  /// </summary>
  internal List<ProbeGraftTargetSnapshot> CollectFromPaths(MCGSIterator iterator, int maxCandidates = int.MaxValue)
  {
    candidates.Clear();
    visited.Clear();

    MCGSEngine engine = iterator.Engine;
    graph = engine.Graph;
    GNode root = engine.SearchRootNode;
    int nCut = Math.Max(Params.TargetNCutAbs, (int)(Params.TargetCutFraction * root.N));
    int nMax = Params.ProbeStampMaxN;
    int rootIndex = root.Index.Index;

    foreach (MCGSPath path in iterator.PathsSet.Paths)
    {
      if (candidates.Count >= maxCandidates)
      {
        break;   // no more can be submitted this pass (prober saturated); the rest are seen again when revisited
      }

      // Ordinary batch paths carry no root slot: slot k (root-to-leaf order) is the node at depth
      // k + 1, its ParentChildEdge leading from the node at depth k (the search root for k = 0).
      // The engine's root-initialization path (one slot, IsRootInitializationPath) and inner-node
      // rollouts (a different root) are skipped.
      if (path == null || path.NumVisitsInPath < 1 || path.IsRootInitializationPath || !path.InnerSearchStartNode.IsNull)
      {
        continue;
      }

      // Pass 1 (cheap, no position copies): does this path hold any threshold-crossing node?
      bool anyCandidate = false;
      int depthFromLeaf = 0;
      int numVisits = path.NumVisitsInPath;
      foreach (MCGSPathVisitMember member in path.PathVisitsLeafToRoot)
      {
        ref readonly MCGSPathVisit visit = ref member.PathVisitRef;
        int depth = numVisits - depthFromLeaf;
        depthFromLeaf++;
        if (visit.IsRootInitializationPath || visit.ParentChildEdge.Type != GEdgeStruct.EdgeType.ChildEdge)
        {
          continue;
        }
        GNode node = visit.ParentChildEdge.ChildNode;
        if (node.IsNull)
        {
          continue;
        }
        ref readonly GNodeStruct nodeRef = ref node.NodeRef;
        int n = nodeRef.N;
        if (n < nCut || n > nMax || node.Index.Index == rootIndex || visited.Contains(node.Index.Index))
        {
          continue;
        }
        if (ProbedTooRecently(in nodeRef, depth))
        {
          continue;
        }
        anyCandidate = true;
        break;
      }
      if (!anyCandidate)
      {
        continue;
      }

      // Pass 2: materialize the path root-to-leaf (positions come free from the visits) and
      // snapshot every qualifying node with its move path.
      pathVisitsScratch.Clear();
      foreach (MCGSPathVisitMember member in path.PathVisitsLeafToRoot)
      {
        ref readonly MCGSPathVisit visit = ref member.PathVisitRef;
        int nodeIndex = visit.ParentChildEdge.Type == GEdgeStruct.EdgeType.ChildEdge && !visit.ParentChildEdge.ChildNodeIndex.IsNull
                      ? visit.ParentChildEdge.ChildNodeIndex.Index : -1;
        pathVisitsScratch.Add((nodeIndex, visit.ParentChildEdge, visit.ChildPosition));
      }
      pathVisitsScratch.Add((rootIndex, default, engine.SearchRootPosMG));
      pathVisitsScratch.Reverse();   // now index 0 is the search root

      pathScratch.Clear();
      if (pathVisitsScratch[0].NodeIndex != rootIndex)
      {
        NumPathsInconsistent++;
        if (NumPathsInconsistent <= 3) Console.WriteLine($"[ProbeGraft-inc] path does not start at the search root: first={pathVisitsScratch[0].NodeIndex} root={rootIndex} visits={pathVisitsScratch.Count}/{path.NumVisitsInPath}");
        continue;
      }
      for (int i = 1; i < pathVisitsScratch.Count; i++)
      {
        (int nodeIndex, GEdge edge, MGPosition pos) = pathVisitsScratch[i];
        if (nodeIndex < 0)
        {
          break;   // path left the graph (should not happen after backup); stop here
        }
        if (edge.ParentNode.Index.Index != pathVisitsScratch[i - 1].NodeIndex)
        {
          NumPathsInconsistent++;
          if (NumPathsInconsistent <= 3) Console.WriteLine($"[ProbeGraft-inc] edge parent mismatch at slot {i}: edgeParent={edge.ParentNode.Index.Index} prev={pathVisitsScratch[i - 1].NodeIndex} root={rootIndex} visits={pathVisitsScratch.Count}/{path.NumVisitsInPath}");
          break;
        }
        MGPosition parentPos = pathVisitsScratch[i - 1].Pos;
        MGMove move = edge.MoveMGFromPos(in parentPos);
        pathScratch.Add(move);

        if (nodeIndex == rootIndex || !visited.Add(nodeIndex))
        {
          continue;
        }
        GNode node = graph[nodeIndex];
        ref readonly GNodeStruct nodeRef = ref node.NodeRef;
        int n = nodeRef.N;
        if (n < nCut || n > nMax || ProbedTooRecently(in nodeRef, i))
        {
          continue;
        }
        bool eligible = node.IsEvaluated
                     && n > 0
                     && !node.Terminal.IsTerminal()
                     && node.NumPolicyMoves > 0
                     && !node.IsSearchRoot;
        if (!eligible)
        {
          continue;
        }

        if (candidates.Count >= maxCandidates)
        {
          break;
        }
        candidates.Add(new ProbeGraftTargetSnapshot()
        {
          NodeIndex = nodeIndex,
          N = n,
          Q = node.Q,
          Depth = i,
          DominantChildIndex = -1,
          DominantVisitFraction = 0,
          DominantMove = default,
          DominantFractionGateHeld = false,
          MovesFromRoot = pathScratch.ToArray(),
          Priority = n,
          RequiredDelta = double.NaN,
        });
      }
    }

    if (candidates.Count > 1)
    {
      candidates.Sort((a, b) => a.Priority != b.Priority ? b.Priority.CompareTo(a.Priority)
                                                         : a.Depth.CompareTo(b.Depth));
    }
    return candidates;
  }


  /// <summary>
  /// Recursive descent below the given node (whose position is passed alongside so edge
  /// moves convert without CalcPosition walks). Children are descended only when above the
  /// visit cut; each qualifying non-root node is considered as a candidate once per sweep.
  /// </summary>
  void Walk(GNode node, in MGPosition nodePos, int depth, int nCut, int rootChildN, double rootChildGap)
  {
    if (depth > 0 || Params.TargetIncludeSearchRoot)
    {
      ConsiderCandidate(node, in nodePos, depth, rootChildN, rootChildGap);
    }

    if (depth >= Params.TargetMaxDepth)
    {
      return;
    }

    int nodeIdx = node.Index.Index;
    onStack.Add(nodeIdx);

    int numExpanded = node.NumEdgesExpanded;
    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = node.ChildEdgeAtIndex(slot);
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeIndex.IsNull)
      {
        continue;
      }

      GNode child = edge.ChildNode;
      int childIdx = child.Index.Index;
      if (child.N < nCut || onStack.Contains(childIdx) || !visited.Add(childIdx))
      {
        continue;
      }

      MGMove move = edge.MoveMGFromPos(in nodePos);
      MGPosition childPos = nodePos;
      childPos.MakeMove(move);

      // At depth 0 each child begins a new root-child subtree, whose visit count and
      // distance-to-best gap are carried down to every node below it.
      int childRootChildN = depth == 0 ? child.N : rootChildN;
      double childRootChildGap = depth == 0 ? RootChildGap(slot) : rootChildGap;

      pathScratch.Add(move);
      Walk(child, in childPos, depth + 1, nCut, childRootChildN, childRootChildGap);
      pathScratch.RemoveAt(pathScratch.Count - 1);
    }

    onStack.Remove(nodeIdx);
  }


  /// <summary>
  /// Evaluates one walked node for candidacy (eligibility, re-probe backoff) and
  /// records a snapshot when it qualifies.
  /// </summary>
  void ConsiderCandidate(GNode node, in MGPosition nodePos, int depth, int rootChildN, double rootChildGap)
  {
    bool eligible = node.IsEvaluated
                 && node.N > 0
                 && !node.Terminal.IsTerminal()
                 && node.NumPolicyMoves > 0
                 && (!node.IsSearchRoot || Params.TargetIncludeSearchRoot);
    if (!eligible)
    {
      return;
    }

    // Nodes near the root carry the most decision leverage, so they use the (smaller) shallow
    // re-probe spacing: the geometric rule alone allows only O(log N) probes per node, which
    // saturates their coverage long before probe capacity is exhausted.
    int nodeIdx = node.Index.Index;
    if (ProbedTooRecently(in node.NodeRef, depth))
    {
      return; // probed too recently relative to visit growth
    }

    // Dominant child by visit count (over expanded edges; unexpanded children have N = 0).
    int dominantIndex = -1;
    int dominantN = 0;
    int numExpanded = node.NumEdgesExpanded;
    for (int slot = 0; slot < numExpanded; slot++)
    {
      GEdge edge = node.ChildEdgeAtIndex(slot);
      if (edge.N > dominantN)
      {
        dominantN = edge.N;
        dominantIndex = slot;
      }
    }

    float dominantFraction = dominantIndex < 0 ? 0 : (float)dominantN / node.N;
    bool gateHeld = dominantFraction >= Params.DominantChildMinVisitFraction;

    double requiredDelta = double.NaN;
    double priority;
    if (Params.TargetPriorityMode == ParamsProbeGraft.TargetPriorityModeType.RootFlipDistance)
    {
      requiredDelta = RequiredDeltaToFlipRoot(node.N, depth, rootChildN, rootChildGap);
      if (requiredDelta > Params.TargetMaxRequiredDelta)
      {
        return; // no plausible refutation here could change the move played
      }

      // Rank by ascending required change; the dominant-visit gate still favors nodes where a
      // refutation trigger (as opposed to a discovery trigger) is possible at all.
      priority = 1.0 / (requiredDelta + REQUIRED_DELTA_EPSILON);
      if (!gateHeld)
      {
        priority *= NON_DOMINANT_PRIORITY_DISCOUNT;
      }
    }
    else
    {
      priority = gateHeld ? node.N : 0;
    }

    if (Params.ProbeStampMode == ParamsProbeGraft.ProbeStampModeType.Active)
    {
      if (node.N > Params.ProbeStampMaxN)
      {
        return;   // the ratio principle: above the cap the probe is a peer, not an oracle
      }
      priority = node.N;   // no dominant-child gate for value stamps; larger subtrees first
    }

    ProbeGraftTargetSnapshot snapshot = new()
    {
      NodeIndex = nodeIdx,
      N = node.N,
      Q = node.Q,
      Depth = depth,
      DominantChildIndex = dominantIndex,
      DominantVisitFraction = dominantFraction,
      DominantMove = dominantIndex < 0 ? default
                                       : node.ChildEdgeAtIndex(dominantIndex).MoveMGFromPos(in nodePos),
      DominantFractionGateHeld = gateHeld,
      MovesFromRoot = pathScratch.ToArray(),
      Priority = priority,
      RequiredDelta = requiredDelta,
    };

    candidates.Add(snapshot);
  }
}
