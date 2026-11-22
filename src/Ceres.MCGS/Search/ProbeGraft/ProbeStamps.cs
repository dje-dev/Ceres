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
using System.Globalization;
using System.Collections.Generic;

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Probe "stamps" (SWARM experiment, 2026-08): an external probe engine's evaluation A of a
/// node's position, recorded on the node together with the node's visit count n0 at the time,
/// and applied as a FADING pseudo-visit term wherever the node's value is projected onto a
/// parent edge (edge.QChild):
///
///     q' = (N q + K A) / (N + K),   K = kappa * n0
///
/// The node's own Q, N, visit record and volatility are never touched, so the stamp biases
/// neither the node's FPU anchor nor its leaf-volatility tracker; every parent (including DAG
/// co-parents) sees the same projection through its own edge, and both PUCT select and the
/// backup read that edge. The weight K/(N+K) is kappa/(1+kappa) at probe time and decays like
/// 1/N as the net's own visits accumulate, so a stamp the net cannot confirm simply fades.
///
/// Under the TPS (tempered posterior) backup a parent recomputes its value from its children
/// with an exponentially tilted policy, so a change in one child's projected value reaches the
/// root without visit dilution once the ancestors are recomputed (RecomputeUpward, CPU only).
///
/// Enabled by ParamsSelect.TPS_ProbeStampKappa > 0 (mirrored into Graph.ProbeStampKappa at every
/// search start); with kappa == 0 ProjectChildQ is the identity and the engine is unchanged.
/// </summary>
public static class ProbeStamps
{

  /// <summary>
  /// Parses a weight table "0:1.5,1:0.5,50:0,100:0.7,150:1,200:1.5" (bucket lower bound in |cp| : weight;
  /// separators ',' or '/'). Null/empty yields null (all weights 1).
  /// </summary>
  public static (int MinCp, float Weight)[] ParseCpWeights(string spec)
  {
    if (string.IsNullOrWhiteSpace(spec)) return null;
    List<(int, float)> list = new();
    foreach (string part in spec.Split(new[] { ',', '/' }, StringSplitOptions.RemoveEmptyEntries))
    {
      string[] kv = part.Split(':');
      if (kv.Length != 2) throw new ArgumentException("bad cp weight entry: " + part);
      int minCp = int.Parse(kv[0].Trim(), CultureInfo.InvariantCulture);
      float w = float.Parse(kv[1].Trim(), CultureInfo.InvariantCulture);
      if (w < 0 || w > 8) throw new ArgumentException("cp weight out of [0,8]: " + part);
      list.Add((minCp, w));
    }
    list.Sort((a, b) => a.Item1.CompareTo(b.Item1));
    return list.ToArray();
  }


  /// <summary>Weight of a stamp whose probe reported the given cp (side to move; sign ignored),
  /// under the given table (from ParseCpWeights; null = every stamp weighs 1).</summary>
  public static float CpWeight(int cp, (int MinCp, float Weight)[] table)
  {
    if (table == null || table.Length == 0) return 1.0f;
    int acp = Math.Abs(cp);
    if (acp >= 20_000) acp = 20_000;   // mate-range scores fall in the top bucket
    float w = 1.0f;
    foreach ((int minCp, float weight) in table)
    {
      if (acp >= minCp) w = weight; else break;
    }
    return w;
  }


  /// <summary>n0 scaled by a stamp weight (at least 1 when the weight is positive; 0 when the weight is 0).</summary>
  public static int WeightedN0(int n0, float weight)
    => weight <= 0 ? 0 : Math.Max(1, (int)Math.Round(n0 * weight));


  /// <summary>Outcome of an upward recompute after a stamp.</summary>
  public readonly record struct RecomputeResult(int Recomputed, int AncestorsMoved, double RootQBefore, double RootQAfter, bool ReachedRoot);

  const double MOVED_THRESHOLD = 0.01;
  const int MAX_RECOMPUTE_NODES = 8192;


  /// <summary>Encodes a visit count as 1 + round(8 log2 n) in one byte (0 reserved for "none").</summary>
  public static byte EncodeN0(int n0)
  {
    if (n0 < 1) n0 = 1;
    int code = 1 + (int)Math.Round(8.0 * Math.Log2(n0));
    return (byte)Math.Clamp(code, 1, 255);
  }


  /// <summary>Inverse of EncodeN0 (0 maps to 0).</summary>
  public static int DecodeN0(byte code) => code == 0 ? 0 : (int)Math.Round(Math.Pow(2.0, (code - 1) / 8.0));


  /// <summary>
  /// Records a probe stamp on the node: value a (node mover's perspective, Ceres Q units,
  /// clamped to [-1,1]) taken when the node had n0 visits.
  /// </summary>
  public static void Stamp(GNode node, double a, int n0)
  {
    ref GNodeStruct s = ref node.NodeRef;
    s.ProbeStampQ = (sbyte)Math.Round(Math.Clamp(a, -1.0, 1.0) * 127.0);
    s.ProbeStampN = EncodeN0(n0);
  }


  /// <summary>Removes the stamp from the node.</summary>
  public static void Clear(GNode node)
  {
    ref GNodeStruct s = ref node.NodeRef;
    s.ProbeStampN = 0;
    s.ProbeStampQ = 0;
  }


  /// <summary>
  /// Weight K/(N+K) the stamp currently carries in the projection of this node (0 if none).
  /// </summary>
  public static double CurrentWeight(GNode node)
  {
    ref GNodeStruct s = ref node.NodeRef;
    float kappa = node.Graph.ProbeStampKappa;
    if (s.ProbeStampN == 0 || kappa <= 0) return 0;
    double k = kappa * DecodeN0(s.ProbeStampN);
    return k / (s.N + k);
  }


  /// <summary>
  /// The value a parent edge should carry for this child: the child's own Q blended with its
  /// probe stamp (if any) by the fading pseudo-visit rule. Identity when no stamp or Kappa == 0.
  /// </summary>
  public static double ProjectChildQ(GNode child, double q)
  {
    ref GNodeStruct s = ref child.NodeRef;
    byte code = s.ProbeStampN;
    if (code == 0 || double.IsNaN(q))
    {
      return q;
    }
    float kappa = child.Graph.ProbeStampKappa;
    if (kappa <= 0)
    {
      return q;
    }
    double k = kappa * DecodeN0(code);
    double a = s.ProbeStampQ / 127.0;
    double n = s.N;
    return Math.Clamp((n * q + k * a) / (n + k), -1.0, 1.0);
  }




  /// <summary>
  /// After a stamp on <paramref name="node"/>: marks all of its parent edges stale and recomputes
  /// every ancestor up to the search root (deepest first, each node once) from its now-current
  /// children, using the engine's own backup rule (QRecomputeHelper: TPS tempered posterior or the
  /// visit-weighted mean). No NN evaluations. Must be called where no select or backup can run
  /// concurrently: between staged searches, at the legacy single-iterator post-batch pump, or
  /// inside the backup gate (in-flight visits of an evaluating iterator are invisible to the
  /// recompute, which skips edges with N == 0). Hot-path callers pass reusable scratch
  /// collections (cleared here) to avoid two allocations per stamp.
  /// </summary>
  public static RecomputeResult RecomputeUpward(GNode node, GNode searchRoot, ParamsSelect paramsSelect,
                                                PriorityQueue<GNode, int> queueScratch = null,
                                                HashSet<int> seenScratch = null)
  {
    double rootBefore = searchRoot.Q;
    bool regularized = paramsSelect.RegularizedBackupActive;
    int recomputed = 0, moved = 0;
    bool reachedRoot = false;

    if (node.IsNull || node.IsSearchRoot || node.IsGraphRoot)
    {
      return new RecomputeResult(0, 0, rootBefore, rootBefore, node.IsSearchRoot);
    }

    // Deepest-first so that each ancestor is recomputed after all of its changed children.
    PriorityQueue<GNode, int> queue = queueScratch ?? new();
    HashSet<int> seen = seenScratch ?? new();
    queue.Clear();
    seen.Clear();
    foreach (GEdge pe in node.ParentEdges)
    {
      pe.IsStale = true;
      GNode parent = pe.ParentNode;
      if (!parent.IsNull && seen.Add(parent.Index.Index))
      {
        int depth = SafeDepth(parent);
        if (depth >= 0)   // only ancestors inside the search root's subtree
        {
          queue.Enqueue(parent, -depth);
        }
      }
    }

    while (queue.Count > 0 && recomputed < MAX_RECOMPUTE_NODES)
    {
      GNode cur = queue.Dequeue();
      if (QRecomputeHelper.IsEligibleForRecompute(cur))
      {
        double before = cur.Q;
        double after = QRecomputeHelper.RecomputeNodeQ(cur, ReadOnlySpan<double>.Empty, paramsSelect, regularized);
        recomputed++;
        if (Math.Abs(after - before) > MOVED_THRESHOLD)
        {
          moved++;
        }
      }

      if (cur.IsSearchRoot)
      {
        reachedRoot = true;
        continue;   // never recompute above the search root
      }
      if (cur.IsGraphRoot)
      {
        continue;
      }
      foreach (GEdge pe in cur.ParentEdges)
      {
        pe.IsStale = true;
        GNode parent = pe.ParentNode;
        if (!parent.IsNull && seen.Add(parent.Index.Index))
        {
          int depth = SafeDepth(parent);
          if (depth >= 0)   // only ancestors inside the search root's subtree
          {
            queue.Enqueue(parent, -depth);
          }
        }
      }
    }

    return new RecomputeResult(recomputed, moved, rootBefore, searchRoot.Q, reachedRoot);
  }


  /// <summary>
  /// Depth of the node below the search root following tree-parent links, or -1 when the walk
  /// leaves the search root's subtree (graph root reached, parentless node, or a 512-step guard):
  /// under graph reuse a transposition may be reachable from nodes above the search root, and
  /// those must be neither recomputed nor walked (GNode.DepthFromSearchRoot would fault there).
  /// </summary>
  static int SafeDepth(GNode node)
  {
    int depth = 0;
    GNode cur = node;
    while (!cur.IsSearchRoot)
    {
      if (cur.IsNull || cur.IsGraphRoot || cur.NodeRef.ParentsHeader.IsEmpty || depth > 512)
      {
        return -1;
      }
      cur = cur.Graph[cur.TreeParentNodeIndex];
      depth++;
    }
    return depth;
  }
}
